#include <libfreenect/libfreenect.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "pose/rtmpose_tracker_onnxruntime.h"
#include "pose/skeleton_3d.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sys/time.h>
#include <thread>
#include <vector>

static const int DW = 640, DH = 480;

// ── shared camera buffers ─────────────────────────────────────────────────────
static std::mutex        s_depth_mtx, s_video_mtx;
static uint8_t           s_depth_pix[DW * DH * 3]{};
static uint16_t          s_depth_raw[DW * DH]{};
static uint8_t           s_video_pix[DW * DH * 3]{};
static uint8_t           s_video_back[DW * DH * 3]{};
static std::atomic<bool>    s_depth_dirty{false}, s_video_dirty{false}, s_pose_video_dirty{false};
static int                  s_video_skip = 30;
static std::atomic<bool>    s_running{true};
static std::atomic<int>     s_video_cb_count{0};   // total video_cb invocations
static std::atomic<int>     s_video_frame_count{0}; // frames actually written
static std::atomic<int>     s_depth_cb_count{0};   // total depth_cb invocations
static GLFWwindow*       s_win = nullptr;

// ── pose results ──────────────────────────────────────────────────────────────
struct PersonPose {
    DetectBox              box;
    std::vector<PosePoint> kps;
    std::vector<Joint3D>   joints;
};
static std::mutex              s_pose_mtx;
static std::vector<PersonPose> s_persons;
static std::atomic<bool>       s_pose_ready{false};

// ── 3D scene shaders ──────────────────────────────────────────────────────────
static const char* SCENE3D_VERT = R"glsl(
#version 410 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;
out vec4 vCol;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vCol = aCol;
}
)glsl";

static const char* SCENE3D_FRAG = R"glsl(
#version 410 core
in vec4 vCol;
out vec4 FragColor;
void main() { FragColor = vCol; }
)glsl";

// ── 3D scene state ────────────────────────────────────────────────────────────
static const int SCENE_W = 640, SCENE_H = 480;
static GLuint g_fbo = 0, g_fbo_col = 0, g_fbo_dep = 0;
static GLuint g_prog3d = 0, g_vao3d = 0, g_vbo3d = 0;
static float  g_cam_az   = 0.f;
static float  g_cam_el   = 0.35f;
static float  g_cam_dist = 2000.f;

struct SceneVert { float x,y,z,r,g,b,a; };
static std::vector<SceneVert> g_scene_buf;

// ── column-major 4×4 matrix ───────────────────────────────────────────────────
struct M4 { float m[16]{}; };

static M4 m4_mul(const M4& a, const M4& b) {
    M4 c{};
    for (int col=0; col<4; col++)
        for (int row=0; row<4; row++)
            for (int k=0; k<4; k++)
                c.m[col*4+row] += a.m[k*4+row] * b.m[col*4+k];
    return c;
}

static M4 m4_perspective(float fov_rad, float aspect, float zn, float zf) {
    float f = 1.f / tanf(fov_rad * 0.5f);
    M4 r{};
    r.m[0]=f/aspect; r.m[5]=f;
    r.m[10]=(zf+zn)/(zn-zf); r.m[11]=-1.f;
    r.m[14]=2.f*zf*zn/(zn-zf);
    return r;
}

// Look at origin (0,0,0) from (ex,ey,ez), up=(0,1,0).
static M4 m4_lookat(float ex, float ey, float ez) {
    float fx=-ex, fy=-ey, fz=-ez;
    float fl=sqrtf(fx*fx+fy*fy+fz*fz);
    if (fl<1e-6f) fl=1.f; fx/=fl; fy/=fl; fz/=fl;

    float upx=0.f, upy=1.f, upz=0.f;
    if (fabsf(fy) > 0.99f) { upx=0.f; upy=0.f; upz=(fy>0.f?-1.f:1.f); }

    float sx=fy*upz-fz*upy, sy=fz*upx-fx*upz, sz=fx*upy-fy*upx;
    float sl=sqrtf(sx*sx+sy*sy+sz*sz);
    if (sl<1e-6f) sl=1.f; sx/=sl; sy/=sl; sz/=sl;

    float ux=sy*fz-sz*fy, uy=sz*fx-sx*fz, uz=sx*fy-sy*fx;

    M4 r{};
    r.m[0]=sx; r.m[1]=ux; r.m[2]=-fx; r.m[3]=0.f;
    r.m[4]=sy; r.m[5]=uy; r.m[6]=-fy; r.m[7]=0.f;
    r.m[8]=sz; r.m[9]=uz; r.m[10]=-fz; r.m[11]=0.f;
    r.m[12]=-(sx*ex+sy*ey+sz*ez);
    r.m[13]=-(ux*ex+uy*ey+uz*ez);
    r.m[14]= (fx*ex+fy*ey+fz*ez);
    r.m[15]=1.f;
    return r;
}

static Vec3 quat_apply(Quat q, Vec3 v) {
    float tx=2.f*(q.y*v.z-q.z*v.y), ty=2.f*(q.z*v.x-q.x*v.z), tz=2.f*(q.x*v.y-q.y*v.x);
    return { v.x+q.w*tx+q.y*tz-q.z*ty,
             v.y+q.w*ty+q.z*tx-q.x*tz,
             v.z+q.w*tz+q.x*ty-q.y*tx };
}

// ── scene geometry builders ───────────────────────────────────────────────────
static void push_line(float x0,float y0,float z0, float x1,float y1,float z1,
                      float r,float g,float b, float a=1.f) {
    g_scene_buf.push_back({x0,y0,z0, r,g,b,a});
    g_scene_buf.push_back({x1,y1,z1, r,g,b,a});
}

// Draw a disk at (cx,cy,cz) with given normal and tangent (both unit vectors).
// The orientation indicator line points along +tangent direction.
static void push_disk(float cx, float cy, float cz,
                      float nx, float ny, float nz,
                      float tx, float ty, float tz,
                      float rad, float r, float g, float b) {
    // bitangent = cross(normal, tangent)
    float bx=ny*tz-nz*ty, by=nz*tx-nx*tz, bz=nx*ty-ny*tx;
    float bl=sqrtf(bx*bx+by*by+bz*bz);
    if (bl<1e-6f) return;
    bx/=bl; by/=bl; bz/=bl;

    static const int N = 24;
    static const float TAU = 6.28318530718f;
    float px=cx+rad*tx, py=cy+rad*ty, pz=cz+rad*tz;
    for (int i=1; i<=N; i++) {
        float a = TAU*i/N;
        float qx=cx+rad*(cosf(a)*tx+sinf(a)*bx);
        float qy=cy+rad*(cosf(a)*ty+sinf(a)*by);
        float qz=cz+rad*(cosf(a)*tz+sinf(a)*bz);
        push_line(px,py,pz, qx,qy,qz, r,g,b, 0.6f);
        px=qx; py=qy; pz=qz;
    }
    // Orientation indicator: yellow line from center to tangent direction
    push_line(cx,cy,cz, cx+rad*tx, cy+rad*ty, cz+rad*tz, 1.f,1.f,0.f);
}

// ── init 3D scene GL resources ────────────────────────────────────────────────
static void init_scene3d() {
    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);

    glGenTextures(1, &g_fbo_col);
    glBindTexture(GL_TEXTURE_2D, g_fbo_col);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCENE_W, SCENE_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_fbo_col, 0);

    glGenRenderbuffers(1, &g_fbo_dep);
    glBindRenderbuffer(GL_RENDERBUFFER, g_fbo_dep);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, SCENE_W, SCENE_H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_fbo_dep);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto compile = [](GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) { char buf[512]; glGetShaderInfoLog(s,512,nullptr,buf); fprintf(stderr,"Shader: %s\n",buf); }
        return s;
    };
    GLuint vs=compile(GL_VERTEX_SHADER, SCENE3D_VERT);
    GLuint fs=compile(GL_FRAGMENT_SHADER, SCENE3D_FRAG);
    g_prog3d = glCreateProgram();
    glAttachShader(g_prog3d, vs); glAttachShader(g_prog3d, fs);
    glLinkProgram(g_prog3d);
    glDeleteShader(vs); glDeleteShader(fs);

    glGenVertexArrays(1, &g_vao3d);
    glBindVertexArray(g_vao3d);
    glGenBuffers(1, &g_vbo3d);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo3d);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

// Render skeleton to FBO. Returns MVP for text projection.
static const float PERSON_COLORS[][3] = {
    {0.9f, 0.9f, 0.9f},
    {1.0f, 0.55f, 0.1f},
    {0.2f, 0.85f, 1.0f},
    {0.85f, 0.3f, 0.85f},
};
static const int NUM_PERSON_COLORS = (int)(sizeof(PERSON_COLORS)/sizeof(PERSON_COLORS[0]));

static M4 render_scene3d(const std::vector<PersonPose>& persons) {
    float el=g_cam_el, az=g_cam_az, dist=g_cam_dist;
    float ex=dist*cosf(el)*sinf(az), ey=dist*sinf(el), ez=dist*cosf(el)*cosf(az);
    M4 view = m4_lookat(ex, ey, ez);
    M4 proj = m4_perspective(0.785f, (float)SCENE_W/SCENE_H, 50.f, 10000.f);
    M4 mvp  = m4_mul(proj, view);

    g_scene_buf.clear();

    // World axes at origin (200 mm)
    push_line(0,0,0, 200,0,0,  1.f,.2f,.2f);
    push_line(0,0,0, 0,200,0,  .2f,1.f,.2f);
    push_line(0,0,0, 0,0,-200, .2f,.2f,1.f);

    // Centre view on first person's head
    Vec3 h{0,0,0};
    if (!persons.empty() && !persons[0].joints.empty() && persons[0].joints[0].valid)
        h = persons[0].joints[0].pos;

    auto kd = [&](Vec3 p) -> Vec3 {
        return { p.x-h.x, -(p.y-h.y), -(p.z-h.z) };
    };

    for (int pi = 0; pi < (int)persons.size(); pi++) {
        const auto& joints = persons[pi].joints;
        if (joints.empty()) continue;
        const float* col = PERSON_COLORS[pi % NUM_PERSON_COLORS];

        // Bones
        for (auto& [pa, ci] : COCO_BONES) {
            if (pa>=(int)joints.size()||ci>=(int)joints.size()) continue;
            if (!joints[pa].valid || !joints[ci].valid) continue;
            Vec3 a=kd(joints[pa].pos), b=kd(joints[ci].pos);
            push_line(a.x,a.y,a.z, b.x,b.y,b.z, col[0],col[1],col[2],.8f);
        }

        // Orientation disks at each valid joint
        for (int i=0; i<(int)joints.size(); i++) {
            if (!joints[i].valid) continue;
            Vec3 p = kd(joints[i].pos);

            Vec3 ax_k = quat_apply(joints[i].orientation, {0,1,0});
            Vec3 ax = v3norm({ax_k.x, -ax_k.y, -ax_k.z});
            Vec3 t_k = quat_apply(joints[i].orientation, {1,0,0});
            Vec3 t = v3norm({t_k.x, -t_k.y, -t_k.z});

            push_disk(p.x,p.y,p.z, ax.x,ax.y,ax.z, t.x,t.y,t.z,
                      40.f, col[0]*0.3f, col[1]*0.7f, col[2]*1.f);
        }
    }

    // Upload and draw
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glViewport(0, 0, SCENE_W, SCENE_H);
    glClearColor(0.07f, 0.07f, 0.12f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    if (!g_scene_buf.empty()) {
        glBindVertexArray(g_vao3d);
        glBindBuffer(GL_ARRAY_BUFFER, g_vbo3d);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(g_scene_buf.size()*sizeof(SceneVert)),
                     g_scene_buf.data(), GL_STREAM_DRAW);
        glUseProgram(g_prog3d);
        glUniformMatrix4fv(glGetUniformLocation(g_prog3d,"uMVP"), 1, GL_FALSE, mvp.m);
        glDrawArrays(GL_LINES, 0, (GLsizei)g_scene_buf.size());
        glBindVertexArray(0);
    }

    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return mvp;
}

// ── freenect callbacks ────────────────────────────────────────────────────────
static void sig_handler(int) {
    if (s_win) glfwSetWindowShouldClose(s_win, GLFW_TRUE);
}

static void depth_cb(freenect_device*, void* buf, uint32_t) {
    int n = ++s_depth_cb_count;
    if (n == 1) fprintf(stderr, "[depth] first callback received\n");

    auto* d = static_cast<uint16_t*>(buf);
    std::lock_guard<std::mutex> lk(s_depth_mtx);
    for (int i = 0; i < DW * DH; i++) {
        uint32_t raw = d[i] & 0x7FFu;
        s_depth_raw[i] = static_cast<uint16_t>(raw);
        uint8_t v = (raw == 2047u) ? 0u : static_cast<uint8_t>((2047u - raw) * 255u / 2047u);
        s_depth_pix[3*i+0] = v;
        s_depth_pix[3*i+1] = v;
        s_depth_pix[3*i+2] = v;
    }
    s_depth_dirty = true;
}

static void video_cb(freenect_device* dev, void* buf, uint32_t timestamp) {
    int n = ++s_video_cb_count;
    if (n == 1) fprintf(stderr, "[video] first callback received (ts=%u)\n", timestamp);

    if (s_video_skip > 0) {
        int remaining = --s_video_skip;
        if (remaining == 0)
            fprintf(stderr, "[video] skip done at callback #%d, now writing frames\n", n);
        return;
    }

    if (buf == nullptr) {
        fprintf(stderr, "[video] WARNING: null buffer at callback #%d\n", n);
        return;
    }

    std::lock_guard<std::mutex> lk(s_video_mtx);
    memcpy(s_video_pix, buf, DW * DH * 3);
    freenect_set_video_buffer(dev, s_video_back);
    s_video_dirty       = true;
    s_pose_video_dirty  = true;

    int fc = ++s_video_frame_count;
    if (fc == 1) fprintf(stderr, "[video] first frame written\n");
}

static void freenect_thread_fn(freenect_context* ctx) {
    fprintf(stderr, "[freenect] event thread started\n");
    timeval tv{0, 100'000};
    int iters = 0;
    while (s_running) {
        int ret = freenect_process_events_timeout(ctx, &tv);
        if (ret < 0)
            fprintf(stderr, "[freenect] process_events_timeout error: %d\n", ret);
        ++iters;
    }
    fprintf(stderr, "[freenect] event thread exiting\n");
}

// ── pose inference thread ─────────────────────────────────────────────────────
static float box_iou(const DetectBox& a, const DetectBox& b) {
    int ix = std::max(a.left, b.left),   iy = std::max(a.top, b.top);
    int ax = std::min(a.right, b.right), ay = std::min(a.bottom, b.bottom);
    if (ax <= ix || ay <= iy) return 0.f;
    float inter = (float)(ax-ix) * (float)(ay-iy);
    float ua    = (float)(a.right-a.left) * (float)(a.bottom-a.top);
    float ub    = (float)(b.right-b.left) * (float)(b.bottom-b.top);
    return inter / (ua + ub - inter);
}

static void pose_thread_fn() {
    try {
        RTMPoseTrackerOnnxruntime tracker(
            MODEL_DIR "/rtmdet-nano/end2end.onnx",
            MODEL_DIR "/rtmpose-s/end2end.onnx");

        cv::Mat               frame(DH, DW, CV_8UC3);
        std::vector<uint16_t> depth(DW * DH);

        // EMA state per tracked person (matched by box IoU across frames)
        std::vector<std::vector<Joint3D>> smooth_joints;
        std::vector<DetectBox>            smooth_boxes;
        constexpr float EMA_ALPHA = 0.4f; // weight on new sample; lower = smoother

        int pose_frame = 0;
        while (s_running) {
            if (!s_pose_video_dirty.exchange(false)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            {
                std::lock_guard<std::mutex> lk(s_video_mtx);
                memcpy(frame.data, s_video_pix, DW * DH * 3);
            }
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
            {
                std::lock_guard<std::mutex> lk(s_depth_mtx);
                memcpy(depth.data(), s_depth_raw, sizeof(s_depth_raw));
            }

            fprintf(stderr, "[pose] frame=%d video_frames=%d\n",
                    ++pose_frame, s_video_frame_count.load());

            auto detections = tracker.Inference(frame);
            int  n_new  = (int)detections.size();
            int  n_prev = (int)smooth_joints.size();

            // Greedy IoU matching: new detection → previous smoothed slot
            std::vector<int>  match(n_new, -1);
            std::vector<bool> used(n_prev, false);
            for (int i = 0; i < n_new; i++) {
                float best = 0.3f; int best_j = -1;
                for (int j = 0; j < n_prev; j++) {
                    if (used[j]) continue;
                    float iou = box_iou(detections[i].first, smooth_boxes[j]);
                    if (iou > best) { best = iou; best_j = j; }
                }
                if (best_j >= 0) { match[i] = best_j; used[best_j] = true; }
            }

            std::vector<std::vector<Joint3D>> new_smooth(n_new);
            std::vector<DetectBox>            new_boxes(n_new);
            std::vector<PersonPose>           persons(n_new);

            for (int i = 0; i < n_new; i++) {
                auto& [box, kps] = detections[i];
                auto joints = lift_to_3d(kps, depth.data());

                if (match[i] >= 0) {
                    const auto& prev = smooth_joints[match[i]];
                    for (int k = 0; k < (int)joints.size() && k < (int)prev.size(); k++) {
                        if (joints[k].valid && prev[k].valid) {
                            joints[k].pos.x = EMA_ALPHA*joints[k].pos.x + (1.f-EMA_ALPHA)*prev[k].pos.x;
                            joints[k].pos.y = EMA_ALPHA*joints[k].pos.y + (1.f-EMA_ALPHA)*prev[k].pos.y;
                            joints[k].pos.z = EMA_ALPHA*joints[k].pos.z + (1.f-EMA_ALPHA)*prev[k].pos.z;
                        }
                    }
                }

                new_smooth[i] = joints;
                new_boxes[i]  = box;
                persons[i]    = { box, kps, std::move(joints) };
            }

            smooth_joints = std::move(new_smooth);
            smooth_boxes  = std::move(new_boxes);

            {
                std::lock_guard<std::mutex> lk(s_pose_mtx);
                s_persons = std::move(persons);
            }
            s_pose_ready = true;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[pose] fatal: %s\n", e.what());
    }
}

// ── OpenGL helpers ────────────────────────────────────────────────────────────
static GLuint make_tex() {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, DW, DH, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    return t;
}

static void upload_tex(GLuint t, const uint8_t* rgb) {
    glBindTexture(GL_TEXTURE_2D, t);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DW, DH, GL_RGB, GL_UNSIGNED_BYTE, rgb);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    freenect_context* ctx = nullptr;
    if (freenect_init(&ctx, nullptr) < 0) {
        fprintf(stderr, "freenect_init failed\n");
        return 1;
    }
    freenect_set_log_level(ctx, FREENECT_LOG_DEBUG);

    if (freenect_num_devices(ctx) < 1) {
        fprintf(stderr, "No Kinect devices found\n");
        freenect_shutdown(ctx);
        return 1;
    }

    freenect_device* dev = nullptr;
    for (int i = 0; i < 5; i++) {
        if (i > 0) {
            freenect_shutdown(ctx);
            ctx = nullptr;
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
            if (freenect_init(&ctx, nullptr) < 0) break;
            freenect_set_log_level(ctx, FREENECT_LOG_DEBUG);
            if (freenect_num_devices(ctx) < 1) continue;
        }
        if (freenect_open_device(ctx, &dev, 0) == 0) break;
        fprintf(stderr, "Open attempt %d/5 failed, retrying...\n", i + 1);
        dev = nullptr;
    }
    if (!dev || !ctx) {
        fprintf(stderr, "Failed to open Kinect device\n");
        if (ctx) freenect_shutdown(ctx);
        return 1;
    }

    freenect_set_depth_callback(dev, depth_cb);
    freenect_set_video_callback(dev, video_cb);
    freenect_set_depth_mode(dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT));
    freenect_set_video_mode(dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));
    freenect_set_video_buffer(dev, s_video_back);
    {
        int r = freenect_start_depth(dev);
        fprintf(stderr, "[init] freenect_start_depth: %s (ret=%d)\n", r < 0 ? "FAILED" : "ok", r);
    }
    {
        int r = freenect_start_video(dev);
        fprintf(stderr, "[init] freenect_start_video: %s (ret=%d)\n", r < 0 ? "FAILED" : "ok", r);
    }

    std::thread fk_thread(freenect_thread_fn, ctx);
    std::thread pose_thread(pose_thread_fn);

    glfwSetErrorCallback([](int e, const char* m) { fprintf(stderr, "GLFW %d: %s\n", e, m); });
    if (!glfwInit()) { s_running = false; fk_thread.join(); pose_thread.join(); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    const int WIN_W = 3 * (DW + 16);
    GLFWwindow* win = glfwCreateWindow(WIN_W, 640, "Kinect + RTMPose", nullptr, nullptr);
    if (!win) { glfwTerminate(); s_running = false; fk_thread.join(); pose_thread.join(); return 1; }
    s_win = win;
    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    glewInit();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    GLuint depth_tex = make_tex();
    GLuint video_tex = make_tex();
    init_scene3d();

    // UI state
    static bool  g_show_boxes    = true;
    static bool  g_show_skeleton = true;
    static float g_conf_thresh   = 0.3f;

    // Joint names for text overlay
    static const char* JNAMES[17] = {
        "nose","Leye","Reye","Lear","Rear",
        "Lshl","Rshl","Lelb","Relb",
        "Lwst","Rwst","Lhip","Rhip",
        "Lkne","Rkne","Lank","Rank"
    };

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        if (s_depth_dirty.exchange(false)) {
            std::lock_guard<std::mutex> lk(s_depth_mtx);
            upload_tex(depth_tex, s_depth_pix);
        }
        if (s_video_dirty.exchange(false)) {
            std::lock_guard<std::mutex> lk(s_video_mtx);
            upload_tex(video_tex, s_video_pix);
        }

        std::vector<PersonPose> persons;
        if (s_pose_ready) {
            std::lock_guard<std::mutex> lk(s_pose_mtx);
            persons = s_persons;
        }

        // Render 3D scene to FBO before starting ImGui frame
        M4 scene_mvp = render_scene3d(persons);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ── depth ─────────────────────────────────────────────────────────────
        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({DW + 16.f, DH + 36.f}, ImGuiCond_Always);
        ImGui::Begin("Depth", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        ImGui::Image((ImTextureID)(uintptr_t)depth_tex, {DW, DH});
        ImGui::End();

        // ── video + skeleton overlay ──────────────────────────────────────────
        ImGui::SetNextWindowPos({DW + 16.f, 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({DW + 16.f, DH + 36.f}, ImGuiCond_Always);
        ImGui::Begin("Video", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
        ImGui::Image((ImTextureID)(uintptr_t)video_tex, {DW, DH});

        if (!persons.empty()) {
            static const ImU32 BOX_COLS[] = {
                IM_COL32(0, 255, 80, 220),
                IM_COL32(255, 180, 0, 220),
                IM_COL32(50, 220, 255, 220),
                IM_COL32(220, 75, 220, 220),
            };
            static const ImU32 KPS_BONE_COLS[] = {
                IM_COL32(0, 220, 0, 220),
                IM_COL32(255, 140, 25, 220),
                IM_COL32(50, 220, 255, 220),
                IM_COL32(220, 75, 220, 220),
            };
            static const ImU32 KPS_DOT_COLS[] = {
                IM_COL32(255, 100, 0, 230),
                IM_COL32(255, 200, 0, 230),
                IM_COL32(0, 200, 255, 230),
                IM_COL32(200, 0, 255, 230),
            };
            constexpr int N_COLS = 4;
            ImDrawList* dl  = ImGui::GetWindowDrawList();
            ImVec2      org = ImGui::GetItemRectMin();

            for (int pi = 0; pi < (int)persons.size(); pi++) {
                const auto& box = persons[pi].box;
                const auto& kps = persons[pi].kps;
                ImU32 box_col  = BOX_COLS[pi % N_COLS];
                ImU32 bone_col = KPS_BONE_COLS[pi % N_COLS];
                ImU32 dot_col  = KPS_DOT_COLS[pi % N_COLS];

                if (g_show_boxes && box.score >= g_conf_thresh) {
                    dl->AddRect(
                        {org.x + box.left,  org.y + box.top},
                        {org.x + box.right, org.y + box.bottom},
                        box_col, 0.f, 0, 2.f);
                    char conf_buf[16];
                    snprintf(conf_buf, sizeof(conf_buf), "%.0f%%", box.score * 100.f);
                    float tx = org.x + box.left + 3.f;
                    float ty = org.y + box.top  - 14.f;
                    dl->AddText({tx+1, ty+1}, IM_COL32(0,0,0,200),     conf_buf);
                    dl->AddText({tx,   ty},   box_col,                  conf_buf);
                }

                if (g_show_skeleton) {
                    for (auto& [a, b] : COCO_BONES) {
                        if (a >= (int)kps.size() || b >= (int)kps.size()) continue;
                        if (kps[a].score < g_conf_thresh || kps[b].score < g_conf_thresh) continue;
                        dl->AddLine({org.x + kps[a].x, org.y + kps[a].y},
                                    {org.x + kps[b].x, org.y + kps[b].y},
                                    bone_col, 2.f);
                    }
                    for (auto& kp : kps) {
                        if (kp.score < g_conf_thresh) continue;
                        dl->AddCircleFilled({org.x + kp.x, org.y + kp.y}, 4.f, dot_col);
                    }
                }
            }
        }
        ImGui::End();

        // ── controls panel ────────────────────────────────────────────────────
        ImGui::SetNextWindowPos({DW + 16.f, DH + 36.f}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({DW + 16.f, 68.f}, ImGuiCond_Always);
        ImGui::Begin("Controls", nullptr,
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
        ImGui::Checkbox("Boxes",    &g_show_boxes);
        ImGui::SameLine();
        ImGui::Checkbox("Skeleton", &g_show_skeleton);
        ImGui::SliderFloat("Confidence threshold", &g_conf_thresh, 0.f, 1.f, "%.2f");
        ImGui::End();

        // ── 3D skeleton scene ─────────────────────────────────────────────────
        ImGui::SetNextWindowPos({2.f * (DW + 16.f), 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({DW + 16.f, DH + 36.f}, ImGuiCond_Always);
        ImGui::Begin("3D View", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        // Display FBO texture (flip Y: uv0=(0,1), uv1=(1,0))
        ImGui::Image((ImTextureID)(uintptr_t)g_fbo_col, {SCENE_W, SCENE_H}, {0,1}, {1,0});
        ImVec2 scene_origin = ImGui::GetItemRectMin();

        // Orbit camera input
        if (ImGui::IsItemHovered()) {
            ImGuiIO& io = ImGui::GetIO();
            if (io.MouseDown[0]) {
                g_cam_az -= io.MouseDelta.x * 0.005f;
                g_cam_el += io.MouseDelta.y * 0.005f;
                g_cam_el  = std::clamp(g_cam_el, -1.4f, 1.4f);
            }
            g_cam_dist -= io.MouseWheel * 150.f;
            g_cam_dist  = std::max(300.f, g_cam_dist);
        }

        // Floating confidence text overlay (billboarded via projection)
        if (!persons.empty()) {
            ImDrawList* dl = ImGui::GetWindowDrawList();
            Vec3 h{0,0,0};
            if (!persons[0].joints.empty() && persons[0].joints[0].valid)
                h = persons[0].joints[0].pos;

            // Only annotate first person to avoid clutter
            const auto& pose_joints = persons[0].joints;
            for (int i=0; i<(int)pose_joints.size(); i++) {
                if (!pose_joints[i].valid) continue;

                float px = pose_joints[i].pos.x - h.x;
                float py = -(pose_joints[i].pos.y - h.y);
                float pz = -(pose_joints[i].pos.z - h.z);

                float cx = scene_mvp.m[0]*px + scene_mvp.m[4]*py + scene_mvp.m[8]*pz + scene_mvp.m[12];
                float cy = scene_mvp.m[1]*px + scene_mvp.m[5]*py + scene_mvp.m[9]*pz + scene_mvp.m[13];
                float cw = scene_mvp.m[3]*px + scene_mvp.m[7]*py + scene_mvp.m[11]*pz + scene_mvp.m[15];
                if (cw <= 0.f) continue;

                float sx = (cx/cw * 0.5f + 0.5f) * SCENE_W;
                float sy = (-cy/cw * 0.5f + 0.5f) * SCENE_H;

                char buf[32];
                snprintf(buf, sizeof(buf), "%s %.2f", JNAMES[i], pose_joints[i].confidence);
                float tx = scene_origin.x + sx + 4.f;
                float ty = scene_origin.y + sy - 8.f;
                dl->AddText({tx+1,ty+1}, IM_COL32(0,0,0,200), buf);
                dl->AddText({tx,ty},     IM_COL32(255,255,128,220), buf);
            }
        }

        ImGui::End();

        ImGui::Render();
        int fw, fh;
        glfwGetFramebufferSize(win, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.1f, 0.1f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    fprintf(stderr, "closing video streams\n");
    freenect_stop_depth(dev);
    freenect_stop_video(dev);
    s_running = false;
    fk_thread.join();
    pose_thread.join();

    freenect_close_device(dev);
    freenect_shutdown(ctx);

    glDeleteFramebuffers(1, &g_fbo);
    glDeleteTextures(1, &g_fbo_col);
    glDeleteRenderbuffers(1, &g_fbo_dep);
    glDeleteProgram(g_prog3d);
    glDeleteVertexArrays(1, &g_vao3d);
    glDeleteBuffers(1, &g_vbo3d);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
