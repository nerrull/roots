#include <libfreenect/libfreenect.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "pose/rtmpose_tracker_onnxruntime.h"
#include "pose/skeleton_3d.h"

#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sys/time.h>
#include <thread>

static const int DW = 640, DH = 480;

// ── shared camera buffers ─────────────────────────────────────────────────────
static std::mutex        s_depth_mtx, s_video_mtx;
static uint8_t           s_depth_pix[DW * DH * 3]{};
static uint16_t          s_depth_raw[DW * DH]{};     // 11-bit raw disparity per pixel
static uint8_t           s_video_pix[DW * DH * 3]{};
static uint8_t           s_video_back[DW * DH * 3]{};
static std::atomic<bool> s_depth_dirty{false}, s_video_dirty{false};
static std::atomic<bool> s_running{true};
static GLFWwindow*       s_win = nullptr;

// ── pose results ──────────────────────────────────────────────────────────────
static std::mutex              s_pose_mtx;
static DetectBox               s_pose_box;
static std::vector<PosePoint>  s_pose_kps;
static std::vector<Joint3D>    s_pose_joints;
static std::atomic<bool>       s_pose_ready{false};

// ── freenect callbacks ────────────────────────────────────────────────────────
static void sig_handler(int) {
    if (s_win) glfwSetWindowShouldClose(s_win, GLFW_TRUE);
}

static void depth_cb(freenect_device*, void* buf, uint32_t) {
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

static void video_cb(freenect_device* dev, void* buf, uint32_t) {
    std::lock_guard<std::mutex> lk(s_video_mtx);
    memcpy(s_video_pix, buf, DW * DH * 3);
    freenect_set_video_buffer(dev, s_video_back);
    s_video_dirty = true;
}

static void freenect_thread_fn(freenect_context* ctx) {
    timeval tv{0, 100'000};
    while (s_running)
        freenect_process_events_timeout(ctx, &tv);
}

// ── pose inference thread ─────────────────────────────────────────────────────
static void pose_thread_fn() {
    try {
        RTMPoseTrackerOnnxruntime tracker(
            MODEL_DIR "/rtmdet-nano/end2end.onnx",
            MODEL_DIR "/rtmpose-s/end2end.onnx");

        cv::Mat              frame(DH, DW, CV_8UC3);
        std::vector<uint16_t> depth(DW * DH);

        while (s_running) {
            {
                std::lock_guard<std::mutex> lk(s_video_mtx);
                memcpy(frame.data, s_video_pix, DW * DH * 3);
            }
            // Kinect gives RGB; RTMDet/Pose pipeline expects BGR
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

            {
                std::lock_guard<std::mutex> lk(s_depth_mtx);
                memcpy(depth.data(), s_depth_raw, sizeof(s_depth_raw));
            }

            auto [box, kps] = tracker.Inference(frame);
            auto joints     = lift_to_3d(kps, depth.data());

            {
                std::lock_guard<std::mutex> lk(s_pose_mtx);
                s_pose_box    = box;
                s_pose_kps    = kps;
                s_pose_joints = joints;
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
    freenect_set_log_level(ctx, FREENECT_LOG_WARNING);

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
            freenect_set_log_level(ctx, FREENECT_LOG_WARNING);
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
    if (freenect_start_depth(dev) < 0) fprintf(stderr, "freenect_start_depth failed\n");
    if (freenect_start_video(dev) < 0) fprintf(stderr, "freenect_start_video failed\n");

    std::thread fk_thread(freenect_thread_fn, ctx);
    std::thread pose_thread(pose_thread_fn);

    glfwSetErrorCallback([](int e, const char* m) { fprintf(stderr, "GLFW %d: %s\n", e, m); });
    if (!glfwInit()) { s_running = false; fk_thread.join(); pose_thread.join(); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* win = glfwCreateWindow(1650, 560, "Kinect + RTMPose", nullptr, nullptr);
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

        // snapshot pose state for this frame
        DetectBox              pose_box;
        std::vector<PosePoint> pose_kps;
        std::vector<Joint3D>   pose_joints;
        if (s_pose_ready) {
            std::lock_guard<std::mutex> lk(s_pose_mtx);
            pose_box    = s_pose_box;
            pose_kps    = s_pose_kps;
            pose_joints = s_pose_joints;
        }

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

        if (!pose_kps.empty()) {
            ImDrawList* dl  = ImGui::GetWindowDrawList();
            ImVec2      org = ImGui::GetItemRectMin();

            for (auto& [a, b] : COCO_BONES) {
                if (a >= (int)pose_kps.size() || b >= (int)pose_kps.size()) continue;
                if (pose_kps[a].score < 0.3f || pose_kps[b].score < 0.3f) continue;
                dl->AddLine({org.x + pose_kps[a].x, org.y + pose_kps[a].y},
                             {org.x + pose_kps[b].x, org.y + pose_kps[b].y},
                             IM_COL32(0, 220, 0, 220), 2.f);
            }
            for (auto& kp : pose_kps) {
                if (kp.score < 0.3f) continue;
                dl->AddCircleFilled({org.x + kp.x, org.y + kp.y},
                                    4.f, IM_COL32(255, 100, 0, 230));
            }
        }
        ImGui::End();

        // ── 3D joint info ─────────────────────────────────────────────────────
        ImGui::SetNextWindowPos({2.f * (DW + 16.f), 0}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({330.f, DH + 36.f}, ImGuiCond_Always);
        ImGui::Begin("Skeleton 3D", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        if (!s_pose_ready) {
            ImGui::TextDisabled("initializing model...");
        } else if (!pose_box.IsValid()) {
            ImGui::TextDisabled("no person detected");
        } else {
            ImGui::Text("bbox  %d %d %d %d  (%.2f)",
                        pose_box.left, pose_box.top,
                        pose_box.right, pose_box.bottom, pose_box.score);
            ImGui::Separator();

            static const char* JOINT_NAMES[] = {
                "nose","L-eye","R-eye","L-ear","R-ear",
                "L-shldr","R-shldr","L-elbow","R-elbow",
                "L-wrist","R-wrist","L-hip","R-hip",
                "L-knee","R-knee","L-ankle","R-ankle"
            };

            if (ImGui::BeginTable("joints", 4,
                    ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY))
            {
                ImGui::TableSetupColumn("joint", ImGuiTableColumnFlags_WidthFixed, 60.f);
                ImGui::TableSetupColumn("X mm",  ImGuiTableColumnFlags_WidthFixed, 70.f);
                ImGui::TableSetupColumn("Y mm",  ImGuiTableColumnFlags_WidthFixed, 70.f);
                ImGui::TableSetupColumn("Z mm",  ImGuiTableColumnFlags_WidthFixed, 70.f);
                ImGui::TableHeadersRow();

                for (int i = 0; i < 17 && i < (int)pose_joints.size(); i++) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(JOINT_NAMES[i]);
                    if (pose_joints[i].valid) {
                        ImGui::TableSetColumnIndex(1); ImGui::Text("%.0f", pose_joints[i].pos.x);
                        ImGui::TableSetColumnIndex(2); ImGui::Text("%.0f", pose_joints[i].pos.y);
                        ImGui::TableSetColumnIndex(3); ImGui::Text("%.0f", pose_joints[i].pos.z);
                    } else {
                        ImGui::TableSetColumnIndex(1); ImGui::TextDisabled("--");
                    }
                }
                ImGui::EndTable();
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

    fprintf(stderr, "closing video streams\n")
    // Stop streams while the event thread is still running.
    freenect_stop_depth(dev);
    freenect_stop_video(dev);
    s_running = false;
    fk_thread.join();
    pose_thread.join();

    freenect_close_device(dev);
    freenect_shutdown(ctx);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
