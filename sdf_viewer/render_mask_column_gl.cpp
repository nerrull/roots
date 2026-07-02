// GPU render of the mask-cavity/attractor root column: grows the sequential-
// reveal root system (same mechanics as export_sequential.cpp) and renders it
// live with RootRenderer's GPU sphere-tracer for root capsules (real depth,
// real shading) plus a rasterized triangle pass for the face-mask reliefs,
// sharing RootRenderer's depth buffer so the two composite correctly. Pipes
// raw RGB frames straight to ffmpeg -- no text dump, no numpy, no per-triangle
// Python loop. Runs a hidden (offscreen) GLFW window, so no display needed.
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "RootRenderer.h"
#include "RootSystem.h"
#include "SegmentAnalyser.h"
#include "MaskCavities.h"
#include "RootAttractors.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace CPlantBox;
using namespace maskcav;

// --- tiny OBJ loader (positions + triangle faces only) --------------------
static void loadObj(const std::string& path, std::vector<float>& verts, std::vector<int>& tris) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("cannot open " + path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tag; ss >> tag;
        if (tag == "v") {
            float x, y, z; ss >> x >> y >> z;
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (tag == "f") {
            int idx[3]; std::string tok;
            for (int i = 0; i < 3 && ss >> tok; i++) idx[i] = std::atoi(tok.c_str()) - 1;
            tris.push_back(idx[0]); tris.push_back(idx[1]); tris.push_back(idx[2]);
        }
    }
}

static void normalizeMesh(std::vector<float>& v) {
    float cx = 0, cy = 0, cz = 0;
    size_t n = v.size() / 3;
    for (size_t i = 0; i < n; i++) { cx += v[i*3]; cy += v[i*3+1]; cz += v[i*3+2]; }
    cx /= n; cy /= n; cz /= n;
    float m = 1e-9f;
    for (size_t i = 0; i < n; i++) {
        v[i*3] -= cx; v[i*3+1] -= cy; v[i*3+2] -= cz;
        m = std::max({m, std::fabs(v[i*3]), std::fabs(v[i*3+1]), std::fabs(v[i*3+2])});
    }
    for (auto& x : v) x /= m;
}

// crop to an oval in the local xy plane (trims the ragged mesh boundary) --
// mirrors columns.scene.crop_oval in the Python pipeline.
static std::vector<int> cropOvalTris(const std::vector<float>& v, const std::vector<int>& tris,
                                     float rx, float ry) {
    std::vector<int> out;
    for (size_t t = 0; t < tris.size(); t += 3) {
        float cx = 0, cy = 0;
        for (int k = 0; k < 3; k++) { cx += v[tris[t+k]*3]; cy += v[tris[t+k]*3+1]; }
        cx /= 3; cy /= 3;
        if ((cx/rx)*(cx/rx) + (cy/ry)*(cy/ry) < 1.0f)
            for (int k = 0; k < 3; k++) out.push_back(tris[t+k]);
    }
    return out;
}

// --- face-mesh GL pass ------------------------------------------------------
struct FaceGL {
    GLuint prog = 0, vao = 0, vbo = 0;
    int uViewProj = -1, uEye = -1, uLightDir = -1;
    int vertCount = 0;

    static GLuint compile(GLenum type, const std::string& src) {
        GLuint s = glCreateShader(type);
        const char* c = src.c_str();
        glShaderSource(s, 1, &c, nullptr);
        glCompileShader(s);
        GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[4096]; glGetShaderInfoLog(s, sizeof(log), nullptr, log);
                  throw std::runtime_error(std::string("face shader error:\n") + log); }
        return s;
    }
    static std::string readFile(const std::string& p) {
        std::ifstream f(p);
        return { std::istreambuf_iterator<char>(f), {} };
    }

    void init(const std::string& shaderDir) {
        GLuint vs = compile(GL_VERTEX_SHADER,   readFile(shaderDir + "/face.vert"));
        GLuint fs = compile(GL_FRAGMENT_SHADER, readFile(shaderDir + "/face.frag"));
        prog = glCreateProgram();
        glAttachShader(prog, vs); glAttachShader(prog, fs);
        glLinkProgram(prog);
        glDeleteShader(vs); glDeleteShader(fs);
        uViewProj = glGetUniformLocation(prog, "u_viewProj");
        uEye      = glGetUniformLocation(prog, "u_eye");
        uLightDir = glGetUniformLocation(prog, "u_lightDir");
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }

    // data: interleaved pos(3) normal(3) color(3) per vertex, already expanded
    // (no index buffer -- fine at this triangle count, keeps upload simple).
    void upload(const std::vector<float>& data) {
        vertCount = (int) (data.size() / 9);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*) 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*) (3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*) (6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glBindVertexArray(0);
    }

    void draw(const float* viewProj, const float* eye, const float* lightDir) {
        if (vertCount == 0) return;
        glUseProgram(prog);
        glUniformMatrix4fv(uViewProj, 1, GL_FALSE, viewProj);
        glUniform3fv(uEye, 1, eye);
        glUniform3fv(uLightDir, 1, lightDir);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, vertCount);
        glBindVertexArray(0);
        glUseProgram(0);
    }
};

static float MARBLE[3] = {0.86f, 0.83f, 0.78f};

// CPlantBox is Z-down (gravitropism pulls -z); RootRenderer's orbit camera
// hardcodes world-up = +Y. Remap render-space geometry (x, -z, y) to match --
// simulation stays in native CPlantBox coordinates, only this copy is swapped.
static Vector3d toYup(const Vector3d& v) { return Vector3d(v.x, -v.z, v.y); }

// build the interleaved pos/normal/color buffer for the currently-revealed
// masks, each holding a cropped-oval copy of the face mesh at its cavity.
static std::vector<float> buildFaceVertexData(const std::vector<MaskNode>& revealed,
                                              const std::vector<float>& fv, const std::vector<int>& ftris,
                                              float faceScale) {
    std::vector<float> out;
    for (const auto& m0 : revealed) {
        Vector3d n = toYup(m0.normal), t = toYup(m0.tangent), b = toYup(m0.bitangent);
        MaskNode m = m0;
        Vector3d p = toYup(m0.pos).minus(n.times(m0.r_depth * 0.5));
        float scale = faceScale * (float) std::min(m.r_width, m.r_height);
        for (size_t i = 0; i < ftris.size(); i += 3) {
            Vector3d verts3[3];
            for (int k = 0; k < 3; k++) {
                int vi = ftris[i+k];
                Vector3d local(fv[vi*3], fv[vi*3+1], fv[vi*3+2]);
                verts3[k] = p.plus(t.times(local.x * scale)).plus(b.times(local.y * scale))
                            .plus(n.times(local.z * scale));
            }
            Vector3d fn = (verts3[1].minus(verts3[0])).cross(verts3[2].minus(verts3[0]));
            double fl = fn.length(); if (fl > 1e-9) fn = fn.times(1.0 / fl);
            for (int k = 0; k < 3; k++) {
                out.push_back((float) verts3[k].x); out.push_back((float) verts3[k].y); out.push_back((float) verts3[k].z);
                out.push_back((float) fn.x); out.push_back((float) fn.y); out.push_back((float) fn.z);
                out.push_back(MARBLE[0]); out.push_back(MARBLE[1]); out.push_back(MARBLE[2]);
            }
        }
    }
    return out;
}

static double minDistToMask(const std::vector<Vector3d>& nodes, const MaskNode& m) {
    double best = std::numeric_limits<double>::max();
    for (const auto& p : nodes) best = std::min(best, p.minus(m.pos).length());
    return best;
}

int main(int argc, char** argv) {
    std::string param = (argc > 1) ? argv[1]
        : "/Users/erichan/Documents/Development/jardins_racine/CPlantBox/"
          "modelparameter/structural/rootsystem/Heliantus_Pages_2013.xml";
    std::string outPath = (argc > 2) ? argv[2] : "growth.mp4";
    int N = (argc > 3) ? std::atoi(argv[3]) : 8;
    double maxDays = (argc > 4) ? std::atof(argv[4]) : 90.0;
    double weight = (argc > 5) ? std::atof(argv[5]) : 0.55;
    double dt = (argc > 6) ? std::atof(argv[6]) : 0.75;
    double reachMult = (argc > 7) ? std::atof(argv[7]) : 1.3;
    double tailDays = (argc > 8) ? std::atof(argv[8]) : 6.0;
    double dwellDays = (argc > 9) ? std::atof(argv[9]) : 9.0;  // keep attracting after first contact
    int rimN = (argc > 10) ? std::atoi(argv[10]) : 6;
    int W = (argc > 11) ? std::atoi(argv[11]) : 800;
    int H = (argc > 12) ? std::atoi(argv[12]) : 1000;
    double spin = (argc > 13) ? std::atof(argv[13]) : 0.5;

    // ---- geometry / growth setup (same mechanics as export_sequential.cpp) --
    double R0 = 12.0, Hh = 48.0, maskR = 2.6;
    double tipRadius = 0.22 * R0;
    auto masks = conePhyllotaxis(N, R0, Hh, maskR, 0.14, 0.94, tipRadius);

    auto rs = std::make_shared<RootSystem>();
    rs->readParameters(param, "plant", true, false);
    rs->initialize(false);

    auto base = std::make_shared<Gravitropism>(rs, 1.0, 0.2);
    std::vector<MaskNode> revealed;
    int target = 0;
    auto applyState = [&]() {
        auto geom = buildCavityGeometry(revealed, R0, Hh, true, 2.0, tipRadius);
        rs->setGeometry(geom);
        if (target < (int) masks.size()) {
            auto attrs = rimAttractors({masks[target]}, rimN, 1.0, 3.0, 1.15);
            rs->setTropism(combinedAttraction(rs, base, attrs, 6.0, 0.25, weight, geom), -1);
        } else {
            rs->setTropism(base, -1);
        }
    };
    applyState();

    // ---- offscreen GL context ------------------------------------------------
    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);   // hidden -- offscreen rendering only
    GLFWwindow* window = glfwCreateWindow(W, H, "render_mask_column_gl", nullptr, nullptr);
    if (!window) { std::cerr << "glfwCreateWindow failed\n"; return 1; }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "glewInit failed\n"; return 1; }

    RootRenderer renderer(W, H);
    renderer.mat.baseColor[0] = 0.55f; renderer.mat.baseColor[1] = 0.40f; renderer.mat.baseColor[2] = 0.26f;
    renderer.mat.ambient = 0.18f;
    renderer.mat.diffuse = 0.75f;
    renderer.radiusScale = 1.4f;
    renderer.fog.density = 0.0f;   // keep it clean for now -- pure geometry
    renderer.wispCount = 0;        // RootRenderer defaults 2 wisps at origin -- don't want them here

    FaceGL faceGL;
    faceGL.init(SHADER_DIR);

    std::vector<float> fv; std::vector<int> ftris;
    loadObj(FACE_OBJ_PATH, fv, ftris);
    normalizeMesh(fv);
    ftris = cropOvalTris(fv, ftris, 0.72f, 0.98f);
    std::cout << "face mesh: " << fv.size()/3 << " verts, " << ftris.size()/3 << " tris (cropped)\n";

    // ---- ffmpeg pipe -----------------------------------------------------
    std::ostringstream cmd;
    cmd << "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s " << W << "x" << H
        << " -r 20 -i - -c:v libx264 -profile:v main -pix_fmt yuv420p -crf 20 -movflags +faststart -vf vflip "
        << "'" << outPath << "'";
    FILE* ff = popen(cmd.str().c_str(), "w");
    if (!ff) { std::cerr << "failed to launch ffmpeg\n"; return 1; }
    std::vector<unsigned char> pixels(W * H * 3);

    // ---- growth + render loop ----------------------------------------------
    double day = 0.0, tailStart = -1.0;
    bool dwelling = false;
    double reachedDay = 0.0;
    int frame = 0;
    float lightDir[3] = {0.5f, 0.8f, 0.35f};
    while (day < maxDays) {
        rs->simulate(dt, false);
        day += dt;

        if (target < (int) masks.size() && !dwelling) {
            double d = minDistToMask(rs->getNodes(), masks[target]);
            double thr = reachMult * std::max({masks[target].r_width, masks[target].r_height});
            if (d < thr) {
                revealed.push_back(masks[target]);
                dwelling = true;
                reachedDay = day;
                applyState();
                std::cout << "day " << day << ": reached mask " << (target) << ", dwelling\n";
            }
        } else if (dwelling && day - reachedDay > dwellDays) {
            target++;
            dwelling = false;
            if (target >= (int) masks.size()) tailStart = day;
            applyState();
        }
        if (tailStart > 0 && day - tailStart > tailDays) break;

        SegmentAnalyser ana(*rs);
        auto radii = ana.getParameter("radius");
        std::vector<Vector3d> nodesYup;
        nodesYup.reserve(ana.nodes.size());
        for (const auto& n : ana.nodes) nodesYup.push_back(toYup(n));
        renderer.uploadSegments(nodesYup, ana.segments, radii);

        auto faceData = buildFaceVertexData(revealed, fv, ftris, 0.85f);
        faceGL.upload(faceData);

        float top = 0.0f, bottom = 0.0f;
        for (const auto& n : ana.nodes) { top = std::max(top, (float) -n.z); bottom = std::min(bottom, (float) -n.z); }
        float target3[3] = {0.0f, (top + bottom) * 0.5f, 0.0f};
        float radius = top * 1.05f + 9.0f;
        float azimuth = 0.8f + 2.0f * (float) M_PI * (float) spin * (frame / 200.0f);   // slow continuous turn

        renderer.render(azimuth, 0.12f, radius, target3, 0.5f, lightDir,
                        [&](const float* vp, const float* eye) {
                            faceGL.draw(vp, eye, lightDir);
                        });

        glBindFramebuffer(GL_FRAMEBUFFER, 0);   // colorTex() lives in the fog FBO; read via glGetTexImage instead
        glBindTexture(GL_TEXTURE_2D, renderer.colorTex());
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        fwrite(pixels.data(), 1, pixels.size(), ff);

        if (frame % 10 == 0) std::cout << "  frame " << frame << "  day " << day
                                       << "  segs " << ana.segments.size() << "\n";
        frame++;
    }
    pclose(ff);
    std::cout << "wrote " << frame << " frames, " << revealed.size() << "/" << masks.size()
              << " masks revealed -> " << outPath << "\n";
    glfwTerminate();
    return 0;
}
