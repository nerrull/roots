// Same "relay" growth as render_relay_gl.cpp (see that file for the full
// design writeup) but opens a VISIBLE window and swaps buffers live instead of
// piping to ffmpeg -- so you can watch the growth happen in real time. Closes
// on ESC or the window close button; after all hops finish, keeps orbiting the
// final result until you close it.
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "RootRenderer.h"
#include "RootSystem.h"
#include "SegmentAnalyser.h"
#include "MaskCavities.h"
#include "RootAttractors.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace CPlantBox;
using namespace maskcav;

// --- tiny OBJ loader + face GL pass (identical to render_mask_column_gl.cpp) -
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
static Vector3d toYup(const Vector3d& v) { return Vector3d(v.x, -v.z, v.y); }

static std::vector<float> buildFaceVertexData(const std::vector<MaskNode>& revealed,
                                              const std::vector<float>& fv, const std::vector<int>& ftris,
                                              float faceScale) {
    std::vector<float> out;
    for (const auto& m0 : revealed) {
        Vector3d n = toYup(m0.normal), t = toYup(m0.tangent), b = toYup(m0.bitangent);
        Vector3d p = toYup(m0.pos).minus(n.times(m0.r_depth * 0.5));
        float scale = faceScale * (float) std::min(m0.r_width, m0.r_height);
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

static double minDist(const std::vector<Vector3d>& nodes, const Vector3d& p) {
    double best = std::numeric_limits<double>::max();
    for (const auto& n : nodes) best = std::min(best, n.minus(p).length());
    return best;
}

// A finished hop's geometry, already translated into global coordinates.
struct FrozenHop {
    std::vector<Vector3d> nodes;
    std::vector<Vector2i> segs;
    std::vector<double> radii;
};

int main(int argc, char** argv) {
    std::string param = (argc > 1) ? argv[1]
        : "/Users/erichan/Documents/Development/jardins_racine/CPlantBox/"
          "modelparameter/structural/rootsystem/Glycine_max.xml";
    int N = (argc > 2) ? std::atoi(argv[2]) : 8;
    double maxHopDays = (argc > 3) ? std::atof(argv[3]) : 60.0;   // per-hop budget
    double weight = (argc > 4) ? std::atof(argv[4]) : 0.68;
    double dt = (argc > 5) ? std::atof(argv[5]) : 0.75;
    double reachMult = (argc > 6) ? std::atof(argv[6]) : 1.6;
    double dwellDays = (argc > 7) ? std::atof(argv[7]) : 18.0;
    int rimN = (argc > 8) ? std::atoi(argv[8]) : 6;
    int W = (argc > 9) ? std::atoi(argv[9]) : 900;
    int H = (argc > 10) ? std::atoi(argv[10]) : 1000;
    double spin = (argc > 11) ? std::atof(argv[11]) : 0.5;
    double R0 = (argc > 12) ? std::atof(argv[12]) : 13.0;
    double Hh = (argc > 13) ? std::atof(argv[13]) : 52.0;
    double startFrac = (argc > 14) ? std::atof(argv[14]) : 0.15;
    double dwellWeight = (argc > 15) ? std::atof(argv[15]) : 0.92;
    double viewCylLen = (argc > 16) ? std::atof(argv[16]) : 8.0;   // 0 disables
    int frameDelayMs = (argc > 17) ? std::atoi(argv[17]) : 90;     // paces playback beyond vsync

    double maskR = 2.6;
    double tipRadius = 0.22 * R0;
    auto masks = conePhyllotaxis(N, R0, Hh, maskR, startFrac, 0.94, tipRadius);

    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);   // live -- an actual window on screen
    GLFWwindow* window = glfwCreateWindow(W, H, "jardins_racine -- mask relay (live)", nullptr, nullptr);
    if (!window) { std::cerr << "glfwCreateWindow failed\n"; return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);   // vsync -- paces the loop to the display refresh
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "glewInit failed\n"; return 1; }

    // On Retina/HiDPI displays the actual framebuffer is larger (e.g. 2x) than
    // the window size passed to glfwCreateWindow (which is in logical/point
    // units). Render at the real pixel size, or the output only fills a
    // corner of the window (the rest of the framebuffer never gets drawn to).
    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    RootRenderer renderer(fbW, fbH);
    renderer.mat.baseColor[0] = 0.55f; renderer.mat.baseColor[1] = 0.40f; renderer.mat.baseColor[2] = 0.26f;
    renderer.mat.ambient = 0.18f;
    renderer.mat.diffuse = 0.75f;
    renderer.radiusScale = 1.4f;
    renderer.fog.density = 0.0f;
    renderer.wispCount = 0;

    FaceGL faceGL;
    faceGL.init(SHADER_DIR);
    std::vector<float> fv; std::vector<int> ftris;
    loadObj(FACE_OBJ_PATH, fv, ftris);
    normalizeMesh(fv);
    ftris = cropOvalTris(fv, ftris, 0.72f, 0.98f);

    // RootRenderer draws into its own internal FBO (colorTex()), not the
    // window's default framebuffer -- blit it to the screen with a trivial
    // fullscreen-triangle passthrough shader (no attributes; gl_VertexID trick).
    const char* blitVS = R"(#version 410 core
        out vec2 v_uv;
        void main() {
            vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
            v_uv = p;
            gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
        })";
    const char* blitFS = R"(#version 410 core
        in vec2 v_uv; out vec4 fragColor; uniform sampler2D u_tex;
        void main() { fragColor = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y)); })";
    GLuint blitProg = glCreateProgram();
    { GLuint vs = FaceGL::compile(GL_VERTEX_SHADER, blitVS), fs = FaceGL::compile(GL_FRAGMENT_SHADER, blitFS);
      glAttachShader(blitProg, vs); glAttachShader(blitProg, fs); glLinkProgram(blitProg);
      glDeleteShader(vs); glDeleteShader(fs); }
    GLint blitTexLoc = glGetUniformLocation(blitProg, "u_tex");
    GLuint blitVAO; glGenVertexArrays(1, &blitVAO);

    auto present = [&]() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbW, fbH);
        glUseProgram(blitProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, renderer.colorTex());
        glUniform1i(blitTexLoc, 0);
        glBindVertexArray(blitVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
        glUseProgram(0);
        glfwSwapBuffers(window);
        glfwPollEvents();
    };

    std::vector<MaskNode> revealed;
    std::vector<FrozenHop> frozen;
    Vector3d prevPos(0, 0, 0);
    float lightDir[3] = {0.5f, 0.8f, 0.35f};
    int frame = 0;

    for (int hop = 0; hop < N; hop++) {
        Vector3d targetGlobal = masks[hop].pos;

        auto rs = std::make_shared<RootSystem>();
        rs->readParameters(param, "plant", true, false);
        rs->initialize(false);
        auto seedNodes = rs->getNodes();
        Vector3d localSeed = seedNodes.empty() ? Vector3d(0, 0, 0) : seedNodes[0];
        Vector3d offset = prevPos.minus(localSeed);
        Vector3d localTarget = targetGlobal.minus(offset);

        std::vector<MaskNode> localRevealed;
        for (const auto& m : revealed) {
            MaskNode lm = m;
            lm.pos = m.pos.minus(offset);
            localRevealed.push_back(lm);
        }
        MaskNode localTargetNode = masks[hop];
        localTargetNode.pos = localTarget;

        auto base = std::make_shared<Gravitropism>(rs, 1.0, 0.2);
        // reach detection only re-steers headings -- CPlantBox has no "stop
        // growing" primitive, so a root already elongating under gravity keeps
        // going even after attraction retargets it. During dwell, gravity's
        // share of the blend drops sharply (dwellWeight >> weight) so growth
        // mostly loiters/curls near the target instead of racing on past it.
        auto rebuildTropism = [&](double w) {
            auto geom = buildCavityGeometry(localRevealed, R0, Hh, false, 2.0, tipRadius, viewCylLen);
            rs->setGeometry(geom);
            auto attrs = rimAttractors({localTargetNode}, rimN, 1.0, 3.0, 1.15);
            rs->setTropism(combinedAttraction(rs, base, attrs, 6.0, 0.25, w, geom), -1);
        };
        rebuildTropism(weight);

        std::cout << "hop " << hop << ": " << prevPos.x << "," << prevPos.y << "," << prevPos.z
                  << " -> " << targetGlobal.x << "," << targetGlobal.y << "," << targetGlobal.z << "\n";

        double day = 0.0, reachedDay = -1.0;
        bool reached = false;
        bool quit = false;
        while (day < maxHopDays) {
            if (glfwWindowShouldClose(window) || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                quit = true; break;
            }
            rs->simulate(dt, false);
            day += dt;

            // Fail-safe: if a hop can't steer close enough within most of its
            // budget (large offset targets, weak attraction pull), don't let it
            // keep falling under near-full gravity for the rest of maxHopDays --
            // force the gravity cutback anyway at whatever position it's reached.
            bool forced = !reached && day > 0.6 * maxHopDays;
            if (!reached) {
                double d = minDist(rs->getNodes(), localTarget);
                double thr = reachMult * std::max(masks[hop].r_width, masks[hop].r_height);
                if (d < thr || forced) {
                    reached = true; reachedDay = day;
                    localRevealed.push_back(localTargetNode);
                    revealed.push_back(masks[hop]);
                    rebuildTropism(dwellWeight);
                    std::cout << "  day " << day << ": " << (forced ? "gave up, capping" : "reached")
                              << ", dwelling (gravity share down to " << (1.0 - dwellWeight) << ")\n";
                }
            }
            if (reached && day - reachedDay > dwellDays) break;

            // ---- render: frozen hops (global) + this hop's live nodes (offset) ----
            SegmentAnalyser ana(*rs);
            auto radii = ana.getParameter("radius");

            std::vector<Vector3d> nodesYup;
            std::vector<Vector2i> segsAll;
            std::vector<double> radAll;
            int base_idx = 0;
            for (const auto& fh : frozen) {
                for (const auto& n : fh.nodes) nodesYup.push_back(toYup(n));
                for (const auto& s : fh.segs) segsAll.push_back(Vector2i(s.x + base_idx, s.y + base_idx));
                for (double r : fh.radii) radAll.push_back(r);
                base_idx += (int) fh.nodes.size();
            }
            for (const auto& n : ana.nodes) nodesYup.push_back(toYup(n.plus(offset)));
            for (const auto& s : ana.segments) segsAll.push_back(Vector2i(s.x + base_idx, s.y + base_idx));
            for (double r : radii) radAll.push_back(r);

            renderer.uploadSegments(nodesYup, segsAll, radAll);

            auto faceData = buildFaceVertexData(revealed, fv, ftris, 0.85f);
            faceGL.upload(faceData);

            // Camera tracks the current frontier (this hop's live growth + the
            // immediately preceding frozen hop + the hop target), not the whole
            // accumulated multi-hop journey -- fitting the entire relay chain in
            // one shot zooms out until later, longer travel cables go sub-pixel.
            std::vector<Vector3d> fitNodes;
            for (const auto& n : ana.nodes) fitNodes.push_back(toYup(n.plus(offset)));
            if (!frozen.empty())
                for (const auto& n : frozen.back().nodes) fitNodes.push_back(toYup(n));
            fitNodes.push_back(toYup(targetGlobal));
            Vector3d lo = fitNodes[0], hi = fitNodes[0];
            for (const auto& n : fitNodes) {
                lo = Vector3d(std::min(lo.x, n.x), std::min(lo.y, n.y), std::min(lo.z, n.z));
                hi = Vector3d(std::max(hi.x, n.x), std::max(hi.y, n.y), std::max(hi.z, n.z));
            }
            float target3[3] = {(float) (lo.x + hi.x) * 0.5f, (float) (lo.y + hi.y) * 0.5f, (float) (lo.z + hi.z) * 0.5f};
            float extent = (float) std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
            float radius = extent * 0.9f + 10.0f;
            float azimuth = 0.8f + 2.0f * (float) M_PI * (float) spin * (frame / 260.0f);

            renderer.render(azimuth, 0.12f, radius, target3, 0.5f, lightDir,
                            [&](const float* vp, const float* eye) { faceGL.draw(vp, eye, lightDir); });
            present();
            if (frameDelayMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(frameDelayMs));

            if (frame % 20 == 0) std::cout << "  frame " << frame << " day " << day
                                           << " segs(total) " << segsAll.size() << "\n";
            frame++;
        }
        if (quit) break;

        // freeze this hop for future hops' obstacle list + background rendering.
        SegmentAnalyser anaFinal(*rs);
        auto radiiFinal = anaFinal.getParameter("radius");
        FrozenHop fh;
        for (const auto& n : anaFinal.nodes) fh.nodes.push_back(n.plus(offset));
        fh.segs = anaFinal.segments;
        fh.radii = radiiFinal;
        frozen.push_back(fh);

        prevPos = targetGlobal;
    }

    std::cout << "growth complete (" << frame << " frames) -- orbiting the result. "
              << "Press ESC or close the window to exit.\n";

    // keep the window open, slowly orbiting the finished result, until closed.
    while (!glfwWindowShouldClose(window) && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS) {
        std::vector<Vector3d> nodesYup;
        std::vector<Vector2i> segsAll;
        std::vector<double> radAll;
        int base_idx = 0;
        for (const auto& fh : frozen) {
            for (const auto& n : fh.nodes) nodesYup.push_back(toYup(n));
            for (const auto& s : fh.segs) segsAll.push_back(Vector2i(s.x + base_idx, s.y + base_idx));
            for (double r : fh.radii) radAll.push_back(r);
            base_idx += (int) fh.nodes.size();
        }
        renderer.uploadSegments(nodesYup, segsAll, radAll);
        auto faceData = buildFaceVertexData(revealed, fv, ftris, 0.85f);
        faceGL.upload(faceData);

        Vector3d lo = nodesYup.empty() ? Vector3d(0, 0, 0) : nodesYup[0], hi = lo;
        for (const auto& n : nodesYup) {
            lo = Vector3d(std::min(lo.x, n.x), std::min(lo.y, n.y), std::min(lo.z, n.z));
            hi = Vector3d(std::max(hi.x, n.x), std::max(hi.y, n.y), std::max(hi.z, n.z));
        }
        float target3[3] = {(float) (lo.x + hi.x) * 0.5f, (float) (lo.y + hi.y) * 0.5f, (float) (lo.z + hi.z) * 0.5f};
        float extent = (float) std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
        float radius = extent * 0.75f + 12.0f;
        float azimuth = 0.8f + 0.15f * (float) frame / 60.0f;

        renderer.render(azimuth, 0.15f, radius, target3, 0.5f, lightDir,
                        [&](const float* vp, const float* eye) { faceGL.draw(vp, eye, lightDir); });
        present();
        frame++;
    }

    glfwTerminate();
    return 0;
}
