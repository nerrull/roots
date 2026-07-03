// Interactive version of the mask-relay live viewer: ImGui panel with sliders
// for every growth parameter (species, mask count, spacing, dwell, attraction
// strength, root thickness range) and a "Regrow" button that reruns the whole
// hop sequence from scratch with the current values. Same relay growth
// mechanics as render_relay_gl.cpp / render_relay_live.cpp -- see those files
// for the full design writeup.
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

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

// --- tiny OBJ loader + face GL pass (same as render_relay_live.cpp) --------
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
    int uLightIntensity = -1, uLightFalloff = -1, uSpecStrength = -1;
    int uVeinColor = -1, uVeinScale = -1, uVeinStrength = -1;
    int vertCount = 0;

    // material knobs -- set from Params before each draw() call.
    float lightIntensity = 3.2f, lightFalloff = 0.012f, specStrength = 1.2f;
    float veinColor[3] = {0.55f, 0.53f, 0.50f};
    float veinScale = 0.6f, veinStrength = 0.5f;

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
        uLightIntensity = glGetUniformLocation(prog, "u_lightIntensity");
        uLightFalloff   = glGetUniformLocation(prog, "u_lightFalloff");
        uSpecStrength   = glGetUniformLocation(prog, "u_specStrength");
        uVeinColor      = glGetUniformLocation(prog, "u_veinColor");
        uVeinScale      = glGetUniformLocation(prog, "u_veinScale");
        uVeinStrength   = glGetUniformLocation(prog, "u_veinStrength");
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }

    void upload(const std::vector<float>& data) {
        vertCount = (int) (data.size() / 12);   // pos3, normal3, color3, lightPos3
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*) 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*) (3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*) (6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*) (9 * sizeof(float)));
        glEnableVertexAttribArray(3);
        glBindVertexArray(0);
    }

    void draw(const float* viewProj, const float* eye, const float* lightDir) {
        if (vertCount == 0) return;
        glUseProgram(prog);
        glUniformMatrix4fv(uViewProj, 1, GL_FALSE, viewProj);
        glUniform3fv(uEye, 1, eye);
        glUniform3fv(uLightDir, 1, lightDir);
        glUniform1f(uLightIntensity, lightIntensity);
        glUniform1f(uLightFalloff, lightFalloff);
        glUniform1f(uSpecStrength, specStrength);
        glUniform3fv(uVeinColor, 1, veinColor);
        glUniform1f(uVeinScale, veinScale);
        glUniform1f(uVeinStrength, veinStrength);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, vertCount);
        glBindVertexArray(0);
        glUseProgram(0);
    }
};

static Vector3d toYup(const Vector3d& v) { return Vector3d(v.x, -v.z, v.y); }

static std::vector<float> buildFaceVertexData(const std::vector<MaskNode>& revealed,
                                              const std::vector<float>& fv, const std::vector<int>& ftris,
                                              float faceScale, float lightDist, const float* maskColor) {
    std::vector<float> out;
    for (const auto& m0 : revealed) {
        Vector3d n = toYup(m0.normal), t = toYup(m0.tangent), b = toYup(m0.bitangent);
        Vector3d p = toYup(m0.pos).minus(n.times(m0.r_depth * 0.5));
        float scale = faceScale * (float) std::min(m0.r_width, m0.r_height);
        // a small point light floating in front of the face, in the same
        // direction the view-clear cylinder keeps open -- so it lights the
        // face directly instead of relying on whatever the scene-wide
        // directional light happens to be doing from the current orbit angle.
        Vector3d lightPos = p.plus(n.times((double) lightDist));
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
                out.push_back(maskColor[0]); out.push_back(maskColor[1]); out.push_back(maskColor[2]);
                out.push_back((float) lightPos.x); out.push_back((float) lightPos.y); out.push_back((float) lightPos.z);
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

struct FrozenHop {
    std::vector<Vector3d> nodes;
    std::vector<Vector2i> segs;
    std::vector<double> radii;
};

// --- tunable params, all exposed in the ImGui panel -------------------------
struct Params {
    int speciesIdx   = 1;    // index into g_species below
    int N            = 8;    // number of masks
    float R0         = 13.0f;
    float Hh         = 52.0f;
    float startFrac  = 0.15f;
    // mask distribution: angle/distance between consecutive masks, independent
    // of N -- see conePhyllotaxis() in MaskCavities.h. angleStepGoldenMult is
    // a multiplier on the golden angle (~137.5deg) rather than an absolute
    // value, so 1.0 is the classic non-repeating phyllotactic spiral and
    // small departures from it (0.9, 1.1, ...) still look organic rather than
    // snapping into a regular repeating pattern the way an arbitrary absolute
    // angle would. distStepFrac <=0 falls back to spreading N masks evenly
    // over the cone (the old behavior).
    float angleStepGoldenMult = 1.0f;
    float distStepFrac = 0.0f;   // 0 = auto (spread N evenly), else explicit step
    // 1.0 = straight cone. <1 opens up fast near the tip and stays close to
    // full width for most of the length (cylinder-ish). >1 stays narrow
    // longer and flares sharply near the base (spike-ish).
    float taperPower = 1.0f;
    float dwellDays  = 18.0f;
    // Default pushed high: at low weight, gravity's residual pull competes
    // with attraction on every dicing trial, and can drag the root off target
    // over a long unconstrained stretch (that's the actual "way off course"
    // mechanism, not just bad luck). At weight ~1.0 the dicing objective is
    // *purely* alignment with the target -- gravity has no vote, so every
    // trial is scored on how well it points at the mask. Organic-looking
    // wander is still available via the "Angular jitter" (sigma) slider,
    // which adds randomness independent of this weight, so reliability and
    // some looseness aren't mutually exclusive the way they'd be if this were
    // the only lever.
    float weight        = 0.9f;    // main-root travel attraction
    float mainTravelTrials = 14.0f;   // dicing trials (n) for the main root during
                                      // travel -- higher = more candidate headings
                                      // tried per step, more reliably finds one that
                                      // actually points at the target
    float lateralWeight = 0.80f;   // offshoot/lateral travel attraction -- usually kept higher
                                    // so laterals still cling for wrapping density
    float dwellWeight        = 0.92f;   // main-root dwell attraction
    float dwellLateralWeight = 0.92f;   // offshoot/lateral dwell attraction
    float sigma      = 0.35f;   // angular jitter -- higher = less railroaded/straight paths
    float viewCylLen = 8.0f;
    float radiusScale= 1.4f;
    float radiusMin  = 0.03f;
    float radiusMax  = 0.35f;
    float maxHopDays = 60.0f;
    float reachMult  = 1.6f;
    // How far the current target's attraction reaches during travel, as a
    // multiple of the hop's own start->target distance (Gaussian radius in
    // AttractionTropism). This is the primary steering lever now that the old
    // hard travel corridor is gone: >=1.0 keeps the pull effectively constant
    // over the whole leg, so the main root always has a heading signal toward
    // the mask and doesn't drift off under gravity the way it did when the
    // target only pulled within a few cm. Larger = straighter/more insistent;
    // smaller loosens the pull on the far part of long hops (more wander, but
    // risks the old "misses the far mask" failure below ~0.8). Naturalness
    // comes from the angular jitter (sigma), independent of this.
    float travelPullReach = 1.2f;
    // "Along the surface" travel variant: confine the travelling root to a
    // thin shell straddling the cone surface the masks sit on, so it crawls
    // over the cone between masks instead of cutting through the interior.
    // Off = original volumetric travel. Shell applies during travel only --
    // dwell/wrapping stays unconstrained so the nests around each mask still
    // bulge into 3D instead of being flattened onto the surface.
    bool  coneSurfaceTravel  = false;
    float coneShellThickness = 7.0f;   // cm; thicker = looser hug, more wander off-surface
    bool invertMode  = false;   // black bg, white roots, overlaps XOR-toggle

    // --- lighting & material -------------------------------------------
    float rootColor[3]  = {0.55f, 0.40f, 0.26f};
    float rootAmbient   = 0.18f;
    float rootMetallic  = 0.18f;
    float rootRoughness = 0.42f;
    float lightAzimuth   = 0.55f;   // radians -- key light direction
    float lightElevation = 0.95f;
    float maskColor[3]   = {0.86f, 0.83f, 0.78f};
    float faceLightIntensity = 3.2f;
    float faceLightFalloff   = 0.012f;   // lower = light reaches further
    float faceLightDist      = 0.6f;     // multiplier on viewCylLen
    float faceSpecStrength   = 1.2f;
    float veinColor[3] = {0.55f, 0.53f, 0.50f};
    float veinScale    = 0.6f;
    float veinStrength = 0.5f;
    // color of the face-light-fed wisps that illuminate nearby roots --
    // previously mistakenly reused veinColor for this (a copy/paste bug),
    // coupling the mask marble veining color to the root accent lighting.
    float faceLightColorForRoots[3] = {1.0f, 0.92f, 0.78f};
    // fog -- default falloff 0 avoids an axis mismatch: fog.frag's falloff
    // term decays along native "z", which after our Y-up coordinate remap for
    // rendering doesn't correspond to height or camera depth anymore, so a
    // nonzero falloff makes density vary with what's really a horizontal
    // world axis -- as the camera orbits through the "thicker" side it reads
    // as the whole scene dimming over time. Uniform density (falloff=0)
    // sidesteps that; raise it deliberately if you want directional fog.
    // Density is a per-unit-path attenuation coefficient, and our camera
    // sits 15-40 units out -- 0.05 was accumulating to near-total haze over
    // that distance. 0.015 stays visible without drowning everything.
    float fogDensity = 0.015f;
    float fogFalloff = 0.0f;
    float fogNoiseStrength = 0.6f;
    float fogNoiseScale = 0.15f;   // was hardcoded at 0.05 (never exposed) -- too
                                    // low a spatial frequency to read as texture
                                    // over our scene's scale, looked like nothing.
    float fogColor[3] = {0.05f, 0.035f, 0.03f};

    // root material abstraction -- second color blended in by world-space
    // noise, breaking up the flat single-tone "pipes" look.
    float rootColor2[3] = {0.30f, 0.42f, 0.38f};
    float colorNoiseScale = 0.4f;
    float colorNoiseStrength = 0.6f;

    // camera: -1 = whole-scene orbit, >=0 = focus on that revealed mask index
    // (fixed/non-rotating -- see idle-orbit code). orbitSpeed only affects
    // the whole-scene orbit; 0 = frozen.
    int cameraFaceIdx = -1;
    float orbitSpeed = 1.0f;

    // internal render resolution as a fraction of the real framebuffer.
    // History: 0.35 was chosen when the fog pass computed a 4-octave
    // procedural fBm 16x per pixel (~96% of frame time; ~78ms/frame at 0.65
    // scale). After moving that noise into a baked 3D texture, replacing the
    // capsule sphere-trace with an analytic intersection, and caching the
    // CPU-side geometry rebuilds, per-pass GPU profiling measures ~8.6ms/frame
    // at FULL scale on a 5120x2880 framebuffer (was ~92ms) -- so 1.0 is now
    // comfortably interactive and this default is just a headroom choice.
    float renderScale = 0.35f;
};

static const std::vector<std::pair<std::string, std::string>> g_species = {
    {"Maize (Zea mays)",     "Zea_mays_6_Leitner_2014.xml"},
    {"Soybean (Glycine max)","Glycine_max.xml"},
    {"Pea (Pisum sativum)",  "Pisum_sativum_a_Pag\xc3\xa8s_2014.xml"},
    {"Sunflower (Heliantus)","Heliantus_Pages_2013.xml"},
    {"Kale (Brassica)",      "Brassica_oleracea_Vansteenkiste_2014.xml"},
    {"Wheat (Triticum)",     "Triticum_aestivum_a_Bingham_2011.xml"},
    {"Lupin (Lupinus)",      "Lupinus_albus_Leitner_2014.xml"},
    {"Pimpernel (Anagallis)","Anagallis_femina_Leitner_2010.xml"},
};

int main(int argc, char** argv) {
    std::string paramDir = (argc > 1) ? argv[1]
        : "/Users/erichan/Documents/Development/jardins_racine/CPlantBox/"
          "modelparameter/structural/rootsystem/";
    int W = (argc > 2) ? std::atoi(argv[2]) : 1100;
    int H = (argc > 3) ? std::atoi(argv[3]) : 1000;
    bool fullscreen = (argc > 4) ? (std::atoi(argv[4]) != 0) : true;
    bool profileMode = (argc > 5) ? (std::atoi(argv[5]) != 0) : false;   // autostart growth, exit after, print PROFILE lines
    bool profileHidden = (argc > 6) ? (std::atoi(argv[6]) != 0) : true;  // profileMode window visibility (isolate swap-cost variable)
    float cliRenderScale = (argc > 7) ? std::atof(argv[7]) : -1.0f;      // override Params default, for isolating renderScale's effect
    int cliN = (argc > 8) ? std::atoi(argv[8]) : -1;                     // override Params.N, for stress-testing mask count
    float cliPullReach = (argc > 9) ? std::atof(argv[9]) : -1.0f;        // override Params.travelPullReach, for steering sweeps
    int cliSurface = (argc > 10) ? std::atoi(argv[10]) : -1;             // >=0 sets coneSurfaceTravel, for the surface-travel variant

    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    if (profileMode) { fullscreen = false; if (profileHidden) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); }

    // Windowed fullscreen (borderless), not exclusive fullscreen: a normal
    // window sized/positioned to cover the monitor with decorations off,
    // rather than an actual display-mode switch. Avoids exclusive
    // fullscreen's mode-switch overhead/flakiness (also the likely cause of
    // some unexplained non-zero exits on window close) and plays nicer with
    // Spaces/Mission Control on macOS.
    int monX = 0, monY = 0;
    if (fullscreen) {
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        glfwGetMonitorPos(monitor, &monX, &monY);
        W = mode->width; H = mode->height;
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    }
    GLFWwindow* window = glfwCreateWindow(W, H, "jardins_racine -- mask relay", nullptr, nullptr);
    if (!window) { std::cerr << "glfwCreateWindow failed\n"; return 1; }
    if (fullscreen) glfwSetWindowPos(window, monX, monY);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "glewInit failed\n"; return 1; }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    int fbW, fbH;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    // Render internally at a fraction of the real framebuffer size and let
    // the blit's bilinear-filtered texture sample upscale it -- profiling
    // showed present()'s cost (blit + swap) scales almost exactly with pixel
    // count (present() alone: ~20ms at 1.1MP vs ~40ms at 4K's 8.3MP, visible
    // window), so rendering fewer pixels internally is a direct, real lever
    // on the actual bottleneck (not the sphere-tracer itself, which profiled
    // at under 1ms regardless of resolution).
    float renderScale = cliRenderScale > 0.0f ? cliRenderScale : 0.35f;
    int rw = std::max(1, (int) (fbW * renderScale)), rh = std::max(1, (int) (fbH * renderScale));
    RootRenderer renderer(rw, rh);
    renderer.profilePasses = profileMode;
    renderer.mat.baseColor[0] = 0.55f; renderer.mat.baseColor[1] = 0.40f; renderer.mat.baseColor[2] = 0.26f;
    renderer.mat.ambient = 0.18f;
    renderer.mat.diffuse = 0.75f;

    // Cheap render-quality wins, already built into RootRenderer, just unused
    // until now: PBR shading (Cook-Torrance/GGX) gives roots real specular
    // response as the camera orbits instead of a flat matte Phong look, and a
    // touch of metallic reads as wet/waxy root fiber. Subtle fog with drifting
    // noise adds depth cueing so the mass reads as occupying real space
    // instead of a flat silhouette. No wisps -- see the per-face point light
    // added to face.vert/.frag instead, below.
    renderer.shaderMode = RootRenderer::ShaderMode::PBR;
    renderer.pbr.metallic = 0.18f;
    renderer.pbr.roughness = 0.42f;
    renderer.fog.density = 0.010f;
    renderer.fog.falloff = 0.045f;
    renderer.fog.noiseStrength = 0.6f;
    renderer.fog.color[0] = 0.05f; renderer.fog.color[1] = 0.035f; renderer.fog.color[2] = 0.03f;
    renderer.wispCount = 0;

    FaceGL faceGL;
    faceGL.init(SHADER_DIR);
    std::vector<float> fv; std::vector<int> ftris;
    loadObj(FACE_OBJ_PATH, fv, ftris);
    normalizeMesh(fv);
    ftris = cropOvalTris(fv, ftris, 0.72f, 0.98f);

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

    std::vector<MaskNode> revealed;
    std::vector<FrozenHop> frozen;
    int frame = 0;
    int orbitFrame = 0;       // whole-scene orbit's own frame counter
    int focusFrame = 0;       // focused-on-a-mask orbit's own frame counter --
                              // separate so switching between the two doesn't
                              // jump or change speed, but both keep spinning.
    bool growing = profileMode;
    int profileIdleFrames = 0;   // idle-orbit frames measured after growth in profileMode
    std::string status = "Press \"Regrow\" to start.";

    Params params, pending = params;
    if (cliRenderScale > 0.0f) { params.renderScale = cliRenderScale; pending.renderScale = cliRenderScale; }
    if (cliN > 0) { params.N = cliN; pending.N = cliN; }
    if (cliPullReach > 0.0f) { params.travelPullReach = cliPullReach; pending.travelPullReach = cliPullReach; }
    if (cliSurface >= 0) { params.coneSurfaceTravel = (cliSurface != 0); pending.coneSurfaceTravel = params.coneSurfaceTravel; }

    auto blit = [&]() {
        int w2, h2; glfwGetFramebufferSize(window, &w2, &h2);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, w2, h2);
        glUseProgram(blitProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, renderer.colorTex());
        glUniform1i(blitTexLoc, 0);
        glBindVertexArray(blitVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
        glUseProgram(0);
    };

    auto drawPanel = [&]() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Mask Relay Controls");
        ImGui::TextWrapped("%s", status.c_str());
        ImGui::Separator();

        ImGui::BeginDisabled(growing);
        if (ImGui::BeginCombo("Root system", g_species[pending.speciesIdx].first.c_str())) {
            for (int i = 0; i < (int) g_species.size(); i++) {
                bool sel = (i == pending.speciesIdx);
                if (ImGui::Selectable(g_species[i].first.c_str(), sel)) pending.speciesIdx = i;
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SliderInt("Masks", &pending.N, 1, 50);
        ImGui::SliderFloat("Cone radius", &pending.R0, 6.0f, 24.0f, "%.1f cm");
        ImGui::SliderFloat("Cone height (spacing)", &pending.Hh, 24.0f, 96.0f, "%.1f cm");
        ImGui::SliderFloat("Cone taper", &pending.taperPower, 0.25f, 4.0f, "%.2f");
        ImGui::TextDisabled("1 = straight cone. <1 opens up fast (cylinder-\nish). >1 stays narrow, flares near the base.");
        ImGui::SliderFloat("Tip start frac", &pending.startFrac, 0.05f, 0.3f);
        ImGui::SliderFloat("Angle step (x golden angle)", &pending.angleStepGoldenMult, 0.5f, 3.0f, "%.3f");
        ImGui::SliderFloat("Distance step per mask", &pending.distStepFrac, 0.0f, 0.15f, "%.3f");
        ImGui::TextDisabled("Distance step 0 = auto (spread N masks evenly\nover the cone). Set explicitly to make mask\ncount and spacing independent.");
        ImGui::Separator();
        ImGui::SliderFloat("Dwell days", &pending.dwellDays, 2.0f, 40.0f, "%.1f");
        ImGui::SliderFloat("Attraction: main root", &pending.weight, 0.0f, 0.99f, "%.2f");
        ImGui::SliderFloat("Main root trials (n)", &pending.mainTravelTrials, 4.0f, 30.0f, "%.0f");
        ImGui::TextDisabled("More trials = more candidate headings tried\nper step -- at high attraction, this is what\nactually makes it reliably point at the target.");
        ImGui::SliderFloat("Attraction: offshoots", &pending.lateralWeight, 0.0f, 0.95f, "%.2f");
        ImGui::SliderFloat("Attraction (dwell): main root", &pending.dwellWeight, 0.0f, 0.99f, "%.2f");
        ImGui::SliderFloat("Attraction (dwell): offshoots", &pending.dwellLateralWeight, 0.0f, 0.99f, "%.2f");
        ImGui::SliderFloat("Angular jitter", &pending.sigma, 0.1f, 0.7f, "%.2f");
        ImGui::SliderFloat("View-clear cylinder", &pending.viewCylLen, 0.0f, 16.0f, "%.1f cm");
        ImGui::Separator();
        ImGui::SliderFloat("Thickness scale", &pending.radiusScale, 0.3f, 3.0f, "%.2f");
        ImGui::SliderFloat("Thickness min", &pending.radiusMin, 0.0f, 0.3f, "%.3f");
        ImGui::SliderFloat("Thickness max", &pending.radiusMax, 0.05f, 1.0f, "%.2f");
        ImGui::Separator();
        ImGui::SliderFloat("Max days/hop", &pending.maxHopDays, 20.0f, 120.0f, "%.0f");
        ImGui::SliderFloat("Reach threshold", &pending.reachMult, 1.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("Travel pull reach", &pending.travelPullReach, 0.6f, 2.5f, "%.2f x hop");
        ImGui::TextDisabled("How far the target's attraction reaches during\ntravel, as a multiple of the hop distance. This\nis the main steering lever (the old hard corridor\nis gone). >=1 keeps a steady pull the whole way\nso the root reliably reaches far masks; lower it\nfor looser paths. Wander comes from jitter, not\nthis, so paths stay natural either way.");
        ImGui::Checkbox("Travel along cone surface", &pending.coneSurfaceTravel);
        ImGui::TextDisabled("Roots crawl over the cone surface between masks\ninstead of cutting through the interior. Wrapping\nat each mask still bulges into 3D.");
        ImGui::BeginDisabled(!pending.coneSurfaceTravel);
        ImGui::SliderFloat("Surface shell thickness", &pending.coneShellThickness, 2.0f, 16.0f, "%.1f cm");
        ImGui::TextDisabled("How tightly roots hug the surface. Thinner =\ntighter (but more repair jitter); thicker = looser.");
        ImGui::EndDisabled();
        ImGui::EndDisabled();

        ImGui::Separator();
        ImGui::Checkbox("Invert mode (black/white, overlaps XOR)", &pending.invertMode);
        ImGui::TextDisabled("Applies live -- works during growth too.");

        if (ImGui::CollapsingHeader("Lighting & Material")) {
            ImGui::Text("Root material (PBR)");
            ImGui::ColorEdit3("Root color", pending.rootColor);
            ImGui::SliderFloat("Root ambient", &pending.rootAmbient, 0.0f, 0.6f, "%.2f");
            ImGui::SliderFloat("Root metallic", &pending.rootMetallic, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Root roughness", &pending.rootRoughness, 0.02f, 1.0f, "%.2f");
            ImGui::ColorEdit3("Root color (variant)", pending.rootColor2);
            ImGui::SliderFloat("Root mottling scale", &pending.colorNoiseScale, 0.05f, 2.0f, "%.2f");
            ImGui::SliderFloat("Root mottling strength", &pending.colorNoiseStrength, 0.0f, 1.0f, "%.2f");
            ImGui::TextDisabled("Mottling blends the two root colors by noise --\nthe main lever against the flat \"pipes\" look.");
            ImGui::Spacing();
            ImGui::Text("Key light");
            ImGui::SliderFloat("Light azimuth", &pending.lightAzimuth, -3.14f, 3.14f, "%.2f");
            ImGui::SliderFloat("Light elevation", &pending.lightElevation, -1.4f, 1.4f, "%.2f");
            ImGui::Spacing();
            ImGui::Text("Mask material (marble)");
            ImGui::ColorEdit3("Mask base color", pending.maskColor);
            ImGui::ColorEdit3("Mask vein color", pending.veinColor);
            ImGui::SliderFloat("Vein scale", &pending.veinScale, 0.05f, 3.0f, "%.2f");
            ImGui::SliderFloat("Vein strength", &pending.veinStrength, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Face light intensity", &pending.faceLightIntensity, 0.0f, 8.0f, "%.2f");
            ImGui::SliderFloat("Face light falloff", &pending.faceLightFalloff, 0.001f, 0.1f, "%.3f");
            ImGui::SliderFloat("Face light distance", &pending.faceLightDist, 0.1f, 2.0f, "%.2f");
            ImGui::SliderFloat("Face specular", &pending.faceSpecStrength, 0.0f, 4.0f, "%.2f");
            ImGui::ColorEdit3("Face light color (on roots)", pending.faceLightColorForRoots);
            ImGui::TextDisabled("Was accidentally tied to vein color -- now\nindependent.");
            ImGui::Spacing();
            ImGui::Text("Fog");
            ImGui::SliderFloat("Fog density", &pending.fogDensity, 0.0f, 0.3f, "%.3f");
            ImGui::SliderFloat("Fog falloff", &pending.fogFalloff, 0.0f, 0.1f, "%.3f");
            ImGui::SliderFloat("Fog noise strength", &pending.fogNoiseStrength, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Fog noise scale", &pending.fogNoiseScale, 0.01f, 1.0f, "%.2f");
            ImGui::ColorEdit3("Fog color", pending.fogColor);
            ImGui::TextDisabled("All of the above apply live -- no regrow needed.");
        }

        if (ImGui::CollapsingHeader("Performance")) {
            ImGui::SliderFloat("Render scale", &pending.renderScale, 0.25f, 1.0f, "%.2f");
            ImGui::TextDisabled("Internal render resolution vs. the real window/\nscreen size. Applies live. After the fog/capsule\nshader optimizations, 1.0 runs at ~8ms/frame on\na 5K framebuffer -- raise this for sharpness.");
        }

        if (ImGui::CollapsingHeader("Camera")) {
            const char* whole = "Whole scene";
            const char* label = whole;
            std::string faceLabel;
            if (pending.cameraFaceIdx >= 0 && pending.cameraFaceIdx < (int) revealed.size()) {
                faceLabel = "Face " + std::to_string(pending.cameraFaceIdx);
                label = faceLabel.c_str();
            }
            if (ImGui::BeginCombo("Focus", label)) {
                bool sel = pending.cameraFaceIdx < 0;
                if (ImGui::Selectable(whole, sel)) pending.cameraFaceIdx = -1;
                if (sel) ImGui::SetItemDefaultFocus();
                for (int i = 0; i < (int) revealed.size(); i++) {
                    std::string item = "Face " + std::to_string(i);
                    bool s2 = (pending.cameraFaceIdx == i);
                    if (ImGui::Selectable(item.c_str(), s2)) pending.cameraFaceIdx = i;
                    if (s2) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::TextDisabled("Applies once growth is finished (idle orbit).\nFocusing on a face freezes rotation --\nit no longer orbits behind it.");
            ImGui::SliderFloat("Orbit speed", &pending.orbitSpeed, 0.0f, 3.0f, "%.2f");
            ImGui::TextDisabled("Whole-scene orbit only -- a focused face\nalready holds still regardless of this.");
        }

        ImGui::Spacing();
        if (growing) {
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.3f, 1), "Growing...");
        } else if (ImGui::Button("Regrow", ImVec2(-1, 32))) {
            params = pending;
            growing = true;
        }
        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    };

    // profileMode + hidden window: skip presenting entirely. Profiling showed
    // glfwSwapBuffers() on an invisible window is *slower* than a visible one
    // at the same resolution on this machine (macOS apparently falls back to
    // a slower composite/software path for offscreen swaps) -- so a hidden
    // window doesn't give a meaningful "headless" number for the actual
    // render work. glFinish() instead forces the GPU to actually complete the
    // frame before the timer stops, so PROFILE numbers reflect real render()
    // cost without the window-system swap confound at all.
    bool skipPresent = profileMode && profileHidden;
    auto present = [&]() {
        if (skipPresent) { glFinish(); return; }
        blit();
        drawPanel();
        glfwSwapBuffers(window);
        glfwPollEvents();
    };

    float lightDir[3] = {0.5f, 0.8f, 0.35f};

    // --- cached segment geometry -----------------------------------------
    // The frozen hops' contribution to the render buffers only changes when a
    // hop completes (or on Regrow), but it was being rebuilt -- toYup over
    // every node of every frozen hop -- on every growth step AND on every
    // idle-orbit frame, then re-uploaded to the GPU each idle frame too
    // (~1.3ms/frame of pure waste at ~82k segments). Cache the frozen prefix;
    // growth steps just truncate back to it and append the live system.
    std::vector<Vector3d> segNodes;
    std::vector<Vector2i> segSegs;
    std::vector<double>   segRads;
    size_t segPrefixHops  = (size_t) -1;   // frozen.size() the prefix was built from
    size_t segPrefixNodes = 0, segPrefixSegs = 0;
    auto refreshFrozenPrefix = [&]() {
        if (segPrefixHops == frozen.size()) return false;
        segNodes.clear(); segSegs.clear(); segRads.clear();
        int base_idx = 0;
        for (const auto& fh : frozen) {
            for (const auto& n : fh.nodes) segNodes.push_back(toYup(n));
            for (const auto& s : fh.segs) segSegs.push_back(Vector2i(s.x + base_idx, s.y + base_idx));
            for (double r : fh.radii) segRads.push_back(r);
            base_idx += (int) fh.nodes.size();
        }
        segPrefixHops  = frozen.size();
        segPrefixNodes = segNodes.size();
        segPrefixSegs  = segSegs.size();
        return true;
    };

    // --- cached face mesh vertex data -------------------------------------
    // Was rebuilt (every revealed mask x every face triangle, CPU-side) and
    // re-uploaded every growth step and every idle frame; it only actually
    // changes when a mask is revealed or a live-editable face param moves.
    size_t faceCacheCount = (size_t) -1;
    float  faceCacheLightDist = -1.0f;
    float  faceCacheColor[3] = {-1.0f, -1.0f, -1.0f};
    auto refreshFaceData = [&]() {
        float ld = std::max(3.0f, params.viewCylLen * pending.faceLightDist);
        if (faceCacheCount == revealed.size() && faceCacheLightDist == ld
            && faceCacheColor[0] == pending.maskColor[0]
            && faceCacheColor[1] == pending.maskColor[1]
            && faceCacheColor[2] == pending.maskColor[2]) return;
        faceGL.upload(buildFaceVertexData(revealed, fv, ftris, 0.85f, ld, pending.maskColor));
        faceCacheCount = revealed.size();
        faceCacheLightDist = ld;
        faceCacheColor[0] = pending.maskColor[0];
        faceCacheColor[1] = pending.maskColor[1];
        faceCacheColor[2] = pending.maskColor[2];
    };

    // idle-orbit camera fit, recomputed only when the geometry changes
    float idleCenter[3] = {0, 0, 0};
    float idleExtent = 10.0f;

    // live edits (material, lighting, fog, shading mode) apply without a full
    // regrow -- everything here is either a plain RootRenderer member or fed
    // straight into the face shader each frame.
    auto applyLiveRenderParams = [&]() {
        int rw2 = std::max(1, (int) (fbW * pending.renderScale));
        int rh2 = std::max(1, (int) (fbH * pending.renderScale));
        renderer.resize(rw2, rh2);   // no-op if unchanged

        renderer.radiusScale = pending.radiusScale;
        renderer.radiusMin = pending.radiusMin;
        renderer.radiusMax = pending.radiusMax;
        renderer.mat.baseColor[0] = pending.rootColor[0];
        renderer.mat.baseColor[1] = pending.rootColor[1];
        renderer.mat.baseColor[2] = pending.rootColor[2];
        renderer.mat.ambient = pending.rootAmbient;
        renderer.pbr.metallic = pending.rootMetallic;
        renderer.pbr.roughness = pending.rootRoughness;
        renderer.mat.baseColor2[0] = pending.rootColor2[0];
        renderer.mat.baseColor2[1] = pending.rootColor2[1];
        renderer.mat.baseColor2[2] = pending.rootColor2[2];
        renderer.mat.colorNoiseScale = pending.colorNoiseScale;
        renderer.mat.colorNoiseStrength = pending.colorNoiseStrength;

        float cosEl = cosf(pending.lightElevation), sinEl = sinf(pending.lightElevation);
        lightDir[0] = cosEl * sinf(pending.lightAzimuth);
        lightDir[1] = sinEl;
        lightDir[2] = cosEl * cosf(pending.lightAzimuth);

        renderer.fog.falloff = pending.fogFalloff;
        renderer.fog.noiseStrength = pending.fogNoiseStrength;
        renderer.fog.noiseScale = pending.fogNoiseScale;
        renderer.fog.color[0] = pending.fogColor[0];
        renderer.fog.color[1] = pending.fogColor[1];
        renderer.fog.color[2] = pending.fogColor[2];

        faceGL.lightIntensity = pending.faceLightIntensity;
        faceGL.lightFalloff = pending.faceLightFalloff;
        faceGL.specStrength = pending.faceSpecStrength;
        faceGL.veinColor[0] = pending.veinColor[0];
        faceGL.veinColor[1] = pending.veinColor[1];
        faceGL.veinColor[2] = pending.veinColor[2];
        faceGL.veinScale = pending.veinScale;
        faceGL.veinStrength = pending.veinStrength;

        if (pending.invertMode) {
            renderer.shaderMode = RootRenderer::ShaderMode::Invert;
            renderer.fog.density = 0.0f;   // fog would muddy the pure black/white toggle
        } else {
            renderer.shaderMode = RootRenderer::ShaderMode::PBR;
            renderer.fog.density = pending.fogDensity;
        }

        // Feed every revealed face's point light into the root shader too
        // (as RootRenderer "wisps"), using the exact same position each face
        // shader itself lights from -- so roots near a mask actually pick up
        // some of that light instead of it only affecting the face mesh.
        // Every revealed mask gets a light (up to MAX_WISPS=50, matching the
        // max mask count) -- previously capped to the 6 most-recently-
        // revealed for performance, but that left most masks permanently
        // unlit with higher counts. Made affordable by (a) the cheap
        // Blinn-Phong wisp shading (see shader.frag) instead of full PBR
        // per light, and (b) a per-pixel distance cutoff in both shader.frag
        // and fog.frag that skips the expensive part of a light entirely once
        // it's clearly too far to matter, so cost tracks how many lights are
        // actually NEARBY a given pixel, not the total count.
        int wc = std::min((int) revealed.size(), RootRenderer::MAX_WISPS);
        renderer.wispCount = wc;
        for (int i = 0; i < wc; i++) {
            const auto& m = revealed[i];
            Vector3d n = toYup(m.normal);
            Vector3d p = toYup(m.pos).minus(n.times(m.r_depth * 0.5));
            Vector3d lp = p.plus(n.times((double) std::max(3.0f, params.viewCylLen * pending.faceLightDist)));
            renderer.wisps[i].basePos[0] = (float) lp.x;
            renderer.wisps[i].basePos[1] = (float) lp.y;
            renderer.wisps[i].basePos[2] = (float) lp.z;
            renderer.wisps[i].color[0] = pending.faceLightColorForRoots[0];
            renderer.wisps[i].color[1] = pending.faceLightColorForRoots[1];
            renderer.wisps[i].color[2] = pending.faceLightColorForRoots[2];
            renderer.wisps[i].intensity = pending.faceLightIntensity * 0.5f;
            renderer.wisps[i].driftRadius = 0.0f;   // static -- these ARE the face lights, not atmosphere
        }
    };

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);
        applyLiveRenderParams();

        if (growing) {
            revealed.clear();
            frozen.clear();
            frame = 0;
            // explicit reset -- don't rely on the wisp-feed loop naturally
            // shrinking to 0 on the next applyLiveRenderParams() call (it
            // does, but only once revealed.size()==0 is actually observed;
            // being explicit here avoids any stale-light flash on Regrow).
            renderer.wispCount = 0;
            std::string param = paramDir + g_species[params.speciesIdx].second;
            double maskR = 2.6;
            double tipRadius = 0.22 * params.R0;
            const double goldenRad = M_PI * (3.0 - std::sqrt(5.0));
            auto masks = conePhyllotaxis(params.N, params.R0, params.Hh, maskR,
                                         params.startFrac, 0.94, tipRadius,
                                         params.angleStepGoldenMult * goldenRad, params.distStepFrac,
                                         params.taperPower);
            Vector3d prevPos(0, 0, 0);
            bool quit = false;

            for (int hop = 0; hop < params.N && !quit; hop++) {
                Vector3d targetGlobal = masks[hop].pos;
                status = "Growing hop " + std::to_string(hop + 1) + "/" + std::to_string(params.N)
                        + " -- " + g_species[params.speciesIdx].first;

                auto rs = std::make_shared<RootSystem>();
                rs->readParameters(param, "plant", true, false);
                // deterministic geometry in profile mode -- before/after
                // benchmark runs must render the exact same root mass for
                // per-pass timings to be comparable.
                if (profileMode) rs->setSeed(42u + (unsigned) hop);
                rs->initialize(false);
                auto seedNodes = rs->getNodes();
                Vector3d localSeed = seedNodes.empty() ? Vector3d(0, 0, 0) : seedNodes[0];
                Vector3d offset = prevPos.minus(localSeed);
                Vector3d localTarget = targetGlobal.minus(offset);

                std::vector<MaskNode> localRevealed;
                for (const auto& m : revealed) {
                    MaskNode lm = m; lm.pos = m.pos.minus(offset);
                    localRevealed.push_back(lm);
                }
                MaskNode localTargetNode = masks[hop];
                localTargetNode.pos = localTarget;

                auto base = std::make_shared<Gravitropism>(rs, 1.0, params.sigma);
                double hopLen = localTarget.minus(localSeed).length();

                // Steering is now soft attraction only -- no hard travel
                // corridor. The old corridor was a geometric backstop for a
                // root that "missed", but it fought the real problem instead
                // of fixing it: CPlantBox's confinement repair (tropism.cpp
                // getHeading) dices a PURELY RANDOM azimuth whenever a
                // candidate heading leaves the allowed region, ignoring
                // attraction while it repairs -- so a root pressed against a
                // long corridor wall steered randomly, not toward the mask.
                //
                // The actual cause of "way off course" was that the target's
                // attraction had almost no reach: rimAttractors used a 3cm
                // Gaussian radius, so AttractionTropism's objective went
                // neutral (0.5, no pull) beyond ~6cm -- i.e. for nearly the
                // whole travel leg the main root felt no target at all and
                // just fell under residual gravity until it happened to drift
                // within 6cm. Late hops at the wide end of the cone, whose
                // targets sit tens of cm from the seed, could never close that
                // gap and gave up ~20cm short.
                //
                // Fix: during travel, pull toward the target with a Gaussian
                // radius scaled to the hop's own length (x the user reach
                // multiplier), so there's a consistent heading signal the
                // entire way. Because AttractionTropism normalizes the pull
                // direction and scores only alignment, a large radius acts as
                // a near-constant "point at the target" objective -- reliable
                // steering -- while the sigma angular jitter still supplies
                // organic wander, so paths stay natural rather than beelining.
                // On reach we swap to the tight rimAttractors ring, which is
                // what actually wants a small radius: local wrapping of the
                // mask rim, not long-range travel.
                auto rebuildTropism = [&](double mainW, double lateralW, bool travel) {
                    auto geom = buildCavityGeometry(localRevealed, params.R0, params.Hh, false, 2.0,
                                                    tipRadius, params.viewCylLen, 0.9, params.taperPower);
                    // "along the surface" variant: during travel, intersect the
                    // cavity-avoidance geometry with a shell hugging the cone
                    // surface, so the root crawls over the cone rather than
                    // through it. The shell only constrains the radial
                    // (off-surface) axis -- orthogonal to travel direction --
                    // so it doesn't fight the tangential target attraction the
                    // way the old axial corridor did. Local frame: the global
                    // cone apex (origin) sits at -offset in grow coords.
                    std::shared_ptr<SignedDistanceFunction> growGeom = geom;
                    if (params.coneSurfaceTravel && travel) {
                        auto shell = buildConeShell(params.R0, params.Hh, tipRadius, params.taperPower,
                                                    params.coneShellThickness, offset.times(-1.0));
                        growGeom = std::make_shared<CPlantBox::SDF_Intersection>(geom, shell);
                    }
                    rs->setGeometry(growGeom);
                    std::vector<Attractor> attrs;
                    if (travel) {
                        double reach = std::max(params.travelPullReach * hopLen, (double) params.R0);
                        attrs.push_back(Attractor{localTarget, 1.0, reach});
                    } else {
                        attrs = rimAttractors({localTargetNode}, 6, 1.0, 3.0, 1.15);
                    }
                    rs->setTropism(combinedAttractionSplit(rs, base, attrs, params.mainTravelTrials, 6.0,
                                                           params.sigma, mainW, lateralW, growGeom, growGeom), -1);
                };
                rebuildTropism(params.weight, params.lateralWeight, true);

                // Scale the day budget by how far this hop actually has to
                // travel: in a golden-angle spiral, masks near the wide end
                // of the cone sit at large radius, so the ARC LENGTH between
                // consecutive masks (~radius * angle) grows with radius --
                // late hops can genuinely need to cover several times the
                // distance early ones do, but were getting the same flat
                // budget. This was the real cause of late-hop failures
                // (confirmed: they persisted identically with the travel
                // corridor effectively disabled, so it wasn't a corridor/
                // repair-loop problem -- just not enough time to get there).
                double hopMaxDays = params.maxHopDays * std::max(1.0, hopLen / std::max(1.0, (double) params.R0));

                double day = 0.0, reachedDay = -1.0;
                bool reached = false;
                double dt = 0.75;
                while (day < hopMaxDays) {
                    if (glfwWindowShouldClose(window)) { quit = true; break; }

                    rs->simulate(dt, false);
                    day += dt;

                    bool forced = !reached && day > 0.6 * hopMaxDays;
                    if (!reached) {
                        double d = minDist(rs->getNodes(), localTarget);
                        double thr = params.reachMult * std::max(masks[hop].r_width, masks[hop].r_height);
                        if (d < thr || forced) {
                            reached = true; reachedDay = day;
                            if (forced) std::cout << "PROFILE hop " << hop << " gave up at day " << day
                                                  << " (dist=" << d << ", threshold=" << thr << ")\n";
                            localRevealed.push_back(localTargetNode);
                            revealed.push_back(masks[hop]);
                            rebuildTropism(params.dwellWeight, params.dwellLateralWeight, false);
                        }
                    }
                    if (reached && day - reachedDay > params.dwellDays) break;

                    auto _t0 = std::chrono::steady_clock::now();
                    SegmentAnalyser ana(*rs);
                    auto radii = ana.getParameter("radius");
                    refreshFrozenPrefix();
                    segNodes.resize(segPrefixNodes);
                    segSegs.resize(segPrefixSegs);
                    segRads.resize(segPrefixSegs);
                    int base_idx = (int) segPrefixNodes;
                    for (const auto& n : ana.nodes) segNodes.push_back(toYup(n.plus(offset)));
                    for (const auto& s : ana.segments) segSegs.push_back(Vector2i(s.x + base_idx, s.y + base_idx));
                    for (double r : radii) segRads.push_back(r);
                    auto _t1 = std::chrono::steady_clock::now();
                    renderer.uploadSegments(segNodes, segSegs, segRads);
                    auto _t2 = std::chrono::steady_clock::now();

                    refreshFaceData();
                    auto _t3 = std::chrono::steady_clock::now();

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
                    float azimuth = 0.8f + 2.0f * (float) M_PI * 0.5f * pending.orbitSpeed * (frame / 260.0f);
                    renderer.fog.driftTime += 0.02f;

                    renderer.render(azimuth, 0.12f, radius, target3, 0.5f, lightDir,
                                    [&](const float* vp, const float* eye) { faceGL.draw(vp, eye, lightDir); });
                    auto _t4 = std::chrono::steady_clock::now();
                    present();
                    auto _t5 = std::chrono::steady_clock::now();

                    if (frame % 10 == 0) {
                        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                        std::cout << "PROFILE frame=" << frame << " segs=" << segSegs.size()
                                  << " rebuild=" << ms(_t0, _t1) << "ms upload=" << ms(_t1, _t2)
                                  << "ms faceBuild+upload=" << ms(_t2, _t3) << "ms render=" << ms(_t3, _t4)
                                  << "ms present=" << ms(_t4, _t5) << "ms TOTAL=" << ms(_t0, _t5) << "ms";
                        if (renderer.profilePasses)
                            std::cout << " geomPass=" << renderer.lastGeomMs
                                      << "ms fogPass=" << renderer.lastFogMs << "ms";
                        std::cout << "\n";
                    }
                    frame++;
                }
                if (quit) break;

                SegmentAnalyser anaFinal(*rs);
                auto radiiFinal = anaFinal.getParameter("radius");
                FrozenHop fh;
                for (const auto& n : anaFinal.nodes) fh.nodes.push_back(n.plus(offset));
                fh.segs = anaFinal.segments;
                fh.radii = radiiFinal;
                frozen.push_back(fh);
                prevPos = targetGlobal;
            }

            growing = false;
            status = quit ? "Stopped." : "Done -- " + g_species[params.speciesIdx].first
                     + ", " + std::to_string((int) revealed.size()) + "/" + std::to_string(params.N) + " masks.";
            if (profileMode) std::cout << "PROFILE: growth complete, measuring idle orbit\n";
            continue;   // re-enter loop in idle-orbit mode
        }

        // ---- idle: orbit the finished result ----
        auto _i0 = std::chrono::steady_clock::now();
        // geometry is static here; rebuild + re-upload only when a growth
        // session actually changed it (or the buffers still carry a live-
        // growth tail past the frozen prefix from the last growth frame)
        bool geomChanged = refreshFrozenPrefix() || segNodes.size() != segPrefixNodes;
        if (geomChanged) {
            segNodes.resize(segPrefixNodes);
            segSegs.resize(segPrefixSegs);
            segRads.resize(segPrefixSegs);
            renderer.uploadSegments(segNodes, segSegs, segRads);
            Vector3d lo = segNodes.empty() ? Vector3d(0, 0, 0) : segNodes[0], hi = lo;
            for (const auto& n : segNodes) {
                lo = Vector3d(std::min(lo.x, n.x), std::min(lo.y, n.y), std::min(lo.z, n.z));
                hi = Vector3d(std::max(hi.x, n.x), std::max(hi.y, n.y), std::max(hi.z, n.z));
            }
            idleCenter[0] = (float) (lo.x + hi.x) * 0.5f;
            idleCenter[1] = (float) (lo.y + hi.y) * 0.5f;
            idleCenter[2] = (float) (lo.z + hi.z) * 0.5f;
            idleExtent = segNodes.empty() ? 10.0f
                       : (float) std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
        }
        refreshFaceData();

        float target3[3];
        float radius;
        float azimuth;
        bool focused = pending.cameraFaceIdx >= 0 && pending.cameraFaceIdx < (int) revealed.size();
        if (focused) {
            // focused on one mask: center on it and frame tight, but keep
            // orbiting around it (its own independent, continuously-advancing
            // angle) rather than either freezing or reusing the whole-scene
            // orbit's arc (which was centered on the whole piece, not this
            // mask, and would swing it out of frame/behind the camera).
            Vector3d fp = toYup(revealed[pending.cameraFaceIdx].pos);
            target3[0] = (float) fp.x; target3[1] = (float) fp.y; target3[2] = (float) fp.z;
            radius = std::max(revealed[pending.cameraFaceIdx].r_width, revealed[pending.cameraFaceIdx].r_height) * 3.5f + 4.0f;
            azimuth = 0.8f + 0.15f * pending.orbitSpeed * (float) focusFrame / 60.0f;
            focusFrame++;
        } else {
            target3[0] = idleCenter[0]; target3[1] = idleCenter[1]; target3[2] = idleCenter[2];
            radius = idleExtent * 0.75f + 12.0f;
            azimuth = 0.8f + 0.15f * pending.orbitSpeed * (float) orbitFrame / 60.0f;
            orbitFrame++;
        }

        auto _i1 = std::chrono::steady_clock::now();
        renderer.render(azimuth, 0.15f, radius, target3, 0.5f, lightDir,
                        [&](const float* vp, const float* eye) { faceGL.draw(vp, eye, lightDir); });
        auto _i2 = std::chrono::steady_clock::now();
        present();
        auto _i3 = std::chrono::steady_clock::now();
        if (profileMode) {
            // visual regression check: dump the composited frame (fog FBO
            // color texture) to a PPM once the idle orbit has settled, so
            // shader optimizations can be A/B-diffed on identical seeded
            // geometry without OS screenshot permissions.
            if (profileIdleFrames == 30) {
                if (const char* dumpPath = std::getenv("MASK_RELAY_DUMP")) {
                    int dw = renderer.width(), dh = renderer.height();
                    std::vector<unsigned char> px((size_t) dw * dh * 4);
                    glBindTexture(GL_TEXTURE_2D, renderer.colorTex());
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
                    glBindTexture(GL_TEXTURE_2D, 0);
                    std::ofstream out(dumpPath, std::ios::binary);
                    out << "P6\n" << dw << " " << dh << "\n255\n";
                    for (int y = 0; y < dh; y++)   // FBO is top-down already for this pipeline
                        for (int x = 0; x < dw; x++)
                            out.write((const char*) &px[((size_t) y * dw + x) * 4], 3);
                    std::cout << "PROFILE: dumped frame to " << dumpPath << "\n";
                }
            }
            auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            std::cout << "PROFILE-IDLE frame=" << profileIdleFrames << " segs=" << segSegs.size()
                      << " rebuild+upload=" << ms(_i0, _i1) << "ms render=" << ms(_i1, _i2)
                      << "ms present=" << ms(_i2, _i3) << "ms TOTAL=" << ms(_i0, _i3) << "ms"
                      << " geomPass=" << renderer.lastGeomMs << "ms fogPass=" << renderer.lastFogMs << "ms\n";
            if (++profileIdleFrames >= 60) { std::cout << "PROFILE: idle orbit done, exiting\n"; break; }
        }
        frame++;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
