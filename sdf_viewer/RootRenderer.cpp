#include "RootRenderer.h"

#include "mymath.h"   // CPlantBox::Vector3d, Vector2i

#include <GL/glew.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny vec3 helpers (avoids pulling in GLM)
// ---------------------------------------------------------------------------
struct V3 { float x, y, z; };

static float dot3(V3 a, V3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
static V3  cross3(V3 a, V3 b)  { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
static V3   norm3(V3 v)        { float l=sqrtf(dot3(v,v)); if(l<1e-7f)l=1.f; return{v.x/l,v.y/l,v.z/l}; }

// ---------------------------------------------------------------------------
// Shader utilities
// ---------------------------------------------------------------------------
static std::string readFile(const char* path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error(std::string("cannot open: ") + path);
    return { std::istreambuf_iterator<char>(f), {} };
}

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096]; glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        glDeleteShader(s);
        throw std::runtime_error(std::string("shader compile error:\n") + log);
    }
    return s;
}

static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096]; glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        glDeleteProgram(p);
        throw std::runtime_error(std::string("program link error:\n") + log);
    }
    glDetachShader(p, vs); glDetachShader(p, fs);
    return p;
}

// ---------------------------------------------------------------------------
// TBO helper
// ---------------------------------------------------------------------------
static void makeTBO(GLuint& buf, GLuint& tex) {
    glGenBuffers(1, &buf);
    glGenTextures(1, &tex);
    float dummy = 0.f;
    glBindBuffer(GL_TEXTURE_BUFFER, buf);
    glBufferData(GL_TEXTURE_BUFFER, sizeof(float), &dummy, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

// ---------------------------------------------------------------------------
// Tiling 3D value-noise fBm, baked once on the CPU.
//
// Profiling (glFinish-synced per-pass timers, 4K, ~82k segments) showed the
// fog pass at ~199ms/frame of a ~208ms total -- ~96% of the entire frame --
// and nearly all of it was the procedural noise: fogFbm() evaluated a
// 4-octave value fBm (32 hash calls) at each of 16 raymarch steps per pixel,
// ~512 hashes/pixel, ~4 billion/frame at 4K. Baking the same fBm into a
// small tiling 3D texture turns each of those 32-hash evaluations into one
// trilinear texture fetch. The bake uses lattice-period-wrapped noise so the
// texture tiles seamlessly under GL_REPEAT; octave frequencies are the
// power-of-two ones (2/4/8) rather than the original irrational-ish
// 2.03/4.07/8.11, which is visually indistinguishable for fog.
// ---------------------------------------------------------------------------

// world-space size (in the shader's noise coordinate units) of one texture
// tile -- shaders must multiply noise coords by 1/NOISE_TILE_PERIOD.
static constexpr float NOISE_TILE_PERIOD = 8.0f;

static float latticeHash(int x, int y, int z) {
    uint32_t h = (uint32_t) x * 374761393u + (uint32_t) y * 668265263u
               + (uint32_t) z * 1274126177u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (float) ((h ^ (h >> 16)) & 0xFFFFFFu) / (float) 0x1000000u;
}

// smoothstep-filtered trilinear value noise, integer lattice wrapped at `per`
static float tilingValueNoise(float px, float py, float pz, int per) {
    int ix = (int) std::floor(px), iy = (int) std::floor(py), iz = (int) std::floor(pz);
    float fx = px - ix, fy = py - iy, fz = pz - iz;
    float ux = fx * fx * (3.f - 2.f * fx);
    float uy = fy * fy * (3.f - 2.f * fy);
    float uz = fz * fz * (3.f - 2.f * fz);
    auto wrap = [per](int v) { return ((v % per) + per) % per; };
    float c[2][2][2];
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++)
                c[dz][dy][dx] = latticeHash(wrap(ix + dx), wrap(iy + dy), wrap(iz + dz));
    auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
    float x00 = lerp(c[0][0][0], c[0][0][1], ux), x10 = lerp(c[0][1][0], c[0][1][1], ux);
    float x01 = lerp(c[1][0][0], c[1][0][1], ux), x11 = lerp(c[1][1][0], c[1][1][1], ux);
    return lerp(lerp(x00, x10, uy), lerp(x01, x11, uy), uz);
}

void RootRenderer::buildNoiseTexture() {
    const int N = 128;   // 128^3 R8 = 2MB; highest octave gets 2 voxels/cell
    std::vector<unsigned char> vox((size_t) N * N * N);
    const float basePer = NOISE_TILE_PERIOD;   // base octave: 8 lattice cells/tile
    for (int z = 0; z < N; z++) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                float px = (float) x / N * basePer;
                float py = (float) y / N * basePer;
                float pz = (float) z / N * basePer;
                float v = 0.5000f * tilingValueNoise(px,      py,      pz,      (int) basePer)
                        + 0.2500f * tilingValueNoise(px * 2,  py * 2,  pz * 2,  (int) basePer * 2)
                        + 0.1250f * tilingValueNoise(px * 4,  py * 4,  pz * 4,  (int) basePer * 4)
                        + 0.0625f * tilingValueNoise(px * 8,  py * 8,  pz * 8,  (int) basePer * 8);
                v *= 1.0f / 0.9375f;   // -> [0,1], mean ~0.5, matches old fogFbm()
                vox[((size_t) z * N + y) * N + x] = (unsigned char) (v * 255.0f + 0.5f);
            }
        }
    }
    glGenTextures(1, &m_noiseTex);
    glBindTexture(GL_TEXTURE_3D, m_noiseTex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, N, N, N, 0, GL_RED, GL_UNSIGNED_BYTE, vox.data());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glBindTexture(GL_TEXTURE_3D, 0);
}

// ===========================================================================
// RootRenderer
// ===========================================================================

RootRenderer::RootRenderer(int w, int h) : m_w(w), m_h(h) {
    buildShader();
    buildFBO();
    buildNoiseTexture();

    makeTBO(m_nodeBuf, m_nodeTex);
    makeTBO(m_segBuf,  m_segTex);
    makeTBO(m_radBuf,  m_radTex);

    glBindTexture(GL_TEXTURE_BUFFER, m_nodeTex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, m_nodeBuf);

    glBindTexture(GL_TEXTURE_BUFFER, m_segTex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32I,  m_segBuf);

    glBindTexture(GL_TEXTURE_BUFFER, m_radTex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F,   m_radBuf);

    glBindTexture(GL_TEXTURE_BUFFER, 0);

    glGenVertexArrays(1, &m_vao);
}

RootRenderer::~RootRenderer() {
    glDeleteProgram(m_prog);
    glDeleteProgram(m_fogProg);
    glDeleteVertexArrays(1, &m_vao);
    glDeleteFramebuffers(1,  &m_fbo);
    glDeleteTextures(1,      &m_colorTex);
    glDeleteTextures(1,      &m_depthTex);
    glDeleteFramebuffers(1,  &m_fogFbo);
    glDeleteTextures(1,      &m_fogColorTex);
    glDeleteBuffers(1,  &m_nodeBuf);
    glDeleteBuffers(1,  &m_segBuf);
    glDeleteBuffers(1,  &m_radBuf);
    glDeleteTextures(1, &m_nodeTex);
    glDeleteTextures(1, &m_segTex);
    glDeleteTextures(1, &m_radTex);
    glDeleteTextures(1, &m_noiseTex);
}

void RootRenderer::buildShader() {
    // --- Geometry pass ---
    {
        auto vertSrc = readFile(SHADER_DIR "/shader.vert");
        auto fragSrc = readFile(SHADER_DIR "/shader.frag");
        GLuint vs = compileShader(GL_VERTEX_SHADER,   vertSrc.c_str());
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc.c_str());
        m_prog = linkProgram(vs, fs);
        glDeleteShader(vs); glDeleteShader(fs);

        m_uNodes    = glGetUniformLocation(m_prog, "u_nodes");
        m_uSegs     = glGetUniformLocation(m_prog, "u_segments");
        m_uRads     = glGetUniformLocation(m_prog, "u_radii");
        m_uSegCount = glGetUniformLocation(m_prog, "u_segCount");
        m_uEye      = glGetUniformLocation(m_prog, "u_eye");
        m_uCam      = glGetUniformLocation(m_prog, "u_cam");
        m_uFov      = glGetUniformLocation(m_prog, "u_fov");
        m_uRes      = glGetUniformLocation(m_prog, "u_res");
        m_uViewProj  = glGetUniformLocation(m_prog, "u_viewProj");
        m_uBaseColor = glGetUniformLocation(m_prog, "u_baseColor");
        m_uBaseColor2 = glGetUniformLocation(m_prog, "u_baseColor2");
        m_uColorNoiseScale = glGetUniformLocation(m_prog, "u_colorNoiseScale");
        m_uColorNoiseStrength = glGetUniformLocation(m_prog, "u_colorNoiseStrength");
        m_uAmbient   = glGetUniformLocation(m_prog, "u_ambient");
        m_uDiffuse   = glGetUniformLocation(m_prog, "u_diffuse");
        m_uSpecColor = glGetUniformLocation(m_prog, "u_specColor");
        m_uShininess = glGetUniformLocation(m_prog, "u_shininess");
        m_uLightDir  = glGetUniformLocation(m_prog, "u_lightDir");
        m_uShaderMode    = glGetUniformLocation(m_prog, "u_shaderMode");
        m_uMetallic      = glGetUniformLocation(m_prog, "u_metallic");
        m_uRoughness     = glGetUniformLocation(m_prog, "u_roughness");
        m_uWispCount     = glGetUniformLocation(m_prog, "u_wispCount");
        m_uWispPos       = glGetUniformLocation(m_prog, "u_wispPos[0]");
        m_uWispColor     = glGetUniformLocation(m_prog, "u_wispColor[0]");
        m_uWispIntensity = glGetUniformLocation(m_prog, "u_wispIntensity[0]");
        m_uRadiusScale   = glGetUniformLocation(m_prog, "u_radiusScale");
        m_uRadiusMin     = glGetUniformLocation(m_prog, "u_radiusMin");
        m_uRadiusMax     = glGetUniformLocation(m_prog, "u_radiusMax");
        m_uNoiseTex      = glGetUniformLocation(m_prog, "u_noiseTex");
    }

    // --- Fog post-process pass ---
    {
        auto vertSrc = readFile(SHADER_DIR "/fog.vert");
        auto fragSrc = readFile(SHADER_DIR "/fog.frag");
        GLuint vs = compileShader(GL_VERTEX_SHADER,   vertSrc.c_str());
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc.c_str());
        m_fogProg = linkProgram(vs, fs);
        glDeleteShader(vs); glDeleteShader(fs);

        m_fpColorTex = glGetUniformLocation(m_fogProg, "u_colorTex");
        m_fpDepthTex = glGetUniformLocation(m_fogProg, "u_depthTex");
        m_fpEye      = glGetUniformLocation(m_fogProg, "u_eye");
        m_fpCam      = glGetUniformLocation(m_fogProg, "u_cam");
        m_fpFov      = glGetUniformLocation(m_fogProg, "u_fov");
        m_fpRes      = glGetUniformLocation(m_fogProg, "u_res");
        m_fpNear     = glGetUniformLocation(m_fogProg, "u_near");
        m_fpFar      = glGetUniformLocation(m_fogProg, "u_far");
        m_fpFogColor         = glGetUniformLocation(m_fogProg, "u_fogColor");
        m_fpFogDensity       = glGetUniformLocation(m_fogProg, "u_fogDensity");
        m_fpFogFalloff       = glGetUniformLocation(m_fogProg, "u_fogFalloff");
        m_fpFogNoiseScale    = glGetUniformLocation(m_fogProg, "u_fogNoiseScale");
        m_fpFogNoiseStrength = glGetUniformLocation(m_fogProg, "u_fogNoiseStrength");
        m_fpFogTime          = glGetUniformLocation(m_fogProg, "u_fogTime");
        m_fpFogRefDist       = glGetUniformLocation(m_fogProg, "u_fogRefDist");
        m_fpWispGlowStrength = glGetUniformLocation(m_fogProg, "u_wispGlowStrength");
        m_fpNoiseType        = glGetUniformLocation(m_fogProg, "u_noiseType");
        m_fpShowAxes         = glGetUniformLocation(m_fogProg, "u_showAxes");
        m_fpAxisLength       = glGetUniformLocation(m_fogProg, "u_axisLength");
        m_fpShowGrid         = glGetUniformLocation(m_fogProg, "u_showGrid");
        m_fpGridSpacing      = glGetUniformLocation(m_fogProg, "u_gridSpacing");
        m_fpWispCount        = glGetUniformLocation(m_fogProg, "u_wispCount");
        m_fpWispPos          = glGetUniformLocation(m_fogProg, "u_wispPos[0]");
        m_fpWispColor        = glGetUniformLocation(m_fogProg, "u_wispColor[0]");
        m_fpWispIntensity    = glGetUniformLocation(m_fogProg, "u_wispIntensity[0]");
        m_fpNoiseTex         = glGetUniformLocation(m_fogProg, "u_noiseTex");
    }
}

void RootRenderer::buildFBO() {
    // --- Geometry FBO ---
    if (m_fbo)      { glDeleteFramebuffers(1, &m_fbo);      m_fbo      = 0; }
    if (m_colorTex) { glDeleteTextures(1,     &m_colorTex); m_colorTex = 0; }
    if (m_depthTex) { glDeleteTextures(1,     &m_depthTex); m_depthTex = 0; }

    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    glGenTextures(1, &m_colorTex);
    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_w, m_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_colorTex, 0);

    // Depth as a texture so the fog pass can sample it
    glGenTextures(1, &m_depthTex);
    glBindTexture(GL_TEXTURE_2D, m_depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, m_w, m_h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_depthTex, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Geometry FBO incomplete");

    // --- Fog FBO (colour only) ---
    if (m_fogFbo)      { glDeleteFramebuffers(1, &m_fogFbo);      m_fogFbo      = 0; }
    if (m_fogColorTex) { glDeleteTextures(1,     &m_fogColorTex); m_fogColorTex = 0; }

    glGenFramebuffers(1, &m_fogFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fogFbo);

    glGenTextures(1, &m_fogColorTex);
    glBindTexture(GL_TEXTURE_2D, m_fogColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_w, m_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fogColorTex, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Fog FBO incomplete");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void RootRenderer::resize(int w, int h) {
    if (w < 1 || h < 1 || (w == m_w && h == m_h)) return;
    m_w = w; m_h = h;
    buildFBO();
}

void RootRenderer::uploadSegments(const std::vector<CPlantBox::Vector3d>& nodes,
                                   const std::vector<CPlantBox::Vector2i>& segs,
                                   const std::vector<double>& radii) {
    m_segCount = static_cast<int>(segs.size());

    std::vector<float> nodeData;
    nodeData.reserve(nodes.size() * 3);
    for (auto& n : nodes) {
        nodeData.push_back(static_cast<float>(n.x));
        nodeData.push_back(static_cast<float>(n.y));
        nodeData.push_back(static_cast<float>(n.z));
    }
    if (nodeData.empty()) nodeData.assign(3, 0.f);

    std::vector<int> segData;
    segData.reserve(segs.size() * 2);
    for (auto& s : segs) { segData.push_back(s.x); segData.push_back(s.y); }
    if (segData.empty()) segData.assign(2, 0);

    std::vector<float> radData;
    radData.reserve(radii.size());
    for (double r : radii) radData.push_back(static_cast<float>(r));
    if (radData.empty()) radData.push_back(0.f);

    glBindBuffer(GL_TEXTURE_BUFFER, m_nodeBuf);
    glBufferData(GL_TEXTURE_BUFFER, static_cast<GLsizeiptr>(nodeData.size() * sizeof(float)),
                 nodeData.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_TEXTURE_BUFFER, m_segBuf);
    glBufferData(GL_TEXTURE_BUFFER, static_cast<GLsizeiptr>(segData.size() * sizeof(int)),
                 segData.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_TEXTURE_BUFFER, m_radBuf);
    glBufferData(GL_TEXTURE_BUFFER, static_cast<GLsizeiptr>(radData.size() * sizeof(float)),
                 radData.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

void RootRenderer::render(float azimuth, float elevation, float radius,
                           const float* target3, float fov,
                           const float* lightDir3, const MidHook& midGeometryHook) {
    float cosEl = cosf(elevation), sinEl = sinf(elevation);
    float cosAz = cosf(azimuth),   sinAz = sinf(azimuth);

    float ex = target3[0] + radius * cosEl * sinAz;
    float ey = target3[1] + radius * sinEl;
    float ez = target3[2] + radius * cosEl * cosAz;

    V3 fwd = norm3({target3[0]-ex, target3[1]-ey, target3[2]-ez});
    V3 wup = {0.f, 1.f, 0.f};
    V3 rawRgt = cross3(fwd, wup);
    V3 rgt = (dot3(rawRgt, rawRgt) > 1e-8f) ? norm3(rawRgt) : V3{1.f, 0.f, 0.f};
    V3 up  = cross3(rgt, fwd);

    float cam[9] = {
        rgt.x, rgt.y, rgt.z,
        up.x,  up.y,  up.z,
        fwd.x, fwd.y, fwd.z
    };

    const float nearZ = 0.01f, farZ = 500.0f;
    float f   = 1.0f / tanf(fov);
    float asp = static_cast<float>(m_w) / static_cast<float>(m_h);
    float fn  = -(farZ + nearZ) / (farZ - nearZ);
    float fn2 = -2.0f * farZ * nearZ / (farZ - nearZ);

    float proj[16] = {
        f/asp, 0,  0,   0,
        0,     f,  0,   0,
        0,     0,  fn, -1,
        0,     0,  fn2, 0
    };

    float dre = dot3(rgt, {ex, ey, ez});
    float due = dot3(up,  {ex, ey, ez});
    float dfe = dot3(fwd, {ex, ey, ez});

    float view[16] = {
        rgt.x,  up.x, -fwd.x, 0,
        rgt.y,  up.y, -fwd.y, 0,
        rgt.z,  up.z, -fwd.z, 0,
       -dre,   -due,   dfe,   1
    };

    float vp[16] = {};
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++)
            for (int k   = 0; k   < 4; k++)
                vp[col*4+row] += proj[k*4+row] * view[col*4+k];

    // -----------------------------------------------------------------------
    // Animate wisps
    // -----------------------------------------------------------------------
    int   activeWisps = (wispCount < MAX_WISPS) ? wispCount : MAX_WISPS;
    float wispPosArr[MAX_WISPS * 3]   = {};
    float wispColorArr[MAX_WISPS * 3] = {};
    float wispIntArr[MAX_WISPS]       = {};
    for (int i = 0; i < activeWisps; i++) {
        const auto& w = wisps[i];
        float t = wispTime;
        wispPosArr[i*3+0] = w.basePos[0] + w.driftRadius * sinf(w.driftSpeed * t            + w.phase[0]);
        wispPosArr[i*3+1] = w.basePos[1] + w.driftRadius * cosf(w.driftSpeed * t * 0.7f     + w.phase[1]);
        wispPosArr[i*3+2] = w.basePos[2] + w.driftRadius * sinf(w.driftSpeed * t * 1.3f     + w.phase[2]);
        wispColorArr[i*3+0] = w.color[0];
        wispColorArr[i*3+1] = w.color[1];
        wispColorArr[i*3+2] = w.color[2];
        wispIntArr[i] = w.intensity;
    }

    // -----------------------------------------------------------------------
    // Pass 1: geometry — shading, no fog
    // -----------------------------------------------------------------------
    auto passClock = [] { return std::chrono::steady_clock::now(); };
    std::chrono::steady_clock::time_point tp0;
    if (profilePasses) { glFinish(); tp0 = passClock(); }

    bool invert = (shaderMode == ShaderMode::Invert);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_w, m_h);
    glClearColor(invert ? 0.f : 0.12f, invert ? 0.f : 0.08f, invert ? 0.f : 0.05f, 1.f);
    glEnable(GL_DEPTH_TEST);
    // Invert mode relaxes the depth test to GL_ALWAYS (depth is still written,
    // just doesn't gate which fragments draw) so every capsule covering a
    // pixel gets a chance to XOR it, not just the nearest one -- that's what
    // produces the "each overlap flips the color" effect.
    glDepthFunc(invert ? GL_ALWAYS : GL_LESS);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (invert) { glEnable(GL_COLOR_LOGIC_OP); glLogicOp(GL_XOR); }

    glUseProgram(m_prog);

    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_BUFFER, m_nodeTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_BUFFER, m_segTex);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_BUFFER, m_radTex);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_3D,     m_noiseTex);

    glUniform1i(m_uNodes,    0);
    glUniform1i(m_uSegs,     1);
    glUniform1i(m_uRads,     2);
    glUniform1i(m_uNoiseTex, 3);
    glUniform1i(m_uSegCount, m_segCount);
    glUniform3f(m_uEye,      ex, ey, ez);
    glUniformMatrix3fv(m_uCam, 1, GL_FALSE, cam);
    glUniform1f(m_uFov,      fov);
    glUniform2f(m_uRes,      static_cast<float>(m_w), static_cast<float>(m_h));
    glUniformMatrix4fv(m_uViewProj, 1, GL_FALSE, vp);
    glUniform3fv(m_uBaseColor, 1, mat.baseColor);
    glUniform3fv(m_uBaseColor2, 1, mat.baseColor2);
    glUniform1f(m_uColorNoiseScale, mat.colorNoiseScale);
    glUniform1f(m_uColorNoiseStrength, mat.colorNoiseStrength);
    glUniform1f (m_uAmbient,      mat.ambient);
    glUniform1f (m_uDiffuse,      mat.diffuse);
    glUniform3fv(m_uSpecColor, 1, mat.specColor);
    glUniform1f (m_uShininess,    mat.shininess);
    glUniform3fv(m_uLightDir,  1, lightDir3);
    glUniform1i (m_uShaderMode,   static_cast<int>(shaderMode));
    glUniform1f (m_uMetallic,     pbr.metallic);
    glUniform1f (m_uRoughness,    pbr.roughness);
    glUniform1i (m_uWispCount,    activeWisps);
    if (activeWisps > 0) {
        glUniform3fv(m_uWispPos,       activeWisps, wispPosArr);
        glUniform3fv(m_uWispColor,     activeWisps, wispColorArr);
        glUniform1fv(m_uWispIntensity, activeWisps, wispIntArr);
    }
    glUniform1f(m_uRadiusScale, radiusScale);
    glUniform1f(m_uRadiusMin, radiusMin);
    glUniform1f(m_uRadiusMax, radiusMax);

    glBindVertexArray(m_vao);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, m_segCount);
    glUseProgram(0);

    // Logic-op XOR is specific to the root capsule pass -- turn it off before
    // any extra geometry (face meshes) draws normally on top, and restore a
    // real depth test so that geometry composites sensibly against whatever
    // depth values the invert pass happened to leave behind.
    if (invert) { glDisable(GL_COLOR_LOGIC_OP); glDepthFunc(GL_LESS); }

    // m_fbo (color+depth) is still bound, GL_DEPTH_TEST still on, so extra
    // triangle geometry drawn here (e.g. face meshes) depth-composites
    // correctly against the sphere-traced root capsules and is included when
    // the fog pass below reads m_colorTex/m_depthTex.
    if (midGeometryHook) { float eye[3] = {ex, ey, ez}; midGeometryHook(vp, eye); }

    std::chrono::steady_clock::time_point tp1;
    if (profilePasses) {
        glFinish(); tp1 = passClock();
        lastGeomMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
    }

    glDisable(GL_DEPTH_TEST);

    // -----------------------------------------------------------------------
    // Pass 2: fog — fullscreen triangle reads colour + depth, writes to fogFbo
    // -----------------------------------------------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, m_fogFbo);
    glViewport(0, 0, m_w, m_h);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(m_fogProg);

    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, m_colorTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, m_depthTex);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_3D, m_noiseTex);

    glUniform1i(m_fpColorTex, 0);
    glUniform1i(m_fpDepthTex, 1);
    glUniform1i(m_fpNoiseTex, 2);
    glUniform3f(m_fpEye,  ex, ey, ez);
    glUniformMatrix3fv(m_fpCam, 1, GL_FALSE, cam);
    glUniform1f(m_fpFov,  fov);
    glUniform2f(m_fpRes,  static_cast<float>(m_w), static_cast<float>(m_h));
    glUniform1f(m_fpNear, nearZ);
    glUniform1f(m_fpFar,  farZ);
    glUniform3fv(m_fpFogColor,        1, fog.color);
    glUniform1f (m_fpFogDensity,         fog.density);
    glUniform1f (m_fpFogFalloff,         fog.falloff);
    glUniform1f (m_fpFogNoiseScale,      fog.noiseScale);
    glUniform1f (m_fpFogNoiseStrength,   fog.noiseStrength);
    glUniform1f (m_fpFogTime,            fog.driftTime);
    glUniform1f (m_fpFogRefDist,         fog.refDist);
    glUniform1f (m_fpWispGlowStrength,   wispGlowStrength);
    glUniform1i (m_fpNoiseType,          fog.noiseType);
    glUniform1i (m_fpShowAxes,           overlay.showAxes  ? 1 : 0);
    glUniform1f (m_fpAxisLength,         overlay.axisLength);
    glUniform1i (m_fpShowGrid,           overlay.showGrid  ? 1 : 0);
    glUniform1f (m_fpGridSpacing,        overlay.gridSpacing);
    glUniform1i (m_fpWispCount,          activeWisps);
    if (activeWisps > 0) {
        glUniform3fv(m_fpWispPos,       activeWisps, wispPosArr);
        glUniform3fv(m_fpWispColor,     activeWisps, wispColorArr);
        glUniform1fv(m_fpWispIntensity, activeWisps, wispIntArr);
    }

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (profilePasses) {
        glFinish();
        lastFogMs = std::chrono::duration<double, std::milli>(passClock() - tp1).count();
    }
}
