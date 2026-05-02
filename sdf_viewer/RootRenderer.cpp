#include "RootRenderer.h"

#include "mymath.h"   // CPlantBox::Vector3d, Vector2i

#include <GL/glew.h>

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

// ===========================================================================
// RootRenderer
// ===========================================================================

RootRenderer::RootRenderer(int w, int h) : m_w(w), m_h(h) {
    buildShader();
    buildFBO();

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
        m_uAmbient   = glGetUniformLocation(m_prog, "u_ambient");
        m_uDiffuse   = glGetUniformLocation(m_prog, "u_diffuse");
        m_uSpecColor = glGetUniformLocation(m_prog, "u_specColor");
        m_uShininess = glGetUniformLocation(m_prog, "u_shininess");
        m_uLightDir  = glGetUniformLocation(m_prog, "u_lightDir");
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
        m_fpNoiseType        = glGetUniformLocation(m_fogProg, "u_noiseType");
        m_fpShowAxes         = glGetUniformLocation(m_fogProg, "u_showAxes");
        m_fpAxisLength       = glGetUniformLocation(m_fogProg, "u_axisLength");
        m_fpShowGrid         = glGetUniformLocation(m_fogProg, "u_showGrid");
        m_fpGridSpacing      = glGetUniformLocation(m_fogProg, "u_gridSpacing");
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
                           const float* lightDir3) {
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
    // Pass 1: geometry — Phong shading, no fog
    // -----------------------------------------------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_w, m_h);
    glClearColor(0.12f, 0.08f, 0.05f, 1.f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(m_prog);

    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_BUFFER, m_nodeTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_BUFFER, m_segTex);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_BUFFER, m_radTex);

    glUniform1i(m_uNodes,    0);
    glUniform1i(m_uSegs,     1);
    glUniform1i(m_uRads,     2);
    glUniform1i(m_uSegCount, m_segCount);
    glUniform3f(m_uEye,      ex, ey, ez);
    glUniformMatrix3fv(m_uCam, 1, GL_FALSE, cam);
    glUniform1f(m_uFov,      fov);
    glUniform2f(m_uRes,      static_cast<float>(m_w), static_cast<float>(m_h));
    glUniformMatrix4fv(m_uViewProj, 1, GL_FALSE, vp);
    glUniform3fv(m_uBaseColor, 1, mat.baseColor);
    glUniform1f (m_uAmbient,      mat.ambient);
    glUniform1f (m_uDiffuse,      mat.diffuse);
    glUniform3fv(m_uSpecColor, 1, mat.specColor);
    glUniform1f (m_uShininess,    mat.shininess);
    glUniform3fv(m_uLightDir,  1, lightDir3);

    glBindVertexArray(m_vao);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, m_segCount);

    glDisable(GL_DEPTH_TEST);
    glUseProgram(0);

    // -----------------------------------------------------------------------
    // Pass 2: fog — fullscreen triangle reads colour + depth, writes to fogFbo
    // -----------------------------------------------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, m_fogFbo);
    glViewport(0, 0, m_w, m_h);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(m_fogProg);

    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, m_colorTex);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, m_depthTex);

    glUniform1i(m_fpColorTex, 0);
    glUniform1i(m_fpDepthTex, 1);
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
    glUniform1i (m_fpNoiseType,          fog.noiseType);
    glUniform1i (m_fpShowAxes,           overlay.showAxes  ? 1 : 0);
    glUniform1f (m_fpAxisLength,         overlay.axisLength);
    glUniform1i (m_fpShowGrid,           overlay.showGrid  ? 1 : 0);
    glUniform1f (m_fpGridSpacing,        overlay.gridSpacing);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
