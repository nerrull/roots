#include "RootRenderer.h"

#include "mymath.h"   // CPlantBox::Vector3d, Vector2i

#include <GL/glew.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny vec3 helpers (avoids pulling in GLM)
// ---------------------------------------------------------------------------
struct V3 { float x, y, z; };

static float dot3(V3 a, V3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
static V3  cross3(V3 a, V3 b)   { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
static V3  norm3(V3 v)          { float l=sqrtf(dot3(v,v)); if(l<1e-7f)l=1.f; return{v.x/l,v.y/l,v.z/l}; }

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
    GLuint s = glCreateShaderwht(type);
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
// TBO helper: create buffer + texture object pair, seed with 1 dummy float
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

    // Bind each texture to its buffer with the correct internal format.
    // These bindings are permanent — we only re-upload buffer data, never rebind.
    glBindTexture(GL_TEXTURE_BUFFER, m_nodeTex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F,  m_nodeBuf);

    glBindTexture(GL_TEXTURE_BUFFER, m_segTex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32I,   m_segBuf);

    glBindTexture(GL_TEXTURE_BUFFER, m_radTex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F,    m_radBuf);

    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // Empty VAO required by the core profile for attribute-less draws
    glGenVertexArrays(1, &m_vao);
}

RootRenderer::~RootRenderer() {
    glDeleteProgram(m_prog);
    glDeleteVertexArrays(1, &m_vao);
    glDeleteFramebuffers(1, &m_fbo);
    glDeleteTextures(1, &m_colorTex);
    glDeleteBuffers(1,  &m_nodeBuf);
    glDeleteBuffers(1,  &m_segBuf);
    glDeleteBuffers(1,  &m_radBuf);
    glDeleteTextures(1, &m_nodeTex);
    glDeleteTextures(1, &m_segTex);
    glDeleteTextures(1, &m_radTex);
}

void RootRenderer::buildShader() {
    auto vertSrc = readFile(SHADER_DIR "/shader.vert");
    auto fragSrc = readFile(SHADER_DIR "/shader.frag");

    GLuint vs = compileShader(GL_VERTEX_SHADER,   vertSrc.c_str());
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc.c_str());
    m_prog = linkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);

    m_uNodes    = glGetUniformLocation(m_prog, "u_nodes");
    m_uSegs     = glGetUniformLocation(m_prog, "u_segments");
    m_uRads     = glGetUniformLocation(m_prog, "u_radii");
    m_uSegCount = glGetUniformLocation(m_prog, "u_segCount");
    m_uEye      = glGetUniformLocation(m_prog, "u_eye");
    m_uCam      = glGetUniformLocation(m_prog, "u_cam");
    m_uFov      = glGetUniformLocation(m_prog, "u_fov");
    m_uRes      = glGetUniformLocation(m_prog, "u_res");
}

void RootRenderer::buildFBO() {
    if (m_fbo)      { glDeleteFramebuffers(1, &m_fbo);      m_fbo      = 0; }
    if (m_colorTex) { glDeleteTextures(1,    &m_colorTex);  m_colorTex = 0; }

    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    glGenTextures(1, &m_colorTex);
    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_w, m_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_colorTex, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("FBO is not complete");

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

    // Pack node positions as float3
    std::vector<float> nodeData;
    nodeData.reserve(nodes.size() * 3);
    for (auto& n : nodes) {
        nodeData.push_back(static_cast<float>(n.x));
        nodeData.push_back(static_cast<float>(n.y));
        nodeData.push_back(static_cast<float>(n.z));
    }
    if (nodeData.empty()) nodeData.assign(3, 0.f);

    // Pack segment indices as int2
    std::vector<int> segData;
    segData.reserve(segs.size() * 2);
    for (auto& s : segs) { segData.push_back(s.x); segData.push_back(s.y); }
    if (segData.empty()) segData.assign(2, 0);

    // Pack radii as float
    std::vector<float> radData;
    radData.reserve(radii.size());
    for (double r : radii) radData.push_back(static_cast<float>(r));
    if (radData.empty()) radData.push_back(0.f);

    glBindBuffer(GL_TEXTURE_BUFFER, m_nodeBuf);
    glBufferData(GL_TEXTURE_BUFFER,
                 static_cast<GLsizeiptr>(nodeData.size() * sizeof(float)),
                 nodeData.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_TEXTURE_BUFFER, m_segBuf);
    glBufferData(GL_TEXTURE_BUFFER,
                 static_cast<GLsizeiptr>(segData.size() * sizeof(int)),
                 segData.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_TEXTURE_BUFFER, m_radBuf);
    glBufferData(GL_TEXTURE_BUFFER,
                 static_cast<GLsizeiptr>(radData.size() * sizeof(float)),
                 radData.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

void RootRenderer::render(float azimuth, float elevation, float radius,
                           const float* target3, float fov) {
    // Compute eye position on a sphere around target
    float cosEl = cosf(elevation), sinEl = sinf(elevation);
    float cosAz = cosf(azimuth),   sinAz = sinf(azimuth);

    float ex = target3[0] + radius * cosEl * sinAz;
    float ey = target3[1] + radius * sinEl;
    float ez = target3[2] + radius * cosEl * cosAz;

    // Camera basis: forward, right, up
    V3 fwd = norm3({target3[0]-ex, target3[1]-ey, target3[2]-ez});
    V3 wup = {0.f, 1.f, 0.f};

    V3 rawRgt = cross3(fwd, wup);
    V3 rgt = (dot3(rawRgt, rawRgt) > 1e-8f) ? norm3(rawRgt) : V3{1.f, 0.f, 0.f};
    V3 up  = cross3(rgt, fwd);   // guaranteed unit length

    // Column-major mat3 for GLSL: columns are [right, up, forward]
    float cam[9] = {
        rgt.x, rgt.y, rgt.z,
        up.x,  up.y,  up.z,
        fwd.x, fwd.y, fwd.z
    };

    // Draw into FBO
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_w, m_h);
    glClearColor(0.12f, 0.08f, 0.05f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

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

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
