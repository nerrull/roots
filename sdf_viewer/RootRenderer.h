#pragma once

#include <GL/glew.h>
#include <vector>

namespace CPlantBox { class Vector3d; class Vector2i; }

class RootRenderer {
public:
    RootRenderer(int w, int h);
    ~RootRenderer();

    // Upload geometry from a SegmentAnalyser snapshot; call whenever segment count changes.
    void uploadSegments(const std::vector<CPlantBox::Vector3d>& nodes,
                        const std::vector<CPlantBox::Vector2i>& segments,
                        const std::vector<double>& radii);

    // Recreate the FBO if the requested size differs from the current size.
    void resize(int w, int h);

    // Render the scene into the internal FBO.
    // azimuth/elevation in radians, radius in cm, target3 is the look-at point,
    // fov is the vertical half-angle in radians.
    void render(float azimuth, float elevation, float radius,
                const float* target3, float fov);

    GLuint colorTex() const { return m_colorTex; }
    int    width()    const { return m_w; }
    int    height()   const { return m_h; }

private:
    void buildFBO();
    void buildShader();

    int    m_w, m_h;
    GLuint m_fbo      = 0;
    GLuint m_colorTex = 0;
    GLuint m_vao      = 0;
    GLuint m_prog     = 0;

    // Texture Buffer Objects: (buffer, texture) pairs
    GLuint m_nodeBuf = 0, m_nodeTex = 0;   // GL_RGB32F   — vec3 node positions
    GLuint m_segBuf  = 0, m_segTex  = 0;   // GL_RG32I    — ivec2 node index pairs
    GLuint m_radBuf  = 0, m_radTex  = 0;   // GL_R32F     — float radius per segment

    int m_segCount = 0;

    // Cached uniform locations
    int m_uNodes = -1, m_uSegs = -1, m_uRads = -1;
    int m_uSegCount = -1;
    int m_uEye = -1, m_uCam = -1, m_uFov = -1, m_uRes = -1;
};
