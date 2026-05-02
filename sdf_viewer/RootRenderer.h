#pragma once

#include <GL/glew.h>
#include <vector>

namespace CPlantBox { class Vector3d; class Vector2i; }

class RootRenderer {
public:
    struct Material {
        float baseColor[3] = {0.60f, 0.55f, 0.45f};
        float ambient      = 0.03f;
        float diffuse      = 0.30f;
        float specColor[3] = {1.00f, 0.95f, 0.85f};
        float shininess    = 150.0f;
    };

    struct Fog {
        float color[3]      = {0.12f, 0.08f, 0.05f};
        float density       = 0.008f;
        float falloff       = 0.05f;
        float noiseScale    = 0.05f;   // noise period ≈ 20 cm
        float noiseStrength = 0.70f;   // 0 = uniform, 1 = fully patchy
        float time          = 0.0f;    // wall-clock seconds; set each frame
    };

    RootRenderer(int w, int h);
    ~RootRenderer();

    void uploadSegments(const std::vector<CPlantBox::Vector3d>& nodes,
                        const std::vector<CPlantBox::Vector2i>& segments,
                        const std::vector<double>& radii);

    void resize(int w, int h);

    void render(float azimuth, float elevation, float radius,
                const float* target3, float fov, const float* lightDir3);

    // Returns the fog-composited output texture for display.
    GLuint colorTex() const { return m_fogColorTex; }
    int    width()    const { return m_w; }
    int    height()   const { return m_h; }

    Material mat;
    Fog      fog;

private:
    void buildFBO();
    void buildShader();

    int    m_w, m_h;

    // Geometry pass FBO
    GLuint m_fbo      = 0;
    GLuint m_colorTex = 0;   // Phong-shaded roots (no fog)
    GLuint m_depthTex = 0;   // depth texture (readable by fog pass)

    // Fog post-process pass
    GLuint m_fogFbo      = 0;
    GLuint m_fogColorTex = 0;   // final composited output
    GLuint m_fogProg     = 0;

    GLuint m_vao  = 0;
    GLuint m_prog = 0;  // geometry pass program

    // Texture Buffer Objects
    GLuint m_nodeBuf = 0, m_nodeTex = 0;
    GLuint m_segBuf  = 0, m_segTex  = 0;
    GLuint m_radBuf  = 0, m_radTex  = 0;

    int m_segCount = 0;

    // Geometry pass uniform locations
    int m_uNodes = -1, m_uSegs = -1, m_uRads = -1, m_uSegCount = -1;
    int m_uEye = -1, m_uCam = -1, m_uFov = -1, m_uRes = -1, m_uViewProj = -1;
    int m_uBaseColor = -1, m_uAmbient = -1, m_uDiffuse = -1;
    int m_uSpecColor = -1, m_uShininess = -1, m_uLightDir = -1;

    // Fog pass uniform locations (prefix fp = fog pass)
    int m_fpColorTex = -1, m_fpDepthTex = -1;
    int m_fpEye = -1, m_fpCam = -1, m_fpFov = -1, m_fpRes = -1;
    int m_fpNear = -1, m_fpFar = -1;
    int m_fpFogColor = -1, m_fpFogDensity = -1, m_fpFogFalloff = -1;
    int m_fpFogNoiseScale = -1, m_fpFogNoiseStrength = -1, m_fpFogTime = -1;
};
