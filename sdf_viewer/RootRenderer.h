#pragma once

#include <GL/glew.h>
#include <functional>
#include <vector>

namespace CPlantBox { class Vector3d; class Vector2i; }

class RootRenderer {
public:
    enum class ShaderMode { Phong = 0, PBR = 1 };

    struct Material {
        float baseColor[3] = {0.60f, 0.55f, 0.45f};
        float ambient      = 0.03f;
        float diffuse      = 0.30f;
        float specColor[3] = {1.00f, 0.95f, 0.85f};
        float shininess    = 150.0f;
    };

    struct PBRParams {
        float metallic  = 0.05f;
        float roughness = 0.70f;
    };

    struct Fog {
        float color[3]      = {0.12f, 0.08f, 0.05f};
        float density       = 0.008f;
        float falloff       = 0.05f;
        float noiseScale    = 0.05f;
        float noiseStrength = 0.70f;
        float driftTime     = 0.0f;    // accumulated; advance by dt*driftSpeed each frame
        float driftSpeed    = 1.0f;
        int   noiseType     = 0;       // 0 = value noise, 1 = simplex
    };

    struct Overlay {
        bool  showAxes    = false;
        float axisLength  = 10.0f;    // cm
        bool  showGrid    = false;
        float gridSpacing = 5.0f;     // cm
    };

    struct WispDef {
        float basePos[3]  = {};
        float color[3]    = {0.8f, 0.9f, 1.0f};
        float intensity   = 3.0f;
        float driftRadius = 5.0f;
        float driftSpeed  = 0.5f;
        float phase[3]    = {};
    };
    static constexpr int MAX_WISPS = 4;

    RootRenderer(int w, int h);
    ~RootRenderer();

    void uploadSegments(const std::vector<CPlantBox::Vector3d>& nodes,
                        const std::vector<CPlantBox::Vector2i>& segments,
                        const std::vector<double>& radii);

    void resize(int w, int h);

    // midGeometryHook runs after the root capsules are sphere-traced (with
    // their real depth already in the shared depth buffer) but before the fog
    // post-process pass -- so a caller can rasterize extra triangle geometry
    // (e.g. the face meshes) into the same FBO, correctly depth-composited
    // against the roots and included in the fog pass that follows. Called with
    // the 4x4 view-projection matrix (16 floats, column-major) and eye pos.
    using MidHook = std::function<void(const float* viewProj, const float* eye3)>;
    void render(float azimuth, float elevation, float radius,
                const float* target3, float fov, const float* lightDir3,
                const MidHook& midGeometryHook = nullptr);

    GLuint colorTex() const { return m_fogColorTex; }
    int    width()    const { return m_w; }
    int    height()   const { return m_h; }

    ShaderMode shaderMode   = ShaderMode::Phong;
    Material   mat;
    PBRParams  pbr;
    Fog        fog;
    Overlay    overlay;
    WispDef    wisps[MAX_WISPS];
    int        wispCount    = 2;
    float      wispTime     = 0.0f;  // advance by dt each frame
    float      radiusScale  = 1.0f;

private:
    void buildFBO();
    void buildShader();

    int    m_w, m_h;

    GLuint m_fbo      = 0;
    GLuint m_colorTex = 0;
    GLuint m_depthTex = 0;

    GLuint m_fogFbo      = 0;
    GLuint m_fogColorTex = 0;
    GLuint m_fogProg     = 0;

    GLuint m_vao  = 0;
    GLuint m_prog = 0;

    GLuint m_nodeBuf = 0, m_nodeTex = 0;
    GLuint m_segBuf  = 0, m_segTex  = 0;
    GLuint m_radBuf  = 0, m_radTex  = 0;

    int m_segCount = 0;

    // Geometry pass
    int m_uNodes = -1, m_uSegs = -1, m_uRads = -1, m_uSegCount = -1;
    int m_uEye = -1, m_uCam = -1, m_uFov = -1, m_uRes = -1, m_uViewProj = -1;
    int m_uBaseColor = -1, m_uAmbient = -1, m_uDiffuse = -1;
    int m_uSpecColor = -1, m_uShininess = -1, m_uLightDir = -1;
    int m_uShaderMode = -1, m_uMetallic = -1, m_uRoughness = -1;
    int m_uWispCount = -1, m_uWispPos = -1, m_uWispColor = -1, m_uWispIntensity = -1;
    int m_uRadiusScale = -1;

    // Fog pass
    int m_fpColorTex = -1, m_fpDepthTex = -1;
    int m_fpEye = -1, m_fpCam = -1, m_fpFov = -1, m_fpRes = -1;
    int m_fpNear = -1, m_fpFar = -1;
    int m_fpFogColor = -1, m_fpFogDensity = -1, m_fpFogFalloff = -1;
    int m_fpFogNoiseScale = -1, m_fpFogNoiseStrength = -1, m_fpFogTime = -1;
    int m_fpNoiseType = -1;
    int m_fpShowAxes = -1, m_fpAxisLength = -1;
    int m_fpShowGrid = -1, m_fpGridSpacing = -1;
    int m_fpWispCount = -1, m_fpWispPos = -1, m_fpWispColor = -1, m_fpWispIntensity = -1;
};
