#pragma once

#include <GL/glew.h>
#include <functional>
#include <vector>

namespace CPlantBox { class Vector3d; class Vector2i; }

class RootRenderer {
public:
    enum class ShaderMode { Phong = 0, PBR = 1, Invert = 2 };

    struct Material {
        float baseColor[3]  = {0.60f, 0.55f, 0.45f};
        float baseColor2[3] = {0.35f, 0.28f, 0.22f};   // blended in by color noise
        float colorNoiseScale    = 0.4f;
        float colorNoiseStrength = 0.0f;               // 0 = flat single color
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
        float refDist       = 0.0f;    // >0: subtract density*refDist from optical
                                       // depth (camera-stable fog); 0 = classic
        int   noiseType     = 0;       // 0 = value noise, 1 = simplex
    };

    // Pulses of light travelling along the roots: an emissive band that scrolls
    // outward along each root's arc-length from its base. Purely additive on top
    // of the normal shading, driven by a per-node "distance from base" buffer.
    struct Pulse {
        bool  enabled   = false;
        float color[3]  = {1.0f, 0.85f, 0.45f};
        float speed     = 14.0f;   // cm of arc-length per unit time
        float spacing   = 22.0f;   // cm between consecutive pulses
        float width     = 3.5f;    // cm; length of the bright band
        float intensity = 1.6f;    // additive brightness at the crest
        float time      = 0.0f;    // accumulated; advance each frame
        // Arc-length added to each successive hop's base, so the pulse train is
        // out of phase from mask to mask -- reads as the wave travelling along
        // the relay rather than every mask firing in unison. Baked into the
        // per-node distance in uploadSegments (0 = all hops in phase).
        float hopOffset = 12.0f;
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
    static constexpr int MAX_WISPS = 50;

    RootRenderer(int w, int h);
    ~RootRenderer();

    // prims: optional per-segment primitive type (0 = capsule, 1 = blade).
    // frames: optional per-segment vec4 (4 floats/segment) -- for blades,
    // xyz = half-width vector, w = curl. Both null -> all capsules (unchanged).
    void uploadSegments(const std::vector<CPlantBox::Vector3d>& nodes,
                        const std::vector<CPlantBox::Vector2i>& segments,
                        const std::vector<double>& radii,
                        const std::vector<int>* groups = nullptr,
                        const std::vector<int>* prims = nullptr,
                        const std::vector<float>* frames = nullptr,
                        const std::vector<float>* aux = nullptr);

    // Per-segment colour groups: a cheap alternative to a full RGB-per-segment
    // attribute. Each segment carries a small integer index (uploaded via the
    // `groups` arg of uploadSegments); the shader looks that index up in this
    // palette for the segment's base colour, so a whole plant of thousands of
    // capsules costs one int per capsule + a handful of palette colours rather
    // than three floats per capsule. paletteCount = 0 disables it and falls
    // back to the single mat.baseColor / baseColor2 material.
    static constexpr int MAX_GROUPS = 8;
    float palette[MAX_GROUPS][3] = {};
    int   paletteCount = 0;
    // Second colour per group: petals/blades lerp palette -> paletteTip along
    // their length (base -> tip) by the per-segment gradient in the aux buffer.
    // paletteTipCount == 0 -> tips default to the base palette (no gradient).
    float paletteTip[MAX_GROUPS][3] = {};
    int   paletteTipCount = 0;

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

    // Per-pass GPU attribution: when enabled, render() brackets the geometry
    // pass (incl. midGeometryHook) and the fog pass with glFinish + a wall
    // clock, so each pass's number is true GPU completion time, not async
    // submission time. Costs pipeline overlap -- leave off outside profiling.
    bool   profilePasses = false;
    double lastGeomMs    = 0.0;
    double lastFogMs     = 0.0;

    ShaderMode shaderMode   = ShaderMode::Phong;
    Material   mat;
    PBRParams  pbr;
    Fog        fog;
    Pulse      pulse;
    Overlay    overlay;
    WispDef    wisps[MAX_WISPS];
    int        wispCount    = 2;
    float      wispGlowStrength = 1.0f;   // fog-pass glow blobs (fog-gated in fog.frag)
    float      wispTime     = 0.0f;  // advance by dt each frame
    float      radiusScale  = 1.0f;
    float      radiusMin    = 0.0f;   // 0 = no floor clamp
    float      radiusMax    = 0.0f;   // 0 = no ceiling clamp

private:
    void buildFBO();
    void buildShader();
    void buildNoiseTexture();

    int    m_w, m_h;

    GLuint m_fbo      = 0;
    GLuint m_colorTex = 0;
    GLuint m_depthTex = 0;

    GLuint m_fogFbo      = 0;
    GLuint m_fogColorTex = 0;
    GLuint m_fogProg     = 0;

    GLuint m_vao  = 0;
    GLuint m_prog = 0;

    GLuint m_noiseTex = 0;   // tiling 3D fBm, replaces per-pixel procedural noise

    GLuint m_nodeBuf = 0, m_nodeTex = 0;
    GLuint m_segBuf  = 0, m_segTex  = 0;
    GLuint m_radBuf  = 0, m_radTex  = 0;
    GLuint m_distBuf = 0, m_distTex = 0;   // per-node arc-length from root base
    GLuint m_grpBuf  = 0, m_grpTex  = 0;   // per-segment palette group index
    GLuint m_primBuf = 0, m_primTex = 0;   // per-segment primitive type
    GLuint m_frameBuf= 0, m_frameTex= 0;   // per-segment blade frame (vec4)
    GLuint m_auxBuf  = 0, m_auxTex  = 0;   // per-segment aux (s0,s1,grad0,grad1)

    int m_segCount = 0;

    // Geometry pass
    int m_uNodes = -1, m_uSegs = -1, m_uRads = -1, m_uSegCount = -1;
    int m_uEye = -1, m_uCam = -1, m_uFov = -1, m_uRes = -1, m_uViewProj = -1;
    int m_uBaseColor = -1, m_uAmbient = -1, m_uDiffuse = -1;
    int m_uBaseColor2 = -1, m_uColorNoiseScale = -1, m_uColorNoiseStrength = -1;
    int m_uSpecColor = -1, m_uShininess = -1, m_uLightDir = -1;
    int m_uShaderMode = -1, m_uMetallic = -1, m_uRoughness = -1;
    int m_uWispCount = -1, m_uWispPos = -1, m_uWispColor = -1, m_uWispIntensity = -1;
    int m_uRadiusScale = -1, m_uRadiusMin = -1, m_uRadiusMax = -1;
    int m_uNoiseTex = -1;
    int m_uDist = -1;
    int m_uGroups = -1, m_uPalette = -1, m_uPaletteCount = -1;
    int m_uPrimType = -1, m_uPrimFrame = -1, m_uPrimAux = -1, m_uPaletteTip = -1;
    int m_uPulseEnabled = -1, m_uPulseColor = -1, m_uPulseSpeed = -1;
    int m_uPulseSpacing = -1, m_uPulseWidth = -1, m_uPulseIntensity = -1, m_uPulseTime = -1;

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
    int m_fpNoiseTex = -1;
    int m_fpFogRefDist = -1, m_fpWispGlowStrength = -1;
};
