// MetalRootRenderer — Metal port of sdf_viewer/RootRenderer.
//
// Same two-pass structure as the GL original (geometry sphere-tracer + fog
// post-process) and the same public knobs (Material/PBRParams/Fog/Pulse/Overlay/
// WispDef), but rendered through the app's shared MetalContext into offscreen
// MTLTextures. GL Texture-Buffer-Objects become plain MTLBuffers; the procedural
// noise is the same baked 128^3 tiling fBm. Segment upload takes flat arrays
// rather than CPlantBox types so this stays independent of the sim (the RootScene
// converts CPlantBox output to these arrays).
//
// Divergence from GL: Invert/XOR mode has no Metal fragment-logic-op equivalent,
// so it renders flat white silhouettes (nearest-depth) rather than XOR overlap.
//
// ObjC++ only.
#pragma once
#ifndef __OBJC__
#error "metal_root_renderer.h is ObjC++ only; include from a .mm file"
#endif

#import <Metal/Metal.h>
#include "root_shared.h"
#include <string>
#include <vector>

class MetalContext;

class MetalRootRenderer {
public:
    enum class ShaderMode { Phong = 0, PBR = 1, Invert = 2 };

    struct Material {
        float baseColor[3]  = {0.60f, 0.55f, 0.45f};
        float baseColor2[3] = {0.35f, 0.28f, 0.22f};
        float colorNoiseScale    = 0.4f;
        float colorNoiseStrength = 0.0f;
        float ambient      = 0.03f;
        float diffuse      = 0.30f;
        float specColor[3] = {1.00f, 0.95f, 0.85f};
        float shininess    = 150.0f;
    };
    struct PBRParams { float metallic = 0.05f; float roughness = 0.70f; };
    struct Fog {
        float color[3]      = {0.12f, 0.08f, 0.05f};
        float density       = 0.008f;
        float falloff       = 0.05f;
        float noiseScale    = 0.05f;
        float noiseStrength = 0.70f;
        float driftTime     = 0.0f;
        float driftSpeed    = 1.0f;
        float refDist       = 0.0f;
        int   noiseType     = 0;
    };
    struct Pulse {
        bool  enabled   = false;
        float color[3]  = {1.0f, 0.85f, 0.45f};
        float speed     = 14.0f;
        float spacing   = 22.0f;
        float width     = 3.5f;
        float intensity = 1.6f;
        float time      = 0.0f;
        float hopOffset = 12.0f;
    };
    struct Overlay {
        bool  showAxes    = false;
        float axisLength  = 10.0f;
        bool  showGrid    = false;
        float gridSpacing = 5.0f;
    };
    // Material for the face mid-geometry pass (mirrors FaceGL's knobs).
    struct FaceParams {
        float lightIntensity = 3.2f;
        float lightFalloff   = 0.012f;
        float specStrength   = 1.2f;
        float veinColor[3]   = {0.55f, 0.53f, 0.50f};
        float veinScale      = 0.6f;
        float veinStrength    = 0.5f;
    };
    struct WispDef {
        float basePos[3]  = {};
        float color[3]    = {0.8f, 0.9f, 1.0f};
        float intensity   = 3.0f;
        float driftRadius = 5.0f;
        float driftSpeed  = 0.5f;
        float phase[3]    = {};
    };
    static constexpr int MAX_WISPS  = ROOT_MAX_WISPS;
    static constexpr int MAX_GROUPS = ROOT_MAX_GROUPS;

    MetalRootRenderer(const MetalContext& ctx, const std::string& shaderDir,
                      const std::string& sharedHeaderPath, int w, int h);

    bool valid() const { return geomPipe_ != nil && fogPipe_ != nil; }

    void resize(int w, int h);

    // Flat-array segment upload. nodesXYZ: 3 floats/node; segs: 2 ints/seg;
    // radii: 1 float/seg. Optional per-segment: groups (1 int), prims (1 int),
    // frames (4 floats), aux (4 floats). Absent optionals default as in GL.
    void uploadSegments(const std::vector<float>& nodesXYZ,
                        const std::vector<int>&   segs,
                        const std::vector<float>& radii,
                        const std::vector<int>*   groups = nullptr,
                        const std::vector<int>*   prims  = nullptr,
                        const std::vector<float>* frames = nullptr,
                        const std::vector<float>* aux    = nullptr);

    // Face mid-geometry mesh: flat interleaved triangles, 12 floats/vertex
    // (pos3, normal3, color3, lightPos3) — same layout as FaceGL's VBO. Drawn
    // into the shared colour+depth target between the capsules and the fog.
    // Empty data clears the face pass.
    void uploadFaceMesh(const std::vector<float>& interleaved);

    // Encode both passes into cb; returns the final fogged colour texture.
    id<MTLTexture> render(id<MTLCommandBuffer> cb,
                          float azimuth, float elevation, float radius,
                          const float target3[3], float fov, const float lightDir3[3]);

    id<MTLTexture> colorTex() const { return fogColorTex_; }
    int width()  const { return w_; }
    int height() const { return h_; }

    // Public knobs (same defaults/meaning as RootRenderer).
    ShaderMode shaderMode = ShaderMode::Phong;
    Material   mat;
    PBRParams  pbr;
    Fog        fog;
    Pulse      pulse;
    Overlay    overlay;
    FaceParams face;
    WispDef    wisps[MAX_WISPS];
    int        wispCount        = 2;
    float      wispGlowStrength = 1.0f;
    float      wispTime         = 0.0f;
    float      radiusScale      = 1.0f;
    float      radiusMin        = 0.0f;
    float      radiusMax        = 0.0f;
    float      palette[MAX_GROUPS][3]    = {};
    int        paletteCount     = 0;
    float      paletteTip[MAX_GROUPS][3] = {};
    int        paletteTipCount  = 0;

private:
    void buildTargets();
    void buildNoiseTexture();
    id<MTLBuffer> makeBuffer(const void* data, size_t bytes);

    id<MTLDevice> device_ = nil;
    int w_ = 0, h_ = 0;

    id<MTLRenderPipelineState> geomPipe_ = nil;
    id<MTLRenderPipelineState> facePipe_ = nil;
    id<MTLRenderPipelineState> fogPipe_  = nil;
    id<MTLDepthStencilState>   depthState_ = nil;

    id<MTLBuffer> faceBuf_ = nil;
    int faceVertCount_ = 0;

    id<MTLTexture> rootColorTex_ = nil;
    id<MTLTexture> rootDepthTex_ = nil;
    id<MTLTexture> fogColorTex_  = nil;
    id<MTLTexture> noiseTex_     = nil;

    id<MTLBuffer> nodeBuf_ = nil, segBuf_ = nil, radBuf_ = nil, distBuf_ = nil;
    id<MTLBuffer> grpBuf_ = nil, primBuf_ = nil, frameBuf_ = nil, auxBuf_ = nil;
    int segCount_ = 0;

    static constexpr MTLPixelFormat kColorFmt = MTLPixelFormatRGBA16Float;
    static constexpr MTLPixelFormat kDepthFmt = MTLPixelFormatDepth32Float;
};
