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
        // Visibility, not density. Extinction is 1/visibility, and transmittance
        // is exp(-extinction * distance) -- so a slider that is linear in density
        // is linear in the *exponent*, and the whole useful range of it lives in
        // the first sixth of its travel. Everything past that is opaque and
        // everything before it is nothing, which is exactly the "either far too
        // dense or invisible" behaviour this had. Visibility is the distance at
        // which the fog reaches about 63%, so it is linear in something the eye
        // actually measures, and it is in world units you can point at.
        float visibility    = 45.0f;
        bool  enabled       = true;
        // The height gradient pivots about heightRef rather than about the world
        // origin, and heightScale is the distance over which density falls by
        // 1/e (the reciprocal of the old `falloff`, which was hard to reason
        // about at either end of its range).
        float heightRef     = 8.0f;
        float heightScale   = 22.0f;
        bool  heightRefAuto = true;    // track the camera target's Y
        // The noise is sampled at pos * noiseScale and the baked texture tiles
        // with period ROOT_NOISE_TILE_PERIOD (8) in that space, so a feature is
        // 8/noiseScale world units across. At the old 0.05 that is 160 units --
        // and the whole piece is about 20 across, so the scene sat inside a
        // single lobe of the noise. The fog therefore had no visible structure
        // at all, only an overall level that drifted; which is the other half of
        // why it looked like a slab moving along a plane. 0.55 puts a feature at
        // about 15 units, a bit under the size of the piece.
        float noiseScale    = 0.55f;
        float noiseStrength = 0.55f;
        float noiseContrast = 1.20f;   // how far the noise swings about the mean
        float driftTime     = 0.0f;
        float driftSpeed    = 1.0f;
        // The march starts here: the air between the lens and the subject reads
        // as clear. Auto ties it to a fraction of the orbit radius so the subject
        // stays clear at any framing.
        float startDist     = 0.0f;
        float startFrac     = 0.12f;
        bool  startAuto     = true;
        int   steps         = 14;
        // Single-scatter terms. `scatter` is the medium's albedo -- the share of
        // the light it removes from the beam that comes back into it -- and
        // `anisotropy` is how forward-biased that scattering is. Together they
        // are what couples the fog to the key light, so the fog brightens
        // towards it and stays dark away from it.
        float scatter       = 0.04f;
        // The volumetric integral runs at output resolution / this. It is a
        // smooth, low-frequency quantity with no silhouettes of its own, so it
        // does not need the grid the geometry needs -- at 1080p, halving it is
        // indistinguishable from full rate (mean error under 0.5/255) and takes
        // the march from 17 ms to 4.5. The march is by far the most expensive
        // thing in the fog, so this is the knob that pays for all of it.
        int   downscale     = 2;
        float anisotropy    = 0.55f;
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
        // Dimmer than it used to be (3.2): the mask's own point light sits a few
        // centimetres off its face, so at the old intensity it blew the forehead
        // and nose to clipped white and the mask read as a lamp rather than as a
        // lit object. With a tonemap in the chain there is also no longer any
        // need to overdrive it to get the highlights to register.
        float lightIntensity = 1.8f;
        float lightFalloff   = 0.012f;
        float specStrength   = 1.2f;
        float veinColor[3]   = {0.55f, 0.53f, 0.50f};
        float veinScale      = 0.6f;
        float veinStrength   = 0.5f;
        float roughness      = 0.42f;   // polished stone, not a mirror
        float metallic       = 0.0f;
        // Off. A per-pixel normal perturbation at stone-grain frequency reads as
        // mottled skin on a face -- the eye is far more sensitive to shading
        // irregularity on a face than on any other surface, and what looks like
        // pleasant granite on a slab looks like a skin condition on a cheek. The
        // knob stays for non-face uses of this pass; the default is smooth.
        float reliefStrength = 0.0f;
        float reliefScale    = 9.0f;    // relief frequency, as a multiple of veinScale
        // Area-weighted vertex normals rather than one face normal per triangle.
        // Lives here (rather than being unconditional) so the faceted original
        // is still reachable for comparison; RootScene reads it when it builds
        // the mask mesh, so changing it needs a rebuildFace().
        bool  smoothNormals  = true;
        // Spotlight cone for the mask's own light, in degrees off its axis. The
        // outer angle is where it reaches zero, the inner where it is still
        // full; the gap between them is the penumbra. 90 outer disables the cone
        // and gives the bare point light back.
        float spotOuterDeg   = 46.0f;
        float spotInnerDeg   = 20.0f;
        // Must match the lightDist RootScene passes to appendFaceVertexData.
        float spotLightDist  = 3.0f;
    };
    // Shading for the meshed leaves. Separate from FaceParams because a leaf is
    // a matte, thin, translucent sheet and a mask is polished stone; they share
    // the pass slot and the environment, not the material.
    struct LeafParams {
        float diffuse      = 0.95f;
        float specStrength = 0.12f;   // broad and weak: leaves are matte
        float roughness    = 0.55f;
        // Back-lit transmission. A leaf with the light behind it glows, and this
        // is most of what stops a mesh leaf reading as painted cardboard.
        float sssTrans     = 0.55f;
        float sssPower     = 2.6f;
    };
    LeafParams leaf;

    // Environment and organic-shading terms, shared by the capsule/blade pass
    // and the mask pass so both sit in the same light.
    struct EnvParams {
        float skyColor[3]    = {0.16f, 0.19f, 0.24f};   // cool from above
        float groundColor[3] = {0.10f, 0.07f, 0.045f};  // warm bounce from below
        float hemiStrength   = 1.0f;
        float envSpec        = 0.6f;
        float rimStrength    = 0.10f;
        float sssWrap        = 0.55f;
        float sssTrans       = 0.35f;
        float sssPower       = 5.0f;
        float sssTint[3]     = {0.90f, 0.45f, 0.22f};   // light reddens on its way through
        // The directional key. It had no colour or intensity of its own before --
        // it was implicitly white at unity, with `Material::diffuse` doing double
        // duty as both the surface's albedo response and the light's brightness,
        // so warming the key meant warming every material.
        float keyColor[3]    = {1.00f, 0.93f, 0.82f};
        float keyIntensity   = 1.0f;
    };
    // Fibre detail on the capsules and blades. Anisotropic on purpose -- see
    // root_geom.metal; isotropic noise here reads as grit on the surface rather
    // than as the lengthwise structure a root actually has.
    struct DetailParams {
        float strength = 0.55f;   // normal perturbation
        float scale    = 20.0f;   // frequency, world units^-1
        float stretch  = 7.0f;    // elongation along the root axis
        float rough    = 0.45f;   // specular / roughness break-up
        float tint     = 0.14f;   // per-segment albedo jitter
    };
    // Screen-space ambient occlusion over the geometry pass's depth buffer.
    struct AOParams {
        bool  enabled   = true;
        // Radius and intensity are tuned together against the *indirect* term
        // only (see root_fog.metal): occlusion here darkens the environment
        // share of a pixel, which is a fifth of its radiance, so an intensity
        // of 1.0 -- correct for AO applied to a whole image -- barely registers.
        float radius    = 2.2f;    // world units
        float intensity = 2.0f;
        float bias      = 0.04f;
        int   samples   = 10;
        int   downscale = 2;       // AO buffer is this much smaller than the scene
    };
    // The final composite: everything between the fog image and the drawable.
    struct PostParams {
        bool  enabled        = true;   // off = fog output goes straight out, as before
        bool  tonemap        = true;
        // Under 1 on purpose. The bright root groups carry an albedo close to
        // white, and at unity exposure they sat on the tonemap's shoulder where
        // every value desaturates towards the same white -- the roots went flat
        // and none of the material work below was visible on them at all.
        float exposure       = 1.20f;
        bool  bloom          = true;
        // Measured against this scene, not assumed. Its linear luminance runs
        // 0.16 at the median, 0.48 at the 99th percentile and about 1.0 at the
        // brightest pixel -- so the 1.35 this was originally set to (chosen when
        // the travelling pulses were blowing highlights past white) meant the
        // bloom chain, and the halation and streak that read from it, produced
        // *exactly nothing*: bit-identical output to bloom disabled, for 0.9 ms
        // a frame. The knee is subtractive, so only the excess over the
        // threshold blooms, and the 13-tap downsample averages a thin bright
        // root with its dark surroundings before the test -- both push the
        // effective threshold well below the nominal one.
        float bloomThreshold = 0.28f;
        float bloomIntensity = 0.50f;
        float bloomRadius    = 1.0f;
        int   bloomLevels    = 5;
        bool  dof            = true;
        // 0 = follow the camera's orbit radius, which is where the subject is.
        float dofFocus       = 0.0f;
        // Wide and weak. The intent is to take the edge off the far end of the
        // tangle so the eye settles on the mask in focus, not to shoot the scene
        // at f/1.4 -- and a gather this size cannot support a heavy blur without
        // showing its own kernel.
        float dofRange       = 55.0f;
        float dofStrength    = 0.50f;
        float vignette       = 0.22f;
        float grain          = 0.030f;
        bool  dither         = true;
        float fogDither      = 1.0f;
        int   ssaa           = 2;      // supersample factor for the scene passes

        // --- lens ------------------------------------------------------------
        // Chromatic aberration, in pixels of channel separation at the image
        // corner. Zero at the optical axis by construction.
        float caStrength     = 0.0f;
        // Anamorphic highlight streak. Off by default: it is a strong stylistic
        // signature and reads as "someone put a filter on it" if it is not what
        // the piece is going for.
        float streak         = 0.0f;
        float streakLength   = 14.0f;
        float streakTint[3]  = {0.55f, 0.72f, 1.00f};   // the classic cool streak

        // --- film ------------------------------------------------------------
        // Halation: warm re-exposure around genuinely bright areas. A small
        // amount is on by default because it is the least "effect-like" of
        // these and does the most to stop the highlights looking synthetic.
        float halation       = 0.35f;
        float halationTint[3] = {1.00f, 0.42f, 0.20f};
        int   halationMip    = 3;      // which bloom level supplies the spread

        // Print grade, applied after the tonemap.
        float contrast       = 1.0f;
        float saturation     = 1.0f;
        float lift[3]        = {0.0f, 0.0f, 0.0f};
        float gammaC[3]      = {1.0f, 1.0f, 1.0f};
        float gain[3]        = {1.0f, 1.0f, 1.0f};
        // Split toning. Enabled by a non-negative balance; the defaults cool the
        // shadows very slightly against this scene's warm key, which is what
        // separates the tangle from the ground without reading as a colour cast.
        float toneBalance    = 0.45f;
        // Stored at full strength and scaled by splitStrength, so one dial takes
        // the effect from off to full rather than needing two swatches edited in
        // step. The default cools the shadows against this scene's warm key,
        // which is what separates the tangle from the ground.
        float splitStrength    = 0.35f;
        float shadowTint[3]    = {0.88f, 0.95f, 1.14f};
        float highlightTint[3] = {1.08f, 1.00f, 0.90f};

        float grainSize      = 1.6f;   // grain cell, in output pixels
        float grainChroma    = 0.25f;  // 0 = monochrome, 1 = independent channels

        // --- lens geometry ---------------------------------------------------
        // Radial distortion. Negative is barrel (what short lenses do), positive
        // is pincushion (what long ones do). distortZoom re-crops so the bowed
        // corners stay inside the rendered image; 1.0 leaves them empty.
        float distortK1      = 0.0f;
        float distortK2      = 0.0f;
        float distortZoom    = 1.0f;
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

    // Leaf mid-geometry mesh: same 12-floats/vertex layout, but the last three
    // are (s, t, vein) rather than a light position, and it is drawn with leaf
    // shading instead of stone. Leaves are meshed rather than drawn on the
    // capsule/blade path because a blade SDF built from a union of capsules
    // cannot thin its margin to an edge — see sdf_viewer/LeafMesh.h.
    // Empty data clears the leaf pass.
    void uploadLeafMesh(const std::vector<float>& interleaved);

    // --- cached instances (many static root systems, LOD + culling) ----------
    // Placement of a cached system in the world (applied once, baked into the
    // uploaded vertices — cached systems are static).
    struct InstancePlacement {
        float translate[3] = {0.f, 0.f, 0.f};
        float rotYaw       = 0.f;   // radians about world Y
        float scale        = 1.f;
    };
    // Add a cached capsule system, baked to world space, with LOD levels built by
    // radius (thin laterals drop first) and a world bounding sphere for culling.
    // Uploaded once; drawn each frame only if visible, at the LOD its projected
    // size warrants. Returns the instance index.
    int  addInstance(const std::vector<float>& nodesXYZ,
                     const std::vector<int>&   segs,
                     const std::vector<float>& radii,
                     const InstancePlacement&  place);
    void clearInstances();
    int  instanceCount() const { return (int)instances_.size(); }

    // Culling / LOD tuning.
    bool  cullInstances = true;    // frustum-cull whole systems
    float instanceCullPx = 2.0f;   // skip systems whose bound projects smaller than this
    bool  subpixelCull   = true;   // drop sub-pixel capsules in the vertex shader
    float lodBias        = 1.0f;   // >1 favours coarser LODs sooner (cheaper)

    // Stats from the most recent render() (for UI / benchmarking).
    int  lastVisibleInstances = 0;
    int  lastCulledInstances  = 0;
    long lastDrawnSegments    = 0;

    // Encode both passes into cb; returns the final fogged colour texture.
    id<MTLTexture> render(id<MTLCommandBuffer> cb,
                          float azimuth, float elevation, float radius,
                          const float target3[3], float fov, const float lightDir3[3]);

    id<MTLTexture> colorTex() const { return outTex_; }
    int width()  const { return w_; }
    int height() const { return h_; }

    // True when the composite pass ran, i.e. the returned texture holds
    // display-referred sRGB-encoded values rather than linear radiance. The
    // headless capture paths need this: they used to apply a 1/2.2 gamma on the
    // way to a PPM, and doing that on top of the tonemap would wash the image
    // out. Asked of the renderer rather than tracked at each call site, because
    // it is the renderer that decides.
    bool outputIsEncoded() const { return post.enabled; }

    // Quality tranches, for A/B comparison and as a coarse quality dial.
    //   0  baseline    — the original two-pass look, every addition off
    //   1  fundamentals— smooth masks, hemisphere ambient, exposure + tonemap
    //   2  + lighting  — supersampling, SSAO, subsurface, environment specular
    //   3  + post      — bloom, depth of field, vignette, grain, dither
    // Applies to the render settings only; it does not touch materials or fog.
    void setTranche(int level);
    int  tranche() const { return tranche_; }

    // Public knobs (same defaults/meaning as RootRenderer).
    ShaderMode shaderMode = ShaderMode::Phong;
    Material   mat;
    PBRParams  pbr;
    Fog        fog;
    Pulse      pulse;
    Overlay    overlay;
    FaceParams face;
    EnvParams  env;
    DetailParams detail;
    AOParams   ao;
    PostParams post;
    float      postTime = 0.0f;   // drives the grain; advanced by the scene's clock
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
    id<MTLRenderPipelineState> leafPipe_ = nil;
    id<MTLRenderPipelineState> fogPipe_  = nil;
    id<MTLRenderPipelineState> fogVolPipe_ = nil;
    id<MTLRenderPipelineState> aoPipe_   = nil;
    id<MTLRenderPipelineState> aoBlurPipe_ = nil;
    id<MTLRenderPipelineState> bloomDownPipe_ = nil;
    id<MTLRenderPipelineState> bloomUpPipe_   = nil;   // additive blend
    id<MTLRenderPipelineState> postPipe_ = nil;
    id<MTLDepthStencilState>   depthState_ = nil;

    id<MTLBuffer> faceBuf_ = nil;
    int faceVertCount_ = 0;

    id<MTLBuffer> leafBuf_ = nil;
    int leafVertCount_ = 0;

    // The scene passes (geometry, mask, fog) run at sw_ x sh_, which is the
    // output size times the supersample factor; everything from the composite
    // on runs at w_ x h_.
    id<MTLTexture> rootColorTex_ = nil;   // sw_ x sh_, HDR + ambient share in alpha
    id<MTLTexture> rootDepthTex_ = nil;   // sw_ x sh_
    id<MTLTexture> fogColorTex_  = nil;   // sw_ x sh_, HDR, fog applied
    id<MTLTexture> aoTex_        = nil;   // sw_/ao.downscale, R8
    id<MTLTexture> aoBlurTex_    = nil;   // ping-pong for the separable blur
    id<MTLTexture> fogVolTex_    = nil;   // sw_/fog.downscale, (scatter.rgb, transmittance)
    id<MTLTexture> postTex_      = nil;   // w_ x h_, display-referred, presented
    id<MTLTexture> outTex_       = nil;   // whichever of the two the caller gets
    std::vector<id<MTLTexture>> bloomMips_;   // w_/2, w_/4, ... (post.bloomLevels)
    id<MTLTexture> noiseTex_     = nil;
    int sw_ = 0, sh_ = 0;         // scene (supersampled) resolution
    int builtSsaa_ = 0;           // the ssaa the current targets were built for
    int builtAoDs_ = 0;           // ditto for the AO downscale
    int builtFogDs_ = 0;          // ditto for the volumetric fog downscale
    int builtBloomLevels_ = 0;
    int tranche_ = 3;
    // Rebuild only what the changed setting invalidates.
    void ensureTargets();

    id<MTLBuffer> nodeBuf_ = nil, segBuf_ = nil, radBuf_ = nil, distBuf_ = nil;
    id<MTLBuffer> grpBuf_ = nil, primBuf_ = nil, frameBuf_ = nil, auxBuf_ = nil;
    int segCount_ = 0;

    // A cached, static capsule system. node/dist are per-node (shared across LODs);
    // each LOD holds a per-segment {seg, rad} subset. Constant per-segment
    // attributes (prim/aux/grp/frame) for capsule instances come from shared
    // default buffers grown to the largest LOD segment count.
    struct InstanceLod { id<MTLBuffer> seg = nil, rad = nil; int segCount = 0; };
    struct Instance {
        id<MTLBuffer> node = nil, dist = nil;
        std::vector<InstanceLod> lods;    // lods[0] = full detail
        float center[3] = {0, 0, 0};
        float radius = 0.f;               // world-space bounding-sphere radius
    };
    std::vector<Instance> instances_;
    id<MTLBuffer> defPrim_ = nil, defAux_ = nil, defGrp_ = nil, defFrame_ = nil;
    int defCap_ = 0;
    void ensureDefaults(int segCount);

    static constexpr MTLPixelFormat kColorFmt = MTLPixelFormatRGBA16Float;
    static constexpr MTLPixelFormat kDepthFmt = MTLPixelFormatDepth32Float;
    // AO is a single [0,1] visibility factor and 8 bits of it is plenty once
    // the bilateral blur has run over it.
    static constexpr MTLPixelFormat kAOFmt    = MTLPixelFormatR8Unorm;
};
