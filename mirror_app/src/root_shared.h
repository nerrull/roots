// root_shared.h — layout-compatible uniform structs shared by the Metal root
// renderer (C++/ObjC++ host) and its MSL passes (root_geom.metal, root_fog.metal).
//
// The same file is #included from both sides: __METAL_VERSION__ selects MSL
// vector/matrix types, otherwise <simd/simd.h>. simd_* and MSL types have
// matching size/alignment (float3=16B, float4x4=64B, float3x3=48B), so a struct
// written here maps byte-for-byte onto a Metal argument buffer. Vec3 quantities
// are stored as float4 (w unused) so member alignment never diverges.
#ifndef ROOT_SHARED_H
#define ROOT_SHARED_H

#ifdef __METAL_VERSION__
    // This header is prepended to the MSL passes before their own includes, so it
    // must bring the metal types into scope itself.
    #include <metal_stdlib>
    using namespace metal;
    #define RS_F2   float2
    #define RS_F4   float4
    #define RS_F3X3 float3x3
    #define RS_F4X4 float4x4
    #define RS_INT  int
#else
    #include <simd/simd.h>
    #define RS_F2   simd_float2
    #define RS_F4   simd_float4
    #define RS_F3X3 simd_float3x3
    #define RS_F4X4 simd_float4x4
    #define RS_INT  int32_t
#endif

#define ROOT_MAX_WISPS  50
#define ROOT_MAX_GROUPS 8

// Baked-noise tiling period; must match the CPU bake in metal_root_renderer.mm
// and the divisor in both MSL passes.
#define ROOT_NOISE_TILE_PERIOD 8.0f

// One drifting accent light. intensity packed in pos.w; color in .xyz.
struct RootWisp {
    RS_F4 pos;      // xyz = world position, w = intensity
    RS_F4 color;    // xyz = color
};

// Geometry pass (root_geom.metal): capsule/blade sphere-tracer + shading.
struct RootGeomU {
    RS_F4X4 viewProj;
    RS_F3X3 cam;          // camera basis (right, up, fwd) as columns
    RS_F4   eye;          // xyz
    RS_F4   baseColor;
    RS_F4   baseColor2;
    RS_F4   specColor;
    RS_F4   lightDir;
    RS_F4   pulseColor;
    RS_F2   res;
    float   fov;
    float   radiusScale;
    float   radiusMin;
    float   radiusMax;
    float   ambient;
    float   diffuse;
    float   shininess;
    float   colorNoiseScale;
    float   colorNoiseStrength;
    float   metallic;
    float   roughness;
    float   pulseSpeed;
    float   pulseSpacing;
    float   pulseWidth;
    float   pulseIntensity;
    float   pulseTime;
    RS_INT  shaderMode;   // 0 Phong, 1 PBR, 2 Invert
    RS_INT  wispCount;
    RS_INT  paletteCount;
    RS_INT  pulseEnabled;
    float   cullPx;       // drop capsules projecting smaller than this (0 = off)
    RS_INT  _pad1;
    // --- environment / organic shading --------------------------------------
    // A two-colour hemisphere standing in for an environment probe, plus the
    // terms that make a tube read as tissue rather than as painted plastic.
    RS_F4   skyColor;     // xyz, hemisphere upper
    RS_F4   groundColor;  // xyz, hemisphere lower (bounce)
    RS_F4   sssTint;      // xyz, colour light takes on its way through
    float   hemiStrength; // 0 = flat constant ambient (the old look)
    float   envSpec;      // roughness-blurred sky reflection
    float   rimStrength;  // Fresnel edge sheen
    float   sssWrap;      // diffuse wrap: (NdotL + w) / (1 + w)
    float   sssTrans;     // back-lit transmission amount
    float   sssPower;     // transmission lobe tightness
    // --- surface detail (capsules and blades) --------------------------------
    float   detailStrength;  // normal perturbation from the stretched noise
    float   detailScale;     // its frequency, in world units^-1
    float   detailStretch;   // how far features elongate along the root axis
    float   detailRough;     // specular/roughness break-up from the same field
    float   detailTint;      // per-segment albedo jitter
    RS_F4   keyColor;        // xyz, directional key colour x intensity
    RS_F4   palette[ROOT_MAX_GROUPS];
    RS_F4   paletteTip[ROOT_MAX_GROUPS];
};

// Face mid-geometry pass (root_face.metal): mask meshes rasterized into the
// shared colour+depth target between the capsule pass and fog.
struct RootFaceU {
    RS_F4X4 viewProj;
    RS_F4   eye;          // xyz
    RS_F4   lightDir;     // xyz
    RS_F4   veinColor;    // xyz
    float   lightIntensity;
    float   lightFalloff;
    float   specStrength;
    float   veinScale;
    float   veinStrength;
    float   roughness;    // base GGX roughness; the vein field modulates it
    float   metallic;
    float   reliefStrength;   // normal perturbation from the turbulence gradient
    RS_F4   skyColor;
    RS_F4   groundColor;
    RS_F4   sssTint;
    float   hemiStrength;
    float   envSpec;
    float   rimStrength;
    float   sssWrap;
    float   sssTrans;
    float   sssPower;
    float   reliefScale;  // relief frequency, as a multiple of veinScale
    // The mask light as a spotlight rather than a bare point: cosines of the
    // half-angles at which it starts and finishes falling off, aimed along the
    // mask's own facing.
    float   spotCosOuter;
    float   spotCosInner;
    float   spotLightDist;   // the offset the mesh builder used, so the shader
                             // can recover the off-axis angle from the distance
    float   _pad1;
    RS_F4   keyColor;     // xyz, the directional key's colour x intensity
};

// Leaf mid-geometry pass (root_leaf.metal): meshed leaves rasterized into the
// same shared colour+depth target as the face masks, for the same reason -- but
// with leaf shading (matte lamina, back-lit translucency) rather than stone.
struct RootLeafU {
    RS_F4X4 viewProj;
    RS_F4   eye;          // xyz
    RS_F4   lightDir;     // xyz
    RS_F4   skyColor;
    RS_F4   groundColor;
    RS_F4   sssTint;
    float   diffuse;
    float   specStrength;
    float   roughness;
    float   hemiStrength;
    float   rimStrength;
    float   sssTrans;     // back-lit transmission amount
    float   sssPower;     // transmission lobe tightness
    float   _pad0;
};

// Fog post-process pass (root_fog.metal).
struct RootFogU {
    RS_F3X3 cam;
    RS_F4   eye;
    RS_F4   fogColor;
    RS_F2   res;
    float   fov;
    float   nearZ;
    float   farZ;
    float   fogDensity;      // extinction per world unit = 1 / visibility
    float   fogHeightRef;    // world Y the height falloff pivots about
    float   fogHeightScale;  // Y distance over which density drops by 1/e
    float   fogNoiseScale;
    float   fogNoiseStrength;
    float   fogNoiseContrast;
    float   fogStart;        // march begins here: the air near the lens is clear
    RS_F4   fogDrift0;       // xyz, first octave's advection (pre-multiplied by time)
    RS_F4   fogDrift1;       // xyz, second octave's, deliberately not parallel
    float   wispGlowStrength;
    float   axisLength;
    float   gridSpacing;
    RS_INT  showAxes;
    RS_INT  showGrid;
    RS_INT  wispCount;
    RS_INT  aoEnabled;    // multiply the geometry pass's ambient share by the AO
    RS_INT  fogSteps;     // march samples between fogStart and the hit
    float   fogDither;    // jitter the march start, in units of one step
    float   fogScatter;   // scattering albedo: how much of the extinguished
                          // energy comes back into the ray rather than being
                          // absorbed. 0 = smoke, 1 = cloud.
    float   fogAnisotropy;// Henyey-Greenstein g: >0 forward, <0 back scattering
    RS_F4   lightDir;     // xyz, the key's direction (surface -> light)
    RS_F4   keyColor;     // xyz, key colour x intensity
};

// Screen-space ambient occlusion (root_ao.metal), run on the geometry pass's
// depth buffer between the geometry and fog passes.
struct RootAOU {
    RS_F3X3 cam;
    RS_F4   eye;
    RS_F2   res;          // AO buffer resolution (typically half the scene's)
    float   fov;
    float   nearZ;
    float   farZ;
    float   radius;       // world-space sampling radius
    float   intensity;
    float   bias;         // normal-plane offset, suppresses self-occlusion acne
    RS_INT  samples;
    RS_INT  blurDir;      // blur pass: 0 = horizontal, 1 = vertical
    RS_INT  _pad0;
    RS_INT  _pad1;
};

// Bloom down/up-sample (root_bloom.metal). One struct for both directions.
struct RootBloomU {
    RS_F2   srcTexel;     // 1 / source resolution
    float   threshold;    // prefilter knee (down-sample level 0 only)
    float   radius;       // up-sample filter width, in source texels
    RS_INT  prefilter;    // 1 = apply the threshold (level 0 of the chain)
    RS_INT  _pad0;
    RS_INT  _pad1;
    RS_INT  _pad2;
};

// Final composite (root_post.metal): supersample resolve, bloom, depth of
// field, exposure + filmic tonemap, vignette, grain, dither, sRGB encode.
struct RootPostU {
    RS_F2   res;          // output resolution
    RS_F2   srcTexel;     // 1 / scene (supersampled) resolution
    RS_INT  ssaa;         // supersample factor being resolved (1 = none)
    RS_INT  tonemap;      // 0 = clamp only, 1 = filmic
    RS_INT  bloomOn;
    RS_INT  dofOn;
    RS_INT  ditherOn;
    float   exposure;
    float   bloomIntensity;
    float   dofFocus;     // focus distance, world units
    float   dofRange;     // distance over which the blur reaches full strength
    float   dofStrength;
    float   vignette;
    float   grain;
    float   time;         // animates the grain
    float   nearZ;
    float   farZ;
    // --- lens ---------------------------------------------------------------
    float   caStrength;    // radial chromatic aberration, in pixels at the corner
    float   streak;        // anamorphic bloom streak intensity
    float   streakLength;  // its reach, in bloom-mip texels
    RS_F4   streakTint;
    // --- film ---------------------------------------------------------------
    float   halation;      // warm bleed around highlights
    RS_F4   halationTint;
    float   contrast;      // about a mid-grey pivot, display-referred
    float   saturation;
    RS_F4   lift;          // xyz, shadows
    RS_F4   gainC;         // xyz, highlights ("gain" collides with nothing, but
                           //      keep the C suffix to match liftC/gammaC below)
    RS_F4   gammaC;        // xyz, midtones
    RS_F4   shadowTint;    // split toning
    RS_F4   highlightTint;
    float   toneBalance;   // where the split between them sits
    float   grainSize;     // grain cell size, in output pixels
    float   grainChroma;   // 0 = monochrome grain, 1 = independent per channel
    float   splitStrength; // scales both split-tone tints towards neutral
    float   distortK1;     // radial distortion: <0 barrel, >0 pincushion
    float   distortK2;     // fourth-order term, for the corners
    float   distortZoom;   // re-crop so the distorted corners stay in frame
};

#endif // ROOT_SHARED_H
