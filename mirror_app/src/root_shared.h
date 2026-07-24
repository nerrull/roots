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
    RS_INT  _pad0;
    RS_INT  _pad1;
    RS_F4   palette[ROOT_MAX_GROUPS];
    RS_F4   paletteTip[ROOT_MAX_GROUPS];
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
    float   fogDensity;
    float   fogFalloff;
    float   fogNoiseScale;
    float   fogNoiseStrength;
    float   fogTime;
    float   fogRefDist;
    float   wispGlowStrength;
    float   axisLength;
    float   gridSpacing;
    RS_INT  showAxes;
    RS_INT  showGrid;
    RS_INT  wispCount;
    RS_INT  _pad0;
};

#endif // ROOT_SHARED_H
