// The fit view: what is being fitted, what is being tracked, and whether the
// two are in register.
//
// Everything else in the app composites the fit into something -- a pond, a
// root scene, a falling sheet -- where a mask that is subtly misplaced still
// looks like an effect. This draws the pieces flat and on top of each other so
// a misplacement reads as a misplacement:
//
//   layer 0   the fit's own output (or the camera frame it is fitted to)
//   layer 1   the training mask, tinted, so the supervised pixels are visible
//   layer 2   the fitted mesh at its projected position, wearing the colours
//             sampled for it -- if those colours do not match the layer under
//             the mesh, the texture is not pinned
#include <metal_stdlib>
using namespace metal;

struct Params {
    float2 mask_size;      // pixels; 0 = no mask uploaded
    float  mask_tint;      // 0..1 strength of the mask overlay
    float  bg_dim;         // dim the background so the mesh reads on top
    float  mesh_alpha;     // 0 = mesh hidden, 1 = opaque
    float  _pad[3];
};

struct FOut {
    float4 clip [[position]];
    float2 uv;
};

// Fullscreen triangle: three vertices covering the viewport, no buffers.
vertex FOut v_bg(uint vid [[vertex_id]]) {
    const float2 p[3] = { float2(-1.0, -3.0), float2(-1.0, 1.0), float2(3.0, 1.0) };
    FOut o;
    o.clip = float4(p[vid], 0.0, 1.0);
    // Clip space is y-up, images are y-down.
    o.uv = float2((p[vid].x + 1.0) * 0.5, (1.0 - p[vid].y) * 0.5);
    return o;
}

fragment float4 f_bg(FOut in [[stage_in]],
                     texture2d<float> src  [[texture(0)]],
                     texture2d<float> mask [[texture(1)]],
                     constant Params& u    [[buffer(0)]]) {
    constexpr sampler smp(filter::linear, address::clamp_to_edge);
    float3 c = src.sample(smp, in.uv).rgb * u.bg_dim;
    if (u.mask_size.x > 0.5) {
        // Nearest, deliberately: the mask is the exact pixel set the gradient
        // sees, and a filtered edge would suggest a softness the training does
        // not have.
        constexpr sampler nrst(filter::nearest, address::clamp_to_edge);
        const float m = mask.sample(nrst, in.uv).r;
        c = mix(c, mix(c, float3(0.25, 0.85, 0.55), 0.45), m * u.mask_tint);
    }
    return float4(c, 1.0);
}

// ---- the fitted mesh --------------------------------------------------------

// Packed, and it matters: a plain float3 is 16-byte aligned in Metal, so this
// struct would be 32 bytes against the 20 the C++ side writes, and every
// vertex after the first would be read from the wrong offset. The symptom is a
// mesh that silently fails to appear.
struct MeshVert {
    packed_float2 uv;      // normalised frame position, y-down
    packed_float3 col;     // colour sampled for this vertex
};

struct MOut {
    float4 clip [[position]];
    float3 col;
};

vertex MOut v_mesh(uint vid [[vertex_id]],
                   device const MeshVert* verts [[buffer(0)]]) {
    MOut o;
    const float2 uv = verts[vid].uv;
    o.clip = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    o.col = verts[vid].col;
    return o;
}

fragment float4 f_mesh(MOut in [[stage_in]], constant Params& u [[buffer(0)]]) {
    return float4(in.col, u.mesh_alpha);
}

// Flat colour, for the wireframe and the landmark dots.
fragment float4 f_flat(MOut in [[stage_in]], constant Params& u [[buffer(0)]]) {
    return float4(in.col, 1.0);
}
