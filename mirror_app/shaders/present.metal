// present.metal — draw a scene texture to the drawable as a fullscreen triangle,
// linearly sampled (this is the "bilinear upsample" of the low-res mirror image,
// done for free in the sampler).
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 pos [[position]];
    float2 uv;
};

vertex VOut present_vs(uint vid [[vertex_id]]) {
    // Fullscreen triangle: (0,0), (2,0), (0,2) in [0,2] → clip space.
    float2 p = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv  = float2(p.x, 1.0 - p.y);   // flip v so image row 0 is at the bottom
    return o;
}

fragment float4 present_fs(VOut in [[stage_in]],
                           texture2d<float> tex [[texture(0)]]) {
    constexpr sampler smp(mag_filter::linear, min_filter::linear,
                          address::clamp_to_edge);
    return tex.sample(smp, in.uv);
}
