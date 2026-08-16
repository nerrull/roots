// root_bloom.metal — the down/up-sample pair for a progressive bloom chain
// (the "dual filter" / COD-Siggraph-2014 approach).
//
// A single wide Gaussian would need an enormous kernel to spread light the way
// the wisps and the travelling pulses want, and it bands badly at that width. A
// chain of 13-tap downsamples followed by 9-tap tent upsamples, each blended
// back into the level above, reaches the same spread for a fraction of the taps
// and lands on a much smoother falloff -- which matters here because the bloom
// is sitting on top of volumetric fog and any structure in it reads as an
// artefact of the fog rather than of the bloom.
#include <metal_stdlib>
using namespace metal;

struct BloomVOut { float4 pos [[position]]; };

vertex BloomVOut root_bloom_vs(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    BloomVOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// 13-tap downsample: a centre quad, an outer ring and the corners, weighted so
// the result is a partition of unity. The point of the odd tap layout (rather
// than a plain 2x2 box) is that it kills the "fireflies pumping between frames"
// that a box filter gives on a chain this deep.
fragment float4 root_bloom_down_fs(BloomVOut in [[stage_in]],
                                   constant RootBloomU& U   [[buffer(0)]], texture2d<float>     src [[texture(0)]]) {
    constexpr sampler linSmp(mag_filter::linear, min_filter::linear,
                             address::clamp_to_edge);
    // The destination is half the source, so the destination pixel centre maps
    // to a source uv via the *source* texel size times two.
    const float2 uv = in.pos.xy * (U.srcTexel * 2.0);
    const float2 t = U.srcTexel;

    float3 a = src.sample(linSmp, uv + float2(-2, -2) * t).rgb;
    float3 b = src.sample(linSmp, uv + float2( 0, -2) * t).rgb;
    float3 c = src.sample(linSmp, uv + float2( 2, -2) * t).rgb;
    float3 d = src.sample(linSmp, uv + float2(-1, -1) * t).rgb;
    float3 e = src.sample(linSmp, uv + float2( 1, -1) * t).rgb;
    float3 f = src.sample(linSmp, uv + float2(-2,  0) * t).rgb;
    float3 g = src.sample(linSmp, uv).rgb;
    float3 h = src.sample(linSmp, uv + float2( 2,  0) * t).rgb;
    float3 i = src.sample(linSmp, uv + float2(-1,  1) * t).rgb;
    float3 j = src.sample(linSmp, uv + float2( 1,  1) * t).rgb;
    float3 k = src.sample(linSmp, uv + float2(-2,  2) * t).rgb;
    float3 l = src.sample(linSmp, uv + float2( 0,  2) * t).rgb;
    float3 m = src.sample(linSmp, uv + float2( 2,  2) * t).rgb;

    float3 o = (d + e + i + j) * 0.125
             + (a + b + g + f) * 0.03125
             + (b + c + h + g) * 0.03125
             + (f + g + l + k) * 0.03125
             + (g + h + m + l) * 0.03125;

    // Soft knee prefilter, level 0 only. A hard threshold makes the bloom
    // switch on along a contour as the exposure drifts, which on a slow camera
    // orbit is very visible; the knee ramps it in over half a stop instead.
    if (U.prefilter == 1) {
        const float lum = dot(o, float3(0.2126, 0.7152, 0.0722));
        const float knee = max(U.threshold * 0.5, 1e-4);
        const float soft = clamp(lum - U.threshold + knee, 0.0, 2.0 * knee);
        const float w = max(lum - U.threshold, soft * soft / (4.0 * knee + 1e-6));
        o *= w / max(lum, 1e-4);
    }
    return float4(o, 1.0);
}

// 9-tap tent upsample. Blended additively into the destination by the pipeline's
// blend state, which is what makes each level contribute its own scale of glow.
fragment float4 root_bloom_up_fs(BloomVOut in [[stage_in]],
                                 constant RootBloomU& U   [[buffer(0)]], texture2d<float>     src [[texture(0)]]) {
    constexpr sampler linSmp(mag_filter::linear, min_filter::linear,
                             address::clamp_to_edge);
    // Destination is twice the source here, so halve the step.
    const float2 uv = in.pos.xy * (U.srcTexel * 0.5);
    const float2 t = U.srcTexel * U.radius;

    float3 o = src.sample(linSmp, uv + float2(-1, -1) * t).rgb * 1.0
             + src.sample(linSmp, uv + float2( 0, -1) * t).rgb * 2.0
             + src.sample(linSmp, uv + float2( 1, -1) * t).rgb * 1.0
             + src.sample(linSmp, uv + float2(-1,  0) * t).rgb * 2.0
             + src.sample(linSmp, uv).rgb                      * 4.0
             + src.sample(linSmp, uv + float2( 1,  0) * t).rgb * 2.0
             + src.sample(linSmp, uv + float2(-1,  1) * t).rgb * 1.0
             + src.sample(linSmp, uv + float2( 0,  1) * t).rgb * 2.0
             + src.sample(linSmp, uv + float2( 1,  1) * t).rgb * 1.0;
    return float4(o * (1.0 / 16.0), 1.0);
}
