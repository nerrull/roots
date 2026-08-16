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

// --- text overlay ------------------------------------------------------------
// A distance-field title composited over the scene by inverting it. See
// text_overlay.h for why the text is here rather than in the network.
//
// Two things make it belong to the image instead of sitting on top of it: the
// sample coordinate is refracted by the same ripple-gradient warp the feature
// kernel applies to its colour coords (mirror_render.cpp, kRippleSrc), and the
// composite is an inversion, so the text takes its colour from whatever it is
// over rather than from a constant that would fight a generated palette.
//
// Must match TextUniforms in text_overlay.h.
struct TextU {
    float4 place;    // centre.xy, half-extent.xy, in coord space
    float4 tune;     // aspect, strength, softness, dilate (encoded units)
    float4 tune2;    // warp, ring_freq, decay, core_radius^2
    float4 cnt;      // source count, enabled, time, reveal
    float4 diss;     // turbulence, turb scale, turb speed, unused
    float4 src[16];  // cx, cy, phase, amp
    float4 wid[16];  // drop-packet width in .x (0 = a standing field)
};

// How much a drop's ring train lengthens per unit travelled. Must match
// kDropSpread in mirror_render.h -- the text bends by the gradient of the same
// field the feature kernel builds, and two spread rates would bend it by the
// gradient of a field that is not on screen.
constant float kDropSpread = 0.5;

// --- turbulence for the dissolve ---------------------------------------------
// Value noise rather than anything fancier: it is sampled once per fragment and
// only has to be lumpy at a believable scale, which three octaves of it are.
static inline float thash(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

static inline float tnoise(float2 p) {
    const float2 i = floor(p);
    float2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    const float a = thash(i);
    const float b = thash(i + float2(1.0, 0.0));
    const float c = thash(i + float2(0.0, 1.0));
    const float d = thash(i + float2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

static inline float tfbm(float2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 3; ++i) {
        s += a * tnoise(p);
        p *= 2.03;
        a *= 0.5;
    }
    s *= 1.14;   // three halving octaves reach 0.875; renormalise to ~1

    // Summing octaves pulls the distribution hard towards its mean -- most
    // values land near 0.5 and only the rare extreme reaches either end. Used
    // raw as a threshold field that makes the whole word cross at nearly the
    // same moment, which reads as a plain fade with a bit of noise on it rather
    // than as patches coming apart. Stretched about the mean, the thresholds
    // actually span the reveal.
    return saturate((s - 0.5) * 2.6 + 0.5);
}

fragment float4 present_fs(VOut in [[stage_in]],
                           texture2d<float> tex [[texture(0)]],
                           texture2d<float> sdf [[texture(1)]],
                           constant TextU& t [[buffer(0)]]) {
    constexpr sampler smp(mag_filter::linear, min_filter::linear,
                          address::clamp_to_edge);
    float4 c = tex.sample(smp, in.uv);
    if (t.cnt.y == 0.0) return c;

    // Coord space: x over (-aspect, aspect), y over (-1, 1). uv.y = 0 is the
    // texture's first row, which present_vs puts at the top of the screen --
    // and the coord grid's y = -1 is its first row too, so this is a straight
    // remap with no flip.
    const float aspect = t.tune.x;
    float2 p = float2((in.uv.x * 2.0 - 1.0) * aspect, in.uv.y * 2.0 - 1.0);

    // Refraction: the radial wave slope of each ripple, accumulated as a
    // gradient exactly as the feature kernel does it. Same sources, same
    // constants, same 1e-6 inside the sqrt -- the text has to bend by the same
    // amount the colour under it does or the two read as separate effects.
    const float warp = t.tune2.x;
    const int nsrc = int(t.cnt.x);
    if (warp != 0.0 && nsrc > 0) {
        const float k = t.tune2.y, decay = t.tune2.z, core_r2 = t.tune2.w;
        float gx = 0.0, gy = 0.0;
        for (int s = 0; s < nsrc; ++s) {
            const float4 S = t.src[s];
            const float dx = p.x - S.x;
            const float dy = p.y - S.y;
            const float ri = sqrt(dx * dx + dy * dy + 1e-6);
            const float r2 = ri * ri;
            float a = S.w * exp(-decay * ri) * (r2 / (r2 + core_r2));
            const float pkw = t.wid[s].x;
            if (pkw > 0.0) {
                const float rf = S.z / k;                  // wavefront radius
                const float pw = pkw * (1.0 + kDropSpread * rf);
                const float u = (ri - rf) / pw;
                a *= exp(-0.5 * u * u);
            }
            const float slope = a * cos(k * ri - S.z) / ri;
            gx += slope * dx;
            gy += slope * dy;
        }
        p += warp * float2(gx, gy);
    }

    // Into the field's own [0,1] box.
    const float2 luv = (p - t.place.xy) / t.place.zw * 0.5 + 0.5;

    // 0.5 is the contour; the edge width is one pixel's footprint in field
    // units, which is what keeps this crisp at any scale and at any warp -- the
    // derivative already carries however much the refraction stretched the
    // sample here.
    //
    // Sampled clamped and masked afterwards, rather than returning early when
    // the box is missed: fwidth needs the neighbouring lanes in the quad to
    // have run the same code, and a fragment on the box's boundary has
    // neighbours on both sides of it.
    const float d = sdf.sample(smp, clamp(luv, 0.0, 1.0)).r - 0.5 + t.tune.w;

    // The raw footprint and the softened edge width are deliberately separate.
    // The footprint is a measure of how much field one pixel covers, and it is
    // what says whether there is an edge here at all; softness is a look. Using
    // the softened width for both made turning softness up fade the text away.
    const float foot = max(fwidth(d), 1e-5);
    // Capped at most of the encoded half-band: past that the smoothstep runs off
    // the end of the field, where the distance saturates flat, and the ramp
    // stops dead at a fixed offset from the outline -- which reads as a
    // hard-edged halo tracing the text rather than as a soft edge.
    const float w = min(foot * max(t.tune.z, 1e-3), 0.42);
    float cov = smoothstep(-w, w, d) * t.tune.y;

    // Drop fragments whose footprint is too large to resolve anything. A strong
    // warp folds the sample coordinate, and at a fold the derivative explodes
    // while d sits near the contour -- smoothstep then returns ~0.5 and leaves a
    // trail of half-lit specks off the ends of the words. A footprint spanning
    // most of the encoded band is not a soft edge, it is no edge at all.
    cov *= 1.0 - smoothstep(0.15, 0.40, foot);

    // --- turbulent emerge / dissolve ----------------------------------------
    // Not an erosion of the distance: that is bounded by the encoded band, so
    // thick stems would survive untouched and then pop. Instead every pixel gets
    // a threshold from the noise and appears when `reveal` passes it, so the
    // word comes apart into drifting patches that thin out from their edges.
    //
    // The noise is sampled at the *warped* p, so with ripples running the
    // dissolve flows with the water for free -- same coordinate the refraction
    // already bent, no second field to keep in step.
    const float reveal = t.cnt.w;
    if (reveal < 1.0) {
        const float2 drift = float2(0.13, 0.71) * t.cnt.z * t.diss.z;
        const float n = tfbm(p * t.diss.y + drift);
        // Bias the interior earlier than the edges: without this the patches
        // are uniform speckle, and with it the strokes visibly thin out and
        // break, which is what erosion looks like.
        const float inside = saturate(d + 0.5);
        float key = mix(0.5, n, t.diss.x) - 0.25 * t.diss.x * inside;
        key = saturate(key);
        // Remapped so reveal 0 clears every threshold and 1 clears none, band
        // included.
        const float r = reveal * 1.24 - 0.12;
        cov *= smoothstep(key - 0.12, key + 0.12, r);
    }

    if (any(luv < 0.0) || any(luv > 1.0)) cov = 0.0;

    // Clamped before inverting: the root scene's target is RGBA16F and can carry
    // values above 1, where 1 - c would go negative and the text would come out
    // as a black hole rather than an inversion.
    return float4(mix(c.rgb, 1.0 - clamp(c.rgb, 0.0, 1.0), cov), c.a);
}
