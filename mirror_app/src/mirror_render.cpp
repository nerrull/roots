#include "mirror_render.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mirror {

namespace {
// Scalar (fp32) array helper — keeps the arithmetic below unambiguous.
inline mx::array S(float v) { return mx::array(v); }

// --- fused ripple-features kernel --------------------------------------------
//
// The op-graph version below (multi_ripple_features_ops) is ~40 FLOPs per pixel
// expressed as ~75 elementwise MLX ops, each of which round-trips the entire
// 518k-element array through memory. Measured at 960x540: **9.9 ms**, i.e. 38%
// of the whole frame, for arithmetic that should be nearly free.
//
// One kernel per pixel reads 2 floats and writes 8 halves; everything between
// stays in registers. Measured **0.60 ms — 16.6x faster** — which is about what
// the ~12 MB/frame of traffic costs, so this is now memory-bound and there is
// nothing further to win here.
//
// (mx::compile was tried first, as the much smaller change: it manages only
// 1.33x, because concatenate/split break its fusion groups.)
//
// scal layout: [k, decay, core_r2, warp, x_offset, y_offset, z, z_cos]
const char* kRippleSrc = R"MSL(
    const uint i = thread_position_in_grid.x;
    if (i >= nrows[0]) return;

    const float x = coords[2 * i + 0];
    const float y = coords[2 * i + 1];

    const float k       = scal[0];
    const float decay   = scal[1];
    const float core_r2 = scal[2];
    const float warp    = scal[3];
    const float xoff    = scal[4];
    const float yoff    = scal[5];
    const float zval    = scal[6];
    const float zcos    = scal[7];

    float sin_acc = 0.0f, cos_acc = 0.0f, gx = 0.0f, gy = 0.0f;

    #pragma unroll
    for (uint s = 0; s < NSRC; ++s) {
        const float cx  = src[4 * s + 0];
        const float cy  = src[4 * s + 1];
        const float ph  = src[4 * s + 2];
        const float amp = src[4 * s + 3];

        const float dx = x - cx;
        const float dy = y - cy;
        // The +1e-6 inside the sqrt matches the op-graph version exactly and
        // keeps ri >= 1e-3, so both the divide below and the core term are safe.
        const float ri = sqrt(dx * dx + dy * dy + 1e-6f);
        const float r2 = ri * ri;

        // With core_r2 == 0 this factor is exactly 1, so one kernel serves both
        // the rolloff-on and rolloff-off settings without a branch.
        const float a = amp * exp(-decay * ri) * (r2 / (r2 + core_r2));

        const float ang = k * ri - ph;
        const float c = cos(ang);
        const float sn = sin(ang);
        sin_acc += a * sn;
        cos_acc += a * c;

        // Radial wave slope -> gradient, for the refraction warp. Computed
        // unconditionally: with warp == 0 it costs a few registers and the
        // result is multiplied out anyway.
        const float slope = a * c / ri;
        gx += slope * dx;
        gy += slope * dy;
    }

    const float xw = x - xoff + warp * gx;
    const float yw = y - yoff + warp * gy;

    // (N, 8): x, y, z, bias, sin_field, cos_field, z_cos, spare
    out[8 * i + 0] = (half)xw;
    out[8 * i + 1] = (half)yw;
    out[8 * i + 2] = (half)zval;
    out[8 * i + 3] = (half)1.0f;
    out[8 * i + 4] = (half)sin_acc;
    out[8 * i + 5] = (half)cos_acc;
    out[8 * i + 6] = (half)zcos;
    out[8 * i + 7] = (half)0.0f;
)MSL";

const mx::fast::CustomKernelFunction& ripple_kernel() {
    static mx::fast::CustomKernelFunction fn = mx::fast::metal_kernel(
        "ripple_features", {"coords", "src", "scal", "nrows"}, {"out"},
        kRippleSrc, /*header=*/"", /*ensure_row_contiguous=*/true);
    return fn;
}

constexpr int kRippleThreads = 256;

}  // namespace

mx::array make_coord_grid(int h, int w, float x0, float x1, float y0, float y1) {
    auto ys = mx::linspace(y0, y1, h);           // fp32
    auto xs = mx::linspace(x0, x1, w);
    auto g = mx::meshgrid({xs, ys}, /*sparse=*/false, /*indexing=*/"xy");  // (h, w) each
    auto gx = mx::reshape(g[0], {-1});
    auto gy = mx::reshape(g[1], {-1});
    auto coords = mx::stack({gx, gy}, -1);       // (H*W, 2)
    return mx::astype(coords, mx::float16);
}

mx::array multi_ripple_features(const mx::array& coords_in,
                                const std::vector<RippleSource>& sources,
                                float ring_freq, float decay, float z, float z_cos,
                                float warp, float core_radius,
                                float x_offset, float y_offset) {
    const int n = coords_in.shape(0);
    const int nsrc = static_cast<int>(sources.size());

    auto coords = mx::contiguous(mx::astype(coords_in, mx::float32));

    // NSRC=0 is a real case, not an edge case -- it is the *default* now that
    // ripples are off (see PondParams). The kernel handles it correctly: the
    // unrolled loop simply does not execute and the sin/cos accumulators stay
    // zero. It still needs a bindable `src` buffer though, so an empty source
    // list gets a one-row dummy the kernel never reads.
    //
    // Falling back to the op-graph path here instead would have quietly made
    // the slow path the common one: measured 20.4 ms vs 16.3 ms per frame.
    std::vector<float> src_flat;
    src_flat.reserve(static_cast<size_t>(nsrc) * 4);
    for (const auto& s : sources) src_flat.insert(src_flat.end(), s.begin(), s.end());
    if (src_flat.empty()) src_flat.assign(4, 0.f);
    auto src = mx::array(src_flat.data(),
                         {nsrc > 0 ? nsrc : 1, 4}, mx::float32);

    const float k = ring_freq * static_cast<float>(M_PI);
    const std::vector<float> sc = {k,        decay,    core_radius * core_radius,
                                   warp,     x_offset, y_offset,
                                   z,        z_cos};
    auto scal = mx::array(sc.data(), {8}, mx::float32);
    auto nrows = mx::array({n}, mx::uint32);

    const int grid = ((n + kRippleThreads - 1) / kRippleThreads) * kRippleThreads;
    auto outs = ripple_kernel()(
        {coords, src, scal, nrows}, {mx::Shape{n, ENRICHED_DIM}}, {mx::float16},
        std::make_tuple(grid, 1, 1), std::make_tuple(kRippleThreads, 1, 1),
        std::vector<std::pair<std::string, mx::fast::TemplateArg>>{{"NSRC", nsrc}},
        std::nullopt, /*verbose=*/false, {});
    return outs[0];
}

mx::array multi_ripple_features_ops(const mx::array& coords_in,
                                    const std::vector<RippleSource>& sources,
                                    float ring_freq, float decay, float z,
                                    float z_cos, float warp, float core_radius,
                                    float x_offset, float y_offset) {
    auto coords = mx::astype(coords_in, mx::float32);
    auto xy = mx::split(coords, 2, /*axis=*/1);  // x, y : (N, 1) each
    auto x = xy[0];
    auto y = xy[1];
    const int n = coords.shape(0);
    const float k = ring_freq * static_cast<float>(M_PI);

    auto sin_acc = mx::zeros({n, 1}, mx::float32);
    auto cos_acc = mx::zeros({n, 1}, mx::float32);
    auto gx = mx::zeros({n, 1}, mx::float32);
    auto gy = mx::zeros({n, 1}, mx::float32);
    for (const auto& s : sources) {
        const float cx = s[0], cy = s[1], phase = s[2], amp = s[3];
        auto dx = mx::subtract(x, S(cx));
        auto dy = mx::subtract(y, S(cy));
        auto ri = mx::sqrt(mx::add(mx::add(mx::multiply(dx, dx), mx::multiply(dy, dy)),
                                   S(1e-6f)));
        auto a = mx::multiply(S(amp), mx::exp(mx::multiply(S(-decay), ri)));
        if (core_radius != 0.f) {                     // fade the singular high-freq core
            auto r2 = mx::multiply(ri, ri);
            a = mx::multiply(a, mx::divide(r2, mx::add(r2, S(core_radius * core_radius))));
        }
        auto ang = mx::subtract(mx::multiply(S(k), ri), S(phase));
        auto c = mx::cos(ang);
        sin_acc = mx::add(sin_acc, mx::multiply(a, mx::sin(ang)));
        cos_acc = mx::add(cos_acc, mx::multiply(a, c));
        if (warp != 0.f) {                            // radial wave slope → gradient
            auto slope = mx::divide(mx::multiply(a, c), ri);
            gx = mx::add(gx, mx::multiply(slope, dx));
            gy = mx::add(gy, mx::multiply(slope, dy));
        }
    }
    // Color coords: subtract xy_offset (color travel) and add warp*gradient (refraction).
    auto xw = mx::subtract(x, S(x_offset));
    auto yw = mx::subtract(y, S(y_offset));
    if (warp != 0.f) {
        xw = mx::add(xw, mx::multiply(S(warp), gx));
        yw = mx::add(yw, mx::multiply(S(warp), gy));
    }
    auto zc = mx::full({n, 1}, z, mx::float32);
    auto zc2 = mx::full({n, 1}, z_cos, mx::float32);
    auto bias = mx::ones({n, 1}, mx::float32);
    auto zero = mx::zeros({n, 1}, mx::float32);
    auto feats = mx::concatenate({xw, yw, zc, bias, sin_acc, cos_acc, zc2, zero}, /*axis=*/1);
    return mx::astype(feats, mx::float16);
}

mx::array render_pond_lowres(const mx::array& weights, const MLPConfig& cfg,
                             int lh, int lw, const std::vector<RippleSource>& sources,
                             float ring_freq, float decay, float z, float z_cos,
                             float warp, float core_radius, float asp) {
    if (cfg.out_dim != 3) throw std::runtime_error("render_pond_lowres expects out_dim == 3");
    auto coords = make_coord_grid(lh, lw, -asp, asp, -1.f, 1.f);
    auto feats = multi_ripple_features(coords, sources, ring_freq, decay, z, z_cos,
                                       warp, core_radius);
    auto out = fused_mlp_forward(feats, weights, cfg);   // (lh*lw, 3) fp16
    auto img = mx::reshape(out, {lh, lw, 3});
    auto alpha = mx::ones({lh, lw, 1}, mx::float16);
    return mx::concatenate({img, alpha}, /*axis=*/2);    // (lh, lw, 4) fp16
}

}  // namespace mirror
