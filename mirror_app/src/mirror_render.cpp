#include "mirror_render.h"

#include <algorithm>
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
// scal layout: [k, decay, core_r2, warp, x_offset, y_offset, z, z_cos,
//               z_out, z_cos_out, cx, cy, inv_hx, inv_hy, fade_start, region_on,
//               inv_fade_width, use_field, fw, fh, ax]
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

    // Region weight: 0 at the trained region's edge, 1 past the fade, smoothstep
    // in between. Computed from the RAW coords, never the offset ones -- the
    // region is where the subject lands on screen, and with an input shift in
    // play (the head-stabilised fit mode) those two are deliberately different.
    float rw = 0.0f;
    if (scal[15] != 0.0f) {
        float d;   // distance outside the region, in coord units
        if (scal[17] != 0.0f) {
            // Bilinear sample of the distance field. It spans exactly the same
            // extent as the render grid, so this is a straight remap -- and the
            // field is smooth by construction, which is why sampling one built
            // at fit-grid size and drawing it at display size holds up.
            const int fw = (int)scal[18];
            const int fh = (int)scal[19];
            const float gx = clamp((x / scal[20] + 1.0f) * 0.5f * (float)(fw - 1),
                                   0.0f, (float)(fw - 1));
            const float gy = clamp((y + 1.0f) * 0.5f * (float)(fh - 1),
                                   0.0f, (float)(fh - 1));
            const int x0 = (int)gx, y0 = (int)gy;
            const int x1 = min(x0 + 1, fw - 1), y1 = min(y0 + 1, fh - 1);
            const float tx = gx - (float)x0, ty = gy - (float)y0;
            const float a = field[y0 * fw + x0], b = field[y0 * fw + x1];
            const float c = field[y1 * fw + x0], e = field[y1 * fw + x1];
            d = mix(mix(a, b, tx), mix(c, e, tx), ty);
        } else {
            const float ux = fabs(x - scal[10]) * scal[12];
            const float uy = fabs(y - scal[11]) * scal[13];
            // Back to coord units, so the fade is specified the same way in
            // both modes: (u - 1) is a fraction of the half-extent it exceeded.
            d = (max(ux, uy) - 1.0f) / max(scal[12], scal[13]);
        }
        const float t = clamp((d - scal[14]) * scal[16], 0.0f, 1.0f);
        rw = t * t * (3.0f - 2.0f * t);
    }
    // Crossfade the two latents. With no region this is exactly scal[6]/scal[7],
    // so the un-regioned path is unchanged to the bit.
    const float zval = scal[6] + (scal[8] - scal[6]) * rw;
    const float zcos = scal[7] + (scal[9] - scal[7]) * rw;

    float sin_acc = 0.0f, cos_acc = 0.0f, gx = 0.0f, gy = 0.0f;

    #pragma unroll
    for (uint s = 0; s < NSRC; ++s) {
        const float cx  = src[SRCDIM * s + 0];
        const float cy  = src[SRCDIM * s + 1];
        const float ph  = src[SRCDIM * s + 2];
        const float amp = src[SRCDIM * s + 3];
        const float pkw = src[SRCDIM * s + 4];

        const float dx = x - cx;
        const float dy = y - cy;
        // The +1e-6 inside the sqrt matches the op-graph version exactly and
        // keeps ri >= 1e-3, so both the divide below and the core term are safe.
        const float ri = sqrt(dx * dx + dy * dy + 1e-6f);
        const float r2 = ri * ri;

        // With core_r2 == 0 this factor is exactly 1, so one kernel serves both
        // the rolloff-on and rolloff-off settings without a branch.
        float a = amp * exp(-decay * ri) * (r2 / (r2 + core_r2));

        // Drop packet: confine this source's rings to a train riding its own
        // wavefront, at phase/k -- see RippleSource in mirror_render.h. At width
        // 0 the factor is skipped entirely and the source is the standing field
        // it always was.
        if (pkw > 0.0f) {
            const float rf = ph / k;
            const float pw = pkw * (1.0f + SPREAD * rf);
            const float u = (ri - rf) / pw;
            a *= exp(-0.5f * u * u);
        }

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

    // Slot 7 stays zero on purpose: all eight are MLP inputs, and putting the
    // region weight there would feed the network a signal that moves with the
    // subject. It goes out as its own array instead.
    regw[i] = (half)rw;
)MSL";

const mx::fast::CustomKernelFunction& ripple_kernel() {
    // The packet's spread rate reaches the kernel as a #define rather than as a
    // literal, so the kernel, the op graph and the present shader cannot fall
    // out of step over a number the parity test would then have to chase.
    static const std::string header =
        "#define SPREAD " + std::to_string(kDropSpread) + "f\n"
        "#define SRCDIM " + std::to_string(RIPPLE_SRC_DIM) + "\n";
    static mx::fast::CustomKernelFunction fn = mx::fast::metal_kernel(
        "ripple_features", {"coords", "src", "scal", "nrows", "field"},
        {"out", "regw"},
        kRippleSrc, header, /*ensure_row_contiguous=*/true);
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

float region_weight(const FitRegion& r, const float* field, float x, float y) {
    if (!r.on) return 0.f;
    float d;
    if (r.use_field && field && r.fw > 1 && r.fh > 1) {
        auto clampf = [](float v, float lo, float hi) {
            return v < lo ? lo : (v > hi ? hi : v);
        };
        const float gx = clampf((x / std::max(r.ax, 1e-6f) + 1.f) * 0.5f * (r.fw - 1),
                                0.f, float(r.fw - 1));
        const float gy = clampf((y + 1.f) * 0.5f * (r.fh - 1), 0.f, float(r.fh - 1));
        const int x0 = int(gx), y0 = int(gy);
        const int x1 = std::min(x0 + 1, r.fw - 1), y1 = std::min(y0 + 1, r.fh - 1);
        const float tx = gx - x0, ty = gy - y0;
        const float a = field[y0 * r.fw + x0], b = field[y0 * r.fw + x1];
        const float c = field[y1 * r.fw + x0], e = field[y1 * r.fw + x1];
        d = (a + (b - a) * tx) + ((c + (e - c) * tx) - (a + (b - a) * tx)) * ty;
    } else {
        const float ux = std::fabs(x - r.cx) / std::max(r.hx, 1e-6f);
        const float uy = std::fabs(y - r.cy) / std::max(r.hy, 1e-6f);
        d = (std::max(ux, uy) - 1.f) * std::min(r.hx, r.hy);
    }
    float t = (d - r.fade_start) / std::max(r.fade_width, 1e-4f);
    t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
    return t * t * (3.f - 2.f * t);
}

mx::array multi_ripple_features(const mx::array& coords,
                                const std::vector<RippleSource>& sources,
                                float ring_freq, float decay, float z, float z_cos,
                                float warp, float core_radius,
                                float x_offset, float y_offset) {
    return multi_ripple_features_region(coords, sources, ring_freq, decay, z, z_cos,
                                        warp, core_radius, x_offset, y_offset,
                                        FitRegion{}, z, z_cos, std::nullopt)[0];
}

std::vector<mx::array> multi_ripple_features_region(
    const mx::array& coords_in, const std::vector<RippleSource>& sources,
    float ring_freq, float decay, float z, float z_cos, float warp,
    float core_radius, float x_offset, float y_offset,
    const FitRegion& region, float z_out, float z_cos_out,
    const std::optional<mx::array>& field) {
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
    src_flat.reserve(static_cast<size_t>(nsrc) * RIPPLE_SRC_DIM);
    for (const auto& s : sources) src_flat.insert(src_flat.end(), s.begin(), s.end());
    if (src_flat.empty()) src_flat.assign(RIPPLE_SRC_DIM, 0.f);
    auto src = mx::array(src_flat.data(),
                         {nsrc > 0 ? nsrc : 1, RIPPLE_SRC_DIM}, mx::float32);

    // A field is only sampled when one was actually supplied: a region claiming
    // field mode with nothing bound would read a dummy buffer and fade the
    // whole frame at once.
    const bool has_field = region.use_field && field.has_value() &&
                           region.fw > 1 && region.fh > 1;
    // The kernel always needs something bound here, so an unused input gets a
    // one-element dummy -- the same arrangement `src` uses for zero ripples.
    auto field_arr = has_field ? mx::contiguous(mx::astype(*field, mx::float32))
                               : mx::zeros({1}, mx::float32);

    const float k = ring_freq * static_cast<float>(M_PI);
    const std::vector<float> sc = {
        k,        decay,    core_radius * core_radius,
        warp,     x_offset, y_offset,
        z,        z_cos,
        z_out,    z_cos_out,
        region.cx, region.cy,
        1.f / std::max(region.hx, 1e-6f), 1.f / std::max(region.hy, 1e-6f),
        region.fade_start,
        region.on ? 1.f : 0.f,
        1.f / std::max(region.fade_width, 1e-4f),
        has_field ? 1.f : 0.f,
        float(region.fw), float(region.fh), std::max(region.ax, 1e-6f)};
    auto scal = mx::array(sc.data(), {21}, mx::float32);
    auto nrows = mx::array({n}, mx::uint32);

    const int grid = ((n + kRippleThreads - 1) / kRippleThreads) * kRippleThreads;
    return ripple_kernel()(
        {coords, src, scal, nrows, field_arr},
        {mx::Shape{n, ENRICHED_DIM}, mx::Shape{n, 1}}, {mx::float16, mx::float16},
        std::make_tuple(grid, 1, 1), std::make_tuple(kRippleThreads, 1, 1),
        std::vector<std::pair<std::string, mx::fast::TemplateArg>>{{"NSRC", nsrc}},
        std::nullopt, /*verbose=*/false, {});
}

// Box mode only. The field path is checked against a CPU bilinear reference in
// the test rather than a second MLX formulation: expressing a gather-based
// bilinear sample in ops would be a third implementation to keep in step, and
// the CPU one is genuinely independent of both.
mx::array region_weight_ops(const mx::array& coords_in, const FitRegion& region) {
    auto coords = mx::astype(coords_in, mx::float32);
    const int n = coords.shape(0);
    if (!region.on) return mx::zeros({n, 1}, mx::float32);
    auto xy = mx::split(coords, 2, /*axis=*/1);
    auto ux = mx::divide(mx::abs(mx::subtract(xy[0], S(region.cx))),
                         S(std::max(region.hx, 1e-6f)));
    auto uy = mx::divide(mx::abs(mx::subtract(xy[1], S(region.cy))),
                         S(std::max(region.hy, 1e-6f)));
    auto d = mx::multiply(mx::subtract(mx::maximum(ux, uy), S(1.f)),
                          S(std::min(region.hx, region.hy)));
    auto t = mx::clip(mx::divide(mx::subtract(d, S(region.fade_start)),
                                 S(std::max(region.fade_width, 1e-4f))),
                      S(0.f), S(1.f));
    return mx::multiply(mx::multiply(t, t),
                        mx::subtract(S(3.f), mx::multiply(S(2.f), t)));
}

mx::array multi_ripple_features_ops(const mx::array& coords_in,
                                    const std::vector<RippleSource>& sources,
                                    float ring_freq, float decay, float z,
                                    float z_cos, float warp, float core_radius,
                                    float x_offset, float y_offset,
                                    const FitRegion& region,
                                    float z_out, float z_cos_out) {
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
        const float cx = s[0], cy = s[1], phase = s[2], amp = s[3], pkw = s[4];
        auto dx = mx::subtract(x, S(cx));
        auto dy = mx::subtract(y, S(cy));
        auto ri = mx::sqrt(mx::add(mx::add(mx::multiply(dx, dx), mx::multiply(dy, dy)),
                                   S(1e-6f)));
        auto a = mx::multiply(S(amp), mx::exp(mx::multiply(S(-decay), ri)));
        if (core_radius != 0.f) {                     // fade the singular high-freq core
            auto r2 = mx::multiply(ri, ri);
            a = mx::multiply(a, mx::divide(r2, mx::add(r2, S(core_radius * core_radius))));
        }
        if (pkw > 0.f) {                              // ride the wavefront
            const float rf = phase / k;
            const float pw = pkw * (1.f + kDropSpread * rf);
            auto u = mx::divide(mx::subtract(ri, S(rf)), S(pw));
            a = mx::multiply(a, mx::exp(mx::multiply(S(-0.5f), mx::multiply(u, u))));
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
    auto rw = region_weight_ops(coords_in, region);
    auto zc = mx::add(S(z), mx::multiply(S(z_out - z), rw));
    auto zc2 = mx::add(S(z_cos), mx::multiply(S(z_cos_out - z_cos), rw));
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
