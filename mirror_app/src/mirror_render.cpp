#include "mirror_render.h"

#include <cmath>
#include <stdexcept>

namespace mirror {

namespace {
// Scalar (fp32) array helper — keeps the arithmetic below unambiguous.
inline mx::array S(float v) { return mx::array(v); }
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
                                float ring_freq, float decay, float z, float z_cos) {
    auto coords = mx::astype(coords_in, mx::float32);
    auto xy = mx::split(coords, 2, /*axis=*/1);  // x, y : (N, 1) each
    auto x = xy[0];
    auto y = xy[1];
    const int n = coords.shape(0);
    const float k = ring_freq * static_cast<float>(M_PI);

    auto sin_acc = mx::zeros({n, 1}, mx::float32);
    auto cos_acc = mx::zeros({n, 1}, mx::float32);
    for (const auto& s : sources) {
        const float cx = s[0], cy = s[1], phase = s[2], amp = s[3];
        auto dx = mx::subtract(x, S(cx));
        auto dy = mx::subtract(y, S(cy));
        auto ri = mx::sqrt(mx::add(mx::add(mx::multiply(dx, dx), mx::multiply(dy, dy)),
                                   S(1e-6f)));
        auto a = mx::multiply(S(amp), mx::exp(mx::multiply(S(-decay), ri)));
        auto ang = mx::subtract(mx::multiply(S(k), ri), S(phase));
        sin_acc = mx::add(sin_acc, mx::multiply(a, mx::sin(ang)));
        cos_acc = mx::add(cos_acc, mx::multiply(a, mx::cos(ang)));
    }
    auto zc = mx::full({n, 1}, z, mx::float32);
    auto zc2 = mx::full({n, 1}, z_cos, mx::float32);
    auto bias = mx::ones({n, 1}, mx::float32);
    auto zero = mx::zeros({n, 1}, mx::float32);
    auto feats = mx::concatenate({x, y, zc, bias, sin_acc, cos_acc, zc2, zero}, /*axis=*/1);
    return mx::astype(feats, mx::float16);
}

mx::array render_pond_lowres(const mx::array& weights, const MLPConfig& cfg,
                             int lh, int lw, const std::vector<RippleSource>& sources,
                             float ring_freq, float decay, float z, float z_cos) {
    if (cfg.out_dim != 3) throw std::runtime_error("render_pond_lowres expects out_dim == 3");
    auto coords = make_coord_grid(lh, lw);
    auto feats = multi_ripple_features(coords, sources, ring_freq, decay, z, z_cos);
    auto out = fused_mlp_forward(feats, weights, cfg);   // (lh*lw, 3) fp16
    auto img = mx::reshape(out, {lh, lw, 3});
    auto alpha = mx::ones({lh, lw, 1}, mx::float16);
    return mx::concatenate({img, alpha}, /*axis=*/2);    // (lh, lw, 4) fp16
}

}  // namespace mirror
