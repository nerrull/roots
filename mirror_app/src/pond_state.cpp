#include "pond_state.h"

#include <array>
#include <cmath>
#include <random>
#include <stdexcept>

namespace mirror {

namespace {
inline mx::array S(float v) { return mx::array(v); }
inline mx::array clip01(const mx::array& a) { return mx::clip(a, S(0.f), S(1.f)); }

// Per-layer flat offsets + total (bounds has num_layers+1 entries).
std::vector<int> weight_bounds(const MLPConfig& cfg) {
    std::vector<int> b;
    int acc = 0;
    b.push_back(0);
    for (auto& kn : cfg.layer_dims()) { acc += kn.first * kn.second; b.push_back(acc); }
    return b;
}
}  // namespace

Pond::Pond(int seed)
    : cfg_(MLPConfig{ENRICHED_DIM, 64, 3, 6, Act::Tanh, Act::Sigmoid}),
      seed_(seed),
      wb_(make_weights(cfg_, seed)),
      wb_shaped_(mx::zeros({1})),
      w_(mx::zeros({1})) {}

mx::array Pond::make_weights(const MLPConfig& cfg, int seed, float scale) {
    // Same MLX RNG as helpers.make_weights → bit-identical base weights.
    mx::random::seed(static_cast<uint64_t>(seed));
    std::vector<mx::array> parts;
    for (auto& kn : cfg.layer_dims()) {
        const int k = kn.first, n = kn.second;
        const float std_ = scale / std::sqrt(static_cast<float>(k));
        auto w = mx::multiply(mx::random::normal({k * n}, mx::float32), S(std_));
        parts.push_back(mx::astype(w, mx::float16));
    }
    return mx::concatenate(parts);
}

std::vector<float> Pond::layer_scales(const PondParams& p) const {
    const int n_hidden = cfg_.num_layers - 1;
    std::vector<float> s;
    for (int i = 0; i < n_hidden; ++i) {
        float f = (n_hidden > 1) ? (static_cast<float>(i) / (n_hidden - 1) - 0.5f) : 0.0f;
        s.push_back(p.detail * std::exp(p.gain_tilt * f));
    }
    s.push_back(p.contrast);
    return s;
}

const mx::array& Pond::shaped_base(const PondParams& p) {
    if (shaped_key_ && *shaped_key_ == p.uniform_mix) return wb_shaped_;
    const float t = p.uniform_mix;
    if (t <= 0.0f) {
        wb_shaped_ = wb_;
    } else {
        const auto bounds = weight_bounds(cfg_);
        const float root3 = std::sqrt(3.0f);
        std::vector<mx::array> parts;
        for (size_t i = 0; i + 1 < bounds.size(); ++i) {
            auto z = mx::astype(mx::slice(wb_, {bounds[i]}, {bounds[i + 1]}), mx::float32);
            auto sigma = mx::sqrt(mx::mean(mx::multiply(z, z)));   // zero-mean layer std
            auto u = mx::multiply(S(0.5f),
                     mx::add(S(1.0f), mx::erf(mx::divide(z, mx::multiply(sigma, S(std::sqrt(2.0f)))))));
            auto uni = mx::multiply(mx::multiply(sigma, S(root3)),
                                    mx::subtract(mx::multiply(S(2.0f), u), S(1.0f)));
            auto mixed = mx::add(mx::multiply(S(1.0f - t), z), mx::multiply(S(t), uni));
            auto norm = mx::divide(sigma,
                        mx::sqrt(mx::add(mx::mean(mx::multiply(mixed, mixed)), S(1e-12f))));
            parts.push_back(mx::multiply(mixed, norm));
        }
        wb_shaped_ = mx::astype(mx::concatenate(parts), mx::float16);
    }
    shaped_key_ = p.uniform_mix;
    return wb_shaped_;
}

const mx::array& Pond::weights(const PondParams& p) {
    std::array<float, 5> key{p.detail, p.gain_tilt, p.uniform_mix, p.contrast,
                             static_cast<float>(seed_)};
    if (w_key_ && *w_key_ == key) return w_;
    const auto bounds = weight_bounds(cfg_);
    const auto scales = layer_scales(p);
    const auto& base = shaped_base(p);
    std::vector<mx::array> parts;
    for (size_t i = 0; i + 1 < bounds.size(); ++i)
        parts.push_back(mx::multiply(mx::slice(base, {bounds[i]}, {bounds[i + 1]}), S(scales[i])));
    w_ = mx::astype(mx::concatenate(parts), mx::float16);
    w_key_ = key;
    return w_;
}

void Pond::reseed() {
    seed_ += 1;
    wb_ = make_weights(cfg_, seed_);
    shaped_key_.reset();
    w_key_.reset();
}

std::vector<RippleSource> Pond::sources(float asp, double t, const PondParams& p) const {
    const float phase = 2.0f * (float)M_PI * p.speed * (float)t + p.ripple_offset;
    std::vector<RippleSource> src;
    std::mt19937 gen(static_cast<uint32_t>(seed_));
    auto U = [&](float lo, float hi) {
        return lo + (hi - lo) * std::generate_canonical<float, 24>(gen);
    };
    for (int i = 0; i < p.drops; ++i) {
        float cx = U(-asp, asp), cy = U(-1.f, 1.f);
        float rate = U(0.15f, 0.5f), ph = U(0.f, 2.f * (float)M_PI);
        float amp = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * rate * (float)t + ph);
        src.push_back({cx, cy, phase, amp});
    }
    if (p.orbit_on) {
        float ox = 0.6f * asp * std::cos(0.5f * (float)t);
        float oy = 0.6f * std::sin(0.5f * (float)t);
        src.push_back({ox, oy, phase, 1.0f});
    }
    return src;
}

mx::array Pond::apply_transition(const mx::array& img, const mx::array& coords,
                                 int lh, int lw, float asp, const PondParams& p) const {
    const int N = lh * lw;
    const float t = p.transition;
    const float te = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);   // smootherstep

    auto x = mx::reshape(mx::astype(mx::slice(coords, {0, 0}, {N, 1}), mx::float32), {lh, lw});
    auto y = mx::reshape(mx::astype(mx::slice(coords, {0, 1}, {N, 2}), mx::float32), {lh, lw});

    // placeholder relief: ellipsoid dome H = h*sqrt(1 - (x/ax)^2 - (y/ay)^2)
    auto mxn = mx::divide(x, S(p.mask_ax));
    auto myn = mx::divide(y, S(p.mask_ay));
    auto rm2 = mx::add(mx::multiply(mxn, mxn), mx::multiply(myn, myn));
    auto dome = mx::sqrt(mx::maximum(S(0.f), mx::subtract(S(1.f), rm2)));
    auto height = mx::multiply(S(p.relief_h), dome);
    auto rm = mx::sqrt(rm2);

    // normals from the height field (finite differences)
    const float dx = 2.0f * asp / (lw - 1);
    const float dy = 2.0f / (lh - 1);
    auto hx = mx::pad(height, {{0, 0}, {1, 1}});   // (lh, lw+2)
    auto hy = mx::pad(height, {{1, 1}, {0, 0}});   // (lh+2, lw)
    auto d_dx = mx::divide(mx::subtract(mx::slice(hx, {0, 2}, {lh, lw + 2}),
                                        mx::slice(hx, {0, 0}, {lh, lw})), S(2.0f * dx));
    auto d_dy = mx::divide(mx::subtract(mx::slice(hy, {2, 0}, {lh + 2, lw}),
                                        mx::slice(hy, {0, 0}, {lh, lw})), S(2.0f * dy));
    auto inv = mx::divide(S(1.f),
               mx::sqrt(mx::add(mx::add(mx::multiply(d_dx, d_dx), mx::multiply(d_dy, d_dy)), S(1.f))));
    auto nx = mx::multiply(mx::negative(d_dx), inv);
    auto ny = mx::multiply(mx::negative(d_dy), inv);
    auto nz = inv;

    // key light + Blinn specular (view straight-on, +z)
    const float cel = std::cos(p.light_elev), sel = std::sin(p.light_elev);
    const float lx = cel * std::cos(p.light_az), ly = cel * std::sin(p.light_az), lz = sel;
    const float hlen = std::sqrt(lx * lx + ly * ly + (lz + 1.0f) * (lz + 1.0f));
    const float hxl = lx / hlen, hyl = ly / hlen, hzl = (lz + 1.0f) / hlen;
    auto ndl = mx::maximum(S(0.f), mx::add(mx::add(mx::multiply(nx, S(lx)), mx::multiply(ny, S(ly))),
                                           mx::multiply(nz, S(lz))));
    auto ndh = mx::maximum(S(0.f), mx::add(mx::add(mx::multiply(nx, S(hxl)), mx::multiply(ny, S(hyl))),
                                           mx::multiply(nz, S(hzl))));
    auto spec = mx::expand_dims(mx::multiply(S(p.spec_amt), mx::power(ndh, S(p.shininess))), -1);
    auto shade = mx::expand_dims(mx::add(S(p.ambient), mx::multiply(S(p.diff_amt), ndl)), -1);

    auto lit = clip01(mx::add(mx::multiply(img, shade), spec));

    // emergence front: relief sweeps on from center outward
    const float front = 1.25f * te;
    auto reveal = mx::expand_dims(clip01(mx::divide(mx::subtract(S(front), rm), S(0.22f))), -1);

    auto base = mx::multiply(img, S(1.0f - p.bg_dim * te));
    auto scene = mx::add(base, mx::multiply(mx::subtract(lit, base), reveal));

    // closing vignette: periphery collapses to black
    auto ux = mx::divide(x, S(asp));
    auto rs = mx::sqrt(mx::add(mx::multiply(ux, ux), mx::multiply(y, y)));
    const float r_out = 1.7f - 0.86f * te;
    auto edge = mx::expand_dims(clip01(mx::divide(mx::subtract(S(r_out), rs), S(0.28f))), -1);
    return clip01(mx::multiply(scene, edge));
}

mx::array Pond::render(int lh, int lw, double t, const PondParams& p) {
    const int N = lh * lw;
    const float asp = static_cast<float>(lw) / static_cast<float>(lh);
    auto coords = make_coord_grid(lh, lw, -asp, asp, -1.f, 1.f);

    float offx = 0.f, offy = 0.f;
    if (p.color_travel) {
        offx = 0.6f * asp * std::cos(0.5f * (float)t);
        offy = 0.6f * std::sin(0.5f * (float)t);
    }
    const float zv = p.z_amp * std::sin(p.z);
    const float zc = p.z_amp * std::cos(p.z);
    auto feats = multi_ripple_features(coords, sources(asp, t, p), p.ring_freq, p.decay,
                                       zv, zc, p.warp, p.core_rolloff ? p.core_radius : 0.f,
                                       offx, offy);

    auto img = mx::reshape(fused_mlp_forward(feats, weights(p), cfg_), {lh, lw, 3});
    img = clip01(mx::astype(img, mx::float32));

    // greyscale (single channel) → RGB, flat or amplitude-driven per pixel.
    const bool doMix = p.amp_drives_color || p.color_mix < 1.0f;
    mx::array mixv = S(p.color_mix);
    if (p.amp_drives_color) {
        auto disp = mx::reshape(mx::abs(mx::astype(mx::slice(feats, {0, 4}, {N, 5}), mx::float32)),
                                {lh, lw, 1});
        mixv = mx::multiply(clip01(mx::multiply(disp, S(p.amp_gain))), S(p.color_mix));
    }
    if (doMix) {
        int gc = p.grey_channel;
        auto grey = mx::slice(img, {0, 0, gc}, {lh, lw, gc + 1});
        img = mx::add(grey, mx::multiply(mx::subtract(img, grey), mixv));
    }
    if (p.srgb_fix) {   // counter the sRGB framebuffer's encode-on-output
        auto xg = clip01(img);
        auto lo = mx::multiply(S(12.92f), xg);
        auto hi = mx::subtract(mx::multiply(S(1.055f), mx::power(xg, S(1.0f / 2.4f))), S(0.055f));
        img = mx::where(mx::less_equal(xg, S(0.0031308f)), lo, hi);
    }
    if (p.gamma != 1.0f) img = mx::power(img, S(p.gamma));
    if (p.transition > 0.0f) img = apply_transition(img, coords, lh, lw, asp, p);
    if (p.swap_rb) {
        auto ch = mx::split(img, 3, 2);
        img = mx::concatenate({ch[2], ch[1], ch[0]}, 2);
    }
    return img;   // (lh, lw, 3) fp32 in [0,1]
}

}  // namespace mirror
