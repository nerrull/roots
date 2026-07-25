// The fused ripple-features Metal kernel vs. the op-graph reference.
//
// multi_ripple_features() is a hand-written kernel because the op-graph version
// cost 9.9 ms/frame at 960x540 (38% of the frame). multi_ripple_features_ops()
// is that original formulation, kept precisely so the fast path has something
// readable to be checked against -- this test is what keeps them honest.
//
// Needs a Metal device but no fixtures, so it runs anywhere the app builds.
// Exit 0 on pass.

#include "mirror_render.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;

namespace {

int g_fail = 0;

// Both paths emit fp16, where one ulp near 1.0 is ~1e-3. Anything at that
// level is rounding; anything above it is a difference in the arithmetic.
constexpr float kTol = 2e-3f;

struct Case {
    const char* name;
    float ring_freq, decay, z, z_cos, warp, core_radius, xoff, yoff;
    int   nsrc;
};

std::vector<mirror::RippleSource> make_sources(int n) {
    std::vector<mirror::RippleSource> s;
    for (int i = 0; i < n; ++i) {
        const float f = static_cast<float>(i);
        s.push_back({-0.7f + 0.3f * f, 0.5f - 0.21f * f, 0.4f + 0.17f * f,
                     0.35f + 0.12f * f});
    }
    return s;
}

void run(const Case& c) {
    const int lh = 96, lw = 160;
    const float asp = static_cast<float>(lw) / static_cast<float>(lh);
    auto coords = mirror::make_coord_grid(lh, lw, -asp, asp, -1.f, 1.f);
    const auto srcs = make_sources(c.nsrc);

    auto ref = mirror::multi_ripple_features_ops(coords, srcs, c.ring_freq, c.decay,
                                                 c.z, c.z_cos, c.warp,
                                                 c.core_radius, c.xoff, c.yoff);
    auto got = mirror::multi_ripple_features(coords, srcs, c.ring_freq, c.decay,
                                             c.z, c.z_cos, c.warp, c.core_radius,
                                             c.xoff, c.yoff);

    if (ref.shape() != got.shape()) {
        std::printf("  [FAIL] %-22s shape mismatch\n", c.name);
        ++g_fail;
        return;
    }
    auto d = mx::max(mx::abs(mx::subtract(mx::astype(ref, mx::float32),
                                          mx::astype(got, mx::float32))));
    mx::eval(d);
    const float md = d.item<float>();
    const bool ok = std::isfinite(md) && md <= kTol;
    std::printf("  [%s] %-22s max|diff| = %.6f\n", ok ? " ok " : "FAIL", c.name, md);
    if (!ok) ++g_fail;
}

}  // namespace

int main() {
    // Spread over the settings that change which arithmetic paths are live:
    // warp on/off (the gradient terms), core rolloff on/off (the damping term,
    // which must be exactly 1 when off), colour offsets, and source counts --
    // NSRC is a kernel template parameter, so each count is a distinct kernel.
    const Case cases[] = {
        {"defaults",            3.0f, 1.8f,  0.0f,  0.0f, 0.00f, 0.00f,  0.0f,  0.0f, 5},
        {"core rolloff",        3.0f, 1.8f,  0.0f,  0.0f, 0.00f, 0.12f,  0.0f,  0.0f, 5},
        {"warp",                3.0f, 1.8f,  0.0f,  0.0f, 0.40f, 0.00f,  0.0f,  0.0f, 5},
        {"warp + core",         3.0f, 1.8f,  0.3f, -0.2f, 0.40f, 0.12f,  0.0f,  0.0f, 5},
        {"colour travel",       3.0f, 1.8f,  0.3f, -0.2f, 0.00f, 0.12f,  0.31f, -0.17f, 5},
        {"z latents",           3.0f, 1.8f, -0.9f,  0.7f, 0.00f, 0.12f,  0.0f,  0.0f, 5},
        {"high freq, low decay", 9.0f, 0.4f, 0.0f,  0.0f, 0.15f, 0.05f,  0.0f,  0.0f, 5},
        {"one source",          3.0f, 1.8f,  0.0f,  0.0f, 0.40f, 0.12f,  0.0f,  0.0f, 1},
        {"two sources",         3.0f, 1.8f,  0.0f,  0.0f, 0.40f, 0.12f,  0.0f,  0.0f, 2},
        {"twelve sources",      3.0f, 1.8f,  0.0f,  0.0f, 0.40f, 0.12f,  0.0f,  0.0f, 12},
    };
    std::printf("ripple_parity: fused kernel vs op-graph reference\n");
    for (const auto& c : cases) run(c);

    // No sources at all: documented to fall back to the op path, and must not
    // produce NaNs or a wrong shape.
    {
        auto coords = mirror::make_coord_grid(96, 160, -1.f, 1.f, -1.f, 1.f);
        auto f = mirror::multi_ripple_features(coords, {}, 3.f, 1.8f, 0.f, 0.f);
        auto s = mx::sum(mx::astype(f, mx::float32));
        mx::eval(s);
        const bool ok = f.shape(0) == 96 * 160 && f.shape(1) == mirror::ENRICHED_DIM &&
                        std::isfinite(s.item<float>());
        std::printf("  [%s] %-22s\n", ok ? " ok " : "FAIL", "no sources");
        if (!ok) ++g_fail;
    }

    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
