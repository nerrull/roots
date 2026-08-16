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
#include "face_tracker.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;

namespace {

int g_fail = 0;

void check_eq(float got, float want, const char* what) {
    const bool ok = std::fabs(got - want) < 1e-5f;
    std::printf("  [%s] %-38s (%.4f)\n", ok ? " ok " : "FAIL", what, got);
    if (!ok) ++g_fail;
}

// Both paths emit fp16, where one ulp near 1.0 is ~1e-3. Anything at that
// level is rounding; anything above it is a difference in the arithmetic.
constexpr float kTol = 2e-3f;

struct Case {
    const char* name;
    float ring_freq, decay, z, z_cos, warp, core_radius, xoff, yoff;
    int   nsrc;
    // Drop packet width, applied to every source in the case. The envelope is
    // derived from each source's own phase, so a sign error or a mismatched
    // spread rate shows up here as the two paths putting the ring train at
    // different radii, not as a scale factor.
    float packet_w = 0.f;
};

std::vector<mirror::RippleSource> make_sources(int n, float packet_w = 0.f) {
    std::vector<mirror::RippleSource> s;
    for (int i = 0; i < n; ++i) {
        const float f = static_cast<float>(i);
        s.push_back({-0.7f + 0.3f * f, 0.5f - 0.21f * f, 0.4f + 0.17f * f,
                     0.35f + 0.12f * f, packet_w});
    }
    return s;
}

void run(const Case& c) {
    const int lh = 96, lw = 160;
    const float asp = static_cast<float>(lw) / static_cast<float>(lh);
    auto coords = mirror::make_coord_grid(lh, lw, -asp, asp, -1.f, 1.f);
    const auto srcs = make_sources(c.nsrc, c.packet_w);

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

// The region path: the kernel crossfades two latents across a soft box edge and
// emits the weight it used. Both outputs are checked, because the renderer uses
// the weight for its colour falloff and a weight that disagreed with the one the
// latent was faded by would put the two edges in different places.
void run_region(const char* name, const mirror::FitRegion& r,
                float z, float z_cos, float z_out, float z_cos_out, int nsrc) {
    const int lh = 96, lw = 160;
    const float asp = static_cast<float>(lw) / static_cast<float>(lh);
    auto coords = mirror::make_coord_grid(lh, lw, -asp, asp, -1.f, 1.f);
    const auto srcs = make_sources(nsrc);

    auto ref = mirror::multi_ripple_features_ops(coords, srcs, 3.f, 1.8f, z, z_cos,
                                                 0.2f, 0.1f, 0.f, 0.f, r,
                                                 z_out, z_cos_out);
    auto got = mirror::multi_ripple_features_region(coords, srcs, 3.f, 1.8f, z,
                                                    z_cos, 0.2f, 0.1f, 0.f, 0.f,
                                                    r, z_out, z_cos_out);
    auto wref = mirror::region_weight_ops(coords, r);

    auto df = mx::max(mx::abs(mx::subtract(mx::astype(ref, mx::float32),
                                           mx::astype(got[0], mx::float32))));
    auto dw = mx::max(mx::abs(mx::subtract(mx::astype(wref, mx::float32),
                                           mx::astype(got[1], mx::float32))));
    mx::eval(df); mx::eval(dw);
    const float mf = df.item<float>(), mw = dw.item<float>();
    const bool ok = std::isfinite(mf) && mf <= kTol && std::isfinite(mw) && mw <= kTol;
    std::printf("  [%s] %-22s feats %.6f  weight %.6f\n", ok ? " ok " : "FAIL",
                name, mf, mw);
    if (!ok) ++g_fail;
}

// The weight's shape, independent of either implementation: zero on the box,
// one well outside it, monotonic across the band. This is what the fit relies
// on -- a weight that were merely *small* over the trained pixels rather than
// exactly zero would drift the latent under the face.
void test_region_shape() {
    mirror::FitRegion r;
    r.on = true; r.cx = 0.2f; r.cy = -0.3f; r.hx = 0.4f; r.hy = 0.25f;
    r.fade_start = 0.f; r.fade_width = 0.2f;

    check_eq(mirror::region_weight(r, nullptr, r.cx, r.cy), 0.f,
             "centre is fully inside");
    check_eq(mirror::region_weight(r, nullptr, r.cx + r.hx, r.cy), 0.f,
             "the x edge is inside");
    check_eq(mirror::region_weight(r, nullptr, r.cx, r.cy + r.hy), 0.f,
             "the y edge is inside");
    // A box corner is the worst case: with a radial falloff it would sit
    // outside the region and the fit would be animated there.
    check_eq(mirror::region_weight(r, nullptr, r.cx + r.hx, r.cy + r.hy), 0.f,
             "the corner is inside");
    check_eq(mirror::region_weight(r, nullptr, r.cx + r.hx + 0.6f, r.cy), 1.f,
             "past the fade is fully outside");
    check_eq(mirror::region_weight(r, nullptr, r.cx + r.hx + 4.f, r.cy), 1.f,
             "and stays there");

    bool mono = true;
    float prev = -1.f;
    for (int i = 0; i <= 40; ++i) {
        const float x = r.cx + r.hx + 0.6f * i / 40.f;
        const float w = mirror::region_weight(r, nullptr, x, r.cy);
        if (w < prev - 1e-6f) mono = false;
        prev = w;
    }
    std::printf("  [%s] %s\n", mono ? " ok " : "FAIL",
                "the falloff is monotonic across the band");
    if (!mono) ++g_fail;

    // fade_start pushes the whole ramp outward: this is the control for a
    // gradient that pools against the crop's edge.
    mirror::FitRegion pushed = r;
    pushed.fade_start = 0.25f;
    check_eq(mirror::region_weight(pushed, nullptr, r.cx + r.hx + 0.1f, r.cy), 0.f,
             "fade_start keeps a clean margin");
    const float w_near = mirror::region_weight(r, nullptr, r.cx + r.hx + 0.1f, r.cy);
    std::printf("  [%s] %-38s (%.4f vs 0)\n", w_near > 0.f ? " ok " : "FAIL",
                "...where the un-pushed ramp is already on", w_near);
    if (!(w_near > 0.f)) ++g_fail;

    mirror::FitRegion off;
    check_eq(mirror::region_weight(off, nullptr, 0.f, 0.f), 0.f,
             "region off = no weight");
    check_eq(mirror::region_weight(off, nullptr, 9.f, 9.f), 0.f,
             "region off = no weight, anywhere");
}

// Field mode, where the shape comes from a distance map rather than a box.
//
// The kernel's bilinear sample is checked against a CPU one computed here --
// deliberately a different implementation rather than a second MLX formulation,
// since the whole point of a reference is that it not share the original's
// mistakes.
void test_region_field() {
    // The field grid must carry the render grid's aspect: the distance is in
    // pixels, and one pixel is only the same coord distance on both axes when
    // fw/fh == ax. (The app gets this for free -- the fit grid and the render
    // grid are the same framebuffer over the same divisors -- but it is the
    // invariant the whole "even band width" property rests on, so the test
    // holds to it rather than papering over it.)
    const int lh = 96, lw = 160;
    const float asp = static_cast<float>(lw) / static_cast<float>(lh);
    const int fh = 24, fw = int(fh * asp + 0.5f);

    // A disc, so the field's contours are circles: a box distance could not
    // produce this, which is what makes it a real test of the field path.
    std::vector<unsigned char> mask(size_t(fw) * fh, 0);
    for (int y = 0; y < fh; ++y)
        for (int x = 0; x < fw; ++x) {
            const float u = (float(x) / (fw - 1) - 0.5f) * 2.f * asp;
            const float v = (float(y) / (fh - 1) - 0.5f) * 2.f;
            if (u * u + v * v < 0.35f * 0.35f) mask[size_t(y) * fw + x] = 1;
        }
    std::vector<float> dist;
    mirror::DistanceOutside(mask, fw, fh, dist);
    const float per_px = 2.f / float(fh);
    for (float& d : dist) d *= per_px;

    mirror::FitRegion r;
    r.on = true; r.use_field = true;
    r.fw = fw; r.fh = fh; r.ax = asp;
    r.fade_start = 0.05f; r.fade_width = 0.30f;

    auto coords = mirror::make_coord_grid(lh, lw, -asp, asp, -1.f, 1.f);
    auto field = mx::array(dist.data(), {fh, fw}, mx::float32);
    auto got = mirror::multi_ripple_features_region(
        coords, make_sources(3), 3.f, 1.8f, 0.1f, -0.2f, 0.f, 0.f, 0.f, 0.f, r,
        0.8f, 0.3f, field)[1];
    auto gotf = mx::contiguous(mx::astype(got, mx::float32));
    mx::eval(gotf);
    const float* gp = gotf.data<float>();

    float worst = 0.f;
    for (int y = 0; y < lh; ++y) {
        for (int x = 0; x < lw; ++x) {
            const float cx = -asp + 2.f * asp * float(x) / float(lw - 1);
            const float cy = -1.f + 2.f * float(y) / float(lh - 1);
            const float want = mirror::region_weight(r, dist.data(), cx, cy);
            worst = std::max(worst, std::fabs(want - gp[size_t(y) * lw + x]));
        }
    }
    const bool ok = worst <= kTol;
    std::printf("  [%s] %-38s (%.6f)\n", ok ? " ok " : "FAIL",
                "kernel matches a CPU bilinear sample", worst);
    if (!ok) ++g_fail;

    // The disc's centre is deep inside; a frame corner is far outside. If these
    // were reversed the sense of the field would be inverted -- which looks
    // plausible on screen right up until the subject is the thing that fades.
    check_eq(mirror::region_weight(r, dist.data(), 0.f, 0.f), 0.f,
             "field: the subject is fully inside");
    check_eq(mirror::region_weight(r, dist.data(), asp * 0.98f, 0.98f), 1.f,
             "field: the far corner is fully outside");

    // Even band width is the reason for the field: sample the weight along the
    // x and y axes at the same distance beyond the disc and they must agree.
    // The box form cannot do this unless it happens to be square.
    const float probe = 0.35f + 0.15f;
    const float wx = mirror::region_weight(r, dist.data(), probe, 0.f);
    const float wy = mirror::region_weight(r, dist.data(), 0.f, probe);
    std::printf("  [%s] %-38s (x %.3f, y %.3f)\n",
                std::fabs(wx - wy) < 0.06f ? " ok " : "FAIL",
                "the band is the same width on both axes", wx, wy);
    if (!(std::fabs(wx - wy) < 0.06f)) ++g_fail;

    // An empty mask must not claim the whole frame as "inside".
    std::vector<unsigned char> none(size_t(fw) * fh, 0);
    std::vector<float> nodist;
    mirror::DistanceOutside(none, fw, fh, nodist);
    bool all_far = true;
    for (float d : nodist) all_far = all_far && d > 1e6f;
    std::printf("  [%s] %s\n", all_far ? " ok " : "FAIL",
                "an empty mask is infinitely far from everything");
    if (!all_far) ++g_fail;
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
        {"drop packets",        3.0f, 1.8f,  0.0f,  0.0f, 0.00f, 0.12f,  0.0f,  0.0f, 5, 0.14f},
        {"packets + warp",      3.0f, 1.8f,  0.3f, -0.2f, 0.40f, 0.12f,  0.0f,  0.0f, 5, 0.14f},
        {"wide packet",         6.0f, 0.6f,  0.0f,  0.0f, 0.20f, 0.00f,  0.0f,  0.0f, 4, 0.45f},
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

    // The region: a crop in the middle, one off-centre, a very soft edge and a
    // near-hard one, and the case where both latents are equal (which must be
    // identical to having no region at all).
    std::printf("\nfit region\n");
    mirror::FitRegion c{true, 0.f, 0.f, 0.5f, 0.4f, 0.f, 0.3f};
    run_region("centred crop", c, -0.9f, 0.7f, 0.4f, -0.3f, 5);
    mirror::FitRegion o{true, 0.7f, -0.4f, 0.3f, 0.35f, 0.f, 0.3f};
    run_region("off-centre crop", o, -0.9f, 0.7f, 0.4f, -0.3f, 5);
    mirror::FitRegion soft{true, 0.f, 0.f, 0.4f, 0.4f, 0.f, 1.2f};
    run_region("wide feather", soft, 0.2f, 0.9f, -0.6f, 0.1f, 2);
    mirror::FitRegion hard{true, 0.f, 0.f, 0.4f, 0.4f, 0.f, 0.01f};
    run_region("hard edge", hard, 0.2f, 0.9f, -0.6f, 0.1f, 2);
    run_region("latents equal", c, 0.3f, -0.2f, 0.3f, -0.2f, 5);
    run_region("no sources", c, -0.9f, 0.7f, 0.4f, -0.3f, 0);

    std::printf("\nregion weight shape\n");
    test_region_shape();

    std::printf("\nregion field (hull-following falloff)\n");
    test_region_field();

    // The un-regioned call must still go down the identical arithmetic: this is
    // the guarantee that adding the region changed nothing for every existing
    // caller, which is most of them.
    {
        auto coords = mirror::make_coord_grid(96, 160, -1.f, 1.f, -1.f, 1.f);
        auto plain = mirror::multi_ripple_features(coords, make_sources(3), 3.f,
                                                   1.8f, 0.3f, -0.2f, 0.2f, 0.1f);
        auto viaR = mirror::multi_ripple_features_region(
            coords, make_sources(3), 3.f, 1.8f, 0.3f, -0.2f, 0.2f, 0.1f, 0.f, 0.f,
            mirror::FitRegion{}, 0.3f, -0.2f)[0];
        auto d = mx::max(mx::abs(mx::subtract(mx::astype(plain, mx::float32),
                                              mx::astype(viaR, mx::float32))));
        mx::eval(d);
        const bool ok = d.item<float>() == 0.f;
        std::printf("  [%s] %s\n", ok ? " ok " : "FAIL",
                    "region off is bit-identical to the plain call");
        if (!ok) ++g_fail;
    }

    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
