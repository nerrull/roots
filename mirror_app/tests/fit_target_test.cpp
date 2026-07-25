// Fitting a *moving* target, and the resampling that feeds it.
//
// fit_test covers convergence onto a fixed image. A camera feed is a different
// problem: the target moves every frame and the fit never finishes, it tracks.
// The failure that matters there is not "does it converge" but "does it keep
// up, and does it recover when the scene changes" -- which a static test cannot
// see at all.
//
// Uses a synthetic moving target rather than the sensor, so it runs in CI, on a
// machine with nothing plugged in, and deterministically. The Kinect path on
// top of this is frame acquisition and colour-order handling; the fitting
// behaviour underneath is what is checked here.
//
// Exit 0 on pass.

#include "fit_target.h"
#include "pond_state.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++g_fail;
}

// A bright disc on a dark field, at a moving centre. Deliberately a *localised*
// feature: a global gradient would let a barely-working fit score well by
// tracking the mean, where following a disc requires actually placing energy in
// the right part of the frame.
std::vector<float> moving_disc(int h, int w, float cx, float cy) {
    std::vector<float> rgb(size_t(h) * w * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float u = float(x) / float(w - 1);
            const float v = float(y) / float(h - 1);
            const float d = std::sqrt((u - cx) * (u - cx) + (v - cy) * (v - cy));
            const float m = std::exp(-(d * d) / (2.f * 0.09f * 0.09f));
            float* px = &rgb[(size_t(y) * w + x) * 3];
            px[0] = 0.08f + 0.90f * m;
            px[1] = 0.08f + 0.55f * m;
            px[2] = 0.10f + 0.20f * m;
        }
    }
    return rgb;
}

float mse(const std::vector<float>& a, const std::vector<float>& b) {
    double acc = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double e = double(a[i]) - double(b[i]);
        acc += e * e;
    }
    return n ? float(acc / n) : 0.f;
}

std::vector<float> render_to_vec(mirror::Pond& pond, int h, int w,
                                 const mirror::PondParams& p) {
    auto img = mx::contiguous(mx::astype(pond.render(h, w, 0.0, p), mx::float32));
    mx::eval(img);
    const float* d = img.data<float>();
    return std::vector<float>(d, d + size_t(h) * w * 3);
}

void test_downsample() {
    std::printf("\nDownsampleRGB8\n");
    // 4x4 BGRX, left half pure blue, right half pure red. A 2x1 box downsample
    // must give one blue and one red pixel -- point sampling would too, so the
    // real check is the channel order and the averaging below.
    const int W = 4, H = 4;
    std::vector<unsigned char> src(size_t(W) * H * 4, 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            unsigned char* px = &src[(size_t(y) * W + x) * 4];
            if (x < W / 2) { px[0] = 255; px[1] = 0; px[2] = 0; }   // B
            else           { px[0] = 0; px[1] = 0; px[2] = 255; }   // R
        }
    }
    std::vector<float> dst;
    mirror::DownsampleRGB8(src.data(), W, H, 4, /*r_off=*/2, /*b_off=*/0, 2, 1, dst);
    check(dst.size() == 6, "output is 2x1x3");
    check(dst[2] > 0.9f && dst[0] < 0.1f, "BGRX left half decodes as blue");
    check(dst[3] > 0.9f && dst[5] < 0.1f, "BGRX right half decodes as red");

    // Averaging: a half-white/half-black column must land at ~0.5, which point
    // sampling would never produce.
    std::vector<unsigned char> grad(size_t(W) * H * 4, 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            unsigned char* px = &grad[(size_t(y) * W + x) * 4];
            const unsigned char v = (x < W / 2) ? 0 : 255;
            px[0] = px[1] = px[2] = v;
        }
    mirror::DownsampleRGB8(grad.data(), W, H, 4, 2, 0, 1, 1, dst);
    check(std::fabs(dst[0] - 0.5f) < 0.02f, "box filter averages rather than samples");

    // RGBX vs BGRX must actually differ, or the offsets are being ignored.
    std::vector<float> as_bgr, as_rgb;
    mirror::DownsampleRGB8(src.data(), W, H, 4, 2, 0, 2, 1, as_bgr);
    mirror::DownsampleRGB8(src.data(), W, H, 4, 0, 2, 2, 1, as_rgb);
    check(std::fabs(as_bgr[0] - as_rgb[0]) > 0.9f, "channel order is honoured");
}

void test_static_target() {
    std::printf("\nStaticFitTarget\n");
    auto img = moving_disc(24, 32, 0.5f, 0.5f);
    mirror::StaticFitTarget t(img, 32, 24);
    std::vector<float> out;
    check(t.poll(32, 24, out) && out.size() == img.size(), "poll at native size");
    check(mse(out, img) < 1e-12f, "native-size poll is exact");
    check(t.poll(16, 12, out) && out.size() == size_t(16) * 12 * 3,
          "poll resamples when the fit grid differs");
    check(t.frames() == 2, "frame counter advances");
}

void test_tracking() {
    std::printf("\ntracking a moving target\n");
    const int FH = 72, FW = 96;
    mirror::Pond pond(11);
    mirror::PondParams p;
    p.sine_layers = 1;
    p.sine_w0 = 12.f;

    // Settle onto the disc at its starting position.
    auto frame = moving_disc(FH, FW, 0.3f, 0.5f);
    pond.beginFit(frame, FH, FW, p);
    for (int i = 0; i < 500; ++i) pond.fitStep(5e-3f, p);
    const float settled = mse(render_to_vec(pond, FH, FW, p), frame);
    std::printf("       settled MSE %.5f\n", settled);
    check(settled < 0.01f, "converges on the first position");

    // Now move it. Immediately after the jump the render still shows the old
    // position, so error must spike -- if it does not, the fit is not actually
    // tracking anything localised.
    auto moved = moving_disc(FH, FW, 0.7f, 0.5f);
    pond.updateFitTarget(moved, FH, FW);
    const float right_after = mse(render_to_vec(pond, FH, FW, p), moved);
    check(right_after > settled * 2.f, "moving the target spikes the error");

    // ...and it must come back, in a plausible number of frames. 120 steps is
    // 2 s at one step per frame, or 1 s at two.
    for (int i = 0; i < 120; ++i) pond.fitStep(5e-3f, p);
    const float recovered = mse(render_to_vec(pond, FH, FW, p), moved);
    std::printf("       after jump %.5f -> %.5f in 120 steps\n", right_after, recovered);
    check(recovered < right_after * 0.5f, "recovers toward the new position");
    check(recovered < settled * 3.f, "recovery approaches the settled quality");

    // Continuous motion, which is the actual camera case: retarget every step
    // and check the error stays bounded rather than drifting away.
    float worst = 0.f;
    for (int i = 0; i < 240; ++i) {
        const float ph = 2.f * float(M_PI) * i / 240.f;
        auto f = moving_disc(FH, FW, 0.5f + 0.18f * std::cos(ph),
                             0.5f + 0.18f * std::sin(ph));
        // updateFitTarget, not beginFit: the latter resets Adam, which is
        // correct when starting a fit and ruinous when retargeting per frame.
        pond.updateFitTarget(f, FH, FW);
        pond.fitStep(5e-3f, p);
        if (i > 60) worst = std::max(worst, mse(render_to_vec(pond, FH, FW, p), f));
    }
    std::printf("       worst MSE while tracking continuous motion: %.5f\n", worst);
    check(worst < 0.05f, "stays locked on a continuously moving target");
    check(std::isfinite(worst), "no divergence under constant retargeting");
}

// A centred square mask, and the same square as a target.
std::vector<unsigned char> square_mask(int h, int w, float side_frac) {
    std::vector<unsigned char> m(size_t(h) * w, 0);
    const float half_h = 0.5f * side_frac;
    const float half_w = half_h * float(h) / float(w);
    for (int y = 0; y < h; ++y) {
        const float v = float(y) / float(h - 1) - 0.5f;
        for (int x = 0; x < w; ++x) {
            const float u = float(x) / float(w - 1) - 0.5f;
            if (std::fabs(u) <= half_w && std::fabs(v) <= half_h)
                m[size_t(y) * w + x] = 1;
        }
    }
    return m;
}

std::vector<float> square_image(int h, int w, float side_frac) {
    std::vector<float> rgb(size_t(h) * w * 3);
    auto m = square_mask(h, w, side_frac);
    for (size_t i = 0; i < m.size(); ++i) {
        rgb[i * 3 + 0] = m[i] ? 0.95f : 0.05f;
        rgb[i * 3 + 1] = m[i] ? 0.93f : 0.06f;
        rgb[i * 3 + 2] = m[i] ? 0.90f : 0.08f;
    }
    return rgb;
}

// Mean squared error over only the pixels a mask selects (or its complement).
float masked_mse(const std::vector<float>& a, const std::vector<float>& b,
                 const std::vector<unsigned char>& m, bool inside) {
    double acc = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < m.size(); ++i) {
        if (bool(m[i]) != inside) continue;
        for (int c = 0; c < 3; ++c) {
            const double e = double(a[i * 3 + c]) - double(b[i * 3 + c]);
            acc += e * e;
        }
        n += 3;
    }
    return n ? float(acc / n) : 0.f;
}

void test_masked_fit() {
    std::printf("\nmasked fitting\n");
    const int FH = 120, FW = 160;
    const float side = 0.20f;
    const auto target = square_image(FH, FW, side);
    const auto mask = square_mask(FH, FW, side);

    size_t on = 0;
    for (unsigned char v : mask) on += v ? 1 : 0;
    std::printf("       mask selects %zu / %d px (%.1f%%)\n", on, FH * FW,
                100.0 * on / (FH * FW));

    mirror::Pond pond(11);
    mirror::PondParams p;
    p.sine_layers = 1;
    p.sine_w0 = 30.f;

    pond.beginFit(target, FH, FW, p, mask);
    check(pond.fitPixels() == int(on),
          "the training pass runs on exactly the masked pixels");

    for (int i = 0; i < 600; ++i) pond.fitStep(3e-3f, p);
    const auto got = render_to_vec(pond, FH, FW, p);

    const float in_err = masked_mse(got, target, mask, /*inside=*/true);
    const float out_err = masked_mse(got, target, mask, /*inside=*/false);
    std::printf("       MSE inside mask %.5f   outside %.5f\n", in_err, out_err);
    check(in_err < 0.01f, "converges inside the mask");
    // The whole point: nothing constrains the network outside, so it is under
    // no obligation to match there. If it matched anyway, the mask is not
    // actually restricting the training pass.
    check(out_err > in_err * 2.f, "outside the mask is genuinely unconstrained");

    // An empty mask must be a no-op, not a NaN. A segmentation mask can
    // legitimately select nothing when nobody is in frame.
    pond.updateFitTarget(target, FH, FW, std::vector<unsigned char>(size_t(FH) * FW, 0));
    check(pond.fitPixels() == 0, "an empty mask selects nothing");
    const float before = masked_mse(render_to_vec(pond, FH, FW, p), target, mask, true);
    for (int i = 0; i < 10; ++i) pond.fitStep(3e-3f, p);
    const auto after_img = render_to_vec(pond, FH, FW, p);
    const float after = masked_mse(after_img, target, mask, true);
    check(std::isfinite(after) && std::fabs(after - before) < 1e-6f,
          "an empty mask leaves the weights untouched");
    bool finite = true;
    for (float v : after_img) finite = finite && std::isfinite(v);
    check(finite, "no NaN from a zero-pixel mask");
}

}  // namespace

int main() {
    std::printf("fit_target_test: resampling + live-target fitting\n");
    test_downsample();
    test_static_target();
    test_tracking();
    test_masked_fit();
    std::printf("\n%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
