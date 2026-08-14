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

// Switching the fit grid mid-fit. Routine now that the crop and the whole feed
// carry their own downscale: every time someone walks in or out of frame, the
// target changes size under a fit that is already running. The weights are a
// continuous function and the optimiser state is per-weight, so none of it
// depends on the pixel count -- but that is a claim about the code, and this is
// the test of it.
void test_regrid() {
    std::printf("\nregridding mid-fit\n");
    const auto target_a = moving_disc(96, 128, 0.5f, 0.5f);
    mirror::Pond pond(11);
    mirror::PondParams p;
    p.sine_layers = 1;
    p.sine_w0 = 30.f;

    pond.beginFit(target_a, 96, 128, p);
    for (int i = 0; i < 300; ++i) pond.fitStep(3e-3f, p);
    const float settled = mse(render_to_vec(pond, 96, 128, p), target_a);
    const int steps_before = pond.fitSteps();

    // Same picture, finer grid -- what a crop appearing does.
    const auto target_b = moving_disc(192, 256, 0.5f, 0.5f);
    pond.updateFitTarget(target_b, 192, 256);
    const float after_swap = mse(render_to_vec(pond, 96, 128, p), target_a);
    std::printf("       settled %.5f -> %.5f immediately after the regrid\n",
                settled, after_swap);
    check(std::fabs(after_swap - settled) < 1e-6f,
          "the regrid alone does not disturb the image");
    check(pond.fitSteps() == steps_before, "nor does it reset the step count");

    for (int i = 0; i < 60; ++i) pond.fitStep(3e-3f, p);
    const float after_steps = mse(render_to_vec(pond, 96, 128, p), target_a);
    std::printf("       after 60 steps on the new grid: %.5f\n", after_steps);
    check(std::isfinite(after_steps), "training continues on the new grid");
    check(after_steps < settled * 2.f, "and does not undo the fit");

    // ...and back down again, the way losing a crop goes.
    pond.updateFitTarget(target_a, 96, 128);
    for (int i = 0; i < 60; ++i) pond.fitStep(3e-3f, p);
    const float back = mse(render_to_vec(pond, 96, 128, p), target_a);
    std::printf("       back on the coarse grid: %.5f\n", back);
    check(std::isfinite(back) && back < settled * 2.f, "regridding is reversible");
}

// The fit region: the subject holds still while the field around it moves.
//
// This is the one claim of the feature that cannot be read off the code -- z is
// an MLP input, so "the latent animates outside and not inside" is a statement
// about what the network draws, and the only way to know it holds is to render
// twice and compare.
void test_region() {
    std::printf("\nfit region\n");
    const int FH = 96, FW = 128;
    const float side = 0.30f;
    const auto target = square_image(FH, FW, side);
    const auto mask = square_mask(FH, FW, side);

    mirror::Pond pond(11);
    mirror::PondParams p;
    p.sine_layers = 1;
    p.sine_w0 = 30.f;
    p.z = 0.4f;                       // a deliberately non-zero starting latent
    pond.beginFit(target, FH, FW, p, mask);
    for (int i = 0; i < 400; ++i) pond.fitStep(3e-3f, p);
    check(std::fabs(pond.fitZ() - 0.4f) < 1e-6f,
          "the fit remembers the latent it was begun at");

    // A region over the trained square. square_mask is square in pixels, which
    // in coord space (x over (-asp, asp), y over (-1, 1)) is side_frac either
    // way.
    p.region.on = true;
    p.region.cx = 0.f; p.region.cy = 0.f;
    p.region.hx = side; p.region.hy = side;
    p.region.fade_start = 0.f;
    p.region.fade_width = 0.2f;

    // Well outside the feather band, so this is not measuring the crossfade.
    std::vector<unsigned char> outer(size_t(FH) * FW, 0);
    for (int y = 0; y < FH; ++y) {
        const float v = 2.f * (float(y) / float(FH - 1) - 0.5f);
        for (int x = 0; x < FW; ++x) {
            const float u = 2.f * (float(x) / float(FW - 1) - 0.5f) * (float(FW) / FH);
            if (std::max(std::fabs(u), std::fabs(v)) > side * 2.5f)
                outer[size_t(y) * FW + x] = 1;
        }
    }

    p.z_free_outside = true;
    const auto a = render_to_vec(pond, FH, FW, p);
    p.z += 2.0f;                       // the latent moves on
    const auto b = render_to_vec(pond, FH, FW, p);

    const float in_move = masked_mse(a, b, mask, /*inside=*/true);
    const float out_move = masked_mse(a, b, outer, /*inside=*/true);
    std::printf("       z moved by 2.0: subject changed %.8f, field %.8f\n",
                in_move, out_move);
    check(in_move < 1e-9f, "the fitted subject does not move with the latent");
    check(out_move > 1e-4f, "the field around it does");

    // Without the split, z is an ordinary global control again -- a fit must
    // not quietly disable it.
    p.z_free_outside = false;
    const auto c = render_to_vec(pond, FH, FW, p);
    check(masked_mse(a, c, mask, true) > 1e-9f,
          "with the split off, z moves the subject too");

    // Colour: the subject keeps its own, the field drains. Checked as
    // saturation, since a greyscale pixel is one with no channel spread.
    p.z_free_outside = false;
    p.grey_outside = 1.f;
    const auto g = render_to_vec(pond, FH, FW, p);
    auto saturation = [&](const std::vector<float>& img,
                          const std::vector<unsigned char>& m) {
        double acc = 0.0; size_t n = 0;
        for (size_t i = 0; i < m.size(); ++i) {
            if (!m[i]) continue;
            const float r = img[i * 3], gg = img[i * 3 + 1], bb = img[i * 3 + 2];
            acc += std::max({r, gg, bb}) - std::min({r, gg, bb});
            ++n;
        }
        return n ? acc / n : 0.0;
    };
    const double sat_in = saturation(g, mask), sat_out = saturation(g, outer);
    std::printf("       grey outside: subject saturation %.5f, field %.5f\n",
                sat_in, sat_out);
    check(sat_out < 1e-5, "the field outside is fully desaturated");
    check(sat_in > sat_out, "the subject keeps its colour");

    // Half-strength must land between, not snap.
    p.grey_outside = 0.5f;
    const auto h = render_to_vec(pond, FH, FW, p);
    const double sat_half = saturation(h, outer);
    const double sat_full = saturation(a, outer);
    std::printf("       half strength: %.5f (off %.5f, full %.5f)\n",
                sat_half, sat_full, sat_out);
    check(sat_half > sat_out && sat_half < sat_full, "half drains halfway");
}

// The frame shift behind the head-centred fit mode. Worth testing on its own:
// a shift with the sign flipped still produces a stable-looking fit, of a face
// moved twice as far as it should be, which is not obvious from the output.
void test_shift() {
    std::printf("\nframe shift\n");
    const int W = 8, H = 6;
    std::vector<float> img(size_t(W) * H * 3, 0.f);
    auto at = [&](int x, int y) { return &img[(size_t(y) * W + x) * 3]; };
    at(2, 1)[0] = 1.f;                       // a red mark left of centre, high up

    auto shifted = img;
    mirror::ShiftRGBF(W, H, 2, 3, shifted);
    auto px = [&](const std::vector<float>& v, int x, int y) {
        return v[(size_t(y) * W + x) * 3];
    };
    check(px(shifted, 4, 4) > 0.99f, "a +x/+y shift moves the mark down-right");
    check(px(shifted, 2, 1) < 0.01f, "and vacates where it was");

    // Round trip: the mark returns, even though the edges cannot.
    mirror::ShiftRGBF(W, H, -2, -3, shifted);
    check(px(shifted, 2, 1) > 0.99f, "the inverse shift brings it back");

    // Edge clamp, not wrap: shifting right replicates the leftmost column into
    // the strip it vacates. A wrap would drag the far side of the room into the
    // crop the moment anyone stood near the frame's edge.
    std::vector<float> edge(size_t(W) * H * 3, 0.f);
    for (int y = 0; y < H; ++y) {
        edge[(size_t(y) * W + 0) * 3 + 1] = 1.f;          // green leading column
        edge[(size_t(y) * W + (W - 1)) * 3 + 2] = 1.f;    // blue trailing column
    }
    mirror::ShiftRGBF(W, H, 3, 0, edge);
    bool clamped = true, wrapped = false;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x <= 3; ++x)
            clamped = clamped && edge[(size_t(y) * W + x) * 3 + 1] > 0.99f;
        for (int x = 0; x < W; ++x)
            wrapped = wrapped || edge[(size_t(y) * W + x) * 3 + 2] > 0.5f;
    }
    check(clamped, "the vacated strip repeats the leading column");
    check(!wrapped, "content pushed off the far edge is gone, not wrapped");

    auto same = img;
    mirror::ShiftRGBF(W, H, 0, 0, same);
    check(mse(same, img) == 0.f, "a zero shift is exactly a no-op");

    // A shift past the frame must leave something defined everywhere rather
    // than reading out of bounds.
    auto far_ = img;
    mirror::ShiftRGBF(W, H, 99, -99, far_);
    bool finite = true;
    for (float v : far_) finite = finite && std::isfinite(v);
    check(finite, "a shift larger than the frame stays in bounds");
}

}  // namespace

int main() {
    std::printf("fit_target_test: resampling + live-target fitting\n");
    test_downsample();
    test_static_target();
    test_shift();
    test_tracking();
    test_masked_fit();
    test_regrid();
    test_region();
    std::printf("\n%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
