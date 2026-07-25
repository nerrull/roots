// End-to-end fitting: does the training loop actually converge?
//
// The gradient is already checked component-wise (mlp_backward_test). What that
// cannot catch is the loop being wrong around a correct gradient -- a sign
// error in the cotangent, Adam state not persisting, the display path reading
// the pre-fit weights, features not matching the target grid. Every one of
// those still runs, and still produces a plausible-looking number.
//
// So this fits a real target and asserts the loss falls, then asserts the
// *rendered image* moved toward the target too, which is the thing that
// actually matters and is a different claim from "loss went down".
//
// Headless, no fixtures, needs a Metal device. Exit 0 on pass.

#include "mirror_render.h"
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

// A target with structure at several scales: a smooth gradient the network can
// reach immediately, plus rings it can only fit once it has some frequency
// content. Fitting only a gradient would pass even with a crippled network.
std::vector<float> target_image(int h, int w) {
    std::vector<float> rgb(size_t(h) * w * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float u = float(x) / float(w - 1);
            const float v = float(y) / float(h - 1);
            const float r = std::sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f));
            float* px = &rgb[(size_t(y) * w + x) * 3];
            px[0] = u;
            px[1] = v;
            px[2] = 0.5f + 0.5f * std::cos(r * 18.f);
        }
    }
    return rgb;
}

// Mean squared error between a rendered (h, w, 3) image and the target.
float image_mse(const mx::array& img, const std::vector<float>& target) {
    auto c = mx::contiguous(mx::astype(img, mx::float32));
    mx::eval(c);
    const float* d = c.data<float>();
    double acc = 0.0;
    for (size_t i = 0; i < target.size(); ++i) {
        const double e = double(d[i]) - double(target[i]);
        acc += e * e;
    }
    return float(acc / target.size());
}

void run_case(int sine_layers, const char* label) {
    std::printf("\n%s (sine_layers=%d)\n", label, sine_layers);
    const int FH = 96, FW = 160;      // fit grid
    mirror::Pond pond(11);
    mirror::PondParams p;
    p.sine_layers = sine_layers;
    p.sine_w0 = 12.f;
    p.detail = 3.f;
    p.contrast = 6.f;

    const auto target = target_image(FH, FW);

    // Where the untrained network starts, rendered at the fit grid.
    const float mse_before = image_mse(pond.render(FH, FW, 0.0, p), target);

    pond.beginFit(target, FH, FW, p);
    check(pond.fitting() && pond.fitted(), "beginFit arms the trainer");

    const float first = pond.fitStep(3e-3f, p);
    float last = first;
    for (int i = 1; i < 400; ++i) last = pond.fitStep(3e-3f, p);

    std::printf("       loss %.6f -> %.6f after %d steps\n", first, last,
                pond.fitSteps());
    check(std::isfinite(first) && std::isfinite(last), "loss stays finite");
    check(last < first * 0.5f, "loss at least halves");
    check(pond.fitSteps() == 400, "every step counted (Adam state persists)");

    // The claim that matters: the *rendered* image moved toward the target.
    // Loss falling proves the trainer improved; this proves the display path
    // is reading the trained weights rather than the pre-fit ones.
    const float mse_after = image_mse(pond.render(FH, FW, 0.0, p), target);
    std::printf("       render MSE %.6f -> %.6f\n", mse_before, mse_after);
    check(mse_after < mse_before * 0.6f, "rendered image moved toward the target");

    // And it must hold at a resolution the fit never saw -- the network is
    // continuous, so a fit at 96x160 has to render correctly at 192x320.
    auto big = pond.render(FH * 2, FW * 2, 0.0, p);
    const auto big_target = target_image(FH * 2, FW * 2);
    const float mse_big = image_mse(big, big_target);
    std::printf("       render MSE at 2x fit resolution: %.6f\n", mse_big);
    check(mse_big < mse_before * 0.8f, "fit generalises to an unseen resolution");

    // Stopping freezes the weights rather than reverting them.
    pond.stopFit();
    const float mse_stopped = image_mse(pond.render(FH, FW, 0.0, p), target);
    check(std::fabs(mse_stopped - mse_after) < 1e-6f,
          "stopFit keeps the fitted weights");

    // reseed() is the documented way back to an unfitted network.
    pond.reseed();
    check(!pond.fitted(), "reseed clears the fit");
}

}  // namespace

int main() {
    std::printf("fit_test: end-to-end training loop\n");
    run_case(0, "all tanh");
    run_case(1, "STTTT hybrid");
    std::printf("\n%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
