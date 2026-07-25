#include "mirror_train.h"

#include "mlp_backward.h"

#include <cmath>
#include <stdexcept>

namespace mirror {

namespace {
inline mx::array S(float v) { return mx::array(v); }

// Adam's usual constants. beta2 = 0.999 over a fit that runs for a few thousand
// steps means the second moment is still warming up early on, which is exactly
// what the bias correction below is for.
constexpr float kBeta1 = 0.9f;
constexpr float kBeta2 = 0.999f;
constexpr float kEps = 1e-8f;
}  // namespace

void MlpTrainer::reset(const MLPConfig& cfg, const mx::array& init) {
    cfg_ = cfg;
    const int n = cfg.total_weights();
    if (init.size() != static_cast<size_t>(n)) {
        throw std::runtime_error("MlpTrainer::reset: init has the wrong size");
    }
    w_ = mx::astype(mx::reshape(init, {n}), mx::float32);
    m_ = mx::zeros({n}, mx::float32);
    v_ = mx::zeros({n}, mx::float32);
    mx::eval(w_, m_, v_);
    step_ = 0;
    ready_ = true;
}

void MlpTrainer::setTarget(const std::vector<float>& rgb, int h, int w,
                           const std::vector<unsigned char>& mask) {
    if (h <= 0 || w <= 0 || rgb.size() != static_cast<size_t>(h) * w * 3) {
        throw std::runtime_error("MlpTrainer::setTarget: size mismatch");
    }
    if (!mask.empty() && mask.size() != static_cast<size_t>(h) * w) {
        throw std::runtime_error("MlpTrainer::setTarget: mask size mismatch");
    }
    target_h_ = h;
    target_w_ = w;
    masked_ = !mask.empty();

    if (!masked_) {
        target_ = mx::astype(mx::array(rgb.data(), {h * w, 3}, mx::float32),
                             mx::float16);
        idx_ = mx::zeros({1}, mx::uint32);
        trained_px_ = h * w;
        mx::eval(target_);
        return;
    }

    // Compact the selected pixels. Building the index list on the CPU and
    // gathering once is much cheaper than masking every step: the mask changes
    // at camera rate at most, the gradient runs every frame.
    std::vector<uint32_t> idx;
    std::vector<float> vals;
    idx.reserve(mask.size() / 4);
    vals.reserve(mask.size() / 4 * 3);
    for (size_t i = 0; i < mask.size(); ++i) {
        if (!mask[i]) continue;
        idx.push_back(static_cast<uint32_t>(i));
        vals.push_back(rgb[i * 3 + 0]);
        vals.push_back(rgb[i * 3 + 1]);
        vals.push_back(rgb[i * 3 + 2]);
    }
    trained_px_ = static_cast<int>(idx.size());
    if (trained_px_ == 0) {
        // An empty mask would make the loss 0/0. Treat it as "nothing to learn
        // this frame" rather than poisoning the weights with NaN.
        target_ = mx::zeros({0, 3}, mx::float16);
        idx_ = mx::zeros({0}, mx::uint32);
        return;
    }
    target_ = mx::astype(
        mx::array(vals.data(), {trained_px_, 3}, mx::float32), mx::float16);
    idx_ = mx::array(idx.data(), {trained_px_}, mx::uint32);
    mx::eval(target_, idx_);
}

float MlpTrainer::step(const mx::array& features, float lr) {
    if (!ready_ || !hasTarget()) return 0.f;
    const int n = features.shape(0);
    if (n != trained_px_) {
        throw std::runtime_error("MlpTrainer::step: features do not match target");
    }
    if (n == 0) return 0.f;   // empty mask: nothing to learn from this frame

    auto w16 = mx::astype(w_, mx::float16);
    auto out = fused_mlp_forward(features, w16, cfg_);

    // MSE and its cotangent, written directly. d/dout mean((out-t)^2) is
    // 2*(out-t)/count, where count is every element, not every row.
    auto diff = mx::subtract(mx::astype(out, mx::float32),
                             mx::astype(target_, mx::float32));
    auto loss = mx::mean(mx::multiply(diff, diff));
    const float scale = 2.0f / static_cast<float>(n * cfg_.out_dim);
    auto cot = mx::astype(mx::multiply(diff, S(scale)), mx::float16);

    auto grads = fused_mlp_backward(features, w16, cot, cfg_);
    auto g = mx::astype(grads.second, mx::float32);

    ++step_;
    m_ = mx::add(mx::multiply(S(kBeta1), m_), mx::multiply(S(1.f - kBeta1), g));
    v_ = mx::add(mx::multiply(S(kBeta2), v_),
                 mx::multiply(S(1.f - kBeta2), mx::multiply(g, g)));
    // Bias correction. Without it the first few dozen steps take a much larger
    // effective stride than `lr`, which with a periodic activation is enough to
    // leave the basin and never come back.
    const float bc1 = 1.f - std::pow(kBeta1, static_cast<float>(step_));
    const float bc2 = 1.f - std::pow(kBeta2, static_cast<float>(step_));
    auto mhat = mx::divide(m_, S(bc1));
    auto vhat = mx::divide(v_, S(bc2));
    w_ = mx::subtract(w_, mx::multiply(S(lr),
                                       mx::divide(mhat, mx::add(mx::sqrt(vhat), S(kEps)))));

    mx::eval(w_, m_, v_, loss);
    return loss.item<float>();
}

}  // namespace mirror
