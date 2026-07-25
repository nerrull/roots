// mirror_train — live fitting of the neural mirror to a target image.
//
// Closes the loop: mlp_forward renders, mlp_backward differentiates, and this
// applies Adam. No autodiff involved and none needed — the loss is mean squared
// error, so the cotangent handed to the backward is just
// 2*(out - target)/count, written out by hand.
//
// Training resolution is deliberately independent of display resolution. A step
// costs roughly 3x a render over the same points (forward + backward +
// optimiser), so training at display resolution cannot fit in a frame that also
// has to draw. Fitting at a quarter of the display and rendering the result at
// full size is the arrangement that works, and it costs nothing in quality —
// the network is continuous, so it is sampled at whatever resolution is asked
// for regardless of where the gradient came from.
#pragma once

#include <mlx/mlx.h>
#include <vector>

#include "mlp_forward.h"

namespace mirror {

namespace mx = mlx::core;

// Adam over a flat weight vector. State lives in fp32 even though the kernels
// consume fp16: the update is a small accumulation over many steps and fp16
// loses it (weight deltas fall below the fp16 epsilon around the point the fit
// starts converging).
class MlpTrainer {
public:
    MlpTrainer() : w_(mx::zeros({1})), m_(mx::zeros({1})), v_(mx::zeros({1})) {}

    // `init` seeds the trainable weights, so fitting starts from whatever the
    // network currently looks like rather than from noise.
    void reset(const MLPConfig& cfg, const mx::array& init);

    bool ready() const { return ready_; }
    int steps() const { return step_; }
    const mx::array& weights() const { return w_; }
    const MLPConfig& config() const { return cfg_; }

    // Sets the fitting target. `rgb` is h*w*3 floats in [0,1], row-major.
    //
    // `mask` is optional and h*w bytes: non-zero means "train on this pixel".
    // Empty means all of them. Masked pixels are *gathered into a compact
    // batch* rather than being zero-weighted in place, so the training pass
    // genuinely only touches them -- with a mask covering 2% of the frame the
    // step costs about 2% of the unmasked one, rather than the same.
    //
    // The network stays global: nothing constrains it outside the mask, so
    // whatever it does there is extrapolation. That is the point when the mask
    // is a person and the background is meant to stay generative.
    void setTarget(const std::vector<float>& rgb, int h, int w,
                   const std::vector<unsigned char>& mask = {});

    // Number of pixels the last setTarget selected, and the grid it came from.
    int trainedPixels() const { return trained_px_; }
    bool masked() const { return masked_; }
    // Row indices into the h*w fit grid, for gathering the features to match.
    const mx::array& indices() const { return idx_; }
    bool hasTarget() const { return target_h_ > 0; }
    int targetH() const { return target_h_; }
    int targetW() const { return target_w_; }

    // One optimiser step against `features` (N, in_dim), which must be the
    // coordinate features evaluated on the same grid the target was set at.
    // Returns the MSE *before* the update. Evaluates eagerly: a training step
    // whose graph is left unevaluated just piles up until the next render and
    // makes per-frame timing meaningless.
    float step(const mx::array& features, float lr);

private:
    MLPConfig cfg_{8, 32, 3, 6};
    mx::array w_, m_, v_;
    mx::array target_ = mx::zeros({1});
    mx::array idx_ = mx::zeros({1});
    int target_h_ = 0, target_w_ = 0;
    int trained_px_ = 0;
    bool masked_ = false;
    int step_ = 0;
    bool ready_ = false;
};

}  // namespace mirror
