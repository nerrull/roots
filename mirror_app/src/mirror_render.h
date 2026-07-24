// mirror_render — coordinate-grid + enriched features + fused MLP → low-res image.
//
// Pure C++/MLX port of the render half of neuromirror's mlx_fused_mlp
// (display.make_coord_grid, features.multi_ripple_features, forward). Produces a
// small (lh, lw, 4) fp16 RGBA image on the GPU; the display path samples it with
// a linear filter to upscale to the window (the "low-res + bilinear upsample"
// accelerator, done in the sampler instead of a separate MLX kernel).
#pragma once

#include <mlx/mlx.h>
#include <array>
#include <vector>

#include "mlp_forward.h"

namespace mirror {

namespace mx = mlx::core;

// A ripple source: (cx, cy, phase, amp). Center, outward phase (ramp to animate
// propagation), and 0..1 amplitude envelope.
using RippleSource = std::array<float, 4>;

// (H*W, 2) row-major grid of (x, y) coords in the given ranges (fp16).
mx::array make_coord_grid(int h, int w,
                          float x0 = -1.f, float x1 = 1.f,
                          float y0 = -1.f, float y1 = 1.f);

// Port of features.multi_ripple_features (core path: sin/cos ripple accumulation
// + z/z_cos latents; warp/core_radius omitted in v1). coords: (N,2) → (N,8) fp16.
mx::array multi_ripple_features(const mx::array& coords,
                                const std::vector<RippleSource>& sources,
                                float ring_freq = 3.f, float decay = 1.6f,
                                float z = 0.f, float z_cos = 0.f);

// Render a low-res RGBA fp16 image (lh, lw, 4): evaluate the MLP over the grid's
// multi-ripple features and append an opaque alpha channel. out_dim must be 3.
mx::array render_pond_lowres(const mx::array& weights, const MLPConfig& cfg,
                             int lh, int lw,
                             const std::vector<RippleSource>& sources,
                             float ring_freq, float decay, float z, float z_cos);

}  // namespace mirror
