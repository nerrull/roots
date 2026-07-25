// mlp_backward — fused Metal backward for the neural mirror's MLP.
//
// Companion to mlp_forward.h: same weight layout, same config, same hybrid
// split. Together they make the network trainable in-process, which is what
// live fitting needs — the forward alone only lets it be looked at.
//
// Returns (dcoords, dweights), both fp32. dweights accumulates via
// atomic<float> across threadgroups, so it is deterministic only up to
// float atomic ordering (ULP level).
#pragma once

#include <mlx/mlx.h>
#include <utility>

#include "mlp_forward.h"

namespace mirror {

namespace mx = mlx::core;

// Backward tiling. 16 rows rather than the forward's 32: the backward keeps
// *every* layer's activations on-chip simultaneously, so its threadgroup
// footprint grows with depth where the forward's does not.
inline constexpr int kBwTileRows = 16;
inline constexpr int kBwSimdGroups = 8;
inline constexpr int kBwThreads = 32 * kBwSimdGroups;

// Threadgroup memory the backward needs, in bytes:
//   acts   : (num_layers + 1) * TR * HIDDEN halves   -- every layer's output
//   dacts  : max(split, 1)    * TR * HIDDEN halves   -- act'(z) for sine layers
//   wbuf   : HIDDEN * HIDDEN halves
//   dbuf   : 2 * TR * HIDDEN halves
//   scr    : SG * 64 floats
//
// The dacts term is sized by `split`, not by depth, which is the whole reason
// the hybrid is trainable at width 64: storing a derivative for every layer
// needs 43008 bytes there (over the 32768 limit), for one layer only 30720.
int fused_mlp_backward_threadgroup_bytes(const MLPConfig& cfg);

// coords: (N, in_dim) or (N, in_dim_pad). cotangent: (N, out_dim).
// Throws if the config does not fit in threadgroup memory.
std::pair<mx::array, mx::array> fused_mlp_backward(mx::array coords,
                                                   mx::array weights,
                                                   mx::array cotangent,
                                                   const MLPConfig& cfg);

}  // namespace mirror
