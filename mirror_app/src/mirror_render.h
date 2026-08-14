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
#include <optional>
#include <vector>

#include "mlp_forward.h"

namespace mirror {

namespace mx = mlx::core;

// Enriched raw-coordinate feature width (features.ENRICHED_DIM): x, y, z, bias,
// sin_field, cos_field, z_cos, spare.
inline constexpr int ENRICHED_DIM = 8;

// A ripple source: (cx, cy, phase, amp). Center, outward phase (ramp to animate
// propagation), and 0..1 amplitude envelope.
using RippleSource = std::array<float, 4>;

// Where the network is being supervised, in coord space, and how sharply that
// gives way to the untrained field around it.
//
// The fit region is not just a training detail once it is visible: inside it the
// network is reproducing a person and every input must hold still, while outside
// it nothing is constrained and the field is free to move. Handing the same
// region to the renderer is what lets those two live in one frame -- a latent
// that animates only where it cannot disturb the fit, and colour that belongs to
// the subject rather than the whole canvas.
//
// Two ways to say where the region is:
//
//   a box      max of the two axis distances, so weight is exactly zero over
//              the whole trained rectangle. Analytic, needs nothing uploaded.
//   a field    a distance map (see DistanceOutside), bilinearly sampled. The
//              contours are then offset curves of whatever shape was rasterised
//              -- a face outline, not a rectangle -- and, because the distance
//              is Euclidean and in the same units on both axes, the fade band
//              is the same width everywhere instead of flaring at the corners.
//
// The fade runs from `fade_start` to `fade_start + fade_width`, both measured
// in coord units outward from the region's edge. Two numbers rather than one
// because "where the gradient sits" and "how long it takes" are separate
// complaints: a fade pinned to the edge pools against it however wide it is.
struct FitRegion {
    bool  on = false;
    float cx = 0.f, cy = 0.f;   // box centre
    float hx = 1.f, hy = 1.f;   // box half-extent; weight is 0 inside it
    float fade_start = 0.f;
    float fade_width = 0.3f;

    // Field mode. `ax` is the coord-space half-width the field spans; it covers
    // exactly (-ax, ax) x (-1, 1), the same extent as the render grid.
    bool  use_field = false;
    int   fw = 0, fh = 0;
    float ax = 1.f;
};

// Weight outside the region: 0 at or inside the edge, 1 past the fade, smoothstep
// between. `field` may be null in box mode. Mirrors the kernel exactly
// (tests/ripple_parity_test).
float region_weight(const FitRegion& r, const float* field, float x, float y);

// (H*W, 2) row-major grid of (x, y) coords in the given ranges (fp16).
mx::array make_coord_grid(int h, int w,
                          float x0 = -1.f, float x1 = 1.f,
                          float y0 = -1.f, float y1 = 1.f);

// Full port of features.multi_ripple_features: sin/cos ripple accumulation,
// z/z_cos latents, domain-warp refraction, and core-radius damping.
// coords: (N,2) → (N,8) fp16.
//
// Runs as a single fused Metal kernel. The op-graph formulation cost 9.9 ms at
// 960x540 -- 38% of the frame -- because ~40 FLOPs/pixel became ~75 elementwise
// kernels, each round-tripping the whole array through memory. Fused: 0.60 ms.
mx::array multi_ripple_features(const mx::array& coords,
                                const std::vector<RippleSource>& sources,
                                float ring_freq = 3.f, float decay = 1.6f,
                                float z = 0.f, float z_cos = 0.f,
                                float warp = 0.f, float core_radius = 0.f,
                                float x_offset = 0.f, float y_offset = 0.f);

// The same, with a fit region: `z`/`z_cos` are the latent inside it and
// `z_out`/`z_cos_out` the latent outside, crossfaded by the region weight. The
// second returned array is that weight, (N, 1) fp16 -- emitted here rather than
// recomputed downstream because the renderer wants the identical field for its
// colour falloff, and two formulations that drift apart would put the colour
// edge somewhere the latent edge is not.
//
// Returns {features (N, 8), region weight (N, 1)}.
//
// The two latents are crossfaded as (sin, cos) pairs, not as phases: that is a
// chord across the unit circle rather than an arc, so the latent's magnitude
// dips slightly mid-band. Inaudible in a smooth gradient, and it keeps this a
// plain lerp of the two values the caller already computes.
std::vector<mx::array> multi_ripple_features_region(
    const mx::array& coords, const std::vector<RippleSource>& sources,
    float ring_freq, float decay, float z, float z_cos, float warp,
    float core_radius, float x_offset, float y_offset,
    const FitRegion& region, float z_out, float z_cos_out,
    const std::optional<mx::array>& field = std::nullopt);

// The original op-graph implementation, kept as the readable reference the
// kernel is checked against (see tests/ripple_parity_test.cpp) and as the path
// taken when there are no sources at all. Not for per-frame use.
mx::array multi_ripple_features_ops(const mx::array& coords,
                                    const std::vector<RippleSource>& sources,
                                    float ring_freq = 3.f, float decay = 1.6f,
                                    float z = 0.f, float z_cos = 0.f,
                                    float warp = 0.f, float core_radius = 0.f,
                                    float x_offset = 0.f, float y_offset = 0.f,
                                    const FitRegion& region = {},
                                    float z_out = 0.f, float z_cos_out = 0.f);

// The region weight as the op graph computes it, for the same parity check.
mx::array region_weight_ops(const mx::array& coords, const FitRegion& region);

// Render a low-res RGBA fp16 image (lh, lw, 4): evaluate the MLP over the grid's
// multi-ripple features and append an opaque alpha channel. out_dim must be 3.
// asp sets the grid's x-range to (-asp, asp) so pixels stay square in the window.
mx::array render_pond_lowres(const mx::array& weights, const MLPConfig& cfg,
                             int lh, int lw,
                             const std::vector<RippleSource>& sources,
                             float ring_freq, float decay, float z, float z_cos,
                             float warp = 0.f, float core_radius = 0.f, float asp = 1.f);

}  // namespace mirror
