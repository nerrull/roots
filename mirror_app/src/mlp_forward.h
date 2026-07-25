// mlp_forward — C++/MLX port of neuromirror's fused forward MLP.
//
// This is a direct port of neuromirror/mlx_fused_mlp/forward.py + config.py to
// the MLX C++ API (mlx::core::fast::metal_kernel). The Metal Shading Language
// kernel source is reused **verbatim** from forward.py (see mlp_forward.cpp) —
// there is no second, divergent kernel implementation.
//
// Pure C++ (no ObjC): compiled into a .cpp translation unit. MLX runs the kernel
// on the system default MTLDevice, the same device MetalContext uses, so the
// resulting mx::array's backing MTLBuffer can be wrapped as an MTLTexture for the
// render pipeline without a copy.
#pragma once

#include <mlx/mlx.h>
#include <string>
#include <utility>
#include <vector>

namespace mirror {

namespace mx = mlx::core;

// Tile of batch rows per threadgroup / simdgroups per threadgroup. Mirrors the
// FUSED_MLP_TILE_ROWS / FUSED_MLP_SIMDGROUPS env knobs in forward.py; 32/8 was
// measured fastest on M4. THREADS = 32 * SIMDGROUPS.
//
// 32/8 is optimal at hidden_dim=64 (measured: 64/8 -> 17.2 ms and 32/4 -> 18.0
// ms, against 16.2 ms here) and near-optimal below it -- at hidden_dim=32 the
// best retune found was only 6% better (128/8 -> 6.37 ms vs 6.80 ms), which is
// not worth making these per-config.
inline constexpr int kTileRows   = 32;
inline constexpr int kSimdGroups = 8;
inline constexpr int kThreads    = 32 * kSimdGroups;

// Threadgroup memory the kernel needs, in bytes, for a given hidden width:
//   bufA + bufB : 2 * TILE_ROWS * HIDDEN halves
//   wbuf        : HIDDEN * HIDDEN halves
//   scratch     : SIMDGROUPS * 256 floats
constexpr int fused_mlp_threadgroup_bytes(int hidden) {
    return 2 * kTileRows * hidden * 2 + hidden * hidden * 2 + kSimdGroups * 256 * 4;
}

// Apple GPUs allow 32 KiB of threadgroup memory per threadgroup. The wbuf term
// is quadratic in hidden_dim, so this is a hard ceiling on how wide the network
// can be with this kernel: **hidden_dim <= 80**. Past that, Metal refuses to
// load the pipeline at runtime ("Threadgroup memory size (57344) exceeds the
// maximum allowed (32768)" for hidden_dim=128), so fused_mlp_forward checks it
// up front and throws something legible instead.
inline constexpr int kMaxThreadgroupBytes = 32768;

// Activation codes — kept in sync with fused_apply_act in the MSL header.
// Sine is SIREN's periodic activation. NOTE: the *forward* supports it, but the
// fused backward cannot -- it reconstructs act'(z) from the post-activation
// (tanh: 1-a^2, sigmoid: a(1-a)), and cos(z) is not recoverable from sin(z).
// Training with Sine therefore needs a backward that keeps pre-activations.
enum class Act : int { Relu = 0, Tanh = 1, Sigmoid = 2, None = 3, Sine = 4 };

// Shape + activation description of one fused MLP. Port of config.MLPConfig.
// Weight layout (flat, contiguous, fp16, row-major per matrix):
//   [W0 (in_dim_pad x hidden) | inner (hidden x hidden)... | W_last (hidden x out_dim_pad)]
struct MLPConfig {
    int in_dim;
    int hidden_dim;
    int out_dim;
    int num_layers;
    Act activation      = Act::Tanh;
    Act out_activation  = Act::Sigmoid;

    // Hybrid activation: the first `split` hidden layers use `act_first`, the
    // rest use `activation`. split = 0 is the plain single-activation network
    // and is bit-identical to what shipped before this existed.
    //
    // The motivating case is one leading sine layer (SIREN) followed by tanh:
    // the sine layer builds a high-frequency basis, which is what lets a
    // coordinate MLP resolve fine detail at all, while the tanh layers behind
    // it stay responsive to the detail/contrast weight scales. Measured on a
    // face fit, one sine layer is as good as five (loss 0.00250 vs 0.00273) and
    // 2.2x better than none (0.00562).
    int split           = 0;
    Act act_first       = Act::Sine;

    static int pad_dim(int d, int mult = 8) { return ((d + mult - 1) / mult) * mult; }

    int in_dim_pad()  const { return pad_dim(in_dim); }
    int out_dim_pad() const { return pad_dim(out_dim); }
    int act_code()       const { return static_cast<int>(activation); }
    int act_first_code() const { return static_cast<int>(act_first); }
    int out_act_code() const { return static_cast<int>(out_activation); }

    // (K_i, N_i) input/output width of each layer's weight matrix.
    std::vector<std::pair<int, int>> layer_dims() const;
    int total_weights() const;
};

// Run the fused forward MLP. Returns (N, out_dim) fp16 (padding stripped).
// coords: (N, in_dim) or (N, in_dim_pad); cast to fp16 internally.
// weights: flat fp16 buffer matching `cfg` layout.
mx::array fused_mlp_forward(mx::array coords, mx::array weights, const MLPConfig& cfg);

// Forward MLP with the axis-aligned Fourier encoding fused into the kernel.
// Takes raw (N, 2) coords (read as fp32 on-chip). cfg.in_dim must equal 4*n_freqs
// and in_dim_pad() <= hidden_dim.
mx::array fused_mlp_forward_fourier(mx::array coords, mx::array weights,
                                    const MLPConfig& cfg, int n_freqs);

}  // namespace mirror
