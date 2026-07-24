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
inline constexpr int kTileRows   = 32;
inline constexpr int kSimdGroups = 8;
inline constexpr int kThreads    = 32 * kSimdGroups;

// Activation codes — kept in sync with fused_apply_act in the MSL header.
enum class Act : int { Relu = 0, Tanh = 1, Sigmoid = 2, None = 3 };

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

    static int pad_dim(int d, int mult = 8) { return ((d + mult - 1) / mult) * mult; }

    int in_dim_pad()  const { return pad_dim(in_dim); }
    int out_dim_pad() const { return pad_dim(out_dim); }
    int act_code()     const { return static_cast<int>(activation); }
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
