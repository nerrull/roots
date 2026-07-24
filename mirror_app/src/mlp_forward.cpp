#include "mlp_forward.h"

#include <cmath>
#include <stdexcept>

namespace mirror {

namespace {

// ---------------------------------------------------------------------------
// MSL kernel source, reused VERBATIM from neuromirror/mlx_fused_mlp/forward.py.
// Keep these byte-for-byte in sync with the Python strings (_HEADER, _DECLS,
// _LOAD_INPUT, _ENCODE_INPUT, _GEMM_OUT). Template params are substituted by MLX.
// ---------------------------------------------------------------------------

const char* kHeader = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float fused_apply_act(float v, uint act) {
    switch (act) {
        case 0: return max(v, 0.0f);        // relu
        case 1: return precise::tanh(v);    // tanh
        case 2: return 1.0f / (1.0f + precise::exp(-v)); // sigmoid
        default: return v;                  // identity / none
    }
}
)MSL";

const char* kDecls = R"MSL(
    const uint tid = thread_index_in_threadgroup;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint tile = threadgroup_position_in_grid.y;
    const uint base_row = tile * TILE_ROWS;
    const uint N = nrows[0];

    threadgroup half bufA[TILE_ROWS * HIDDEN];
    threadgroup half bufB[TILE_ROWS * HIDDEN];
    threadgroup half wbuf[HIDDEN * HIDDEN];
    threadgroup float scratch[SIMDGROUPS * 256];  // one 16x16 float tile / simdgroup
)MSL";

const char* kLoadInput = R"MSL(
    for (uint idx = tid; idx < TILE_ROWS * IN_DIM_PAD; idx += THREADS) {
        uint r = idx / IN_DIM_PAD;
        uint c = idx % IN_DIM_PAD;
        uint grow = base_row + r;
        bufA[r * HIDDEN + c] = (grow < N) ? coords[grow * IN_DIM_PAD + c] : (half)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
)MSL";

const char* kEncodeInput = R"MSL(
    for (uint i = tid; i < TILE_ROWS * HIDDEN; i += THREADS) bufA[i] = (half)0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < TILE_ROWS * NF; idx += THREADS) {
        uint r = idx / NF;
        uint fi = idx % NF;
        uint grow = base_row + r;
        float x = 0.0f, y = 0.0f;
        if (grow < N) { x = coords[grow * 2 + 0]; y = coords[grow * 2 + 1]; }
        float f = exp2((float)fi) * M_PI_F;
        float fx = f * x, fy = f * y;
        uint c = fi * 4;
        bufA[r * HIDDEN + c + 0] = (half)precise::sin(fx);
        bufA[r * HIDDEN + c + 1] = (half)precise::sin(fy);
        bufA[r * HIDDEN + c + 2] = (half)precise::cos(fx);
        bufA[r * HIDDEN + c + 3] = (half)precise::cos(fy);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
)MSL";

const char* kGemmOut = R"MSL(
    threadgroup half* cur = bufA;
    threadgroup half* nxt = bufB;
    uint woff = 0;

    for (uint layer = 0; layer < NUM_LAYERS; ++layer) {
        uint K  = (layer == 0)              ? IN_DIM_PAD  : HIDDEN;
        uint Nn = (layer == NUM_LAYERS - 1) ? OUT_DIM_PAD : HIDDEN;
        uint act = (layer == NUM_LAYERS - 1) ? OUT_ACT : ACT;

        // Stream this layer's weights (K x Nn, row-major) into wbuf.
        for (uint idx = tid; idx < K * Nn; idx += THREADS) {
            wbuf[idx] = weights[woff + idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint n_col_blk = (Nn + 15) / 16;
        uint n_blks = (TILE_ROWS / 16) * n_col_blk;
        for (uint t = sg; t < n_blks; t += SIMDGROUPS) {
            uint mb = t / n_col_blk;
            uint nb = t % n_col_blk;
            uint col0 = nb * 16;
            bool full = (col0 + 16 <= Nn);   // is there a right 8-col pair?
            simdgroup_float8x8 a00 = simdgroup_float8x8(0), a10 = simdgroup_float8x8(0);
            simdgroup_float8x8 a01 = simdgroup_float8x8(0), a11 = simdgroup_float8x8(0);
            for (uint ki = 0; ki < K / 8; ++ki) {
                simdgroup_half8x8 x0, x1, y0;
                simdgroup_load(x0, cur + (mb * 16 + 0) * HIDDEN + ki * 8, HIDDEN);
                simdgroup_load(x1, cur + (mb * 16 + 8) * HIDDEN + ki * 8, HIDDEN);
                simdgroup_load(y0, wbuf + ki * 8 * Nn + col0, Nn);
                simdgroup_multiply_accumulate(a00, x0, y0, a00);
                simdgroup_multiply_accumulate(a10, x1, y0, a10);
                if (full) {
                    simdgroup_half8x8 y1;
                    simdgroup_load(y1, wbuf + ki * 8 * Nn + col0 + 8, Nn);
                    simdgroup_multiply_accumulate(a01, x0, y1, a01);
                    simdgroup_multiply_accumulate(a11, x1, y1, a11);
                }
            }
            threadgroup float* sc = scratch + sg * 256;  // 16x16, ld = 16
            simdgroup_store(a00, sc + 0, 16);
            simdgroup_store(a10, sc + 8 * 16, 16);
            if (full) {
                simdgroup_store(a01, sc + 8, 16);
                simdgroup_store(a11, sc + 8 * 16 + 8, 16);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // Apply activation + convert to half, writing only valid columns.
            uint colw = full ? 16u : 8u;
            for (uint e = lane; e < 16 * colw; e += 32) {
                uint rr = e / colw;
                uint cc = e % colw;
                float v = fused_apply_act(sc[rr * 16 + cc], act);
                nxt[(mb * 16 + rr) * HIDDEN + (col0 + cc)] = (half)v;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half* tmp = cur; cur = nxt; nxt = tmp;
        woff += K * Nn;
    }

    for (uint idx = tid; idx < TILE_ROWS * OUT_DIM; idx += THREADS) {
        uint r = idx / OUT_DIM;
        uint c = idx % OUT_DIM;
        uint grow = base_row + r;
        if (grow < N) {
            out[grow * OUT_DIM + c] = cur[r * HIDDEN + c];
        }
    }
)MSL";

std::string source_plain()   { return std::string(kDecls) + kLoadInput   + kGemmOut; }
std::string source_fourier() { return std::string(kDecls) + kEncodeInput + kGemmOut; }

// Cached kernel builders (analogue of the @lru_cache _kernel() functions).
const mx::fast::CustomKernelFunction& kernel_plain() {
    static mx::fast::CustomKernelFunction fn = mx::fast::metal_kernel(
        "fused_mlp_forward", {"coords", "weights", "nrows"}, {"out"},
        source_plain(), kHeader, /*ensure_row_contiguous=*/true);
    return fn;
}

const mx::fast::CustomKernelFunction& kernel_fourier() {
    static mx::fast::CustomKernelFunction fn = mx::fast::metal_kernel(
        "fused_mlp_forward_fourier", {"coords", "weights", "nrows"}, {"out"},
        source_fourier(), kHeader, /*ensure_row_contiguous=*/true);
    return fn;
}

using TArg = std::pair<std::string, mx::fast::TemplateArg>;

std::vector<TArg> base_template(const MLPConfig& cfg) {
    return {
        {"HIDDEN", cfg.hidden_dim},
        {"NUM_LAYERS", cfg.num_layers},
        {"IN_DIM_PAD", cfg.in_dim_pad()},
        {"OUT_DIM_PAD", cfg.out_dim_pad()},
        {"OUT_DIM", cfg.out_dim},
        {"ACT", cfg.act_code()},
        {"OUT_ACT", cfg.out_act_code()},
        {"TILE_ROWS", kTileRows},
        {"SIMDGROUPS", kSimdGroups},
        {"THREADS", kThreads},
    };
}

}  // namespace

std::vector<std::pair<int, int>> MLPConfig::layer_dims() const {
    const int ip = in_dim_pad(), op = out_dim_pad(), h = hidden_dim;
    if (num_layers == 1) return {{ip, op}};
    std::vector<std::pair<int, int>> dims;
    dims.emplace_back(ip, h);
    for (int i = 0; i < num_layers - 2; ++i) dims.emplace_back(h, h);
    dims.emplace_back(h, op);
    return dims;
}

int MLPConfig::total_weights() const {
    int acc = 0;
    for (auto& kn : layer_dims()) acc += kn.first * kn.second;
    return acc;
}

mx::array fused_mlp_forward(mx::array coords, mx::array weights, const MLPConfig& cfg) {
    if (kTileRows % 16 != 0) throw std::runtime_error("TILE_ROWS must be a multiple of 16");
    if (coords.ndim() != 2) throw std::runtime_error("coords must be 2D");
    const int n = coords.shape(0);
    const int in_cols = coords.shape(1);
    if (in_cols != cfg.in_dim && in_cols != cfg.in_dim_pad())
        throw std::runtime_error("coords have wrong number of columns");

    coords = mx::astype(coords, mx::float16);
    if (in_cols != cfg.in_dim_pad()) {
        auto pad = mx::zeros({n, cfg.in_dim_pad() - in_cols}, mx::float16);
        coords = mx::concatenate({coords, pad}, 1);
    }
    coords = mx::contiguous(coords);

    weights = mx::contiguous(mx::astype(weights, mx::float16));
    if (weights.size() != static_cast<size_t>(cfg.total_weights()))
        throw std::runtime_error("weights have wrong number of elements");

    if (n == 0) return mx::zeros({0, cfg.out_dim}, mx::float16);

    auto nrows = mx::array({n}, mx::uint32);
    const int num_tiles = (n + kTileRows - 1) / kTileRows;

    auto outs = kernel_plain()(
        {coords, weights, nrows},
        {mx::Shape{n, cfg.out_dim}}, {mx::float16},
        std::make_tuple(kThreads, num_tiles, 1),
        std::make_tuple(kThreads, 1, 1),
        base_template(cfg), std::nullopt, /*verbose=*/false, {});
    return outs[0];
}

mx::array fused_mlp_forward_fourier(mx::array coords, mx::array weights,
                                    const MLPConfig& cfg, int n_freqs) {
    if (cfg.in_dim != 4 * n_freqs)
        throw std::runtime_error("cfg.in_dim must equal 4*n_freqs");
    if (cfg.in_dim_pad() > cfg.hidden_dim)
        throw std::runtime_error("fused Fourier needs in_dim_pad <= hidden");
    if (coords.ndim() != 2 || coords.shape(1) != 2)
        throw std::runtime_error("coords must be (N, 2) raw xy");
    const int n = coords.shape(0);
    if (n == 0) return mx::zeros({0, cfg.out_dim}, mx::float16);

    coords = mx::contiguous(mx::astype(coords, mx::float32));
    weights = mx::contiguous(mx::astype(weights, mx::float16));
    if (weights.size() != static_cast<size_t>(cfg.total_weights()))
        throw std::runtime_error("weights have wrong number of elements");

    auto nrows = mx::array({n}, mx::uint32);
    const int num_tiles = (n + kTileRows - 1) / kTileRows;
    auto tmpl = base_template(cfg);
    tmpl.push_back({"NF", n_freqs});

    auto outs = kernel_fourier()(
        {coords, weights, nrows},
        {mx::Shape{n, cfg.out_dim}}, {mx::float16},
        std::make_tuple(kThreads, num_tiles, 1),
        std::make_tuple(kThreads, 1, 1),
        tmpl, std::nullopt, /*verbose=*/false, {});
    return outs[0];
}

}  // namespace mirror
