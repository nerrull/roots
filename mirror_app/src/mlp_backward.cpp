// mlp_backward — fused Metal backward for the neural mirror's MLP.
//
// Port of neuromirror/mlx_fused_mlp/training.py's fused backward, extended with
// the hybrid split (see mlp_forward.h). Two passes in one threadgroup:
//   1. re-forward, storing every layer's activations on-chip
//   2. backprop: dz = da * act'(z); dW += a_in^T @ dz (atomic into device);
//      da = dz @ W^T
//
// The one substantive change over the reference: **sine layers store act'(z)
// during pass 1**. The reference derives act'(z) from the post-activation
// (tanh: 1-a^2), which is impossible for sine because cos(z) is not recoverable
// from sin(z). Storing it costs one threadgroup buffer sized by `split` rather
// than by depth, which is what keeps a trainable sine network inside the 32 KiB
// budget at hidden_dim=64 (30720 bytes, vs 43008 if every layer stored one).
//
// dweights accumulates through atomic<float> across threadgroups, so it is
// deterministic only to float atomic-ordering (ULP level) -- same caveat as the
// reference.

#include "mlp_backward.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mirror {

namespace {

const char* kBwHeader = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// act'(z) from the POST-activation a = act(z). Cheap, and sufficient for every
// activation whose derivative is expressible in its own output.
inline float fused_dact(half a, uint act) {
    switch (act) {
        case 0: return a > (half)0 ? 1.0f : 0.0f;           // relu
        case 1: return 1.0f - (float)a * (float)a;          // tanh
        case 2: return (float)a * (1.0f - (float)a);        // sigmoid
        default: return 1.0f;                               // identity
    }
}

// act'(z) from the PRE-activation z. Sine needs this: cos(z) cannot be
// recovered from sin(z) (the sign is ambiguous), which is why the sine layers
// store their derivative during the re-forward instead of deriving it later.
inline half fused_dact_pre(float v, uint act) {
    switch (act) {
        case 0: return v > 0.0f ? (half)1 : (half)0;
        case 1: { float t = precise::tanh(v); return (half)(1.0f - t * t); }
        case 2: { float s = 1.0f / (1.0f + precise::exp(-v)); return (half)(s * (1.0f - s)); }
        case 4: return (half)precise::cos(v);               // sine
        default: return (half)1;
    }
}

inline half fused_act(float v, uint act) {
    switch (act) {
        case 0: return (half)max(v, 0.0f);
        case 1: return (half)precise::tanh(v);
        case 2: return (half)(1.0f / (1.0f + precise::exp(-v)));
        case 4: return (half)precise::sin(v);
        default: return (half)v;
    }
}
)MSL";

const char* kBwSource = R"MSL(
    const uint tid = thread_index_in_threadgroup;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint tile = threadgroup_position_in_grid.y;
    const uint base = tile * TR;
    const uint N = nrows[0];

    threadgroup half acts[(NUM_LAYERS + 1) * TR * HIDDEN];
    threadgroup half dacts[(SPLIT > 0 ? SPLIT : 1) * TR * HIDDEN];
    threadgroup half wbuf[HIDDEN * HIDDEN];
    threadgroup half dbuf[TR * HIDDEN];
    threadgroup half dzbuf[TR * HIDDEN];
    threadgroup float scr[SG * 64];

    // a_0 = padded input rows (tail rows / pad cols zero-filled).
    for (uint i = tid; i < TR * HIDDEN; i += THREADS) {
        uint r = i / HIDDEN, c = i % HIDDEN; half v = 0;
        if (c < IN_DIM_PAD) { uint g = base + r; v = (g < N) ? coords[g * IN_DIM_PAD + c] : (half)0; }
        acts[r * HIDDEN + c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 1: reforward, storing a_1 .. a_L.
    uint woff = 0;
    for (uint L = 0; L < NUM_LAYERS; ++L) {
        uint K = (L == 0) ? IN_DIM_PAD : HIDDEN;
        uint Nn = (L == NUM_LAYERS - 1) ? OUT_DIM_PAD : HIDDEN;
        uint act = (L == NUM_LAYERS - 1) ? OUT_ACT : (L < SPLIT ? ACT_FIRST : ACT);
        for (uint i = tid; i < K * Nn; i += THREADS) wbuf[i] = weights[woff + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half* ain = acts + L * TR * HIDDEN;
        threadgroup half* aout = acts + (L + 1) * TR * HIDDEN;
        uint nct = Nn / 8, nt = (TR / 8) * nct;
        for (uint t = sg; t < nt; t += SG) {
            uint mi = t / nct, ni = t % nct;
            simdgroup_float8x8 acc = simdgroup_float8x8(0);
            for (uint ki = 0; ki < K / 8; ++ki) {
                simdgroup_half8x8 a, b;
                simdgroup_load(a, ain + mi * 8 * HIDDEN + ki * 8, HIDDEN);
                simdgroup_load(b, wbuf + ki * 8 * Nn + ni * 8, Nn);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, scr + sg * 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                uint rr = e / 8, cc = e % 8;
                float z = scr[sg * 64 + e];
                aout[(mi * 8 + rr) * HIDDEN + (ni * 8 + cc)] = fused_act(z, act);
                if (L < SPLIT) {
                    dacts[L * TR * HIDDEN + (mi * 8 + rr) * HIDDEN + (ni * 8 + cc)]
                        = fused_dact_pre(z, act);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        woff += K * Nn;
    }

    // da = cotangent wrt a_L (padded to OUT_DIM_PAD; tail rows zero).
    for (uint i = tid; i < TR * HIDDEN; i += THREADS) {
        uint r = i / HIDDEN, c = i % HIDDEN; half v = 0;
        if (c < OUT_DIM) { uint g = base + r; v = (g < N) ? cot[g * OUT_DIM + c] : (half)0; }
        dbuf[r * HIDDEN + c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-layer weight offsets (compile-time small loop).
    uint woffs[NUM_LAYERS]; uint acc_off = 0;
    for (uint L = 0; L < NUM_LAYERS; ++L) {
        uint K = (L == 0) ? IN_DIM_PAD : HIDDEN;
        uint Nn = (L == NUM_LAYERS - 1) ? OUT_DIM_PAD : HIDDEN;
        woffs[L] = acc_off; acc_off += K * Nn;
    }

    // Pass 2: backprop.
    for (int L = NUM_LAYERS - 1; L >= 0; --L) {
        uint K = (L == 0) ? IN_DIM_PAD : HIDDEN;
        uint Nn = (L == NUM_LAYERS - 1) ? OUT_DIM_PAD : HIDDEN;
        uint act = (L == NUM_LAYERS - 1) ? OUT_ACT : (L < SPLIT ? ACT_FIRST : ACT);
        for (uint i = tid; i < K * Nn; i += THREADS) wbuf[i] = weights[woffs[L] + i];
        threadgroup half* aout = acts + (L + 1) * TR * HIDDEN;
        threadgroup half* ain = acts + L * TR * HIDDEN;
        // dz = da * act'(a_out).
        for (uint i = tid; i < TR * Nn; i += THREADS) {
            uint r = i / Nn, c = i % Nn;
            float d = (L < SPLIT)
                ? (float)dacts[L * TR * HIDDEN + r * HIDDEN + c]
                : fused_dact(aout[r * HIDDEN + c], act);
            dzbuf[r * HIDDEN + c] = (half)((float)dbuf[r * HIDDEN + c] * d);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dW = a_in^T @ dz  (K x Nn), atomic-accumulate into device.
        uint nct = Nn / 8, nt = (K / 8) * nct;
        for (uint t = sg; t < nt; t += SG) {
            uint kt = t / nct, ni = t % nct;
            simdgroup_float8x8 acc = simdgroup_float8x8(0);
            for (uint r = 0; r < TR / 8; ++r) {
                simdgroup_half8x8 aT, b;
                simdgroup_load(aT, ain + r * 8 * HIDDEN + kt * 8, HIDDEN, 0, true);
                simdgroup_load(b, dzbuf + r * 8 * HIDDEN + ni * 8, HIDDEN);
                simdgroup_multiply_accumulate(acc, aT, b, acc);
            }
            simdgroup_store(acc, scr + sg * 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                uint rr = e / 8, cc = e % 8;
                atomic_fetch_add_explicit(
                    &dweights[woffs[L] + (kt * 8 + rr) * Nn + (ni * 8 + cc)],
                    scr[sg * 64 + e], memory_order_relaxed);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // da_new = dz @ W^T  (TR x K).
        uint kct = K / 8, mt = (TR / 8) * kct;
        for (uint t = sg; t < mt; t += SG) {
            uint mi = t / kct, ki = t % kct;
            simdgroup_float8x8 acc = simdgroup_float8x8(0);
            for (uint ni = 0; ni < Nn / 8; ++ni) {
                simdgroup_half8x8 a, bT;
                simdgroup_load(a, dzbuf + mi * 8 * HIDDEN + ni * 8, HIDDEN);
                simdgroup_load(bT, wbuf + ki * 8 * Nn + ni * 8, Nn, 0, true);
                simdgroup_multiply_accumulate(acc, a, bT, acc);
            }
            simdgroup_store(acc, scr + sg * 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                uint rr = e / 8, cc = e % 8;
                dbuf[(mi * 8 + rr) * HIDDEN + (ki * 8 + cc)] = (half)scr[sg * 64 + e];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // dcoords = da (first IN_DIM_PAD cols), atomic_store (one writer per row).
    for (uint i = tid; i < TR * IN_DIM_PAD; i += THREADS) {
        uint r = i / IN_DIM_PAD, c = i % IN_DIM_PAD; uint g = base + r;
        if (g < N) atomic_store_explicit(&dco[g * IN_DIM_PAD + c],
                                         (float)dbuf[r * HIDDEN + c], memory_order_relaxed);
    }
)MSL";

const mx::fast::CustomKernelFunction& bw_kernel() {
    static mx::fast::CustomKernelFunction fn = mx::fast::metal_kernel(
        "fused_mlp_backward", {"coords", "weights", "cot", "nrows"},
        {"dco", "dweights"}, kBwSource, kBwHeader,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/true);
    return fn;
}

}  // namespace

int fused_mlp_backward_threadgroup_bytes(const MLPConfig& cfg) {
    const int h = cfg.hidden_dim;
    const int acts = (cfg.num_layers + 1) * kBwTileRows * h * 2;
    const int dacts = std::max(cfg.split, 1) * kBwTileRows * h * 2;
    const int wbuf = h * h * 2;
    const int dbuf = 2 * kBwTileRows * h * 2;
    const int scr = kBwSimdGroups * 64 * 4;
    return acts + dacts + wbuf + dbuf + scr;
}

std::pair<mx::array, mx::array> fused_mlp_backward(
    mx::array coords, mx::array weights, mx::array cotangent,
    const MLPConfig& cfg) {
    const int need = fused_mlp_backward_threadgroup_bytes(cfg);
    if (need > kMaxThreadgroupBytes) {
        throw std::runtime_error(
            "fused MLP backward: hidden_dim=" + std::to_string(cfg.hidden_dim) +
            " with split=" + std::to_string(cfg.split) + " needs " +
            std::to_string(need) + " bytes of threadgroup memory, over the " +
            std::to_string(kMaxThreadgroupBytes) + " byte limit");
    }
    if (coords.ndim() != 2) throw std::runtime_error("coords must be 2D");
    const int n = coords.shape(0);
    const int in_cols = coords.shape(1);
    if (in_cols != cfg.in_dim && in_cols != cfg.in_dim_pad())
        throw std::runtime_error("coords have wrong number of columns");

    coords = mx::astype(coords, mx::float16);
    if (in_cols != cfg.in_dim_pad()) {
        coords = mx::concatenate(
            {coords, mx::zeros({n, cfg.in_dim_pad() - in_cols}, mx::float16)}, 1);
    }
    coords = mx::contiguous(coords);
    weights = mx::contiguous(mx::astype(weights, mx::float16));
    if (weights.size() != static_cast<size_t>(cfg.total_weights()))
        throw std::runtime_error("weights have wrong number of elements");
    cotangent = mx::contiguous(mx::astype(cotangent, mx::float16));

    if (n == 0) {
        return {mx::zeros({0, cfg.in_dim}, mx::float32),
                mx::zeros({cfg.total_weights()}, mx::float32)};
    }

    auto nrows = mx::array({n}, mx::uint32);
    const int num_tiles = (n + kBwTileRows - 1) / kBwTileRows;

    std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmpl = {
        {"HIDDEN", cfg.hidden_dim},
        {"NUM_LAYERS", cfg.num_layers},
        {"IN_DIM_PAD", cfg.in_dim_pad()},
        {"OUT_DIM_PAD", cfg.out_dim_pad()},
        {"OUT_DIM", cfg.out_dim},
        {"ACT", cfg.act_code()},
        {"ACT_FIRST", cfg.act_first_code()},
        {"SPLIT", cfg.split},
        {"OUT_ACT", cfg.out_act_code()},
        {"TR", kBwTileRows},
        {"SG", kBwSimdGroups},
        {"THREADS", kBwThreads},
    };

    auto outs = bw_kernel()(
        {coords, weights, cotangent, nrows},
        {mx::Shape{n, cfg.in_dim_pad()}, mx::Shape{cfg.total_weights()}},
        {mx::float32, mx::float32},
        std::make_tuple(kBwThreads, num_tiles, 1),
        std::make_tuple(kBwThreads, 1, 1),
        tmpl, /*init_value=*/0.f, /*verbose=*/false, {});

    auto dcoords = mx::slice(outs[0], {0, 0}, {n, cfg.in_dim});
    return {dcoords, outs[1]};
}

}  // namespace mirror
