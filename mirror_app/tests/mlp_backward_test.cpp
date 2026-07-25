// The fused backward against an independent MLX-op backprop.
//
// A fast wrong gradient is worse than no gradient: training would still appear
// to run, just converge to the wrong thing. And this kernel is exactly the sort
// of code where that happens quietly -- two transposed simdgroup GEMMs, fp16
// storage, and dW accumulated through float atomics across threadgroups.
//
// So it is checked against a plain matmul-chain backprop in fp32, across every
// axis that changes the kernel's behaviour: hidden width, depth, the hybrid
// split (which changes how act'(z) is obtained per layer), activations, and
// ragged row counts that leave a partial tile.
//
// Needs a Metal device; no fixtures. Exit 0 on pass.

#include "mlp_backward.h"
#include "mlp_forward.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace {

int g_fail = 0;

// Gradient magnitudes span orders of magnitude across layers, so the check is
// relative to the largest component rather than absolute. 2% accommodates fp16
// activations plus atomic accumulation order; a real indexing fault produces
// O(1) relative error, not O(1e-2).
constexpr float kTol = 0.02f;

// ReLU is deliberately NOT covered here. It is not a gap in this kernel and not
// something the split introduced -- it is unbounded, so its activations grow
// with depth and the kernel's fp16 inter-layer storage diverges from the fp32
// reference. Measured relative dW error at weight scale 0.3:
//
//     config              err          config              err
//     relu split=0 L=3   0.0103        relu split=0 L=6   0.0506
//     relu split=1 L=3   0.0087        relu split=1 L=6   0.0239
//     tanh split=0 L=3   0.0005        tanh split=0 L=6   0.0011
//
// relu is *worse without* the split than with it, and tanh on the identical
// code paths is 50x tighter, so this is fp16 range in an unbounded activation,
// inherited from the reference design. The mirror uses tanh/sigmoid/sine, all
// bounded. Testing relu here would be measuring float precision, not this
// kernel -- and picking a tolerance that let it pass would hide the fact that
// relu gradients through this kernel are simply not trustworthy at depth.
//
// The related practical consequence, which does apply: large `detail`/
// `contrast` scales degrade gradient accuracy for *every* activation. The same
// sweep at weight scale 2.4 puts tanh at 0.1598.
constexpr float kTolUnbounded = 0.05f;  // unused; kept for the None case
//

mx::array apply_act(const mx::array& z, mirror::Act a) {
    switch (a) {
        case mirror::Act::Relu:    return mx::maximum(z, mx::array(0.f));
        case mirror::Act::Tanh:    return mx::tanh(z);
        case mirror::Act::Sigmoid: return mx::sigmoid(z);
        case mirror::Act::Sine:    return mx::sin(z);
        default:                   return z;
    }
}

// act'(z), from z. Deliberately computed from the pre-activation for *every*
// activation: that is the definition the kernel must match, however it chooses
// to obtain it internally.
mx::array act_grad(const mx::array& z, mirror::Act a) {
    switch (a) {
        case mirror::Act::Relu:
            return mx::astype(mx::greater(z, mx::array(0.f)), mx::float32);
        case mirror::Act::Tanh: {
            auto t = mx::tanh(z);
            return mx::subtract(mx::array(1.f), mx::multiply(t, t));
        }
        case mirror::Act::Sigmoid: {
            auto s = mx::sigmoid(z);
            return mx::multiply(s, mx::subtract(mx::array(1.f), s));
        }
        case mirror::Act::Sine:
            return mx::cos(z);
        default:
            return mx::ones_like(z);
    }
}

std::vector<int> weight_offsets(const mirror::MLPConfig& cfg) {
    std::vector<int> offs;
    int acc = 0;
    for (auto& kn : cfg.layer_dims()) {
        offs.push_back(acc);
        acc += kn.first * kn.second;
    }
    return offs;
}

mirror::Act layer_act(const mirror::MLPConfig& cfg, int i, int last) {
    if (i == last) return cfg.out_activation;
    return (i < cfg.split) ? cfg.act_first : cfg.activation;
}

// Reference backprop in fp32, keeping pre-activations.
std::pair<mx::array, mx::array> reference_backward(mx::array coords,
                                                   const mx::array& weights,
                                                   const mx::array& cot,
                                                   const mirror::MLPConfig& cfg) {
    const int n = coords.shape(0);
    const int ip = cfg.in_dim_pad();
    coords = mx::astype(coords, mx::float16);
    if (coords.shape(1) < ip) {
        coords = mx::concatenate(
            {coords, mx::zeros({n, ip - coords.shape(1)}, mx::float16)}, 1);
    }
    auto w = mx::astype(weights, mx::float32);
    const auto dims = cfg.layer_dims();
    const auto offs = weight_offsets(cfg);
    const int last = int(dims.size()) - 1;

    std::vector<mx::array> acts{mx::astype(coords, mx::float32)};
    std::vector<mx::array> zs;
    for (int i = 0; i <= last; ++i) {
        const int k = dims[i].first, nn = dims[i].second;
        auto wi = mx::reshape(mx::slice(w, {offs[i]}, {offs[i] + k * nn}), {k, nn});
        auto z = mx::matmul(acts.back(), wi);
        zs.push_back(z);
        acts.push_back(apply_act(z, layer_act(cfg, i, last)));
    }

    // Cotangent lands on the first out_dim columns; the padded tail is zero.
    auto da = mx::concatenate(
        {mx::astype(cot, mx::float32),
         mx::zeros({n, acts.back().shape(1) - cfg.out_dim}, mx::float32)}, 1);

    std::vector<mx::array> dW(dims.size(), mx::zeros({1}));
    for (int i = last; i >= 0; --i) {
        auto dz = mx::multiply(da, act_grad(zs[i], layer_act(cfg, i, last)));
        const int k = dims[i].first, nn = dims[i].second;
        auto wi = mx::reshape(mx::slice(w, {offs[i]}, {offs[i] + k * nn}), {k, nn});
        dW[i] = mx::matmul(mx::transpose(acts[i]), dz);
        da = mx::matmul(dz, mx::transpose(wi));
    }
    std::vector<mx::array> flat;
    for (auto& m : dW) flat.push_back(mx::reshape(m, {-1}));
    return {mx::slice(da, {0, 0}, {n, cfg.in_dim}), mx::concatenate(flat)};
}

float rel_err(const mx::array& got, const mx::array& want) {
    auto d = mx::max(mx::abs(mx::subtract(mx::astype(got, mx::float32),
                                          mx::astype(want, mx::float32))));
    auto s = mx::max(mx::abs(mx::astype(want, mx::float32)));
    mx::eval(d, s);
    const float scale = std::max(s.item<float>(), 1e-6f);
    return d.item<float>() / scale;
}

void check(int hidden, int layers, int split, int n, mirror::Act act,
           mirror::Act act_first, const char* note) {
    mirror::MLPConfig cfg{8, hidden, 3, layers, act, mirror::Act::Sigmoid};
    cfg.split = split;
    cfg.act_first = act_first;

    mx::random::seed(77 + hidden + layers * 13 + split * 101 + n);
    auto coords = mx::astype(mx::random::normal({n, cfg.in_dim}), mx::float16);
    auto w = mx::astype(
        mx::multiply(mx::random::normal({cfg.total_weights()}), mx::array(0.3f)),
        mx::float16);
    auto cot = mx::astype(mx::random::normal({n, cfg.out_dim}), mx::float16);

    auto got = mirror::fused_mlp_backward(coords, w, cot, cfg);
    auto want = reference_backward(coords, w, cot, cfg);

    const float rw = rel_err(got.second, want.second);
    const float rc = rel_err(got.first, want.first);
    const bool unbounded = (act == mirror::Act::Relu || act == mirror::Act::None);
    const float tol = unbounded ? kTolUnbounded : kTol;
    const bool ok = std::isfinite(rw) && std::isfinite(rc) && rw <= tol && rc <= tol;
    if (!ok) ++g_fail;
    std::printf("  [%s] h=%-3d L=%-2d split=%d n=%-7d  dW %.4f  dcoords %.4f  %s\n",
                ok ? " ok " : "FAIL", hidden, layers, split, n, rw, rc, note);
}

}  // namespace

int main() {
    std::printf("mlp_backward: fused kernel vs fp32 MLX backprop\n");
    const auto T = mirror::Act::Tanh;
    const auto S = mirror::Act::Sine;
    const int N = 480 * 270;   // the real training row count

    // split = 0: the plain network, every layer using the post-activation path.
    check(32, 6, 0, N, T, S, "all tanh");
    check(64, 6, 0, N, T, S, "all tanh, wide");

    // The hybrid. split >= 1 sends the leading layers down the stored
    // pre-activation derivative path and the rest down the post-activation one,
    // so the boundary between them is the interesting case.
    check(32, 6, 1, N, T, S, "STTTT (shipping hybrid)");
    check(32, 6, 2, N, T, S, "SSTTT");
    check(32, 6, 5, N, T, S, "all sine");
    check(64, 6, 1, N, T, S, "STTTT at width 64");

    // Depth, including a 2-layer net where 'first' and 'last' coincide.
    check(32, 2, 1, 4096, T, S, "shallow");
    check(32, 10, 3, 4096, T, S, "deep");

    // Ragged row counts: the tail tile is masked by (g < N) and a mistake there
    // corrupts only the last partial tile, which a round row count hides.
    check(32, 6, 1, 1, T, S, "single row");
    check(32, 6, 1, 17, T, S, "tile + 1");
    check(32, 6, 1, 1000, T, S, "ragged tail");

    // Other activations behind the sine layer.
    // (see the note on kTolUnbounded for why relu is not in this sweep)
    check(32, 6, 1, 4096, mirror::Act::Sigmoid, S, "sigmoid tail");
    check(32, 6, 1, 4096, T, mirror::Act::Tanh, "no sine (act_first=tanh)");

    // Over-budget configs must throw rather than produce garbage.
    {
        mirror::MLPConfig big{8, 80, 3, 6, T, mirror::Act::Sigmoid};
        big.split = 1;
        bool threw = false;
        try {
            auto c = mx::zeros({16, 8}, mx::float16);
            auto w = mx::zeros({big.total_weights()}, mx::float16);
            auto ct = mx::zeros({16, 3}, mx::float16);
            auto r = mirror::fused_mlp_backward(c, w, ct, big);
            mx::eval(r.first, r.second);
        } catch (const std::exception&) {
            threw = true;
        }
        if (!threw) ++g_fail;
        std::printf("  [%s] hidden=80 rejected (threadgroup limit)\n",
                    threw ? " ok " : "FAIL");
        std::printf("       budgets: h32/split1 %d  h64/split1 %d  h80/split1 %d "
                    "(limit %d)\n",
                    mirror::fused_mlp_backward_threadgroup_bytes(
                        [] { mirror::MLPConfig c{8, 32, 3, 6}; c.split = 1; return c; }()),
                    mirror::fused_mlp_backward_threadgroup_bytes(
                        [] { mirror::MLPConfig c{8, 64, 3, 6}; c.split = 1; return c; }()),
                    mirror::fused_mlp_backward_threadgroup_bytes(big),
                    mirror::kMaxThreadgroupBytes);
    }

    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
