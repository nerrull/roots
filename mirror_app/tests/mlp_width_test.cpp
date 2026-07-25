// The fused MLP kernel at hidden widths other than 64.
//
// mlp_parity_test pins the kernel against a Python fixture at one shape
// (hidden=64). But hidden_dim is a *template parameter*: every distinct value
// compiles a different kernel, with a different simdgroup work decomposition
// (n_blks = (TILE_ROWS/16) * ceil(N/16)) and different threadgroup allocations.
// A width that happens not to divide evenly is exactly where a tiling bug would
// hide, and the fixture cannot see any of it.
//
// So this checks the kernel against a plain MLX matmul chain across widths,
// layer counts and row counts. No fixtures, no Python -- just a Metal device.
// Exit 0 on pass.

#include "mlp_forward.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;

namespace {

int g_fail = 0;

// The kernel matmuls in fp16 with fp32 simdgroup accumulation, then rounds to
// half between layers; the reference below does the same rounding but
// accumulates in fp32 throughout. Six layers of that diverges a little, and
// the output activation is a sigmoid, so errors stay bounded. 6e-3 is well
// under any visible difference in an 8-bit image (1/255 = 3.9e-3 per step)
// while still being tight enough to catch a real indexing fault, which shows
// up as O(1) garbage rather than O(1e-3) drift.
constexpr float kTol = 6e-3f;

float act(float v, mirror::Act a) {
    switch (a) {
        case mirror::Act::Relu:    return std::max(v, 0.f);
        case mirror::Act::Tanh:    return std::tanh(v);
        case mirror::Act::Sigmoid: return 1.f / (1.f + std::exp(-v));
        default:                   return v;
    }
}

mx::array apply_act(const mx::array& x, mirror::Act a) {
    switch (a) {
        case mirror::Act::Relu:    return mx::maximum(x, mx::array(0.f));
        case mirror::Act::Tanh:    return mx::tanh(x);
        case mirror::Act::Sigmoid: return mx::sigmoid(x);
        default:                   return x;
    }
}

// Straightforward reference: pad the input, then one matmul + activation per
// layer, rounding to fp16 between layers exactly as the kernel does.
mx::array reference_mlp(mx::array coords, const mx::array& weights,
                        const mirror::MLPConfig& cfg) {
    const int n = coords.shape(0);
    const int ip = cfg.in_dim_pad();
    coords = mx::astype(coords, mx::float16);
    if (coords.shape(1) < ip) {
        coords = mx::concatenate(
            {coords, mx::zeros({n, ip - coords.shape(1)}, mx::float16)}, 1);
    }

    auto x = mx::astype(coords, mx::float32);
    const auto dims = cfg.layer_dims();
    int off = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
        const int K = dims[i].first, Nn = dims[i].second;
        auto w = mx::reshape(mx::slice(weights, {off}, {off + K * Nn}), {K, Nn});
        x = mx::matmul(x, mx::astype(w, mx::float32));
        const bool last = (i + 1 == dims.size());
        x = apply_act(x, last ? cfg.out_activation : cfg.activation);
        // The kernel stores each layer's output as half before the next matmul.
        x = mx::astype(mx::astype(x, mx::float16), mx::float32);
        off += K * Nn;
    }
    return mx::slice(x, {0, 0}, {n, cfg.out_dim});
}

void check(int hidden, int layers, int n, mirror::Act a, mirror::Act oa,
           const char* note) {
    mirror::MLPConfig cfg{8, hidden, 3, layers, a, oa};

    // Deterministic pseudo-random inputs and weights, scaled so activations
    // stay in a sane range rather than saturating every tanh.
    mx::random::seed(1234 + hidden * 31 + layers);
    auto coords = mx::astype(
        mx::random::normal({n, cfg.in_dim}, mx::float32), mx::float16);
    auto w = mx::astype(
        mx::multiply(mx::random::normal({cfg.total_weights()}, mx::float32),
                     mx::array(0.35f)),
        mx::float16);

    auto got = mx::astype(mirror::fused_mlp_forward(coords, w, cfg), mx::float32);
    auto ref = reference_mlp(coords, w, cfg);

    auto d = mx::max(mx::abs(mx::subtract(got, ref)));
    mx::eval(d);
    const float md = d.item<float>();
    const bool ok = std::isfinite(md) && md <= kTol;
    if (!ok) ++g_fail;
    std::printf("  [%s] hidden=%-4d layers=%-2d n=%-8d max|diff|=%.5f  %s\n",
                ok ? " ok " : "FAIL", hidden, layers, n, md, note);
}

}  // namespace

int main() {
    std::printf("mlp_width: fused kernel vs MLX matmul chain\n");

    const int N = 960 * 540;  // the real half-res row count

    // Widths. 64 is what ships; 32 is the candidate. 48 and 80 are deliberately
    // *not* multiples of the 16-wide simdgroup tile, which is where an
    // off-by-one in the partial-block path would surface.
    check(64, 6, N, mirror::Act::Tanh, mirror::Act::Sigmoid, "shipping");
    check(32, 6, N, mirror::Act::Tanh, mirror::Act::Sigmoid, "candidate");
    check(16, 6, N, mirror::Act::Tanh, mirror::Act::Sigmoid, "");
    check(48, 6, N, mirror::Act::Tanh, mirror::Act::Sigmoid, "not a 16-multiple");
    check(80, 6, N, mirror::Act::Tanh, mirror::Act::Sigmoid, "widest that fits");

    // Past hidden_dim=80 the quadratic wbuf term blows the 32 KiB threadgroup
    // budget. That must fail as a legible C++ exception up front, not as a
    // Metal pipeline-load abort on the first frame.
    {
        mirror::MLPConfig big{8, 128, 3, 6, mirror::Act::Tanh, mirror::Act::Sigmoid};
        bool threw = false;
        try {
            auto c = mx::zeros({16, 8}, mx::float16);
            auto w = mx::zeros({big.total_weights()}, mx::float16);
            auto r = mirror::fused_mlp_forward(c, w, big);
            mx::eval(r);
        } catch (const std::exception&) {
            threw = true;
        }
        if (!threw) ++g_fail;
        std::printf("  [%s] hidden=128 rejected up front (threadgroup limit)\n",
                    threw ? " ok " : "FAIL");
    }

    // Depths, at the candidate width.
    check(32, 2, N, mirror::Act::Tanh, mirror::Act::Sigmoid, "shallow");
    check(32, 4, N, mirror::Act::Tanh, mirror::Act::Sigmoid, "");
    check(32, 10, N, mirror::Act::Tanh, mirror::Act::Sigmoid, "deep");

    // Row counts that do not fill a tile: the tail rows are masked by `grow < N`
    // and a mistake there corrupts the last partial tile only.
    check(32, 6, 1, mirror::Act::Tanh, mirror::Act::Sigmoid, "single row");
    check(32, 6, 33, mirror::Act::Tanh, mirror::Act::Sigmoid, "tile + 1");
    check(32, 6, 1000, mirror::Act::Tanh, mirror::Act::Sigmoid, "ragged tail");

    // Other activations, since ACT/OUT_ACT are template parameters too.
    check(32, 6, 1000, mirror::Act::Relu, mirror::Act::None, "relu/none");
    check(32, 6, 1000, mirror::Act::Sigmoid, mirror::Act::Tanh, "sigmoid/tanh");

    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
