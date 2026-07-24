// mlp_parity_test — verifies the C++/MLX fused-MLP port matches the Python
// reference on fixtures dumped by dump_fixtures.py, within fp16 tolerance.
//
// Usage:  mlp_parity_test <fixtures_dir>
// Exit 0 on pass, 1 on failure. No window / GPU display needed.
#include "mlp_forward.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace mx = mlx::core;

static std::vector<float> read_f32(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::vector<float> v(count);
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
    if (!f) { fprintf(stderr, "short read on %s\n", path.c_str()); std::exit(2); }
    return v;
}

static std::map<std::string, int> read_meta(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::map<std::string, int> m;
    std::string k; int v;
    while (f >> k >> v) m[k] = v;
    return m;
}

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "fixtures";
    if (!dir.empty() && dir.back() != '/') dir += '/';

    auto meta = read_meta(dir + "meta.txt");
    const int n = meta["n"];
    const int in_dim = meta["in_dim"];
    const int out_dim = meta["out_dim"];
    const int total_w = meta["total_weights"];

    mirror::MLPConfig cfg;
    cfg.in_dim = in_dim;
    cfg.hidden_dim = meta["hidden_dim"];
    cfg.out_dim = out_dim;
    cfg.num_layers = meta["num_layers"];
    cfg.activation = static_cast<mirror::Act>(meta["act"]);
    cfg.out_activation = static_cast<mirror::Act>(meta["out_act"]);

    auto coords_v = read_f32(dir + "coords.f32", (size_t)n * in_dim);
    auto weights_v = read_f32(dir + "weights.f32", (size_t)total_w);
    auto expected_v = read_f32(dir + "expected.f32", (size_t)n * out_dim);

    auto coords = mx::array(coords_v.data(), {n, in_dim}, mx::float32);
    auto weights = mx::array(weights_v.data(), {total_w}, mx::float32);

    auto out = mirror::fused_mlp_forward(coords, weights, cfg);
    out = mx::astype(out, mx::float32);
    mx::eval(out);

    // fp16 tolerance mirrors the Python tests: |a-b| <= atol + rtol*|b|.
    const float atol = 2e-2f, rtol = 1e-2f;
    const float* got = out.data<float>();
    float max_abs = 0.0f; int fails = 0;
    for (size_t i = 0; i < expected_v.size(); ++i) {
        float e = expected_v[i], g = got[i];
        float d = std::fabs(g - e);
        max_abs = std::max(max_abs, d);
        if (d > atol + rtol * std::fabs(e)) ++fails;
    }
    printf("mlp_parity: n=%d out_dim=%d  max_abs_err=%.4e  fails=%d/%zu\n",
           n, out_dim, max_abs, fails, expected_v.size());
    if (fails > 0) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
