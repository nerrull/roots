// pond_parity_test — checks the C++ Pond render against a Python demo_panel
// fixture (dump_pond_fixture.py), with drops=0 + orbit so the comparison is
// deterministic. Exit 0 on pass. u8 tolerance absorbs fp16 rounding at edges.
#include "pond_state.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace mx = mirror::mx;

static std::map<std::string, double> read_meta(const std::string& p) {
    std::ifstream f(p);
    std::map<std::string, double> m;
    std::string k; double v;
    while (f >> k >> v) m[k] = v;
    return m;
}

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "fixtures";
    if (!dir.empty() && dir.back() != '/') dir += '/';

    auto meta = read_meta(dir + "pond_u8.meta");
    const int lh = (int)meta["lh"], lw = (int)meta["lw"];
    const double t = meta["t"], z = meta["z"];

    std::vector<uint8_t> ref((size_t)lh * lw * 3);
    { std::ifstream f(dir + "pond_u8.bin", std::ios::binary);
      f.read(reinterpret_cast<char*>(ref.data()), ref.size());
      if (!f) { fprintf(stderr, "cannot read pond_u8.bin\n"); return 2; } }

    mirror::Pond pond(11);
    mirror::PondParams p;      // defaults match PondState.__init__
    p.drops_on = false;   // the fixture is orbit-only, and drops are not deterministic
    p.orbit_on = true;
    p.z = (float)z;

    auto img = pond.render(lh, lw, t, p);     // (lh, lw, 3) fp32 [0,1]
    img = mx::contiguous(img);
    mx::eval(img);
    const float* got = img.data<float>();

    // image_to_u8 truncates: (uint8)(clip(v,0,1)*255).
    int max_abs = 0; long over1 = 0, over2 = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        int g = (int)(std::min(std::max(got[i], 0.f), 1.f) * 255.0f);
        int d = std::abs(g - (int)ref[i]);
        max_abs = std::max(max_abs, d);
        if (d > 1) ++over1;
        if (d > 2) ++over2;
    }
    double pct1 = 100.0 * over1 / ref.size();
    printf("pond_parity: %dx%d  max|d|=%d  >1: %.3f%%  >2: %.4f%%\n",
           lh, lw, max_abs, pct1, 100.0 * over2 / ref.size());
    // fp16 render + u8 truncation: allow a handful of off-by-±1/±2 at boundaries.
    if (over2 > (long)(0.001 * ref.size())) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
