#include "text_sdf.h"

#include <algorithm>
#include <cmath>

namespace mirror {
namespace {

constexpr float kInf = 1e20f;

// One row of Felzenszwalb & Huttenlocher's transform: the lower envelope of the
// parabolas (q - x)^2 + f[q], evaluated at every x. `v` holds the vertices of
// the envelope's segments and `z` the boundaries between them; both are scratch
// of length n (+1 for z) supplied by the caller so the 2D pass allocates once.
void edt1d(const float* f, float* d, int n, int* v, float* z) {
    int k = 0;
    v[0] = 0;
    z[0] = -kInf;
    z[1] = kInf;
    for (int q = 1; q < n; ++q) {
        // Intersection of the parabola at q with the one currently on top; pop
        // segments until it lands to the right of the last boundary.
        float s = ((f[q] + float(q) * q) - (f[v[k]] + float(v[k]) * v[k])) /
                  (2.f * float(q) - 2.f * float(v[k]));
        while (s <= z[k]) {
            --k;
            s = ((f[q] + float(q) * q) - (f[v[k]] + float(v[k]) * v[k])) /
                (2.f * float(q) - 2.f * float(v[k]));
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = kInf;
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < float(q)) ++k;
        const float dq = float(q) - float(v[k]);
        d[q] = dq * dq + f[v[k]];
    }
}

}  // namespace

void SquaredEDT(std::vector<float>& f, int w, int h) {
    if (w <= 0 || h <= 0) return;
    const int n = std::max(w, h);
    std::vector<float> col(n), out(n), z(size_t(n) + 1);
    std::vector<int> v(n);

    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) col[y] = f[size_t(y) * w + x];
        edt1d(col.data(), out.data(), h, v.data(), z.data());
        for (int y = 0; y < h; ++y) f[size_t(y) * w + x] = out[y];
    }
    for (int y = 0; y < h; ++y) {
        float* row = &f[size_t(y) * w];
        edt1d(row, out.data(), w, v.data(), z.data());
        std::copy(out.begin(), out.begin() + w, row);
    }
}

SdfImage BuildSDF(const unsigned char* cov, int w, int h, float spread) {
    SdfImage img;
    if (!cov || w <= 0 || h <= 0) return img;
    img.w = w;
    img.h = h;
    img.spread = std::max(spread, 1e-3f);

    const size_t n = size_t(w) * h;

    // Two transforms: distance to the nearest inside pixel (nonzero outside the
    // glyph) and distance to the nearest outside pixel (nonzero within it).
    // Their difference is the signed distance. Doing it as a difference rather
    // than as one transform from the boundary is what keeps it exact on both
    // sides without having to extract a contour first.
    std::vector<float> din(n), dout(n);
    for (size_t i = 0; i < n; ++i) {
        const bool inside = cov[i] >= 128;
        din[i]  = inside ? 0.f : kInf;   // seeds inside  -> distance from outside
        dout[i] = inside ? kInf : 0.f;   // seeds outside -> distance from inside
    }
    SquaredEDT(din, w, h);
    SquaredEDT(dout, w, h);

    img.px.resize(n);
    const float inv = 0.5f / img.spread;
    for (size_t i = 0; i < n; ++i) {
        // Positive inside. Both terms are zero-distance on their own side, so
        // this is just whichever one is nonzero.
        const float sd = std::sqrt(dout[i]) - std::sqrt(din[i]);
        float e = 0.5f + sd * inv;
        e = e < 0.f ? 0.f : (e > 1.f ? 1.f : e);
        img.px[i] = static_cast<unsigned char>(std::lround(e * 255.f));
    }
    return img;
}

}  // namespace mirror
