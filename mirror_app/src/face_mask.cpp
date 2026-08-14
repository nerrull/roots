// Landmark outline -> training mask. No MediaPipe here: this is pure geometry,
// so it builds and is tested whether or not the tracker backend is present.

#include "face_tracker.h"

#include <algorithm>
#include <cmath>

namespace mirror {

const std::vector<int>& FaceOvalIndices() {
    // MediaPipe's FACEMESH_FACE_OVAL, walked as a single closed loop. The
    // canonical constant is an unordered set of edges; a polygon fill needs
    // them in traversal order, so the loop is written out rather than derived.
    static const std::vector<int> kOval = {
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109,
    };
    return kOval;
}

namespace {

// Even-odd point-in-polygon. The face oval is convex enough that winding rules
// would agree, but even-odd is one line and does not care.
bool inside(const std::vector<float>& px, const std::vector<float>& py,
            float x, float y) {
    bool in = false;
    const size_t n = px.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const bool straddles = (py[i] > y) != (py[j] > y);
        if (!straddles) continue;
        const float t = (y - py[i]) / (py[j] - py[i] + 1e-20f);
        if (x < px[i] + t * (px[j] - px[i])) in = !in;
    }
    return in;
}

}  // namespace

void RasteriseFaceMask(const std::vector<FaceLandmark>& landmarks,
                       const std::vector<int>& indices, int w, int h,
                       int dilate_px, std::vector<unsigned char>& mask) {
    mask.assign(size_t(std::max(w, 0)) * std::max(h, 0), 0);
    if (w <= 0 || h <= 0 || indices.size() < 3) return;

    std::vector<float> px, py;
    px.reserve(indices.size());
    py.reserve(indices.size());
    for (int i : indices) {
        if (i < 0 || i >= int(landmarks.size())) return;   // wrong landmark set
        px.push_back(landmarks[size_t(i)].x * float(w));
        py.push_back(landmarks[size_t(i)].y * float(h));
    }

    // Bound the scan to the polygon: a face is a small part of the frame, and
    // testing every pixel against 36 edges is most of the cost otherwise.
    float x0f = px[0], x1f = px[0], y0f = py[0], y1f = py[0];
    for (size_t i = 1; i < px.size(); ++i) {
        x0f = std::min(x0f, px[i]); x1f = std::max(x1f, px[i]);
        y0f = std::min(y0f, py[i]); y1f = std::max(y1f, py[i]);
    }
    const int pad = std::max(dilate_px, 0);
    const int x0 = std::max(0, int(std::floor(x0f)) - pad);
    const int x1 = std::min(w - 1, int(std::ceil(x1f)) + pad);
    const int y0 = std::max(0, int(std::floor(y0f)) - pad);
    const int y1 = std::min(h - 1, int(std::ceil(y1f)) + pad);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            if (inside(px, py, float(x) + 0.5f, float(y) + 0.5f))
                mask[size_t(y) * w + x] = 1;
        }
    }

    if (pad <= 0) return;

    // Dilate by a square structuring element, separably (rows then columns):
    // a true disc would cost a lot more for a difference no training run can
    // see, since the point is only to admit some background near the edge.
    std::vector<unsigned char> tmp(mask.size(), 0);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            unsigned char v = 0;
            const int lo = std::max(x0, x - pad), hi = std::min(x1, x + pad);
            for (int k = lo; k <= hi && !v; ++k) v = mask[size_t(y) * w + k];
            tmp[size_t(y) * w + x] = v;
        }
    }
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            unsigned char v = 0;
            const int lo = std::max(y0, y - pad), hi = std::min(y1, y + pad);
            for (int k = lo; k <= hi && !v; ++k) v = tmp[size_t(k) * w + x];
            mask[size_t(y) * w + x] = v;
        }
    }
}

namespace {

// One row of the squared-distance transform: the lower envelope of the
// parabolas f(q) + (x - q)^2. `v` holds the parabola indices in the envelope and
// `z` the boundaries between them; both are scratch of length n.
void DT1D(const float* f, float* d, int n, int* v, float* z) {
    const float kInf = 1e20f;
    int k = 0;
    v[0] = 0;
    z[0] = -kInf;
    z[1] = kInf;
    for (int q = 1; q < n; ++q) {
        // Where this parabola overtakes the one currently on top; pop while it
        // does so before the top parabola's own start.
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

void DistanceOutside(const std::vector<unsigned char>& mask, int w, int h,
                     std::vector<float>& dist_px) {
    dist_px.assign(size_t(std::max(w, 0)) * std::max(h, 0), 0.f);
    if (w <= 0 || h <= 0 || mask.size() != size_t(w) * h) return;

    const float kInf = 1e20f;
    // Seed: zero on the mask, infinity elsewhere. An empty mask has no nearest
    // pixel at all, so it stays infinite -- callers read that as "no region",
    // which is the right answer when nobody is in frame.
    std::vector<float> f(size_t(w) * h);
    bool any = false;
    for (size_t i = 0; i < f.size(); ++i) {
        const bool on = mask[i] != 0;
        any = any || on;
        f[i] = on ? 0.f : kInf;
    }
    if (!any) { dist_px.assign(size_t(w) * h, kInf); return; }

    const int n = std::max(w, h);
    std::vector<float> buf(static_cast<size_t>(n)), out(static_cast<size_t>(n));
    std::vector<float> z(static_cast<size_t>(n) + 1);
    std::vector<int> v(static_cast<size_t>(n));

    for (int x = 0; x < w; ++x) {                      // columns
        for (int y = 0; y < h; ++y) buf[size_t(y)] = f[size_t(y) * w + x];
        DT1D(buf.data(), out.data(), h, v.data(), z.data());
        for (int y = 0; y < h; ++y) f[size_t(y) * w + x] = out[size_t(y)];
    }
    for (int y = 0; y < h; ++y) {                      // rows
        DT1D(&f[size_t(y) * w], out.data(), w, v.data(), z.data());
        for (int x = 0; x < w; ++x)
            dist_px[size_t(y) * w + x] = std::sqrt(out[size_t(x)]);
    }
}

void RasteriseBox(float cx, float cy, float hx, float hy, int w, int h,
                  int pad_px, std::vector<unsigned char>& mask) {
    mask.assign(size_t(std::max(w, 0)) * std::max(h, 0), 0);
    if (w <= 0 || h <= 0 || hx <= 0.f || hy <= 0.f) return;

    const int pad = std::max(pad_px, 0);
    const int x0 = std::max(0, int(std::floor((cx - hx) * float(w))) - pad);
    const int x1 = std::min(w - 1, int(std::ceil((cx + hx) * float(w))) + pad);
    const int y0 = std::max(0, int(std::floor((cy - hy) * float(h))) - pad);
    const int y1 = std::min(h - 1, int(std::ceil((cy + hy) * float(h))) + pad);
    if (x1 < x0 || y1 < y0) return;   // entirely off-frame

    for (int y = y0; y <= y1; ++y)
        std::fill(mask.begin() + size_t(y) * w + x0,
                  mask.begin() + size_t(y) * w + x1 + 1, (unsigned char)1);
}

void RasteriseFaceBox(const std::vector<FaceLandmark>& landmarks, int w, int h,
                      float pad_frac, int pad_px,
                      std::vector<unsigned char>& mask) {
    mask.assign(size_t(std::max(w, 0)) * std::max(h, 0), 0);
    if (w <= 0 || h <= 0 || landmarks.empty()) return;

    float x0f = landmarks[0].x, x1f = x0f;
    float y0f = landmarks[0].y, y1f = y0f;
    for (const FaceLandmark& L : landmarks) {
        x0f = std::min(x0f, L.x); x1f = std::max(x1f, L.x);
        y0f = std::min(y0f, L.y); y1f = std::max(y1f, L.y);
    }
    // Padded in normalised space so the margin tracks the face's size, then
    // taken to pixels once. Doing it the other way round would give a distant
    // face the same absolute margin as a near one.
    const float pf = 1.f + std::max(pad_frac, 0.f);
    RasteriseBox(0.5f * (x0f + x1f), 0.5f * (y0f + y1f),
                 0.5f * (x1f - x0f) * pf, 0.5f * (y1f - y0f) * pf,
                 w, h, pad_px, mask);
}

}  // namespace mirror
