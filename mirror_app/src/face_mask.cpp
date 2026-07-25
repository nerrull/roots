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

}  // namespace mirror
