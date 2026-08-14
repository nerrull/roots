#include "fit_target.h"

#include <algorithm>
#include <cmath>

namespace mirror {

namespace {

// The box filter itself. Two callers want the same resampling with different
// output types -- the fit target wants floats in [0,1], MediaPipe wants bytes
// -- and having them share this is what keeps the two from drifting apart and
// giving the tracker a subtly different image from the one being fitted.
template <typename T, typename Store>
void BoxDownsample(const unsigned char* src, int src_w, int src_h,
                   int stride_px, int r_off, int b_off,
                   int dst_w, int dst_h, std::vector<T>& dst, Store store) {
    dst.assign(size_t(dst_w) * dst_h * 3, T(0));
    if (!src || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;

    const int g_off = (r_off + b_off) / 2;   // green sits between them either way

    for (int dy = 0; dy < dst_h; ++dy) {
        // Source rows covered by this destination row. Computed as a half-open
        // span so every source pixel lands in exactly one box -- rounding both
        // ends independently would double-count or drop rows.
        const int y0 = int(int64_t(dy) * src_h / dst_h);
        const int y1 = std::max(y0 + 1, int(int64_t(dy + 1) * src_h / dst_h));
        for (int dx = 0; dx < dst_w; ++dx) {
            const int x0 = int(int64_t(dx) * src_w / dst_w);
            const int x1 = std::max(x0 + 1, int(int64_t(dx + 1) * src_w / dst_w));

            uint32_t acc_r = 0, acc_g = 0, acc_b = 0, n = 0;
            for (int y = y0; y < y1; ++y) {
                const unsigned char* row = src + size_t(y) * src_w * stride_px;
                for (int x = x0; x < x1; ++x) {
                    const unsigned char* px = row + size_t(x) * stride_px;
                    acc_r += px[r_off];
                    acc_g += px[g_off];
                    acc_b += px[b_off];
                    ++n;
                }
            }
            if (!n) continue;
            T* o = &dst[(size_t(dy) * dst_w + dx) * 3];
            o[0] = store(float(acc_r) / n);
            o[1] = store(float(acc_g) / n);
            o[2] = store(float(acc_b) / n);
        }
    }
}

}  // namespace

void DownsampleRGB8(const unsigned char* src, int src_w, int src_h,
                    int stride_px, int r_off, int b_off,
                    int dst_w, int dst_h, std::vector<float>& dst) {
    BoxDownsample(src, src_w, src_h, stride_px, r_off, b_off, dst_w, dst_h, dst,
                  [](float v) { return v * (1.f / 255.f); });
}

void DownsampleToRGB8(const unsigned char* src, int src_w, int src_h,
                      int stride_px, int r_off, int b_off,
                      int dst_w, int dst_h, std::vector<unsigned char>& dst) {
    BoxDownsample(src, src_w, src_h, stride_px, r_off, b_off, dst_w, dst_h, dst,
                  [](float v) {
                      return (unsigned char)(v < 0.f ? 0.f : (v > 255.f ? 255.f : v) + 0.5f);
                  });
}

void MirrorRGB8(int w, int h, std::vector<unsigned char>& rgb) {
    if (w <= 1 || h <= 0 || rgb.size() != size_t(w) * h * 3) return;
    for (int y = 0; y < h; ++y) {
        unsigned char* row = &rgb[size_t(y) * w * 3];
        for (int x = 0; x < w / 2; ++x) {
            unsigned char* a = row + size_t(x) * 3;
            unsigned char* b = row + size_t(w - 1 - x) * 3;
            for (int c = 0; c < 3; ++c) std::swap(a[c], b[c]);
        }
    }
}

void ShiftRGBF(int w, int h, int dx, int dy, std::vector<float>& rgb) {
    if (w <= 0 || h <= 0 || rgb.size() != size_t(w) * h * 3) return;
    if (dx == 0 && dy == 0) return;

    static std::vector<float> tmp;
    tmp = rgb;
    for (int y = 0; y < h; ++y) {
        const int sy = std::min(h - 1, std::max(0, y - dy));
        const float* srow = &tmp[size_t(sy) * w * 3];
        float* drow = &rgb[size_t(y) * w * 3];
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(w - 1, std::max(0, x - dx));
            const float* s = srow + size_t(sx) * 3;
            float* d = drow + size_t(x) * 3;
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
        }
    }
}

void PlaceRGBF(int w, int h, float src_cx, float src_cy, float scale,
               std::vector<float>& rgb) {
    if (w <= 1 || h <= 1 || rgb.size() != size_t(w) * h * 3) return;
    if (scale <= 1e-4f) return;

    static std::vector<float> tmp;
    tmp = rgb;
    const float inv = 1.f / scale;
    for (int y = 0; y < h; ++y) {
        // Destination normalised -> source normalised. The inverse map, so
        // every destination pixel is written exactly once; forward-mapping
        // would leave holes wherever scale > 1.
        const float v = (float(y) + 0.5f) / float(h);
        const float sv = (v - 0.5f) * inv + src_cy;
        const float fy = sv * float(h) - 0.5f;
        const float cy = std::min(std::max(fy, 0.f), float(h - 1));
        const int y0 = int(cy), y1 = std::min(y0 + 1, h - 1);
        const float ty = cy - float(y0);
        for (int x = 0; x < w; ++x) {
            const float u = (float(x) + 0.5f) / float(w);
            const float su = (u - 0.5f) * inv + src_cx;
            const float fx = su * float(w) - 0.5f;
            const float cx = std::min(std::max(fx, 0.f), float(w - 1));
            const int x0 = int(cx), x1 = std::min(x0 + 1, w - 1);
            const float tx = cx - float(x0);

            float* d = &rgb[(size_t(y) * w + x) * 3];
            for (int c = 0; c < 3; ++c) {
                const float a = tmp[(size_t(y0) * w + x0) * 3 + c];
                const float b = tmp[(size_t(y0) * w + x1) * 3 + c];
                const float e = tmp[(size_t(y1) * w + x0) * 3 + c];
                const float f = tmp[(size_t(y1) * w + x1) * 3 + c];
                d[c] = (a + (b - a) * tx) + ((e + (f - e) * tx) - (a + (b - a) * tx)) * ty;
            }
        }
    }
}

bool StaticFitTarget::poll(int w, int h, std::vector<float>& out) {
    if (w <= 0 || h <= 0 || rgb_.empty()) return false;
    if (w == w_ && h == h_) {
        out = rgb_;
    } else {
        // Nearest-neighbour is fine here: the source was already decoded at a
        // requested size, so this only runs when the fit grid changed mid-session.
        out.assign(size_t(w) * h * 3, 0.f);
        for (int y = 0; y < h; ++y) {
            const int sy = std::min(h_ - 1, int(int64_t(y) * h_ / h));
            for (int x = 0; x < w; ++x) {
                const int sx = std::min(w_ - 1, int(int64_t(x) * w_ / w));
                const float* s = &rgb_[(size_t(sy) * w_ + sx) * 3];
                float* d = &out[(size_t(y) * w + x) * 3];
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
            }
        }
    }
    ++frames_;
    return true;
}

}  // namespace mirror
