#include "fit_target.h"

#include <algorithm>
#include <cmath>

namespace mirror {

void DownsampleRGB8(const unsigned char* src, int src_w, int src_h,
                    int stride_px, int r_off, int b_off,
                    int dst_w, int dst_h, std::vector<float>& dst) {
    dst.assign(size_t(dst_w) * dst_h * 3, 0.f);
    if (!src || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;

    const int g_off = (r_off + b_off) / 2;   // green sits between them either way
    const float inv255 = 1.f / 255.f;

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
            float* o = &dst[(size_t(dy) * dst_w + dx) * 3];
            o[0] = float(acc_r) / n * inv255;
            o[1] = float(acc_g) / n * inv255;
            o[2] = float(acc_b) / n * inv255;
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
