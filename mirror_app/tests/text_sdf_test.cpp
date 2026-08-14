// text_sdf_test — the distance transform behind the text overlay.
//
// Checked against a shape whose distance is known in closed form (a disc), which
// is the only way to catch the failure this transform actually has: an
// approximate or subtly wrong envelope is still smooth and still monotone, so it
// looks correct in a rendered frame and is only wrong by a fraction of a pixel
// in places. That fraction is what makes a long straight stem wobble.
//
// No Metal and no fonts: this is the part of the overlay that is pure geometry.

#include "text_sdf.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

}  // namespace

int main() {
    // --- a disc, against its analytic distance ------------------------------
    const int w = 129, h = 129;
    const float cx = 64.f, cy = 64.f, R = 32.f;
    std::vector<unsigned char> cov(size_t(w) * h, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dx = float(x) - cx, dy = float(y) - cy;
            cov[size_t(y) * w + x] = (std::sqrt(dx * dx + dy * dy) <= R) ? 255 : 0;
        }
    }

    const float spread = 16.f;
    mirror::SdfImage sdf = mirror::BuildSDF(cov.data(), w, h, spread);
    check(sdf.w == w && sdf.h == h, "field keeps the bitmap's dimensions");
    check(sdf.px.size() == size_t(w) * h, "field is one byte per texel");

    // Only inside the band the encoding can represent, and away from the
    // rasterised boundary itself, where the pixel grid's own quantisation of the
    // circle is the error and not the transform.
    float worst = 0.f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float dx = float(x) - cx, dy = float(y) - cy;
            const float truth = R - std::sqrt(dx * dx + dy * dy);  // + inside
            if (std::fabs(truth) > spread - 1.f || std::fabs(truth) < 1.5f) continue;
            const float got =
                (float(sdf.px[size_t(y) * w + x]) / 255.f - 0.5f) * 2.f * spread;
            worst = std::max(worst, std::fabs(got - truth));
        }
    }
    std::printf("disc: worst error %.3f px\n", worst);
    // One pixel covers the rasterisation of the circle plus the 8-bit
    // quantisation of the encoding (spread/128 = 0.125 px here).
    check(worst < 1.0f, "distances match the analytic disc within a pixel");

    // --- the sign convention ------------------------------------------------
    check(sdf.px[size_t(64) * w + 64] == 255, "deep inside saturates high");
    check(sdf.px[0] == 0, "far outside saturates low");
    // The contour lands at 0.5: the texel just inside the edge is above it and
    // the one just outside is below. The shader's whole edge test is this.
    const int ex = int(cx + R);
    check(sdf.px[size_t(64) * w + (ex - 2)] > 128, "just inside is above 0.5");
    check(sdf.px[size_t(64) * w + (ex + 2)] < 128, "just outside is below 0.5");

    // --- degenerate inputs --------------------------------------------------
    std::vector<unsigned char> empty(size_t(w) * h, 0);
    mirror::SdfImage none = mirror::BuildSDF(empty.data(), w, h, spread);
    bool all_low = true;
    for (unsigned char v : none.px) all_low = all_low && v == 0;
    check(all_low, "an empty bitmap is everywhere outside");

    check(mirror::BuildSDF(nullptr, w, h, spread).px.empty(), "null bitmap");
    check(mirror::BuildSDF(cov.data(), 0, 0, spread).px.empty(), "empty bitmap");

    // A single row and a single column: the 1D pass is where an off-by-one in
    // the envelope's boundaries would hide.
    std::vector<unsigned char> row(size_t(w), 0);
    row[10] = 255;
    mirror::SdfImage r1 = mirror::BuildSDF(row.data(), w, 1, 8.f);
    check(!r1.px.empty(), "a one-row bitmap transforms");
    const float d15 = (float(r1.px[15]) / 255.f - 0.5f) * 2.f * 8.f;
    check(std::fabs(d15 - (-5.f)) < 0.2f, "1D distance is exact");

    if (failures == 0) std::printf("text_sdf_test: OK\n");
    return failures == 0 ? 0 : 1;
}
