// Landmark outline -> training mask.
//
// This is the geometry that decides which pixels the gradient sees, so it is
// worth testing independently of MediaPipe: a mask that is subtly wrong
// (mirrored, off by a scale factor, inside-out) still produces a plausible
// training run that fits the wrong part of the frame.
//
// Uses synthetic landmark rings rather than real detections, so it needs
// neither the tracker backend nor a face.
//
// Exit 0 on pass.

#include "face_tracker.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++g_fail;
}

// A circle of landmarks at (cx, cy) with radius r, all in normalised space.
// Indices 0..n-1 are the ring; the array is padded to `total` so index-based
// outlines (like FaceOvalIndices) can be exercised against a full-size set.
std::vector<mirror::FaceLandmark> ring(int n, float cx, float cy, float r,
                                       int total = 0) {
    std::vector<mirror::FaceLandmark> lm(std::max(n, total));
    for (int i = 0; i < n; ++i) {
        const float a = 2.f * float(M_PI) * i / n;
        lm[size_t(i)].x = cx + r * std::cos(a);
        lm[size_t(i)].y = cy + r * std::sin(a);
    }
    return lm;
}

size_t count(const std::vector<unsigned char>& m) {
    size_t n = 0;
    for (unsigned char v : m) n += v ? 1 : 0;
    return n;
}

void test_basic() {
    std::printf("\nrasterisation\n");
    const int W = 200, H = 200;
    std::vector<int> idx;
    for (int i = 0; i < 24; ++i) idx.push_back(i);
    auto lm = ring(24, 0.5f, 0.5f, 0.25f);

    std::vector<unsigned char> mask;
    mirror::RasteriseFaceMask(lm, idx, W, H, 0, mask);
    check(mask.size() == size_t(W) * H, "mask is w*h");

    // A 24-gon of radius 0.25 covers ~pi*r^2 of the frame; the polygon is
    // slightly inside the circle, so allow a little under.
    const double frac = double(count(mask)) / (W * H);
    const double want = M_PI * 0.25 * 0.25;
    std::printf("       covers %.4f of frame (circle would be %.4f)\n", frac, want);
    check(frac > want * 0.95 && frac < want * 1.02, "area matches the polygon");

    check(mask[size_t(H / 2) * W + W / 2] != 0, "centre is inside");
    check(mask[0] == 0 && mask[mask.size() - 1] == 0, "corners are outside");

    // Orientation: a ring offset to the RIGHT must fill on the right. This is
    // the check that catches an x/y swap or a mirrored mask, which is otherwise
    // invisible on a centred test shape.
    auto off = ring(24, 0.75f, 0.5f, 0.15f);
    mirror::RasteriseFaceMask(off, idx, W, H, 0, mask);
    size_t left = 0, right = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            (x < W / 2 ? left : right) += mask[size_t(y) * W + x] ? 1 : 0;
    check(right > 0 && left == 0, "an offset ring fills the correct side");

    // Same, vertically.
    auto down = ring(24, 0.5f, 0.8f, 0.12f);
    mirror::RasteriseFaceMask(down, idx, W, H, 0, mask);
    size_t top = 0, bot = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            (y < H / 2 ? top : bot) += mask[size_t(y) * W + x] ? 1 : 0;
    check(bot > 0 && top == 0, "y is not flipped");
}

void test_dilate() {
    std::printf("\ndilation\n");
    const int W = 200, H = 200;
    std::vector<int> idx;
    for (int i = 0; i < 24; ++i) idx.push_back(i);
    auto lm = ring(24, 0.5f, 0.5f, 0.2f);

    std::vector<unsigned char> a, b, c;
    mirror::RasteriseFaceMask(lm, idx, W, H, 0, a);
    mirror::RasteriseFaceMask(lm, idx, W, H, 4, b);
    mirror::RasteriseFaceMask(lm, idx, W, H, 12, c);
    std::printf("       px: dilate 0 = %zu, 4 = %zu, 12 = %zu\n",
                count(a), count(b), count(c));
    check(count(b) > count(a), "dilation grows the mask");
    check(count(c) > count(b), "more dilation grows it further");

    // Dilation must be a superset: it may only add pixels.
    bool superset = true;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] && !b[i]) superset = false;
    check(superset, "dilation never removes a pixel");

    // Growth should be roughly the perimeter times the radius, not the area
    // times something -- a sanity check that the structuring element is the
    // size asked for and not, say, applied twice.
    const double grew = double(count(b) - count(a));
    const double perim = 2.0 * M_PI * 0.2 * W;   // px
    std::printf("       dilate 4 added %.0f px, perimeter*4*2 ~ %.0f\n",
                grew, perim * 4 * 2);
    check(grew > perim * 2 && grew < perim * 4 * 3, "growth is perimeter-scaled");
}

void test_face_oval() {
    std::printf("\nFaceOvalIndices\n");
    const auto& oval = mirror::FaceOvalIndices();
    check(oval.size() == 36, "36 points in the oval loop");
    int mx = 0;
    for (int i : oval) mx = std::max(mx, i);
    check(mx < 468, "indices are inside the 468-point mesh");
    // No duplicates: a repeated vertex in a fill polygon creates a zero-length
    // edge and can flip the even-odd parity.
    std::vector<int> seen(478, 0);
    bool dup = false;
    for (int i : oval) dup = dup || seen[size_t(i)]++;
    check(!dup, "no repeated vertices");

    // Rasterise it against a full 478-landmark set laid out as a ring, which
    // is the shape a real detection has.
    auto lm = ring(478, 0.5f, 0.5f, 0.3f, 478);
    std::vector<unsigned char> mask;
    mirror::RasteriseFaceMask(lm, oval, 160, 120, 0, mask);
    check(count(mask) > 0, "the oval rasterises against a 478-point set");
}

void test_degenerate() {
    std::printf("\ndegenerate input\n");
    std::vector<unsigned char> mask;
    std::vector<int> idx = {0, 1, 2};
    auto lm = ring(3, 0.5f, 0.5f, 0.2f);

    mirror::RasteriseFaceMask(lm, idx, 0, 0, 0, mask);
    check(mask.empty(), "zero size yields an empty mask");

    mirror::RasteriseFaceMask({}, idx, 32, 32, 0, mask);
    check(count(mask) == 0, "no landmarks yields an empty mask, not a crash");

    // An index past the end must bail rather than read out of bounds -- a
    // detection with the wrong landmark count would otherwise corrupt memory.
    std::vector<int> bad = {0, 1, 999};
    mirror::RasteriseFaceMask(lm, bad, 32, 32, 0, mask);
    check(count(mask) == 0, "out-of-range index is rejected safely");

    std::vector<int> two = {0, 1};
    mirror::RasteriseFaceMask(lm, two, 32, 32, 0, mask);
    check(count(mask) == 0, "fewer than 3 points cannot enclose anything");
}

}  // namespace

int main() {
    std::printf("face_mask_test: landmark outline -> training mask\n");
    std::printf("  (backend compiled in: %s)\n",
                mirror::FaceTracker::available() ? "yes" : "no");
    test_basic();
    test_dilate();
    test_face_oval();
    test_degenerate();
    std::printf("\n%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
