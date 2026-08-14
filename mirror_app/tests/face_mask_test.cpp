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

void test_box() {
    std::printf("\nlandmark box\n");
    const int W = 200, H = 200;
    auto lm = ring(24, 0.5f, 0.5f, 0.2f);

    std::vector<unsigned char> box;
    mirror::RasteriseFaceBox(lm, W, H, 0.f, 0, box);
    check(box.size() == size_t(W) * H, "mask is w*h");
    // The bounding box of a radius-0.2 ring is 0.4 x 0.4 of the frame.
    const double frac = double(count(box)) / (W * H);
    std::printf("       covers %.4f of frame (box would be %.4f)\n", frac, 0.16);
    check(frac > 0.15 && frac < 0.18, "area matches the bounding box");

    // The box must contain the hull: this is the property the two modes are
    // chosen between on, so a box that ever cut inside the outline would make
    // "box" the tighter option and invert the whole comparison.
    std::vector<int> idx;
    for (int i = 0; i < 24; ++i) idx.push_back(i);
    std::vector<unsigned char> hull;
    mirror::RasteriseFaceMask(lm, idx, W, H, 0, hull);
    bool contains = true;
    for (size_t i = 0; i < hull.size(); ++i)
        if (hull[i] && !box[i]) contains = false;
    check(contains, "the box contains the hull");
    check(count(box) > count(hull), "and is strictly larger");

    // Orientation, same as the hull's: an offset face must box the right side.
    auto off = ring(24, 0.75f, 0.5f, 0.1f);
    mirror::RasteriseFaceBox(off, W, H, 0.f, 0, box);
    size_t left = 0, right = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            (x < W / 2 ? left : right) += box[size_t(y) * W + x] ? 1 : 0;
    check(right > 0 && left == 0, "an offset face boxes the correct side");

    // Padding is a fraction of the box's own size, so the margin scales with
    // how close the person is -- that scale-following is the whole reason it
    // is a fraction and not a pixel count. Measured as widths rather than
    // areas: a small box is only a few dozen pixels across and the rounding at
    // its edges swamps an area ratio.
    auto width_of = [](const std::vector<unsigned char>& m, int w, int h) {
        int lo = w, hi = -1;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                if (m[size_t(y) * w + x]) { lo = std::min(lo, x); hi = std::max(hi, x); }
        return hi < lo ? 0 : hi - lo + 1;
    };
    std::vector<unsigned char> s0, s1, b0, b1;
    auto small = ring(24, 0.5f, 0.5f, 0.1f);   // 0.2 of the frame across
    auto big   = ring(24, 0.5f, 0.5f, 0.2f);   // twice that
    mirror::RasteriseFaceBox(small, W, H, 0.f,   0, s0);
    mirror::RasteriseFaceBox(small, W, H, 0.5f,  0, s1);
    mirror::RasteriseFaceBox(big,   W, H, 0.f,   0, b0);
    mirror::RasteriseFaceBox(big,   W, H, 0.5f,  0, b1);
    const int gs = width_of(s1, W, H) - width_of(s0, W, H);
    const int gb = width_of(b1, W, H) - width_of(b0, W, H);
    std::printf("       pad 0.5 widens small by %d px, large by %d px\n", gs, gb);
    check(gs > 0 && gb > 0, "fractional pad widens the box");
    check(gb > gs * 1.7 && gb < gs * 2.3,
          "a face twice the size gets twice the margin");
    // ...and the fraction is of the whole box: pad 0.5 is +50% across. Checked
    // on the larger face, where the ±1px the fill rounds outward is a smaller
    // share of the width.
    const double ratio = double(width_of(b1, W, H)) / width_of(b0, W, H);
    std::printf("       pad 0.5 widens the box x%.3f (want 1.5)\n", ratio);
    check(ratio > 1.42 && ratio < 1.58, "pad 0.5 is half again as wide");

    // A face partly out of frame must clip, not wrap or crash.
    auto edge = ring(24, 0.02f, 0.5f, 0.2f);
    mirror::RasteriseFaceBox(edge, W, H, 0.2f, 4, box);
    bool row_ok = true;
    for (int y = 0; y < H; ++y) {
        // Any filled row must be a single run touching the left edge.
        const unsigned char* r = &box[size_t(y) * W];
        if (!r[0]) continue;
        int run = 0;
        while (run < W && r[run]) ++run;
        for (int x = run; x < W; ++x) if (r[x]) row_ok = false;
    }
    check(row_ok, "an off-frame face clips instead of wrapping");
    check(count(box) > 0, "and still selects pixels");
}

// The distance transform the soft edge is built on. Worth its own test because
// it is exact arithmetic with an easy off-by-one: a field that is right to
// within a pixel still looks fine on screen and still puts the fade in slightly
// the wrong place everywhere.
void test_distance() {
    std::printf("\ndistance outside\n");
    const int W = 64, H = 64;
    std::vector<unsigned char> mask(size_t(W) * H, 0);
    // A single pixel: the distance from it is exactly the Euclidean radius, so
    // every value in the field has a closed form to check against.
    mask[size_t(32) * W + 32] = 1;

    std::vector<float> d;
    mirror::DistanceOutside(mask, W, H, d);
    check(d.size() == size_t(W) * H, "field is w*h");
    check(d[size_t(32) * W + 32] == 0.f, "zero on the mask");

    float worst = 0.f;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float want = std::sqrt(float((x - 32) * (x - 32) +
                                               (y - 32) * (y - 32)));
            worst = std::max(worst, std::fabs(want - d[size_t(y) * W + x]));
        }
    std::printf("       worst error vs the exact radius: %.6f px\n", worst);
    check(worst < 1e-3f, "exact Euclidean distance, not an approximation");

    // A filled rectangle: zero throughout, and the distance just outside an
    // edge is 1 px -- the check that catches a field measured from the mask's
    // centre or from its bounding box rather than from the mask itself.
    std::vector<unsigned char> rect(size_t(W) * H, 0);
    for (int y = 20; y <= 40; ++y)
        for (int x = 10; x <= 50; ++x) rect[size_t(y) * W + x] = 1;
    mirror::DistanceOutside(rect, W, H, d);
    bool inside_zero = true;
    for (int y = 20; y <= 40; ++y)
        for (int x = 10; x <= 50; ++x)
            inside_zero = inside_zero && d[size_t(y) * W + x] == 0.f;
    check(inside_zero, "zero everywhere inside a filled shape");
    check(std::fabs(d[size_t(19) * W + 30] - 1.f) < 1e-4f, "1 px above the top edge");
    check(std::fabs(d[size_t(30) * W + 9] - 1.f) < 1e-4f, "1 px left of the side");
    // Diagonally off a corner is sqrt(2), not 1: the transform is Euclidean,
    // not the cheap chamfer that would make the fade band square.
    check(std::fabs(d[size_t(19) * W + 9] - std::sqrt(2.f)) < 1e-4f,
          "sqrt(2) off a corner, so the band is round");

    // Symmetry: equal distances in every direction from a centred shape. This
    // is the property the whole "even band width" fix rests on.
    const float up = d[size_t(20 - 5) * W + 30], down = d[size_t(40 + 5) * W + 30];
    const float left = d[size_t(30) * W + (10 - 5)], right = d[size_t(30) * W + (50 + 5)];
    check(std::fabs(up - down) < 1e-4f && std::fabs(left - right) < 1e-4f &&
              std::fabs(up - left) < 1e-4f,
          "the same distance out in all four directions");

    mirror::DistanceOutside({}, 0, 0, d);
    check(d.empty(), "zero size yields an empty field");
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

    mirror::RasteriseFaceBox(lm, 0, 0, 0.f, 0, mask);
    check(mask.empty(), "box: zero size yields an empty mask");
    mirror::RasteriseFaceBox({}, 32, 32, 0.f, 0, mask);
    check(count(mask) == 0, "box: no landmarks yields an empty mask, not a crash");
    // Entirely off-frame: nothing to select, and nothing to write out of bounds.
    auto gone = ring(8, -0.9f, 0.5f, 0.05f);
    mirror::RasteriseFaceBox(gone, 32, 32, 0.f, 0, mask);
    check(count(mask) == 0, "box: a face fully off-frame selects nothing");
}

}  // namespace

int main() {
    std::printf("face_mask_test: landmark outline -> training mask\n");
    std::printf("  (backend compiled in: %s)\n",
                mirror::FaceTracker::available() ? "yes" : "no");
    test_basic();
    test_dilate();
    test_face_oval();
    test_box();
    test_distance();
    test_degenerate();
    std::printf("\n%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
