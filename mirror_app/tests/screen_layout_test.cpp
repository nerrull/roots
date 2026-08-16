// screen_layout_test — the composition/letterbox arithmetic, and the camera
// crop that follows from it.
//
// Both are pixel arithmetic whose failure mode is being slightly wrong rather
// than visibly broken: a composition one pixel over the drawable, a crop that
// silently changes scale when it hits the edge of its travel, an aspect that is
// right on a 16:9 window and wrong on 16:10. None of that is legible in a
// screenshot, which is why it is checked here rather than by looking.

#include "fit_target.h"
#include "screen_layout.h"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

}  // namespace

int main() {
    using namespace mirror;
    const float P = 9.f / 16.f;

    // --- Auto fills, always ------------------------------------------------
    for (int w : {1280, 1920, 1080, 1000}) {
        for (int h : {720, 1080, 1920, 1000}) {
            const ScreenLayout L = ComputeLayout(w, h, Orientation::Auto, P);
            check(L.comp_w == w && L.comp_h == h, "auto composes at the drawable");
            check(!L.letterboxed, "auto never letterboxes");
            check(L.vp_x == 0 && L.vp_y == 0, "auto starts at the origin");
        }
    }

    // --- the installation: a portrait drawable, portrait forced ------------
    // The case that has to be exactly right, because it is the one that ships.
    {
        const ScreenLayout L = ComputeLayout(1080, 1920, Orientation::Portrait, P);
        check(L.comp_w == 1080 && L.comp_h == 1920, "portrait panel fills");
        check(!L.letterboxed, "portrait panel has no bars");
    }

    // --- developing: portrait forced on a landscape monitor ----------------
    {
        const ScreenLayout L = ComputeLayout(1920, 1080, Orientation::Portrait, P);
        check(L.comp_h == 1080, "preview is as tall as the window");
        check(L.comp_w == 608, "preview is 9:16 of that height");
        check(near(float(L.comp_w) / L.comp_h, P, 0.002f),
              "preview matches the panel's aspect, not the monitor's");
        check(L.letterboxed, "preview letterboxes");
        check(L.vp_x == (1920 - 608) / 2 && L.vp_y == 0, "preview is centred");
    }

    // --- the aspect is the panel's, not the window's -----------------------
    // A 16:10 dev monitor must still preview a 9:16 frame. Deriving the target
    // by inverting the window's aspect would pass on 16:9 and be wrong here.
    {
        const ScreenLayout L = ComputeLayout(1680, 1050, Orientation::Portrait, P);
        check(near(float(L.comp_w) / L.comp_h, P, 0.002f),
              "16:10 window still previews 9:16");
    }

    // --- landscape forced on a portrait drawable ---------------------------
    {
        const ScreenLayout L = ComputeLayout(1080, 1920, Orientation::Landscape, P);
        check(L.comp_w == 1080, "landscape preview is as wide as the window");
        check(near(float(L.comp_w) / L.comp_h, 1.f / P, 0.01f),
              "landscape preview is the panel on its side");
        check(L.letterboxed, "landscape on a tall window letterboxes");
    }

    // --- invariants over a sweep -------------------------------------------
    for (int w = 200; w <= 2600; w += 137) {
        for (int h = 200; h <= 2600; h += 149) {
            for (Orientation o : {Orientation::Auto, Orientation::Landscape,
                                  Orientation::Portrait}) {
                const ScreenLayout L = ComputeLayout(w, h, o, P);
                check(L.comp_w >= 1 && L.comp_h >= 1, "composition is never empty");
                check(L.comp_w <= w && L.comp_h <= h,
                      "composition never exceeds the drawable");
                check(L.vp_x >= 0 && L.vp_y >= 0 &&
                      L.vp_x + L.vp_w <= w && L.vp_y + L.vp_h <= h,
                      "viewport stays inside the drawable");
            }
        }
    }

    // --- degenerate input ---------------------------------------------------
    {
        const ScreenLayout L = ComputeLayout(0, 0, Orientation::Portrait, 0.f);
        check(L.comp_w >= 1 && L.comp_h >= 1, "a zero drawable still yields a frame");
    }

    // --- the camera crop ----------------------------------------------------
    // 1920x1080 sensor into a 9:16 frame: a tall rect the full height of the
    // sensor, about a third of its width. Losing the sides is the point.
    {
        const SrcRect r = ComputeFeedRect(1920, 1080, 608, 1080, FeedCrop{});
        check(r.h == 1080, "the tall crop keeps the sensor's full height");
        check(std::abs(r.w - 607) <= 2, "and 9:16 of it in width");
        check(std::abs(r.x - (1920 - r.w) / 2) <= 1, "centred by default");
        check(near(float(r.w) / r.h, 608.f / 1080.f, 0.01f),
              "crop matches the destination's aspect");
    }
    // A landscape frame from the same sensor is the whole thing.
    {
        const SrcRect r = ComputeFeedRect(1920, 1080, 1920, 1080, FeedCrop{});
        check(r.x == 0 && r.y == 0 && r.w == 1920 && r.h == 1080,
              "matching aspects crop nothing");
    }
    // Zoom moves in; panning to the edge slides the rect and must not resize it,
    // or someone walking across the frame would change size at the extremes.
    {
        FeedCrop c;
        c.zoom = 2.f;
        const SrcRect mid = ComputeFeedRect(1920, 1080, 608, 1080, c);
        check(std::abs(mid.h - 540) <= 2, "zoom 2 halves the rect");
        c.cx = 0.f;
        const SrcRect left = ComputeFeedRect(1920, 1080, 608, 1080, c);
        check(left.x == 0, "panned hard left, the rect sits at the edge");
        check(left.w == mid.w && left.h == mid.h, "panning does not resize");
        c.cx = 1.f;
        const SrcRect right = ComputeFeedRect(1920, 1080, 608, 1080, c);
        check(right.x + right.w == 1920, "panned hard right, flush to the edge");
        check(right.w == mid.w && right.h == mid.h, "still no resize");
    }
    // Absurd zoom must stay inside the image rather than producing a rect the
    // resampler would index off the end of.
    {
        FeedCrop c;
        c.zoom = 1000.f;
        const SrcRect r = ComputeFeedRect(1920, 1080, 608, 1080, c);
        check(r.w >= 1 && r.h >= 1, "extreme zoom still yields a rect");
        check(r.x >= 0 && r.y >= 0 && r.x + r.w <= 1920 && r.y + r.h <= 1080,
              "extreme zoom stays inside the sensor");
    }
    {
        FeedCrop c;
        c.zoom = 0.f;   // a preset written before zoom was clamped
        const SrcRect r = ComputeFeedRect(1920, 1080, 608, 1080, c);
        check(r.x >= 0 && r.y >= 0 && r.x + r.w <= 1920 && r.y + r.h <= 1080,
              "zero zoom stays inside the sensor");
    }
    check(ComputeFeedRect(0, 0, 100, 100, FeedCrop{}).w == 0, "empty source");

    // --- crop and resample agree -------------------------------------------
    // The rect the tracker's image is taken from must be the rect the fit's is,
    // or the landmarks describe a different picture from the one being fitted.
    {
        std::vector<unsigned char> src(size_t(64) * 32 * 3, 0);
        for (int y = 0; y < 32; ++y)
            for (int x = 0; x < 64; ++x)
                src[(size_t(y) * 64 + x) * 3 + 0] = (unsigned char)(x * 4);

        const SrcRect r = ComputeFeedRect(64, 32, 8, 16, FeedCrop{});
        std::vector<float> f;
        std::vector<unsigned char> u8;
        DownsampleRectRGB8(src.data(), 64, 32, 3, 0, 2, r, 8, 16, f);
        DownsampleRectToRGB8(src.data(), 64, 32, 3, 0, 2, r, 8, 16, u8);
        check(f.size() == size_t(8) * 16 * 3 && u8.size() == f.size(),
              "both crops produce the requested frame");
        bool same = true;
        for (size_t i = 0; i < f.size(); ++i)
            same = same && std::fabs(f[i] * 255.f - float(u8[i])) <= 1.5f;
        check(same, "the float and byte crops are the same image");
        // The crop is the middle of a left-to-right ramp, so it must not start
        // at the darkest column -- that would mean the rect was ignored.
        check(f[0] > 0.1f, "the crop is taken from the middle, not the origin");
    }

    if (failures == 0) std::printf("screen_layout_test: OK\n");
    return failures == 0 ? 0 : 1;
}
