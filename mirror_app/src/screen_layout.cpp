#include "screen_layout.h"

#include <algorithm>
#include <cmath>

namespace mirror {

ScreenLayout ComputeLayout(int draw_w, int draw_h, Orientation o,
                           float portrait_aspect) {
    ScreenLayout L;
    const int dw = std::max(1, draw_w);
    const int dh = std::max(1, draw_h);

    // Guarded rather than trusted: this comes from a slider and a preset file,
    // and a zero here would divide the composition height by nothing.
    const float pa = std::min(std::max(portrait_aspect, 0.05f), 20.f);
    const float draw_aspect = float(dw) / float(dh);

    float target = draw_aspect;
    if (o == Orientation::Portrait) target = pa;
    else if (o == Orientation::Landscape) target = 1.f / pa;

    // The largest rect of `target` aspect that fits. Auto lands on the drawable
    // exactly, since its target *is* the drawable's aspect.
    if (draw_aspect > target) {
        L.comp_h = dh;
        L.comp_w = std::max(1, int(std::lround(double(dh) * target)));
    } else {
        L.comp_w = dw;
        L.comp_h = std::max(1, int(std::lround(double(dw) / target)));
    }
    // Rounding can push a dimension one pixel past the drawable when the aspects
    // are within a rounding step of each other; the composition must never
    // exceed the window it is drawn into.
    L.comp_w = std::min(L.comp_w, dw);
    L.comp_h = std::min(L.comp_h, dh);

    L.vp_w = L.comp_w;
    L.vp_h = L.comp_h;
    L.vp_x = (dw - L.comp_w) / 2;
    L.vp_y = (dh - L.comp_h) / 2;
    L.letterboxed = (L.comp_w != dw) || (L.comp_h != dh);
    return L;
}

}  // namespace mirror
