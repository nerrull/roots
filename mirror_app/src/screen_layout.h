// screen_layout — the frame the work is composed for, separately from the window
// it is being shown in.
//
// The installation runs on a portrait screen; development happens on a landscape
// one. Those are the same piece rendered into differently-shaped frames, and
// almost everything downstream is derived from that shape: the mirror's coord
// space spans (-aspect, aspect) x (-1, 1), the root renderer builds its frustum
// from w/h, the text overlay places itself in coord units, and the camera feed
// has to be cropped into it. Composing for the drawable directly means the only
// way to see the installation's framing is to be standing in front of it.
//
// So the composition size is its own thing. Scenes render at it, and the present
// pass blits that into whatever viewport the drawable can give it -- exactly the
// drawable when the two agree, a centred letterbox when they do not. Forcing
// Portrait on a landscape monitor is then a true preview: the same aspect the
// panel will have, the same crop out of the camera, the same place the text
// lands, in a tall box in the middle of the window.
//
// No rotation is involved. The installation's display is rotated by macOS, so
// its drawable is genuinely tall and the app only has to compose tall. A
// physically rotated panel driven with a landscape signal would need the blit to
// rotate as well, which this deliberately does not do.
#pragma once

namespace mirror {

enum class Orientation : int {
    Auto = 0,       // compose for whatever the drawable is; never letterboxes
    Landscape = 1,  // force the panel's landscape aspect
    Portrait = 2,   // force the panel's portrait aspect
};

struct ScreenLayout {
    // What the scenes render at.
    int comp_w = 1, comp_h = 1;
    // Where that lands in the drawable, in pixels, origin top-left.
    int vp_x = 0, vp_y = 0, vp_w = 1, vp_h = 1;
    // True when the composition does not fill the drawable, i.e. there are bars.
    bool letterboxed = false;
};

// `portrait_aspect` is the installation panel's width/height in portrait --
// 9/16 = 0.5625 for a 1920x1080 panel stood on its end. It is a *stated* number
// rather than the dev monitor's aspect inverted, because the preview is only
// worth anything if it matches the real screen and not the one it is being
// previewed on. Landscape forces its reciprocal.
//
// The composition is the largest rect of the target aspect that fits inside the
// drawable, centred. When the drawable already has that aspect -- the
// installation, running portrait -- that is the drawable itself, so the same
// rule covers both cases and there is no fills-vs-letterboxes branch to get
// wrong.
ScreenLayout ComputeLayout(int draw_w, int draw_h, Orientation o,
                           float portrait_aspect = 9.f / 16.f);

}  // namespace mirror
