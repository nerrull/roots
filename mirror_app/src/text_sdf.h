// text_sdf — an exact signed distance field from a coverage bitmap.
//
// The overlay needs text that stays crisp at any size, over a scene that is
// itself only a few hundred pixels wide before upsampling. A glyph *bitmap*
// cannot do that: it is sharp at exactly one scale and mush either side of it.
// A distance field is resolution-independent by construction -- the edge is
// wherever the interpolated distance crosses zero, so the fragment shader
// recovers a clean, antialiased contour at whatever scale it is drawn, and can
// dilate or erode it for free.
//
// It is also what makes the ripple warp affordable. Refracting a bitmap means
// resampling glyphs every frame and watching them alias as they move; refracting
// a distance field is a perturbation of the *sample coordinate*, and the edge
// stays exactly as sharp as it was before it moved.
//
// The transform is the exact Euclidean one (Felzenszwalb & Huttenlocher's O(n)
// squared-distance transform, run per axis), not an approximation like 8SSEDT.
// Text at large scale is exactly where a chamfer approximation's error shows up
// as visible facets on the near-straight edges of a stem.
#pragma once

#include <cstddef>
#include <vector>

namespace mirror {

// A signed distance field, 8-bit. Encoding: 0.5 is the edge, values above it
// are inside the glyph, and the full 0..1 range spans +/- `spread` pixels of the
// source bitmap. 8 bits is enough because the value is only ever used near the
// crossing, where a linear sampler interpolates between adjacent texels anyway.
struct SdfImage {
    int w = 0, h = 0;
    float spread = 0.f;             // pixels mapped to half the 0..1 range
    std::vector<unsigned char> px;  // w*h, one byte per texel
};

// Build a field from `cov`, a w*h 8-bit coverage bitmap (0 = outside,
// 255 = inside). The threshold is 128, so an antialiased rasterisation places
// the contour at the half-covered pixel -- which is where the rasteriser itself
// thinks the edge is.
//
// Rasterise `cov` well above the size the field will be sampled at: the
// transform is exact with respect to the bitmap it is given, so the bitmap's own
// quantisation of the outline is the only error left, and it is the one that
// shows up as wobble along a long straight stem.
SdfImage BuildSDF(const unsigned char* cov, int w, int h, float spread);

// The squared Euclidean distance transform of `f`, in place: on entry, 0 at
// seed sites and a large value elsewhere; on exit, the squared distance to the
// nearest seed. Exposed because both passes of BuildSDF use it and because it is
// the part worth testing directly.
void SquaredEDT(std::vector<float>& f, int w, int h);

}  // namespace mirror
