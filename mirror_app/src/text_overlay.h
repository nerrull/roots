// text_overlay — legible text over the composited scene, refracted by the pond.
//
// The neural mirror cannot render text. Its input basis is eight features
// (x, y, z, bias, sin_field, cos_field, z_cos, spare) through six tanh layers of
// width 32: a few thousand weights over one low-frequency radial field, which is
// why it reconstructs faces well and would turn letterforms into smudges. Fitting
// text would also mean taking the network away from the face it is tracking.
//
// So the text is not in the network at all. It is a distance field composited in
// the present pass, which makes it crisp at any size -- and it is sampled through
// the *same* ripple-gradient warp the features use, so it refracts with the water
// instead of floating on top of it as an unrelated title card. The sharpness
// comes from the field, the motion comes from the sim.
//
// Compositing is an inversion of what is underneath rather than a fill. That
// needs no colour choice to work against a scene whose palette is generated and
// changes constantly, and it keeps the text legible over both the bright and dark
// halves of a ripple. (The invert in root_face.metal falls back to flat white
// because Metal has no framebuffer logic ops; here the scene colour is sampled in
// the fragment, so a real inversion is just 1 - c.)
#pragma once

#ifndef __OBJC__
#error "text_overlay.h is ObjC++ only"
#endif

#import <Metal/Metal.h>
#include <simd/simd.h>

#include <string>

class MetalContext;

namespace mirror {

// Every knob, so the UI and the preset system see one struct.
struct TextParams {
    bool  on = false;
    // '\n' splits lines; they are centred on each other.
    std::string text = "JARDINS RACINE";
    std::string font = "HelveticaNeue-Bold";

    // Placement, in the render's coord space: x spans (-aspect, aspect) and y
    // spans (-1, 1), the same convention as the feature grid, so a position set
    // here means the same thing the ripple centres do.
    float cx = 0.f, cy = 0.f;
    float size = 0.16f;      // half-height of the text block, coord units

    float strength = 1.f;    // 0 = invisible, 1 = full inversion
    // Refraction by the ripple gradient. Low by default: the gradient is a
    // large number in coord units, and past ~0.15 it displaces neighbouring
    // fragments far enough apart to tear the letterforms open. That is a usable
    // effect, but it is not the one you want the first time you switch this on.
    float warp = 0.08f;
    // Edge width, in units of one pixel's footprint. This is an antialiasing
    // control, not a glow: the field only encodes distance out to `spread`
    // pixels, so past a few pixels of blur the ramp runs off the end of what is
    // represented and the shader clamps it.
    float softness = 1.f;
    float dilate = 0.f;      // thicken (+) / thin (-) the strokes, coord units
    float tracking = 0.02f;  // letter spacing, em

    // --- turbulent emerge / dissolve ---------------------------------------
    // `reveal` is the timeline: 0 is gone, 1 is whole, and everything between is
    // the word coming apart. Each pixel gets a threshold from a noise field and
    // appears once reveal passes it, rather than the outline being eroded --
    // erosion is bounded by the encoded band, so thick stems would sit
    // untouched and then pop out all at once.
    float reveal = 1.f;
    float turbulence = 0.8f;   // 0 = an even fade, 1 = fully broken up
    float turb_scale = 12.f;    // noise frequency, cycles per coord unit
    float turb_speed = 0.15f;  // how fast the patches drift

    // Rasterisation height per line, in pixels: the quality knob. The distance
    // transform is exact with respect to this bitmap, so this sets how faithful
    // the outline is, not how large the text draws.
    int raster_px = 256;
};

// The ripple field the text refracts through -- the same terms the fused feature
// kernel accumulates its warp gradient from, passed through plainly so this
// header does not have to pull in MLX.
struct TextRipple {
    float k = 3.f;         // ring_freq
    float decay = 1.8f;
    float core_r2 = 0.f;   // core_radius^2, or 0 with the rolloff off
    int   n = 0;           // active sources
    float src[16][4] = {}; // cx, cy, phase, amp
};

// Matches TextU in present.metal. Packed as float4s so the C and MSL layouts
// agree without either side needing alignment attributes.
struct TextUniforms {
    simd::float4 place = {0, 0, 0, 0};   // centre.xy, half-extent.xy (coord units)
    simd::float4 tune  = {1, 0, 1, 0};   // aspect, strength, softness, dilate
    simd::float4 tune2 = {0, 3, 1.8f, 0};// warp, k, decay, core_r2
    simd::float4 cnt   = {0, 0, 0, 1};   // source count, enabled, time, reveal
    simd::float4 diss  = {0, 6, 0, 0};   // turbulence, turb scale, turb speed, -
    simd::float4 src[16] = {};
};

class TextOverlay {
public:
    explicit TextOverlay(const MetalContext& ctx);

    // Rebuilds the field when the text, font, tracking or raster height change;
    // otherwise cheap enough to call every frame. Everything else (position,
    // size, strength, warp) is a uniform and needs no rebuild -- which is what
    // makes those the live-performance controls and these the setup ones.
    void update(const TextParams& p);

    // The uniforms for this frame. `aspect` is the drawable's, and must be the
    // one the coord grid was built with or the text and the ripples will not
    // agree about where anything is. `time` drives the dissolve's drift only.
    TextUniforms uniforms(const TextParams& p, float aspect,
                          const TextRipple& r, double time) const;

    id<MTLTexture> texture() const { return tex_; }
    bool valid() const { return tex_ != nil; }

    // Bitmap dimensions of the field, for the UI to report.
    int fieldW() const { return fw_; }
    int fieldH() const { return fh_; }

private:
    const MetalContext& ctx_;
    id<MTLTexture> tex_ = nil;

    // What the current texture was built from, so update() can no-op.
    std::string built_text_, built_font_;
    float built_tracking_ = -1.f;
    int   built_raster_ = -1;

    int   fw_ = 0, fh_ = 0;
    float spread_ = 1.f;      // pixels spanned by half the encoded range
    // Ratio of the rasterised block's height to the text's own typographic
    // height. The padding needed by the distance field is part of the bitmap but
    // must not change how large the text draws, so the placement extent is
    // scaled by this rather than being the bitmap outright.
    float pad_ratio_ = 1.f;
    float aspect_ratio_ = 1.f;  // bitmap width / height
};

}  // namespace mirror
