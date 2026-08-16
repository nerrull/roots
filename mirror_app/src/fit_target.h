// fit_target — what the neural mirror is being fitted to.
//
// A still image is a fixed target the network converges onto. A camera feed is
// a *moving* one: the fit never finishes, it tracks. That is the mirror — the
// network is always a few hundred milliseconds behind whoever is standing in
// front of the sensor, and the lag is the effect rather than a defect.
//
// Both are the same interface because the training loop should not care, and
// because the still-image path stays useful for testing something that would
// otherwise need a person and a sensor to exercise at all.

#pragma once

#include <string>
#include <vector>

namespace mirror {

// A target frame: h*w*3 floats in [0, 1], row-major RGB.
class FitTarget {
public:
    virtual ~FitTarget() = default;

    // Fills `rgb` with the current frame resampled to (w, h). Returns false if
    // no new frame is available -- for a live source that just means "reuse the
    // last one", not an error, so the caller keeps training against whatever it
    // last received rather than stalling.
    virtual bool poll(int w, int h, std::vector<float>& rgb) = 0;

    virtual const char* name() const = 0;
    // Empty while healthy.
    virtual std::string error() const { return {}; }
    // Frames delivered so far; lets a stalled source be spotted from the UI.
    virtual uint64_t frames() const = 0;
};

// --- helpers shared by the implementations ----------------------------------

// How a source frame maps into a differently-shaped output frame.
//
// The sensor is 16:9 and does not turn around when the screen does. Resampling
// it straight into a portrait fit grid stretches whoever is standing there into
// a caricature -- and it is the kind of wrong that looks almost right in a
// preview thumbnail and unmistakable at scale. So a rect of the *output's*
// aspect is taken out of the source instead, which means a portrait frame throws
// away most of a 16:9 image and where that rect sits becomes a real decision.
// Hence the framing controls: with only a third of the sensor's width surviving,
// nobody can be relied on to stand in the middle of it.
struct FeedCrop {
    float cx = 0.5f, cy = 0.5f;  // rect centre, normalised source coords
    float zoom = 1.f;            // > 1 takes a smaller rect, i.e. moves in
};

// A rect in source pixels.
struct SrcRect { int x = 0, y = 0, w = 0, h = 0; };

// The rect `c` selects when resampling into a dst_w x dst_h frame: the largest
// rect of the output's aspect at zoom 1, divided by zoom, centred on (cx, cy)
// and shifted -- never shrunk -- to stay inside the source. Clamping by shifting
// keeps the scale the framing controls ask for, so panning to the edge slides
// the crop rather than silently zooming it out.
SrcRect ComputeFeedRect(int src_w, int src_h, int dst_w, int dst_h,
                        const FeedCrop& c);

// Box-filter downsample of an 8-bit interleaved image into h*w*3 floats.
//
// Box rather than point sampling: the colour camera is 1920x1080 and the fit
// grid is a few hundred pixels wide, so point sampling would alias hard and
// the network would chase sampling noise between frames. `stride_px` is the
// source's bytes per pixel, `r_off`/`b_off` select channel order (BGRX vs RGBX).
void DownsampleRGB8(const unsigned char* src, int src_w, int src_h,
                    int stride_px, int r_off, int b_off,
                    int dst_w, int dst_h, std::vector<float>& dst);

// The same, over a sub-rect of the source (see ComputeFeedRect). The full-frame
// versions are these with the rect set to the whole image, so there is one
// filter and the cropped and uncropped paths cannot drift apart.
void DownsampleRectRGB8(const unsigned char* src, int src_w, int src_h,
                        int stride_px, int r_off, int b_off, const SrcRect& rect,
                        int dst_w, int dst_h, std::vector<float>& dst);
void DownsampleRectToRGB8(const unsigned char* src, int src_w, int src_h,
                          int stride_px, int r_off, int b_off,
                          const SrcRect& rect, int dst_w, int dst_h,
                          std::vector<unsigned char>& dst);

// The same resampling, to 8-bit RGB. MediaPipe wants bytes, and it wants them
// at a higher resolution than the fit grid -- the fit runs at a couple of
// hundred pixels wide, where a face is too few pixels to land landmarks well.
// Sharing the filter with the float path keeps the tracker looking at the same
// image the network is being fitted to, only larger.
void DownsampleToRGB8(const unsigned char* src, int src_w, int src_h,
                      int stride_px, int r_off, int b_off,
                      int dst_w, int dst_h, std::vector<unsigned char>& dst);

// Flip an RGB8 image horizontally, in place.
void MirrorRGB8(int w, int h, std::vector<unsigned char>& rgb);

// Translate an h*w*3 float image by (dx, dy) pixels, clamping at the edges.
//
// This is how a moving subject is held still for the fit: shift the frame by
// the head's displacement and the head lands in the same place every time, so
// the network is always shown the same problem. Whole pixels only -- a
// subpixel shift would resample the face every frame, and feeding the fit a
// slightly differently-filtered image each time is exactly the noise the
// stabilising is meant to remove.
//
// Edge clamp rather than black fill: the vacated strip is outside the mask and
// never trained, but a hard black band there would still show up the moment
// anyone widened the crop.
void ShiftRGBF(int w, int h, int dx, int dy, std::vector<float>& rgb);

// Place a region of the frame: resample so that the normalised point
// (src_cx, src_cy) lands at the centre and everything is scaled about it by
// `scale`. Bilinear, clamped at the edges.
//
// This is the head-centred mode with a size control on it: scale > 1 makes the
// subject bigger on screen. Unlike ShiftRGBF it must interpolate, so it does
// refilter the image every frame -- the cost of choosing the size rather than
// accepting whatever distance the person is standing at.
void PlaceRGBF(int w, int h, float src_cx, float src_cy, float scale,
               std::vector<float>& rgb);

// A still image loaded once. `load_rgb` is supplied by the caller because
// decoding is platform code (main.mm uses NSImage) and this header is not.
class StaticFitTarget : public FitTarget {
public:
    StaticFitTarget(std::vector<float> rgb, int w, int h)
        : rgb_(std::move(rgb)), w_(w), h_(h) {}

    bool poll(int w, int h, std::vector<float>& out) override;
    const char* name() const override { return "image"; }
    uint64_t frames() const override { return frames_; }

private:
    std::vector<float> rgb_;
    int w_ = 0, h_ = 0;
    uint64_t frames_ = 0;
};

}  // namespace mirror
