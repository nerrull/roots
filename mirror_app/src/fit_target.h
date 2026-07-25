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

// Box-filter downsample of an 8-bit interleaved image into h*w*3 floats.
//
// Box rather than point sampling: the colour camera is 1920x1080 and the fit
// grid is a few hundred pixels wide, so point sampling would alias hard and
// the network would chase sampling noise between frames. `stride_px` is the
// source's bytes per pixel, `r_off`/`b_off` select channel order (BGRX vs RGBX).
void DownsampleRGB8(const unsigned char* src, int src_w, int src_h,
                    int stride_px, int r_off, int b_off,
                    int dst_w, int dst_h, std::vector<float>& dst);

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
