// kinect_target — the Kinect v2 colour camera as a live fitting target.
//
// Wraps kinect_v2_validate's KinectSource rather than reimplementing it. That
// module already handles the parts that are easy to get wrong and expensive to
// rediscover: the USB reset policy, independent latest-wins listeners per
// stream (a SyncMultiFrameListener would gate depth on colour), and per-stream
// poll-rate control. See kinect_v2_validate/src/demo/kinect_source.h.
//
// Compiled only when freenect2 is available (MIRROR_HAVE_KINECT); the app
// builds and runs without it, with this target simply absent from the UI.

#pragma once

#include "fit_target.h"

#include <memory>
#include <string>

namespace mirror {

class KinectFitTarget : public FitTarget {
public:
    KinectFitTarget();
    ~KinectFitTarget() override;

    // Opens the sensor. Returns false with `err` set if it is missing, already
    // held by another process, or fails to start.
    bool open(std::string& err);
    void close();
    bool isOpen() const;

    // The colour camera is 1920x1080; the fit grid is a few hundred px, so
    // frames are box-filtered down (see DownsampleRGB8). Returns false when no
    // new sensor frame has arrived since the last call -- the caller should
    // keep training on the previous target rather than treating it as an error.
    bool poll(int w, int h, std::vector<float>& rgb) override;

    // The most recently polled frame as RGB8 at (w, h), for the face tracker.
    //
    // Deliberately reads the *retained* snapshot rather than pulling a new one:
    // a second pollColor() here would race the fit path for frames, and the two
    // would end up looking at different moments -- the mask would then be a
    // face outline from one frame applied to the pixels of another, which shows
    // up as the fit smearing whenever anyone moves. Returns false before the
    // first successful poll().
    bool lastFrameRGB8(int w, int h, std::vector<unsigned char>& rgb) const;

    const char* name() const override { return "kinect"; }
    std::string error() const override;
    uint64_t frames() const override;

    // Mirror the image horizontally. On by default: a mirror should show you
    // your own left hand on your left, and the sensor does not do that.
    void setMirrored(bool m);
    bool mirrored() const;

    // Cap how often frames are pulled. The sensor free-runs at 30 Hz; pulling
    // less often costs freshness, not stability.
    void setRateHz(float hz);

    std::string deviceInfo() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mirror
