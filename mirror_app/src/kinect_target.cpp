#include "kinect_target.h"

#include "kinect_source.h"

#include <algorithm>

namespace mirror {

struct KinectFitTarget::Impl {
    KinectSource src;
    FrameSnapshot frame;      // most recent colour frame, kept across polls
    std::string err;
    uint64_t frames = 0;
    bool mirrored = true;
    bool have_frame = false;
};

KinectFitTarget::KinectFitTarget() : impl_(new Impl()) {}
KinectFitTarget::~KinectFitTarget() { close(); }

bool KinectFitTarget::open(std::string& err) {
    if (impl_->src.isOpen()) return true;
    // OpenGL depth pipeline and the USB reset are both the defaults from the
    // validator. The reset is what clears a sensor left wedged by a previous
    // run, which matters here because this app and kinect_v2_demo cannot hold
    // the device at the same time.
    // Colour only. The fit target is the RGB image; starting the depth stream
    // would run libfreenect2's OpenGL depth pipeline on the same GPU as the
    // training kernels for a result nothing reads.
    const bool ok = impl_->src.open(/*use_opengl=*/true,
                                    KinectSource::UsbReset::kReset, err,
                                    /*want_depth=*/false);
    if (!ok) {
        impl_->err = err;
        return false;
    }
    impl_->err.clear();
    impl_->src.setColorRate(30.f);
    return true;
}

void KinectFitTarget::close() {
    if (impl_ && impl_->src.isOpen()) impl_->src.close();
}

bool KinectFitTarget::isOpen() const { return impl_->src.isOpen(); }
std::string KinectFitTarget::error() const { return impl_->err; }
uint64_t KinectFitTarget::frames() const { return impl_->frames; }
void KinectFitTarget::setMirrored(bool m) { impl_->mirrored = m; }
bool KinectFitTarget::mirrored() const { return impl_->mirrored; }
void KinectFitTarget::setRateHz(float hz) { impl_->src.setColorRate(hz); }

std::string KinectFitTarget::deviceInfo() const {
    if (!impl_->src.isOpen()) return "not open";
    return impl_->src.serial() + "  fw " + impl_->src.firmware() + "  " +
           impl_->src.pipelineName();
}

bool KinectFitTarget::poll(int w, int h, std::vector<float>& rgb) {
    if (!impl_->src.isOpen() || w <= 0 || h <= 0) return false;

    // A new sensor frame is not required every call: the colour camera runs at
    // 30 Hz (15 under long auto-exposure) while the render loop may be well
    // above that, so most frames legitimately have nothing new. Returning false
    // leaves the caller training against the previous target, which is correct
    // -- the alternative would be a stutter in the fit every other frame.
    if (!impl_->src.pollColor(impl_->frame)) return false;
    if (!impl_->frame.valid || impl_->frame.data.empty()) return false;
    impl_->have_frame = true;
    ++impl_->frames;

    const FrameSnapshot& f = impl_->frame;
    // libfreenect2 delivers BGRX or RGBX depending on pipeline; bytes_per_pixel
    // is 4 either way. Guessing wrong swaps red and blue, which on a face is
    // unmistakable but easy to leave in, so it is keyed off the reported format
    // rather than assumed.
    const bool rgbx = (f.format == libfreenect2::Frame::RGBX);
    const int r_off = rgbx ? 0 : 2;
    const int b_off = rgbx ? 2 : 0;

    DownsampleRGB8(f.data.data(), f.width, f.height, f.bytes_per_pixel,
                   r_off, b_off, w, h, rgb);

    if (impl_->mirrored) {
        for (int y = 0; y < h; ++y) {
            float* row = &rgb[size_t(y) * w * 3];
            for (int x = 0; x < w / 2; ++x) {
                float* a = row + size_t(x) * 3;
                float* b = row + size_t(w - 1 - x) * 3;
                for (int c = 0; c < 3; ++c) std::swap(a[c], b[c]);
            }
        }
    }
    return true;
}

}  // namespace mirror
