// Kinect v2 capture for the ImGui demo.
//
// Design note (this is the fix for what the plain validator mis-reports):
// the validator uses SyncMultiFrameListener over Color|Ir|Depth, which only
// releases a bundle once *every* stream has a frame. With RGB at 15 fps under
// long auto-exposure that gates depth down to 15 fps too, and the depth frames
// the listener throws away show up as gaps in Frame::sequence -- looking like
// hundreds of dropped frames when the USB path is actually clean.
//
// Here each stream gets its own "latest wins" listener instead. Color and depth
// are fully independent, nothing is gated, and a frame that arrives before the
// UI polled the previous one is counted as *skipped* (expected, benign) rather
// than dropped. Genuine transport loss still shows up as a sequence gap.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <libfreenect2/frame_listener.hpp>
#include <libfreenect2/libfreenect2.hpp>

// A detached copy of one frame, owned by the caller. We copy pixels out under
// the lock rather than handing the UI a libfreenect2::Frame* so that frame
// lifetime can never straddle the USB delivery thread and the render thread.
struct FrameSnapshot {
  std::vector<unsigned char> data;
  int width = 0;
  int height = 0;
  int bytes_per_pixel = 0;
  libfreenect2::Frame::Format format = libfreenect2::Frame::Invalid;
  uint32_t sequence = 0;
  uint32_t timestamp = 0;
  float exposure = 0.f;
  float gain = 0.f;
  float gamma = 0.f;
  bool valid = false;
};

struct StreamStats {
  uint64_t delivered = 0;   // frames handed to us by libfreenect2
  uint64_t polled = 0;      // frames the UI actually consumed
  uint64_t skipped = 0;     // superseded before the UI polled them (benign)
  uint64_t seq_gaps = 0;    // real transport loss: holes in Frame::sequence
  double delivered_hz = 0;  // measured, sliding window
  double polled_hz = 0;
};

// Keeps only the newest frame per type. onNewFrame() never blocks on the UI.
class LatestFrameListener : public libfreenect2::FrameListener {
 public:
  ~LatestFrameListener() override;

  bool onNewFrame(libfreenect2::Frame::Type type,
                  libfreenect2::Frame* frame) override;

  // Copies the pending frame into `out` and marks it consumed.
  // Returns false if nothing new has arrived since the last take().
  bool take(libfreenect2::Frame::Type type, FrameSnapshot& out);

  StreamStats stats(libfreenect2::Frame::Type type) const;

  // Streams we don't display (IR) are declined so libfreenect2 recycles them.
  void setAccepted(unsigned int type_mask) { accepted_ = type_mask; }

 private:
  struct Slot {
    libfreenect2::Frame* frame = nullptr;  // owned
    bool fresh = false;
    uint32_t last_seq = 0;
    bool seq_seen = false;
    uint64_t delivered = 0, polled = 0, skipped = 0, seq_gaps = 0;
    // Sliding-window rate estimate.
    double win_start = 0;
    uint64_t win_delivered = 0, win_polled = 0;
    double delivered_hz = 0, polled_hz = 0;
  };

  Slot* slotFor(libfreenect2::Frame::Type type);
  const Slot* slotFor(libfreenect2::Frame::Type type) const;

  mutable std::mutex mu_;
  Slot color_, depth_;
  unsigned int accepted_ =
      libfreenect2::Frame::Color | libfreenect2::Frame::Depth;
};

class KinectSource {
 public:
  ~KinectSource();

  enum class UsbReset {
    // Reset the sensor on open. This is the useful default: it clears a sensor
    // left in a weird state by a previous process. It transiently detaches
    // AppleUSBAudio, so the CoreAudio device disappears and comes back a moment
    // later -- start audio *after* opening, with a wait (see AudioCapture::start
    // `wait_seconds`), and the mic array works fine alongside video/depth.
    kReset,
    // Skip the reset. Only useful if the reset itself is what fails, or to keep
    // an already-running audio stream alive across an open. Requires the local
    // libfreenect2 patch in patches/ (applied by setup.sh); without it this
    // degrades to kReset.
    kSkip,
  };

  // `use_opengl` picks the OpenGL depth pipeline (fast) over the CPU one.
  // If the requested reset policy fails to open the device, the other one is
  // tried once before giving up; usedUsbReset() reports what actually happened.
  //
  // On failure returns false and fills `err`.
  bool open(bool use_opengl, UsbReset reset, std::string& err);
  void close();
  bool isOpen() const { return dev_ != nullptr; }

  // Whether the device was actually opened with a USB reset.
  bool usedUsbReset() const { return used_usb_reset_; }

  const std::string& serial() const { return serial_; }
  const std::string& firmware() const { return firmware_; }
  const std::string& pipelineName() const { return pipeline_name_; }

  // Depth (IR) camera intrinsics, for unprojecting depth pixels to metres.
  struct DepthIntrinsics {
    float fx = 0, fy = 0, cx = 0, cy = 0;
    bool valid = false;
  };
  const DepthIntrinsics& depthIntrinsics() const { return intrinsics_; }

  // Target poll rates, in Hz. <= 0 means "every UI frame, unthrottled".
  // These are how often the UI *pulls* a frame; the sensor always runs at its
  // own native rate, so lowering these costs no stability, just freshness.
  void setColorRate(float hz) { color_hz_.store(hz); }
  void setDepthRate(float hz) { depth_hz_.store(hz); }
  void setColorPaused(bool p) { color_paused_.store(p); }
  void setDepthPaused(bool p) { depth_paused_.store(p); }

  // Respecting the configured rate, copy the newest frame into `out`.
  // Returns true only when `out` was refreshed.
  bool pollColor(FrameSnapshot& out);
  bool pollDepth(FrameSnapshot& out);

  StreamStats colorStats() const {
    return listener_.stats(libfreenect2::Frame::Color);
  }
  StreamStats depthStats() const {
    return listener_.stats(libfreenect2::Frame::Depth);
  }

 private:
  bool pollStream(libfreenect2::Frame::Type type, FrameSnapshot& out,
                  std::atomic<float>& hz, std::atomic<bool>& paused,
                  double& next_due);

  bool openOnce(bool use_opengl, bool with_reset, std::string& err);

  libfreenect2::Freenect2 fn2_;
  libfreenect2::Freenect2Device* dev_ = nullptr;
  LatestFrameListener listener_;
  std::string serial_, firmware_, pipeline_name_;
  bool used_usb_reset_ = true;
  DepthIntrinsics intrinsics_;

  std::atomic<float> color_hz_{30.f}, depth_hz_{30.f};
  std::atomic<bool> color_paused_{false}, depth_paused_{false};
  double color_next_due_ = 0, depth_next_due_ = 0;
};

// Seconds since an arbitrary fixed origin (steady clock).
double NowSeconds();
