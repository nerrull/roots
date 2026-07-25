// 4-channel capture from the Kinect v2 mic array via CoreAudio.
//
// libfreenect2 has no audio support, but it doesn't need to: macOS exposes the
// sensor as an ordinary CoreAudio input device ("Xbox NUI Sensor", 4ch @ 16kHz),
// so we open it with an AUHAL input unit like any other interface.
//
// Device selection is deterministic and never silently falls back. Matching by
// display name is not good enough: names are not unique (two identical USB
// interfaces report the same name), they are localisable, and if the match
// fails, falling back to the *system default input* silently captures whatever
// happens to be default -- a virtual/remote-desktop mic, say. So we key on the
// USB VID:PID carried in the device's ModelUID, verify the selection by reading
// the device back off the audio unit, and treat "not found" as an error.

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "dsp.h"

// The Kinect v2 sensor's USB vendor:product, as CoreAudio spells it in
// kAudioDevicePropertyModelUID ("Xbox NUI Sensor:045E:02C4"). Matches the
// 045e:02c4 reported by USB enumeration.
inline constexpr const char* kKinectV2ModelUid = "045E:02C4";

struct AudioDeviceInfo {
  uint32_t id = 0;
  std::string name;
  std::string uid;        // stable, unique; encodes the USB serial
  std::string model_uid;  // encodes VID:PID; same for every unit of a model
  std::string manufacturer;
  int input_channels = 0;
  double sample_rate = 0;
  bool is_usb = false;
  bool is_default_input = false;
};

// Every device with at least one input channel.
std::vector<AudioDeviceInfo> ListAudioInputDevices();

// How to pick a device, tried in this order. The first non-empty criterion that
// matches wins; a criterion that is set but matches nothing is an error rather
// than a reason to move on, so behaviour never depends on enumeration order.
struct AudioSelector {
  std::string uid;            // exact match on kAudioDevicePropertyDeviceUID
  std::string model_uid;      // substring of ModelUID, e.g. "045E:02C4"
  std::string name;           // substring of the display name (last resort)
  int min_input_channels = 0; // reject a match with fewer inputs than this
  bool allow_default_fallback = false;  // opt in explicitly; off by default
};

class AudioCapture {
 public:
  static constexpr int kMaxChannels = 8;
  // ~2 s of history at 16 kHz. Power of two so the ring wraps with a mask.
  static constexpr int kRingSize = 32768;

  ~AudioCapture();

  // Returns false with `err` describing what was looked for and what was
  // present, so a mis-selection is diagnosable from the message alone.
  //
  // `wait_seconds` > 0 re-polls until the device shows up, which is what you
  // want right after libfreenect2 opens the sensor: its USB reset detaches
  // AppleUSBAudio, and the audio interface takes a moment to re-attach once the
  // device re-enumerates (measured ~0.1 s). Ambiguous matches are not retried
  // -- waiting cannot resolve them.
  bool start(const AudioSelector& sel, std::string& err,
             double wait_seconds = 0.0);
  void stop();
  bool running() const { return running_; }

  // What we actually opened, read back off the audio unit after initialisation
  // rather than remembered from the request.
  const AudioDeviceInfo& device() const { return device_; }
  bool usedFallback() const { return used_fallback_; }
  int channels() const { return channels_; }
  double sampleRate() const { return sample_rate_; }

  // Total input frames delivered, and callbacks that returned an error.
  uint64_t framesCaptured() const { return frames_captured_.load(); }
  uint64_t callbackErrors() const { return callback_errors_.load(); }

  // Copies the most recent `count` samples of `ch` into `out`, oldest first.
  //
  // Lock-free by design: the render callback runs on a realtime thread, where
  // taking a mutex risks priority inversion and audio glitches. A snapshot can
  // therefore tear at the write cursor if it races a callback -- harmless for a
  // scope display, and never unsafe (the ring is fixed-size and preallocated).
  void snapshot(int ch, int count, std::vector<float>& out) const;

  // Peak/RMS over the most recent `count` samples.
  void levels(int ch, int count, float* peak, float* rms) const;

  // All channels reading exact digital silence usually means the process was
  // denied microphone access rather than a dead sensor.
  bool looksLikeSilence() const;

  // --- DSP -------------------------------------------------------------------
  //
  // Parameters are individual atomics rather than a locked struct: the render
  // callback is a realtime thread, so it must not block on the UI. Each atomic
  // is read once per block; a UI edit that lands mid-block simply takes effect
  // on the next one, and no combination of values is unsafe.
  struct Controls {
    std::atomic<bool> enabled{true};

    // Pre-gain, then a highpass to shed room rumble and handling noise, then
    // the compressor, then a safety limiter.
    std::atomic<float> input_gain_db{12.f};
    std::atomic<float> highpass_hz{110.f};
    std::atomic<float> comp_threshold_db{-34.f};
    std::atomic<float> comp_ratio{4.f};
    std::atomic<float> comp_attack_ms{5.f};
    std::atomic<float> comp_release_ms{140.f};
    std::atomic<float> comp_knee_db{8.f};
    std::atomic<float> comp_makeup_db{8.f};
    std::atomic<bool> limiter{true};

    // Delay-and-sum beam. When off, the chain runs on mic 1 alone, which makes
    // the beamformer's contribution easy to A/B.
    std::atomic<bool> beamform{true};
    std::atomic<float> steer_deg{0.f};
    // End-to-end array aperture in metres. Not published for the Kinect v2, so
    // it is adjustable -- see dsp.h and the README on calibrating it.
    std::atomic<float> aperture_m{0.16f};
  };
  Controls controls;

  // Processed mono output of the chain, same ring geometry as the raw channels.
  void beamSnapshot(int count, std::vector<float>& out) const;
  void beamLevels(int count, float* peak, float* rms) const;

  // Gap-free sequential read of the beam ring.
  //
  // beamSnapshot() returns the most recent N samples, which is right for a
  // scope and wrong for anything that must not miss audio: poll it twice and
  // the samples between the two windows are simply gone. A transcriber fed that
  // way drops syllables. So consumers that need continuity hold a cursor
  // instead and drain forward from it.
  //
  // Start with beamCursorNow(), then call beamDrain() as often as you like.
  // Appends everything written since `cursor` to `out` and advances it.
  // A consumer that falls more than kRingSize behind cannot be served the
  // samples it missed -- they have been overwritten -- so those are skipped and
  // counted in the return value rather than silently papered over.
  uint64_t beamCursorNow() const { return write_pos_.load(std::memory_order_acquire); }
  uint64_t beamDrain(uint64_t& cursor, std::vector<float>& out) const;

  // Compressor gain reduction (negative dB) as of the last processed block.
  float gainReductionDb() const { return gain_reduction_db_.load(); }
  // Beam latency introduced by the steering delay lines.
  float beamLatencyMs() const { return beam_latency_ms_.load(); }

 private:
  // Runs the DSP chain over `frames` samples ending at ring position
  // `base + frames`. Called from the render callback only.
  void processBlock(uint64_t base, int frames);
  struct Impl;
  Impl* impl_ = nullptr;

  AudioDeviceInfo device_;
  bool used_fallback_ = false;
  bool running_ = false;
  int channels_ = 0;
  double sample_rate_ = 0;

  std::atomic<uint64_t> frames_captured_{0};
  std::atomic<uint64_t> callback_errors_{0};

  // ring_[ch] is kRingSize floats; write_pos_ counts total samples per channel.
  // beam_ring_ shares that index space and cursor, so a snapshot of the beam
  // lines up sample-for-sample with a snapshot of the raw channels.
  std::vector<float> ring_[kMaxChannels];
  std::vector<float> beam_ring_;
  std::atomic<uint64_t> write_pos_{0};

  // DSP state, touched only by the render callback.
  dsp::Biquad hp_filter_;
  dsp::Compressor compressor_;
  dsp::Limiter limiter_;
  float hp_cached_hz_ = -1.f;
  dsp::BiquadCoeffs hp_coeffs_;

  std::atomic<float> gain_reduction_db_{0.f};
  std::atomic<float> beam_latency_ms_{0.f};

  friend struct AudioCaptureBridge;
};
