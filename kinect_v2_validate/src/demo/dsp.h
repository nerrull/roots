// Realtime DSP primitives for the Kinect v2 mic array.
//
// Everything here is allocation-free and branch-light so it can run inside the
// CoreAudio render callback. No state is shared between instances, and no
// function here locks, allocates, or logs.
//
// Written from standard formulations rather than pulled in as a dependency:
// ODAS (the usual open-source choice for array DOA/beamforming) drags in
// libconfig, FFTW and its own audio I/O for what amounts to a few hundred lines
// of textbook DSP on a 4-mic array. Sources for the formulations used:
//   * RBJ audio-EQ-cookbook biquad coefficients.
//   * Soft-knee compressor curve: Reiss & McPherson, "Audio Effects" (2014),
//     eq. 6.5 -- the standard piecewise-quadratic knee.
//   * Delay-and-sum + steered response power: Benesty, Chen & Huang,
//     "Microphone Array Signal Processing" (2008), ch. 3 and 9.

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace dsp {

inline float DbToLin(float db) { return std::pow(10.f, db * 0.05f); }
inline float LinToDb(float x) {
  return 20.f * std::log10(std::max(x, 1e-9f));
}

// --- Biquad ------------------------------------------------------------------

struct BiquadCoeffs {
  float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
};

BiquadCoeffs HighpassCoeffs(float fs, float fc, float q = 0.7071f);
BiquadCoeffs LowpassCoeffs(float fs, float fc, float q = 0.7071f);

// Transposed direct form II: one multiply-add chain, good float behaviour.
class Biquad {
 public:
  void reset() { z1_ = z2_ = 0.f; }
  inline float process(float x, const BiquadCoeffs& c) {
    const float y = c.b0 * x + z1_;
    z1_ = c.b1 * x - c.a1 * y + z2_;
    z2_ = c.b2 * x - c.a2 * y;
    return y;
  }

 private:
  float z1_ = 0, z2_ = 0;
};

// --- Compressor --------------------------------------------------------------

struct CompressorParams {
  float threshold_db = -32.f;
  float ratio = 4.f;
  float attack_ms = 5.f;
  float release_ms = 120.f;
  float knee_db = 6.f;   // total width of the soft knee, centred on threshold
  float makeup_db = 6.f;
};

// Feed-forward peak compressor: detect level, compute target gain from the
// static curve, smooth that gain with attack/release, apply. Smoothing the gain
// (rather than the detector) is what keeps it from pumping on transients.
class Compressor {
 public:
  void prepare(float fs) {
    fs_ = std::max(fs, 1.f);
    reset();
  }
  void reset() { gr_db_ = 0.f; }

  inline float process(float x, const CompressorParams& p) {
    const float level_db = LinToDb(std::fabs(x));
    const float target_db = StaticCurveGainDb(level_db, p);

    // Attack when we need *more* reduction, release when we need less.
    const float tau_ms = (target_db < gr_db_) ? p.attack_ms : p.release_ms;
    const float coef =
        std::exp(-1.f / std::max(tau_ms * 0.001f * fs_, 1.f));
    gr_db_ = target_db + (gr_db_ - target_db) * coef;

    return x * DbToLin(gr_db_ + p.makeup_db);
  }

  // Negative dB, or 0 when not compressing.
  float gainReductionDb() const { return gr_db_; }

  // The static (unsmoothed) curve, exposed for tests and for drawing it.
  static float StaticCurveGainDb(float level_db, const CompressorParams& p);

 private:
  float fs_ = 16000.f;
  float gr_db_ = 0.f;
};

// Brick-wall-ish safety limiter: a fast compressor at a high ratio plus a hard
// clamp, so nothing can leave the chain outside [-1, 1].
class Limiter {
 public:
  void prepare(float fs) { comp_.prepare(fs); }
  void reset() { comp_.reset(); }

  inline float process(float x, float ceiling_db = -1.f) {
    CompressorParams p;
    p.threshold_db = ceiling_db;
    p.ratio = 50.f;
    p.attack_ms = 1.f;
    p.release_ms = 60.f;
    p.knee_db = 1.f;
    p.makeup_db = 0.f;
    const float y = comp_.process(x, p);
    const float ceil_lin = DbToLin(ceiling_db);
    return std::min(std::max(y, -ceil_lin), ceil_lin);
  }

  float gainReductionDb() const { return comp_.gainReductionDb(); }

 private:
  Compressor comp_;
};

// --- Array geometry ----------------------------------------------------------

// Mic positions along the array axis, in metres, 0 = array centre, +x = right
// as seen from the sensor looking out at the scene.
struct MicArray {
  static constexpr int kMaxMics = 8;
  int count = 4;
  float x_m[kMaxMics] = {0, 0, 0, 0, 0, 0, 0, 0};
  float speed_of_sound = 343.f;

  float maxAbsX() const {
    float m = 0.f;
    for (int i = 0; i < count; ++i) m = std::max(m, std::fabs(x_m[i]));
    return m;
  }

  // Highest frequency free of spatial aliasing, for the *smallest* spacing.
  float aliasingFreeHz() const;
};

// Uniform linear array of `count` mics spanning `aperture_m` end to end.
//
// NOTE: the Kinect v2's true mic positions are not published, and libfreenect2
// does not model audio at all, so the default aperture is an estimate based on
// the 249 mm sensor bar. Beam steering is only as accurate as this number --
// calibrate it by putting a talker at a known angle and matching the mic DOA
// readout to the depth-tracked azimuth (the demo shows both side by side).
MicArray UniformLinearArray(int count, float aperture_m);
MicArray KinectV2MicArray(float aperture_m = 0.16f);

// --- Fractional delay --------------------------------------------------------

// 4-point Lagrange interpolation. `taps` are samples at integer positions
// -1, 0, +1, +2 relative to the target; `frac` in [0,1) selects between taps[1]
// and taps[2]. Cubic keeps the beamformer's high end intact, which linear
// interpolation audibly dulls.
inline float Lagrange4(const float taps[4], float frac) {
  const float d1 = frac - 1.f;
  const float d2 = frac - 2.f;
  const float dp1 = frac + 1.f;
  const float c0 = -frac * d1 * d2 / 6.f;
  const float c1 = dp1 * d1 * d2 * 0.5f;
  const float c2 = -dp1 * frac * d2 * 0.5f;
  const float c3 = dp1 * frac * d1 / 6.f;
  return c0 * taps[0] + c1 * taps[1] + c2 * taps[2] + c3 * taps[3];
}

// Per-mic delays, in samples, that time-align a plane wave arriving from
// `azimuth_deg` (0 = straight ahead, positive = toward +x).
//
// A mic nearer the source hears it earlier, so aligning means delaying that mic
// more. A common bulk offset is added so every delay is >= 0 and can be read as
// history; the beam output is therefore latent by that offset (~0.5 ms here).
// Returns the bulk offset actually used.
float SteeringDelays(const MicArray& arr, float azimuth_deg, float fs,
                     float* delays_out);

// --- Direction of arrival ----------------------------------------------------

struct DoaResult {
  float azimuth_deg = 0.f;
  float confidence = 0.f;  // 0 = flat response, 1 = sharply peaked
  bool valid = false;
};

// Steered response power: delay-and-sum toward each candidate angle and keep
// the loudest. Time domain, so no FFT dependency; with 4 mics and a few dozen
// angles this is a fraction of a millisecond per estimate.
//
// Feed it band-limited audio (speech band, and below aliasingFreeHz()) --
// broadband input makes the response surface ambiguous.
class SrpDoa {
 public:
  void configure(const MicArray& arr, float fs, float min_deg = -90.f,
                 float max_deg = 90.f, float step_deg = 3.f);

  // `channels[c]` holds `n` samples for mic c. Needs n > ~128 to be meaningful.
  DoaResult estimate(const float* const* channels, int count, int n) const;

  int angleCount() const { return static_cast<int>(angles_.size()); }
  float angleAt(int i) const { return angles_[i]; }
  // Normalised response from the last estimate(), for plotting.
  const std::vector<float>& lastResponse() const { return response_; }

 private:
  std::vector<float> angles_;
  std::vector<float> delays_;  // angleCount() * mics, samples
  mutable std::vector<float> response_;
  int mics_ = 0;
  int guard_ = 0;  // samples to skip at the block start for delay history
};

}  // namespace dsp
