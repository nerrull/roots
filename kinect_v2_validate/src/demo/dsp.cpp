#include "dsp.h"

#include <limits>

namespace dsp {

// --- Biquad ------------------------------------------------------------------

// RBJ audio-EQ-cookbook, normalised so a0 == 1.
BiquadCoeffs HighpassCoeffs(float fs, float fc, float q) {
  BiquadCoeffs c;
  fc = std::min(std::max(fc, 1.f), fs * 0.49f);
  const float w0 = 2.f * float(M_PI) * fc / fs;
  const float cw = std::cos(w0), sw = std::sin(w0);
  const float alpha = sw / (2.f * std::max(q, 0.01f));
  const float a0 = 1.f + alpha;
  c.b0 = ((1.f + cw) * 0.5f) / a0;
  c.b1 = (-(1.f + cw)) / a0;
  c.b2 = c.b0;
  c.a1 = (-2.f * cw) / a0;
  c.a2 = (1.f - alpha) / a0;
  return c;
}

BiquadCoeffs LowpassCoeffs(float fs, float fc, float q) {
  BiquadCoeffs c;
  fc = std::min(std::max(fc, 1.f), fs * 0.49f);
  const float w0 = 2.f * float(M_PI) * fc / fs;
  const float cw = std::cos(w0), sw = std::sin(w0);
  const float alpha = sw / (2.f * std::max(q, 0.01f));
  const float a0 = 1.f + alpha;
  c.b0 = ((1.f - cw) * 0.5f) / a0;
  c.b1 = (1.f - cw) / a0;
  c.b2 = c.b0;
  c.a1 = (-2.f * cw) / a0;
  c.a2 = (1.f - alpha) / a0;
  return c;
}

// --- Compressor --------------------------------------------------------------

float Compressor::StaticCurveGainDb(float level_db, const CompressorParams& p) {
  const float ratio = std::max(p.ratio, 1.f);
  const float knee = std::max(p.knee_db, 0.f);
  const float over = level_db - p.threshold_db;

  float out_db;
  if (knee > 0.f && 2.f * over > -knee && 2.f * over < knee) {
    // Quadratic interpolation across the knee.
    const float t = over + knee * 0.5f;
    out_db = level_db + (1.f / ratio - 1.f) * t * t / (2.f * knee);
  } else if (over <= 0.f) {
    out_db = level_db;  // below threshold: unity
  } else {
    out_db = p.threshold_db + over / ratio;
  }
  return std::min(out_db - level_db, 0.f);
}

// --- Array geometry ----------------------------------------------------------

float MicArray::aliasingFreeHz() const {
  if (count < 2) return speed_of_sound;
  float min_spacing = std::numeric_limits<float>::max();
  for (int i = 1; i < count; ++i) {
    min_spacing = std::min(min_spacing, std::fabs(x_m[i] - x_m[i - 1]));
  }
  if (!(min_spacing > 0.f)) return speed_of_sound;
  return speed_of_sound / (2.f * min_spacing);
}

MicArray UniformLinearArray(int count, float aperture_m) {
  MicArray a;
  a.count = std::min(std::max(count, 1), MicArray::kMaxMics);
  if (a.count == 1) {
    a.x_m[0] = 0.f;
    return a;
  }
  const float step = aperture_m / float(a.count - 1);
  for (int i = 0; i < a.count; ++i) {
    a.x_m[i] = -aperture_m * 0.5f + step * float(i);
  }
  return a;
}

MicArray KinectV2MicArray(float aperture_m) {
  return UniformLinearArray(4, aperture_m);
}

// --- Steering ----------------------------------------------------------------

float SteeringDelays(const MicArray& arr, float azimuth_deg, float fs,
                     float* delays_out) {
  const float s = std::sin(azimuth_deg * float(M_PI) / 180.f);
  // Bulk offset large enough that every per-mic delay stays non-negative, plus
  // one sample of margin for the interpolator's taps[0] at position -1.
  const float bulk = arr.maxAbsX() / arr.speed_of_sound * fs + 2.f;
  for (int c = 0; c < arr.count; ++c) {
    delays_out[c] = bulk + (arr.x_m[c] * s / arr.speed_of_sound) * fs;
  }
  return bulk;
}

// --- SRP DOA -----------------------------------------------------------------

void SrpDoa::configure(const MicArray& arr, float fs, float min_deg,
                       float max_deg, float step_deg) {
  angles_.clear();
  delays_.clear();
  mics_ = arr.count;
  step_deg = std::max(step_deg, 0.25f);

  for (float a = min_deg; a <= max_deg + 1e-3f; a += step_deg) {
    angles_.push_back(a);
  }
  delays_.resize(angles_.size() * size_t(mics_));

  float max_delay = 0.f;
  for (size_t i = 0; i < angles_.size(); ++i) {
    SteeringDelays(arr, angles_[i], fs, &delays_[i * size_t(mics_)]);
    for (int c = 0; c < mics_; ++c) {
      max_delay = std::max(max_delay, delays_[i * size_t(mics_) + size_t(c)]);
    }
  }
  guard_ = int(std::ceil(max_delay)) + 2;
  response_.assign(angles_.size(), 0.f);
}

DoaResult SrpDoa::estimate(const float* const* channels, int count,
                           int n) const {
  DoaResult r;
  if (angles_.empty() || count < 2 || mics_ < 2) return r;
  const int mics = std::min(count, mics_);
  const int start = guard_;
  if (n - start < 64) return r;  // too little data to be meaningful

  response_.assign(angles_.size(), 0.f);

  float best_power = -1.f;
  int best_idx = 0;
  for (size_t ai = 0; ai < angles_.size(); ++ai) {
    const float* d = &delays_[ai * size_t(mics_)];
    float power = 0.f;
    for (int i = start; i < n; ++i) {
      float sum = 0.f;
      for (int c = 0; c < mics; ++c) {
        const float pos = float(i) - d[c];
        const int ip = int(pos);
        const float frac = pos - float(ip);
        // ip-1 >= 0 is guaranteed by the guard band.
        const float taps[4] = {channels[c][ip - 1], channels[c][ip],
                               channels[c][ip + 1], channels[c][ip + 2]};
        sum += Lagrange4(taps, frac);
      }
      power += sum * sum;
    }
    response_[ai] = power;
    if (power > best_power) {
      best_power = power;
      best_idx = int(ai);
    }
  }

  // Confidence: how far the peak stands above the average response. A diffuse
  // or silent field gives a flat surface and so a near-zero score.
  double mean = 0.0;
  for (float p : response_) mean += p;
  mean /= double(response_.size());
  r.confidence = (best_power > 0.f)
                     ? float((best_power - mean) / double(best_power))
                     : 0.f;

  // Parabolic interpolation across the peak for sub-grid resolution.
  float az = angles_[size_t(best_idx)];
  if (best_idx > 0 && best_idx + 1 < int(angles_.size())) {
    const float y0 = response_[size_t(best_idx - 1)];
    const float y1 = response_[size_t(best_idx)];
    const float y2 = response_[size_t(best_idx + 1)];
    const float denom = y0 - 2.f * y1 + y2;
    if (std::fabs(denom) > 1e-12f) {
      const float delta = 0.5f * (y0 - y2) / denom;
      if (std::fabs(delta) <= 1.f) {
        const float step = angles_[1] - angles_[0];
        az += delta * step;
      }
    }
  }

  // Normalise the response for plotting.
  if (best_power > 0.f) {
    for (float& p : response_) p /= best_power;
  }

  r.azimuth_deg = az;
  r.valid = true;
  return r;
}

}  // namespace dsp
