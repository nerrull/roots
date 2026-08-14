// Rational sample-rate conversion between the host rate and the rate an MI
// module was written for.
//
// Clouds and Elements both run their DSP at 32 kHz. At a 48 kHz host rate that
// is exactly 2/3, so a polyphase rational resampler converts both ways with no
// drift and no interpolation error accumulating over time. Running the modules
// at the host rate instead would transpose them by a fifth and scale every
// delay and decay time by 1.5, which is not something a pitch offset can undo.
//
// The design is the textbook interpolate-by-L / lowpass / decimate-by-M, with
// the filter evaluated in polyphase form so the zero-stuffed samples are never
// actually multiplied:
//
//     y[n] = sum_k h[k*L + phase] * x[floor(n*M/L) - k],  phase = (n*M) mod L
//
// Push input samples one at a time; Pop drains however many output samples have
// become available (0, 1 or more, depending on the phase).

#ifndef MI_RESAMPLER_H_
#define MI_RESAMPLER_H_

#include <cmath>
#include <cstddef>
#include <cstring>

namespace mi {

// L: interpolation factor. M: decimation factor. kTapsPerPhase: filter length
// per polyphase branch, i.e. how many input samples each output looks at.
template <int L, int M, int kTapsPerPhase = 24>
class PolyphaseResampler {
 public:
  PolyphaseResampler() {
    DesignFilter();
    Reset();
  }

  void Reset() {
    memset(x_, 0, sizeof(x_));
    next_out_ = 0;
    input_index_ = -1;
  }

  // Feeds one input sample. Afterwards, Pop() until it returns false.
  inline void Push(float in) {
    // Newest sample first; the history is short enough that shifting beats the
    // bookkeeping of a ring buffer.
    for (int k = kTapsPerPhase - 1; k > 0; --k) {
      x_[k] = x_[k - 1];
    }
    x_[0] = in;
    ++input_index_;

    // Keep the counters small. Advancing the output index by L moves the input
    // position by exactly M, so subtracting this pair leaves both the phase and
    // the input/output alignment untouched.
    if (next_out_ >= kRenormalize * L) {
      next_out_ -= kRenormalize * L;
      input_index_ -= kRenormalize * M;
    }
  }

  // Emits every output sample whose source position has arrived. Returns false
  // once the caller has drained them all.
  inline bool Pop(float* out) {
    if ((next_out_ * M) / L > input_index_) {
      return false;
    }

    const int phase = (int)((next_out_ * M) % L);

    const float* h = &h_[phase * kTapsPerPhase];
    float acc = 0.0f;
    for (int k = 0; k < kTapsPerPhase; ++k) {
      acc += h[k] * x_[k];
    }
    *out = acc;

    ++next_out_;
    return true;
  }

 private:
  // Windowed-sinc lowpass at the intermediate rate (fs_in * L), stored already
  // split into polyphase branches so Pop() is a straight dot product.
  void DesignFilter() {
    const int kTotalTaps = L * kTapsPerPhase;

    // Cut below both Nyquist limits, with margin for the transition band.
    const double decimation = (L > M) ? L : M;
    const double cutoff = 0.45 / decimation;  // cycles/sample at the intermediate rate

    double taps[L * kTapsPerPhase];
    double sum = 0.0;
    const double center = 0.5 * (kTotalTaps - 1);

    for (int i = 0; i < kTotalTaps; ++i) {
      const double t = i - center;

      double sinc;
      if (fabs(t) < 1e-9) {
        sinc = 2.0 * cutoff;
      } else {
        sinc = sin(2.0 * M_PI * cutoff * t) / (M_PI * t);
      }

      // Blackman window: the stopband rejection matters more here than a tight
      // transition, since anything left above Nyquist folds straight back in.
      const double w = 0.42
          - 0.5 * cos(2.0 * M_PI * i / (kTotalTaps - 1))
          + 0.08 * cos(4.0 * M_PI * i / (kTotalTaps - 1));

      taps[i] = sinc * w;
      sum += taps[i];
    }

    // Normalize to unity passband gain, then scale by L to make up for the
    // zeros introduced by upsampling.
    const double scale = (sum != 0.0) ? ((double)L / sum) : 1.0;

    for (int phase = 0; phase < L; ++phase) {
      for (int k = 0; k < kTapsPerPhase; ++k) {
        const int idx = k * L + phase;
        h_[phase * kTapsPerPhase + k] =
            (idx < kTotalTaps) ? (float)(taps[idx] * scale) : 0.0f;
      }
    }
  }

  static const long kRenormalize = 1 << 20;

  float h_[L * kTapsPerPhase];
  float x_[kTapsPerPhase];

  long next_out_;
  long input_index_;
};

// 48 kHz host -> 32 kHz module, and back.
typedef PolyphaseResampler<2, 3> HostToModule48to32;
typedef PolyphaseResampler<3, 2> ModuleToHost32to48;

}  // namespace mi

#endif  // MI_RESAMPLER_H_
