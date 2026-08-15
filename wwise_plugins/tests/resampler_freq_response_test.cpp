// Stationary-tone frequency response of the resampler round trip (no block
// adapter): for each test frequency, render N cycles, measure the output
// amplitude at that exact frequency via a Goertzel-style DFT bin, relative to
// the input amplitude. Cleaner than a sweep for pinning down where the
// passband actually stops being flat.

#include <cmath>
#include <cstdio>
#include <vector>

#include "mi_common/mi_resampler.h"

namespace {
const uint32_t kRate = 48000;

std::vector<float> RoundTrip(const std::vector<float>& in) {
  mi::HostToModule48to32 down;
  mi::ModuleToHost32to48 up;
  std::vector<float> out;
  out.reserve(in.size());
  float s32;
  for (float x : in) {
    down.Push(x);
    while (down.Pop(&s32)) {
      up.Push(s32);
      float s48;
      while (up.Pop(&s48)) out.push_back(s48);
    }
  }
  return out;
}

// Goertzel magnitude at freq, over the last nSamples of x (skips the first
// portion to let the resampler's group delay/settling pass).
double GoertzelMag(const std::vector<float>& x, double freq, uint32_t rate, size_t skip) {
  size_t n = x.size() > skip ? x.size() - skip : 0;
  if (n == 0) return 0.0;
  double w = 2.0 * M_PI * freq / rate;
  double coeff = 2.0 * cos(w);
  double s0 = 0, s1 = 0, s2 = 0;
  for (size_t i = 0; i < n; ++i) {
    s0 = x[skip + i] + coeff * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  double real = s1 - s2 * cos(w);
  double imag = s2 * sin(w);
  return 2.0 * sqrt(real * real + imag * imag) / n;
}

}  // namespace

int main() {
  double freqs[] = {200, 500, 1000, 2000, 4000, 6000, 8000, 9000, 10000,
                     11000, 12000, 13000, 14000, 15000, 16000};
  double durationSec = 0.5;
  size_t n = (size_t)(durationSec * kRate);

  printf("%8s %10s %10s %10s\n", "freq_Hz", "in_mag", "out_mag", "gain_dB");
  for (double f : freqs) {
    std::vector<float> in(n);
    for (size_t i = 0; i < n; ++i) {
      in[i] = 0.5f * (float)sin(2.0 * M_PI * f * i / kRate);
    }
    std::vector<float> out = RoundTrip(in);
    size_t skip = out.size() / 4;  // let group delay settle
    double inMag = GoertzelMag(in, f, kRate, in.size() / 4);
    double outMag = GoertzelMag(out, f, kRate, skip);
    double gainDb = 20.0 * log10((outMag + 1e-12) / (inMag + 1e-12));
    printf("%8.0f %10.5f %10.5f %10.2f\n", f, inMag, outMag, gainDb);
  }
  return 0;
}
