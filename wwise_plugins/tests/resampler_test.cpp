// Checks the 48k<->32k polyphase resampler: rate ratios, round-trip fidelity
// on in-band content, and rejection of content above the 32 kHz Nyquist.

#include <cmath>
#include <cstdio>
#include <vector>

#include "mi_common/mi_resampler.h"

namespace {

const double kPi = 3.14159265358979323846;

// Measures the amplitude of a sine at frequency f in a signal, by projecting
// onto it (skipping the start so filter ramp-up doesn't count).
double Amplitude(const std::vector<float>& x, double f, double rate) {
  const size_t skip = x.size() / 4;
  double re = 0.0, im = 0.0;
  size_t n = 0;
  for (size_t i = skip; i < x.size(); ++i, ++n) {
    const double t = 2.0 * kPi * f * (double)i / rate;
    re += x[i] * cos(t);
    im += x[i] * sin(t);
  }
  return 2.0 * sqrt(re * re + im * im) / (double)n;
}

int failures = 0;

void Check(bool ok, const char* what, double value) {
  printf("  %-46s %10.5f  %s\n", what, value, ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

}  // namespace

int main() {
  const double kHostRate = 48000.0;
  const double kModuleRate = 32000.0;
  const size_t kFrames = 48000;

  // ---- rate ratio ----------------------------------------------------------
  {
    mi::HostToModule48to32 down;
    size_t produced = 0;
    for (size_t i = 0; i < kFrames; ++i) {
      down.Push(0.0f);
      float y;
      while (down.Pop(&y)) ++produced;
    }
    const double ratio = (double)produced / (double)kFrames;
    printf("48k -> 32k\n");
    Check(fabs(ratio - 2.0 / 3.0) < 1e-3, "output/input ratio (want 0.66667)", ratio);
  }
  {
    mi::ModuleToHost32to48 up;
    size_t produced = 0;
    const size_t in = 32000;
    for (size_t i = 0; i < in; ++i) {
      up.Push(0.0f);
      float y;
      while (up.Pop(&y)) ++produced;
    }
    const double ratio = (double)produced / (double)in;
    printf("32k -> 48k\n");
    Check(fabs(ratio - 1.5) < 1e-3, "output/input ratio (want 1.50000)", ratio);
  }

  // ---- round trip on an in-band tone ---------------------------------------
  for (double freq : {220.0, 1000.0, 5000.0}) {
    mi::HostToModule48to32 down;
    mi::ModuleToHost32to48 up;

    std::vector<float> out;
    out.reserve(kFrames);

    for (size_t i = 0; i < kFrames; ++i) {
      const float in = (float)sin(2.0 * kPi * freq * (double)i / kHostRate);
      down.Push(in);
      float mid;
      while (down.Pop(&mid)) {
        up.Push(mid);
        float y;
        while (up.Pop(&y)) out.push_back(y);
      }
    }

    const double amp = Amplitude(out, freq, kHostRate);
    char label[80];
    snprintf(label, sizeof(label), "round-trip amplitude @ %.0f Hz (want 1.0)", freq);
    printf("round trip\n");
    Check(fabs(amp - 1.0) < 0.05, label, amp);
  }

  // ---- aliasing rejection --------------------------------------------------
  // 20 kHz is above the 16 kHz Nyquist of the 32 kHz module rate. Without a
  // proper anti-alias filter it would fold down to 12 kHz and be plainly
  // audible; it must be strongly attenuated instead.
  {
    mi::HostToModule48to32 down;
    std::vector<float> mid;
    mid.reserve(kFrames);

    for (size_t i = 0; i < kFrames; ++i) {
      const float in = (float)sin(2.0 * kPi * 20000.0 * (double)i / kHostRate);
      down.Push(in);
      float y;
      while (down.Pop(&y)) mid.push_back(y);
    }

    // 20 kHz sampled at 32 kHz folds to |20000 - 32000| = 12000 Hz.
    const double fold = Amplitude(mid, 12000.0, kModuleRate);
    const double dB = 20.0 * log10(fold + 1e-12);
    printf("aliasing\n");
    Check(dB < -50.0, "folded 20 kHz image, dB (want < -50)", dB);
  }

  if (failures) {
    printf("\nFAIL: %d check(s) failed\n", failures);
    return 1;
  }
  printf("\nPASS: resampler checks passed\n");
  return 0;
}
