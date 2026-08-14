// Offline harness for the Peaks core, mirroring the Wwise source plug-in's
// gate handling. Triggers each percussion model and checks it sounds.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "stmlib/utils/gate_flags.h"
#include "peaks/processors.h"

namespace {

const uint32_t kSampleRate = 48000;
const size_t kChunk = 64;

struct Model { int id; peaks::ProcessorFunction fn; const char* name; };

const Model kModels[] = {
  { 0, peaks::PROCESSOR_FUNCTION_BASS_DRUM,  "bass drum" },
  { 1, peaks::PROCESSOR_FUNCTION_SNARE_DRUM, "snare"     },
  { 2, peaks::PROCESSOR_FUNCTION_HIGH_HAT,   "hi-hat"    },
  { 3, peaks::PROCESSOR_FUNCTION_FM_DRUM,    "fm drum"   },
};

}  // namespace

int main() {
  int failures = 0;
  const uint32_t gate_len = (uint32_t)(0.002f * kSampleRate);

  for (size_t m = 0; m < sizeof(kModels) / sizeof(kModels[0]); ++m) {
    peaks::Processors proc;
    proc.Init(0);
    proc.set_control_mode(peaks::CONTROL_MODE_FULL);
    proc.set_function(kModels[m].fn);
    for (uint8_t i = 0; i < 4; ++i) {
      proc.set_parameter(i, 32768);
    }

    stmlib::GateFlags flags[kChunk];
    int16_t out[kChunk];

    float peak = 0.0f;
    bool finite = true;
    const uint32_t total = kSampleRate;  // 1 s

    for (uint32_t s = 0; s < total; s += kChunk) {
      const size_t n = (total - s) < kChunk ? (total - s) : kChunk;
      for (size_t i = 0; i < n; ++i) {
        const uint32_t t = s + (uint32_t)i;
        if (t == 0) flags[i] = stmlib::GATE_FLAG_RISING | stmlib::GATE_FLAG_HIGH;
        else if (t < gate_len) flags[i] = stmlib::GATE_FLAG_HIGH;
        else if (t == gate_len) flags[i] = stmlib::GATE_FLAG_FALLING;
        else flags[i] = stmlib::GATE_FLAG_LOW;
      }
      proc.Process(flags, out, n);
      for (size_t i = 0; i < n; ++i) {
        const float v = (float)out[i] / 32768.0f;
        if (!std::isfinite(v)) finite = false;
        const float a = fabsf(v);
        if (a > peak) peak = a;
      }
    }

    const bool ok = finite && peak > 1e-3f && peak <= 4.0f;
    printf("model %d %-10s peak=%.4f %s\n", kModels[m].id, kModels[m].name, peak, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
  }

  if (failures) { printf("FAIL: %d model(s) silent\n", failures); return 1; }
  printf("PASS: all percussion models rendered\n");
  return 0;
}
