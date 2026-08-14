// Offline harness for Elements, mirroring the Wwise effect's pipeline:
// 48 kHz in -> 2/3 downsample -> 16-frame blocks at 32 kHz -> elements::Part
// -> 3/2 upsample -> 48 kHz out, with the same output FIFO.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <vector>

#include "mi_common/mi_resampler.h"

#include "elements/dsp/part.h"
#include "elements/dsp/patch.h"

namespace {

const uint32_t kHostRate = 48000;
const size_t kBlockSize = elements::kMaxBlockSize;
const size_t kFifoSize = 256;

uint16_t g_reverb[32768];

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  // Same reasoning as Clouds: Part is a global in the firmware, so it must be
  // zeroed explicitly when it is not in BSS.
  static elements::Part part;
  memset((void*)&part, 0, sizeof(part));
  memset(g_reverb, 0, sizeof(g_reverb));

  part.Init(g_reverb);
  uint32_t seed[3] = { 0x9e3779b9, 0x243f6a88, 0xb7e15162 };
  part.Seed(seed, 3);

  elements::Patch* p = part.mutable_patch();
  p->exciter_envelope_shape = 0.5f;
  p->exciter_bow_level = 0.0f;
  p->exciter_bow_timbre = 0.5f;
  p->exciter_blow_level = 0.0f;
  p->exciter_blow_meta = 0.5f;
  p->exciter_blow_timbre = 0.5f;
  p->exciter_strike_level = 0.6f;
  p->exciter_strike_meta = 0.5f;
  p->exciter_strike_timbre = 0.5f;
  p->exciter_signature = 0.0f;
  p->resonator_geometry = 0.4f;
  p->resonator_brightness = 0.5f;
  p->resonator_damping = 0.7f;
  p->resonator_position = 0.3f;
  p->resonator_modulation_frequency = 0.5f;
  p->resonator_modulation_offset = 0.0f;
  p->reverb_diffusion = 0.625f;
  p->reverb_lp = 0.7f;
  p->space = 0.1f;
  p->modulation_frequency = 0.0f;

  elements::PerformanceState state;
  memset(&state, 0, sizeof(state));
  state.note = 48.0f;
  state.modulation = 0.0f;
  state.strength = 0.7f;
  state.gate = false;

  mi::HostToModule48to32 down;
  mi::ModuleToHost32to48 up[2];

  float blowIn[kBlockSize], strikeIn[kBlockSize];
  float mainOut[kBlockSize], auxOut[kBlockSize];
  memset(blowIn, 0, sizeof(blowIn));
  memset(strikeIn, 0, sizeof(strikeIn));
  memset(mainOut, 0, sizeof(mainOut));
  memset(auxOut, 0, sizeof(auxOut));
  size_t inPos = 0;

  static float fifo[2][kFifoSize];
  memset(fifo, 0, sizeof(fifo));
  size_t rd = 0, wr = 24;

  const uint32_t frames = kHostRate * 6;
  float peak = 0.0f;
  bool finite = true;
  uint32_t underruns = 0;

  const clock_t t0 = clock();

  for (uint32_t i = 0; i < frames; ++i) {
    // A short noise burst every second, as an impact would arrive on a bus.
    const uint32_t phase = i % kHostRate;
    float in = 0.0f;
    if (phase < kHostRate / 250) {
      in = 0.5f * (2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f);
    }

    // Open the gate briefly with each burst so the exciter envelope fires.
    state.gate = (phase < kHostRate / 100);

    down.Push(in);

    float x;
    while (down.Pop(&x)) {
      blowIn[inPos] = x;
      strikeIn[inPos] = x;
      if (++inPos == kBlockSize) {
        inPos = 0;
        part.Process(state, blowIn, strikeIn, mainOut, auxOut, kBlockSize);
        for (size_t k = 0; k < kBlockSize; ++k) {
          up[0].Push(mainOut[k]);
          up[1].Push(auxOut[k]);
          float ol, orr;
          while (up[0].Pop(&ol)) {
            if (!up[1].Pop(&orr)) orr = ol;
            fifo[0][wr] = ol;
            fifo[1][wr] = orr;
            wr = (wr + 1) & (kFifoSize - 1);
          }
        }
      }
    }

    const size_t count = (wr - rd) & (kFifoSize - 1);
    float wl = 0.0f, wrr = 0.0f;
    if (count > 0) {
      wl = fifo[0][rd];
      wrr = fifo[1][rd];
      rd = (rd + 1) & (kFifoSize - 1);
    } else if (i > 2000) {
      ++underruns;
    }

    if (!std::isfinite(wl) || !std::isfinite(wrr)) finite = false;
    const float a = fabsf(wl);
    if (a > peak) peak = a;
  }

  const double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
  printf("elements: peak=%.4f underruns=%u finite=%d (%.2fs cpu for %.0fs audio)\n",
         peak, underruns, (int)finite, secs, (double)frames / kHostRate);

  if (!finite || !(peak > 1e-4f) || peak > 8.0f || underruns != 0) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
