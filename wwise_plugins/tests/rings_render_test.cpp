// Offline harness that drives the Rings core through the same BlockAdapter the
// Wwise plug-in uses, so the DSP path can be verified without loading Wwise.
//
// Renders a series of short noise bursts (standing in for impacts on a bus)
// into a stereo WAV and reports peak/RMS so a silent or exploding build is
// obvious.
//
// Build: see tests/build_tests.sh

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "mi_common/mi_block_adapter.h"

#include "rings/dsp/part.h"
#include "rings/dsp/patch.h"
#include "rings/dsp/performance_state.h"
#include "rings/dsp/strummer.h"

namespace {

const uint32_t kSampleRate = 48000;
const size_t kBlockSize = rings::kMaxBlockSize;

void WriteWav(const char* path, const std::vector<float>& interleaved, int channels, uint32_t rate) {
  FILE* fp = fopen(path, "wb");
  if (!fp) { perror("fopen"); exit(1); }
  const uint32_t frames = (uint32_t)(interleaved.size() / channels);
  uint32_t l; uint16_t s;
  fwrite("RIFF", 4, 1, fp);
  l = 36 + frames * 2 * channels; fwrite(&l, 4, 1, fp);
  fwrite("WAVE", 4, 1, fp);
  fwrite("fmt ", 4, 1, fp);
  l = 16; fwrite(&l, 4, 1, fp);
  s = 1; fwrite(&s, 2, 1, fp);
  s = (uint16_t)channels; fwrite(&s, 2, 1, fp);
  l = rate; fwrite(&l, 4, 1, fp);
  l = rate * 2 * channels; fwrite(&l, 4, 1, fp);
  s = (uint16_t)(2 * channels); fwrite(&s, 2, 1, fp);
  s = 16; fwrite(&s, 2, 1, fp);
  fwrite("data", 4, 1, fp);
  l = frames * 2 * channels; fwrite(&l, 4, 1, fp);
  for (size_t i = 0; i < interleaved.size(); ++i) {
    float v = interleaved[i];
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    int16_t q = (int16_t)(v * 32767.0f);
    fwrite(&q, 2, 1, fp);
  }
  fclose(fp);
}

}  // namespace

int main(int argc, char** argv) {
  const char* out_path = argc > 1 ? argv[1] : "rings_out.wav";

  // 64 KB, matching what the plug-in allocates through Wwise.
  static uint16_t reverb_buffer[32768];
  memset(reverb_buffer, 0, sizeof(reverb_buffer));

  rings::Part part;
  rings::Strummer strummer;
  rings::Patch patch;
  rings::PerformanceState state;
  mi::BlockAdapter<kBlockSize, 2> adapter;

  part.Init(reverb_buffer);
  strummer.Init(0.01f, rings::kSampleRate / kBlockSize);
  memset(&patch, 0, sizeof(patch));
  memset(&state, 0, sizeof(state));

  patch.structure = 0.25f;
  patch.brightness = 0.5f;
  patch.damping = 0.7f;
  patch.position = 0.25f;

  state.note = 60.0f;
  state.tonic = 12.0f;
  state.fm = 0.0f;
  state.chord = 0;
  state.internal_exciter = false;
  state.internal_strum = true;
  state.internal_note = true;

  part.set_model(rings::RESONATOR_MODEL_MODAL);
  part.set_polyphony(1);

  auto render = [&](const float* in, float* const* out, size_t n) {
    state.strum = false;
    strummer.Process(in, n, &state);
    part.Process(state, patch, in, out[0], out[1], n);
  };

  const uint32_t duration = kSampleRate * 8;
  std::vector<float> interleaved;
  interleaved.reserve(duration * 2);

  float peak = 0.0f;
  double sumsq = 0.0;

  srand(1);
  for (uint32_t i = 0; i < duration; ++i) {
    // A 4 ms noise burst every second: a percussive exciter.
    const uint32_t phase = i % kSampleRate;
    float in = 0.0f;
    if (phase < kSampleRate / 250) {
      in = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
      in *= 0.5f;
    }

    float wet[2];
    adapter.Tick(in, wet, render);
    interleaved.push_back(wet[0]);
    interleaved.push_back(wet[1]);

    for (int c = 0; c < 2; ++c) {
      float a = fabsf(wet[c]);
      if (a > peak) peak = a;
      sumsq += (double)wet[c] * wet[c];
    }
  }

  const double rms = sqrt(sumsq / (duration * 2));
  printf("rendered %u frames, peak=%.4f rms=%.5f -> %s\n", duration, peak, rms, out_path);

  if (!(peak > 1e-4)) {
    printf("FAIL: output is silent\n");
    return 1;
  }
  if (!std::isfinite(peak) || peak > 8.0f) {
    printf("FAIL: output diverged (peak=%f)\n", peak);
    return 1;
  }
  WriteWav(out_path, interleaved, 2, kSampleRate);
  printf("PASS\n");
  return 0;
}
