// Ad hoc verification: does the exciter AGC restore reliable strum triggering
// on quiet real-world program material? Compares strum count and output RMS
// with and without mi::ExciterAGC, using the exact BlockAdapter + Strummer
// pipeline ModalResonatorFX.cpp uses.
//
// Not part of the committed test suite -- a one-off check for the "modal
// plugins sound fucked" investigation.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mi_common/mi_block_adapter.h"
#include "mi_common/mi_exciter_agc.h"

#include "rings/dsp/part.h"
#include "rings/dsp/patch.h"
#include "rings/dsp/performance_state.h"
#include "rings/dsp/strummer.h"

namespace {
const uint32_t kRate = 48000;

bool ReadWavPcm16Mono(const std::string& path, std::vector<float>* out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> data(size);
  fread(data.data(), 1, size, f);
  fclose(f);
  if (size < 44) return false;
  long pos = 12;
  uint16_t channels = 0, bits = 0;
  const uint8_t* pcmData = nullptr;
  uint32_t pcmSize = 0;
  while (pos + 8 <= size) {
    char id[5] = {0};
    memcpy(id, &data[pos], 4);
    uint32_t chunkSize;
    memcpy(&chunkSize, &data[pos + 4], 4);
    if (memcmp(id, "fmt ", 4) == 0) {
      memcpy(&channels, &data[pos + 8 + 2], 2);
      memcpy(&bits, &data[pos + 8 + 14], 2);
    } else if (memcmp(id, "data", 4) == 0) {
      pcmData = &data[pos + 8];
      pcmSize = chunkSize;
    }
    pos += 8 + chunkSize + (chunkSize & 1);
  }
  if (!pcmData || bits != 16) return false;
  size_t numSamples = pcmSize / 2 / channels;
  out->resize(numSamples);
  const int16_t* samples = (const int16_t*)pcmData;
  for (size_t i = 0; i < numSamples; ++i) {
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) sum += samples[i * channels + c] / 32768.0f;
    (*out)[i] = sum / channels;
  }
  return true;
}

void Run(const std::vector<float>& input, bool useAGC, const char* label) {
  static uint16_t reverb[32768];
  memset(reverb, 0, sizeof(reverb));
  static rings::Part part;
  memset((void*)&part, 0, sizeof(part));
  part.Init(reverb);
  rings::Strummer strummer;
  strummer.Init(0.01f, rings::kSampleRate / rings::kMaxBlockSize);
  rings::Patch patch;
  memset(&patch, 0, sizeof(patch));
  patch.structure = 0.4f;
  patch.brightness = 0.5f;
  patch.damping = 0.7f;
  patch.position = 0.3f;
  rings::PerformanceState state;
  memset(&state, 0, sizeof(state));
  state.note = 48.0f;
  state.tonic = 12.0f;
  state.internal_exciter = false;
  state.internal_strum = true;
  state.internal_note = true;
  part.set_polyphony(1);

  mi::BlockAdapter<rings::kMaxBlockSize, 2> adapter;
  mi::ExciterAGC agc;

  uint32_t strumCount = 0;
  double sumSq = 0.0;
  float peak = 0.0f;

  for (size_t i = 0; i < input.size(); ++i) {
    const float x = useAGC ? agc.Process(input[i]) : input[i];
    float wet[2];
    adapter.Tick(x, wet, [&](const float* blk, float* const* o, size_t n) {
      state.strum = false;
      strummer.Process(blk, n, &state);
      if (state.strum) ++strumCount;
      part.Process(state, patch, blk, o[0], o[1], n);
    });
    sumSq += (double)wet[0] * wet[0];
    if (fabsf(wet[0]) > peak) peak = fabsf(wet[0]);
  }

  double rms = sqrt(sumSq / input.size());
  double seconds = (double)input.size() / kRate;
  printf("[%s] strums: %u (%.2f/s), output RMS: %.6f, peak: %.4f\n",
         label, strumCount, strumCount / seconds, rms, peak);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: %s file.wav\n", argv[0]); return 1; }
  std::vector<float> input;
  if (!ReadWavPcm16Mono(argv[1], &input)) { printf("failed to read %s\n", argv[1]); return 1; }
  printf("loaded %s: %.2f s\n", argv[1], (double)input.size() / kRate);

  Run(input, false, "no AGC (old behavior)");
  Run(input, true, "with AGC (fixed)");
  return 0;
}
