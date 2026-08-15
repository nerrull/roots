// Reproduces exactly what ModalVoiceFX.cpp does -- Init(), UpdatePatch()'s
// default patch, the resampler+block-adapter pipeline, External Exciter on --
// fed with a single short click (simulating a one-shot event hitting the bus)
// to see whether the resonator actually produces audible output.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mi_common/mi_resampler.h"

#include "elements/dsp/part.h"
#include "elements/dsp/patch.h"

namespace {
const uint32_t kRate = 48000;
const size_t kBlockSize = elements::kMaxBlockSize;

// Minimal mono 16-bit PCM WAV reader, for feeding a real test sound in
// instead of a synthetic click.
bool ReadWavPcm16Mono(const std::string& path, std::vector<float>* out, uint32_t* rate) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> data(size);
  fread(data.data(), 1, size, f);
  fclose(f);

  if (size < 44 || memcmp(&data[0], "RIFF", 4) != 0 || memcmp(&data[8], "WAVE", 4) != 0) {
    return false;
  }
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
      memcpy(rate, &data[pos + 8 + 4], 4);
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
    for (int c = 0; c < channels; ++c) {
      sum += samples[i * channels + c] / 32768.0f;
    }
    (*out)[i] = sum / channels;
  }
  return true;
}

void WriteWav(const char* path, const std::vector<float>& s, uint32_t rate) {
  FILE* f = fopen(path, "wb");
  uint32_t dataBytes = (uint32_t)(s.size() * sizeof(float));
  uint32_t byteRate = rate * sizeof(float);
  uint16_t blockAlign = sizeof(float), bits = 32, fmt = 3, ch = 1;
  uint32_t fmtSize = 16, riffSize = 4 + 8 + fmtSize + 8 + dataBytes;
  fwrite("RIFF", 1, 4, f); fwrite(&riffSize, 4, 1, f); fwrite("WAVE", 1, 4, f);
  fwrite("fmt ", 1, 4, f); fwrite(&fmtSize, 4, 1, f); fwrite(&fmt, 2, 1, f);
  fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f); fwrite(&byteRate, 4, 1, f);
  fwrite(&blockAlign, 2, 1, f); fwrite(&bits, 2, 1, f);
  fwrite("data", 1, 4, f); fwrite(&dataBytes, 4, 1, f);
  fwrite(s.data(), sizeof(float), s.size(), f);
  fclose(f);
}
}  // namespace

int main(int argc, char** argv) {
  static elements::Part part;
  memset((void*)&part, 0, sizeof(part));
  static uint16_t reverb[32768];
  memset(reverb, 0, sizeof(reverb));
  part.Init(reverb);
  uint32_t seed[3] = { 0x9e3779b9, 0x243f6a88, 0xb7e15162 };
  part.Seed(seed, 3);

  // Exact defaults from ModalVoiceFXParams::Init(block==0) / UpdatePatch().
  elements::Patch* p = part.mutable_patch();
  p->exciter_envelope_shape = 0.5f;   // fContour
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
  p->space = 0.1f;                    // fSpace
  p->modulation_frequency = 0.0f;

  elements::PerformanceState state;
  memset(&state, 0, sizeof(state));
  state.note = 48.0f;                 // fPitch
  state.modulation = 0.0f;
  state.strength = 0.7f;              // fStrength
  state.gate = false;                 // NonRTPC.bGate default

  std::vector<float> input;
  if (argc > 1) {
    uint32_t fileRate = 0;
    if (!ReadWavPcm16Mono(argv[1], &input, &fileRate)) {
      printf("failed to read %s\n", argv[1]);
      return 1;
    }
    if (fileRate != kRate) {
      printf("warning: file is %u Hz, expected %u Hz (no rate conversion applied)\n", fileRate, kRate);
    }
    printf("loaded %s: %zu frames (%.2f s)\n", argv[1], input.size(), (double)input.size() / kRate);
  } else {
    // Default: 0.5s of silence, then a single-sample unit-amplitude click,
    // then 2s of silence -- simulating a one-shot sound event hitting the bus.
    size_t totalIn = kRate * 3;
    input.assign(totalIn, 0.0f);
    input[kRate / 2] = 1.0f;
  }
  size_t totalIn = input.size();

  mi::HostToModule48to32 down;
  mi::ModuleToHost32to48 up[2];

  float blowIn[kBlockSize], strikeIn[kBlockSize];
  float mainOut[kBlockSize], auxOut[kBlockSize];
  memset(blowIn, 0, sizeof(blowIn));
  memset(strikeIn, 0, sizeof(strikeIn));
  size_t inPos = 0;

  std::vector<float> outMain, outAux;
  outMain.reserve(totalIn);
  outAux.reserve(totalIn);

  float peakOut = 0.0f;
  size_t peakOutIdx = 0;

  for (size_t i = 0; i < totalIn; ++i) {
    down.Push(input[i]);
    float x32;
    while (down.Pop(&x32)) {
      // External Exciter on: same signal to both, per ModalVoiceFX.cpp.
      blowIn[inPos] = x32;
      strikeIn[inPos] = x32;
      if (++inPos == kBlockSize) {
        inPos = 0;
        part.Process(state, blowIn, strikeIn, mainOut, auxOut, kBlockSize);
        for (size_t k = 0; k < kBlockSize; ++k) {
          up[0].Push(mainOut[k]);
          up[1].Push(auxOut[k]);
          float m, a;
          while (up[0].Pop(&m)) {
            if (!up[1].Pop(&a)) a = m;
            outMain.push_back(m);
            outAux.push_back(a);
            if (fabsf(m) > peakOut) { peakOut = fabsf(m); peakOutIdx = outMain.size(); }
          }
        }
      }
    }
  }

  double rmsOut = 0.0, rmsIn = 0.0;
  for (float v : outMain) rmsOut += v * v;
  for (float v : input) rmsIn += v * v;
  rmsOut = sqrt(rmsOut / outMain.size());
  rmsIn = sqrt(rmsIn / input.size());

  printf("output frames: %zu\n", outMain.size());
  printf("peak output magnitude: %.6f at sample %zu (%.1f ms)\n",
         peakOut, peakOutIdx, 1000.0 * peakOutIdx / kRate);
  printf("input RMS: %.6f, output RMS: %.6f\n", rmsIn, rmsOut);

  WriteWav("Q:/Development/git/roots/wwise_plugins/.analysis_out/modalvoice_click_main.wav", outMain, kRate);
  WriteWav("Q:/Development/git/roots/wwise_plugins/.analysis_out/modalvoice_click_aux.wav", outAux, kRate);

  return 0;
}
