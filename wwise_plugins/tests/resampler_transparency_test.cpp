// Isolates the resampler + block adapter from the MI DSP entirely: feeds a
// sine sweep and white noise through HostToModule48to32 -> BlockAdapter
// (identity passthrough, no MI core) -> ModuleToHost32to48, and writes both
// the dry and round-tripped signal to WAV so a spectral analysis can tell
// whether the *glue code* (not the MI algorithms) is introducing artifacts.
//
// Usage: resampler_transparency_test <output-dir>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mi_common/mi_resampler.h"
#include "mi_common/mi_block_adapter.h"

namespace {

const uint32_t kRate = 48000;

void WriteWav(const std::string& path, const std::vector<float>& samples, uint32_t rate) {
  FILE* f = fopen(path.c_str(), "wb");
  uint32_t dataBytes = (uint32_t)(samples.size() * sizeof(float));
  uint32_t byteRate = rate * sizeof(float);
  uint16_t blockAlign = sizeof(float);
  uint16_t bitsPerSample = 32;
  uint16_t audioFormat = 3;  // IEEE float
  uint32_t fmtSize = 16;
  uint32_t riffSize = 4 + (8 + fmtSize) + (8 + dataBytes);

  fwrite("RIFF", 1, 4, f);
  fwrite(&riffSize, 4, 1, f);
  fwrite("WAVE", 1, 4, f);
  fwrite("fmt ", 1, 4, f);
  fwrite(&fmtSize, 4, 1, f);
  fwrite(&audioFormat, 2, 1, f);
  uint16_t channels = 1;
  fwrite(&channels, 2, 1, f);
  fwrite(&rate, 4, 1, f);
  fwrite(&byteRate, 4, 1, f);
  fwrite(&blockAlign, 2, 1, f);
  fwrite(&bitsPerSample, 2, 1, f);
  fwrite("data", 1, 4, f);
  fwrite(&dataBytes, 4, 1, f);
  fwrite(samples.data(), sizeof(float), samples.size(), f);
  fclose(f);
}

// Exponential sweep 20 Hz -> 20 kHz over durationSec.
std::vector<float> MakeSweep(double durationSec, uint32_t rate) {
  size_t n = (size_t)(durationSec * rate);
  std::vector<float> out(n);
  double f0 = 20.0, f1 = 20000.0;
  double k = log(f1 / f0) / durationSec;
  for (size_t i = 0; i < n; ++i) {
    double t = (double)i / rate;
    double phase = 2.0 * M_PI * f0 * (exp(k * t) - 1.0) / k;
    out[i] = 0.5f * (float)sin(phase);
  }
  return out;
}

std::vector<float> MakeNoise(double durationSec, uint32_t rate) {
  size_t n = (size_t)(durationSec * rate);
  std::vector<float> out(n);
  uint32_t s = 12345;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    out[i] = ((float)(int32_t)s / 2147483648.0f) * 0.5f;
  }
  return out;
}

// Round trip through the resampler pair only (no block adapter, no MI core):
// 48k -> down to 32k -> back up to 48k. Isolates the polyphase filter itself.
std::vector<float> RoundTripResamplerOnly(const std::vector<float>& in) {
  mi::HostToModule48to32 down;
  mi::ModuleToHost32to48 up;
  std::vector<float> out;
  out.reserve(in.size());
  float sample32;
  for (float x : in) {
    down.Push(x);
    while (down.Pop(&sample32)) {
      up.Push(sample32);
      float sample48;
      while (up.Pop(&sample48)) {
        out.push_back(sample48);
      }
    }
  }
  return out;
}

// Round trip through resampler + block adapter with an identity "renderer"
// (copies its input block straight to its output block), matching exactly
// what GranularTextureFX/ModalVoiceFX do around the MI core, minus the core.
std::vector<float> RoundTripWithBlockAdapter(const std::vector<float>& in) {
  const size_t kBlock = 32;
  mi::HostToModule48to32 down;
  mi::ModuleToHost32to48 up;
  mi::BlockAdapter<kBlock, 1> adapter;

  auto identity = [](const float* inBlock, float* const* outBlock, size_t n) {
    memcpy(outBlock[0], inBlock, n * sizeof(float));
  };

  std::vector<float> out;
  out.reserve(in.size());
  float sample32;
  for (float x : in) {
    down.Push(x);
    while (down.Pop(&sample32)) {
      float adapterOut;
      adapter.Tick(sample32, &adapterOut, identity);
      up.Push(adapterOut);
      float sample48;
      while (up.Pop(&sample48)) {
        out.push_back(sample48);
      }
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string outDir = argc > 1 ? argv[1] : ".";

  std::vector<float> sweep = MakeSweep(3.0, kRate);
  std::vector<float> noise = MakeNoise(1.0, kRate);

  WriteWav(outDir + "/sweep_dry.wav", sweep, kRate);
  WriteWav(outDir + "/sweep_resampler_only.wav", RoundTripResamplerOnly(sweep), kRate);
  WriteWav(outDir + "/sweep_resampler_blockadapter.wav", RoundTripWithBlockAdapter(sweep), kRate);

  WriteWav(outDir + "/noise_dry.wav", noise, kRate);
  WriteWav(outDir + "/noise_resampler_only.wav", RoundTripResamplerOnly(noise), kRate);
  WriteWav(outDir + "/noise_resampler_blockadapter.wav", RoundTripWithBlockAdapter(noise), kRate);

  printf("wrote WAVs to %s\n", outDir.c_str());
  return 0;
}
