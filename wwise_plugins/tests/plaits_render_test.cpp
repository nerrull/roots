// Offline harness for the Plaits core, mirroring the Wwise source plug-in's
// signal path. Renders a triggered note from every engine and checks each one
// produces finite, audible output.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "stmlib/utils/buffer_allocator.h"
#include "plaits/dsp/voice.h"

namespace {

const uint32_t kSampleRate = 48000;

void WriteWav(const char* path, const std::vector<float>& mono, uint32_t rate) {
  FILE* fp = fopen(path, "wb");
  if (!fp) { perror("fopen"); return; }
  const uint32_t frames = (uint32_t)mono.size();
  uint32_t l; uint16_t s;
  fwrite("RIFF", 4, 1, fp);
  l = 36 + frames * 2; fwrite(&l, 4, 1, fp);
  fwrite("WAVE", 4, 1, fp);
  fwrite("fmt ", 4, 1, fp);
  l = 16; fwrite(&l, 4, 1, fp);
  s = 1; fwrite(&s, 2, 1, fp);
  s = 1; fwrite(&s, 2, 1, fp);
  l = rate; fwrite(&l, 4, 1, fp);
  l = rate * 2; fwrite(&l, 4, 1, fp);
  s = 2; fwrite(&s, 2, 1, fp);
  s = 16; fwrite(&s, 2, 1, fp);
  fwrite("data", 4, 1, fp);
  l = frames * 2; fwrite(&l, 4, 1, fp);
  for (size_t i = 0; i < mono.size(); ++i) {
    float v = mono[i];
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    int16_t q = (int16_t)(v * 32767.0f);
    fwrite(&q, 2, 1, fp);
  }
  fclose(fp);
}

}  // namespace

int main(int argc, char** argv) {
  const char* out_path = argc > 1 ? argv[1] : "plaits_out.wav";

  static uint8_t arena[16384];
  std::vector<float> all;
  int failures = 0;

  for (int engine = 0; engine < plaits::kMaxEngines; ++engine) {
    memset(arena, 0, sizeof(arena));
    stmlib::BufferAllocator allocator(arena, sizeof(arena));

    plaits::Voice voice;
    voice.Init(&allocator);

    plaits::Patch patch;
    plaits::Modulations modulations;
    memset(&patch, 0, sizeof(patch));
    memset(&modulations, 0, sizeof(modulations));

    patch.note = 60.0f;
    patch.harmonics = 0.5f;
    patch.timbre = 0.5f;
    patch.morph = 0.5f;
    patch.decay = 0.5f;
    patch.lpg_colour = 0.5f;
    patch.engine = engine;

    modulations.trigger_patched = true;
    modulations.trigger = 0.0f;

    // 1 second per engine.
    const uint32_t blocks = kSampleRate / plaits::kBlockSize;
    plaits::Voice::Frame frames[plaits::kBlockSize];

    float peak = 0.0f;
    bool finite = true;

    for (uint32_t b = 0; b < blocks; ++b) {
      // Held gate, matching the plug-in: percussive engines take the rising
      // edge, the six-op FM engines need the level to stay high.
      modulations.trigger = (b > 0) ? 1.0f : 0.0f;
      voice.Render(patch, modulations, frames, plaits::kBlockSize);
      for (size_t i = 0; i < plaits::kBlockSize; ++i) {
        const float v = (float)frames[i].out / 32768.0f;
        if (!std::isfinite(v)) finite = false;
        const float a = fabsf(v);
        if (a > peak) peak = a;
        all.push_back(v);
      }
    }

    const bool ok = finite && peak > 1e-4f && peak <= 4.0f;
    printf("engine %2d (active=%2d): peak=%.4f %s\n",
           engine, voice.active_engine(), peak, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
  }

  WriteWav(out_path, all, kSampleRate);
  printf("wrote %s (%zu frames)\n", out_path, all.size());

  if (failures) {
    printf("FAIL: %d engine(s) produced no usable output\n", failures);
    return 1;
  }
  printf("PASS: all %d engines rendered\n", plaits::kMaxEngines);
  return 0;
}
