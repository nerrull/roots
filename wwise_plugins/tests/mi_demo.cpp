// Renders demo audio for the four plug-ins that are hard to judge from a
// pass/fail number: the three effects and the drum synth.
//
// Drum Synth plays a pattern, and that pattern is then used as the source
// material for the three effects, which is how they are meant to be used --
// inserted on a bus carrying real content.
//
// Each render drives the MI core through the same wrappers the plug-in uses.
//
// Build: see tests/run_demo.sh

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mi_common/mi_block_adapter.h"
#include "mi_common/mi_resampler.h"

#include "peaks/processors.h"
#include "stmlib/utils/gate_flags.h"

#include "rings/dsp/part.h"
#include "rings/dsp/patch.h"
#include "rings/dsp/performance_state.h"
#include "rings/dsp/strummer.h"

#include "clouds/dsp/granular_processor.h"
#include "clouds/dsp/frame.h"

#include "elements/dsp/part.h"
#include "elements/dsp/patch.h"

namespace {

const uint32_t kRate = 48000;

// ---------------------------------------------------------------------------

void WriteWav(const std::string& path, const std::vector<float>& interleaved, int channels) {
  FILE* fp = fopen(path.c_str(), "wb");
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
  l = kRate; fwrite(&l, 4, 1, fp);
  l = kRate * 2 * channels; fwrite(&l, 4, 1, fp);
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
  printf("  wrote %s (%.1f s)\n", path.c_str(), (double)frames / kRate);
}

float Peak(const std::vector<float>& x) {
  float p = 0.0f;
  for (size_t i = 0; i < x.size(); ++i) {
    const float a = fabsf(x[i]);
    if (a > p) p = a;
  }
  return p;
}

// ---------------------------------------------------------------------------
// Drum Synth: a pattern, one Peaks processor per voice.

struct Hit { uint32_t step; int model; float p1, p2, p3, p4, gain; };

std::vector<float> RenderDrums() {
  const double kBPM = 112.0;
  const uint32_t kStepsPerBar = 16;              // sixteenth notes
  const uint32_t kBars = 8;
  const uint32_t samplesPerStep = (uint32_t)(kRate * 60.0 / kBPM / 4.0);
  const uint32_t totalSteps = kBars * kStepsPerBar;
  const uint32_t frames = totalSteps * samplesPerStep + kRate;  // + tail

  // model: 0 bass drum, 1 snare, 2 hi-hat, 3 fm drum
  // Peaks' four knobs mean different things per model; these are chosen by ear.
  std::vector<Hit> pattern;
  for (uint32_t bar = 0; bar < kBars; ++bar) {
    const uint32_t b = bar * kStepsPerBar;

    // Kick: four on the floor, with a push into bar 2 and 6.
    pattern.push_back({ b + 0,  0, 0.22f, 0.62f, 0.35f, 0.55f, 1.00f });
    pattern.push_back({ b + 6,  0, 0.22f, 0.55f, 0.35f, 0.45f, 0.70f });
    pattern.push_back({ b + 8,  0, 0.22f, 0.62f, 0.35f, 0.55f, 0.95f });
    if (bar % 2 == 1) {
      pattern.push_back({ b + 14, 0, 0.26f, 0.50f, 0.40f, 0.40f, 0.60f });
    }

    // Snare on 2 and 4, with ghost notes.
    pattern.push_back({ b + 4,  1, 0.40f, 0.55f, 0.50f, 0.45f, 0.90f });
    pattern.push_back({ b + 12, 1, 0.40f, 0.55f, 0.50f, 0.45f, 0.90f });
    pattern.push_back({ b + 7,  1, 0.40f, 0.30f, 0.50f, 0.25f, 0.22f });
    pattern.push_back({ b + 15, 1, 0.40f, 0.30f, 0.50f, 0.25f, 0.28f });

    // Hats on eighths, accented on the beat, opening up in the last bar.
    for (uint32_t s = 0; s < kStepsPerBar; s += 2) {
      const bool accent = (s % 4) == 0;
      const float decay = (bar == kBars - 1) ? 0.62f : 0.30f;
      pattern.push_back({ b + s, 2, 0.55f, 0.50f, 0.55f, decay,
                          accent ? 0.42f : 0.24f });
    }

    // An FM drum answering every other bar: the metallic one.
    if (bar % 2 == 1) {
      pattern.push_back({ b + 10, 3, 0.45f, 0.60f, 0.55f, 0.55f, 0.55f });
    }
    if (bar == kBars - 1) {
      pattern.push_back({ b + 13, 3, 0.60f, 0.70f, 0.65f, 0.70f, 0.70f });
    }
  }

  // One processor per model so overlapping hits do not cut each other off.
  const int kVoices = 4;
  peaks::Processors proc[kVoices];
  const peaks::ProcessorFunction fns[kVoices] = {
    peaks::PROCESSOR_FUNCTION_BASS_DRUM,
    peaks::PROCESSOR_FUNCTION_SNARE_DRUM,
    peaks::PROCESSOR_FUNCTION_HIGH_HAT,
    peaks::PROCESSOR_FUNCTION_FM_DRUM,
  };
  for (int v = 0; v < kVoices; ++v) {
    memset((void*)&proc[v], 0, sizeof(proc[v]));
    proc[v].Init(0);
    proc[v].set_control_mode(peaks::CONTROL_MODE_FULL);
    proc[v].set_function(fns[v]);
  }

  std::vector<float> out(frames, 0.0f);
  const uint32_t gateLen = (uint32_t)(0.002f * kRate);

  // Render each voice across the whole timeline, summing as we go.
  for (int v = 0; v < kVoices; ++v) {
    // Gate timeline and per-hit gain for this voice.
    std::vector<uint8_t> gate(frames, 0);
    std::vector<float> gain(frames, 0.0f);
    float currentGain = 1.0f;

    for (size_t h = 0; h < pattern.size(); ++h) {
      if (pattern[h].model != v) continue;
      const uint32_t t = pattern[h].step * samplesPerStep;
      if (t >= frames) continue;
      for (uint32_t i = 0; i < gateLen && t + i < frames; ++i) {
        gate[t + i] = 1;
      }
      gain[t] = pattern[h].gain;
    }

    // Knob values are taken from the most recent hit of this voice.
    float p1 = 0.5f, p2 = 0.5f, p3 = 0.5f, p4 = 0.5f;
    size_t nextHit = 0;
    std::vector<const Hit*> mine;
    for (size_t h = 0; h < pattern.size(); ++h) {
      if (pattern[h].model == v) mine.push_back(&pattern[h]);
    }

    const size_t kChunk = 64;
    stmlib::GateFlags flags[kChunk];
    int16_t samples[kChunk];
    bool prevHigh = false;

    for (uint32_t i = 0; i < frames; i += kChunk) {
      const size_t n = (frames - i) < kChunk ? (frames - i) : kChunk;

      // Apply the knob settings of any hit starting in this chunk.
      while (nextHit < mine.size() &&
             mine[nextHit]->step * samplesPerStep < i + n) {
        p1 = mine[nextHit]->p1; p2 = mine[nextHit]->p2;
        p3 = mine[nextHit]->p3; p4 = mine[nextHit]->p4;
        ++nextHit;
      }
      proc[v].set_parameter(0, (uint16_t)(p1 * 65535.0f));
      proc[v].set_parameter(1, (uint16_t)(p2 * 65535.0f));
      proc[v].set_parameter(2, (uint16_t)(p3 * 65535.0f));
      proc[v].set_parameter(3, (uint16_t)(p4 * 65535.0f));

      for (size_t k = 0; k < n; ++k) {
        const bool high = gate[i + k] != 0;
        if (high && !prevHigh) {
          flags[k] = stmlib::GATE_FLAG_RISING | stmlib::GATE_FLAG_HIGH;
          if (gain[i + k] > 0.0f) currentGain = gain[i + k];
        } else if (high) {
          flags[k] = stmlib::GATE_FLAG_HIGH;
        } else if (!high && prevHigh) {
          flags[k] = stmlib::GATE_FLAG_FALLING;
        } else {
          flags[k] = stmlib::GATE_FLAG_LOW;
        }
        prevHigh = high;
      }

      proc[v].Process(flags, samples, n);

      for (size_t k = 0; k < n; ++k) {
        out[i + k] += (float)samples[k] / 32768.0f * currentGain * 0.5f;
      }
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Rings: steps through all six resonator models, driven by the drum loop.

std::vector<float> RenderRings(const std::vector<float>& in) {
  static uint16_t reverb[32768];
  memset(reverb, 0, sizeof(reverb));

  static rings::Part part;
  memset((void*)&part, 0, sizeof(part));
  rings::Strummer strummer;
  rings::Patch patch;
  rings::PerformanceState state;
  memset(&patch, 0, sizeof(patch));
  memset(&state, 0, sizeof(state));

  part.Init(reverb);
  strummer.Init(0.01f, rings::kSampleRate / rings::kMaxBlockSize);

  state.tonic = 12.0f;
  state.internal_exciter = false;
  state.internal_strum = true;
  state.internal_note = true;
  part.set_polyphony(2);

  mi::BlockAdapter<rings::kMaxBlockSize, 2> adapter;

  const char* names[6] = { "modal", "sympathetic string", "string",
                           "FM voice", "quantized sympathetic", "string + reverb" };
  const uint32_t secsPerModel = 5;
  const uint32_t framesPerModel = kRate * secsPerModel;

  std::vector<float> out;
  out.reserve(framesPerModel * 6 * 2);

  for (int model = 0; model < 6; ++model) {
    printf("    model %d: %s\n", model, names[model]);
    part.set_model((rings::ResonatorModel)model);

    for (uint32_t i = 0; i < framesPerModel; ++i) {
      const float t = (float)i / kRate;

      // Sweep the macro controls so each model shows its range.
      patch.structure = 0.15f + 0.7f * (0.5f - 0.5f * cosf(2.0f * M_PI * t / secsPerModel));
      patch.brightness = 0.35f + 0.45f * (0.5f + 0.5f * sinf(2.0f * M_PI * t / 3.0f));
      patch.damping = 0.75f;
      patch.position = 0.15f + 0.5f * (0.5f + 0.5f * sinf(2.0f * M_PI * t / 4.3f));

      // Walk the note around a minor pentatonic so it is musical, not static.
      static const float scale[5] = { 0.0f, 3.0f, 5.0f, 7.0f, 10.0f };
      const int step = ((int)(t * 2.0f)) % 5;
      state.note = 48.0f + scale[step];
      state.chord = (int32_t)(t) % rings::kNumChords;

      const size_t src = (size_t)((i + model * framesPerModel) % in.size());
      const float x = in[src] * 0.8f;

      float wet[2];
      adapter.Tick(x, wet, [&](const float* blk, float* const* o, size_t n) {
        state.strum = false;
        strummer.Process(blk, n, &state);
        part.Process(state, patch, blk, o[0], o[1], n);
      });

      out.push_back(wet[0]);
      out.push_back(wet[1]);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Clouds: each playback mode in turn, with the controls moving.

std::vector<float> RenderClouds(const std::vector<float>& in) {
  static uint8_t large[118784];
  static uint8_t small[65536 - 128];

  const int modes[4] = { clouds::PLAYBACK_MODE_GRANULAR,
                         clouds::PLAYBACK_MODE_STRETCH,
                         clouds::PLAYBACK_MODE_LOOPING_DELAY,
                         clouds::PLAYBACK_MODE_SPECTRAL };
  const char* names[4] = { "granular", "stretch", "looping delay", "spectral" };
  const uint32_t secsPerMode = 10;
  const uint32_t framesPerMode = kRate * secsPerMode;

  std::vector<float> out;
  out.reserve(framesPerMode * 4 * 2);

  for (int mi_ = 0; mi_ < 4; ++mi_) {
    const int mode = modes[mi_];
    printf("    mode %d: %s\n", mode, names[mi_]);

    memset(large, 0, sizeof(large));
    memset(small, 0, sizeof(small));

    static clouds::GranularProcessor proc;
    memset((void*)&proc, 0, sizeof(proc));
    proc.Init(large, sizeof(large), small, sizeof(small));
    proc.set_playback_mode((clouds::PlaybackMode)mode);
    proc.set_quality(0);

    clouds::Parameters* p = proc.mutable_parameters();
    p->dry_wet = 1.0f;
    p->stereo_spread = 0.7f;
    p->feedback = 0.0f;
    p->reverb = 0.25f;
    p->freeze = false;
    p->trigger = false;
    p->gate = false;

    mi::HostToModule48to32 down[2];
    mi::ModuleToHost32to48 up[2];

    clouds::ShortFrame inBlock[32], outBlock[32];
    memset(inBlock, 0, sizeof(inBlock));
    memset(outBlock, 0, sizeof(outBlock));
    size_t inPos = 0;

    const size_t kFifo = 256;
    static float fifo[2][256];
    memset(fifo, 0, sizeof(fifo));
    size_t rd = 0, wr = 48;

    for (uint32_t i = 0; i < framesPerMode; ++i) {
      const float t = (float)i / kRate;

      // Grain position drifts back through the buffer while size and density
      // sweep, so you hear the buffer being scanned.
      p->position = 0.5f + 0.45f * sinf(2.0f * M_PI * t / secsPerMode);
      p->size = 0.25f + 0.6f * (0.5f + 0.5f * sinf(2.0f * M_PI * t / 6.0f));
      // Density is bipolar with a dead zone at exactly 0.5 (grains fire only
      // from the trigger input there), so this sweeps the dense half.
      p->density = 0.62f + 0.36f * (0.5f + 0.5f * sinf(2.0f * M_PI * t / 4.0f));
      p->texture = 0.3f + 0.5f * (0.5f + 0.5f * cosf(2.0f * M_PI * t / 7.0f));

      // Drop an octave for the second half of each mode.
      p->pitch = (t > secsPerMode * 0.6f) ? -12.0f : 0.0f;

      // Freeze near the end so the buffer keeps playing without new input.
      p->freeze = (t > secsPerMode * 0.82f);

      const size_t src = (size_t)((i + mi_ * framesPerMode) % in.size());
      const float x = in[src] * 0.8f;

      down[0].Push(x);
      down[1].Push(x);

      float l, r;
      while (down[0].Pop(&l)) {
        if (!down[1].Pop(&r)) r = l;
        inBlock[inPos].l = (short)(fmaxf(-1.0f, fminf(1.0f, l)) * 32767.0f);
        inBlock[inPos].r = (short)(fmaxf(-1.0f, fminf(1.0f, r)) * 32767.0f);
        if (++inPos == 32) {
          inPos = 0;
          proc.Prepare();
          proc.Process(inBlock, outBlock, 32);
          for (size_t k = 0; k < 32; ++k) {
            up[0].Push((float)outBlock[k].l / 32768.0f);
            up[1].Push((float)outBlock[k].r / 32768.0f);
            float ol, orr;
            while (up[0].Pop(&ol)) {
              if (!up[1].Pop(&orr)) orr = ol;
              fifo[0][wr] = ol; fifo[1][wr] = orr;
              wr = (wr + 1) & (kFifo - 1);
            }
          }
        }
      }

      float wl = 0.0f, wrr = 0.0f;
      if (((wr - rd) & (kFifo - 1)) > 0) {
        wl = fifo[0][rd]; wrr = fifo[1][rd];
        rd = (rd + 1) & (kFifo - 1);
      }
      out.push_back(wl);
      out.push_back(wrr);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Elements: struck, then bowed, then blown, sweeping the resonator.

std::vector<float> RenderElements(const std::vector<float>& in) {
  static uint16_t reverb[32768];
  memset(reverb, 0, sizeof(reverb));

  static elements::Part part;
  memset((void*)&part, 0, sizeof(part));
  part.Init(reverb);
  uint32_t seed[3] = { 0x9e3779b9, 0x243f6a88, 0xb7e15162 };
  part.Seed(seed, 3);

  elements::Patch* p = part.mutable_patch();
  p->exciter_bow_timbre = 0.5f;
  p->exciter_blow_timbre = 0.5f;
  p->exciter_strike_timbre = 0.5f;
  p->exciter_signature = 0.2f;
  p->exciter_envelope_shape = 0.5f;
  p->resonator_modulation_frequency = 0.5f;
  p->resonator_modulation_offset = 0.0f;
  p->modulation_frequency = 0.0f;
  p->reverb_diffusion = 0.625f;
  p->reverb_lp = 0.7f;
  p->space = 0.3f;

  elements::PerformanceState state;
  memset(&state, 0, sizeof(state));
  state.modulation = 0.0f;
  state.strength = 0.8f;

  mi::HostToModule48to32 down;
  mi::ModuleToHost32to48 up[2];

  float blowIn[elements::kMaxBlockSize], strikeIn[elements::kMaxBlockSize];
  float mainOut[elements::kMaxBlockSize], auxOut[elements::kMaxBlockSize];
  memset(blowIn, 0, sizeof(blowIn));
  memset(strikeIn, 0, sizeof(strikeIn));
  memset(mainOut, 0, sizeof(mainOut));
  memset(auxOut, 0, sizeof(auxOut));
  size_t inPos = 0;

  const size_t kFifo = 256;
  static float fifo[2][256];
  memset(fifo, 0, sizeof(fifo));
  size_t rd = 0, wr = 24;

  struct Section { const char* name; float bow, blow, strike; uint32_t secs; };
  const Section sections[3] = {
    { "struck (drums into the exciter)", 0.0f, 0.0f, 0.75f, 10 },
    { "bowed",                           0.7f, 0.0f, 0.0f,  8 },
    { "blown",                           0.0f, 0.7f, 0.15f, 8 },
  };

  std::vector<float> out;
  uint32_t elapsed = 0;

  for (int s = 0; s < 3; ++s) {
    printf("    %s\n", sections[s].name);
    p->exciter_bow_level = sections[s].bow;
    p->exciter_blow_level = sections[s].blow;
    p->exciter_strike_level = sections[s].strike;
    p->exciter_blow_meta = 0.45f;
    p->exciter_strike_meta = 0.55f;

    const uint32_t frames = kRate * sections[s].secs;
    for (uint32_t i = 0; i < frames; ++i) {
      const float t = (float)i / kRate;

      p->resonator_geometry = 0.15f + 0.7f * (0.5f + 0.5f * sinf(2.0f * M_PI * t / 9.0f));
      p->resonator_brightness = 0.3f + 0.5f * (0.5f + 0.5f * sinf(2.0f * M_PI * t / 5.0f));
      p->resonator_damping = 0.6f + 0.35f * (0.5f + 0.5f * cosf(2.0f * M_PI * t / 11.0f));
      p->resonator_position = 0.1f + 0.4f * (0.5f + 0.5f * sinf(2.0f * M_PI * t / 3.7f));

      static const float scale[5] = { 0.0f, 3.0f, 5.0f, 7.0f, 10.0f };
      state.note = 36.0f + scale[((int)(t * 1.5f)) % 5];

      // The bowed and blown sections need the gate held; the struck section is
      // excited by the drum audio itself.
      state.gate = (s != 0) || (fmodf(t, 1.0f) < 0.02f);

      const size_t src = (size_t)((elapsed + i) % in.size());
      const float x = (s == 0) ? in[src] * 0.9f : in[src] * 0.15f;

      down.Push(x);

      float v;
      while (down.Pop(&v)) {
        blowIn[inPos] = v;
        strikeIn[inPos] = v;
        if (++inPos == elements::kMaxBlockSize) {
          inPos = 0;
          part.Process(state, blowIn, strikeIn, mainOut, auxOut,
                       elements::kMaxBlockSize);
          for (size_t k = 0; k < elements::kMaxBlockSize; ++k) {
            up[0].Push(mainOut[k]);
            up[1].Push(auxOut[k]);
            float ol, orr;
            while (up[0].Pop(&ol)) {
              if (!up[1].Pop(&orr)) orr = ol;
              fifo[0][wr] = ol; fifo[1][wr] = orr;
              wr = (wr + 1) & (kFifo - 1);
            }
          }
        }
      }

      float wl = 0.0f, wrr = 0.0f;
      if (((wr - rd) & (kFifo - 1)) > 0) {
        wl = fifo[0][rd]; wrr = fifo[1][rd];
        rd = (rd + 1) & (kFifo - 1);
      }
      out.push_back(wl);
      out.push_back(wrr);
    }
    elapsed += frames;
  }
  return out;
}

// Trims to a comfortable level without squashing dynamics.
void Normalize(std::vector<float>& x, float target) {
  const float p = Peak(x);
  if (p <= 1e-6f) return;
  const float g = target / p;
  for (size_t i = 0; i < x.size(); ++i) x[i] *= g;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const std::string dir = argc > 1 ? argv[1] : ".";

  printf("Drum Synth (Peaks): 8 bars at 112 BPM\n");
  std::vector<float> drums = RenderDrums();
  Normalize(drums, 0.89f);
  {
    std::vector<float> stereo;
    stereo.reserve(drums.size() * 2);
    for (size_t i = 0; i < drums.size(); ++i) {
      stereo.push_back(drums[i]);
      stereo.push_back(drums[i]);
    }
    WriteWav(dir + "/demo_1_drumsynth.wav", stereo, 2);
  }

  printf("Modal Resonator (Rings): six models, drums as exciter\n");
  {
    std::vector<float> x = RenderRings(drums);
    Normalize(x, 0.89f);
    WriteWav(dir + "/demo_2_modal_resonator.wav", x, 2);
  }

  printf("Granular Texture (Clouds): all four playback modes\n");
  {
    std::vector<float> x = RenderClouds(drums);
    Normalize(x, 0.89f);
    WriteWav(dir + "/demo_3_granular_texture.wav", x, 2);
  }

  printf("Modal Voice (Elements): struck, bowed, blown\n");
  {
    std::vector<float> x = RenderElements(drums);
    Normalize(x, 0.89f);
    WriteWav(dir + "/demo_4_modal_voice.wav", x, 2);
  }

  printf("done\n");
  return 0;
}
