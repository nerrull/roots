#include "audio_capture.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#import <AudioToolbox/AudioToolbox.h>
#import <CoreAudio/CoreAudio.h>

namespace {

constexpr UInt32 kInputBus = 1;
constexpr UInt32 kMaxFramesPerSlice = 4096;

std::string LowerCase(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Reads a CFString device property, or "" if absent.
std::string StringProp(AudioObjectID dev, AudioObjectPropertySelector sel) {
  AudioObjectPropertyAddress addr{sel, kAudioObjectPropertyScopeGlobal,
                                  kAudioObjectPropertyElementMain};
  CFStringRef cf = nullptr;
  UInt32 size = sizeof(cf);
  if (AudioObjectGetPropertyData(dev, &addr, 0, nullptr, &size, &cf) != noErr ||
      !cf) {
    return {};
  }
  char buf[512] = {0};
  CFStringGetCString(cf, buf, sizeof(buf), kCFStringEncodingUTF8);
  CFRelease(cf);
  return std::string(buf);
}

std::string DeviceName(AudioObjectID dev) {
  return StringProp(dev, kAudioObjectPropertyName);
}

int InputChannelCount(AudioObjectID dev) {
  AudioObjectPropertyAddress addr{kAudioDevicePropertyStreamConfiguration,
                                  kAudioDevicePropertyScopeInput,
                                  kAudioObjectPropertyElementMain};
  UInt32 size = 0;
  if (AudioObjectGetPropertyDataSize(dev, &addr, 0, nullptr, &size) != noErr ||
      size == 0) {
    return 0;
  }
  std::vector<uint8_t> storage(size);
  auto* list = reinterpret_cast<AudioBufferList*>(storage.data());
  if (AudioObjectGetPropertyData(dev, &addr, 0, nullptr, &size, list) != noErr) {
    return 0;
  }
  int channels = 0;
  for (UInt32 i = 0; i < list->mNumberBuffers; ++i) {
    channels += list->mBuffers[i].mNumberChannels;
  }
  return channels;
}

AudioObjectID DefaultInputDevice() {
  AudioObjectPropertyAddress addr{kAudioHardwarePropertyDefaultInputDevice,
                                  kAudioObjectPropertyScopeGlobal,
                                  kAudioObjectPropertyElementMain};
  AudioObjectID dev = kAudioObjectUnknown;
  UInt32 size = sizeof(dev);
  AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, nullptr, &size,
                             &dev);
  return dev;
}

bool Contains(const std::string& haystack, const std::string& needle) {
  return LowerCase(haystack).find(LowerCase(needle)) != std::string::npos;
}

// A one-line rendering of the device list, for error messages.
std::string DescribeDevices(const std::vector<AudioDeviceInfo>& devs) {
  std::string s;
  for (const AudioDeviceInfo& d : devs) {
    s += "\n  - \"" + d.name + "\"  " + std::to_string(d.input_channels) +
         "ch @ " + std::to_string(int(d.sample_rate)) + " Hz" +
         (d.is_default_input ? "  [system default input]" : "") +
         "\n      uid=" + d.uid + "\n      model=" + d.model_uid;
  }
  if (s.empty()) s = " (none)";
  return s;
}

}  // namespace

std::vector<AudioDeviceInfo> ListAudioInputDevices() {
  std::vector<AudioDeviceInfo> out;

  AudioObjectPropertyAddress addr{kAudioHardwarePropertyDevices,
                                  kAudioObjectPropertyScopeGlobal,
                                  kAudioObjectPropertyElementMain};
  UInt32 size = 0;
  if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0,
                                     nullptr, &size) != noErr) {
    return out;
  }
  std::vector<AudioObjectID> devs(size / sizeof(AudioObjectID));
  if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, nullptr,
                                 &size, devs.data()) != noErr) {
    return out;
  }

  const AudioObjectID def = DefaultInputDevice();
  for (AudioObjectID d : devs) {
    const int in = InputChannelCount(d);
    if (in <= 0) continue;

    AudioDeviceInfo info;
    info.id = d;
    info.name = DeviceName(d);
    info.uid = StringProp(d, kAudioDevicePropertyDeviceUID);
    info.model_uid = StringProp(d, kAudioDevicePropertyModelUID);
    info.manufacturer = StringProp(d, kAudioObjectPropertyManufacturer);
    info.input_channels = in;
    info.is_default_input = (d == def);

    UInt32 transport = 0, tsize = sizeof(transport);
    AudioObjectPropertyAddress ta{kAudioDevicePropertyTransportType,
                                  kAudioObjectPropertyScopeGlobal,
                                  kAudioObjectPropertyElementMain};
    if (AudioObjectGetPropertyData(d, &ta, 0, nullptr, &tsize, &transport) ==
        noErr) {
      info.is_usb = (transport == kAudioDeviceTransportTypeUSB);
    }

    Float64 rate = 0;
    UInt32 rsize = sizeof(rate);
    AudioObjectPropertyAddress ra{kAudioDevicePropertyNominalSampleRate,
                                  kAudioObjectPropertyScopeGlobal,
                                  kAudioObjectPropertyElementMain};
    if (AudioObjectGetPropertyData(d, &ra, 0, nullptr, &rsize, &rate) == noErr) {
      info.sample_rate = rate;
    }
    out.push_back(std::move(info));
  }
  return out;
}

namespace {

// Applies the selector. On no match, `err` explains what was sought and lists
// what was available. Never returns an arbitrary device.
// `ambiguous` reports a multi-match, which callers must not retry: waiting can
// only ever make an ambiguous match worse, never resolve it.
bool SelectDevice(const AudioSelector& sel,
                  const std::vector<AudioDeviceInfo>& devs,
                  AudioDeviceInfo* chosen, bool* fallback, std::string& err,
                  bool* ambiguous) {
  *fallback = false;
  if (ambiguous) *ambiguous = false;

  struct Criterion {
    const char* label;
    const std::string& value;                    // what to look for
    const std::string AudioDeviceInfo::* field;  // where to look for it
    bool exact;                                  // exact match vs. substring
  };
  const Criterion criteria[] = {
      {"device UID", sel.uid, &AudioDeviceInfo::uid, true},
      {"model UID", sel.model_uid, &AudioDeviceInfo::model_uid, false},
      {"name", sel.name, &AudioDeviceInfo::name, false},
  };

  for (const Criterion& c : criteria) {
    if (c.value.empty()) continue;

    std::vector<const AudioDeviceInfo*> hits;
    for (const AudioDeviceInfo& d : devs) {
      const std::string& field = d.*(c.field);
      const bool match =
          c.exact ? (field == c.value) : Contains(field, c.value);
      if (match && d.input_channels >= sel.min_input_channels) hits.push_back(&d);
    }

    if (hits.size() == 1) {
      *chosen = *hits[0];
      return true;
    }
    if (hits.size() > 1) {
      // Ambiguous: refuse rather than pick by enumeration order. The UID of any
      // one of them disambiguates, so name them in the error.
      if (ambiguous) *ambiguous = true;
      err = std::string("audio ") + c.label + " \"" + c.value +
            "\" matches " + std::to_string(hits.size()) +
            " devices; pass --audio-uid to choose one:";
      for (const AudioDeviceInfo* d : hits) {
        err += "\n  - \"" + d->name + "\" uid=" + d->uid;
      }
      return false;
    }

    err = std::string("no input device matched ") + c.label + " \"" + c.value +
          "\"" +
          (sel.min_input_channels > 0
               ? " with >=" + std::to_string(sel.min_input_channels) +
                     " input channels"
               : "") +
          ". Available inputs:" + DescribeDevices(devs);
    // A criterion that was asked for but found nothing is an error; do not
    // quietly try the next, looser one. Only an explicit opt-in reaches the
    // default-input fallback below.
    if (!sel.allow_default_fallback) return false;
    break;
  }

  if (sel.allow_default_fallback) {
    const AudioObjectID def = DefaultInputDevice();
    for (const AudioDeviceInfo& d : devs) {
      if (d.id == def) {
        *chosen = d;
        *fallback = true;
        err.clear();
        return true;
      }
    }
    err += "\n...and no usable system default input to fall back to";
    return false;
  }
  if (err.empty()) {
    err = "no audio device selector given and default-input fallback is disabled";
  }
  return false;
}

}  // namespace

struct AudioCapture::Impl {
  AudioUnit unit = nullptr;
  // Preallocated non-interleaved buffer list handed to AudioUnitRender().
  std::vector<uint8_t> abl_storage;
  std::vector<float> scratch;  // kMaxFramesPerSlice * channels
  int channels = 0;
};

// Gives the C render callback access to AudioCapture's privates.
struct AudioCaptureBridge {
  static OSStatus Render(void* refcon, AudioUnitRenderActionFlags* flags,
                         const AudioTimeStamp* ts, UInt32 bus, UInt32 frames,
                         AudioBufferList* /*io*/) {
    auto* self = static_cast<AudioCapture*>(refcon);
    AudioCapture::Impl* impl = self->impl_;
    if (!impl || !impl->unit || frames == 0) return noErr;
    if (frames > kMaxFramesPerSlice) {
      self->callback_errors_.fetch_add(1);
      return noErr;
    }

    const int ch_count = impl->channels;
    auto* abl = reinterpret_cast<AudioBufferList*>(impl->abl_storage.data());
    abl->mNumberBuffers = static_cast<UInt32>(ch_count);
    for (int c = 0; c < ch_count; ++c) {
      abl->mBuffers[c].mNumberChannels = 1;
      abl->mBuffers[c].mDataByteSize = frames * sizeof(float);
      abl->mBuffers[c].mData = impl->scratch.data() + c * kMaxFramesPerSlice;
    }

    const OSStatus st =
        AudioUnitRender(impl->unit, flags, ts, bus, frames, abl);
    if (st != noErr) {
      self->callback_errors_.fetch_add(1);
      return noErr;  // keep the unit running; the UI surfaces the counter
    }

    // Single producer: publish the new write cursor only after the samples are
    // in place, so a concurrent snapshot() can't read past valid data.
    const uint64_t base = self->write_pos_.load(std::memory_order_relaxed);
    for (int c = 0; c < ch_count; ++c) {
      const float* src =
          static_cast<const float*>(abl->mBuffers[c].mData);
      float* ring = self->ring_[c].data();
      for (UInt32 i = 0; i < frames; ++i) {
        ring[(base + i) & (AudioCapture::kRingSize - 1)] = src[i];
      }
    }

    // Beamform + dynamics over the block we just wrote. This reads the raw
    // rings behind `base` for the steering delay lines, which is why it has to
    // happen after the writes above and before the cursor is published.
    self->processBlock(base, int(frames));

    self->write_pos_.store(base + frames, std::memory_order_release);
    self->frames_captured_.fetch_add(frames, std::memory_order_relaxed);
    return noErr;
  }
};

AudioCapture::~AudioCapture() { stop(); }

// Realtime: no allocation, no locks, no syscalls.
void AudioCapture::processBlock(uint64_t base, int frames) {
  float* out = beam_ring_.data();
  if (!out || channels_ <= 0) return;

  const bool on = controls.enabled.load(std::memory_order_relaxed);
  const float gain =
      dsp::DbToLin(controls.input_gain_db.load(std::memory_order_relaxed));

  // Steering delays for this block. Recomputed per block (a couple of trig
  // calls per ~32 ms) so the beam follows the tracker without zipper noise
  // within a block.
  const bool beam_on = controls.beamform.load(std::memory_order_relaxed) &&
                       channels_ >= 2;
  float delays[dsp::MicArray::kMaxMics] = {0};
  int beam_mics = 1;
  if (beam_on) {
    const dsp::MicArray arr = dsp::UniformLinearArray(
        std::min(channels_, dsp::MicArray::kMaxMics),
        controls.aperture_m.load(std::memory_order_relaxed));
    const float bulk = dsp::SteeringDelays(
        arr, controls.steer_deg.load(std::memory_order_relaxed),
        float(sample_rate_), delays);
    beam_mics = arr.count;
    beam_latency_ms_.store(
        float(bulk / std::max(sample_rate_, 1.0) * 1000.0),
        std::memory_order_relaxed);
  } else {
    beam_latency_ms_.store(0.f, std::memory_order_relaxed);
  }

  // Highpass coefficients only change when the UI moves the slider.
  const float hp_hz = controls.highpass_hz.load(std::memory_order_relaxed);
  if (hp_hz != hp_cached_hz_) {
    hp_coeffs_ = dsp::HighpassCoeffs(float(sample_rate_), hp_hz);
    hp_cached_hz_ = hp_hz;
  }

  dsp::CompressorParams cp;
  cp.threshold_db = controls.comp_threshold_db.load(std::memory_order_relaxed);
  cp.ratio = controls.comp_ratio.load(std::memory_order_relaxed);
  cp.attack_ms = controls.comp_attack_ms.load(std::memory_order_relaxed);
  cp.release_ms = controls.comp_release_ms.load(std::memory_order_relaxed);
  cp.knee_db = controls.comp_knee_db.load(std::memory_order_relaxed);
  cp.makeup_db = controls.comp_makeup_db.load(std::memory_order_relaxed);
  const bool limit = controls.limiter.load(std::memory_order_relaxed);

  for (int i = 0; i < frames; ++i) {
    const uint64_t n = base + uint64_t(i);

    float x;
    if (beam_on) {
      // Delay-and-sum: read each mic at its steered fractional offset. All
      // offsets are >= 2 samples into the past, so every tap is already written.
      float sum = 0.f;
      for (int c = 0; c < beam_mics; ++c) {
        const float pos = float(i) - delays[c];
        const int ip = int(std::floor(pos));
        const float frac = pos - float(ip);
        const float* ring = ring_[c].data();
        const uint64_t b = base + uint64_t(int64_t(ip));
        const float taps[4] = {
            ring[(b - 1) & (kRingSize - 1)], ring[b & (kRingSize - 1)],
            ring[(b + 1) & (kRingSize - 1)], ring[(b + 2) & (kRingSize - 1)]};
        sum += dsp::Lagrange4(taps, frac);
      }
      x = sum / float(beam_mics);
    } else {
      x = ring_[0][n & (kRingSize - 1)];
    }

    if (on) {
      x = hp_filter_.process(x * gain, hp_coeffs_);
      x = compressor_.process(x, cp);
      if (limit) x = limiter_.process(x, -1.f);
    }
    out[n & (kRingSize - 1)] = x;
  }

  gain_reduction_db_.store(compressor_.gainReductionDb(),
                           std::memory_order_relaxed);
}

bool AudioCapture::start(const AudioSelector& sel, std::string& err,
                         double wait_seconds) {
  if (running_) return true;

  // Poll for the device rather than failing on the first look: after a USB
  // reset the audio interface reappears a moment later.
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(static_cast<long>(wait_seconds * 1000));
  bool ambiguous = false;
  for (;;) {
    const std::vector<AudioDeviceInfo> devs = ListAudioInputDevices();
    if (SelectDevice(sel, devs, &device_, &used_fallback_, err, &ambiguous)) {
      break;
    }
    // An ambiguous match will never become unambiguous by waiting.
    if (ambiguous || std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  const AudioObjectID dev = device_.id;

  AudioComponentDescription desc{};
  desc.componentType = kAudioUnitType_Output;
  desc.componentSubType = kAudioUnitSubType_HALOutput;
  desc.componentManufacturer = kAudioUnitManufacturer_Apple;
  AudioComponent comp = AudioComponentFindNext(nullptr, &desc);
  if (!comp) {
    err = "AUHAL component not found";
    return false;
  }

  impl_ = new Impl();
  if (AudioComponentInstanceNew(comp, &impl_->unit) != noErr || !impl_->unit) {
    err = "AudioComponentInstanceNew failed";
    stop();
    return false;
  }

  // AUHAL defaults to output-only: enable the input bus, disable bus 0.
  UInt32 on = 1, off = 0;
  if (AudioUnitSetProperty(impl_->unit, kAudioOutputUnitProperty_EnableIO,
                           kAudioUnitScope_Input, kInputBus, &on,
                           sizeof(on)) != noErr) {
    err = "failed to enable AUHAL input";
    stop();
    return false;
  }
  AudioUnitSetProperty(impl_->unit, kAudioOutputUnitProperty_EnableIO,
                       kAudioUnitScope_Output, 0, &off, sizeof(off));

  if (AudioUnitSetProperty(impl_->unit, kAudioOutputUnitProperty_CurrentDevice,
                           kAudioUnitScope_Global, 0, &dev,
                           sizeof(dev)) != noErr) {
    err = "failed to select device '" + device_.name + "' (uid=" + device_.uid +
          ")";
    stop();
    return false;
  }

  // Read the device back rather than trusting the set: if the unit is somehow
  // bound elsewhere (the default input, say), we want to fail loudly here
  // instead of silently scoping the wrong microphone.
  AudioObjectID bound = kAudioObjectUnknown;
  UInt32 bsize = sizeof(bound);
  if (AudioUnitGetProperty(impl_->unit, kAudioOutputUnitProperty_CurrentDevice,
                           kAudioUnitScope_Global, 0, &bound, &bsize) != noErr ||
      bound != dev) {
    err = "audio unit bound to device id " + std::to_string(bound) +
          " but we selected \"" + device_.name + "\" (id " +
          std::to_string(dev) + ", uid=" + device_.uid + ")";
    stop();
    return false;
  }

  // Take the hardware's own rate and channel count so no sample-rate or
  // channel conversion is inserted (the Kinect reports 4ch @ 16 kHz).
  AudioStreamBasicDescription hw{};
  UInt32 size = sizeof(hw);
  if (AudioUnitGetProperty(impl_->unit, kAudioUnitProperty_StreamFormat,
                           kAudioUnitScope_Input, kInputBus, &hw,
                           &size) != noErr) {
    err = "failed to read hardware stream format";
    stop();
    return false;
  }
  channels_ = std::min<int>(static_cast<int>(hw.mChannelsPerFrame), kMaxChannels);
  sample_rate_ = hw.mSampleRate;
  // Report what the unit actually gave us, not what enumeration predicted.
  device_.input_channels = channels_;
  device_.sample_rate = sample_rate_;
  if (channels_ <= 0) {
    err = "device reports 0 input channels";
    stop();
    return false;
  }

  // Ask for deinterleaved float32 so each channel lands in its own buffer.
  AudioStreamBasicDescription want{};
  want.mSampleRate = hw.mSampleRate;
  want.mFormatID = kAudioFormatLinearPCM;
  want.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked |
                      kAudioFormatFlagIsNonInterleaved;
  want.mChannelsPerFrame = static_cast<UInt32>(channels_);
  want.mBitsPerChannel = 32;
  want.mFramesPerPacket = 1;
  want.mBytesPerFrame = sizeof(float);
  want.mBytesPerPacket = sizeof(float);
  if (AudioUnitSetProperty(impl_->unit, kAudioUnitProperty_StreamFormat,
                           kAudioUnitScope_Output, kInputBus, &want,
                           sizeof(want)) != noErr) {
    err = "device rejected float32 non-interleaved format";
    stop();
    return false;
  }

  UInt32 slice = kMaxFramesPerSlice;
  AudioUnitSetProperty(impl_->unit, kAudioUnitProperty_MaximumFramesPerSlice,
                       kAudioUnitScope_Global, 0, &slice, sizeof(slice));

  // Allocate everything the realtime callback touches up front.
  impl_->channels = channels_;
  impl_->scratch.assign(static_cast<size_t>(kMaxFramesPerSlice) * channels_, 0.f);
  impl_->abl_storage.assign(
      sizeof(AudioBufferList) + sizeof(AudioBuffer) * channels_, 0);
  for (int c = 0; c < channels_; ++c) ring_[c].assign(kRingSize, 0.f);
  beam_ring_.assign(kRingSize, 0.f);
  hp_filter_.reset();
  hp_cached_hz_ = -1.f;  // force a coefficient recompute on the first block
  compressor_.prepare(float(sample_rate_));
  limiter_.prepare(float(sample_rate_));
  gain_reduction_db_.store(0.f);
  beam_latency_ms_.store(0.f);
  write_pos_.store(0);
  frames_captured_.store(0);
  callback_errors_.store(0);

  AURenderCallbackStruct cb{};
  cb.inputProc = &AudioCaptureBridge::Render;
  cb.inputProcRefCon = this;
  if (AudioUnitSetProperty(impl_->unit,
                           kAudioOutputUnitProperty_SetInputCallback,
                           kAudioUnitScope_Global, 0, &cb, sizeof(cb)) != noErr) {
    err = "failed to install input callback";
    stop();
    return false;
  }

  if (AudioUnitInitialize(impl_->unit) != noErr) {
    err = "AudioUnitInitialize failed";
    stop();
    return false;
  }
  if (AudioOutputUnitStart(impl_->unit) != noErr) {
    err = "AudioOutputUnitStart failed";
    stop();
    return false;
  }

  running_ = true;
  return true;
}

void AudioCapture::stop() {
  if (impl_) {
    if (impl_->unit) {
      AudioOutputUnitStop(impl_->unit);
      AudioUnitUninitialize(impl_->unit);
      AudioComponentInstanceDispose(impl_->unit);
      impl_->unit = nullptr;
    }
    delete impl_;
    impl_ = nullptr;
  }
  running_ = false;
}

namespace {

// Shared by the raw-channel and beam accessors: both rings use the same index
// space and the same published cursor.
void SnapshotRing(const float* ring, uint64_t w, int count,
                  std::vector<float>& out) {
  out.resize(count);
  if (!ring || count <= 0) {
    std::fill(out.begin(), out.end(), 0.f);
    return;
  }
  // Before the ring has filled once, pad the missing history with zeros.
  const uint64_t start = (w >= static_cast<uint64_t>(count)) ? w - count : 0;
  const int lead =
      (w >= static_cast<uint64_t>(count)) ? 0 : count - static_cast<int>(w);
  for (int i = 0; i < lead; ++i) out[i] = 0.f;
  for (int i = lead; i < count; ++i) {
    out[i] = ring[(start + (i - lead)) & (AudioCapture::kRingSize - 1)];
  }
}

void LevelsFromRing(const float* ring, uint64_t w, int count, float* peak,
                    float* rms) {
  float pk = 0.f;
  double sum = 0.0;
  if (ring && count > 0) {
    const int n = static_cast<int>(std::min<uint64_t>(w, count));
    for (int i = 0; i < n; ++i) {
      const float v = ring[(w - 1 - i) & (AudioCapture::kRingSize - 1)];
      pk = std::max(pk, std::fabs(v));
      sum += static_cast<double>(v) * v;
    }
    if (n > 0) sum /= n;
  }
  if (peak) *peak = pk;
  if (rms) *rms = static_cast<float>(std::sqrt(sum));
}

}  // namespace

void AudioCapture::snapshot(int ch, int count, std::vector<float>& out) const {
  count = std::min(count, kRingSize);
  if (ch < 0 || ch >= channels_) {
    out.assign(std::max(count, 0), 0.f);
    return;
  }
  SnapshotRing(ring_[ch].data(), write_pos_.load(std::memory_order_acquire),
               count, out);
}

void AudioCapture::levels(int ch, int count, float* peak, float* rms) const {
  count = std::min(count, kRingSize);
  const bool ok = (ch >= 0 && ch < channels_);
  LevelsFromRing(ok ? ring_[ch].data() : nullptr,
                 write_pos_.load(std::memory_order_acquire), count, peak, rms);
}

void AudioCapture::beamSnapshot(int count, std::vector<float>& out) const {
  count = std::min(count, kRingSize);
  SnapshotRing(beam_ring_.empty() ? nullptr : beam_ring_.data(),
               write_pos_.load(std::memory_order_acquire), count, out);
}

void AudioCapture::beamLevels(int count, float* peak, float* rms) const {
  count = std::min(count, kRingSize);
  LevelsFromRing(beam_ring_.empty() ? nullptr : beam_ring_.data(),
                 write_pos_.load(std::memory_order_acquire), count, peak, rms);
}

bool AudioCapture::looksLikeSilence() const {
  if (!running_ || framesCaptured() == 0) return false;
  for (int c = 0; c < channels_; ++c) {
    float pk = 0.f;
    levels(c, 8000, &pk, nullptr);
    if (pk > 0.f) return false;
  }
  return true;
}
