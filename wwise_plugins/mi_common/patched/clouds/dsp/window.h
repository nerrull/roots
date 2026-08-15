// -----------------------------------------------------------------------------
// PATCHED COPY of clouds/dsp/window.h -- one line changed. See the marked line
// in Start() below.
//
// This directory is placed ahead of the eurorack checkout on the include path
// so this file shadows the original, which is left untouched.
//
// Why:
//   Window::done_ is assigned in exactly two places upstream. Init() sets it
//   true, and OverlapAdd() sets it from the envelope phase -- but OverlapAdd()
//   early-returns when done_ is already true, so it can never clear it.
//   Start() does not touch done_ at all.
//
//   The result is a one-way latch. WSOLASamplePlayer::Init() calls
//   windows_[0].Init() and windows_[1].Init(), marking both done, and from that
//   point OverlapAdd() returns immediately forever. PLAYBACK_MODE_STRETCH
//   therefore emits about a second of audio (whatever is in flight) and then
//   goes permanently silent.
//
//   Adding done_ = false to Start() -- which is what "start a new window"
//   plainly means -- restores it. Verified by A/B: sustained RMS goes from
//   0.00000 to ~0.06-0.12 with no other change, and the correlator starts
//   latching correlator_loaded_ as it should.
//
//   Not reproducible from the wrapper: the failure is identical when the core
//   is driven directly at 32 kHz with no resampler, block adapter or FIFO.
//
// Upstream is MIT licensed, Copyright 2014 Emilie Gillet; the original header
// follows unchanged.
// -----------------------------------------------------------------------------

// Copyright 2014 Emilie Gillet.
//
// Author: Emilie Gillet (emilie.o.gillet@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// 
// See http://creativecommons.org/licenses/MIT/ for more information.
//
// -----------------------------------------------------------------------------
//
// Variant of the grain class used for WSOLA.

#ifndef CLOUDS_DSP_WINDOW_H_
#define CLOUDS_DSP_WINDOW_H_

#include "stmlib/stmlib.h"

#include "stmlib/dsp/dsp.h"

#include "clouds/dsp/audio_buffer.h"

#include "clouds/resources.h"

namespace clouds {

enum WindowFlags {
  WINDOW_FLAGS_HALF_DONE = 1,
  WINDOW_FLAGS_REGENERATED = 2,
  WINDOW_FLAGS_DONE = 4
};

class Window {
 public:
  Window() { }
  ~Window() { }
  
  void Init() {
    done_ = true;
    regenerated_ = false;
    half_ = false;
  }
  
  void Start(
      int32_t buffer_size,
      int32_t start,
      int32_t width,
      int32_t phase_increment) {
    first_sample_ = (start + buffer_size) % buffer_size;
    phase_increment_ = phase_increment;
    phase_ = 0;
    // THE PATCH: upstream never clears done_ here.
    done_ = false;
    regenerated_ = false;
    envelope_phase_increment_ = 2.0f / static_cast<float>(width);
  }
  
  template<Resolution resolution>
  inline void OverlapAdd(
      const AudioBuffer<resolution>* buffer,
      float* samples,
      int32_t channels) {
    if (done_) {
      return;
    }
    int32_t phase_integral = phase_ >> 16;
    int32_t phase_fractional = phase_ & 0xffff;
    int32_t sample_index = first_sample_ + phase_integral;
    
    float envelope_phase = phase_integral * envelope_phase_increment_;
    done_ = envelope_phase >= 2.0f;
    half_ = envelope_phase >= 1.0f;
    float gain = envelope_phase >= 1.0f
        ? 2.0f - envelope_phase
        : envelope_phase;
    
    float l = buffer[0].ReadHermite(sample_index, phase_fractional) * gain;
    if (channels == 1) {
      *samples++ += l;
      *samples++ += l;
    } else if (channels == 2) {
      float r = buffer[1].ReadHermite(sample_index, phase_fractional) * gain;
      *samples++ += l;
      *samples++ += r;
    }
    phase_ += phase_increment_;
  }
  
  inline bool done() { return done_; }
  inline bool needs_regeneration() { return half_ && !regenerated_; }
  inline void MarkAsRegenerated() { regenerated_ = true; }
  
 private:
  Window* next_;
  int32_t first_sample_;
  int32_t phase_;
  int32_t phase_increment_;
  float envelope_phase_increment_;
  
  bool done_;
  bool half_;
  bool regenerated_;
  
  DISALLOW_COPY_AND_ASSIGN(Window);
};

}  // namespace clouds

#endif  // CLOUDS_DSP_WINDOW_H_
