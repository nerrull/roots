// Denormal-float protection for the MI DSP cores.
//
// None of the MI filter/resonator code (rings::Resonator's bandpass bank,
// Elements' modal resonator, Clouds' FX chain, etc.) guards against denormals
// in its own decay paths, and Wwise does not enable flush-to-zero for plug-in
// DSP threads on its own -- RacineCombDSP.cpp needed its own additive guard
// for exactly this reason. A resonator decaying for several seconds (Rings'
// tail can run to 12s at high damping) spends a lot of that time with sample
// values in denormal range, where floating-point ops are commonly 10-100x
// slower without FTZ/DAZ, which reads as real-time-only glitching/dropouts
// that never shows up in an offline (non-realtime-constrained) render.
//
// Call EnableFlushToZero() once per Execute()/Process() call; it is cheap
// and idempotent, so there is no need to save or restore the previous state.

#ifndef MI_DENORMAL_GUARD_H_
#define MI_DENORMAL_GUARD_H_

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#include <pmmintrin.h>
#define MI_HAS_SSE_FTZ_DAZ 1
#endif

namespace mi {

inline void EnableFlushToZero() {
#if defined(MI_HAS_SSE_FTZ_DAZ)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

}  // namespace mi

#endif  // MI_DENORMAL_GUARD_H_
