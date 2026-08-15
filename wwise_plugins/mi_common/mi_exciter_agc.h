// Normalizes an audio-rate exciter signal toward the level Rings and Elements
// were actually tuned for before it reaches their DSP cores.
//
// Both cores were designed for Eurorack line level -- a few volts peak to
// peak, consistently "hot" by construction of the hardware. Real game audio
// (an ambience bed, dialogue, a music stem) commonly sits tens of dB quieter
// and swings in level constantly, which two absolute-level assumptions in the
// MI code don't tolerate:
//   - Rings' onset detector (rings/dsp/onset_detector.h) requires the energy
//     derivative to clear a fixed absolute floor (0.01) in addition to its
//     z-score test, so quiet input produces sparse or missed strums instead
//     of steady triggering.
//   - Elements' external exciter input (blow_in/strike_in in part.cc) is fed
//     straight into the resonator with no gain stage, so quiet input yields
//     a proportionally quiet -- effectively inaudible at high Dry/Wet -- wet
//     signal.
//
// This is a fast-attack/slow-release peak follower that brings the exciter
// signal up to a consistent operating level regardless of how quiet or loud
// the source bus is, exactly the job the Eurorack modules' front panel gain
// trim would otherwise do by hand.

#ifndef MI_EXCITER_AGC_H_
#define MI_EXCITER_AGC_H_

#include <cmath>

namespace mi
{

class ExciterAGC
{
public:
    ExciterAGC() : m_fEnvelope(0.0f) {}

    inline float Process(float in)
    {
        const float fRect = fabsf(in);
        const float fCoeff = (fRect > m_fEnvelope) ? kAttack : kRelease;
        m_fEnvelope += fCoeff * (fRect - m_fEnvelope);

        const float fEnvFloor = (m_fEnvelope < kFloor) ? kFloor : m_fEnvelope;
        float fGain = kTarget / fEnvFloor;
        if (fGain > kMaxGain) fGain = kMaxGain;
        if (fGain < 1.0f) fGain = 1.0f;

        float fOut = in * fGain;
        // The envelope follower reacts to average level, not to individual
        // transients, so a sharp attack (a rain droplet, a click) can still
        // slip through at full gain before the envelope catches up. Clamp
        // the result rather than let an outlier sample feed an extreme value
        // into the resonator, where its own gain can ring it out further.
        if (fOut > kOutputClamp) fOut = kOutputClamp;
        if (fOut < -kOutputClamp) fOut = -kOutputClamp;

        return fOut;
    }

    void Reset() { m_fEnvelope = 0.0f; }

private:
    // ~1 ms attack, ~40 ms release at 48 kHz. Fast enough to catch onsets,
    // slow enough that the gain isn't audibly pumping.
    static constexpr float kAttack = 0.02f;
    static constexpr float kRelease = 0.0005f;
    // Envelope floor to bound gain on near-silence; ceiling so hiss/noise
    // floor isn't amplified into an unwanted signal.
    static constexpr float kFloor = 0.001f;
    static constexpr float kMaxGain = 12.0f;
    // Target peak level, matched to the range these cores were tuned for.
    static constexpr float kTarget = 0.2f;
    // Absolute ceiling on the boosted sample, independent of the gain
    // formula above -- see the comment in Process().
    static constexpr float kOutputClamp = 1.0f;

    float m_fEnvelope;
};

}  // namespace mi

#endif  // MI_EXCITER_AGC_H_
