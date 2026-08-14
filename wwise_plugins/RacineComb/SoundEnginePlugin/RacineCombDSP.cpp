/*******************************************************************************
Racine Comb - tunable feedback comb filter with realtime peak sweep.

Structure is the classic feedback comb, H(z) = z^-D / (1 - g z^-D), with a
one-pole lowpass in the loop for damping. Peaks land on integer multiples of
fs/D, so tuning D tunes the whole harmonic series.

D is fractional and interpolated with a 4-point cubic Hermite (Catmull-Rom)
kernel, and slewed per sample. Sweeping D resamples whatever is circulating in
the loop, so moving the peaks pitch-glides the resonating content. That is
intended here.
*******************************************************************************/

#include "RacineCombDSP.h"

#include <cmath>
#include <cstring>

namespace
{
    /// Keeps the feedback state out of denormal range without audible offset.
    const float kDenormalGuard = 1e-18f;

    /// Decay target for the tail estimate: -60 dB.
    const float kTailDecay = 0.001f;

    /// Ceiling on the tail estimate, in seconds. High feedback with a long
    /// delay would otherwise keep a voice alive far longer than is useful.
    const float kMaxTailSeconds = 8.0f;

    inline float Clamp(float v, float lo, float hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    /// 4-point, 3rd-order Hermite. y1 is the sample at the integer index, t the
    /// fractional part in [0, 1).
    inline float Hermite(float y0, float y1, float y2, float y3, float t)
    {
        const float c0 = y1;
        const float c1 = 0.5f * (y2 - y0);
        const float c2 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
        const float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);
        return ((c3 * t + c2) * t + c1) * t + c0;
    }

    /// Damping 0..1 -> one-pole coefficient, cutoff swept logarithmically.
    /// Returns exactly 1 (bypass) at zero damping so the loop stays sample-exact.
    inline float DampingCoef(float in_fDamping, unsigned int in_uSampleRate)
    {
        const float d = Clamp(in_fDamping, 0.0f, 1.0f);
        if (d <= 0.0f)
            return 1.0f;

        const float fCutoff = RacineComb::kDampingMaxCutoff *
            powf(RacineComb::kDampingMinCutoff / RacineComb::kDampingMaxCutoff, d);
        if (fCutoff >= in_uSampleRate * 0.49f)
            return 1.0f;

        const float a = 1.0f - expf(-2.0f * 3.14159265358979f * fCutoff / (float)in_uSampleRate);
        return Clamp(a, 1e-4f, 1.0f);
    }

    /// Phase delay the one-pole contributes at in_fFrequency, in samples.
    ///
    /// The comb resonates where the *total* loop phase comes to 2*pi, so the
    /// filter's share has to come back off the delay line or the peaks sit
    /// flat. Evaluated at the fundamental: the one-pole's phase delay falls
    /// with frequency, so the upper peaks stay very slightly stretched, the
    /// same mild inharmonicity a damped physical resonator has.
    inline float OnePolePhaseDelay(float in_fCoef, float in_fFrequency, unsigned int in_uSampleRate)
    {
        if (in_fCoef >= 1.0f)
            return 0.0f;

        const float w = 2.0f * 3.14159265358979f * in_fFrequency / (float)in_uSampleRate;
        if (w <= 0.0f)
            return 0.0f;

        const float p = 1.0f - in_fCoef;
        const float fPhase = atan2f(p * sinf(w), 1.0f - p * cosf(w));
        return fPhase / w;
    }
}

namespace RacineComb
{
    unsigned int DelayFramesPerChannel(unsigned int in_uSampleRate)
    {
        // Longest delay is the one that puts the fundamental at kMinFrequency,
        // plus the interpolator's forward reach.
        const unsigned int uNeeded = (unsigned int)(in_uSampleRate / kMinFrequency) + 4;
        unsigned int uFrames = 64;
        while (uFrames < uNeeded)
            uFrames <<= 1;
        return uFrames;
    }

    Processor::Processor()
        : m_uSampleRate(48000)
        , m_uNumChannels(0)
        , m_uDelayFrames(0)
        , m_uDelayMask(0)
        , m_fMaxDelaySamples(0.0f)
    {
        memset(m_channels, 0, sizeof(m_channels));
    }

    void Processor::Setup(unsigned int in_uSampleRate, unsigned int in_uNumChannels, float* in_pDelayMemory)
    {
        m_uSampleRate = in_uSampleRate;
        m_uNumChannels = in_uNumChannels < kMaxChannels ? in_uNumChannels : kMaxChannels;
        m_uDelayFrames = DelayFramesPerChannel(in_uSampleRate);
        m_uDelayMask = m_uDelayFrames - 1;

        // Leave the interpolator's forward reach inside the buffer.
        m_fMaxDelaySamples = (float)(m_uDelayFrames - 4);

        for (unsigned int c = 0; c < m_uNumChannels; ++c)
        {
            m_channels[c].pDelay = in_pDelayMemory + (size_t)c * m_uDelayFrames;
            m_channels[c].uWritePos = 0;
            m_channels[c].fCurDelay = 0.0f;
            m_channels[c].fDampState = 0.0f;
        }
    }

    float Processor::DelaySamplesForFrequency(float in_fFrequency) const
    {
        const float f = in_fFrequency < kMinFrequency ? kMinFrequency : in_fFrequency;
        return Clamp((float)m_uSampleRate / f, kMinDelaySamples, m_fMaxDelaySamples);
    }

    void Processor::Reset(float in_fFrequency)
    {
        const float fDelay = DelaySamplesForFrequency(in_fFrequency);
        for (unsigned int c = 0; c < m_uNumChannels; ++c)
        {
            if (m_channels[c].pDelay)
                memset(m_channels[c].pDelay, 0, sizeof(float) * m_uDelayFrames);
            m_channels[c].uWritePos = 0;
            m_channels[c].fCurDelay = fDelay;
            m_channels[c].fDampState = 0.0f;
        }
    }

    unsigned int Processor::TailFrames(const Settings& in_settings) const
    {
        const float fDelay = DelaySamplesForFrequency(in_settings.fFrequency);
        const float g = fabsf(Clamp(in_settings.fFeedback, -kMaxFeedback, kMaxFeedback));

        // Number of round trips for g^n to fall below the decay target.
        float fTrips = 1.0f;
        if (g > 0.001f)
            fTrips = logf(kTailDecay) / logf(g);

        float fFrames = fTrips * fDelay;
        const float fCap = kMaxTailSeconds * (float)m_uSampleRate;
        if (fFrames > fCap)
            fFrames = fCap;
        return (unsigned int)fFrames;
    }

    void Processor::ProcessChannel(
        float* in_pSamples,
        unsigned int in_uNumFrames,
        unsigned int in_uChannel,
        const Settings& in_prev,
        const Settings& in_cur)
    {
        if (in_uChannel >= m_uNumChannels || in_uNumFrames == 0)
            return;

        ChannelState& state = m_channels[in_uChannel];
        float* pDelay = state.pDelay;
        if (!pDelay)
            return;

        const float fInvFrames = 1.0f / (float)in_uNumFrames;

        // Delay length: ramp the *target* across the buffer, then slew toward it
        // per sample. The ramp removes the per-buffer staircase, the slew sets
        // how fast the peaks are allowed to travel.
        const float fDelayStart = DelaySamplesForFrequency(in_prev.fFrequency);
        const float fDelayEnd = DelaySamplesForFrequency(in_cur.fFrequency);
        const float fDelayStep = (fDelayEnd - fDelayStart) * fInvFrames;

        const float fGlideMs = in_cur.fGlideMs < kMinGlideMs ? kMinGlideMs : in_cur.fGlideMs;
        const float fGlideSamples = fGlideMs * 0.001f * (float)m_uSampleRate;
        const float fSlew = Clamp(1.0f - expf(-1.0f / fGlideSamples), 0.0f, 1.0f);

        const float fFbStart = Clamp(in_prev.fFeedback, -kMaxFeedback, kMaxFeedback);
        const float fFbStep = (Clamp(in_cur.fFeedback, -kMaxFeedback, kMaxFeedback) - fFbStart) * fInvFrames;

        const float fDampStart = DampingCoef(in_prev.fDamping, m_uSampleRate);
        const float fDampEnd = DampingCoef(in_cur.fDamping, m_uSampleRate);
        const float fDampStep = (fDampEnd - fDampStart) * fInvFrames;

        const float fPhaseCompStart = OnePolePhaseDelay(fDampStart, in_prev.fFrequency, m_uSampleRate);
        const float fPhaseCompEnd = OnePolePhaseDelay(fDampEnd, in_cur.fFrequency, m_uSampleRate);
        const float fPhaseCompStep = (fPhaseCompEnd - fPhaseCompStart) * fInvFrames;

        const float fWetStart = Clamp(in_prev.fWetDryMix, 0.0f, 1.0f);
        const float fWetStep = (Clamp(in_cur.fWetDryMix, 0.0f, 1.0f) - fWetStart) * fInvFrames;

        const float fGainStart = in_prev.fOutputGain;
        const float fGainStep = (in_cur.fOutputGain - fGainStart) * fInvFrames;

        float fDelayTarget = fDelayStart;
        float fFeedback = fFbStart;
        float fDampCoef = fDampStart;
        float fPhaseComp = fPhaseCompStart;
        float fWet = fWetStart;
        float fGain = fGainStart;

        float fCurDelay = state.fCurDelay;
        float fDampState = state.fDampState;
        unsigned int uWritePos = state.uWritePos;

        for (unsigned int n = 0; n < in_uNumFrames; ++n)
        {
            const float fIn = in_pSamples[n];

            fCurDelay += (fDelayTarget - fCurDelay) * fSlew;

            // The damping filter is part of the loop, so its group delay counts
            // toward the tuning; take it back out of the delay line.
            const float fDelay = Clamp(fCurDelay - fPhaseComp, kMinDelaySamples, m_fMaxDelaySamples);

            // Read before writing this frame: with D >= 3 the interpolator's
            // [i-1 .. i+2] window stays strictly behind the write head, and the
            // damped sample can then be written back within the same frame.
            const float fReadPos = (float)uWritePos - fDelay;
            const int iRead = (int)floorf(fReadPos);
            const float t = fReadPos - (float)iRead;

            const unsigned int i0 = (unsigned int)(iRead - 1 + (int)m_uDelayFrames) & m_uDelayMask;
            const unsigned int i1 = (unsigned int)(iRead + (int)m_uDelayFrames) & m_uDelayMask;
            const unsigned int i2 = (unsigned int)(iRead + 1 + (int)m_uDelayFrames) & m_uDelayMask;
            const unsigned int i3 = (unsigned int)(iRead + 2 + (int)m_uDelayFrames) & m_uDelayMask;

            const float fDelayed = Hermite(pDelay[i0], pDelay[i1], pDelay[i2], pDelay[i3], t);

            // One-pole lowpass closes the feedback loop. Its DC gain is 1, so
            // the stability bound stays |fFeedback| < 1.
            fDampState += fDampCoef * (fDelayed - fDampState) + kDenormalGuard;

            pDelay[uWritePos] = fIn + fFeedback * fDampState;

            in_pSamples[n] = (fIn * (1.0f - fWet) + fDelayed * fWet) * fGain;

            uWritePos = (uWritePos + 1) & m_uDelayMask;
            fDelayTarget += fDelayStep;
            fFeedback += fFbStep;
            fDampCoef += fDampStep;
            fPhaseComp += fPhaseCompStep;
            fWet += fWetStep;
            fGain += fGainStep;
        }

        state.fCurDelay = fCurDelay;
        state.fDampState = fDampState;
        state.uWritePos = uWritePos;
    }
}
