/*******************************************************************************
Racine Comb - tunable feedback comb filter with realtime peak sweep.

This header is deliberately free of Wwise SDK dependencies so the DSP can be
built and validated standalone (see tests/comb_response_test.cpp).
*******************************************************************************/

#ifndef RacineCombDSP_H
#define RacineCombDSP_H

namespace RacineComb
{
    /// Lowest tunable fundamental. Fixes the delay line allocation, since the
    /// frequency is RTPC-driven and may reach the bottom of its range at any time.
    static const float kMinFrequency = 20.0f;

    /// Shortest usable delay, in samples. The 4-point interpolator reads
    /// [i-1 .. i+2], so the read head must stay at least 3 samples behind the
    /// write head to avoid tapping the sample written this frame.
    static const float kMinDelaySamples = 3.0f;

    /// Feedback magnitude is clamped here: at 1.0 the comb is an oscillator.
    static const float kMaxFeedback = 0.98f;

    /// Shortest glide. Even a "no glide" setting slews slightly, otherwise the
    /// per-buffer parameter steps become audible as clicks.
    static const float kMinGlideMs = 0.5f;

    /// Upper bound on channels handled by one instance. Covers 7.1.4 with room
    /// to spare; anything wider is passed through untouched.
    static const unsigned int kMaxChannels = 16;

    /// Damping maps onto a one-pole lowpass in the feedback path, sweeping its
    /// cutoff between these bounds.
    static const float kDampingMaxCutoff = 20000.0f;
    static const float kDampingMinCutoff = 200.0f;

    /// Parameters in DSP-native units (as opposed to the authoring units used
    /// in the property XML).
    struct Settings
    {
        float fFrequency;   ///< Hz. Comb peaks land on integer multiples of this.
        float fFeedback;    ///< -kMaxFeedback .. kMaxFeedback
        float fDamping;     ///< 0 .. 1
        float fWetDryMix;   ///< 0 = dry, 1 = wet
        float fGlideMs;     ///< Slew time of the delay length
        float fOutputGain;  ///< Linear
    };

    /// Per-channel delay line state. The slew state is per channel but evolves
    /// identically across channels, so they stay phase-coherent.
    struct ChannelState
    {
        float* pDelay;      ///< Not owned; see Processor::Setup
        unsigned int uWritePos;
        float fCurDelay;    ///< Slewed delay length, in fractional samples
        float fDampState;
    };

    /// Frames of delay memory needed per channel, rounded up to a power of two
    /// so the read/write indices can be masked rather than wrapped.
    unsigned int DelayFramesPerChannel(unsigned int in_uSampleRate);

    class Processor
    {
    public:
        Processor();

        /// in_pDelayMemory must hold DelayFramesPerChannel() * in_uNumChannels
        /// floats and outlive the processor. Ownership stays with the caller.
        void Setup(unsigned int in_uSampleRate, unsigned int in_uNumChannels, float* in_pDelayMemory);

        /// Clears the delay lines and snaps the glide to the given frequency.
        void Reset(float in_fFrequency);

        /// Processes one channel in place. Parameters ramp from in_prev to
        /// in_cur across the buffer.
        void ProcessChannel(
            float* in_pSamples,
            unsigned int in_uNumFrames,
            unsigned int in_uChannel,
            const Settings& in_prev,
            const Settings& in_cur);

        /// Frames for the feedback to decay below -60 dB, for tail handling.
        unsigned int TailFrames(const Settings& in_settings) const;

        float DelaySamplesForFrequency(float in_fFrequency) const;

        unsigned int NumChannels() const { return m_uNumChannels; }

    private:
        ChannelState m_channels[kMaxChannels];
        unsigned int m_uSampleRate;
        unsigned int m_uNumChannels;
        unsigned int m_uDelayFrames;   ///< Power of two
        unsigned int m_uDelayMask;
        float m_fMaxDelaySamples;
    };
}

#endif // RacineCombDSP_H
