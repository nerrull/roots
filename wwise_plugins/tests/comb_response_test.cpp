// Offline validation for the Racine Comb DSP. No Wwise SDK required: the DSP
// carries no Wwise headers, so it links straight into this harness.
//
// Unlike the MI harnesses alongside it, this one asserts rather than renders a
// WAV — the comb's correctness is a claim about where its peaks sit, which is
// measurable. Checks that the peaks land where the frequency parameter says,
// that a swept parameter stays click-free and stable, and that the feedback
// loop cannot run away.
//
// Build (from wwise_plugins/):
//   c++ -std=c++17 -O2 -I. tests/comb_response_test.cpp \
//       RacineComb/SoundEnginePlugin/RacineCombDSP.cpp -o comb_response_test

#include "RacineComb/SoundEnginePlugin/RacineCombDSP.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
    const unsigned int kSampleRate = 48000;
    const unsigned int kBlockSize = 256;

    int g_failures = 0;

    void Check(bool in_bOk, const char* in_szLabel, const char* in_szDetail = "")
    {
        printf("  [%s] %s %s\n", in_bOk ? "PASS" : "FAIL", in_szLabel, in_szDetail);
        if (!in_bOk)
            ++g_failures;
    }

    RacineComb::Settings MakeSettings(float in_fFrequency, float in_fFeedback)
    {
        RacineComb::Settings s;
        s.fFrequency = in_fFrequency;
        s.fFeedback = in_fFeedback;
        s.fDamping = 0.0f;
        s.fWetDryMix = 1.0f;
        s.fGlideMs = 0.0f;
        s.fOutputGain = 1.0f;
        return s;
    }

    /// Runs a steady sine through the comb and returns the settled RMS.
    float SteadyStateRMS(float in_fTestHz, const RacineComb::Settings& in_settings)
    {
        const unsigned int uFrames = RacineComb::DelayFramesPerChannel(kSampleRate);
        std::vector<float> delay(uFrames, 0.0f);

        RacineComb::Processor dsp;
        dsp.Setup(kSampleRate, 1, delay.data());
        dsp.Reset(in_settings.fFrequency);

        // Long enough for the feedback to settle at g = 0.9.
        const unsigned int uTotalBlocks = 400;
        const unsigned int uMeasureFrom = 300;

        std::vector<float> block(kBlockSize);
        double dSumSq = 0.0;
        unsigned int uCounted = 0;
        double dPhase = 0.0;
        const double dStep = 2.0 * M_PI * in_fTestHz / kSampleRate;

        for (unsigned int b = 0; b < uTotalBlocks; ++b)
        {
            for (unsigned int n = 0; n < kBlockSize; ++n)
            {
                block[n] = (float)sin(dPhase);
                dPhase += dStep;
            }

            dsp.ProcessChannel(block.data(), kBlockSize, 0, in_settings, in_settings);

            if (b >= uMeasureFrom)
            {
                for (unsigned int n = 0; n < kBlockSize; ++n)
                {
                    dSumSq += (double)block[n] * block[n];
                    ++uCounted;
                }
            }
        }

        return (float)sqrt(dSumSq / uCounted);
    }

    void TestPeakPlacement()
    {
        printf("Peak placement (f0 = 440 Hz, g = 0.9)\n");

        const RacineComb::Settings s = MakeSettings(440.0f, 0.9f);

        // A comb tuned to 440 should resonate at 440 and its harmonics, and
        // reject the half-integer points between them.
        const float fAt440 = SteadyStateRMS(440.0f, s);
        const float fAt880 = SteadyStateRMS(880.0f, s);
        const float fAt1320 = SteadyStateRMS(1320.0f, s);
        const float fAt660 = SteadyStateRMS(660.0f, s);   // midway: notch
        const float fAt220 = SteadyStateRMS(220.0f, s);   // midway: notch

        printf("    440 Hz: %.3f   880 Hz: %.3f  1320 Hz: %.3f\n", fAt440, fAt880, fAt1320);
        printf("    660 Hz: %.3f   220 Hz: %.3f\n", fAt660, fAt220);

        // With g = 0.9 the peak/notch ratio is (1+g)/(1-g) = 19, ~25 dB.
        Check(fAt440 / fAt660 > 8.0f, "peak at f0 well above adjacent notch");
        Check(fAt880 / fAt660 > 8.0f, "peak at 2*f0 well above adjacent notch");
        Check(fAt1320 / fAt660 > 8.0f, "peak at 3*f0 well above adjacent notch");
        Check(fAt220 < fAt440, "half-integer point rejected");
    }

    void TestPeakTracksFrequency()
    {
        printf("Peak tracks the frequency parameter\n");

        // Deliberately non-integer delay lengths: 48000/700 = 68.57 samples,
        // 48000/1234.5 = 38.88. Integer-only interpolation would miss these.
        const float kFreqs[] = { 137.0f, 700.0f, 1234.5f, 3333.0f };

        for (float f0 : kFreqs)
        {
            const RacineComb::Settings s = MakeSettings(f0, 0.9f);
            const float fOnPeak = SteadyStateRMS(f0, s);
            const float fOffPeak = SteadyStateRMS(f0 * 1.5f, s);

            char szDetail[128];
            snprintf(szDetail, sizeof(szDetail), "f0=%.1f Hz  on=%.3f off=%.3f ratio=%.1f",
                f0, fOnPeak, fOffPeak, fOnPeak / fOffPeak);
            Check(fOnPeak / fOffPeak > 8.0f, "resonance sits on the requested frequency", szDetail);
        }
    }

    /// Locates the response maximum near in_fAround by direct search.
    float FindPeakNear(float in_fAround, const RacineComb::Settings& in_settings)
    {
        float fBestHz = in_fAround;
        float fBest = -1.0f;
        // +/- 6% at 0.25% resolution.
        for (int i = -24; i <= 24; ++i)
        {
            const float f = in_fAround * (1.0f + 0.0025f * i);
            const float fMag = SteadyStateRMS(f, in_settings);
            if (fMag > fBest)
            {
                fBest = fMag;
                fBestHz = f;
            }
        }
        return fBestHz;
    }

    void TestTuningAccuracy()
    {
        printf("Tuning accuracy\n");

        // Regression guard: an extra sample anywhere in the feedback loop (the
        // damping filter, a read/write ordering slip) detunes the comb, and the
        // error grows with frequency as a fraction of the delay length.
        // Undamped, the tuning should be exact to within the search resolution.
        // Damped, it cannot be: the loop gain falls with frequency, which drags
        // the response maximum slightly below the point where the loop phase
        // closes. That residue is a property of a damped comb, so the tolerance
        // opens up rather than the DSP compensating for it.
        struct Case { float fFreq; float fDamping; float fTolPct; };
        const Case kCases[] = {
            { 220.0f,  0.0f, 0.3f },
            { 1234.5f, 0.0f, 0.3f },
            { 3333.0f, 0.0f, 0.3f },
            { 1234.5f, 0.5f, 2.0f },
            { 3333.0f, 0.5f, 2.0f },
        };

        for (const Case& c : kCases)
        {
            RacineComb::Settings s = MakeSettings(c.fFreq, 0.9f);
            s.fDamping = c.fDamping;

            const float fPeak = FindPeakNear(c.fFreq, s);
            const float fErrPct = 100.0f * (fPeak - c.fFreq) / c.fFreq;

            char szDetail[160];
            snprintf(szDetail, sizeof(szDetail), "asked %.1f Hz, peak at %.1f Hz (%+.2f%%), damping %.0f%%, tol %.1f%%",
                c.fFreq, fPeak, fErrPct, c.fDamping * 100.0f, c.fTolPct);
            Check(fabsf(fErrPct) < c.fTolPct, "peak lands on the requested frequency", szDetail);
        }
    }

    void TestSweepIsClickFree()
    {
        printf("Frequency sweep stays continuous\n");

        const unsigned int uFrames = RacineComb::DelayFramesPerChannel(kSampleRate);
        std::vector<float> delay(uFrames, 0.0f);

        RacineComb::Processor dsp;
        dsp.Setup(kSampleRate, 1, delay.data());
        dsp.Reset(100.0f);

        RacineComb::Settings prev = MakeSettings(100.0f, 0.9f);
        prev.fGlideMs = 5.0f;

        std::vector<float> block(kBlockSize);
        double dPhase = 0.0;
        const double dStep = 2.0 * M_PI * 300.0 / kSampleRate;

        float fLastSample = 0.0f;
        float fMaxJump = 0.0f;
        float fMaxAbs = 0.0f;
        bool bFinite = true;

        // Sweep 100 Hz -> 4000 Hz over ~1.3 s, then back down.
        const unsigned int uBlocks = 500;
        for (unsigned int b = 0; b < uBlocks; ++b)
        {
            const float t = (float)b / (uBlocks - 1);
            const float fTri = t < 0.5f ? t * 2.0f : (1.0f - t) * 2.0f;
            RacineComb::Settings cur = prev;
            cur.fFrequency = 100.0f * powf(40.0f, fTri);

            for (unsigned int n = 0; n < kBlockSize; ++n)
            {
                block[n] = 0.5f * (float)sin(dPhase);
                dPhase += dStep;
            }

            dsp.ProcessChannel(block.data(), kBlockSize, 0, prev, cur);

            for (unsigned int n = 0; n < kBlockSize; ++n)
            {
                const float v = block[n];
                if (!std::isfinite(v))
                    bFinite = false;
                const float fJump = fabsf(v - fLastSample);
                if (b > 2 && fJump > fMaxJump)
                    fMaxJump = fJump;
                if (fabsf(v) > fMaxAbs)
                    fMaxAbs = fabsf(v);
                fLastSample = v;
            }

            prev = cur;
        }

        char szDetail[128];
        snprintf(szDetail, sizeof(szDetail), "max sample-to-sample jump %.4f, peak %.3f", fMaxJump, fMaxAbs);

        Check(bFinite, "sweep output stays finite");
        // A 300 Hz sine at this amplitude moves at most ~0.02 per sample; the
        // comb's own transients are larger, but a discontinuity would be
        // order-of-magnitude worse than the signal peak.
        Check(fMaxJump < 0.5f * fMaxAbs, "no discontinuity during sweep", szDetail);
    }

    void TestStabilityAtMaxFeedback()
    {
        printf("Stability at maximum feedback\n");

        const unsigned int uFrames = RacineComb::DelayFramesPerChannel(kSampleRate);
        std::vector<float> delay(uFrames, 0.0f);

        RacineComb::Processor dsp;
        dsp.Setup(kSampleRate, 1, delay.data());
        dsp.Reset(220.0f);

        // Ask for more feedback than is stable; the DSP must clamp it.
        RacineComb::Settings s = MakeSettings(220.0f, 5.0f);
        s.fGlideMs = 1.0f;

        std::vector<float> block(kBlockSize);
        double dPhase = 0.0;
        const double dStep = 2.0 * M_PI * 220.0 / kSampleRate;

        float fPeak = 0.0f;
        bool bFinite = true;

        // 20 s of on-resonance excitation, the worst case for a feedback comb.
        const unsigned int uBlocks = 20 * kSampleRate / kBlockSize;
        for (unsigned int b = 0; b < uBlocks; ++b)
        {
            for (unsigned int n = 0; n < kBlockSize; ++n)
            {
                block[n] = (float)sin(dPhase);
                dPhase += dStep;
            }

            dsp.ProcessChannel(block.data(), kBlockSize, 0, s, s);

            for (unsigned int n = 0; n < kBlockSize; ++n)
            {
                if (!std::isfinite(block[n]))
                    bFinite = false;
                if (fabsf(block[n]) > fPeak)
                    fPeak = fabsf(block[n]);
            }
        }

        char szDetail[128];
        snprintf(szDetail, sizeof(szDetail), "peak %.2f after 20 s on resonance", fPeak);

        Check(bFinite, "output stays finite");
        // Bounded by 1/(1 - kMaxFeedback) = 50 for a unit-amplitude input.
        Check(fPeak < 60.0f, "feedback clamp bounds the resonance", szDetail);
    }

    void TestTailLength()
    {
        printf("Tail estimate\n");

        const unsigned int uFrames = RacineComb::DelayFramesPerChannel(kSampleRate);
        std::vector<float> delay(uFrames, 0.0f);

        RacineComb::Processor dsp;
        dsp.Setup(kSampleRate, 1, delay.data());

        const RacineComb::Settings sQuiet = MakeSettings(220.0f, 0.0f);
        const RacineComb::Settings sRing = MakeSettings(220.0f, 0.95f);

        const unsigned int uQuiet = dsp.TailFrames(sQuiet);
        const unsigned int uRing = dsp.TailFrames(sRing);

        char szDetail[128];
        snprintf(szDetail, sizeof(szDetail), "g=0: %u frames, g=0.95: %u frames (%.2f s)",
            uQuiet, uRing, (float)uRing / kSampleRate);

        Check(uRing > uQuiet, "more feedback means a longer tail", szDetail);
        Check(uRing <= 8 * kSampleRate, "tail is capped");
    }

    void TestDryPassthrough()
    {
        printf("Dry path is transparent\n");

        const unsigned int uFrames = RacineComb::DelayFramesPerChannel(kSampleRate);
        std::vector<float> delay(uFrames, 0.0f);

        RacineComb::Processor dsp;
        dsp.Setup(kSampleRate, 1, delay.data());
        dsp.Reset(220.0f);

        RacineComb::Settings s = MakeSettings(220.0f, 0.9f);
        s.fWetDryMix = 0.0f;

        std::vector<float> block(kBlockSize);
        std::vector<float> expected(kBlockSize);
        double dPhase = 0.0;
        const double dStep = 2.0 * M_PI * 1000.0 / kSampleRate;

        float fMaxErr = 0.0f;
        for (unsigned int b = 0; b < 50; ++b)
        {
            for (unsigned int n = 0; n < kBlockSize; ++n)
            {
                expected[n] = (float)sin(dPhase);
                block[n] = expected[n];
                dPhase += dStep;
            }

            dsp.ProcessChannel(block.data(), kBlockSize, 0, s, s);

            for (unsigned int n = 0; n < kBlockSize; ++n)
            {
                const float fErr = fabsf(block[n] - expected[n]);
                if (fErr > fMaxErr)
                    fMaxErr = fErr;
            }
        }

        char szDetail[64];
        snprintf(szDetail, sizeof(szDetail), "max error %.2e", fMaxErr);
        Check(fMaxErr < 1e-5f, "wet/dry at 0 returns the input unchanged", szDetail);
    }
}

int main()
{
    printf("Racine Comb DSP validation @ %u Hz\n\n", kSampleRate);

    TestPeakPlacement();
    TestPeakTracksFrequency();
    TestTuningAccuracy();
    TestSweepIsClickFree();
    TestStabilityAtMaxFeedback();
    TestTailLength();
    TestDryPassthrough();

    printf("\n%s (%d failure%s)\n",
        g_failures == 0 ? "ALL PASS" : "FAILURES",
        g_failures, g_failures == 1 ? "" : "s");

    return g_failures == 0 ? 0 : 1;
}
