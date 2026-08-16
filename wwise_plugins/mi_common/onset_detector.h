// Transient detection with a threshold that follows the material.
//
// Used by the OnsetTap plug-in to turn whatever passes through a bus into
// discrete "something just happened" events for an external program to react to
// (see onset_shm.h for how they get there).
//
// A fixed threshold cannot do this job. The same show carries a rain bed at
// -45 dBFS and a struck resonator peaking near 0, and a level that catches the
// rain fires continuously on the resonator while one tuned to the resonator
// never hears the rain at all. Both are the *same* musical event -- a hit -- and
// what distinguishes a hit is not how loud it is but how much louder it is than
// the moment before it.
//
// So the detection function is the positive rise of the signal's level in dB,
// measured per hop, and the threshold is the recent statistics of that function:
//   thr = mean(rise) + sensitivity * stddev(rise)
// A steady bed has a small mean rise and a small spread, so a modest transient
// clears the bar; a busy percussive passage raises both and only the genuinely
// exceptional hits get through. Because the whole thing is in dB it is invariant
// to absolute level -- turning the bus down does not change what it detects.
//
// Three guards keep it honest, and each exists because of a specific way an
// adaptive threshold embarrasses itself:
//   floor      an absolute level gate, so the noise in "silence" (whose rises
//              are, statistically, just as exceptional) does not fire at all
//   min rise   a floor under the threshold itself, so the quietest possible
//              adaptation window cannot make a 0.2 dB wobble into an onset
//   refractory a minimum gap, so one hit's attack is one event and not the six
//              consecutive hops it actually spans
//
// Header-only, no Wwise and no allocation after Init(): it runs on the audio
// thread and it has to be usable by an offline test that links nothing.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mi {

struct OnsetParams
{
    // How exceptional a rise has to be, in standard deviations above the recent
    // mean. ~1.5 is twitchy, ~2.5 musical, ~4 only the big hits.
    float sensitivity = 2.5f;
    // Absolute gate: material quieter than this never produces an onset,
    // whatever its statistics say.
    float floorDb = -60.f;
    // The smallest rise that counts, whatever the adaptation says. Keeps a very
    // steady source from adapting its way down onto its own noise.
    float minRiseDb = 2.0f;
    // Refractory period: one hit is one event.
    float minIntervalMs = 80.f;
    // How far back the mean/stddev look. Long enough to average over a bar or
    // so, short enough to follow a scene change.
    float windowMs = 1000.f;
    // Envelope follower. Fast attack so a transient is not smeared into the hop
    // before it; slower release so the rise is measured against where the sound
    // actually was, not against the follower already falling away.
    float attackMs = 1.f;
    float releaseMs = 60.f;
};

struct OnsetHit
{
    uint32_t frameOffset = 0; // where in the processed block it landed
    float strength = 0.f;     // 0..1, loudness of the hit within the floor..0 dB range
    float levelDb = 0.f;      // the envelope at the hit
    float excessDb = 0.f;     // how far over the threshold the rise was
};

class OnsetDetector
{
public:
    // Hop size in frames. ~1.3 ms at 48 kHz: short enough that a snare's attack
    // lands in one or two hops, long enough that the per-hop RMS is a level and
    // not a sample.
    static constexpr uint32_t kHop = 64;

    void Init(uint32_t sampleRate)
    {
        m_sampleRate = sampleRate ? sampleRate : 48000;
        Reset();
    }

    void Reset()
    {
        m_acc = 0.f;
        m_accCount = 0;
        m_env = 0.f;
        m_prevDb = -200.f;
        m_history.assign(HistoryLen(), 0.f);
        m_histPos = 0;
        m_histFilled = 0;
        m_sum = 0.0;
        m_sumSq = 0.0;
        m_sinceHit = 1e9f;
        m_threshold = 0.f;
        m_levelDb = -200.f;
    }

    void SetParams(const OnsetParams& p)
    {
        m_p = p;
        const uint32_t want = HistoryLen();
        if (want != m_history.size())
        {
            // The window changed: start its statistics over rather than mixing
            // two window lengths' worth of history into one mean.
            m_history.assign(want, 0.f);
            m_histPos = 0;
            m_histFilled = 0;
            m_sum = 0.0;
            m_sumSq = 0.0;
        }
    }

    // Feeds one block of mono samples. Returns how many hits were written to
    // out[0..maxOut); extra hits in the same block are dropped, which at the
    // refractory periods this runs at means never in practice.
    uint32_t Process(const float* mono, uint32_t numFrames, OnsetHit* out, uint32_t maxOut)
    {
        uint32_t found = 0;
        const float hopSeconds = float(kHop) / float(m_sampleRate);

        for (uint32_t i = 0; i < numFrames; ++i)
        {
            const float s = mono[i];
            m_acc += s * s;
            if (++m_accCount < kHop)
                continue;

            const float rms = std::sqrt(m_acc / float(kHop));
            m_acc = 0.f;
            m_accCount = 0;
            m_sinceHit += hopSeconds;

            // Follower, per hop: coefficients are exp(-hop / tau), so the
            // attack/release times mean the same thing at any sample rate.
            const float atk = Coeff(m_p.attackMs, hopSeconds);
            const float rel = Coeff(m_p.releaseMs, hopSeconds);
            const float c = (rms > m_env) ? atk : rel;
            m_env = c * m_env + (1.f - c) * rms;

            const float db = 20.f * std::log10(m_env + 1e-9f);
            m_levelDb = db;
            const float rise = (m_prevDb <= -199.f) ? 0.f : (db - m_prevDb);
            m_prevDb = db;
            const float odf = rise > 0.f ? rise : 0.f;

            // The threshold is computed from the window *before* this hop, so a
            // hit never contributes to the bar it has to clear -- otherwise a
            // loud enough transient partly hides itself.
            const float mean = Mean();
            const float sd = StdDev(mean);
            float thr = mean + m_p.sensitivity * sd;
            if (thr < m_p.minRiseDb)
                thr = m_p.minRiseDb;
            m_threshold = thr;

            PushHistory(odf);

            const bool loudEnough = db > m_p.floorDb;
            const bool clearedBar = odf > thr;
            const bool armed = m_sinceHit >= m_p.minIntervalMs * 0.001f;
            if (loudEnough && clearedBar && armed)
            {
                m_sinceHit = 0.f;
                if (found < maxOut)
                {
                    OnsetHit& h = out[found++];
                    h.frameOffset = i;
                    h.levelDb = db;
                    h.excessDb = odf - thr;
                    // Loudness within the usable range, which is what a visual
                    // wants: the floor is silence and 0 dBFS is as loud as it
                    // gets, so a hit maps to 0..1 across exactly the span the
                    // gate already says is audible.
                    const float span = -m_p.floorDb > 1.f ? -m_p.floorDb : 1.f;
                    float st = (db - m_p.floorDb) / span;
                    st = st < 0.f ? 0.f : (st > 1.f ? 1.f : st);
                    h.strength = st;
                }
            }
        }
        return found;
    }

    // Live values for a meter: the follower's level in dB and the rise (in dB)
    // a hit would currently have to clear.
    float LevelDb() const { return m_levelDb; }
    float ThresholdDb() const { return m_threshold; }

private:
    uint32_t HistoryLen() const
    {
        const float hops = m_p.windowMs * 0.001f * float(m_sampleRate) / float(kHop);
        uint32_t n = uint32_t(hops);
        if (n < 8) n = 8;
        if (n > 8192) n = 8192;
        return n;
    }

    static float Coeff(float ms, float dtSeconds)
    {
        if (ms <= 0.f)
            return 0.f;
        return std::exp(-dtSeconds / (ms * 0.001f));
    }

    // Running sums rather than a pass over the window per hop: this is on the
    // audio thread, and a 1 s window at 48 kHz is 750 entries to re-add 750
    // times a second otherwise.
    void PushHistory(float v)
    {
        const float old = m_history[m_histPos];
        if (m_histFilled == m_history.size())
        {
            m_sum -= old;
            m_sumSq -= double(old) * double(old);
        }
        else
        {
            ++m_histFilled;
        }
        m_history[m_histPos] = v;
        m_sum += v;
        m_sumSq += double(v) * double(v);
        m_histPos = (m_histPos + 1) % m_history.size();
    }

    float Mean() const
    {
        return m_histFilled ? float(m_sum / double(m_histFilled)) : 0.f;
    }

    float StdDev(float mean) const
    {
        if (m_histFilled < 2)
            return 0.f;
        const double var = m_sumSq / double(m_histFilled) - double(mean) * double(mean);
        return var > 0.0 ? float(std::sqrt(var)) : 0.f;
    }

    OnsetParams m_p;
    uint32_t m_sampleRate = 48000;

    float m_acc = 0.f;
    uint32_t m_accCount = 0;
    float m_env = 0.f;
    float m_prevDb = -200.f;
    float m_levelDb = -200.f;
    float m_threshold = 0.f;
    float m_sinceHit = 1e9f;

    std::vector<float> m_history;
    size_t m_histPos = 0;
    size_t m_histFilled = 0;
    double m_sum = 0.0;
    double m_sumSq = 0.0;
};

} // namespace mi
