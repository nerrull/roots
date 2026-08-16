// The onset detector, and the shared-memory stream it publishes through.
//
// Two halves, both offline and both without Wwise:
//
//   detection   synthetic material whose transients are known exactly, so
//               "did it fire, and when" is checkable rather than a matter of
//               listening. The cases are chosen to be the ways an adaptive
//               threshold fails: a quiet bed (fires on nothing), a loud steady
//               tone (fires on nothing), the same hits at two very different
//               levels (fires identically), and a dense roll (fires once per
//               hit, not once per analysis hop).
//
//   transport   the writer and the reader against each other, in one process.
//               The layout is shared memory read by a program that was not
//               compiled at the same time, so the sequencing rules -- never
//               replay, never hand back a lapped slot -- are the contract.
//
// Optionally takes a WAV to run the detector over instead, which is how you
// tune sensitivity against real material:
//   onset_test [file.wav]
//
// Exit 0 on pass.
#include "../mi_common/onset_detector.h"
#include "../mi_common/onset_shm.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++g_fail;
}

constexpr uint32_t kSR = 48000;

// White-ish noise, deterministic so a failure is reproducible.
struct Noise {
    uint32_t s = 22222;
    float operator()() {
        s = s * 1664525u + 1013904223u;
        return float(int32_t(s >> 8) - 0x800000) / float(0x800000);
    }
};

// A bed at `bedAmp` with an exponentially decaying hit at each time in `hits`.
std::vector<float> material(float seconds, float bedAmp,
                            const std::vector<float>& hits, float hitAmp,
                            float decayMs = 120.f) {
    std::vector<float> x(size_t(seconds * kSR), 0.f);
    Noise n;
    for (float& v : x) v = bedAmp * n();
    const float tau = decayMs * 0.001f * kSR;
    for (float t : hits) {
        const size_t start = size_t(t * kSR);
        Noise hn;
        for (size_t i = start; i < x.size(); ++i) {
            const float env = std::exp(-float(i - start) / tau);
            if (env < 1e-4f) break;
            x[i] += hitAmp * env * hn();
        }
    }
    return x;
}

// Runs the detector over a signal in 256-frame blocks (a typical Wwise block)
// and returns the time of each onset, in seconds.
std::vector<float> detect(const std::vector<float>& x, const mi::OnsetParams& p) {
    mi::OnsetDetector d;
    d.Init(kSR);
    d.SetParams(p);
    std::vector<float> times;
    mi::OnsetHit hits[8];
    const uint32_t block = 256;
    for (size_t i = 0; i < x.size(); i += block) {
        const uint32_t n = uint32_t(std::min<size_t>(block, x.size() - i));
        const uint32_t got = d.Process(x.data() + i, n, hits, 8);
        for (uint32_t h = 0; h < got; ++h)
            times.push_back(float(i + hits[h].frameOffset) / float(kSR));
    }
    return times;
}

// How many of `want` were detected within `tol` seconds, and how many
// detections matched nothing (false positives).
void score(const std::vector<float>& got, const std::vector<float>& want,
           float tol, int& hit, int& spurious) {
    hit = 0;
    spurious = 0;
    std::vector<bool> used(want.size(), false);
    for (float g : got) {
        bool matched = false;
        for (size_t i = 0; i < want.size(); ++i) {
            if (!used[i] && std::fabs(g - want[i]) <= tol) {
                used[i] = true;
                matched = true;
                break;
            }
        }
        if (matched) ++hit; else ++spurious;
    }
}

void test_finds_hits() {
    std::printf("it finds the hits and nothing else\n");
    const std::vector<float> when{0.6f, 1.2f, 1.9f, 2.5f, 3.3f};
    auto x = material(4.f, 0.02f, when, 0.5f);

    mi::OnsetParams p;
    p.floorDb = -60.f;
    auto got = detect(x, p);
    int hit = 0, spurious = 0;
    score(got, when, 0.03f, hit, spurious);
    char msg[128];
    std::snprintf(msg, sizeof msg, "5 hits over a bed -> %d found, %d spurious",
                  hit, spurious);
    check(hit == 5 && spurious == 0, msg);

    // The point of the whole design: the same material 30 dB quieter must give
    // the same answer. A fixed threshold cannot do this.
    std::vector<float> quiet = x;
    for (float& v : quiet) v *= 0.0316f;   // -30 dB
    mi::OnsetParams pq = p;
    pq.floorDb = -90.f;                    // the gate has to allow for the level
    auto gotq = detect(quiet, pq);
    score(gotq, when, 0.03f, hit, spurious);
    std::snprintf(msg, sizeof msg, "...and 30 dB quieter -> %d found, %d spurious",
                  hit, spurious);
    check(hit == 5 && spurious == 0, msg);
}

void test_silence_and_steady() {
    std::printf("\nit does not invent events\n");
    mi::OnsetParams p;

    // Near-silence. Its rises are, statistically, just as exceptional as a
    // hit's -- the absolute gate is the only thing standing between this and a
    // continuous stream of onsets from room tone.
    auto quiet = material(4.f, 0.0005f, {}, 0.f);
    auto got = detect(quiet, p);
    char msg[128];
    std::snprintf(msg, sizeof msg, "4 s of near-silence -> %d events", (int)got.size());
    check(got.empty(), msg);

    // A loud steady tone: plenty of level, no transients. The rise is what
    // matters, not the level.
    std::vector<float> tone(4 * kSR);
    for (size_t i = 0; i < tone.size(); ++i)
        tone[i] = 0.5f * std::sin(2.f * 3.14159265f * 220.f * float(i) / float(kSR));
    got = detect(tone, p);
    std::snprintf(msg, sizeof msg, "4 s of a loud steady tone -> %d events",
                  (int)got.size());
    check(got.size() <= 1, msg);   // at most the one where it starts

    // A bed that rises slowly over four seconds: a fade-in is not an onset,
    // however far the level travels in the end.
    auto swell = material(4.f, 0.02f, {}, 0.f);
    for (size_t i = 0; i < swell.size(); ++i)
        swell[i] *= 0.05f + 10.f * float(i) / float(swell.size());
    got = detect(swell, p);
    std::snprintf(msg, sizeof msg, "a 46 dB swell over 4 s -> %d events", (int)got.size());
    check(got.size() <= 1, msg);
}

void test_refractory_and_sensitivity() {
    std::printf("\none hit is one event\n");
    // Hits 150 ms apart, i.e. comfortably outside the refractory period, but
    // each spanning many analysis hops.
    std::vector<float> when;
    for (int i = 0; i < 12; ++i) when.push_back(0.4f + 0.15f * i);
    auto x = material(3.f, 0.02f, when, 0.5f, 40.f);

    mi::OnsetParams p;
    p.minIntervalMs = 80.f;
    auto got = detect(x, p);
    char msg[128];
    std::snprintf(msg, sizeof msg, "12 hits at 150 ms apart -> %d events",
                  (int)got.size());
    check(got.size() == 12, msg);

    // A long refractory period deliberately thins a roll out -- this is the
    // control for "I want a drop per beat, not per grain".
    mi::OnsetParams slow = p;
    slow.minIntervalMs = 400.f;
    auto thin = detect(x, slow);
    std::snprintf(msg, sizeof msg, "...and %d with a 400 ms minimum interval",
                  (int)thin.size());
    check(thin.size() >= 3 && thin.size() <= 6, msg);

    // Sensitivity has to be monotonic, or it is not a knob anyone can use.
    auto busy = material(4.f, 0.05f, {0.5f, 1.0f, 1.4f, 2.2f, 3.0f}, 0.35f);
    mi::OnsetParams lo = p, hi = p;
    lo.sensitivity = 1.f;
    hi.sensitivity = 6.f;
    const size_t nlo = detect(busy, lo).size(), nhi = detect(busy, hi).size();
    std::snprintf(msg, sizeof msg, "sensitivity 1 -> %d events, 6 -> %d",
                  (int)nlo, (int)nhi);
    check(nlo >= nhi, msg);
}

void test_strength() {
    std::printf("\nstrength tracks how hard the hit was\n");
    mi::OnsetParams p;
    p.floorDb = -60.f;

    auto measure = [&](float amp) {
        auto x = material(2.f, 0.01f, {1.0f}, amp);
        mi::OnsetDetector d;
        d.Init(kSR);
        d.SetParams(p);
        mi::OnsetHit hits[8];
        float strength = -1.f;
        const uint32_t block = 256;
        for (size_t i = 0; i < x.size(); i += block) {
            const uint32_t n = uint32_t(std::min<size_t>(block, x.size() - i));
            const uint32_t got = d.Process(x.data() + i, n, hits, 8);
            for (uint32_t h = 0; h < got; ++h)
                if (float(i) / kSR > 0.9f && strength < 0.f) strength = hits[h].strength;
        }
        return strength;
    };

    const float soft = measure(0.05f), hard = measure(0.9f);
    char msg[128];
    std::snprintf(msg, sizeof msg, "soft %.2f < hard %.2f", soft, hard);
    check(soft >= 0.f && hard > soft, msg);
    check(soft >= 0.f && soft <= 1.f && hard <= 1.f, "and both stay in 0..1");
}

// --- the shared-memory stream ----------------------------------------------

void test_transport() {
    std::printf("\nthe stream carries them across\n");
    const uint32_t tap = 13;   // a tap ID nothing else is likely to hold

    mi::onset::TapWriter w;
    check(w.Open(tap, "Test Tap", kSR), "the writer opens a stream");

    mi::onset::DirectoryReader dir;
    bool listed = false;
    for (const auto& t : dir.ListActive())
        if (t.tapId == tap && t.label == "Test Tap") listed = true;
    check(listed, "and registers in the directory");

    mi::onset::TapReader r;
    check(r.Open(tap), "a reader attaches to it");

    // A reader that attaches mid-show must not replay history.
    for (int i = 0; i < 5; ++i) {
        mi::onset::Event e;
        e.strength = 0.5f;
        w.Push(e);
    }
    mi::onset::TapReader late;
    late.Open(tap);
    std::vector<mi::onset::Event> got;
    check(late.Poll(got) == 0, "a reader attaching late starts from now");

    // ...but one that was already attached sees everything since its last poll.
    got.clear();
    const uint32_t n = r.Poll(got);
    char msg[128];
    std::snprintf(msg, sizeof msg, "an attached reader sees all 5 (%u)", n);
    check(n == 5 && got.size() == 5, msg);
    // Contiguous and increasing, but not necessarily starting at 1: the segment
    // outlives the process that made it, and a fresh writer picking up a stale
    // one deliberately continues the count rather than resetting it -- a cursor
    // that went backwards would look to every attached reader like a stream it
    // has already seen.
    bool ordered = got.size() == 5;
    for (size_t i = 1; i < got.size(); ++i)
        if (got[i].seq != got[i - 1].seq + 1) ordered = false;
    check(ordered, "in order, with contiguous sequence numbers");

    got.clear();
    check(r.Poll(got) == 0, "and polling again returns nothing");

    // Values survive the crossing intact -- this is a struct being reinterpreted
    // through a mapping, so a layout mistake shows up as garbage here.
    mi::onset::Event sent;
    sent.strength = 0.75f;
    sent.levelDb = -12.5f;
    sent.excessDb = 3.25f;
    sent.pan = -0.5f;
    sent.channels = 2;
    sent.hostTimeNs = 123456789ull;
    w.Push(sent);
    got.clear();
    r.Poll(got);
    const bool intact = got.size() == 1 && got[0].strength == sent.strength &&
                        got[0].levelDb == sent.levelDb &&
                        got[0].excessDb == sent.excessDb && got[0].pan == sent.pan &&
                        got[0].channels == sent.channels &&
                        got[0].hostTimeNs == sent.hostTimeNs;
    check(intact, "every field arrives unchanged");

    // Meters are readable without any event having fired: that is how a UI
    // distinguishes "silent" from "threshold too high".
    w.PublishMeter(-24.f, 5.5f, 42);
    check(std::fabs(r.LevelDb() + 24.f) < 1e-4f &&
          std::fabs(r.ThresholdDb() - 5.5f) < 1e-4f && r.HeartbeatNs() == 42,
          "the meter is live between events");

    // Lapping the reader: the events that fell out of the ring are counted as
    // missed rather than handed back as stale slots.
    for (uint32_t i = 0; i < mi::onset::kEventCapacity + 20; ++i) {
        mi::onset::Event e;
        e.strength = float(i);
        w.Push(e);
    }
    got.clear();
    const uint32_t after = r.Poll(got);
    std::snprintf(msg, sizeof msg, "a lapped reader keeps %u and counts %u missed",
                  after, r.Missed());
    check(after == mi::onset::kEventCapacity && r.Missed() == 20, msg);

    w.Close();
    bool stillListed = false;
    for (const auto& t : dir.ListActive())
        if (t.tapId == tap) stillListed = true;
    check(!stillListed, "closing the writer clears its directory slot");
}

// --- optional: run over a real file -----------------------------------------

bool ReadWavMono(const char* path, std::vector<float>& out, uint32_t& sampleRate) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    char id[4];
    uint32_t size = 0;
    std::fread(id, 1, 4, f);
    std::fread(&size, 4, 1, f);
    std::fread(id, 1, 4, f);
    uint16_t channels = 1, bits = 16, format = 1;
    bool haveFmt = false;
    while (std::fread(id, 1, 4, f) == 4) {
        uint32_t chunk = 0;
        if (std::fread(&chunk, 4, 1, f) != 1) break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            std::vector<uint8_t> fmt(chunk);
            std::fread(fmt.data(), 1, chunk, f);
            std::memcpy(&format, fmt.data() + 0, 2);
            std::memcpy(&channels, fmt.data() + 2, 2);
            std::memcpy(&sampleRate, fmt.data() + 4, 4);
            std::memcpy(&bits, fmt.data() + 14, 2);
            haveFmt = true;
        } else if (std::memcmp(id, "data", 4) == 0 && haveFmt) {
            const uint32_t frames = chunk / (channels * (bits / 8));
            out.assign(frames, 0.f);
            std::vector<uint8_t> raw(chunk);
            std::fread(raw.data(), 1, chunk, f);
            for (uint32_t i = 0; i < frames; ++i) {
                float acc = 0.f;
                for (uint16_t c = 0; c < channels; ++c) {
                    const uint8_t* s = raw.data() + (size_t(i) * channels + c) * (bits / 8);
                    if (bits == 16) {
                        int16_t v;
                        std::memcpy(&v, s, 2);
                        acc += float(v) / 32768.f;
                    } else if (bits == 32 && format == 3) {
                        float v;
                        std::memcpy(&v, s, 4);
                        acc += v;
                    }
                }
                out[i] = acc / float(channels);
            }
            std::fclose(f);
            return !out.empty();
        } else {
            std::fseek(f, chunk + (chunk & 1), SEEK_CUR);
        }
    }
    std::fclose(f);
    return false;
}

void report_file(const char* path) {
    std::vector<float> x;
    uint32_t sr = kSR;
    if (!ReadWavMono(path, x, sr)) {
        std::printf("\ncannot read %s (16-bit or float32 PCM WAV only)\n", path);
        ++g_fail;
        return;
    }
    std::printf("\n%s: %.1f s at %u Hz\n", path, float(x.size()) / float(sr), sr);
    for (float sens : {1.5f, 2.5f, 4.f}) {
        mi::OnsetDetector d;
        d.Init(sr);
        mi::OnsetParams p;
        p.sensitivity = sens;
        d.SetParams(p);
        mi::OnsetHit hits[8];
        int n = 0;
        float sumStrength = 0.f;
        const uint32_t block = 256;
        for (size_t i = 0; i < x.size(); i += block) {
            const uint32_t got = d.Process(x.data() + i,
                                           uint32_t(std::min<size_t>(block, x.size() - i)),
                                           hits, 8);
            for (uint32_t h = 0; h < got; ++h) { ++n; sumStrength += hits[h].strength; }
        }
        std::printf("  sensitivity %.1f -> %d onsets (%.2f/s), mean strength %.2f\n",
                    sens, n, float(n) / (float(x.size()) / float(sr)),
                    n ? sumStrength / float(n) : 0.f);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("onset_detector_test: transients, and the stream they travel on\n");
    test_finds_hits();
    test_silence_and_steady();
    test_refractory_and_sensitivity();
    test_strength();
    test_transport();
    if (argc > 1) report_file(argv[1]);
    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
