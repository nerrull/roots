// The whole path from a published onset to a drop on the water.
//
// The two halves are tested where they live -- the detector and the stream in
// wwise_plugins/tests/onset_detector_test.cpp, the spawner in drop_test.cpp --
// and this is the join between them, which is where the interesting mistakes
// are: a reader that replays history the moment it connects, strengths that
// arrive but are ignored, a backlog that lands as one enormous splash. It plays
// the writer's part directly (the same header the Wwise plug-in writes through)
// so it needs neither Wwise nor a running sound engine.
//
// Exit 0 on pass.
#include "audio_pulse.h"
#include "pond_state.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++g_fail;
}

// A tap ID for tests only, well away from the low IDs a show would use.
constexpr uint32_t kTestTap = 15;

mi::onset::Event hit(float strength, float pan) {
    mi::onset::Event e;
    e.strength = strength;
    e.pan = pan;
    e.levelDb = -20.f;
    e.channels = 2;
    return e;
}

void test_events_arrive() {
    std::printf("onsets reach the app\n");
    mi::onset::TapWriter w;
    check(w.Open(kTestTap, "Test Bus", 48000), "a tap publishes");

    mirror::AudioPulses pulses;
    bool listed = false;
    for (const mirror::AudioTap& t : pulses.taps())
        if (t.tapId == kTestTap) listed = true;
    check(listed, "the app lists it");
    check(pulses.connect(kTestTap), "and connects to it");
    check(pulses.label() == std::string("Test Bus"), "with the name it was given");

    // Anything published before connecting is history, and history must not
    // arrive as a burst of drops the moment a show connects mid-run.
    check(pulses.poll().empty(), "nothing waiting on a fresh connection");

    w.Push(hit(0.25f, -0.5f));
    w.Push(hit(0.9f, 0.75f));
    auto got = pulses.poll();
    check(got.size() == 2, "two hits, two events");
    const bool intact = got.size() == 2 && std::fabs(got[0].strength - 0.25f) < 1e-6f &&
                        std::fabs(got[1].pan - 0.75f) < 1e-6f;
    check(intact, "strength and pan survive the crossing");
    check(pulses.poll().empty(), "and they are not delivered twice");

    // The meter is what tells a quiet bus from a threshold set too high, so it
    // has to be readable with nothing firing at all.
    w.PublishMeter(-33.5f, 4.f, 1000);
    check(std::fabs(pulses.levelDb() + 33.5f) < 1e-3f, "the level meter reads through");
    check(std::fabs(pulses.thresholdDb() - 4.f) < 1e-3f, "and so does the threshold");

    w.Close();
}

void test_hits_become_drops() {
    std::printf("\nand land as drops\n");
    mirror::PondParams p;
    p.drops_on = true;
    p.spawn.rain_on = false;      // audio only: every drop here came from a hit
    p.spawn.amp_jitter = 0.f;
    p.spawn.width_jitter = 0.f;
    p.spawn.hit_amp = 1.f;
    p.spawn.hit_width = 1.f;
    p.spawn.hit_pan = 1.f;

    mi::onset::TapWriter w;
    w.Open(kTestTap, "Test Bus", 48000);
    mirror::AudioPulses pulses;
    pulses.connect(kTestTap);

    mirror::Pond pond(11);
    // A frame with nothing published: no hits, no drops. Establishes that what
    // follows came from the audio and not from the schedule.
    for (const mirror::AudioOnset& e : pulses.poll())
        pond.triggerDrop(e.strength, e.pan);
    pond.render(32, 48, 0.0, p);
    check(pond.lastSources().empty(), "a silent bus makes no drops");

    w.Push(hit(1.f, 0.9f));
    w.Push(hit(0.1f, -0.9f));
    for (const mirror::AudioOnset& e : pulses.poll())
        pond.triggerDrop(e.strength, e.pan);
    pond.render(32, 48, 0.2, p);
    const auto& src = pond.lastSources();
    check(src.size() == 2, "two onsets, two drops");
    if (src.size() == 2) {
        // Ordered as published, so [0] is the loud one panned right.
        char msg[128];
        std::snprintf(msg, sizeof msg, "the loud hit is the bigger drop (%.2f vs %.2f)",
                      src[0][4], src[1][4]);
        check(src[0][4] > src[1][4] * 1.5f, msg);
        std::snprintf(msg, sizeof msg, "and they land where they were heard (%.2f vs %.2f)",
                      src[0][0], src[1][0]);
        check(src[0][0] > 0.f && src[1][0] < 0.f, msg);
    }

    // With the drops switched off the tap keeps running and the hits are simply
    // not drawn -- turning the rain off must not leave a queue building up that
    // empties all at once when it comes back.
    mirror::PondParams off = p;
    off.drops_on = false;
    for (int i = 0; i < 20; ++i) {
        w.Push(hit(0.8f, 0.f));
        for (const mirror::AudioOnset& e : pulses.poll())
            pond.triggerDrop(e.strength, e.pan);
        pond.render(32, 48, 0.3 + i * 0.05, off);
    }
    pond.render(32, 48, 1.5, p);
    char msg[128];
    std::snprintf(msg, sizeof msg, "no backlog when the rain is off (%d drops)",
                  (int)pond.lastSources().size());
    check(pond.lastSources().empty(), msg);

    w.Close();
}

}  // namespace

int main() {
    std::printf("audio_drops_test: an onset in Wwise, a drop on the water\n");
    test_events_arrive();
    test_hits_become_drops();
    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
