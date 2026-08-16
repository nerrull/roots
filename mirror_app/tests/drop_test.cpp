// A raindrop must behave like an impact, not like a field being turned up.
//
// The old drops were standing ring fields whose amplitude was modulated by a
// slow cosine: rings everywhere at once, breathing in place, from a position
// that never changed. This checks what replaced that, on both sides of the
// split -- the renderer's ring train rides its own wavefront and leaves flat
// water behind it (mirror_render.h), and the spawner lands each drop as its own
// event, whether the schedule or an outside trigger asked for it
// (drop_spawner.h). Both are easy to break silently (a sign, a stale phase, a
// shared clock) in ways that still render *something* plausible.
//
// Needs a Metal device, no fixtures. Exit 0 on pass.
#include "drop_spawner.h"
#include "mirror_render.h"

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++g_fail;
}

constexpr float kRingFreq = 3.f;
constexpr float kK = kRingFreq * (float)M_PI;

// The wave field sampled along a ray from the source, as N points from r = 0 to
// r = rmax. A radial slice is enough: the field is radially symmetric about the
// source by construction, and it makes "where is the energy" a 1-D question.
std::vector<float> radial(float phase, float packet_w, int n = 512,
                          float rmax = 2.f) {
    std::vector<float> xy;
    xy.reserve(n * 2);
    for (int i = 0; i < n; ++i) {
        xy.push_back(rmax * float(i) / float(n - 1));
        xy.push_back(0.f);
    }
    auto coords = mx::array(xy.data(), {n, 2}, mx::float32);
    std::vector<mirror::RippleSource> src{{0.f, 0.f, phase, 1.f, packet_w}};
    auto f = mirror::multi_ripple_features(coords, src, kRingFreq, 1.8f, 0.f, 0.f);
    // Column 4 is sin_field, the ripple term the MLP actually sees.
    auto col = mx::contiguous(mx::astype(mx::slice(f, {0, 4}, {n, 5}), mx::float32));
    mx::eval(col);
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) out[i] = col.data<float>()[i];
    return out;
}

// Radius of the largest |field| in the slice, and that magnitude.
std::pair<float, float> peak(const std::vector<float>& v, float rmax = 2.f) {
    int best = 0;
    for (size_t i = 0; i < v.size(); ++i)
        if (std::fabs(v[i]) > std::fabs(v[best])) best = int(i);
    return {rmax * float(best) / float(v.size() - 1), std::fabs(v[best])};
}

// The largest |field| inside `r`, i.e. how much is still going on behind the
// front. Water the drop has finished with should be near still.
float max_inside(const std::vector<float>& v, float r, float rmax = 2.f) {
    const int upto = int(r / rmax * float(v.size() - 1));
    float m = 0.f;
    for (int i = 0; i <= upto && i < int(v.size()); ++i) m = std::max(m, std::fabs(v[i]));
    return m;
}

void test_packet_travels() {
    std::printf("the ring train rides the wavefront\n");
    const float w = 0.12f;
    // Three ages of the same drop, as phases: the front sits at phase / k.
    for (float rf : {0.3f, 0.8f, 1.3f}) {
        auto v = radial(rf * kK, w);
        auto [pr, pv] = peak(v);
        // Within a wavelength of the front. Not tighter: the peak is where a
        // crest falls inside the envelope, and crests are 2*pi/k apart.
        const float lambda = 2.f * (float)M_PI / kK;
        char msg[128];
        std::snprintf(msg, sizeof msg,
                      "front at %.2f -> peak at %.2f (within one wavelength)", rf, pr);
        check(std::fabs(pr - rf) < lambda && pv > 1e-3f, msg);
    }

    // Behind the front is flat. This is the one the old formulation could never
    // satisfy: a standing field's largest crest is at r ~ 0, always.
    auto late = radial(1.3f * kK, w);
    const float behind = max_inside(late, 0.5f);
    const float front = peak(late).second;
    char msg[128];
    std::snprintf(msg, sizeof msg, "behind the front is still (%.4f vs %.4f at the front)",
                  behind, front);
    check(behind < 0.1f * front, msg);

    // With width 0 the same source is the old standing field, whose energy is
    // all at the centre. Keeps the packet from being a no-op the checks above
    // happen to tolerate, and covers the orbit source, which stays standing.
    auto standing = radial(1.3f * kK, 0.f);
    check(max_inside(standing, 0.5f) > 0.5f * peak(standing).second,
          "width 0 leaves the old standing field alone");

    // A bigger drop is a bigger splash: more of the frame is moving at the same
    // age. This is what an audio hit's strength buys.
    auto small = radial(0.8f * kK, 0.06f);
    auto big = radial(0.8f * kK, 0.30f);
    auto energy = [](const std::vector<float>& v) {
        float e = 0.f;
        for (float x : v) e += x * x;
        return e;
    };
    check(energy(big) > 2.f * energy(small), "a wider packet disturbs more water");
}

// --- the spawner ------------------------------------------------------------

// Runs the spawner over `secs` at 60 fps and returns every source it emitted,
// frame by frame.
std::vector<std::vector<mirror::RippleSource>> run(mirror::DropSpawner& sp,
                                                   const mirror::DropSpawnParams& p,
                                                   double secs, double t0 = 0.0) {
    std::vector<std::vector<mirror::RippleSource>> frames;
    const double dt = 1.0 / 60.0;
    for (int i = 0; i * dt < secs; ++i)
        frames.push_back(sp.update(t0 + i * dt, 1.6f, kRingFreq, 1.2f, p));
    return frames;
}

void test_rate() {
    std::printf("\nthe schedule spawns at the rate it is asked for\n");
    mirror::DropSpawnParams p;
    p.rate = 4.f;
    p.rate_jitter = 0.f;      // metronomic, so the count is exactly checkable
    p.max_active = 24;

    mirror::DropSpawner sp(7);
    run(sp, p, 10.0);
    const int n = sp.spawnCount();
    char msg[128];
    std::snprintf(msg, sizeof msg, "4/s for 10s -> %d drops", n);
    check(n >= 38 && n <= 42, msg);

    // Jitter must change the *timing*, not the average rate: a "randomness"
    // control that also halves the rainfall is two controls in a trench coat.
    mirror::DropSpawnParams j = p;
    j.rate_jitter = 1.f;
    mirror::DropSpawner sp2(7);
    run(sp2, j, 10.0);
    std::snprintf(msg, sizeof msg, "...and %d with full jitter, same mean rate",
                  sp2.spawnCount());
    check(sp2.spawnCount() >= 28 && sp2.spawnCount() <= 52, msg);

    // Rain off means nothing falls, but a trigger still lands: that is the
    // audio-only setting, and it is the whole point of the split.
    mirror::DropSpawnParams off = p;
    off.rain_on = false;
    mirror::DropSpawner sp3(7);
    run(sp3, off, 5.0);
    check(sp3.spawnCount() == 0, "rain off spawns nothing on its own");
    sp3.trigger(1.f, 0.f);
    sp3.update(5.0, 1.6f, kRingFreq, 1.2f, off);
    check(sp3.spawnCount() == 1 && sp3.drops().size() == 1,
          "...but a trigger still lands one");
}

void test_each_drop_is_its_own() {
    std::printf("\nevery drop is its own event\n");
    mirror::DropSpawnParams p;
    p.rate = 3.f;
    p.max_active = 24;
    mirror::DropSpawner sp(11);
    auto frames = run(sp, p, 6.0);

    // Some frame has several drops live at once, and in it no two share a
    // position or a phase. A shared clock or a shared slot would show up here
    // as identical phases across sources -- every front at the same radius.
    bool multi = false, distinct = true;
    for (const auto& f : frames) {
        if (f.size() < 3) continue;
        multi = true;
        for (size_t i = 0; i < f.size(); ++i)
            for (size_t j = i + 1; j < f.size(); ++j) {
                if (std::fabs(f[i][2] - f[j][2]) < 1e-6f) distinct = false;
                if (std::fabs(f[i][0] - f[j][0]) < 1e-6f &&
                    std::fabs(f[i][1] - f[j][1]) < 1e-6f) distinct = false;
            }
    }
    check(multi, "several drops are live at once");
    check(distinct, "and no two share a place or a phase");

    // Phases only ever grow while a drop lives, and each drop starts at ~0: the
    // renderer reads the wavefront radius off the phase, so a drop born at a
    // carried-over phase would appear as a ring already out from a centre
    // nothing happened at.
    float max_birth_phase = 0.f;
    for (const auto& f : frames)
        for (const auto& s : f)
            if (s[3] < 0.35f)   // still in the attack, i.e. just born
                max_birth_phase = std::max(max_birth_phase, s[2]);
    char msg[128];
    std::snprintf(msg, sizeof msg, "a new drop starts at its centre (phase %.3f)",
                  max_birth_phase);
    check(max_birth_phase < 0.5f, msg);

    // Widths and amplitudes scatter -- the "light randomization" that keeps a
    // shower from reading as one drop repeated.
    float wmin = 1e9f, wmax = 0.f, amin = 1e9f, amax = 0.f;
    for (const auto& f : frames)
        for (const auto& s : f) {
            wmin = std::min(wmin, s[4]); wmax = std::max(wmax, s[4]);
            if (s[3] > 0.9f) { amin = std::min(amin, s[3]); amax = std::max(amax, s[3]); }
        }
    check(wmax > wmin * 1.3f, "drops come in a range of sizes");
    check(amax > amin * 1.1f, "and a range of strengths");

    // Zero jitter is zero jitter: the controls have to be able to turn off.
    mirror::DropSpawnParams flat = p;
    flat.width_jitter = flat.amp_jitter = flat.speed_jitter = 0.f;
    mirror::DropSpawner sp2(11);
    auto ff = run(sp2, flat, 4.0);
    bool same_width = true;
    for (const auto& f : ff)
        for (const auto& s : f)
            if (std::fabs(s[4] - flat.width) > 1e-5f) same_width = false;
    check(same_width, "jitter at 0 makes every drop identical");
}

void test_budget_and_retirement() {
    std::printf("\ndrops are retired, and the budget holds\n");
    mirror::DropSpawnParams p;
    p.rate = 30.f;            // far more than can be alive at once
    p.rate_jitter = 0.f;
    p.max_active = 6;
    mirror::DropSpawner sp(3);
    auto frames = run(sp, p, 6.0);

    size_t worst = 0;
    for (const auto& f : frames) worst = std::max(worst, f.size());
    char msg[128];
    std::snprintf(msg, sizeof msg, "never more than the budget live (%d, cap 6)",
                  (int)worst);
    check(worst <= 6, msg);

    // Retirement: with the schedule off, the drops in flight must drain away on
    // their own. A drop that is never retired is a source the kernel keeps
    // paying for, forever.
    mirror::DropSpawnParams none = p;
    none.rain_on = false;
    for (int i = 0; i < 30; ++i)
        sp.update(6.0 + i / 60.0, 1.6f, kRingFreq, 1.2f, none);
    const size_t before = sp.drops().size();
    for (int i = 0; i < 60 * 20; ++i)
        sp.update(7.0 + i / 60.0, 1.6f, kRingFreq, 1.2f, none);
    std::snprintf(msg, sizeof msg, "they drain away once they leave the frame (%d -> %d)",
                  (int)before, (int)sp.drops().size());
    check(sp.drops().empty(), msg);
}

// One triggered drop, sampled past its attack -- at the instant it lands its
// amplitude is still ramping, so reading it there would compare two zeros.
mirror::RippleSource hit(mirror::DropSpawner& sp, const mirror::DropSpawnParams& p,
                         float strength, float pan) {
    sp.trigger(strength, pan);
    sp.update(0.0, 1.6f, kRingFreq, 1.2f, p);
    return sp.update(0.2, 1.6f, kRingFreq, 1.2f, p)[0];
}

void test_triggers() {
    std::printf("\nan outside trigger decides what its drop is like\n");
    mirror::DropSpawnParams p;
    p.rain_on = false;
    p.amp_jitter = p.width_jitter = 0.f;
    p.hit_amp = 1.f;
    p.hit_width = 1.f;

    mirror::DropSpawner sp(5);
    auto weak = hit(sp, p, 0.05f, 0.f);      // a whisper
    mirror::DropSpawner sp2(5);
    auto strong = hit(sp2, p, 1.0f, 0.f);    // a slam

    char msg[128];
    std::snprintf(msg, sizeof msg, "a loud hit is bigger than a quiet one (%.2f vs %.2f wide)",
                  strong[4], weak[4]);
    check(strong[4] > weak[4] * 1.5f, msg);
    check(strong[3] > weak[3] * 1.5f, "and stronger");
    // The quiet one still has to be visible: an onset that fires and shows
    // nothing is indistinguishable from a dropped trigger.
    check(weak[3] > 0.05f * p.amp && weak[4] > 0.02f, "a quiet hit still lands");

    // Ignoring strength is a setting, not an oversight: at 0 the audio decides
    // only *when*.
    mirror::DropSpawnParams fixed = p;
    fixed.hit_amp = fixed.hit_width = 0.f;
    mirror::DropSpawner sp3(5);
    auto a = hit(sp3, fixed, 0.05f, 0.f);
    mirror::DropSpawner sp4(5);
    auto b = hit(sp4, fixed, 1.0f, 0.f);
    check(std::fabs(a[3] - b[3]) < 1e-5f && std::fabs(a[4] - b[4]) < 1e-5f,
          "with the mapping at 0, strength changes nothing");

    // Pan: at full weight a hard-left hit lands left of centre.
    mirror::DropSpawnParams panned = p;
    panned.hit_pan = 1.f;
    mirror::DropSpawner sp5(5);
    auto left = hit(sp5, panned, 1.f, -1.f);
    mirror::DropSpawner sp6(5);
    auto right = hit(sp6, panned, 1.f, 1.f);
    std::snprintf(msg, sizeof msg, "pan places the drop (%.2f vs %.2f)", left[0], right[0]);
    check(left[0] < -1.f && right[0] > 1.f, msg);
}

void test_clock_moved_back() {
    std::printf("\na clock that jumps does not dump a shower\n");
    mirror::DropSpawnParams p;
    p.rate = 2.f;
    p.max_active = 12;
    mirror::DropSpawner sp(9);
    run(sp, p, 4.0);
    const int before = sp.spawnCount();

    // A scrub backwards, then a long jump forwards -- a preset load, a pause,
    // a show cue. Neither may spawn a backlog: the arrears belong to a run that
    // no longer exists.
    sp.update(0.0, 1.6f, kRingFreq, 1.2f, p);
    sp.update(600.0, 1.6f, kRingFreq, 1.2f, p);
    char msg[128];
    std::snprintf(msg, sizeof msg, "a 10-minute jump adds %d drops, not hundreds",
                  sp.spawnCount() - before);
    check(sp.spawnCount() - before <= p.max_active + 2, msg);
    check(sp.drops().size() <= size_t(p.max_active), "and the budget still holds");
}

}  // namespace

int main() {
    std::printf("drop_test: a raindrop is one impact spreading outward\n");
    test_packet_travels();
    test_rate();
    test_each_drop_is_its_own();
    test_budget_and_retirement();
    test_triggers();
    test_clock_moved_back();
    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
