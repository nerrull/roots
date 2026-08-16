#include "drop_spawner.h"

#include <algorithm>
#include <cmath>

namespace mirror {

namespace {

// A drop's wavefront radius at age `age`: the phase the renderer will see,
// divided by k. Kept in one place because the retirement test and the phase the
// source carries have to agree about where the rings are.
inline float front_radius(double age, float ring_freq, float speed) {
    const float k = ring_freq * (float)M_PI;
    return (float)(2.0 * M_PI * speed * age) / std::max(k, 1e-4f);
}

}  // namespace

Drop DropSpawner::makeDrop(double t, float asp, const DropSpawnParams& p,
                           float strength, float pan, bool from_audio) {
    Drop d;
    d.birth = t;
    d.from_audio = from_audio;

    const float ax = std::max(p.area_x, 0.f) * asp;
    const float ay = std::max(p.area_y, 0.f);
    d.cx = p.bias_x + uniform(-ax, ax);
    d.cy = p.bias_y + uniform(-ay, ay);

    d.amp = std::max(scatter(p.amp, p.amp_jitter), 0.f);
    d.width = std::max(scatter(p.width, p.width_jitter), 0.01f);
    d.speed = std::max(scatter(p.speed, p.speed_jitter), 0.05f);

    if (strength >= 0.f) {
        const float s = std::clamp(strength, 0.f, 1.f);
        // A hit's strength scales *toward* it rather than replacing it: at
        // hit_amp 0 the drop is whatever the sliders say and the audio only
        // decides timing, at 1 a weak hit is a faint drop. The 0.15 floor keeps
        // the quietest onset from spawning a drop nobody can see -- an event
        // that fires and shows nothing reads as a dropped trigger.
        const float k = std::clamp(p.hit_amp, 0.f, 1.f);
        d.amp *= (1.f - k) + k * (0.15f + 0.85f * s);
        const float kw = std::clamp(p.hit_width, 0.f, 1.f);
        d.width *= (1.f - kw) + kw * (0.4f + 1.2f * s);
    }
    if (from_audio && p.hit_pan > 0.f) {
        // Pan places the drop across the frame; the blend keeps some scatter so
        // a mono source does not put every drop on the centre line.
        const float k = std::clamp(p.hit_pan, 0.f, 1.f);
        const float px = std::clamp(pan, -1.f, 1.f) * asp;
        d.cx = (1.f - k) * d.cx + k * px;
    }
    return d;
}

void DropSpawner::retire(double t, float ring_freq, float speed,
                         const DropSpawnParams& p) {
    // The frame's far corner: once a drop's train has passed it, nothing it does
    // is on screen any more. The +3 widths is the tail of the Gaussian, which
    // has to clear the corner too or drops visibly blink out at the edge.
    const float corner = std::sqrt(1.f + 1.f) + 0.5f;
    drops_.erase(std::remove_if(drops_.begin(), drops_.end(),
                                [&](const Drop& d) {
                                    const double age = t - d.birth;
                                    if (age < 0.0) return true;   // clock moved back
                                    if (age > p.life) return true;
                                    const float rf = front_radius(age, ring_freq,
                                                                  speed * d.speed);
                                    const float w = d.width * (1.f + kDropSpread * rf);
                                    return rf - 3.f * w > corner;
                                }),
                 drops_.end());
}

const std::vector<RippleSource>& DropSpawner::update(double t, float asp,
                                                     float ring_freq, float speed,
                                                     const DropSpawnParams& p) {
    // A clock that moved backwards is a scrub or a reset, not elapsed time:
    // reschedule from here rather than spawning a burst to "catch up" to a
    // schedule that belongs to a run that no longer exists.
    if (t < last_t_ || next_spawn_ < 0.0) next_spawn_ = t;
    last_t_ = t;

    const int budget = std::max(p.max_active, 1);

    // Externally triggered hits first: they are the ones with a deadline.
    for (const Pending& hit : pending_) {
        if ((int)drops_.size() >= budget) drops_.erase(drops_.begin());
        drops_.push_back(makeDrop(t, asp, p, hit.strength, hit.pan, true));
        ++spawns_;
    }
    pending_.clear();

    if (p.rain_on && p.rate > 0.f) {
        const double mean = 1.0 / std::max(p.rate, 1e-3f);
        // Bounded catch-up: a stall or a long pause must not dump a frame's
        // worth of arrears onto the surface at once.
        int guard = budget;
        while (t >= next_spawn_ && guard-- > 0) {
            if ((int)drops_.size() >= budget) drops_.erase(drops_.begin());
            drops_.push_back(makeDrop(next_spawn_, asp, p, -1.f, 0.f, false));
            ++spawns_;

            // Regularity: the metronomic gap and an exponential one, blended.
            // The exponential is what makes rain sound and look like rain --
            // gaps that cluster -- and the blend is there because pure Poisson
            // at a low rate leaves stretches with nothing happening at all.
            const float u = std::clamp(uniform(0.f, 1.f), 1e-4f, 0.9999f);
            const double expo = -std::log(1.0 - u);
            const double j = std::clamp(p.rate_jitter, 0.f, 1.f);
            next_spawn_ += mean * ((1.0 - j) + j * expo);
        }
        if (guard <= 0) next_spawn_ = t + mean;   // gave up catching up
    } else {
        next_spawn_ = t;   // rain off: don't accrue a backlog while it is off
    }

    retire(t, ring_freq, speed, p);

    src_.clear();
    src_.reserve(drops_.size());
    for (const Drop& d : drops_) {
        const double age = std::max(t - d.birth, 0.0);
        // Phase from *this drop's* birth, so the renderer's wavefront radius
        // (phase / k) is where this drop's rings actually are. A shared clock
        // would put every drop's front at the same radius, which is the one
        // thing a drop is not allowed to have in common with the others.
        const float phase = 2.f * (float)M_PI * speed * d.speed * (float)age;
        // A few milliseconds of attack: at age 0 the packet is a spike at a
        // point, and stepping it in whole would pop a full ring on one frame.
        const float attack = std::min((float)age / 0.05f, 1.f);
        src_.push_back({d.cx, d.cy, phase, d.amp * attack * attack, d.width});
    }
    return src_;
}

}  // namespace mirror
