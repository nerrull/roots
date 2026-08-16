// audio_pulse — onsets from the Wwise OnsetTap plug-in, as things to react to.
//
// The detection happens in the sound engine, on the audio thread, where the
// audio actually is (wwise_plugins/OnsetTap). This side only reads conclusions
// out of shared memory: "a transient landed, this hard, panned here". Polled
// once per frame from the render loop, which is all the resolution a visual can
// use anyway -- a drop that lands one frame late is a drop that lands on time.
//
// A thin wrapper over mi::onset rather than using it directly, because the panel
// needs three things the raw reader does not provide: a tap list that is not
// re-enumerated 60 times a second, a rate to display, and a notion of "live"
// that survives the two processes not sharing a clock epoch.
#pragma once

#include <cstdint>
#include <chrono>
#include <string>
#include <vector>

#include "onset_shm.h"

namespace mirror {

using AudioTap = mi::onset::TapInfo;
using AudioOnset = mi::onset::Event;

class AudioPulses {
public:
    // Taps currently published by any OnsetTap instance. Cached: enumerating
    // walks a shared directory, and nothing about it changes at frame rate.
    const std::vector<AudioTap>& taps() {
        const double now = Now();
        if (now - listed_at_ > 0.5 || taps_.empty()) {
            taps_ = dir_.ListActive();
            listed_at_ = now;
        }
        return taps_;
    }

    bool connect(uint32_t tapId) {
        connected_ = reader_.Open(tapId);
        if (connected_) {
            tap_id_ = tapId;
            label_ = reader_.Label();
            last_beat_ = 0;
            beat_seen_at_ = Now();
            recent_.clear();
        }
        return connected_;
    }

    void disconnect() {
        reader_.Close();
        connected_ = false;
        label_.clear();
        recent_.clear();
    }

    bool connected() const { return connected_; }
    uint32_t tapId() const { return tap_id_; }
    const std::string& label() const { return label_; }

    // Everything published since the last call, oldest first.
    std::vector<AudioOnset> poll() {
        std::vector<AudioOnset> out;
        if (!connected_) return out;
        reader_.Poll(out);

        const double now = Now();
        // Liveness from the heartbeat *changing*, not from its age: the stamp
        // comes from the writer's steady clock, whose epoch this process has no
        // right to assume is its own. A value that keeps moving means a sound
        // engine is running the plug-in right now, whatever it counts from.
        const uint64_t beat = reader_.HeartbeatNs();
        if (beat != last_beat_) {
            last_beat_ = beat;
            beat_seen_at_ = now;
        }

        for (size_t i = 0; i < out.size(); ++i) recent_.push_back(now);
        while (!recent_.empty() && now - recent_.front() > 5.0)
            recent_.erase(recent_.begin());
        if (!out.empty()) last_onset_at_ = now;
        return out;
    }

    // True while the plug-in is actually processing audio. A tap can be listed
    // (the instance exists) while its bus is silent or its Wwise session is
    // stopped, which looks exactly like "no onsets" unless this is shown.
    bool live() const { return connected_ && (Now() - beat_seen_at_) < 0.5; }

    float levelDb() const { return connected_ ? reader_.LevelDb() : -200.f; }
    float thresholdDb() const { return connected_ ? reader_.ThresholdDb() : 0.f; }
    // Onsets per second over the last 5 s.
    float rate() const { return float(recent_.size()) / 5.f; }
    // Seconds since the last onset, for a "last hit" indicator.
    double sinceLast() const { return Now() - last_onset_at_; }
    uint32_t missed() const { return reader_.Missed(); }

private:
    static double Now() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }

    mi::onset::DirectoryReader dir_;
    mi::onset::TapReader reader_;
    std::vector<AudioTap> taps_;
    std::vector<double> recent_;
    std::string label_;
    double listed_at_ = -1e9;
    double beat_seen_at_ = 0.0;
    double last_onset_at_ = -1e9;
    uint64_t last_beat_ = 0;
    uint32_t tap_id_ = 0;
    bool connected_ = false;
};

}  // namespace mirror
