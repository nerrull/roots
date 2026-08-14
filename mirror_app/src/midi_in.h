// midi_in — CoreMIDI control-change input.
//
// One job: turn every MIDI source on the machine into a queue of control-change
// messages the render loop can drain. No note handling, no clock, no output.
//
// Sources are connected as they appear rather than enumerated once at startup:
// a controller plugged in after the app is running is the normal case, not an
// edge case, and an app that only saw devices present at launch would look
// broken in exactly the situation where someone is trying to get one working.
//
// The read callback runs on CoreMIDI's own high-priority thread, so it does the
// least possible work -- push three bytes under a mutex -- and everything else
// happens when the frame loop drains it.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace midi {

struct CC {
    int channel;   // 0-15
    int cc;        // controller number
    int value;     // 0-127
};

class Input {
public:
    Input();
    ~Input();

    // Opens the client and connects every current source. Returns false with
    // `err` set if CoreMIDI is unavailable; having no devices is not an error.
    bool open(std::string& err);
    void close();
    bool isOpen() const;

    // Connect any sources that have appeared since the last call. Cheap enough
    // to call once a second from the frame loop.
    void rescan();

    int  sourceCount() const;
    std::string deviceInfo() const;

    // Hand over everything received since the last drain, oldest first.
    void drain(std::vector<CC>& out);

    // Total messages seen, so a device that is connected but silent can be
    // told from one that is not connected at all.
    uint64_t received() const;

    // Public only because CoreMIDI's read callback is a free function that has
    // to reach it. Incomplete here, so nothing leaks.
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace midi
