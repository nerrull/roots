// Shared-memory layout for the OnsetTap plug-in and whatever reacts to it.
//
// The sibling of signal_scope_shm.h, and deliberately not the same thing.
// SignalScope publishes *audio*: a big ring the monitor app draws. This
// publishes *events*: "a transient just landed, this hard, panned here". A
// visual program that wanted onsets out of the audio ring would have to run its
// own detector over samples it is always slightly behind on, on a thread that
// has a frame to draw -- the detection belongs next to the audio, on the audio
// thread, and only the conclusion needs to cross the process boundary.
//
// One writer (the plug-in, in the Wwise sound engine or Authoring's preview),
// any number of readers. Two segments, following the same shape as the scope:
//  - one Directory every tap registers in, so a reader can list what is live
//    without being told tap IDs out of band;
//  - one Stream per tap, holding a small ring of events plus the live meter
//    values (level and the threshold it would have to clear) so a reader can
//    show why nothing is firing.
//
// Same lock-free-atomics-across-processes assumption as signal_scope_shm.h; see
// the note there. A reader that misses an event because the writer lapped it is
// a dropped visual cue, not a corruption -- and the ring is sized so that only
// happens if the reader stops polling for seconds at a time.
#pragma once

#include "shm_region.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mi::onset {

constexpr uint32_t kDirectoryMagic = 0x4F544B44; // 'OTKD'
constexpr uint32_t kStreamMagic = 0x4F544B53;    // 'OTKS'
constexpr uint32_t kLayoutVersion = 1;

constexpr int kMaxTaps = 16;
constexpr int kMaxLabelLen = 32;

// Power of two so the cursor wraps with a mask. 256 events is ~30 s of dense
// percussion, i.e. a reader has to stall for a very long time to lose one.
constexpr uint32_t kEventCapacity = 256;

inline const char* DirectoryName() { return "WwiseOnsetTap_Directory_v1"; }

inline std::string StreamName(uint32_t tapId)
{
    return "WwiseOnsetTap_Stream_v1_" + std::to_string(tapId);
}

// One detected transient.
struct Event
{
    // Monotonic, starting at 1. A reader tracks the last sequence it saw, so it
    // can tell "nothing new" from "I missed some" -- and never replays an event
    // it already spawned something for.
    uint64_t seq = 0;
    // The writer's steady clock at detection, nanoseconds. Meaningful only
    // against other events from the same writer (both sides are on one machine,
    // but not necessarily on one clock base).
    uint64_t hostTimeNs = 0;
    float strength = 0.f;   // 0..1 loudness within the detector's floor..0 dB
    float levelDb = 0.f;    // absolute level at the hit
    float excessDb = 0.f;   // how far the rise cleared the adaptive threshold
    float pan = 0.f;        // -1 left .. +1 right; 0 for mono or centred
    uint32_t channels = 0;  // channels the detector summed
    uint32_t reserved = 0;
};

struct DirectorySlot
{
    std::atomic<uint32_t> active{ 0 };
    uint32_t tapId = 0;
    char label[kMaxLabelLen] = {};
    uint32_t sampleRate = 0;
    std::atomic<uint64_t> generation{ 0 };
};

struct Directory
{
    uint32_t magic = kDirectoryMagic;
    uint32_t version = kLayoutVersion;
    DirectorySlot slots[kMaxTaps];
};

struct StreamHeader
{
    uint32_t magic = kStreamMagic;
    uint32_t version = kLayoutVersion;
    uint32_t sampleRate = 0;
    uint32_t tapId = 0;
    char label[kMaxLabelLen] = {};

    // Total events ever published. Also the ring's write position, masked.
    std::atomic<uint64_t> writeCursor{ 0 };

    // Live meter, republished every block. Floats as bit patterns because
    // std::atomic<float> is not guaranteed lock-free everywhere this has to be
    // ABI-compatible across two processes.
    std::atomic<uint32_t> levelDbBits{ 0 };
    std::atomic<uint32_t> thresholdDbBits{ 0 };
    // Steady-clock stamp of the last block processed: a reader shows "live" or
    // "silent" from this rather than from a plug-in that may simply not be
    // running right now.
    std::atomic<uint64_t> heartbeatNs{ 0 };

    Event events[kEventCapacity];
};

inline uint32_t FloatBits(float v)
{
    uint32_t b;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

inline float BitsFloat(uint32_t b)
{
    float v;
    std::memcpy(&v, &b, sizeof(v));
    return v;
}

// ---------------------------------------------------------------------------
// Writer: owned by the OnsetTap sound-engine plug-in instance.
// ---------------------------------------------------------------------------
class TapWriter
{
public:
    ~TapWriter() { Close(); }

    bool Open(uint32_t tapId, const char* label, uint32_t sampleRate)
    {
        Close();

        if (!m_dir.CreateOrOpen(DirectoryName(), sizeof(Directory)))
            return false;
        Directory* dir = static_cast<Directory*>(m_dir.Data());
        if (dir->magic != kDirectoryMagic)
        {
            dir->magic = kDirectoryMagic;
            dir->version = kLayoutVersion;
        }

        int slot = -1;
        for (int i = 0; i < kMaxTaps; ++i)
        {
            if (dir->slots[i].active.load(std::memory_order_relaxed) &&
                dir->slots[i].tapId == tapId)
            {
                slot = i;
                break;
            }
        }
        if (slot < 0)
        {
            for (int i = 0; i < kMaxTaps; ++i)
            {
                if (!dir->slots[i].active.load(std::memory_order_relaxed))
                {
                    slot = i;
                    break;
                }
            }
        }
        if (slot < 0)
        {
            m_dir.Close();
            return false;
        }

        if (!m_stream.CreateOrOpen(StreamName(tapId), sizeof(StreamHeader)))
        {
            m_dir.Close();
            return false;
        }
        m_hdr = static_cast<StreamHeader*>(m_stream.Data());

        // A fresh OS page is zero-filled, so magic == 0 means we created it.
        // Reopening a live stream must not reset the cursor: a reader watching
        // it would see the sequence go backwards and replay old events.
        if (m_hdr->magic != kStreamMagic)
        {
            m_hdr->magic = kStreamMagic;
            m_hdr->version = kLayoutVersion;
            m_hdr->writeCursor.store(0, std::memory_order_relaxed);
        }
        m_hdr->sampleRate = sampleRate;
        m_hdr->tapId = tapId;
        std::memset(m_hdr->label, 0, sizeof(m_hdr->label));
        if (label)
            std::strncpy(m_hdr->label, label, sizeof(m_hdr->label) - 1);

        DirectorySlot& s = dir->slots[slot];
        s.tapId = tapId;
        std::memset(s.label, 0, sizeof(s.label));
        if (label)
            std::strncpy(s.label, label, sizeof(s.label) - 1);
        s.sampleRate = sampleRate;
        s.generation.fetch_add(1, std::memory_order_relaxed);
        s.active.store(1, std::memory_order_release);

        m_slot = slot;
        return true;
    }

    void Close()
    {
        if (m_dir.IsValid() && m_slot >= 0)
        {
            Directory* dir = static_cast<Directory*>(m_dir.Data());
            dir->slots[m_slot].active.store(0, std::memory_order_release);
        }
        m_stream.Close();
        m_dir.Close();
        m_hdr = nullptr;
        m_slot = -1;
    }

    bool IsOpen() const { return m_hdr != nullptr; }

    // Publishes one event. The slot is filled before the cursor advances, so a
    // reader that sees the new cursor always sees a complete event behind it.
    void Push(const Event& e)
    {
        if (!m_hdr)
            return;
        const uint64_t cursor = m_hdr->writeCursor.load(std::memory_order_relaxed);
        Event& dst = m_hdr->events[cursor & (kEventCapacity - 1)];
        dst = e;
        dst.seq = cursor + 1;
        m_hdr->writeCursor.store(cursor + 1, std::memory_order_release);
    }

    void PublishMeter(float levelDb, float thresholdDb, uint64_t nowNs)
    {
        if (!m_hdr)
            return;
        m_hdr->levelDbBits.store(FloatBits(levelDb), std::memory_order_relaxed);
        m_hdr->thresholdDbBits.store(FloatBits(thresholdDb), std::memory_order_relaxed);
        m_hdr->heartbeatNs.store(nowNs, std::memory_order_release);
    }

private:
    ShmRegion m_dir;
    ShmRegion m_stream;
    StreamHeader* m_hdr = nullptr;
    int m_slot = -1;
};

// ---------------------------------------------------------------------------
// Reader: used by whatever is reacting to the onsets. Read-only, never creates.
// ---------------------------------------------------------------------------
struct TapInfo
{
    uint32_t tapId = 0;
    std::string label;
    uint32_t sampleRate = 0;
    uint64_t generation = 0;
};

class DirectoryReader
{
public:
    bool Open()
    {
        if (m_dir.IsValid())
            return true;
        return m_dir.OpenExisting(DirectoryName(), sizeof(Directory));
    }

    // Every tap currently registered. Cheap; safe to call every frame.
    std::vector<TapInfo> ListActive()
    {
        std::vector<TapInfo> out;
        if (!Open())
            return out;
        const Directory* dir = static_cast<const Directory*>(m_dir.Data());
        if (dir->magic != kDirectoryMagic)
            return out;
        for (int i = 0; i < kMaxTaps; ++i)
        {
            const DirectorySlot& s = dir->slots[i];
            if (!s.active.load(std::memory_order_acquire))
                continue;
            TapInfo info;
            info.tapId = s.tapId;
            info.label.assign(s.label, strnlen(s.label, kMaxLabelLen));
            info.sampleRate = s.sampleRate;
            info.generation = s.generation.load(std::memory_order_relaxed);
            out.push_back(std::move(info));
        }
        return out;
    }

private:
    ShmRegion m_dir;
};

class TapReader
{
public:
    bool Open(uint32_t tapId)
    {
        Close();
        if (!m_stream.OpenExisting(StreamName(tapId), sizeof(StreamHeader)))
            return false;
        m_hdr = static_cast<StreamHeader*>(m_stream.Data());
        if (m_hdr->magic != kStreamMagic)
        {
            Close();
            return false;
        }
        m_tapId = tapId;
        // Start from now, not from the beginning of the ring: a program that
        // connects mid-show must not spawn a visual for every hit that landed
        // while it was not looking.
        m_seen = m_hdr->writeCursor.load(std::memory_order_acquire);
        return true;
    }

    void Close()
    {
        m_stream.Close();
        m_hdr = nullptr;
        m_seen = 0;
        m_missed = 0;
    }

    bool IsOpen() const { return m_hdr != nullptr; }
    uint32_t TapId() const { return m_tapId; }
    const char* Label() const { return m_hdr ? m_hdr->label : ""; }
    uint32_t SampleRate() const { return m_hdr ? m_hdr->sampleRate : 0; }

    float LevelDb() const
    {
        return m_hdr ? BitsFloat(m_hdr->levelDbBits.load(std::memory_order_relaxed)) : -200.f;
    }
    float ThresholdDb() const
    {
        return m_hdr ? BitsFloat(m_hdr->thresholdDbBits.load(std::memory_order_relaxed)) : 0.f;
    }
    uint64_t HeartbeatNs() const
    {
        return m_hdr ? m_hdr->heartbeatNs.load(std::memory_order_acquire) : 0;
    }

    // Appends every event published since the last call. Returns how many.
    //
    // If the writer has lapped the reader, the events that were overwritten are
    // gone: this skips to the oldest one still intact and counts the loss in
    // Missed() rather than handing back garbage or replaying stale hits.
    uint32_t Poll(std::vector<Event>& out)
    {
        if (!m_hdr)
            return 0;
        const uint64_t cursor = m_hdr->writeCursor.load(std::memory_order_acquire);
        if (cursor <= m_seen)
        {
            m_seen = cursor;   // writer restarted: resync rather than replay
            return 0;
        }
        uint64_t from = m_seen;
        if (cursor - from > kEventCapacity)
        {
            m_missed += uint32_t(cursor - from - kEventCapacity);
            from = cursor - kEventCapacity;
        }
        uint32_t n = 0;
        for (uint64_t i = from; i < cursor; ++i)
        {
            const Event& e = m_hdr->events[i & (kEventCapacity - 1)];
            // The sequence check catches the one race worth catching: the
            // writer overwriting this very slot between the cursor read and
            // the copy. A stale slot is dropped, not shown.
            if (e.seq != i + 1)
                continue;
            out.push_back(e);
            ++n;
        }
        m_seen = cursor;
        return n;
    }

    uint32_t Missed() const { return m_missed; }

private:
    ShmRegion m_stream;
    StreamHeader* m_hdr = nullptr;
    uint64_t m_seen = 0;
    uint32_t m_missed = 0;
    uint32_t m_tapId = 0;
};

} // namespace mi::onset
