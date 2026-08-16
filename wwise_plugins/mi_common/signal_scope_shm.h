// Shared-memory layout for the SignalScope capture plug-in and its external
// monitor app (wwise_plugins/scope_monitor). One process (the plug-in, running
// inside the Wwise sound engine or Wwise Authoring's local preview) writes;
// any number of processes (the monitor app) read.
//
// Two kinds of segment:
//  - one Directory: a small fixed-size table every SignalScope instance
//    registers itself in, so the monitor app can list "what's live" without
//    being told scope IDs out of band.
//  - one Ring per active instance, named by that instance's ScopeId, holding
//    the actual captured audio as a lock-free ring of planar float blocks.
//
// This relies on std::atomic<uint32_t/uint64_t> being lock-free and having the
// same in-memory representation as the plain integer on every platform this
// project targets (true for MSVC and libstdc++/libc++ on x86-64/ARM64) -- the
// C++ standard doesn't guarantee that across process boundaries, but no lock
// table or auxiliary state is involved for lock-free atomics on these ABIs, so
// the plain shared memory works. A visualizer reading a torn/stale value once
// in a while is a cosmetic non-issue, not a correctness one.
#pragma once

#include "shm_region.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mi::scope {

// A local min(), not std::min: AK/SoundEngine headers can drag in <windows.h>
// without NOMINMAX ahead of this header, and the resulting min/max macros
// mangle any std::min(...) call textually. Sidestep it rather than depend on
// include order.
template <typename T>
inline T Min(T a, T b) { return a < b ? a : b; }

constexpr uint32_t kDirectoryMagic = 0x53534B44; // 'SSKD'
constexpr uint32_t kRingMagic = 0x53534B52;       // 'SSKR'
constexpr uint32_t kLayoutVersion = 1;

constexpr int kMaxScopes = 64;
constexpr int kMaxLabelLen = 32;
constexpr int kMaxChannelsSupported = 16;
constexpr int kMaxChannelNameLen = 8; // "FL", "FR", "FC", "LFE", "BL", "SR", ... plus a nul

// ~1.4s of history at 48kHz; comfortably covers a spectrogram/waveform window
// without the shared segment getting unreasonably large per instance.
constexpr uint32_t kDefaultCapacityFrames = 65536;

inline const char* DirectoryName() { return "WwiseSignalScope_Directory_v1"; }

inline std::string RingName(uint32_t scopeId)
{
    return "WwiseSignalScope_Ring_v1_" + std::to_string(scopeId);
}

struct DirectorySlot
{
    std::atomic<uint32_t> active{ 0 }; // 0 = free/closed, 1 = live
    uint32_t scopeId = 0;
    char label[kMaxLabelLen] = {};
    uint32_t sampleRate = 0;
    uint32_t numChannels = 0;
    uint32_t capacityFrames = 0;
    std::atomic<uint64_t> generation{ 0 }; // bumped every Open(); lets a reader notice a re-open mid-read
};

struct Directory
{
    uint32_t magic = kDirectoryMagic;
    uint32_t version = kLayoutVersion;
    DirectorySlot slots[kMaxScopes];
};

struct RingHeader
{
    uint32_t magic = kRingMagic;
    uint32_t version = kLayoutVersion;
    uint32_t sampleRate = 0;
    uint32_t numChannels = 0;
    uint32_t capacityFrames = 0;
    std::atomic<uint64_t> writeCursorFrames{ 0 }; // total frames ever written, monotonic
    char channelNames[kMaxChannelsSupported][kMaxChannelNameLen] = {}; // e.g. "FL"/"FR"; blank if unknown
};

inline size_t RingBytes(uint32_t numChannels, uint32_t capacityFrames)
{
    return sizeof(RingHeader) + size_t(numChannels) * size_t(capacityFrames) * sizeof(float);
}

// Per-channel planar layout: channel c's ring occupies
// samples[c * capacityFrames .. c * capacityFrames + capacityFrames).
inline float* RingSamples(RingHeader* hdr)
{
    return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(RingHeader));
}

// ---------------------------------------------------------------------------
// Writer side: owned by the SignalScope sound-engine plug-in instance.
// ---------------------------------------------------------------------------
class ScopeWriter
{
public:
    ~ScopeWriter() { Close(); }

    // channelNames is optional: pass numChannels entries (each a short string
    // like "FL"/"FR"/"LFE") to label the channels in the monitor app, or
    // nullptr to leave them blank (the UI falls back to "Ch <n>").
    bool Open(uint32_t scopeId, const char* label, uint32_t sampleRate, uint32_t numChannels,
        const char* const* channelNames = nullptr, uint32_t capacityFrames = kDefaultCapacityFrames)
    {
        Close();

        if (numChannels == 0 || numChannels > kMaxChannelsSupported)
            return false;

        if (!m_dir.CreateOrOpen(DirectoryName(), sizeof(Directory)))
            return false;
        Directory* dir = static_cast<Directory*>(m_dir.Data());
        if (dir->magic != kDirectoryMagic)
        {
            // First opener of a freshly created segment: lay out the header.
            dir->magic = kDirectoryMagic;
            dir->version = kLayoutVersion;
        }

        // Reuse the slot already claimed by this scopeId if there is one (e.g. the
        // plug-in was re-Init()'d), otherwise take the first free slot.
        int slot = -1;
        for (int i = 0; i < kMaxScopes; ++i)
        {
            if (dir->slots[i].active.load(std::memory_order_relaxed) &&
                dir->slots[i].scopeId == scopeId)
            {
                slot = i;
                break;
            }
        }
        if (slot < 0)
        {
            for (int i = 0; i < kMaxScopes; ++i)
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
            return false; // directory full
        }

        const size_t ringBytes = RingBytes(numChannels, capacityFrames);
        if (!m_ring.CreateOrOpen(RingName(scopeId), ringBytes))
        {
            m_dir.Close();
            return false;
        }

        m_hdr = static_cast<RingHeader*>(m_ring.Data());
        // A fresh OS-backed page is zero-filled, so magic == 0 reliably means
        // "we're the first to create this segment" -- reinitialize the header
        // and reset the write cursor. Reopening one that's already live must NOT
        // reset the cursor, or every reader watching it stutters.
        if (m_hdr->magic != kRingMagic || m_hdr->numChannels != numChannels ||
            m_hdr->capacityFrames != capacityFrames)
        {
            m_hdr->magic = kRingMagic;
            m_hdr->version = kLayoutVersion;
            m_hdr->sampleRate = sampleRate;
            m_hdr->numChannels = numChannels;
            m_hdr->capacityFrames = capacityFrames;
            m_hdr->writeCursorFrames.store(0, std::memory_order_relaxed);
        }
        else
        {
            m_hdr->sampleRate = sampleRate; // sample rate can legitimately change across Init()
        }

        // Refreshed on every Open(), independent of the reinit branch above --
        // names are cheap to rewrite and should always reflect this binding.
        for (uint32_t c = 0; c < kMaxChannelsSupported; ++c)
        {
            char* dst = m_hdr->channelNames[c];
            std::memset(dst, 0, kMaxChannelNameLen);
            if (channelNames && c < numChannels && channelNames[c])
                std::strncpy(dst, channelNames[c], kMaxChannelNameLen - 1);
        }

        m_samples = RingSamples(m_hdr);
        m_numChannels = numChannels;
        m_capacityFrames = capacityFrames;

        DirectorySlot& s = dir->slots[slot];
        s.scopeId = scopeId;
        std::memset(s.label, 0, sizeof(s.label));
        if (label)
            std::strncpy(s.label, label, sizeof(s.label) - 1);
        s.sampleRate = sampleRate;
        s.numChannels = numChannels;
        s.capacityFrames = capacityFrames;
        s.generation.fetch_add(1, std::memory_order_relaxed);
        s.active.store(1, std::memory_order_release);

        m_slot = slot;
        m_scopeId = scopeId;
        return true;
    }

    void Close()
    {
        if (m_dir.IsValid() && m_slot >= 0)
        {
            Directory* dir = static_cast<Directory*>(m_dir.Data());
            dir->slots[m_slot].active.store(0, std::memory_order_release);
        }
        m_ring.Close();
        m_dir.Close();
        m_hdr = nullptr;
        m_samples = nullptr;
        m_slot = -1;
    }

    bool IsOpen() const { return m_hdr != nullptr; }

    // Writes one block of planar (per-channel contiguous) audio. channelPtrs must
    // have at least the numChannels passed to Open(); extra input channels beyond
    // that are ignored.
    void Write(const float* const* channelPtrs, uint32_t numFrames)
    {
        if (!m_hdr || numFrames == 0)
            return;

        const uint64_t startCursor = m_hdr->writeCursorFrames.load(std::memory_order_relaxed);
        // A block larger than the ring is only kept for its tail (via srcOffset
        // below); the rest would be immediately overwritten anyway.
        const uint32_t framesLeft = Min(numFrames, m_capacityFrames);

        uint64_t cursor = startCursor;
        for (uint32_t c = 0; c < m_numChannels; ++c)
        {
            float* ring = m_samples + size_t(c) * m_capacityFrames;
            const float* src = channelPtrs[c] ? channelPtrs[c] : nullptr;
            uint32_t pos = uint32_t(cursor % m_capacityFrames);
            uint32_t remaining = framesLeft;
            uint32_t srcOffset = numFrames - framesLeft;
            while (remaining > 0)
            {
                const uint32_t run = Min(remaining, m_capacityFrames - pos);
                if (src)
                    std::memcpy(ring + pos, src + srcOffset, run * sizeof(float));
                else
                    std::memset(ring + pos, 0, run * sizeof(float));
                pos = (pos + run) % m_capacityFrames;
                srcOffset += run;
                remaining -= run;
            }
        }

        m_hdr->writeCursorFrames.store(startCursor + framesLeft, std::memory_order_release);
    }

private:
    ShmRegion m_dir;
    ShmRegion m_ring;
    RingHeader* m_hdr = nullptr;
    float* m_samples = nullptr;
    uint32_t m_numChannels = 0;
    uint32_t m_capacityFrames = 0;
    int m_slot = -1;
    uint32_t m_scopeId = 0;
};

// ---------------------------------------------------------------------------
// Reader side: used by the monitor app. Read-only, never creates a segment.
// ---------------------------------------------------------------------------
struct ScopeInfo
{
    uint32_t scopeId = 0;
    std::string label;
    uint32_t sampleRate = 0;
    uint32_t numChannels = 0;
    uint32_t capacityFrames = 0;
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

    // Snapshot of every currently-active scope. Safe to call every frame; cheap.
    std::vector<ScopeInfo> ListActive()
    {
        std::vector<ScopeInfo> out;
        if (!Open())
            return out;
        const Directory* dir = static_cast<const Directory*>(m_dir.Data());
        if (dir->magic != kDirectoryMagic)
            return out;
        for (int i = 0; i < kMaxScopes; ++i)
        {
            const DirectorySlot& s = dir->slots[i];
            if (!s.active.load(std::memory_order_acquire))
                continue;
            ScopeInfo info;
            info.scopeId = s.scopeId;
            info.label.assign(s.label, strnlen(s.label, kMaxLabelLen));
            info.sampleRate = s.sampleRate;
            info.numChannels = s.numChannels;
            info.capacityFrames = s.capacityFrames;
            info.generation = s.generation.load(std::memory_order_relaxed);
            out.push_back(std::move(info));
        }
        return out;
    }

private:
    ShmRegion m_dir;
};

class RingReader
{
public:
    bool Open(uint32_t scopeId)
    {
        Close();

        ShmRegion peek;
        if (!peek.OpenExisting(RingName(scopeId), sizeof(RingHeader)))
            return false;
        const RingHeader* peekHdr = static_cast<const RingHeader*>(peek.Data());
        if (peekHdr->magic != kRingMagic)
            return false;
        const uint32_t numChannels = peekHdr->numChannels;
        const uint32_t capacityFrames = peekHdr->capacityFrames;
        peek.Close();

        if (!m_ring.OpenExisting(RingName(scopeId), RingBytes(numChannels, capacityFrames)))
            return false;

        m_hdr = static_cast<RingHeader*>(m_ring.Data());
        m_samples = RingSamples(m_hdr);
        m_scopeId = scopeId;
        return true;
    }

    void Close()
    {
        m_ring.Close();
        m_hdr = nullptr;
        m_samples = nullptr;
    }

    bool IsOpen() const { return m_hdr != nullptr; }
    uint32_t NumChannels() const { return m_hdr ? m_hdr->numChannels : 0; }
    uint32_t CapacityFrames() const { return m_hdr ? m_hdr->capacityFrames : 0; }
    uint32_t SampleRate() const { return m_hdr ? m_hdr->sampleRate : 0; }

    // Empty string if the writer didn't provide a name for this channel.
    const char* ChannelName(uint32_t channel) const
    {
        if (!m_hdr || channel >= m_hdr->numChannels)
            return "";
        return m_hdr->channelNames[channel];
    }
    uint64_t WriteCursor() const { return m_hdr ? m_hdr->writeCursorFrames.load(std::memory_order_acquire) : 0; }

    // Copies the most recent `numFrames` frames of channel `channel` (0-based)
    // into out[0..numFrames), oldest first. Frames not yet written (cursor hasn't
    // reached capacityFrames yet) come back as zero. Not glitch-proof if the
    // writer laps the reader mid-copy -- acceptable for a visualizer.
    void ReadLatest(uint32_t channel, float* out, uint32_t numFrames) const
    {
        if (!m_hdr || channel >= m_hdr->numChannels)
        {
            std::memset(out, 0, numFrames * sizeof(float));
            return;
        }
        const uint32_t capacity = m_hdr->capacityFrames;
        const uint64_t cursor = WriteCursor();
        const uint64_t available = Min<uint64_t>(cursor, capacity);
        const uint32_t validCount = uint32_t(Min<uint64_t>(available, numFrames));
        const uint32_t zeroCount = numFrames - validCount;

        std::memset(out, 0, zeroCount * sizeof(float));

        const float* ring = m_samples + size_t(channel) * capacity;
        uint64_t startFrame = cursor - validCount;
        uint32_t pos = uint32_t(startFrame % capacity);
        uint32_t remaining = validCount;
        float* dst = out + zeroCount;
        while (remaining > 0)
        {
            const uint32_t run = Min(remaining, capacity - pos);
            std::memcpy(dst, ring + pos, run * sizeof(float));
            dst += run;
            pos = (pos + run) % capacity;
            remaining -= run;
        }
    }

private:
    ShmRegion m_ring;
    RingHeader* m_hdr = nullptr;
    float* m_samples = nullptr;
    uint32_t m_scopeId = 0;
};

} // namespace mi::scope
