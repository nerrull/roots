// Dev-only console smoke test for the reader side, paired with test_writer.cpp.
// Not part of the CMake build. From this directory:
//   cl /std:c++17 /EHsc src/test_reader.cpp /Fe:test_reader.exe
#include "../../mi_common/signal_scope_shm.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

int main()
{
    mi::scope::DirectoryReader dir;
    auto scopes = dir.ListActive();
    std::printf("Active scopes: %zu\n", scopes.size());
    for (auto& s : scopes)
    {
        std::printf("  #%u '%s' %u ch @ %u Hz, capacity %u frames, gen %llu\n",
            s.scopeId, s.label.c_str(), s.numChannels, s.sampleRate, s.capacityFrames,
            (unsigned long long)s.generation);
    }
    if (scopes.empty())
        return 0;

    mi::scope::RingReader ring;
    if (!ring.Open(scopes[0].scopeId))
    {
        std::printf("Failed to open ring for scope %u\n", scopes[0].scopeId);
        return 1;
    }

    std::vector<float> buf(256);
    ring.ReadLatest(0, buf.data(), (uint32_t)buf.size());
    float minV = buf[0], maxV = buf[0], sumAbs = 0.0f;
    for (float v : buf)
    {
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
        sumAbs += std::fabs(v);
    }
    std::printf("channel 0 latest 256 frames: min=%.4f max=%.4f meanAbs=%.4f writeCursor=%llu\n",
        minV, maxV, sumAbs / buf.size(), (unsigned long long)ring.WriteCursor());
    return 0;
}
