// Standalone smoke-test writer: publishes a synthetic scope (sine + noise)
// so scope_monitor has something to display without needing a full Wwise
// session. Not part of the CMake build; compiled ad hoc from this directory:
//   cl /std:c++17 /EHsc src/test_writer.cpp /Fe:test_writer.exe
// Not shipped -- purely a development smoke test for the shared-memory path.
#include "../../mi_common/signal_scope_shm.h"

#include <cmath>
#include <cstdio>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <vector>

std::atomic<bool> g_running{ true };

int main(int argc, char** argv)
{
    uint32_t scopeId = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 5;
    const uint32_t sampleRate = 48000;
    const uint32_t numChannels = 2;
    const uint32_t blockFrames = 512;

    const char* channelNames[2] = { "FL", "FR" };

    mi::scope::ScopeWriter writer;
    if (!writer.Open(scopeId, "Test Sine (named buffer demo)", sampleRate, numChannels, channelNames))
    {
        std::fprintf(stderr, "Failed to open scope writer\n");
        return 1;
    }
    std::printf("Publishing scope %u at %u Hz, %u ch (FL/FR). Ctrl+C to stop.\n", scopeId, sampleRate, numChannels);

    std::vector<float> left(blockFrames), right(blockFrames);
    const float* chans[2] = { left.data(), right.data() };
    double phase = 0.0;
    const double freq = 440.0;
    uint64_t frame = 0;

    while (g_running)
    {
        for (uint32_t i = 0; i < blockFrames; ++i)
        {
            const double t = double(frame + i) / sampleRate;
            const float s = 0.5f * float(std::sin(2.0 * 3.14159265358979323846 * freq * t))
                          + 0.1f * float(std::sin(2.0 * 3.14159265358979323846 * freq * 3.0 * t));
            left[i] = s;
            right[i] = s * 0.8f;
        }
        writer.Write(chans, blockFrames);
        frame += blockFrames;
        std::this_thread::sleep_for(std::chrono::milliseconds(blockFrames * 1000 / sampleRate));
    }
    return 0;
}
