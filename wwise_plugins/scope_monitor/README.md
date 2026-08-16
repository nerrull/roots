# scope_monitor

Standalone imgui app that reads the shared-memory buffers written by the
`SignalScope` Wwise effect plug-in (`../SignalScope`) and displays a
waveform, spectrogram and spectral envelope for whichever capture instance
is selected. Read-only: it never writes to the shared memory and doesn't
need Wwise installed, only a `SignalScope`-carrying process (Wwise Authoring
in Play/Preview, or a game linking the sound-engine plug-in) running
somewhere that shares this machine's OS shared-memory namespace.

See `../mi_common/signal_scope_shm.h` for the shared layout both sides
speak, and `../mi_common/shm_region.h` for the underlying cross-platform
named-shared-memory wrapper (Win32 `CreateFileMapping`/POSIX `shm_open`).

## Building

GLFW is fetched via CMake `FetchContent` (source build, no system package
manager needed); everything else is either vendored (`../../imgui`,
`../mi_common`) or the platform's own OpenGL.

```
git submodule update --init imgui   # from the repo root, if not already done
cmake -S wwise_plugins/scope_monitor -B wwise_plugins/scope_monitor/build -A x64
cmake --build wwise_plugins/scope_monitor/build --config Release
wwise_plugins/scope_monitor/build/Release/scope_monitor.exe
```

## Using it

1. Insert the **Signal Scope** effect (built from `../SignalScope`) anywhere
   in a bus or Actor-Mixer effect chain in Wwise. It passes audio through
   unchanged -- it's a tap, not a processor.
2. Give it a **Scope ID** (0-63). Two inserts sharing an ID overwrite the
   same shared-memory slot, so give each point you want to watch
   simultaneously a distinct ID -- e.g. one instance before and one after
   another effect, to compare input vs. output.
3. Enter Play/Preview (or run the game) so the sound engine actually
   executes the plug-in -- nothing is captured while it's not processing
   audio.
4. Launch `scope_monitor`. Every active Scope ID shows up in the **Scope**
   dropdown, labeled with its channel count and sample rate; pick one to
   view its waveform, spectrogram and spectral envelope live. Switching
   plug-ins on/off, changing Scope ID, or restarting Wwise Authoring is
   picked up automatically (the app polls the shared directory every
   frame and rebinds when a scope's generation counter changes).

## Layout

```
src/main.cpp        app: GLFW+OpenGL3 window, per-scope view state, drawing
src/fft.h            small vendored radix-2 FFT (no external DSP library)
src/test_writer.cpp  dev-only smoke test: publishes a synthetic sine-wave
                      scope without needing Wwise running. Not part of the
                      CMake build; compile ad hoc if you need it, from this
                      directory: cl /std:c++17 /EHsc src/test_writer.cpp
```
