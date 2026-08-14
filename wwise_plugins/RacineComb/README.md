# Racine Comb

A Wwise effect plug-in: a feedback comb filter whose spectral peaks can be swept
in realtime from an RTPC.

## What it does

The structure is the classic feedback comb, `H(z) = z^-D / (1 - g·z^-D)`, with a
one-pole lowpass in the loop for damping. Peaks land on integer multiples of
`fs/D`, so the **Frequency** parameter (which sets `D = fs/f0`) moves the whole
harmonic series at once.

`D` is fractional, interpolated with a 4-point cubic Hermite (Catmull-Rom)
kernel, and slewed per sample rather than per buffer. Sweeping it resamples
whatever is circulating in the feedback loop, so moving the peaks pitch-glides
the resonating content — flanger/tape-warp behaviour. That is intended. If you
ever need the peaks to relocate *without* the glide, a comb cannot do it; that
calls for a bank of retunable resonators instead.

## Parameters

| Property | Range | Default | Notes |
|---|---|---|---|
| Frequency | 20 – 16000 Hz | 220 | Fundamental of the peak series. Exclusive RTPC. |
| Glide | 0 – 2000 ms | 20 | How fast the peaks travel. Also the anti-zipper slew, so the DSP floors it at 0.5 ms. |
| Feedback | −98 – 98 % | 80 | Resonance. Negative shifts the response to odd harmonics only. |
| Damping | 0 – 100 % | 20 | Loop lowpass, 20 kHz down to 200 Hz. Damps the upper peaks faster. |
| Wet/Dry Mix | 0 – 100 % | 100 | |
| Output Level | −24 – 24 dB | 0 | |

All six are RTPC-able. Parameters arrive at buffer boundaries and are ramped
across the buffer internally.

## Design notes

**Delay memory** is sized for the bottom of the frequency range at the running
sample rate (20 Hz ⇒ 2400 frames at 48 kHz, rounded up to 4096 so indices can be
masked), because Frequency is RTPC-driven and can reach it at any time.

**Tuning depends on the whole loop, not just the delay line.** Two things had to
be handled or the comb tunes flat:

- The damped sample must be written back within the same frame. Feeding the
  filter's *previous* output back adds a full sample to the loop — 2.8% flat at
  1.2 kHz, 6% at 3.3 kHz.
- The one-pole itself contributes phase delay, so its share is subtracted from
  the delay line, evaluated at the fundamental.

With damping at 0 the tuning is then exact. With damping engaged a small
downward residue remains (−1.25% at 3.3 kHz / 50% damping): the loop gain falls
with frequency, which drags the response maximum below the point where the loop
phase closes. That is a property of a damped comb, not a defect, so the DSP does
not try to correct it.

**Stability** — feedback is clamped to ±0.98, bounding the on-resonance gain at
1/(1−g) = 50. The loop lowpass has unity DC gain so it does not move that bound.

## Layout

- `SoundEnginePlugin/RacineCombDSP.{h,cpp}` — the filter. No Wwise headers, so it
  builds and runs standalone.
- `SoundEnginePlugin/RacineCombFX.{h,cpp}` — Wwise effect: allocation, tail
  handling, unit conversion, per-buffer parameter ramps.
- `SoundEnginePlugin/RacineCombFXParams.{h,cpp}` — parameter block and RTPC handling.
- `WwisePlugin/` — authoring-side plug-in and the property XML.
- `tests/comb_response_test.cpp` — standalone validation.

Parameter order appears in three places that must stay in sync: the IDs in
`RacineCombFXParams.h`, the reads in `SetParamsBlock`, and the writes in
`RacineCombPlugin::GetBankParameters`.

## Tests

```sh
cd tests
c++ -std=c++17 -O2 -I../SoundEnginePlugin comb_response_test.cpp \
    ../SoundEnginePlugin/RacineCombDSP.cpp -o comb_response_test
./comb_response_test
```

Measures the steady-state response of the actual DSP: that peaks land on the
requested frequency and its harmonics (including non-integer delay lengths, which
only work because of the fractional interpolation), that the notches between them
are rejected, that a 100 Hz → 4 kHz → 100 Hz sweep stays continuous and finite,
that the feedback clamp holds over 20 s of on-resonance excitation, and that
Wet/Dry at 0 is bit-transparent.

## Building

Sound engine plug-in, macOS:

```sh
export WWISEROOT=/Applications/Audiokinetic/Wwise_2025.1.9.9197
python3 $WWISEROOT/Scripts/Build/Plugins/premake.py Mac
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -workspace RacineComb_Mac.xcworkspace -scheme All \
             -configuration Profile -destination 'generic/platform=macOS'
```

Note the SDK's `build.py` wrapper hardcodes the toolchain path
`/Applications/Xcode26.app/Contents/Developer` (from
`Scripts/ToolchainSetup/Mac/ToolchainVers.txt`). With Xcode installed at the
normal `/Applications/Xcode.app`, invoke `xcodebuild` directly as above, or
symlink the expected path.

Outputs land in the Wwise install, not in this directory:

- `SDK/Mac_Xcode2600/Profile/lib/libRacineCombFX.a` — link this into the game.
- `SDK/Mac_Xcode2600/Profile/bin/libRacineComb.dylib`

### The authoring plug-in needs a Windows build

Wwise Authoring on macOS is the Windows build running under CrossOver — see
`Wwise.app/Contents/SharedSupport/Wwise/share/{crossover,wine}`. Accordingly the
SDK ships only `Scripts/Build/Plugins/common/platform/authoring_windows.py`;
there is no macOS authoring target, and `premake.py Mac` runs with
`--authoring=no`.

So the `WwisePlugin/` half — the DLL that makes the effect appear in the Wwise
UI with the property page above — has to be built on Windows:

```
python %WWISEROOT%\Scripts\Build\Plugins\premake.py Authoring
python %WWISEROOT%\Scripts\Build\Plugins\build.py Authoring
python %WWISEROOT%\Scripts\Build\Plugins\package.py Authoring
```

Until that exists you cannot insert the effect from the authoring tool or create
a ShareSet for it, so the sound engine half — though built and validated here —
has nothing in the project to attach to yet.

The upside of the CrossOver arrangement: that one Windows DLL is what the Mac
authoring app loads too, so there is no second authoring binary to maintain.

Cross-compiling it from macOS is not supported — `authoring.py` picks the
authoring target from `platform.system()` at build time, so the build has to run
on Windows. (The API itself is header-only, with no import library, but the
scaffolded `WwisePlugin/Win32/` GUI class derives from `PluginMFCWindows<>`, and
MFC is MSVC-only.) A Linux authoring build would only produce a `.so` that
neither machine can load.
