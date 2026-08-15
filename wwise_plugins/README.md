# Mutable Instruments Wwise plug-ins

Wwise plug-ins wrapping five Mutable Instruments DSP cores. The sound-engine
(runtime) side was built for Apple Silicon macOS against Wwise 2025.1.9; the
authoring side (the Windows DLL that makes each plug-in appear in the Wwise UI)
was built separately against Wwise 2025.1.10 on Windows — see
[Authoring plug-in (Windows)](#authoring-plug-in-windows).

| Plug-in | Type | MI module | Rate | Notes |
|---|---|---|---|---|
| Modal Resonator | Effect | Rings | 48 kHz native | Bus audio excites the resonator; onsets strum it |
| Granular Texture | Effect | Clouds | 32 kHz, resampled | All four playback modes (needs one upstream fix, below) |
| Modal Voice | Effect | Elements | 32 kHz, resampled | Bus audio feeds the exciter inputs |
| Macro Oscillator | Source | Plaits | 48 kHz native | All 24 synthesis engines |
| Drum Synth | Source | Peaks | 48 kHz native | Bass drum, snare, hi-hat, FM drum |

The MI code is MIT licensed (Copyright Emilie Gillet) and is compiled
**unmodified** from a checkout of `pichenettes/eurorack` for the macOS build;
everything specific to Wwise lives in this directory. The Windows authoring
build needs a handful of MSVC-portability patches applied to the `eurorack`
checkout itself — see
[Patches required in the `eurorack` checkout](#patches-required-in-the-eurorack-checkout).
Note that the MIT grant covers the code, not the Mutable Instruments trademark
or the module names, which is why the plug-ins are named for what they do.

## Layout

```
mi_common/          shared adaptation layer
  mi_block_adapter.h  Wwise's variable buffers -> MI's fixed block sizes
  mi_resampler.h      polyphase 48k <-> 32k rational resampler
  mi_arena.h          IAkPluginMemAlloc-backed arena for stmlib::BufferAllocator
  patched/            headers that shadow the checkout (one file, one line)
<PluginName>/       one directory per plug-in, from wp.py's scaffolding
tests/              offline DSP harnesses (no Wwise required)
```

## Prerequisites

- Wwise 2025.1.9 at `/Applications/Audiokinetic/Wwise_2025.1.9.9197`
- A `pichenettes/eurorack` checkout. By default it is expected next to the
  project, at `../../eurorack` relative to this directory. Override with
  `MI_EURORACK_DIR`.
- Xcode. Wwise's toolchain scripts look for `/Applications/Xcode26.app`; if your
  Xcode is elsewhere, point `AK_XCODE_DEVELOPER_DIR_2600` at it (see below).

## Building

```sh
export WWISEROOT=/Applications/Audiokinetic/Wwise_2025.1.9.9197
export WWISESDK=$WWISEROOT/SDK
export AK_XCODE_DEVELOPER_DIR_2600=/Applications/Xcode.app/Contents/Developer
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

for P in ModalResonator MacroOscillator GranularTexture ModalVoice DrumSynth; do
  (cd $P \
    && python3 $WWISEROOT/Scripts/Build/Plugins/wp.py premake Mac \
    && xcodebuild -workspace ${P}_Mac.xcworkspace -scheme All \
         -configuration Debug -destination 'generic/platform=macOS' \
         ARCHS=arm64 VALID_ARCHS=arm64)
done
```

Artifacts land in `$WWISESDK/Mac_Xcode2600/<Config>/`: `bin/lib<Name>.dylib` for
dynamic loading and `lib/lib<Name>FX.a` to link statically into the game binary.

`wp.py build` swallows xcodebuild's output, which makes failures look like
successes; calling `xcodebuild` directly as above is worth it.

## Testing

```sh
tests/run_tests.sh
```

Each harness drives the MI core through the same wrappers the plug-in uses and
writes a WAV you can listen to. They check for silence, divergence, NaNs and
(for the resampled plug-ins) FIFO underruns, and the resampler has its own test
covering rate ratios, round-trip gain and alias rejection.

Note that these measure **sustained RMS over the second half of the render**,
not peak. Peak alone is not a useful check here: a mode that emits one burst
and then dies still shows a healthy peak, which is exactly how the Clouds
stretch problem below went unnoticed at first.

## Demo audio

```sh
tests/run_demo.sh
```

Writes four WAVs: a Drum Synth pattern, then that pattern run through each of
the three effects (Rings through all six resonator models, Clouds through all
four playback modes, Elements struck/bowed/blown).

## Things worth knowing

**`TEST` is required, not optional.** It selects stmlib's portable code paths
instead of Cortex-M inline assembly, and makes `IN_RAM` a no-op rather than an
STM32-only `.ramtext` section attribute that will not link on macOS.

**The MI cores assume BSS zero-initialization.** On the modules these objects
are globals; as plug-in members they are heap memory with garbage in it. Clouds
in particular will hang: its WSOLA correlator searches on uninitialized window
bounds and grinds through billions of candidates. The Clouds and Elements
plug-ins `memset` their processor before `Init()`.

**Clouds' `Prepare()` must run often.** The firmware calls it continuously from
its main loop while `Process()` runs in the audio interrupt. It does buffer
housekeeping and advances the WSOLA correlator search, so the plug-in calls it
once per 32-frame block, not once per Wwise buffer.

**Plaits' trigger is a gate, not a pulse.** The percussive engines respond to
its rising edge, but the three six-operator FM engines read its level as a
note-on/note-off gate. A one-block pulse leaves those three silent.

**`AkClamp` is a macro with no outer parentheses.** `AkClamp(x, a, b) * 0.5f`
scales only the last branch of its ternary. Every use here is parenthesized.

**48 kHz only for Clouds and Elements.** Their 2/3 polyphase conversion is exact
only from 48 kHz; at other host rates those two plug-ins pass audio through dry
rather than silently detune. Rings and Plaits run at any rate with a pitch
correction applied.

## Upstream fix: Clouds stretch mode

`PLAYBACK_MODE_STRETCH` (the WSOLA time-stretch) does not work with the sources
as published. It emits about a second of audio and then goes permanently
silent. This is not a porting artifact -- it fails identically when
`GranularProcessor` is driven directly at 32 kHz with no resampler, block
adapter or FIFO.

The cause is in `clouds/dsp/window.h`. `Window::done_` is assigned in exactly
two places: `Init()` sets it true, and `OverlapAdd()` derives it from the
envelope phase. But `OverlapAdd()` early-returns while `done_` is true, so it
can never clear it, and `Start()` -- which is what "begin a new window" means --
does not touch it at all. So it is a one-way latch: `WSOLASamplePlayer::Init()`
calls `Init()` on both windows, and from then on `OverlapAdd()` returns
immediately forever.

Adding `done_ = false;` to `Start()` fixes it. Sustained RMS goes from 0.00000
to ~0.096 with no other change, and `correlator_loaded_` begins latching as it
should.

Rather than edit the eurorack checkout, `mi_common/patched/clouds/dsp/window.h`
is a copy carrying that single line, and the patched directory is placed ahead
of the checkout on the include path so it shadows the original. The checkout
stays pristine and can be updated freely. If upstream ever fixes this, delete
the patched copy and the extra include directory.

This is worth reporting upstream.

## Gotcha: Clouds' density control is bipolar

Density has a dead zone at exactly 0.5, where grains fire only from the trigger
input -- which nothing is patched to here, so the granular mode goes quiet.
Sustained RMS at density 0.0 or 1.0 is around 0.49; at 0.5 it is zero. Sweep the
control off centre, and do not use 0.5 as a "neutral" default.

## Authoring plug-in (Windows)

Wwise Authoring on macOS is the Windows application running under
CrossOver/Wine, so the authoring component — the DLL that makes a plug-in
appear in the Wwise UI with a property page — has to be built on Windows. That
build now exists for all six plug-ins.

### Prerequisites

- Wwise SDK for Windows, vc170 (VS2022) toolset — e.g.
  `Wwise_2025.1.10.9233\SDK`. Set `WWISEROOT` / `WWISESDK` accordingly.
- Visual Studio 2022 (or Build Tools 2022) with the **Desktop development with
  C++** workload, plus the **C++ MFC for latest v143 build tools** component —
  the GUI plug-in derives from `PluginMFCWindows<>`, and
  `AK/Wwise/Plugin/PluginMFCWindows.h` compiles to nothing without MFC on the
  include path.
- The `eurorack` checkout for the five MI-based plug-ins (see Prerequisites
  above), including its `stmlib` submodule: `git submodule update --init stmlib`.

### Building

```
set WWISEROOT=Q:\Development\Audiokinetic\Wwise_2025.1.10.9233
set WWISESDK=%WWISEROOT%\SDK
set MI_EURORACK_DIR=Q:\Development\git\eurorack

for %P in (RacineComb ModalResonator GranularTexture MacroOscillator ModalVoice DrumSynth) do (
  cd %P
  python %WWISEROOT%\Scripts\Build\Plugins\wp.py premake Authoring
  python %WWISEROOT%\Scripts\Build\Plugins\wp.py build Authoring -t vc170 -c Release
  cd ..
)
```

`build Authoring` compiles both the sound-engine static lib and the authoring
DLL and drops the DLL straight into
`<Wwise install>\Authoring\x64\<Config>\bin\Plugins\`, where Wwise Authoring
loads plug-ins from — no separate install step needed for local testing.
`package.py Authoring` (see the SDK docs) produces a distributable bundle.

### Patches required in the `eurorack` checkout

The MI DSP code was only ever compiled with Clang/GCC on macOS before. Getting
it through MSVC needs a few small, non-functional patches to the checkout
itself (not to anything in this repo) — apply these to whatever `eurorack`
checkout `MI_EURORACK_DIR` points at:

- **`NOMINMAX` / `_USE_MATH_DEFINES`** — already added to each plug-in's
  `PremakePlugin.lua` defines. Without `NOMINMAX`, `windows.h`'s `min`/`max`
  macros corrupt every `std::min`/`std::max` call in the MI sources (surfaces
  as bizarre `error C2589: illegal token on right side of '::'`). Without
  `_USE_MATH_DEFINES`, MSVC's `<cmath>` does not define `M_PI`.
- **`stmlib/stmlib.h`** — add `#define __attribute__(x)` under
  `#if defined(_MSC_VER)`. The GCC/Clang-only `__attribute__((always_inline))`
  syntax in `stmlib/utils/dsp.h` doesn't compile under MSVC; it's a hint, so
  dropping it is safe.
- **`{rings,elements,plaits}/resources.cc`** — three empty-initializer arrays
  (`const T* table[] = {};`) are a GCC/Clang extension MSVC rejects
  (`error C2466`). Guard with `#if defined(_MSC_VER)` and give them one
  `nullptr` element; the arrays are unused by these modules either way.
- **`plaits/dsp/fm/algorithms.h`** — drop the four explicit-specialization
  forward declarations for `Algorithms<4>`/`Algorithms<6>`'s `opcodes_` and
  `renderers_`. MSVC parses incomplete-array static-member specializations
  differently from GCC/Clang and errors either way (with or without `extern`);
  the definitions in `algorithms.cc` are sufficient on their own.

None of these change behavior — they're compiler-portability fixes only —
but they do mean the checkout at `MI_EURORACK_DIR` is no longer byte-identical
to upstream `pichenettes/eurorack`.

### Installing on your Mac

`wwise_plugins/dist/` has the built `<Plugin>.dll` + `<Plugin>.xml` pairs
(Release config) for all six plug-ins, checked into the repo. Install them
into:

```
/Library/Application Support/Audiokinetic/Wwise <Version>/Authoring/x64/Release/bin/Plugins
```

1. Copy all twelve files from `wwise_plugins/dist/` (six `.dll` + six `.xml`)
   into that `Plugins/` folder, substituting your installed `<Version>` (e.g.
   `2025.1.9`).
2. Quit and relaunch Wwise Authoring. Racine Comb, Modal Resonator, Granular
   Texture, Macro Oscillator, Modal Voice and Drum Synth should now show up in
   the Effects/Sources insert lists.

**Version caveat:** these DLLs were built against the **2025.1.10** SDK, one
point release ahead of the **2025.1.9** Authoring app this repo's macOS
prerequisites section names. `AK::Wwise::Plugin` (the API these plug-ins use)
has been stable since Wwise 2022.1, so 2025.1.10 plug-ins are likely to load
fine in a 2025.1.9 Authoring host, but this hasn't been verified against an
actual 2025.1.9 install. If a plug-in fails to appear or Wwise logs a load
error, that version gap is the first thing to check — rebuild against the
2025.1.9 SDK's `Win32_vc170`/`x64_vc170` headers and libs instead (or update
Wwise Authoring to 2025.1.10) to rule it out.
