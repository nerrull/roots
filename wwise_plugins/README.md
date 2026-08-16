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
| Signal Scope | Effect | (none) | Any | Pure audio tap -- publishes whatever passes through it to shared memory for the external `scope_monitor` app; doesn't touch the signal |
| Onset Tap | Effect | (none) | Any | Pure analysis tap -- detects transients against a self-adjusting threshold and publishes the events to shared memory for a program outside Wwise; doesn't touch the signal |

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
  onset_detector.h    transient detection with a self-adjusting threshold
  onset_shm.h         the event stream Onset Tap publishes through
  signal_scope_shm.h  the audio ring Signal Scope publishes through
  shm_region.h        named shared memory, Win32 and POSIX
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

for P in ModalResonator MacroOscillator GranularTexture ModalVoice DrumSynth SignalScope OnsetTap; do
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

### Ad hoc diagnostic harnesses (Windows/MSVC)

`tests/resampler_transparency_test.cpp`, `tests/resampler_freq_response_test.cpp`,
`tests/modalvoice_trigger_test.cpp` and `tests/rings_agc_test.cpp` are
standalone, MSVC-buildable harnesses used to investigate specific bug reports
(whether the resampler or block adapter introduce audible artifacts; whether
Modal Voice/Modal Resonator respond usefully to quiet real-world program
material rather than the hot signal levels used elsewhere in testing). They
don't need Wwise headers, only this directory and an
`eurorack` checkout on the include path:

```
cl.exe /std:c++17 /O2 /EHsc /D_USE_MATH_DEFINES /DTEST ^
  /I"<wwise_plugins>" /I"<eurorack>" ^
  tests\resampler_freq_response_test.cpp
```

`resampler_freq_response_test` needs only `mi_resampler.h`; the other two link
against the relevant MI `.cc` files directly (see the comments at the top of
each for exactly which ones). Findings from these are recorded in commit
messages and PR discussion rather than kept as a permanent regression suite.

## Demo audio

```sh
tests/run_demo.sh
```

Writes four WAVs: a Drum Synth pattern, then that pattern run through each of
the three effects (Rings through all six resonator models, Clouds through all
four playback modes, Elements struck/bowed/blown).

## Signal Scope: audio capture + external monitor

`SignalScope` is a seventh plug-in, structurally separate from the five MI
cores: an authoring+sound-engine **Effect** that never alters the audio
passing through it. Instead, on every `Execute()` it copies the block into a
named OS shared-memory ring buffer (`mi_common/shm_region.h` -- Win32
`CreateFileMapping`/`MapViewOfFile`, POSIX `shm_open`/`mmap`), keyed by a
**Scope ID** property (0-63) set on the instance. `scope_monitor/` is a
separate, standalone imgui app (GLFW + OpenGL3, not part of this repo's
Wwise plug-in build) that reads those buffers read-only and renders a
waveform, scrolling spectrogram and log-frequency spectral envelope for
whichever Scope ID is selected -- multiple `scope_monitor` instances, or one
instance switching between IDs, can watch different insert points
(pre/post another effect, different busses) at once. See
`scope_monitor/README.md` for build and usage instructions, and
`mi_common/signal_scope_shm.h` for the shared layout both sides speak.

This only works when the plug-in and the monitor app run on the same
machine (same OS shared-memory namespace) -- e.g. Wwise Authoring's local
Play/Preview alongside `scope_monitor` on this machine, not a remote
CrossOver/Wine Authoring session watched from elsewhere.

## Onset Tap: transients out of Wwise, into another program

`OnsetTap` is an authoring+sound-engine **Effect** that, like Signal Scope,
never alters the audio passing through it. What it publishes is not audio but
*events*: on every `Execute()` it runs a transient detector over the block and,
when one fires, writes "a hit landed, this hard, panned here" into a named
shared-memory ring keyed by a **Tap ID** property (0-15). It also republishes
the live level and the threshold that level would have to clear, every block,
so a receiving program can show why nothing is firing.

Built for driving visuals from sound -- in this repo, `mirror_app`'s raindrops
(`mirror_app/src/audio_pulse.h` reads the stream; see that app's README).

**Why the detection is in the plug-in.** A program on the other side of a
SignalScope-style audio ring could run its own detector, but it would be doing
it on samples it is always slightly behind on, on a thread that also has a
frame to draw, at whatever rate it happens to poll. Onsets are a millisecond-
scale phenomenon; the audio thread is where they are visible. Only the
conclusion needs to cross the process boundary, and a conclusion is 48 bytes.

**The threshold adapts, and that is the whole point.** A fixed level cannot
serve one show that carries a rain bed at -45 dBFS and a struck resonator
peaking near 0: a threshold that catches the rain fires continuously on the
resonator, and one tuned to the resonator never hears the rain. What makes a
hit a hit is not its level but how much louder it is than the moment before it,
so the detection function is the positive rise of the level in dB per ~1.3 ms
hop, and the bar is that function's own recent statistics --
`mean + sensitivity x stddev` over the adaptation window. Everything is in dB,
so turning a bus down does not change what it detects. Three guards keep it
honest, each against a specific way an adaptive threshold embarrasses itself:
**Level Floor** (an absolute gate, or room tone's statistically-exceptional
rises fire constantly), **Minimum Rise** (a floor under the adaptive threshold,
or a very steady source adapts down onto its own noise), and **Minimum
Interval** (a refractory period, or one hit is reported once per analysis hop
for as long as its attack lasts).

The detector is mono -- a transient is an event in time, and running it per
channel would report one stereo hit twice a millisecond apart. The channels are
downmixed for detection and their balance measured separately, travelling with
the event as `pan` so a receiving program can place it where it was heard.

Properties: **Tap ID**, **Name**, **Detection Enabled**, **Sensitivity**
(~1.5 twitchy, 2.5 musical, 4+ only the big hits), **Level Floor (dB)**,
**Minimum Rise (dB)**, **Minimum Interval (ms)**, **Adaptation Window (ms)**.

`mi_common/onset_detector.h` is the detector (header-only, no Wwise, no
allocation after `Init()`), `mi_common/onset_shm.h` the layout both sides
speak, and `tests/onset_detector_test.cpp` drives both offline -- synthetic
material with known transients, plus the writer and reader against each other.
Pass it a WAV to print onset counts at three sensitivities over real material,
which is how to pick a Sensitivity without guessing:

```sh
tests/run_tests.sh /tmp/out path/to/ambience.wav
```

Same-machine only, for the same reason Signal Scope is (see above).

## Things worth knowing

**`GetBankParameters()` in the authoring plug-in must actually write every
property, in the same order `SetParamsBlock()` reads them.** `wp.py new`
scaffolds it as a single placeholder float write. RacineComb's was filled in
correctly from the start (see the comment in `RacineCombPlugin.cpp`); the five
MI-based plug-ins' authoring `GetBankParameters()` were still the unmodified
placeholder until this was caught, meaning every SoundBank built from them
carried garbage parameter data. This is invisible to any offline test that
calls into the DSP core directly (as all the test harnesses in this repo do,
by design, to run without a Wwise install) -- it only shows up in the actual
sound engine running against a real bank, whether that's "Play" inside Wwise
Authoring or a shipped game build, because only that path exercises bank
(de)serialization at all. If a plug-in sounds fine when you nudge its sliders
live in the property editor but wrong/silent/glitchy whenever a sound actually
plays back through it, this mismatch is the first thing to check -- read
`SetParamsBlock()` in `SoundEnginePlugin/*FXParams.cpp` (or `*SourceParams.cpp`)
for the true order, and confirm `GetBankParameters()` in `WwisePlugin/*Plugin.cpp`
writes every one of those properties, by the same names as the XML, in the
same order, with the matching `Write<Type>`/`Get<Type>` pair for each.

**Rings and Elements were tuned for Eurorack line level, not typical game
audio level, and their exciter inputs have no gain stage of their own.**
Rings' onset detector (`rings/dsp/onset_detector.h`) requires the energy
derivative to clear a fixed *absolute* floor (0.01) in addition to its
z-score test, and Elements' external exciter (`blow_in`/`strike_in` in
`elements/dsp/part.cc`) is fed straight into the resonator with no
normalization. Both assume a consistently "hot" signal, which is what a
Eurorack module actually receives but is not what a typical game audio bus
looks like -- an ambience bed or dialogue track commonly sits tens of dB
quieter and varies constantly. Measured on a 45s quiet rain-ambience file
(input RMS ~0.005) with `tests/rings_agc_test.cpp` and
`tests/modalvoice_trigger_test.cpp <wav>`: Modal Resonator's output RMS was
8x too quiet and Modal Voice's was 18x too quiet before the fix, which reads
as intermittent/weak resonance ("glitchy/scratchy") or an effectively silent
wet signal, even though the exact same effect sounds fine on a hot test
signal near 0 dBFS. Both plug-ins now run the exciter feed through
`mi::ExciterAGC` (`mi_common/mi_exciter_agc.h`), a fast-attack/slow-release
peak follower that brings quiet input up to the level these cores expect,
with a hard output clamp since the follower can't fully catch a sharp
transient (a rain droplet, a click) before it reaches the resonator's own
gain. If a build ever needs to second-guess whether this is still doing its
job, re-run those two harnesses against real program material, not a
synthetic full-scale test tone -- that's exactly the gap that hid this bug
originally.

**None of the MI cores guard against denormals, and Wwise doesn't enable
flush-to-zero for you.** A resonator or filter decaying toward silence over
several seconds (Rings' tail runs up to 12s at high damping) spends a lot of
that time with sample values in denormal range, where floating-point ops are
commonly 10-100x slower without FTZ/DAZ enabled -- this reads as real-time-only
audio glitching/dropouts that never shows up in an offline render, since only
the live sound engine has a hard deadline to miss. `RacineCombDSP.cpp` already
guarded its own feedback loop with an additive constant; the five MI-based
plug-ins now call `mi::EnableFlushToZero()` (see `mi_common/mi_denormal_guard.h`)
once per `Execute()`/`Process()` instead, since none of MI's own filter code
carries any denormal protection.

**Clouds' granular mode needs its buffer to fill before it sounds "full."**
`GranularProcessor` records incoming audio into a rolling buffer (roughly a
second's worth, at these buffer sizes) and reads grains from a `Position`
within that recorded history. Right after the effect starts, most of that
history is still silence, so granular texture builds up over about a second
rather than being immediate -- this is inherent to the algorithm, not
wrapper latency (the resampler + block adapter's actual structural latency is
under 2 ms). `Position` near 0 (most recent audio) shortens the perceived
delay; `Position` near the default 0.5 (mid-buffer) waits longer for real
content to reach that point in the loop.

**Peaks doesn't have separate Snare and Hi-Hat models.** Despite `DrumSynth`'s
`Model` property listing them as distinct choices, `peaks::Processors::Configure()`
(`peaks/processors.h`) shares one algorithm between them and silently forces
Snare unless Tone (`Param2`) **and** Snappy (`Param3`) are both `>= ~99.2%`
(`65000/65535`) -- below that band it reverts to Snare regardless of what
`set_function()` was called with, no matter what `Model` says. Hi-Hat presets
need `Param2`/`Param3` pinned near 1.0.

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
build now exists for all six MI-based plug-ins plus Signal Scope.

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

for %P in (RacineComb ModalResonator GranularTexture MacroOscillator ModalVoice DrumSynth SignalScope OnsetTap) do (
  cd %P
  python %WWISEROOT%\Scripts\Build\Plugins\wp.py premake Authoring
  python %WWISEROOT%\Scripts\Build\Plugins\wp.py build Authoring -t vc170 -c Release
  cd ..
)
```

Signal Scope and Onset Tap have no `eurorack`/MI dependency (they don't need
`MI_EURORACK_DIR`) and also build for the plain Windows sound-engine target, same as
the other plug-ins: `wp.py build Windows_vc170 -c Release` from inside the plug-in's
directory. Onset Tap's Windows authoring DLL has **not** been built yet -- only the
macOS sound-engine side (`libOnsetTap.dylib` / `libOnsetTapFX.a`) has, so it does not
appear in `dist/` alongside the others.

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
(Release config) for all seven plug-ins, checked into the repo. Install them
into:

```
/Library/Application Support/Audiokinetic/Wwise <Version>/Authoring/x64/Release/bin/Plugins
```

1. Copy all fourteen files from `wwise_plugins/dist/` (seven `.dll` + seven
   `.xml`) into that `Plugins/` folder, substituting your installed
   `<Version>` (e.g. `2025.1.9`).
2. Quit and relaunch Wwise Authoring. Racine Comb, Modal Resonator, Granular
   Texture, Macro Oscillator, Modal Voice, Drum Synth and Signal Scope should
   now show up in the Effects/Sources insert lists.

**Version caveat:** these DLLs were built against the **2025.1.10** SDK, one
point release ahead of the **2025.1.9** Authoring app this repo's macOS
prerequisites section names. `AK::Wwise::Plugin` (the API these plug-ins use)
has been stable since Wwise 2022.1, so 2025.1.10 plug-ins are likely to load
fine in a 2025.1.9 Authoring host, but this hasn't been verified against an
actual 2025.1.9 install. If a plug-in fails to appear or Wwise logs a load
error, that version gap is the first thing to check — rebuild against the
2025.1.9 SDK's `Win32_vc170`/`x64_vc170` headers and libs instead (or update
Wwise Authoring to 2025.1.10) to rule it out.
