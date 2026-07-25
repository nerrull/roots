# kinect_v2_validate

Tools to validate Kinect v2 (Xbox One sensor) streams on macOS via
[libfreenect2](https://github.com/OpenKinect/libfreenect2):

- **`kinect_v2_validate`** — headless. Opens the sensor, captures for N seconds,
  and reports each stream's resolution, format, measured FPS, frame count, and
  dropped-frame count, plus a depth sanity check.
- **`kinect_v2_demo`** — ImGui viewer: RGB left, depth right, 4-channel mic-array
  scope underneath, with per-stream poll-rate control. See
  [Demo viewer](#demo-viewer).

Validated streams: **RGB (1920×1080), IR (512×424), Depth (512×424 float mm),
audio (4ch @ 16 kHz).**

## Audio

The Kinect v2's 4-mic array **does work on macOS** — just not through
libfreenect2, which has never supported audio. macOS itself exposes the sensor
as a normal CoreAudio input device: `Xbox NUI Sensor`, **4 channels @ 16 kHz**.
Verified with all four channels carrying distinct live signal (peaks ≈ −34 dB,
so no duplicated or dead channels):

```sh
ffmpeg -f avfoundation -list_devices true -i ""   # find its index
ffmpeg -f avfoundation -i ":0" -t 5 nui4ch.wav    # 4.0, 16000 Hz, pcm_f32le
```

So no external mic array (ReSpeaker etc.) is needed. `kinect_v2_demo` below
captures the same device directly via CoreAudio and scopes all 4 channels.

**Mic permission:** these are plain CLI binaries, so macOS attributes microphone
access to the *parent terminal*. Grant your terminal (or IDE) mic access under
System Settings → Privacy & Security → Microphone; otherwise capture starts but
every channel reads digital silence.

### ⚠️ The USB reset and audio: open the sensor *first*

Opening the sensor with libfreenect2 momentarily destroys the audio device.
`Freenect2::openDevice()` calls `libusb_reset_device()`, and a USB reset detaches
every kernel driver bound to the device — including AppleUSBAudio. The
`Xbox NUI Sensor` CoreAudio device disappears; an already-running capture freezes
mid-stream and `AudioUnitRender` starts erroring. Measured, with audio opened
*before* the sensor:

| | audio frames over 5 s | `AudioUnitRender` errors |
| --- | --- | --- |
| Sensor opened with USB reset | frozen at 34816 | 503 |
| Sensor opened with reset skipped | 31744 → 118784 | 0 |

The fix is ordering, not avoidance. The audio interface **re-attaches ~0.1 s after
the re-enumeration**, so the sequence that works is:

1. Open the sensor (with the reset — this is what clears a wedged sensor).
2. Wait for the mic array to reappear in CoreAudio.
3. Start audio capture.

That gets you a freshly reset sensor *and* working audio. `kinect_v2_demo` does
exactly this: `AudioCapture::start()` takes a `wait_seconds` and polls until the
device shows up. Verified: `usedUsbReset=no` → audio climbing ~16 k frames/s on
all 4 channels with color+depth flowing and 0 errors.

`patches/0001-optional-usb-reset.patch` additionally makes the reset opt-out via
`FREENECT2_NO_RESET=1` (`setup.sh` applies it idempotently after cloning), which
backs `--no-usb-reset`. Useful when the reset itself is the thing failing. If the
requested policy can't open the device, `KinectSource::open()` tries the other one
once before giving up.

Note that only one process can hold the sensor: if `openDevice` fails with
`LIBUSB_ERROR_NO_DEVICE` on reset, check you don't already have the demo running.

## Build

```sh
./setup.sh
```

This clones + builds libfreenect2 into `external/libfreenect2/install` (no
Homebrew formula exists), then builds the validator into `build/`. Requires
Homebrew deps: `libusb`, `jpeg-turbo`, `glfw` (the script installs them).

## Run

```sh
./build/kinect_v2_validate -t 10     # capture for 10 seconds
```

- `devices found: 0` → plug the sensor into a genuine **USB3** port via its
  power/adapter brick, then unplug/replug and re-run.
- Exit code: `0` = PASS (all streams healthy), non-zero = FAIL.

`external/libfreenect2/build/bin/Protonect` is also built — the upstream GUI
viewer, handy for an eyeball check of the depth/RGB image.

### On the validator's `drops=` column

Expect large drop counts on IR/DEPTH (and 15 fps on all three) whenever the room
is dim. Both are artefacts of how this tool is wired, not transport faults:

- The colour camera halves its rate to **15 fps** under long auto-exposure.
- The validator uses one `SyncMultiFrameListener` over Color|Ir|Depth, which
  only releases a bundle once *every* stream has a frame. RGB at 15 fps therefore
  gates depth to 15 fps, and the depth frames the listener discards leave holes
  in `Frame::sequence` — counted here as drops.

libfreenect2's own log line (`N packets were lost`) is the honest number. Ungated,
depth runs at **29.8 fps with zero sequence gaps**; `kinect_v2_demo` uses
independent per-stream listeners and reports it correctly.

## Demo viewer

```sh
./build/kinect_v2_demo
```

RGB left, Turbo-colourmapped depth right, four mic-array scopes underneath.

| Flag | Effect |
| --- | --- |
| `--cpu-pipeline` | Decode depth on the CPU instead of OpenGL |
| `--no-audio` | Skip mic-array capture |
| `--list-audio` | Print all audio inputs with their UIDs, then exit |
| `--audio-uid <uid>` | Pin an exact device UID |
| `--audio-model <substr>` | Match a model-UID substring |
| `--audio-name <substr>` | Match a display-name substring |
| `--audio-allow-fallback` | Permit falling back to the system default input |
| `--no-usb-reset` | Skip the sensor USB reset on open (it is on by default) |

**Poll rate.** `video poll Hz` and `depth poll Hz` (1–30, independent, each with a
`pause` box, plus an `unthrottled` box) set how often the *UI pulls* the newest
frame. The sensor always runs at its native rate, so lowering these costs
freshness, not stability. Each panel header shows both numbers, e.g.
`sensor 29.9 Hz -> polled 10.0 Hz`.

**Audio device selection is deterministic.** By default it pins the sensor by the
USB VID:PID in its CoreAudio `ModelUID` (`045E:02C4`) and requires ≥4 input
channels — not by display name, which is neither unique (two identical USB
interfaces report the same name) nor stable across locales. The selection is then
verified by reading the device back off the audio unit. If the selector matches
nothing, or matches ambiguously, capture **fails with the device list** rather
than falling back to the system default input — a silent fallback is how you end
up scoping a virtual/remote-desktop mic that records pure silence. Opt in with
`--audio-allow-fallback` if you actually want that. The audio panel shows the
bound device name in green when it is the Kinect, amber otherwise, with its UID
underneath.

To pin one specific sensor when several are attached, use its UID (it encodes the
USB serial):

```sh
./build/kinect_v2_demo --list-audio
./build/kinect_v2_demo --audio-uid 'AppleUSBAudioEngine:Microsoft:Xbox NUI Sensor:059731334047:3'
```

## Audio DSP and beam steering

The 4 raw mics feed a beamformer, then a dynamics chain, producing one mono
"beam" output shown as its own scope above the raw channels:

```
mic 1..4 ──▶ delay-and-sum beam ──▶ pre-gain ──▶ highpass ──▶ compressor ──▶ limiter ──▶ beam out
                     ▲
              steering azimuth
                     ▲
        ┌────────────┴────────────┐
   depth: closest person     mic: SRP over the array
```

All of it runs inside the CoreAudio render callback: allocation-free, lock-free,
and parameterised through individual atomics so a UI edit can never block the
realtime thread. Measured beam latency is **0.36 ms** (the steering delay lines),
and live parameter churn produces **0 `AudioUnitRender` errors**.

### Making the audio clear

`in gain dB` → `HPF Hz` (sheds room rumble and handling noise) → compressor
(`thresh`/`ratio`/`attack`/`release`/`knee`/`makeup`) → limiter with a −1 dBFS
ceiling that nothing can pass. The compressor is a feed-forward peak design with
a soft knee, smoothing the *gain* rather than the detector so it does not pump on
transients. A gain-reduction meter sits above the DOA plot. Defaults are tuned for
a person a couple of metres from the sensor: +12 dB in, 110 Hz highpass, −34 dB
threshold at 4:1 with 8 dB makeup.

### Steering: two independent locators

| Mode | How it finds the talker |
| --- | --- |
| `manual` | Fixed angle from the slider |
| `depth (closest)` | Nearest depth surface with real support |
| `mic (loudest)` | SRP azimuth scan across the 4 mics |
| `depth, mic fallback` | Depth when tracked, mic estimate when not (default) |

Depth is preferred because it still works in a silent room; the mic estimate
covers the case where the person is outside the depth frustum.

**Depth locator.** "Closest point" taken literally is useless — one hot noise
pixel beats a real person. Instead the frame is scored on a coarse 16 px block
grid, each block ranked by a low percentile of its valid depths and required to
have enough valid pixels; the centroid of everything in a slab behind the nearest
qualifying surface becomes the target, unprojected to metres through the IR
intrinsics. Azimuth is one-pole smoothed and coasts for a second when the track
drops, so the beam follows a person instead of chasing centroid jitter. A
crosshair on the depth image shows the current centroid (green tracked, amber
holding).

**Mic locator.** Steered response power: delay-and-sum toward each candidate angle
over ±90° and keep the loudest, with parabolic interpolation across the peak for
sub-grid resolution. Time domain, so no FFT dependency. Input is bandpassed to
300–3000 Hz — speech range, and below the array's spatial-aliasing limit `c/2d`.
A confidence score (peak height above the mean response) gates the estimate so
silence cannot steer the beam. Runs at 15 Hz, far faster than a person moves.
The half-polar plot draws the whole response surface plus each locator's estimate.

### ⚠️ Calibrating the array aperture

**The Kinect v2's true mic positions are not published**, and libfreenect2 does not
model audio at all, so the geometry is a uniform linear array with an estimated
0.16 m aperture. Beam steering and DOA accuracy are only as good as that number.
To calibrate: put a talker at a known angle, set steering to `depth (closest)`,
and adjust `aperture m` until the `mic` readout agrees with the `depth` readout —
the panel shows both side by side for exactly this purpose. If the beam steers
*away* from the person, flip `mirror depth az`.

Both locators' sign conventions are verified against synthetic ground truth in the
tests, but the mapping from the depth camera's axis to the mic array's axis is a
physical property of the bar that only a real measurement can confirm.

**Full step-by-step procedure, including what to do if it will not converge:**
[`documentation/notes/kinect_mic_array_calibration.md`](../documentation/notes/kinect_mic_array_calibration.md)

### Tests

Headless, hardware-free, synthetic-signal tests for the parts that are otherwise
only checkable by ear:

```sh
ctest --test-dir build --output-on-failure
```

- `dsp` — compressor static curve against closed-form values, attack/release
  behaviour, limiter ceiling, biquad magnitude response, array geometry, beam
  on/off-axis rejection (measured 6.17 dB, theory 10·log₁₀4 = 6.02 dB), and DOA
  accuracy (within ~1.3° across ±60°, 8/8 noisy trials within 10°, and silence
  correctly yielding near-zero confidence).
- `person_tracker` — azimuth against analytically-derived angles, a near person
  beating a full-frame back wall, **depth speckle failing to hijack the track**,
  undersized-blob rejection, hold-then-release timing, and smoothing convergence.

**Stability notes.** The UI is GLFW + Metal, so it never shares an OpenGL context
with libfreenect2's depth pipeline. Colour and depth get independent "latest wins"
listeners (see `src/demo/kinect_source.h`), so neither can gate the other; frames
superseded before the UI polls them count as `skipped`, and only `seq gaps`
indicate real loss. Each stream uploads into a 3-deep Metal texture ring, so an
upload never lands on a texture the GPU is still sampling. Pixels are copied out
under lock rather than handing the render thread a `libfreenect2::Frame*`. The
audio ring is lock-free, so the CoreAudio realtime callback never takes a mutex.
The header tracks worst frame time and a count of frames over 33 ms, with a reset
button, for soak testing. A missing sensor or missing mic permission degrades to
an on-screen warning instead of a crash.
