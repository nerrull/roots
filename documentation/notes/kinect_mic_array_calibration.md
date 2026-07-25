# Kinect v2 mic-array calibration

25 July 2026

Beam steering in `kinect_v2_validate/build/kinect_v2_demo` depends on one number
that cannot be looked up: the mic array's **aperture** (end-to-end spacing of the
4 mics). Microsoft never published the Kinect v2's mic positions and libfreenect2
does not model audio at all, so the code assumes a uniform linear array with an
estimated **0.16 m** aperture (`dsp::KinectV2MicArray`, default in
`AudioCapture::Controls::aperture_m`).

Everything downstream — the delay-and-sum beam direction and the SRP direction
finding — is only as accurate as that estimate. Calibrating it needs a person in
the room, so it could not be done when the feature was built.

## Why this works

There are two independent locators, and one of them needs no calibration:

- **Depth** (closest tracked person) is derived from the IR camera intrinsics,
  which the sensor reports. It is correct by construction.
- **Mic** (SRP over the array) depends entirely on the assumed geometry.

So the depth azimuth is the ground truth, and the aperture is tuned until the mic
azimuth agrees with it. The DSP panel prints both side by side for exactly this:

```
depth ok +14.2 | mic ok +21.8 (c 0.63)
```

## Procedure

1. Run `./build/kinect_v2_demo` (from `kinect_v2_validate/`).
2. Set steering to **`depth (closest)`** so the beam is driven by the reliable
   locator while calibrating the other one.
3. Stand **2–3 m** from the sensor, clearly **off to one side** (roughly 20–40°).
   Dead ahead is useless — every aperture value agrees at 0°, because broadside
   delays are equal regardless of spacing.
4. Talk continuously. Speech is what the SRP band filters (300–3000 Hz) expect;
   check the `mic` readout shows `ok` and confidence `c` is above ~0.4. If it
   stays `--`, either the room is too quiet or `DOA conf min` is set too high.
5. Confirm the green crosshair on the depth image is actually on you, and that
   the `person:` line shows a plausible distance.
6. Adjust **`aperture m`** until the `mic` angle matches the `depth` angle.
   - mic reads **larger** than depth → aperture is too **small**
   - mic reads **smaller** than depth → aperture is too **large**
7. Verify on the other side (mirror your position) and at a second angle. One
   aperture value should fit both; if it does not, the array is probably not
   uniformly spaced and the model needs per-mic positions instead (see
   "If it will not converge").
8. Note the value that works and set it as the default in
   `src/demo/dsp.h` (`KinectV2MicArray`'s `aperture_m` default) and
   `src/demo/audio_capture.h` (`Controls::aperture_m`).

## Sign check

Separate from magnitude: if the beam steers **away** from you rather than toward
you, the depth camera's x-axis and the mic array's axis disagree in sign. Tick
**`mirror depth az`** in the panel, then persist it via
`PersonTracker::Params::mirror` in `src/demo/person_tracker.h`.

Both sign conventions are verified against synthetic ground truth in the tests
(`ctest --test-dir build`), but the physical mapping from camera axis to mic axis
is a property of the sensor bar that only a real measurement settles.

## If it will not converge

The uniform-linear-array assumption may simply be wrong — Kinect **v1** used a
non-uniform layout (mics at roughly −113, −36, +36, +113 mm), so v2 plausibly does
too. In that case replace `UniformLinearArray` with explicit per-mic `x_m[]`
positions in `dsp::KinectV2MicArray`; nothing else needs to change, since
`SteeringDelays` and `SrpDoa` already read arbitrary positions.

A quick sanity bound: the sensor bar is ~249 mm wide, so the aperture cannot
exceed that, and the slider is capped at 0.30 m.

## Other things worth knowing

- **Spatial aliasing** limits useful beamforming to `c / 2d`. At the default
  0.16 m aperture the spacing is 0.053 m, giving ~3.2 kHz — which is why the SRP
  input is lowpassed at 3 kHz. A wider aperture sharpens the beam but lowers this
  ceiling.
- A 4-mic delay-and-sum beam only buys about **6 dB** of on/off-axis rejection
  (measured 6.17 dB, theory 10·log₁₀4 = 6.02). It is a gentle focus, not
  isolation. Do not expect it to null out a competing talker.
- Only one process can hold the sensor. If the demo fails to open with
  `LIBUSB_ERROR_NO_DEVICE`, check another copy is not already running.
