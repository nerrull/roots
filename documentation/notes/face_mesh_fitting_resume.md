# Face mesh fitting — done, and what the decisions were

Written 25 July 2026 as a handoff while this was paused waiting on NVIDIA's
`face_model2.nvf`. **Rewritten the same day: the model arrived, its layout was
validated, and the fitting is built and wired.** Kept rather than deleted
because the reasoning below is not recoverable from the code.

Current state lives in
[`mirror_app/plans/PLAN.md`](../../mirror_app/plans/PLAN.md) under
*Face mesh fitting*. This note is the why.

## What shipped

| piece | file |
| --- | --- |
| basis exporter (two bases, one coefficient set) | `mirror_app/tools/export_face_basis.py` |
| basis loader | `mirror_app/src/face_basis.{h,cpp}` |
| the fitter | `mirror_app/src/face_fit.{h,cpp}` |
| test vs. the Python reference | `mirror_app/tests/face_fit_test.cpp` |
| wiring | `mirror_app/src/main.mm`, `root_scene.mm`, `kinect_target.cpp` |

`tools/export_ict_basis.py` is superseded by `export_face_basis.py` and was
removed; the new one produces ICT topology too, with `--nvf` pointed at nothing.

## The decisions worth keeping

### The mask did not get the mesh

The note's original "open question worth settling early" was whether a fitted
mesh actually beats the landmark-hull mask that already worked
(`RasteriseFaceMask` + `FaceOvalIndices`) — the hull being ~40 lines with no
model file, no fit, and no per-frame solve.

**It was settled by not asking the mask to use the mesh.** The training mask
takes the hull; the mesh goes to the root scene. The hull is sufficient for a
region-of-supervision — it only has to say "person here, background there" — and
the mesh's advantages (3D pose, a surface to place) are exactly the things the
root scene needs and the mask has no use for. Neither path pays for the other.

### Fit against ICT, draw with Maxine

Both models are 100 identity + 53 ARKit blendshapes (Maxine's *is* a modified
ICT-FaceKit), but they are trustworthy in different places:

- ICT has dlib-68 landmark ordering right, from `vertex_indices.json`.
- NVF's `LMRK` chunk stores landmarks in an internal order that neuromirror
  recovered by **2D nearest-neighbour matching against MediaPipe** on a frontal
  face (`_LM68_REORDER` in `nvf_model.py`). That is a heuristic inside an
  otherwise byte-validated parser, and it is the half a fit would be most
  sensitive to.
- NVF has the better render topology: 2056 verts / 4048 tris, already a face
  mask — no region crop, no straddling triangles.

So the export carries **both** bases and one coefficient set, with ICT's modes
resampled onto NVF's vertices by nearest neighbour (the meshes are aligned
closely enough that this is a resample, not a registration — neuromirror
measured nose tips agreeing to <0.02 mm; the export reports mean residual
0.196 model units). This is `nvf_live.py`'s bridge, moved to export time so the
C++ side has no mapping table.

### Head pose came free

`ict_fit.solve_pose` ran `cv2.solvePnP` on brow/eye/nose/mouth landmarks,
deliberately dropping the dlib jawline because the contour points bias
pitch/depth. None of that was ported: MediaPipe's Tasks API returns a 4×4
facial transformation matrix, so the rotation is read off it directly. **That
is the whole reason the C++ side has no OpenCV dependency.** The 2D similarity
still runs, for scale and placement in *this* image — MediaPipe's translation
is in its own metric space and is not what is wanted.

`useTrackerPose(false)` falls back to the similarity alone, which places the
mask correctly but does not turn it.

## Gotchas, confirmed and new

Carried over and still true:

- **12 of the 68 landmark vertices fall outside ICT's face region.** The
  landmark basis is extracted from the *full* ICT mesh rather than the render
  mesh, which makes this structurally impossible to hit again — the crop can no
  longer reach the landmarks.
- **ICT units are roughly centimetres**, extent ±7.5 × −10..9.6 × 2.6..13. Not
  normalised. `RootScene::setFittedFace` normalises.
- **The y-flip.** MediaPipe is y-down pixels, the basis is y-up. Targets flip on
  the way in, projections flip on the way out. `face_fit_test` checks the
  projected mesh lands *inside* the image, because a lost flip puts it above.
- `mirror_app/external/` is gitignored: `face_basis.bin` is generated.

New, found while building this:

- **`RootScene` must not re-normalise the fitted mesh per frame.** An expression
  changes the mesh's extent, so per-frame normalisation rescales the mask every
  time the person opens their mouth. The transform is captured once, from the
  first mesh seen, and reused — the mask holds still and the face moves inside
  it.
- **Landmarks 234 and 454 are MP68 targets** (dlib 1 and 15, jawline) *and* the
  points `Frontality` reads. A test that wrote to them to fake a frontal score
  silently corrupted two of the 68 points the fit solves against, and the fit
  degraded from 0.76 correlation to 0.11. Anything that synthesises landmarks
  must leave the MP68 set alone. (Landmark 1, the nose tip, is *not* in MP68 and
  is safe to set — which is how the test gets a frontal score honestly.)
- **Identity frames must be ranked, not thresholded.** The first cut gated on
  `frontality > 0.55 && neutrality > 0.45` and never accepted a single real
  frame: MediaPipe reports substantial baseline activation on an ordinary face,
  and a frame of someone mid-sentence scores **0.04** neutrality. That is
  correct, not a bug — verified against the Python `_neutrality` on the same
  image, which returns 0.0443. A fixed threshold either accepts everything or
  nothing depending on the person and the lighting, so collection would stall
  forever. `nvf_live.py` had this right: score candidates over a fixed time
  window, sort, keep the best N. `offerIdentityFrame` now maintains the best
  `max_frames` seen and the app collects on a clock.
- **The tracker must not pull its own camera frame.** `KinectFitTarget::
  lastFrameRGB8` reads the *retained* snapshot rather than calling `pollColor`
  again; a second poll races the fit path and the two end up on different
  moments, which shows as the fit smearing whenever anyone moves.
- **Bazel's default macOS deployment target is too low for flatbuffers.** The
  MediaPipe build now fails with ~20 `'value' is unavailable: introduced in
  macOS 10.13` errors in `binary_annotator.cpp`, naming the SDK rather than the
  flag responsible. Fixed in `setup-mediapipe.sh` with `--macos_minimum_os=11.0`
  and `--host_macos_minimum_os=11.0` (host too — `flatc` hits the same wall).
  This did not reproduce on the first build; it surfaced after `/private/var/tmp`
  was purged and everything rebuilt from scratch.

## What the fit is worth

`face_fit_test` against neuromirror's `fit_identity_frames` on identical
synthetic input, correlation of recovered vs. ground-truth identity:

| ridge | python | this port |
| --- | --- | --- |
| 6.0 | 0.7609 | 0.7609 |
| 1.0 | 0.8061 | 0.8061 |
| 0.1 | 0.8443 | 0.8443 |

Not near 1, and that is the algorithm rather than a defect: ridge shrinks toward
the mean face, and the identity modes are far from orthogonal once projected to
2D, so many coefficient vectors explain the same 68 points about equally well.
The number rules out a fit that found a *different* face. Mean landmark residual
is 0.27 px.

## Still open

- **Texture the mask from the camera** (`sample_texture` in `ict_fit.py`, and
  neuromirror's own remaining TODO for both ICT and NVF). The mask currently
  renders in a flat material colour.
- **Photometric refinement** of identity — 68 landmarks is a thin constraint,
  which is what the ridge term is compensating for.
- The fit has only been validated against synthetic ground truth and live
  eyeballing. There is no regression fixture from a real face.
