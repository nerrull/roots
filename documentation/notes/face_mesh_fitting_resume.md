# Face mesh fitting — where this was left, and how to resume

Written 25 July 2026. Paused waiting on the NVIDIA Maxine model file, which
lives on another machine.

## The goal

Bring neuromirror's face-mesh fitting into `mirror_app`: fit a 3D morphable
face model to the MediaPipe landmarks the C++ tracker now produces, so the
fitted mesh can drive (a) the masked-training region and (b) the root scene's
face-mask placement.

MediaPipe face tracking itself is **done and committed** — 478 landmarks, 52
blendshapes, 4×4 pose matrix, 4.5 ms/frame. See `mirror_app/src/face_tracker.h`
and the `MediaPipe face tracking` section of `mirror_app/plans/PLAN.md`.

## ⚠️ The Maxine model is not on this machine

Searched all of `~/Documents/Development` and `~/Downloads`: **no `.nvf` file,
no AR SDK install.** `face_model2.nvf` ships inside the NVIDIA AR SDK installer
— free account, EULA-gated, not in the public MAXINE-AR-SDK repo, and not
redistributable (keep it gitignored).

`neuromirror/emotion/nvf_model.py` is a parser for it, but has **only ever been
validated against a synthetic chunk file**. Its own docstring says to run
`--inspect` against the real file before trusting the arrays, and
`neuromirror/emotion/TODO.md` lists all four Maxine items as unchecked:

- ☐ Obtain `face_model2.nvf`
- ☐ Validate the binary layout (`python -m emotion.nvf_model --inspect <file>`)
  — the docs give the chunk tree but not every dtype
- ☐ Confirm the 53-blendshape order matches `_ARKIT_53`
- ☐ `ict_demo --nvf face_model2.nvf` retargets onto the genuine mesh

**Do the `--inspect` pass first.** It is the step that makes the parser
trustworthy, and it is cheap. Until it passes, any C++ port built on
`nvf_model`'s assumed layout is speculative.

## Why ICT is a legitimate stand-in

From `neuromirror/emotion/TODO.md`:

> Maxine's model **IS** a modified ICT-FaceKit (MIT), 100 identity + 53 ARKit
> blendshapes — so the ICT path (`ict_model.py`) is a faithful stand-in, and
> runs on the M4 with no NVIDIA GPU / no FLAME gate.

Same structure means the *fitting* code is model-agnostic: it consumes a basis
(neutral + identity modes + expression modes + landmark indices) and does not
care which file produced it. So the fitter can be written and tested against
ICT now and re-pointed at Maxine later without changing.

## What already exists

### Done, committed
- `mirror_app/tools/export_ict_basis.py` — exports the ICT basis to a flat
  binary for C++. **Runs clean:**
  ```
  vertices: 7813 kept of 26719 (7801 region + 12 landmarks outside it)
  faces:    15166 of 52220
  wrote external/ict_basis.bin (14.6 MB)
  neutral (7813,3)  id (100,7813,3)  ex (53,7813,3)  lm (68,)
  extent [-9.20 -15.76 -1.81] .. [9.20 11.66 13.09]
  ```

### Source material in `../neuromirror`
| file | what |
| --- | --- |
| `emotion/data/ict_basis.npz` | built ICT basis (26719 verts, 100 id, 53 ex) |
| `emotion/ict_model.py` | builds that cache from the ICT `.obj` files |
| `emotion/ict_fit.py` | **the fitting to port** — 127 lines |
| `emotion/nvf_model.py` | Maxine `.nvf` parser, unvalidated |
| `third_party/ICT-FaceKit/` | the MIT mesh source |

## What `ict_fit.py` actually does

Three separable pieces, in rough order of value:

1. **`fit_identity_frames`** — the expensive one-time fit. Alternates: solve a
   2D similarity (Umeyama) for pose given shape, then a ridge-regularised
   linear solve for the 100 identity coefficients given pose, ~12 iterations.
   Expression is *known* (from MediaPipe blendshapes) and subtracted first, so
   identity is not polluted by expression. Accepts multiple frames sharing one
   identity, which averages out landmark noise.

2. **`mp_expression`** — MediaPipe blendshape dict → ICT expression weights by
   normalised ARKit name (`mouthSmileLeft` → `mouthSmile_L`). Nearly 1:1.

3. **`solve_pose`** — 3D head rotation via `cv2.solvePnP`, deliberately dropping
   the dlib jawline (0–16) because the contour points are unreliable and bias
   pitch/depth.

`MP68` at the top maps dlib-68 indices onto MediaPipe FaceMesh vertex indices —
needed regardless of which mesh is used.

## Resuming: suggested order

1. Copy `face_model2.nvf` across; gitignore it.
2. `python -m emotion.nvf_model --inspect face_model2.nvf` in neuromirror. Pin
   the dtypes/ordering (`BSIS` modes-first vs interleaved, `TRNG` uint16,
   `IBUG` order) against the real chunk sizes. **Do not skip this.**
3. Confirm the 53-blendshape order matches `_ARKIT_53`, else fix the
   `mp_expression` name mapping.
4. Generalise `export_ict_basis.py` to take either source — the output format is
   already model-agnostic, so this is a source switch, not a rewrite.
5. Port the fitter to C++ (`face_fit.{h,cpp}`): Umeyama similarity + ridge solve.
   Both are small dense problems (68×2 points, 80×80 normal equations); no new
   dependency needed. `solve_pose` needs PnP — either implement it or use the
   2D similarity alone at first, which is enough for mask placement.
6. Wire `FaceTracker` → fitter → `Pond::updateFitTarget(rgb, mask)`, replacing
   the face-oval hull mask (`RasteriseFaceMask`) with the projected mesh.

## Gotchas already found

- **12 of the 68 landmark vertices fall outside ICT's face region.** Exporting
  the mask region alone silently drops them and skews the identity fit. The
  exporter keeps the union; re-check this on the Maxine mesh, whose region
  split may differ.
- Only 15166 of 52220 faces have all three corners inside the kept set. Faces
  straddling the boundary are dropped, else they render as stray triangles
  across the mask edge.
- ICT units are roughly centimetres, extent about ±9 × ±15 × 13 — not
  normalised. Anything consuming the mesh needs to scale.
- `ict_fit` flips y (`* [1, -1]`) to work in a y-up space against MediaPipe's
  y-down pixels. Easy to lose in translation.
- `mirror_app/external/` is gitignored, so `ict_basis.bin` is generated, not
  committed — same arrangement as the MediaPipe build.

## Open question worth settling early

Whether the fitted mesh is actually better than the landmark-hull mask already
working (`RasteriseFaceMask` + `FaceOvalIndices`). The hull is ~40 lines with no
model file, no fit, and no per-frame solve. The mesh wins on 3D pose and on
giving the root scene something to place, but if the *only* consumer were the
training mask, the hull may be enough. Worth a side-by-side before committing to
the full path.
