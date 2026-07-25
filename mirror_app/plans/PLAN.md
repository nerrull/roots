# Neuromirror ⇄ Roots — unified Metal C++ app

> Working plan, versioned with the code. **Status as of 2026-07-24:** the
> scaffold, the C++ neural mirror, the full GLSL→Metal root/face port, and live
> CPlantBox growth are all done and validated; a cached-system LOD/culling path is
> in. Remaining: **libmediapipe face tracking** for the mirror, and the (still
> undefined) **transition** between the two scenes. Wwise + ICT-FaceKit fitting
> stay deferred.

## Status snapshot

Done (validated, on `jardins_racine` main):

- **Scaffold** — Metal window shell (GLFW + CAMetalLayer + ImGui-metal +
  `MetalContext`), superbuild wiring.
- **Neural mirror** — MLX C++ fused-MLP + features + upsample, bit-exact vs the
  Python reference; full `demo_panel` parity (weight shaping, ripples, z, colour,
  hydro-dip transition); performance-matched to Python; `MirrorScene` → texture.
- **Root render (Metal)** — `MetalRootRenderer`: capsule/blade sphere-tracer +
  Phong/PBR, wisps, pulses (`root_geom.metal`); FXAA-lite + volumetric fog +
  overlays + wisp glow (`root_fog.metal`); the face mask **mid-geometry pass**
  (`root_face.metal`, depth-composited). Invert/XOR falls back to flat white (no
  Metal fragment logic ops).
- **Live CPlantBox** — `RootSim`: GL-free, incremental port of the "mask relay"
  growth (per-frame state machine), reusing `MaskCavities`/`RootAttractors` +
  CPlantBox behind a pimpl. `RootScene` steps it and drives the face pass.
- **Scaling** — cached static instances (`addInstance`, baked, uploaded once) with
  **frustum culling + distance LOD + sub-pixel capsule cull**; auto render-scale of
  the roots' internal resolution. Profiled: the pass is overdraw-bound (lighting is
  ~free); 256-system field 19.1→10.1 ms with cull+LOD.
- **Mirror perf** — fused ripple-features kernel + cached coord grid:
  **26.2 → 16.2 ms at 960×540 (38 → 62 fps)**, 108.8 → 64.5 ms at 1080p. See
  [Neural mirror performance](#neural-mirror-performance).

Not yet built: **libmediapipe face tracking** (needs the one-time Bazel build) and
the **transition**. Headless coverage: `--selftest`, `--roottest`, `--rootshot`,
`--growshot`, `--fieldshot`, `--rootbench`, `--fieldbench`, `mlp_parity_test`,
`pond_parity_test`, `ripple_parity_test`.

## Context

Two working codebases revolve around the **same face mask**
(`canonical_face_model.obj`):

- **neuromirror** — a shader-driven "neural mirror": a coordinate-MLP reconstructs
  the live face + face tracking/fitting. Today **Python only** (MLX fused-MLP with
  inline Metal, MediaPipe tracking, ICT-FaceKit fitting). Per-frame output is
  *an RGB grid → upsample → display* — i.e. a texture.
- **jardins_racine/sdf_viewer** — a 3D lit scene (`RootRenderer`, a **GLSL**
  sphere-tracer over CPlantBox root geometry). The flagship `mask_relay_gui` grows
  roots **around the same face mask**. Currently **OpenGL 4.1 + GLEW + ImGui**.

Goal: one native C++ app that runs the neural mirror and the 3D root scene and
(eventually) transitions between them, reusing each codebase as a module rather
than reimplementing in parallel.

## Confirmed decisions

- All-native C++; **unified Metal backend with a common shading pipeline** (port
  the GL root renderer to Metal — no GL/IOSurface interop).
- Face tracking via **libmediapipe** (one-time Bazel build → `.dylib`, then
  CMake-only, full 478-landmark + 52-blendshape + pose parity).
- Roots via **live CPlantBox** simulation.
- App lives at **`jardins_racine/mirror_app/`**; `sdf_viewer` refactored to expose
  reusable GL-free sim targets.
- **Only one simulation runs at a time** — running both sims tanks framerate, so any
  eventual transition is a well-defined handoff, not a blend of two live sims.

## Architecture — one Metal pipeline, two scenes

```
GLFW window → CAMetalLayer (Metal), ImGui (imgui_impl_metal)
 MetalContext (device/queue shared with MLX, MSL loader, RT/texture helpers)
 ├─ MirrorScene → MTLTexture   (MLX MLP zero-copy + libmediapipe tracking)
 ├─ RootScene   → MTLTexture   (Metal-ported RootRenderer, live CPlantBox)
 └─ TransitionPass → (deferred; effect over fromTex/toTex/t/mask)
```

MLX arrays are Metal-buffer-backed, so the mirror stays **zero-copy** (same
`MTLCreateSystemDefaultDevice()` as the compositor). Both scenes produce Metal
textures + depth in one pipeline, so an eventual transition can be a real shader
effect, not an alpha fade.

## Reuse map

| Piece | Source | Disposition |
|---|---|---|
| App shell (GLFW+CAMetalLayer+ImGui-metal + runtime MSL compile) | `neuromirror/reactor_cpp/src/main.mm` | **Reuse** as scaffold |
| Root growth sim + geometry (GL-free) | `sdf_viewer/FlowerLSystem.h`, `MaskCavities.h`, `RootAttractors.h` + **CPlantBox** | **Reuse as-is** |
| Root **rendering**: sphere-tracer + fog + capsule TBOs | `sdf_viewer/RootRenderer.{h,cpp}`, `shader/fog/face .vert/.frag` | **Port GLSL→MSL** (`MetalRootRenderer`) |
| Fused MLP kernel | `neuromirror/mlx_fused_mlp/forward.py` (MSL) + `features.py` (`ENRICHED_DIM=8`) + `upsample.py` | **Port to MLX C++** reusing verbatim MSL |
| Face tracking | MediaPipe `face_landmarker.task` | via **libmediapipe** `.dylib` |
| ICT-FaceKit fitting | `neuromirror/emotion/ict_*` | **Deferred** |

## Work breakdown

1. ✅ **Scaffold** `mirror_app/`: CMake, `MetalContext`, GLFW+CAMetalLayer+
   ImGui-metal window shell.
2. 🟡 **C++ neural mirror**: `mlp_forward` (MLX C++, verbatim MSL) + features +
   upsample + `MirrorScene` **done** and parity-tested; **`face_tracker`
   (libmediapipe) still to do** — this is the one remaining piece of item 2.
3. ✅ **Port root/mesh scene GLSL→Metal**: `MetalRootRenderer` (geometry + fog +
   face mid-pass), GL TBOs → Metal buffers, validated visually.
4. ✅ **Live CPlantBox**: shipped as the GL-free `RootSim` module reusing
   `MaskCavities`/`RootAttractors` + CPlantBox directly (rather than refactoring
   `sdf_viewer`'s CMake into a separate `sdfsim` target — same "no parallel
   reimplementation" goal, less churn to the GL apps).
5. ✅ **Cached-system scaling**: instances + LOD + frustum/sub-pixel culling +
   auto render-scale (`addInstance`, `RootScene::buildField`, `--fieldbench`).

## Next

1. **libmediapipe face tracking** — one-time Bazel build → `.dylib`; feed 478
   landmarks + 52 blendshapes + pose into the mirror and the root mask placement.
   (Bazel may not be installed on the dev box — flagged.)
2. **Transition** — still undefined. When defined: a `SceneDirector` single-sim
   handoff (snapshot the outgoing scene to a static texture, stop its sim, run the
   incoming one) + a `TransitionPass` MSL effect over `(fromTex, toTex, t, mask)`.
   Both scenes already produce Metal textures + depth in one pipeline, so this can
   be a real effect, not an alpha fade.

## Deferred

- Wwise audio layer.
- ICT-FaceKit identity fitting (v1 drives the mask directly from tracked
  landmarks + blendshapes).
- Per-instance model matrix (cached instances are baked-static today; add a
  per-draw transform if the transition needs to move/scale whole systems).

## Neural mirror performance

Measured on **Apple M4, 10-core, 32 GB**, `--bench`, 25 July 2026.

### Where the frame went (960×540, before)

| phase | ms | |
| --- | --- | --- |
| coord grid | 0.65 | rebuilt every frame, depends only on size |
| ripple features | 9.90 | **38% of the frame** |
| MLP + tone | 15.10 | |
| **total** | **25.65** | 38 fps |

### The fix

**Ripple features: 9.90 → 0.60 ms (16.6×).** The op-graph formulation expressed
~40 FLOPs/pixel as ~75 elementwise MLX ops, each round-tripping the whole
518k-element array through memory. One fused `metal_kernel` reads 2 floats and
writes 8 halves per pixel with everything in between in registers — now
memory-bound at ~12 MB/frame, so there is nothing further to win.

`mx::compile` was tried first as the smaller change and manages only 1.33×:
`concatenate`/`split` break its fusion groups.

**Coord grid: 0.65 → 0 ms.** Cached on `(lh, lw)`, evaluated eagerly so the
linspace/meshgrid graph is not spliced into a later frame.

### Result

| | before | after | |
| --- | --- | --- | --- |
| 960×540 (half) | 26.16 ms (38 fps) | **16.21 ms (62 fps)** | 1.61× |
| 1920×1080 | 108.76 ms (9 fps) | **64.48 ms (16 fps)** | 1.69× |

Verified by `pond_parity_test` (max deviation 1/255, 0% above), `mlp_parity_test`
(exact), and `ripple_parity_test` (fused vs op-graph, within 1 fp16 ulp across 11
parameter settings including every branch and 1/2/5/12 sources).

### What is left, and what it costs (960×540)

| | ms | |
| --- | --- | --- |
| ripple features | 0.82 | fused kernel; memory-bound |
| **MLP + clip/astype** | **15.43** | **95% of the frame** |
| shipping defaults, total | 16.22 | |

Tone mapping is **not** the problem — every optional stage is sub-millisecond:

| stage | added ms |
| --- | --- |
| gamma ≠ 1 | +0.08 |
| colour mix | +0.44 |
| swap_rb | +0.56 |
| amp_drives_color | +0.72 |
| srgb_fix | +0.79 |
| **transition (relief + lighting)** | **+3.60** |

Only the emergence transition is expensive, and it is transient by nature — but
it does push a 16.2 ms frame to 19.9 ms, i.e. it breaks 60 fps *during the
transition*. Worth folding into a kernel if the transition must hold 60.

**The MLP is now the whole budget.** It is already a fused kernel (8→64×6→3,
fp16, simdgroup matmul). Further headroom has to come from the network itself —
fewer layers, narrower hidden, or a smaller internal resolution — not from the
surrounding code, which is now ~5% of the frame.

### hidden_dim: what narrowing buys

`hidden_dim` was already configurable — it is a kernel *template* parameter
(`HIDDEN`), so each width compiles its own specialisation and there is **no
runtime cost to configurability itself**. Only the Pond constructor hardcodes 64.

Measured at 960×540, `--bench 2 200`, whole frame:

| hidden | ms | fps | vs 64 |
| --- | --- | --- | --- |
| 16 | 3.74 | 267 | 4.3× |
| **32** | **6.80** | **147** | **2.4×** |
| 48 | 10.85 | 92 | 1.5× |
| 64 (ships) | 16.24 | 61 | — |

Not the 4× that hidden² would predict, because ~0.8 ms of the frame is features
and tone, and the per-layer weight streaming and barriers do not shrink with
width.

**Retuning the tiling is not worth it.** `n_blks = (TILE_ROWS/16) · ceil(N/16)`
drops from 8 to 4 at hidden=32, so half the simdgroups idle — but fixing that
recovers only ~6%:

| hidden=32 tiling | ms |
| --- | --- |
| 32/8 (current) | 6.80 |
| 32/4 | 6.58 |
| 64/8 | 6.49 |
| 128/8 | 6.37 |

And 32/8 is genuinely optimal at hidden=64 (64/8 → 17.25 ms, 32/4 → 17.99 ms),
so the shipping constants stay.

**Ceiling: hidden_dim ≤ 80.** `wbuf` is `HIDDEN²` halves, so threadgroup memory
is quadratic; hidden=128 needs 57344 bytes against Apple's 32768 limit and Metal
refuses to load the pipeline. `fused_mlp_forward` now checks this up front and
throws something legible instead of aborting on the first frame.

`mlp_width_test` validates the kernel against a plain MLX matmul chain at widths
16/32/48/64/80 (48 and 80 deliberately not multiples of the 16-wide simdgroup
tile), depths 2/4/6/10, ragged row counts (1, 33, 1000), and all four
activations — plus that hidden=128 is rejected cleanly.

**Switching to 32 is a one-line change** in the `Pond` constructor
(`MLPConfig{ENRICHED_DIM, 64, ...}` → `32`). Left at 64 pending a look at what
the narrower network does to the image: it is a different network, so the
pattern changes and the fixtures would need regenerating.

### Training budget at hidden_dim=32

Measured with the **real fused training path** (fused forward + fused backward
vjp + Adam) from `neuromirror/mlx_fused_mlp/training.py`, at the mirror's config
(8 → N → 3, 6 layers, tanh/sigmoid), M4, 25 July 2026.

⚠️ **The fused backward kernel is not ported to C++ yet.** `mlp_forward.cpp` is
forward-only; `fused_mlp_backward` + the custom vjp exist only in the Python
reference. These numbers are what the C++ side will get *after* that port, and
they are the reason to do it: the explicit (non-fused) backward is much slower.

Budget at 60 fps, hidden=32: 16.67 ms frame − 6.80 ms display = **9.87 ms**.

| train res | points | ms/step | steps/frame |
| --- | --- | --- | --- |
| 960×540 (half) | 518 400 | 23.10 | **0.4** |
| 480×270 (quarter) | 129 600 | 5.83 | 1.7 |
| 320×180 | 57 600 | 2.63 | 3.8 |
| 240×135 (eighth) | 32 400 | 1.52 | 6.5 |
| 160×90 | 14 400 | 0.72 | 13.7 |
| 120×68 | 8 160 | 0.44 | 22.3 |
| 64×36 | 2 304 | 0.18 | 54.5 |

**Half resolution does not fit.** One step at 518 400 points costs 23.1 ms —
2.3 whole frames — against a 9.87 ms budget. A training step is ~3.4× a display
render over the same points (forward + backward + optimiser), so training at
display resolution can never fit while the display is also drawing.

Cost is linear in point count, so the tradeoff is a straight line: ~600×340 for
one step/frame, ~310×175 for four, ~220×125 for eight.

**At hidden_dim=64 training is impossible at 60 fps** — the display alone leaves
0.43 ms, which is under a single step at any resolution. If live training is
required, hidden=32 is not an optimisation, it is a precondition.

### Hybrid sine/tanh activations (SIREN split)

`MLPConfig` gained `split` + `act_first`: the first `split` hidden layers use
`act_first` (sine), the rest use `activation` (tanh). **`split = 0` is the
original network and is bit-identical to it** — verified by `pond_parity_test`
and `mlp_parity_test` still passing unchanged.

Why: raw coordinates through tanh cannot resolve fine detail (spectral bias),
and a Fourier input encoding fixes that but stamps an axis-aligned grid over the
whole image. A leading sine layer builds the high-frequency basis instead, with
no encoding and no grid, while the tanh layers behind it stay responsive to the
existing `detail`/`contrast` weight scales.

**One sine layer is enough.** Face-fit MSE at hidden=32, 6 layers, 1500 steps:

| split | pattern | fit loss |
| --- | --- | --- |
| 0 | TTTTT | 0.00562 |
| **1** | **STTTT** | **0.00250** |
| 2–4 | SS.. | 0.00249–0.00263 |
| 5 | SSSSS | 0.00273 |

### Two independent controls

| knob | what it does |
| --- | --- |
| `sine_w0` | how many regions the field breaks into (composition) |
| `detail` | how hard the boundaries are (articulation) |

They genuinely decouple at low `sine_w0`. Fraction of frame with near-zero
gradient, measured in C++ at a fixed z:

| sine_w0 | detail 0.8 | 1.5 | 2.5 | 4.0 |
| --- | --- | --- | --- | --- |
| 2 | 100% | 76% | 58% | 57% |
| 5 | 100% | 63% | 56% | 55% |
| 10 | 99% | 64% | 59% | 52% |
| 20 | 62% | 48% | 42% | 29% |
| 40 | 45% | 31% | 25% | 21% |

At `sine_w0` 2–10 roughly half the frame stays open while mean gradient rises
~10x across the `detail` sweep — large flat areas survive much sharper edges. By
40 that is gone and the field is uniform texture with no background. **The
open-composition regime is `sine_w0` 5–10 with `detail` 2.5–4.**

### Cost

| | 960×540, hidden=64 |
| --- | --- |
| `sine_layers = 0` | 16.31 ms (61 fps) |
| `sine_layers = 1` | 18.70 ms (53 fps) |
| `sine_layers = 2` | 18.16 ms (55 fps) |

The hybrid costs ~15%: it keeps the expensive `tanh` calls and adds a `sin`.
Pure sine would be *cheaper* than pure tanh (measured −17% on the forward), so
the cost here buys the tunable tanh stack, not the sine.

### Not yet ported: the backward

`mlp_forward.cpp` is forward-only, so this is the aesthetic/render path.
Training needs the sine-capable backward prototyped in Python: the shipping one
reconstructs `act'(z)` from post-activations (`tanh: 1-a²`), and `cos(z)` is not
recoverable from `sin(z)`. The fix is to store `act'(z)` for the sine layers
only during the re-forward pass — validated against explicit backprop at 0.0010
relative, and it costs one threadgroup buffer sized by `split` rather than by
depth:

| hidden | shipping | all-layer preact | split=1 preact |
| --- | --- | --- | --- |
| 32 | 13312 ✓ | 20480 ✓ | 14336 ✓ |
| 64 | 28672 ✓ | 43008 ✗ | 30720 ✓ |

i.e. the split is what keeps a trainable sine network within budget at
hidden=64 at all.

## MediaPipe face tracking

Built from **upstream MediaPipe's official C Tasks API**
(`mediapipe/tasks/c/vision/face_landmarker`) as a shared library — 478
landmarks, 52 blendshapes, 4x4 facial transformation matrix, Apache-2.0.

**Not cpvrlab/libmediapipe**, which this plan previously named: it pins
MediaPipe v0.8.11, which predates the Tasks API entirely (legacy 468-point
face_mesh, no blendshapes, no pose matrix) and is GPL-3.0.

```sh
./setup-mediapipe.sh      # installs deps, clones, patches, builds, verifies
cmake -S . -B build       # -> "MediaPipe face tracking enabled"
```

Measured: **4.5 ms/frame**, 50/50 detection, resolution-independent (the graph
crops to the detected face ROI). Face-oval mask rasterises to ~3.8% of frame at
640x480 — directly usable as the masked-training region.

### The build needs five fixes, three of them as patches

`external/` is gitignored, so the source edits live in `patches/` and
`setup-mediapipe.sh` re-applies them idempotently. None are cosmetic; each is a
distinct failure that does not name itself:

| # | problem | symptom |
| --- | --- | --- |
| — | system python 3.13+ | "Could not find requirements_lock.txt matching 3.14" — fixed by `--repo_env=HERMETIC_PYTHON_VERSION=3.12` |
| — | no JDK | `no such package '@@rules_java~//tools/jdk'` — fixed by `brew install openjdk` + `JAVA_HOME` |
| 0001 | OpenCV path/version | `fatal error: opencv2/core/version.hpp not found`. Upstream expects opencv@3 at `/usr/local`; arm64 Homebrew is `/opt/homebrew` and the plain `opencv` formula is now **5.x**, which MediaPipe does not build against. Needs `opencv@4` and deeper header paths. |
| 0002 | protobuf conflict | **Segfault before `main()`**, inside `DescriptorPool::InternalAddGeneratedFile`. `libopencv_video` pulls `libopencv_dnn` -> Homebrew `libprotobuf.35`, colliding with MediaPipe's static protobuf during dyld initialisers. MediaPipe's Image path does not use the video module, so it is dropped. |
| 0003 | no exported symbols | A 14 MB dylib exporting **zero** `MpFaceLandmarker*` symbols. The `.dylib` target depends on `face_landmarker_lib`, which lacks `alwayslink = 1`, so nothing inside the shared library references the C entry points and the linker discards them. Repointed at `face_landmarker_c_lib`; same fix for `:image`. |

`setup-mediapipe.sh` asserts the symbol count after building, because 0003's
failure mode is a build that reports success and produces a useless library.

CMake stages the dylib out of `bazel-bin` and rewrites its `install_name` to
`@rpath/...` — bazel emits a bare filename, which dyld treats as a literal path
and never resolves against an rpath.

### Face mesh fitting — paused

Fitting a 3D morphable face model to the MediaPipe landmarks (for the training
mask and the root scene's mask placement) is **paused waiting on NVIDIA's
`face_model2.nvf`**, which is EULA-gated and lives on another machine.

`tools/export_ict_basis.py` works and produces `external/ict_basis.bin`
(7813 verts, 100 identity + 53 ARKit expression modes, 14.6 MB) from
ICT-FaceKit, which Maxine's model is a modified version of -- so the fitter can
be written against ICT and re-pointed later.

Full handoff, including the order to resume in and the gotchas already found:
[`documentation/notes/face_mesh_fitting_resume.md`](../../documentation/notes/face_mesh_fitting_resume.md)
