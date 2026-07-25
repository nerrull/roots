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

Not yet built: **libmediapipe face tracking** (needs the one-time Bazel build) and
the **transition**. Headless coverage: `--selftest`, `--roottest`, `--rootshot`,
`--growshot`, `--fieldshot`, `--rootbench`, `--fieldbench`, `mlp_parity_test`,
`pond_parity_test`.

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
