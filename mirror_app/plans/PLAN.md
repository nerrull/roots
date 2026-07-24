# Neuromirror ⇄ Roots — unified Metal C++ app

> Working plan, versioned with the code. The transition effect is **not yet
> defined** — current focus is (1) scaffolding, (2) the C++ neural-mirror
> implementation, (3) porting the mesh/root scene to Metal. The transition
> state machine / effect and Wwise are deferred until the above lands.

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

## Current work breakdown (active scope)

1. **Scaffold** `mirror_app/`: CMake, `MetalContext`, minimal GLFW+CAMetalLayer+
   ImGui-metal window shell (adapted from `reactor_cpp/main.mm`). Runnable checkpoint.
2. **C++ neural mirror**: `mlp_forward` (MLX C++ port reusing verbatim MSL + features
   + upsample), `face_tracker` (libmediapipe), `MirrorScene` → texture. MLP parity
   test vs the Python reference.
3. **Port root/mesh scene GLSL→Metal**: `MetalRootRenderer` mirroring `RootRenderer`'s
   API; port `shader.frag`/`fog.frag`/`face.*` to MSL, GL TBOs → Metal buffers; live
   CPlantBox growth; validate visually against `mask_relay_gui`.
4. **Refactor** `sdf_viewer/CMakeLists.txt` to expose a GL-free `sdfsim` target;
   make `RootRenderer` shader/param paths runtime-overridable; keep GL apps building.

## Deferred (revisit after the above)

- Transition state machine (`SceneDirector`) + transition effect (`TransitionPass`).
- Wwise audio layer.
- ICT-FaceKit identity fitting (v1 drives the mask directly from tracked
  landmarks + blendshapes).
