# mirror_app

Unified **Metal** C++ app hosting neuromirror's shader-driven **neural mirror** and
jardins_racine's **3D lit root scene**, in one shading pipeline. See
[`plans/PLAN.md`](plans/PLAN.md) for the architecture and current scope.

Part of the JardinsRacine superbuild (`add_subdirectory(mirror_app)` in the repo-root
`CMakeLists.txt`), and it reaches sideways into the sibling `../../neuromirror`
checkout for MSL sources, `face_landmarker.task`, and the canonical face model.

## Build & run

```sh
# from the jardins_racine repo root
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mirror_app -j$(sysctl -n hw.logicalcpu)
./build/mirror_app/mirror_app
```

VS Code: **Build: mirror_app** / **Run: mirror_app**.

## Status

- [x] Metal window shell (GLFW + CAMetalLayer + ImGui-metal + MetalContext)
- [ ] Neural mirror (MLX C++ fused MLP + libmediapipe face tracking)
- [ ] Metal root renderer (GLSL→MSL port of sdf_viewer's `RootRenderer`)
- [ ] Transition (deferred — not yet defined)

## Layout

- `src/main.mm` — window/app shell + main loop.
- `src/metal_context.{h,mm}` — the shared Metal device/queue (same one MLX uses) + MSL loader.
- `shaders/` — app MSL (compositor / transition / ported root passes).
- `plans/` — working plan, versioned with the code.
