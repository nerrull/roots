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
- [x] Neural mirror render: MLX C++ fused MLP (bit-exact vs Python) → ripple
      features → low-res RGBA16F → Metal texture, presented fullscreen (linear
      upsample).
- [x] **Full `demo_panel` feature parity**: weight shaping (detail / gain-tilt /
      gauss↔uniform / contrast / reseed), ripple/z/colour/time knobs, and the
      mask-emergence (hydro-dip) transition — the complete ImGui panel.
      Numerically matched vs the Python panel (`pond_parity_test`: max u8 |Δ|=1,
      0% pixels off by >1).
- [ ] Neural mirror: libmediapipe face tracking driving the field
- [x] **Metal root renderer** (GLSL→MSL port of sdf_viewer's `RootRenderer`):
      instanced capsule/blade sphere-tracer + Phong/PBR, wisp point lights,
      travelling pulses (`root_geom.metal`); FXAA-lite + volumetric fog +
      axes/grid + wisp glow (`root_fog.metal`); two-pass `MetalRootRenderer`
      into offscreen RGBA16F/Depth32 targets. Divergence: Invert/XOR has no Metal
      fragment-logic-op equivalent → flat-white silhouette.
- [ ] Root scene: live CPlantBox growth (currently a procedural stand-in in
      `RootScene`; needs the GL-free `sdfsim` target) + the face mid-geometry pass
- [ ] Transition (deferred — not yet defined)

Headless checks: `mirror_app --selftest` (MLX→texture path), `mirror_app
--roottest` (root MSL compile + render + readback), `mirror_app --rootshot
<out.ppm> [az el radius]` (render the root scene to an image), and
`mlp_parity_test mirror_app/tests/fixtures` (MLP vs Python reference).
Regenerate the pond weights with `assets/gen_pond_weights.py` (needs neuromirror's venv).

## Layout

- `src/main.mm` — window/app shell + main loop.
- `src/metal_context.{h,mm}` — the shared Metal device/queue (same one MLX uses) + MSL loader.
- `shaders/` — app MSL (compositor / transition / ported root passes).
- `plans/` — working plan, versioned with the code.
