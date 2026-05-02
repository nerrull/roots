# SDF Root System Viewer — Implementation Plan

Interactive Dear ImGui window that sphere-traces a single growing root system in real time.

## Constraints

- **macOS OpenGL 4.1 max** — no SSBOs (require 4.3). Use Texture Buffer Objects (TBOs) instead; available since OpenGL 3.1, same random-access semantics.
- Links against the existing `CPlantBox` shared library.

---

## File Layout

```
gui/sdf_viewer/
  main.cpp            app entry, GLFW window, ImGui setup, render loop
  RootRenderer.h      class declaration
  RootRenderer.cpp    FBO, fullscreen quad, shader, TBO management
  shader.vert         passthrough (clip-space quad → UV)
  shader.frag         sphere tracer
  CMakeLists.txt
```

---

## Phase 1 — Project Scaffold

`CMakeLists.txt` adds an executable target linking:

- `CPlantBox` (shared lib, already built)
- `glfw3`
- `imgui` (GLFW + OpenGL3 backends)
- OpenGL + GLEW (or `glad`)

`main.cpp` responsibilities:
- Create GLFW window with OpenGL 4.1 core-profile context
- Initialize ImGui (`ImGui_ImplGlfw_InitForOpenGL`, `ImGui_ImplOpenGL3_Init("#version 410")`)
- Instantiate `CPlantBox::Plant`, call `readParameters(...)` and `initialize()`
- Run render loop (Phase 6)

---

## Phase 2 — Offscreen Framebuffer

`RootRenderer` owns one FBO with a `GL_RGBA8` color attachment (default 800×600).

```
glGenFramebuffers / glGenTextures / glFramebufferTexture2D
```

The color texture handle is exposed as `GLuint colorTex()` so ImGui can display it:

```cpp
ImGui::Image((ImTextureID)(intptr_t)renderer.colorTex(), ImVec2(800, 600));
```

Resize support: if the ImGui window changes size, recreate the FBO texture to match.

---

## Phase 3 — GPU Segment Data (TBOs)

After each `simulate(dt)`, extract geometry via `SegmentAnalyser`:

```cpp
SegmentAnalyser ana(plant);
// ana.nodes      → std::vector<Vector3d>
// ana.segments   → std::vector<Vector2i>
// ana.getParameter("radius") → std::vector<double>
```

Three TBOs, updated whenever segment count changes:

| Buffer | Internal Format | GLSL Sampler | Content |
|--------|----------------|--------------|---------|
| nodes | `GL_RGB32F` | `samplerBuffer` | `vec3` positions |
| segments | `GL_RG32I` | `isamplerBuffer` | `ivec2` node index pairs |
| radii | `GL_R32F` | `samplerBuffer` | `float` per-segment radius |

Upload path: `glBindBuffer(GL_TEXTURE_BUFFER, ...)` → `glBufferData(...)`. Full re-upload each step is fine for a single root system; incremental updates (only new segments) can be added later.

---

## Phase 4 — Raymarching Fragment Shader

### `shader.vert`

Passthrough — outputs `v_uv` in `[0,1]²` from a clip-space fullscreen quad (two triangles, no attributes needed beyond `gl_VertexID`).

### `shader.frag`

**Uniforms:**
```glsl
uniform samplerBuffer  u_nodes;
uniform isamplerBuffer u_segments;
uniform samplerBuffer  u_radii;
uniform int            u_segCount;
uniform vec3           u_eye;
uniform mat3           u_cam;   // [right, up, -forward]
uniform float          u_fov;   // vertical half-angle in radians
uniform vec2           u_res;
```

**`sceneSDF(vec3 p)`** — direct port of `SDF_RootSystem::getDist()`:
```glsl
float sceneSDF(vec3 p) {
    float d = 1e9;
    for (int i = 0; i < u_segCount; i++) {
        ivec2 seg = texelFetch(u_segments, i).xy;
        vec3  a   = texelFetch(u_nodes, seg.x).xyz;
        vec3  b   = texelFetch(u_nodes, seg.y).xyz;
        float r   = texelFetch(u_radii, i).r;
        vec3 ab = b - a;
        vec3 ap = p - a;
        float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
        d = min(d, length(ap - t * ab) - r);
    }
    return d;
}
```

**Sphere tracer:** max 96 steps, hit threshold 0.001 cm, max distance 200 cm.

**Normal:** central differences, epsilon 0.01.

**Shading:**
- Roots: warm tan `vec3(0.55, 0.40, 0.20)`, Phong with one directional light from above-left
- Background: dark soil `vec3(0.12, 0.08, 0.05)`, fog falloff with depth
- Optional: cheap soft shadow by stepping back toward light and sampling SDF

---

## Phase 5 — Orbit Camera

State: `float azimuth, elevation, radius`. Target fixed at scene centroid (updated as roots grow).

Compute `eye` and `u_cam` matrix each frame:

```cpp
eye = target + radius * vec3(
    cos(elevation)*sin(azimuth),
    sin(elevation),
    cos(elevation)*cos(azimuth));
```

Input (only when ImGui image is hovered):
- **Left drag** → `azimuth += dx * sensitivity`, `elevation += dy * sensitivity`
- **Scroll** → `radius *= pow(0.9, scroll)`

Read input via `ImGui::GetIO()` — no raw GLFW callbacks needed.

---

## Phase 6 — Render Loop

```
each frame:
  1. simulate(dt * speed)               if playing and not paused
  2. if (ana.segments.size() changed):
       SegmentAnalyser → upload TBOs
  3. glBindFramebuffer(FBO)
       glViewport(0,0,w,h)
       use raymarcher shader
       set uniforms (camera, segCount)
       draw fullscreen quad (6 vertices, no VBO)
     glBindFramebuffer(0)
  4. ImGui::NewFrame()
       ImGui::Begin("Root System Viewer")
         ImGui::Image(colorTex, size)   ← raymarched output
         [▶ Play] [↺ Reset]   Speed ──●── 1.0x
         Day: 12.4   Segments: 847
       ImGui::End()
     ImGui::Render()
  5. glfwSwapBuffers
```

---

## Phase 7 — Polish (post-MVP)

- Incremental TBO update (append-only as roots grow)
- Ambient occlusion pass (short-range SDF samples)
- Ground plane with grid or soil texture
- Color roots by organ type (tap root vs laterals) using `GetGeometryNodeIds` or segment metadata
- Export frame as PNG

---

## Build Order

1. Shader strings compiling and FBO clearing (black window in ImGui)
2. Static root system loaded from XML, displayed as sphere-traced SDF
3. Simulation loop — watch it grow
4. Camera orbit
5. UI controls
