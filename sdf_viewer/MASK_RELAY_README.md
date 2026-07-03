# Mask Relay Viewer

Interactive viewer for the mask-cavity/attractor root system: a phyllotactic
spiral of face-mask cavities on a cone, with a fresh root system "relayed"
from mask to mask (rather than one plant growing the whole journey, which
front-loads density near its origin and can blow up exponentially for some
species).

## Running

In VS Code: **Terminal > Run Task > Run: Mask Relay Viewer** (builds first if
needed). There's also a windowed (non-fullscreen) variant if you want the
window to coexist with other apps on screen.

From the command line:

```sh
cmake -S . -B build
cmake --build build --target mask_relay_gui -j4
./build/sdf_viewer/mask_relay_gui \
    CPlantBox/modelparameter/structural/rootsystem/   # param file directory
    1920 1080   # width height (ignored in fullscreen)
    1           # fullscreen: 1 = windowed fullscreen (default), 0 = windowed
```

## Controls

- **Regrow**: reruns the entire mask sequence from scratch with the panel's
  current values. Growth parameters (masks, spacing, dwell, attraction,
  thickness range) only take effect on the next Regrow.
- **Lighting & Material**: root PBR material, key light direction, mask
  marble material (base/vein color + procedural veining), per-face point
  light, fog. All of these apply live -- no regrow needed.
- **Camera**: pick a specific mask to focus the view on once growth finishes,
  or "Whole scene" for the default framing.
- **Invert mode**: black background, white roots, overlapping capsules XOR
  the pixel color instead of just the nearest one winning -- a distinct
  graphic/op-art look, toggles live.

ESC or close the window to quit.

## Known limits

- **Performance** scales with total segment count (which grows with mask
  count × dwell time × species branching habit) and, more significantly, with
  overdraw from many overlapping instanced capsule quads in a dense wrapped
  mass -- there's no depth pre-pass or spatial acceleration structure (macOS
  caps at OpenGL 4.1, so no compute shaders to build one properly). If it gets
  heavy: fewer masks, shorter dwell, or a less aggressively-branching species
  (maize/soybean/pea/sunflower are the ones that have tested well; lupin and
  pimpernel don't -- see commit history for why).
- Per-face point lights feeding the root shader are capped to the 6
  most-recently-revealed masks regardless of total mask count, both for
  performance and because far-away masks don't meaningfully light nearby
  roots anyway.
- `render_relay_gl.cpp`, `render_relay_live.cpp`, and
  `render_mask_column_gl.cpp` are earlier, non-interactive iterations of this
  same idea (batch video export, and a plain live viewer without the ImGui
  panel). They share `face.vert`/`face.frag` with this tool but weren't kept
  in sync with its later shader changes (marble material, per-face lights) --
  `render_relay_gui.cpp` is the one actively maintained.
