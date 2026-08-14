# mirror_app — controls

Every control in the panel, what it does, and why it exists. Grouped the way the
panel is grouped, so this reads top to bottom alongside it.

The panel is one window, `neuromirror — controls`. Sections that belong to a
scene only appear when that scene is selected; the rest are always there,
because the source and the tracker feed every scene.

> **Status: first pass.** The grouping below is a proposal as much as a
> description — several placements are arguable and are flagged inline with
> **[?]**. That is what to push back on.

---

## Parameter names, MIDI and presets

Every slider, checkbox and colour picker registers itself as it draws, under the
section it draws in. Its **path** is `section/label` — `mirror/ring freq`,
`roots/growth/taper`. That one name is what MIDI bindings and settings presets
are written in terms of, so:

- **Renaming a control or a section invalidates saved bindings and presets** for
  it. The loader reports names no control claimed rather than dropping them
  silently — see *settings → N key(s) no control claimed*.
- Two controls in the same section may not share a label. There is a headless
  check (`--uitest`) that would catch a collision, and one was found this way:
  face size and camera-overlay size were both `face tracking/size`, which would
  have made a preset apply one to both.

There is no separate list of parameters to maintain. The panel *is* the list.

### MIDI

**settings → open MIDI** connects every CoreMIDI source on the machine, and
rescans every two seconds so a controller plugged in mid-session is picked up.
Only control-change messages are read; notes and clock are ignored.

To bind: **right-click any control → MIDI learn**, then move a knob. The first
CC that arrives is bound to that control. Learn *consumes* that message rather
than acting on it — the knob is somewhere arbitrary when you bind it, and having
the parameter jump there is a surprise at exactly the wrong moment.

- A bound control shows `[cc<n>]` next to it.
- **settings → incoming** lists the last eight messages received. This is the
  difference between "nothing is bound" and "nothing is arriving".
- **settings → bindings** lists everything bound, with an `x` per row.
- The CC's 0–127 range maps onto the control's own slider range. For a checkbox,
  anything at or past halfway is on — which is also how a controller's button
  behaves when it sends 0/127.
- Colour pickers are not bindable: one CC cannot say three numbers, and quietly
  binding it to the red channel would be worse than not offering it.

### Settings presets

**settings → save / load**, written to `mirror_app/presets/<name>.set` as flat
`key = value` text. Holds every registered parameter *and* the MIDI map.

Two behaviours worth knowing:

- **Values apply when a control next draws.** Anything inside a collapsed header
  is applied the moment you open it, not at load. The panel says so.
- **Unknown keys are skipped, absent keys keep their current value.** A preset
  written before a control existed still loads afterwards.

Root *growth* parameters additionally have their own presets (`roots → presets`,
`.root` files). That split is deliberate: growth parameters are a recipe you
regrow from, while settings are the look of a running session. **[?]** They could
be folded together.

---

## Always visible

### scene

`mirror` · `roots` · `transition` · `fit view` · `cam mask`

Which scene renders. `fit view` and `cam mask` are diagnostic/authoring scenes:

- **fit view** — the fit's output, the training mask tinted over it, and the
  fitted mesh at its projected position wearing its sampled colours. Stacked
  flat in one coordinate space so anything out of register looks wrong rather
  than interesting. If the mesh's colours don't match the layer underneath it,
  the texture sampling is reading the wrong pixels.
- **cam mask** — the camera frame with the mask applied, and drag handles on the
  rectangle.

### camera mask

Keeps a rectangle of the sensor's view and blacks out the rest. The mirror only
sells if the frame holds the person and nothing that says *room* — a doorway, a
window, the edge of the rig.

Applied in **camera space, ahead of everything**: the fit never sees the masked
pixels and the tracker cannot find a face in them. It does not move with the
subject, because it is aimed at the room.

| control | what it does |
|---|---|
| `mask the camera` | on/off |
| `x`, `y` | the rectangle, as normalised ranges |
| `soft edge` | fade width at the border, as a fraction of the frame |
| `reset` | back to the default rectangle |

In the `cam mask` scene the same rectangle has four corner handles and a
draggable body.

### face tracking

The tracker feeds the mirror (crop + soft edge) and the roots (fitted mesh), so
it lives outside both.

| control | what it does |
|---|---|
| `track faces` | run MediaPipe at all |
| `hold on loss` | how long a detection survives after the tracker stops returning one |
| `acquire` | consecutive detections before a face is believed |

**Why the hysteresis.** MediaPipe drops frames on a blink, a turn, a hand across
the face. Treated as "nobody there", the fit target flips from a crop to the
whole frame and back — resizing the trained pixel set, rebuilding the feature
gather, and snapping the soft edge on and off. Several frames of upheaval to
report something already over. The status reads `held 0.24s` in amber during a
gap.

#### the crop

| control | what it does |
|---|---|
| `crop the fit to the face` | with a face tracked, train only on its crop; with none, fit the whole feed |
| `landmark box` / `hull` | the crop's shape |
| `pad` | grows the box by a fraction of its own size, so the margin scales with how close the person is |
| `dilate` | a fixed margin in fit-grid pixels on top |

Box is the default. The hull is tighter but supervises skin only — the network
never sees the boundary between a person and the room, so it has no reason to
draw one.

#### head movement

Three answers to "the subject moves and the network is a function of position",
none strictly better:

| mode | what it does |
|---|---|
| `centre it` | shift the frame so the head sits mid-frame and fit it there. Most stable weights; the mirror stops showing where in the room anyone is |
| `follow it` | fit the head where the camera found it. Honest, least stable |
| `shift the inputs` | fit it where it is, but offset the network's input coordinates by its displacement. The picture of *follow*, the weights of *centre* |

In `shift the inputs` the whole field translates with the head, background
included — the offset is on the coordinates, not on the subject.

| control | what it does |
|---|---|
| `set face size` + `size` | (centred only) resample so the head is a chosen fraction of the frame. Costs a bilinear resample; at 1:1 the centred mode only shifts by whole pixels, deliberately |
| `head smoothing` | how fast the tracked box follows the landmarks. 1 is raw |

#### overlay

A corner picture-in-picture of the raw source — the same image the tracker sees,
mirroring included. The mirror's own output cannot tell a closed sensor from a
stale frame from the photo still being selected; this can. Shows landmarks and
the crop rectangle, and captions with the source, frame count and whether the
fit is consuming it.

#### identity, and the rest

`secs` / `modes` / `ridge` / `frames` govern the morphable-model identity fit;
`head pose from tracker` rotates the mesh by MediaPipe's 4×4 rather than the
flat 2D similarity. `fitted mesh drives the root masks` and `texture the mask
from the neural fit` are handoffs to the root scene.

**[?]** Identity arguably belongs in its own top-level section rather than
nested under tracking.

---

### text overlay

`show text`, the text box, `size`, `x`, `y`, `inversion`, `text refraction`,
`edge softness`, `stroke weight`, `reveal`, `turbulence`, `turb scale`,
`turb drift`, `tracking`, `font`, `raster px`.

Always visible, because the text composites in the **present pass** rather than
inside a scene: it draws over the mirror, the roots and the transition alike.

The glyphs are CoreText outlines turned into a signed distance field, not a
bitmap and not something the network produces — the mirror's eight-feature basis
cannot hold letterforms, and pointing it at text would take it away from the face
it is tracking. The field is what keeps the text crisp at any `size`, and what
makes `stroke weight` a free parameter rather than a re-render.

The composite is an **inversion** of whatever is underneath. There is no colour
control because there is no colour: against a palette the network generates and
keeps changing, `1 - c` is the only choice that stays legible over all of it.

`text refraction` bends the text through the pond's own ripple gradient — the
same accumulation the feature kernel warps its colour coords with, so the text
moves with the water rather than floating over it. It has no effect in the root
scene, which has no ripples; the panel says so when that is the case. Past ~0.15
it starts to tear the letterforms open, which is an effect rather than a bug, but
legibility goes with it.

**`reveal` is the emerge/dissolve timeline** — 0 is gone, 1 is whole, and the
word comes apart turbulently in between. This is the one to put on a fader. The
three under it shape what the journey looks like and are set once: `turbulence`
0 is an even fade and 1 is fully broken up; `turb scale` is how fine the patches
are (low numbers swallow whole letters, high ones eat into the strokes);
`turb drift` moves them. The noise is sampled at the same refracted coordinate
the text is, so with ripples running the dissolve flows with the water at no
extra cost.

It works by giving every pixel a threshold from the noise and revealing it when
`reveal` passes — not by eroding the outline. Erosion is bounded by how far the
distance field encodes, so thick stems would sit untouched and then pop.

`edge softness` is antialiasing, **not a glow**. The field only carries distance
out to its spread, so the shader clamps the ramp there; past that it would run
off the end of what is represented and leave a hard-edged halo tracing the text
rather than a soft edge. Useful range is about 0.5–3.

The last three rebuild the field rather than moving a uniform, so they are the
setup controls and the ones above them are the performance controls. `raster px`
is quality-of-outline, not size on screen: the distance transform is exact with
respect to the bitmap it is handed, so this sets how faithful the curve is.

The text and font strings are **not** saved in presets — the registry is numeric.

---

## mirror

### ripples

`ring freq`, `ripple decay`, `ripple speed`, `ripple phase`, `refraction (warp)`,
`raindrops`, `moving ripple`, `soft centers` + `radius`.

A decorative field. Off by default (`raindrops` 0) because they dominate the
MLP's input features, which is wrong once the network is being *fitted* to
something — during training they are signal the target does not contain.

### LIVE / the fit

| control | what it does |
|---|---|
| `open sensor` | only one process can hold the Kinect |
| `track live feed` | arm the camera feed. **Does not start training** — it turns on the frame pull, the tracker and the preview |
| `fit` / `stop` | start (or restart) training. With the feed armed, on the frame the camera is showing right now, cropped the way the per-frame retarget will crop it |
| `clear fit` | back to the generated field |
| `mirror image` | a mirror should put your left hand on your left; the sensor does not |

**Two tunings**, switched automatically by whether a crop is active, the live one
marked in green:

| | `grid` | `steps` | `lr` |
|---|---|---|---|
| `face crop` | 1 | 4 | 2e-3 |
| `whole feed` | 3 | 1 | 3e-3 |

A crop is a few percent of the pixels, so a step costs a few percent as much and
many more fit in a frame — but each gradient sees far less data, so a smaller
step keeps it off the crop box's own jitter. The whole feed is the opposite. One
shared set of numbers meant every crop/no-crop transition quietly changed what
they meant. **The defaults are reasoned, not measured.**

Changing `grid` mid-fit is safe: the weights are continuous and the optimiser
state is per-weight, so a regrid leaves the image bit-identical and keeps the
step count.

### outside the crop

Inside the crop the network is reproducing a person and every input must hold
still. Outside, nothing is constrained. These are what that difference is
allowed to look like.

| control | what it does |
|---|---|
| `soft edge` | fade the effects below across a band instead of switching at the border |
| `follow the outline` | fade outward from the mask's actual shape (a face outline when the crop is a hull) rather than its bounding box |
| `fade starts` | how far out the fade begins, in coord units (the frame is 2 tall) |
| `fade width` | how far it runs |
| `animate z outside` + `z rate -s` | let the latent move everywhere except on the subject |
| `grey outside` | drain colour outside the crop; 1 leaves the subject in colour on a greyscale field |

**Why start and width are separate.** "Where the gradient sits" and "how long it
takes" are separate complaints — a fade pinned to the crop's edge pools against
it however wide you make it.

**Why `follow the outline` also fixes pooling.** The box form measures distance
as a fraction of *each* half-extent, so a tall crop fades over a longer distance
vertically than horizontally and flares at the corners. The outline form uses a
Euclidean distance field, so the band is one width all the way round.

`animate z outside` is not a colour tweak: z is an MLP *input*, so moving it
under a fitted network asks the same weights a different question and the face
comes apart. Inside the crop it is pinned to the value the fit was begun at.
With the split off, `z` stays an ordinary global control.

### network

`sine layers`, `sine w0`, `detail`, `gain tilt`, `w shape`, `contrast`,
`reseed network`.

Once the network is fitted these stop applying — the weights were learned, not
derived from them.

### colour & tone

`sRGB fix`, `gamma`, `color mix`, `grey ch`, `ripple amp -> color` + `amp gain`,
`swap R-B`, `color travel`.

### z latent

`z`, `z amplitude`, `z auto-rate -s`, `z step size`, and step buttons.

### clock & render

`ripple time scale`, `pause`, `downscale` (mirror render-resolution divisor).

### mask emergence (transition)

The screen-space hydro-dip: `transition`, `auto-play` + `play rate`, `relief
height`, `mask width`/`height`, `light azimuth`/`elevation`, `wet sheen`,
`sheen tightness`, `background dim`.

**[?]** This overlaps confusingly with the `transition` *scene*, which is a
different (3D cloth) effect with a similar name.

---

## roots

### growth

`species` picks the CPlantBox parameter set. The rest shape the mask relay:
masks are placed by phyllotaxis on a cone and a root system grows hop by hop
from mask to mask, travelling then dwelling/wrapping.

| control | what it does |
|---|---|
| `masks` | number of masks / hops |
| `cone radius`, `cone height`, `taper` | the cone the masks sit on |
| `spiral x golden` | multiplier on the golden angle; 1.0 is the classic non-repeating spiral |
| `jitter` | angular randomness — looseness independent of the pull weights |
| `travel pull`, `pull reach` | how insistently the main root homes on the next mask |
| `lateral` | offshoot attraction during travel, kept low so laterals dangle rather than converge |
| `dwell`, `dwell days` | wrapping around a mask once reached |
| `hop days` | budget per hop, scaled by how far it has to travel |
| `crawl the cone surface` + `shell` | confine the travelling root to a shell around the cone, so it crawls over the surface instead of cutting through |
| `days - step`, `steps-frame` | playback speed; read live, no regrow needed |
| `regrow` / `reseed` | apply structural changes / new random seed |

`reseed` reseeds the growth. It used to drop a synthetic stand-in structure over
a running grow which the next frame overwrote; the stand-in now only appears
when CPlantBox's parameter files are absent.

### presets

Growth parameters to and from `.root` files.

### camera

| control | what it does |
|---|---|
| `frame automatically` | derive target and distance from the scene's own bounds |
| `focus` | whole scene, or centre on one revealed mask and frame tight to it |
| `zoom` | multiplier on the framed distance |
| `auto-orbit`, `orbit rate` | a focused mask orbits on its own angle — borrowing the whole-scene arc would sweep it out of frame |
| `radius` | manual distance (disabled while auto-framing) |
| `azimuth`, `elevation`, `fov` | |

### material / fog & atmosphere / travelling pulses / face masks

Renderer knobs, ported from the GL reference. `face masks` also carries
`show faces` and `face scale` for the placed masks.

### cached field: LOD & culling

A tiled field of cached root instances, for profiling the culling and LOD path.
`tile field` / `clear field`.

### overlays

`axes`, `grid`, `grid spacing`.

---

## Headless modes

Not controls, but the same surface from the command line:

| flag | what it checks |
|---|---|
| `--selftest` | the MLX→Metal mirror path |
| `--uitest` | the parameter registry: declaration, MIDI routing, preset round-trip |
| `--fitviewtest` | the fit view's three layers render |
| `--maskframes` | the root masks' orientation, as numbers |
| `--presettest` | growth-parameter round-trip |
| `--growshot` | render the grown root system to a PPM |
| `--textshot` | the text overlay over a live pond, through the real present pass |
| `--roottest`, `--rootshot`, `--fieldshot`, `--maskshot`, `--transhot`, `--mirrorclip` | older shot/bench modes |
