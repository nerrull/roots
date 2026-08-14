// TransitionScene — the pond -> face hydro-dip transition.
//
// A port of neuromirror/cloth_cpp, which prototyped the whole effect against
// static assets. Four phases on one timeline, one locked front-on camera (the
// face never moves):
//
//   1 hold     the flat neural pond fills the frame
//   2 emerge   the face rises, its normals refracting and embossing the pond
//   3 swap     the flat pond becomes a 3D cloth over the *same screen pixels*
//   4 fall     the cloth sim runs and the sheet falls away behind the face
//
// The art is that phases 1-2 (a flat screen-space pass) and 3-4 (real 3D) meet
// at the swap with no visible discontinuity. Three things have to match on that
// frame -- brightness, face shading, and texture content -- and the notes in
// transition.metal say how each is held.
//
// ## What is improved over the prototype
//
// cloth_cpp ran on two baked assets: `pond.ppm`, one saved PondState frame, and
// `face.bin`, the neutral ICT mask flattened to a height+coverage grid. Both
// were static, and its own TODO list led with the consequences. The combined app
// has the live versions of both, so:
//
//   * **The pond is live.** The texture is MirrorScene's actual output, so the
//     film the sheet is skinned with is the network's current frame -- and, when
//     the mirror has been fitted, the person's own face. (cloth_cpp TODO: "live
//     pond and face".)
//
//   * **The face is the real fitted mesh.** Maxine/NVF topology, this person's
//     identity, this frame's expression and head pose, straight from FaceFitter
//     -- not a displaced grid of a baked neutral. This is what fixes the two
//     artifacts the prototype documented and could not fix: the seam line around
//     the silhouette and the boxy neck edge were both properties of the
//     heightmap *shell*. Real geometry is watertight, so they cannot occur.
//     (cloth_cpp TODOs: "seam line", "real ICT mesh", "head pose".)
//
//   * **The relief G-buffer is rendered, not baked.** `f_relief` rasterises the
//     fitted mesh into the same (normal.xy, height, coverage) layout face.bin
//     had, every frame. The emerge pass is unchanged -- same consumer, live
//     producer -- and it now tracks expression and pose for free.
//
//   * **The rim follows the face.** cloth_cpp pinned the cloth to a fixed
//     ellipse because its face never moved. A fitted face is a different shape
//     per person, so the pin ring is derived from the mesh silhouette
//     (`Cloth::pinTo`).
//
//   * **Refraction is velocity-driven** rather than a single constant, so the
//     distortion peaks while the face is actually moving through the film and
//     not merely at the midpoint of the timeline. (cloth_cpp TODO.)
//
//   * **The swap is crossfaded** over a few frames instead of being a hard cut,
//     which hides any residual mismatch. (cloth_cpp TODO: "swap softening".)
//
//   * **Timing is data-driven** -- the phase durations are fields, not
//     hard-coded constants, so they can be matched to audio or a live cue.
//     (cloth_cpp TODO.)
//
// Deliberately NOT ported: cloth self-collision and sheet<->face collision. Both
// were listed as expensive and unnecessary while the sheet is pinned behind the
// face, and both still are.
#pragma once
#ifndef __OBJC__
#error "transition_scene.h is ObjC++ only"
#endif

#import <Metal/Metal.h>

#include <memory>
#include <string>
#include <vector>

#include "cloth.h"

class MetalContext;

class TransitionScene {
public:
    TransitionScene(const MetalContext& ctx, int w = 1280, int h = 720);
    ~TransitionScene();

    bool valid() const;
    void ensureSize(int w, int h);
    int  width() const;
    int  height() const;

    // Phase durations in seconds. Data-driven so the effect can be matched to a
    // cue rather than recompiled.
    struct Timing {
        float hold   = 0.4f;    // flat pond before anything happens
        float emerge = 1.6f;    // face rising through the film
        float settle = 0.3f;    // fully emerged, before the sheet lets go
        float fade   = 0.15f;   // crossfade across the swap; the sheet is
                                // held flat for its duration
    };
    Timing timing;

    // Look.
    float refract      = 0.06f;  // base refraction strength
    float refractVel   = 0.5f;   // extra refraction proportional to d(emergence)/dt
    float faceScale    = 1.0f;
    bool  showCloth    = true;
    bool  wireframe    = false;

    // Cloth knobs (passed through to the solver).
    // Slower than the prototype's 9.8: at this scale (the sheet half-extent is
    // ~1.24 world units) 9.8 puts the sheet behind the camera in well under a
    // second, so the drape never reads.
    float gravityBack  = 3.5f;   // -z, behind the mask
    float gravityDown  = 0.3f;   // -y
    int   substeps     = 2;
    int   iterations   = 24;

    // The face the transition reveals. `verts` is 3 floats/vertex in the
    // fitter's model units and `tris` its topology; passing an empty `tris`
    // keeps the previous one. Without a mesh the scene falls back to the
    // prototype's behaviour and shows the pond alone.
    void setFaceMesh(const std::vector<float>& verts, const std::vector<int>& tris);
    bool hasFace() const;

    // The pond film. This is MirrorScene's own output texture -- the sheet is
    // skinned with whatever the network is currently rendering.
    void setPondTexture(id<MTLTexture> pond);

    // Timeline control.
    void restart();
    void advance(double dt);
    double clock() const;
    float  emergence() const;    // 0..1
    bool   done() const;
    const char* phaseName() const;

    // Encode the whole effect into `cb` and return the colour texture.
    id<MTLTexture> render(id<MTLCommandBuffer> cb);

    const Cloth& cloth() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
