// RootScene — the 3D lit root scene. Owns a MetalRootRenderer, an orbit camera,
// and (for now) a procedurally generated branching structure standing in for the
// live CPlantBox growth (Task: sdfsim wiring). advance() spins the camera and
// drives the fog/pulse/wisp clocks; render() encodes the two Metal passes into the
// caller's command buffer and returns the fogged colour texture to present.
#pragma once
#ifndef __OBJC__
#error "root_scene.h is ObjC++ only"
#endif

#import <Metal/Metal.h>
#include "metal_root_renderer.h"
#include "root_sim.h"
#include "root_sim.h"

#include <memory>
#include <string>
#include <vector>

class MetalContext;

class RootScene {
public:
    explicit RootScene(const MetalContext& ctx, int w = 1280, int h = 720);

    bool valid() const { return rr_ && rr_->valid(); }

    void ensureSize(int w, int h);
    void advance(double dt);
    id<MTLTexture> render(id<MTLCommandBuffer> cb);

    double clock() const { return t_; }
    int    width()  const { return rr_ ? rr_->width() : 0; }
    int    height() const { return rr_ ? rr_->height() : 0; }

    MetalRootRenderer& renderer() { return *rr_; }

    // A new random seed for the growth. Reseeds the *simulation* when one is
    // running -- the synthetic branching structure below is a fallback for when
    // CPlantBox's parameter files are missing, not something to drop on top of
    // a live grow.
    void reseed(uint32_t seed);

    // Rebuild + re-upload the face mask mesh (call after changing faceScale /
    // showFace / face placement).
    void rebuildFace();

    // --- live face fitting ---------------------------------------------------
    // Replace the static canonical_face_model.obj with a mesh fitted to whoever
    // is in front of the sensor (see face_fit.h). `verts` is 3 floats/vertex in
    // the fitter's model units; `tris` is its topology, and may be passed empty
    // on subsequent calls to keep the previous one.
    //
    // The normalisation is captured once, from the first mesh seen, and reused:
    // re-normalising per frame would rescale the mask every time the person
    // opened their mouth, since an expression changes the mesh's extent. So the
    // mask holds still and the face moves inside it, which is the intent.
    void setFittedFace(const std::vector<float>& verts, const std::vector<int>& tris);
    void clearFittedFace();
    bool usingFittedFace() const { return fitted_face_; }

    // Per-vertex RGB for the fitted mesh (3 floats/vertex, [0,1]), sampled from
    // the neural mirror by FaceFitter::sampleTexture. Empty reverts the mask to
    // its flat material colour.
    //
    // This is deliberately a *stored* colour rather than a live texture lookup.
    // The mirror and the roots never run at the same time -- only one sim runs
    // at a time by design -- so by the time the root scene is drawing, the
    // mirror has stopped and there is no live neural texture left to sample.
    // Capturing it at the handoff is what lets the mask keep the face.
    void setFaceColors(const std::vector<float>& rgb);
    bool hasFaceColors() const { return !faceColors_.empty(); }

    // --- camera ------------------------------------------------------------
    // Frame the scene from its own bounds rather than from constants: the cone
    // is sized by R0/Hh, so any change to those left hardcoded framing pointing
    // at the wrong part of it. `focusMask` >= 0 centres on one revealed mask
    // and frames tight to its radius, orbiting on its own angle -- reusing the
    // whole-scene arc would swing a mask out of frame, since that arc is
    // centred on the piece and not on the mask.
    bool  autoFrame = true;
    int   focusMask = -1;      // -1 = whole scene
    float zoom      = 1.0f;
    int   maskCount() const { return (int)revealedMasks().size(); }

    // Restart the live CPlantBox growth (no-op if the sim failed to load).
    void regrow();
    // The growth parameters, editable in place; call regrow() to apply. Held
    // here rather than rebuilt at each call site so a species change and a
    // seed change go through the same door.
    rootsim::SimParams& simParams() { return simParams_; }
    const rootsim::SimParams& simParams() const { return simParams_; }
    // Species available to the sim: display name and parameter file, mirroring
    // the reference GUI's list.
    static const std::vector<std::pair<std::string, std::string>>& species();
    int  speciesIndex() const;
    void setSpeciesIndex(int i);

    // --- presets -----------------------------------------------------------
    // Growth parameters to and from a flat key=value file. Deliberately not a
    // binary blob or a versioned schema: an unknown key is skipped and a
    // missing one keeps its default, so a preset written before a parameter
    // existed still loads afterwards.
    bool saveConfig(const std::string& path) const;
    bool loadConfig(const std::string& path);
    static std::string presetDir();
    static std::vector<std::string> listPresets();
    bool simActive() const { return useSim_; }
    // The revealed masks, in render space. Exposed so the frames the faces are
    // placed on can be checked as numbers -- their orientation is not reliably
    // readable off a render at the scale they occupy.
    const std::vector<rootsim::SimMask>& revealedMasks() const;
    bool simDone()   const;

    // Grow one system, then cache it and tile a gridN x gridN field of instances
    // (varied yaw/scale) to exercise LOD + frustum culling. Stops the live path.
    void buildField(int gridN, float spacing);

    bool  showFace  = true;
    float faceScale = 0.85f;
    int   simStepsPerFrame = 2;   // growth steps advanced per rendered frame

    // Camera / lighting (mirrors mask_relay_gui's controls).
    float azimuth   = 0.6f;
    float elevation = 0.35f;
    float radius    = 42.0f;
    float fov       = 0.6f;
    float target[3] = {0.f, 8.f, 0.f};
    float lightDir[3] = {0.4f, 0.8f, 0.35f};
    bool  autoOrbit = true;
    float orbitRate = 0.15f;   // rad/s

private:
    void buildSyntheticRoots(uint32_t seed);
    void uploadFaceFromMasks();      // build face verts from the live sim's masks

    std::unique_ptr<MetalRootRenderer> rr_;
    std::unique_ptr<rootsim::RootSim>  sim_;
    rootsim::SimParams simParams_;
    bool useSim_ = false;
    // Whether CPlantBox is usable at all. Distinct from useSim_, which also
    // goes false when a cached instance field takes over the renderer -- and
    // reseeding after that must not silently fall back to the stand-in.
    bool simAvailable_ = false;
    std::vector<float> faceVerts_;   // face model, local, normalised (3/vert)
    std::vector<int>   faceTris_;    // triangle indices
    // The canonical model, kept so clearFittedFace() can go back to it without
    // re-reading the .obj.
    std::vector<float> canonVerts_;
    float idleCentre_[3] = {0.f, 0.f, 0.f};
    float idleExtent_ = 10.f;
    float focusAngle_ = 0.f;
    void  updateBounds(const std::vector<float>& nodes);
    void  applyFraming();
    std::vector<int>   canonTris_;
    std::vector<float> faceColors_;   // per-vertex RGB, empty = flat material
    bool  fitted_face_ = false;
    bool  fit_norm_set_ = false;     // normalisation captured from the first fit
    float fit_centre_[3] = {0, 0, 0};
    float fit_scale_ = 1.0f;
    double t_ = 0.0;
};
