// RootSim — GL-free, incremental CPlantBox "mask relay" root growth.
//
// A faithful, restructured port of render_relay_gui.cpp's growth driver: masks
// are placed by phyllotaxis on a cone, and a root system grows hop-by-hop from
// mask to mask (travel toward the target, then dwell/wrap around it), confined by
// the cavity-avoidance geometry and steered by attraction tropism. The GL app's
// blocking per-hop while-loop is turned into a per-frame state machine so the
// Metal app can step it alongside rendering.
//
// CPlantBox types are hidden behind a pimpl so this header stays includable from
// ObjC++ (RootScene) without pulling Eigen/CPlantBox into the ObjC compile.
// Geometry and masks are returned in RENDER space (Y-up, the toYup remap applied).
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace rootsim {

struct SimParams {
    int   N            = 6;        // number of masks / hops
    float R0           = 13.0f;    // cone base radius
    float Hh           = 52.0f;    // cone height
    float startFrac    = 0.15f;
    float endFrac      = 0.94f;
    float taperPower   = 1.0f;
    float angleStepGoldenMult = 1.0f;
    float distStepFrac = 0.0f;
    float dwellDays    = 18.0f;
    float weight       = 0.9f;     // main-root travel attraction
    float mainTravelTrials = 14.0f;
    float lateralWeight = 0.20f;
    float dwellWeight        = 0.92f;
    float dwellLateralWeight = 0.92f;
    float sigma        = 0.35f;    // angular jitter
    float viewCylLen   = 8.0f;
    float maxHopDays   = 60.0f;
    float reachMult    = 1.6f;
    float travelPullReach = 1.2f;
    // "Along the surface" travel: confine the travelling root to a thin shell
    // straddling the cone the masks sit on, so it crawls over the cone between
    // masks instead of cutting through its interior. Travel only -- the dwell
    // wrapping stays unconstrained, or the nests around each mask would be
    // flattened onto the surface instead of bulging into 3D.
    bool  coneSurfaceTravel  = false;
    float coneShellThickness = 7.0f;   // cm; thicker = looser hug, more wander
    float growthDt     = 0.75f;    // sim days advanced per step()
    float targetLift   = 0.0f;
    float spawnBehind  = 0.0f;
    unsigned seed      = 42u;
    std::string paramDir;          // CPlantBox modelparameter dir (trailing slash)
    std::string speciesXml = "Zea_mays_6_Leitner_2014.xml";
};

// A revealed face mask in render (Y-up) space, for the face mid-geometry pass.
struct SimMask {
    float pos[3];
    float normal[3];
    float tangent[3];
    float bitangent[3];
    float rDepth, rWidth, rHeight;
};

class RootSim {
public:
    RootSim();
    ~RootSim();

    // (Re)start growth. Returns false if the parameter file cannot be loaded.
    bool reset(const SimParams& p);

    // Advance the live growth by one sim step (growthDt days). No-op once done.
    void step();

    bool valid() const;
    bool done()  const;

    // Current geometry in render space: nodesXYZ 3/node, segs 2/seg, radii 1/seg.
    void geometry(std::vector<float>& nodesXYZ,
                  std::vector<int>&   segs,
                  std::vector<float>& radii) const;

    // Masks revealed so far (render space), for the face pass.
    const std::vector<SimMask>& revealedMasks() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rootsim
