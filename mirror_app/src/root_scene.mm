#include "root_scene.h"
#include "metal_context.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <map>
#include <vector>

// ---------------------------------------------------------------------------
// Face model loading + placement (ports of render_relay_gui.cpp helpers, but in
// render world space directly — no CPlantBox toYup remap needed here).
// ---------------------------------------------------------------------------
namespace {
struct F3 { float x, y, z; };
inline F3 sub(F3 a, F3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline F3 add(F3 a, F3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline F3 mul(F3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
inline float dot(F3 a, F3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline F3 cross(F3 a, F3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
inline F3 norm(F3 v) { float l = std::sqrt(dot(v,v)); if (l < 1e-9f) l = 1.f; return {v.x/l, v.y/l, v.z/l}; }

void loadObj(const std::string& path, std::vector<float>& verts, std::vector<int>& tris) {
    std::ifstream f(path);
    if (!f.is_open()) return;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tag; ss >> tag;
        if (tag == "v") {
            float x, y, z; ss >> x >> y >> z;
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (tag == "f") {
            int idx[3]; std::string tok;
            for (int i = 0; i < 3 && ss >> tok; i++) idx[i] = std::atoi(tok.c_str()) - 1;
            tris.push_back(idx[0]); tris.push_back(idx[1]); tris.push_back(idx[2]);
        }
    }
}

void normalizeMesh(std::vector<float>& v) {
    float cx = 0, cy = 0, cz = 0;
    size_t n = v.size() / 3;
    if (n == 0) return;
    for (size_t i = 0; i < n; i++) { cx += v[i*3]; cy += v[i*3+1]; cz += v[i*3+2]; }
    cx /= n; cy /= n; cz /= n;
    float m = 1e-9f;
    for (size_t i = 0; i < n; i++) {
        v[i*3] -= cx; v[i*3+1] -= cy; v[i*3+2] -= cz;
        m = std::max({m, std::fabs(v[i*3]), std::fabs(v[i*3+1]), std::fabs(v[i*3+2])});
    }
    for (auto& x : v) x /= m;
}

std::vector<int> cropOvalTris(const std::vector<float>& v, const std::vector<int>& tris,
                              float rx, float ry) {
    std::vector<int> out;
    for (size_t t = 0; t < tris.size(); t += 3) {
        float cx = 0, cy = 0;
        for (int k = 0; k < 3; k++) { cx += v[tris[t+k]*3]; cy += v[tris[t+k]*3+1]; }
        cx /= 3; cy /= 3;
        if ((cx/rx)*(cx/rx) + (cy/ry)*(cy/ry) < 1.0f)
            for (int k = 0; k < 3; k++) out.push_back(tris[t+k]);
    }
    return out;
}

// One placed mask (render world space; Y-up).
struct Mask { F3 pos, normal, tangent, bitangent; float rDepth, rWidth, rHeight; };

// Build interleaved face triangles (12 floats/vert) for a set of masks.
// `vcol`, when non-empty, is a per-vertex RGB (3 floats/vertex, same indexing as
// `fv`) that overrides the flat `color` -- this is how a mask wears a sampled
// face rather than a material tint.
void appendFaceVertexData(std::vector<float>& out, const Mask& m,
                          const std::vector<float>& fv, const std::vector<int>& ftris,
                          float faceScale, float lightDist, const float color[3],
                          const std::vector<float>& vcol = {}) {
    F3 n = m.normal, t = m.tangent, b = m.bitangent;
    F3 p = sub(m.pos, mul(n, m.rDepth * 0.5f));
    float scale = faceScale * std::min(m.rWidth, m.rHeight);
    F3 lightPos = add(p, mul(n, lightDist));
    for (size_t i = 0; i < ftris.size(); i += 3) {
        F3 v3[3];
        for (int k = 0; k < 3; k++) {
            int vi = ftris[i+k];
            F3 local = {fv[vi*3], fv[vi*3+1], fv[vi*3+2]};
            v3[k] = add(add(add(p, mul(t, local.x * scale)), mul(b, local.y * scale)),
                        mul(n, local.z * scale));
        }
        F3 fn = norm(cross(sub(v3[1], v3[0]), sub(v3[2], v3[0])));
        for (int k = 0; k < 3; k++) {
            const int vi = ftris[i+k];
            const bool haveCol = !vcol.empty() && size_t(vi) * 3 + 2 < vcol.size();
            out.push_back(v3[k].x); out.push_back(v3[k].y); out.push_back(v3[k].z);
            out.push_back(fn.x); out.push_back(fn.y); out.push_back(fn.z);
            if (haveCol) {
                out.push_back(vcol[vi*3]); out.push_back(vcol[vi*3+1]);
                out.push_back(vcol[vi*3+2]);
            } else {
                out.push_back(color[0]); out.push_back(color[1]); out.push_back(color[2]);
            }
            out.push_back(lightPos.x); out.push_back(lightPos.y); out.push_back(lightPos.z);
        }
    }
}

// Right/up basis for a mask facing `n`, using world up (0,1,0).
Mask makeMask(F3 pos, F3 n, float r) {
    n = norm(n);
    F3 up = {0, 1, 0};
    F3 t = cross(up, n);
    if (dot(t, t) < 1e-6f) t = (F3){1, 0, 0};
    t = norm(t);
    F3 bit = norm(cross(n, t));
    return {pos, n, t, bit, r, r, r};
}
}  // namespace

RootScene::RootScene(const MetalContext& ctx, int w, int h) {
    const std::string shaderDir    = std::string(MIRROR_APP_SHADER_DIR);
    const std::string sharedHeader = std::string(MIRROR_APP_SRC_DIR) + "/root_shared.h";
    rr_ = std::make_unique<MetalRootRenderer>(ctx, shaderDir, sharedHeader, w, h);
    if (!rr_->valid()) return;

    // A warm, mottled root material with soft fog and a couple of drifting wisps,
    // echoing mask_relay_gui's default look.
    rr_->mat.baseColor[0] = 0.55f; rr_->mat.baseColor[1] = 0.42f; rr_->mat.baseColor[2] = 0.28f;
    rr_->mat.baseColor2[0] = 0.28f; rr_->mat.baseColor2[1] = 0.18f; rr_->mat.baseColor2[2] = 0.12f;
    rr_->mat.colorNoiseStrength = 0.6f;
    rr_->mat.colorNoiseScale = 0.35f;
    rr_->mat.ambient = 0.06f;
    rr_->mat.diffuse = 0.55f;
    rr_->fog.density = 0.012f;
    rr_->fog.noiseStrength = 0.6f;
    rr_->fog.refDist = radius;

    rr_->wispCount = 2;
    rr_->wisps[0].basePos[0] = 6.f;  rr_->wisps[0].basePos[1] = 14.f; rr_->wisps[0].basePos[2] = 4.f;
    rr_->wisps[0].color[0] = 1.0f; rr_->wisps[0].color[1] = 0.85f; rr_->wisps[0].color[2] = 0.5f;
    rr_->wisps[1].basePos[0] = -7.f; rr_->wisps[1].basePos[1] = 6.f;  rr_->wisps[1].basePos[2] = -3.f;
    rr_->wisps[1].color[0] = 0.6f; rr_->wisps[1].color[1] = 0.8f; rr_->wisps[1].color[2] = 1.0f;

    rr_->pulse.enabled = true;

    // Canonical face model (shared with the GL sdf_viewer assets).
    loadObj(std::string(SDF_VIEWER_DIR) + "/assets/canonical_face_model.obj",
            faceVerts_, faceTris_);
    normalizeMesh(faceVerts_);
    faceTris_ = cropOvalTris(faceVerts_, faceTris_, 0.72f, 0.98f);
    canonVerts_ = faceVerts_;
    canonTris_  = faceTris_;

    // Try the live CPlantBox growth; fall back to the procedural stand-in if the
    // parameter files aren't present.
    sim_ = std::make_unique<rootsim::RootSim>();
    simParams_.N = 5;
    regrow();
    if (useSim_) {
        // Framing is derived from the geometry's own bounds each frame (see
        // applyFraming); the constants that used to live here were tuned to one
        // particular R0/Hh and pointed at the wrong part of any other cone.
        rr_->radiusScale = 1.4f;
        rr_->radiusMin = 0.03f;
        rr_->radiusMax = 0.35f;
        // The masks are small; soften the per-face light so they read as faces
        // rather than blown-out blobs.
        rr_->face.lightIntensity = 1.8f;
        rr_->face.lightFalloff   = 0.05f;
    } else {
        buildSyntheticRoots(1u);
        rebuildFace();
    }
}

const std::vector<std::pair<std::string, std::string>>& RootScene::species() {
    // The reference GUI's list, verbatim -- these are the parameter sets that
    // are known to grow rather than every file in the directory.
    static const std::vector<std::pair<std::string, std::string>> kSpecies = {
        {"Maize (Zea mays)",      "Zea_mays_6_Leitner_2014.xml"},
        {"Soybean (Glycine max)", "Glycine_max.xml"},
        {"Pea (Pisum sativum)",   "Pisum_sativum_a_Pag\xc3\xa8s_2014.xml"},
        {"Sunflower (Heliantus)", "Heliantus_Pages_2013.xml"},
        {"Kale (Brassica)",       "Brassica_oleracea_Vansteenkiste_2014.xml"},
        {"Wheat (Triticum)",      "Triticum_aestivum_a_Bingham_2011.xml"},
        {"Lupin (Lupinus)",       "Lupinus_albus_Leitner_2014.xml"},
        {"Pimpernel (Anagallis)", "Anagallis_femina_Leitner_2010.xml"},
    };
    return kSpecies;
}

int RootScene::speciesIndex() const {
    const auto& sp = species();
    for (size_t i = 0; i < sp.size(); ++i)
        if (sp[i].second == simParams_.speciesXml) return (int)i;
    return -1;
}

void RootScene::setSpeciesIndex(int i) {
    const auto& sp = species();
    if (i >= 0 && i < (int)sp.size()) simParams_.speciesXml = sp[size_t(i)].second;
}

// One list, walked by both the writer and the reader, so a parameter cannot be
// saved and then not loaded (or the reverse) -- which is the failure a preset
// system has that nobody notices until a preset comes back subtly different.
template <class Fn>
static void visitSimParams(rootsim::SimParams& p, Fn&& f) {
    f("speciesXml", p.speciesXml);
    f("N", p.N);
    f("R0", p.R0);              f("Hh", p.Hh);
    f("startFrac", p.startFrac); f("endFrac", p.endFrac);
    f("taperPower", p.taperPower);
    f("angleStepGoldenMult", p.angleStepGoldenMult);
    f("distStepFrac", p.distStepFrac);
    f("dwellDays", p.dwellDays);
    f("weight", p.weight);      f("mainTravelTrials", p.mainTravelTrials);
    f("lateralWeight", p.lateralWeight);
    f("dwellWeight", p.dwellWeight);
    f("dwellLateralWeight", p.dwellLateralWeight);
    f("sigma", p.sigma);        f("viewCylLen", p.viewCylLen);
    f("maxHopDays", p.maxHopDays); f("reachMult", p.reachMult);
    f("travelPullReach", p.travelPullReach);
    f("coneSurfaceTravel", p.coneSurfaceTravel);
    f("coneShellThickness", p.coneShellThickness);
    f("growthDt", p.growthDt);
    f("targetLift", p.targetLift); f("spawnBehind", p.spawnBehind);
    f("seed", p.seed);
}

std::string RootScene::presetDir() {
    return std::string(MIRROR_APP_SRC_DIR) + "/../presets";
}

std::vector<std::string> RootScene::listPresets() {
    std::vector<std::string> out;
    DIR* d = opendir(presetDir().c_str());
    if (!d) return out;
    while (struct dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".root") == 0)
            out.push_back(n.substr(0, n.size() - 5));
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

bool RootScene::saveConfig(const std::string& path) const {
    mkdir(presetDir().c_str(), 0755);
    std::ofstream f(path);
    if (!f.is_open()) return false;
    rootsim::SimParams copy = simParams_;
    visitSimParams(copy, [&](const char* k, auto& v) { f << k << " = " << v << "\n"; });
    return true;
}

bool RootScene::loadConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(f, line)) {
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        auto trim = [](std::string& s) {
            const size_t a = s.find_first_not_of(" \t\r");
            const size_t b = s.find_last_not_of(" \t\r");
            s = (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
        };
        trim(k); trim(v);
        if (!k.empty()) kv[k] = v;
    }
    visitSimParams(simParams_, [&](const char* k, auto& v) {
        auto it = kv.find(k);
        if (it == kv.end()) return;              // absent: keep the default
        std::istringstream ss(it->second);
        ss >> v;
    });
    return true;
}

void RootScene::regrow() {
    if (!sim_) return;
    simParams_.paramDir = ROOTSIM_PARAM_DIR;
    useSim_ = sim_->reset(simParams_);
    simAvailable_ = simAvailable_ || useSim_;
}

bool RootScene::simDone() const { return sim_ && sim_->done(); }

const std::vector<rootsim::SimMask>& RootScene::revealedMasks() const {
    static const std::vector<rootsim::SimMask> kNone;
    return sim_ ? sim_->revealedMasks() : kNone;
}

void RootScene::buildField(int gridN, float spacing) {
    if (!rr_ || !sim_) return;
    // Ensure the template system is fully grown.
    for (int i = 0; i < 6000 && !sim_->done(); ++i) sim_->step();
    std::vector<float> nodes, radii; std::vector<int> segs;
    sim_->geometry(nodes, segs, radii);
    if (nodes.empty() || segs.empty()) return;

    rr_->clearInstances();
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    const float half = (gridN - 1) * 0.5f;
    for (int gz = 0; gz < gridN; gz++)
        for (int gx = 0; gx < gridN; gx++) {
            MetalRootRenderer::InstancePlacement pl;
            pl.translate[0] = (gx - half) * spacing;
            pl.translate[1] = 0.f;
            pl.translate[2] = (gz - half) * spacing;
            pl.rotYaw = U(rng) * 6.2831853f;
            pl.scale  = 0.8f + 0.4f * U(rng);
            rr_->addInstance(nodes, segs, radii, pl);
        }
    // The field is the cached instances; stop the single live system + face pass.
    rr_->uploadSegments({}, {}, {});
    rr_->uploadFaceMesh({});
    useSim_ = false;
}

void RootScene::uploadFaceFromMasks() {
    if (!rr_ || !sim_) return;
    if (!showFace || faceVerts_.empty() || faceTris_.empty()) { rr_->uploadFaceMesh({}); return; }
    const float maskColor[3] = {0.86f, 0.83f, 0.78f};
    std::vector<float> data;
    for (const auto& sm : sim_->revealedMasks()) {
        Mask m;
        m.pos = {sm.pos[0], sm.pos[1], sm.pos[2]};
        m.normal = {sm.normal[0], sm.normal[1], sm.normal[2]};
        m.tangent = {sm.tangent[0], sm.tangent[1], sm.tangent[2]};
        m.bitangent = {sm.bitangent[0], sm.bitangent[1], sm.bitangent[2]};
        m.rDepth = sm.rDepth; m.rWidth = sm.rWidth; m.rHeight = sm.rHeight;
        appendFaceVertexData(data, m, faceVerts_, faceTris_, faceScale, 3.0f, maskColor,
                             faceColors_);
    }
    rr_->uploadFaceMesh(data);
}

void RootScene::rebuildFace() {
    if (!rr_) return;
    if (useSim_) { uploadFaceFromMasks(); return; }
    if (!showFace || faceVerts_.empty() || faceTris_.empty()) {
        rr_->uploadFaceMesh({});
        return;
    }
    const float maskColor[3] = {0.86f, 0.83f, 0.78f};
    // A few masks around the trunk at staggered heights, facing outward, so the
    // orbit reveals them and they depth-composite against the roots.
    std::vector<float> data;
    const int N = 3;
    for (int i = 0; i < N; i++) {
        float ang = 2.0f * (float)M_PI * i / N + 0.4f;
        float h = 6.0f + 5.0f * i;
        F3 dir = {std::sin(ang), 0.15f, std::cos(ang)};
        F3 center = {target[0] + 5.0f * std::sin(ang),
                     target[1] + h - 6.0f,
                     target[2] + 5.0f * std::cos(ang)};
        Mask m = makeMask(center, dir, 4.5f);
        appendFaceVertexData(data, m, faceVerts_, faceTris_, faceScale, 3.0f, maskColor,
                             faceColors_);
    }
    rr_->uploadFaceMesh(data);
}

void RootScene::setFittedFace(const std::vector<float>& verts,
                              const std::vector<int>& tris) {
    if (verts.size() < 9) return;
    if (!tris.empty()) faceTris_ = tris;
    if (faceTris_.empty()) return;

    // Capture the normalisation once. Centre on the mesh centroid and divide by
    // the largest absolute coordinate, matching what normalizeMesh() does to
    // the canonical model -- so the placement code, faceScale, and every
    // material knob downstream keep the ranges they were tuned against.
    if (!fit_norm_set_) {
        const size_t n = verts.size() / 3;
        double cx = 0, cy = 0, cz = 0;
        for (size_t i = 0; i < n; i++) {
            cx += verts[i * 3]; cy += verts[i * 3 + 1]; cz += verts[i * 3 + 2];
        }
        fit_centre_[0] = float(cx / n);
        fit_centre_[1] = float(cy / n);
        fit_centre_[2] = float(cz / n);
        float m = 1e-9f;
        for (size_t i = 0; i < n; i++) {
            m = std::max({m, std::fabs(verts[i * 3]     - fit_centre_[0]),
                             std::fabs(verts[i * 3 + 1] - fit_centre_[1]),
                             std::fabs(verts[i * 3 + 2] - fit_centre_[2])});
        }
        fit_scale_ = 1.0f / m;
        fit_norm_set_ = true;
    }

    faceVerts_.resize(verts.size());
    for (size_t i = 0; i < verts.size() / 3; i++) {
        faceVerts_[i * 3]     = (verts[i * 3]     - fit_centre_[0]) * fit_scale_;
        faceVerts_[i * 3 + 1] = (verts[i * 3 + 1] - fit_centre_[1]) * fit_scale_;
        faceVerts_[i * 3 + 2] = (verts[i * 3 + 2] - fit_centre_[2]) * fit_scale_;
    }
    fitted_face_ = true;
    rebuildFace();
}

void RootScene::setFaceColors(const std::vector<float>& rgb) {
    faceColors_ = rgb;
    rebuildFace();
}

void RootScene::clearFittedFace() {
    if (!fitted_face_) return;
    faceVerts_ = canonVerts_;
    faceTris_  = canonTris_;
    fitted_face_ = false;
    fit_norm_set_ = false;
    rebuildFace();
}

// Scene bounds, for framing. Recomputed whenever the geometry is re-uploaded,
// which during a grow is every frame -- it is a min/max over the node array
// that was just built anyway.
void RootScene::updateBounds(const std::vector<float>& nodes) {
    if (nodes.size() < 3) return;
    float lo[3] = {nodes[0], nodes[1], nodes[2]};
    float hi[3] = {nodes[0], nodes[1], nodes[2]};
    for (size_t i = 0; i + 2 < nodes.size(); i += 3)
        for (int c = 0; c < 3; ++c) {
            lo[c] = std::min(lo[c], nodes[i + c]);
            hi[c] = std::max(hi[c], nodes[i + c]);
        }
    for (int c = 0; c < 3; ++c) idleCentre_[c] = 0.5f * (lo[c] + hi[c]);
    idleExtent_ = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1.f});
}

void RootScene::applyFraming() {
    if (!autoFrame) return;
    const auto& ms = revealedMasks();
    if (focusMask >= 0 && focusMask < (int)ms.size()) {
        const auto& m = ms[size_t(focusMask)];
        target[0] = m.pos[0]; target[1] = m.pos[1]; target[2] = m.pos[2];
        radius = (std::max(m.rWidth, m.rHeight) * 3.5f + 4.0f) * zoom;
        // Its own angle, advanced independently: the whole-scene orbit is
        // centred on the piece, so borrowing it would sweep this mask out of
        // frame and eventually behind the camera.
        if (autoOrbit) azimuth = focusAngle_;
    } else {
        target[0] = idleCentre_[0];
        target[1] = idleCentre_[1];
        target[2] = idleCentre_[2];
        radius = (idleExtent_ * 0.75f + 12.0f) * zoom;
    }
    if (rr_) rr_->fog.refDist = radius;
}

void RootScene::ensureSize(int w, int h) {
    if (rr_) rr_->resize(std::max(1, w), std::max(1, h));
}

void RootScene::reseed(uint32_t seed) {
    // Reseeding used to build the synthetic stand-in unconditionally. With a
    // grow running, that uploaded a whole different structure which advance()
    // then overwrote on the next frame -- a flash of something that was never
    // part of the scene. The stand-in exists for when CPlantBox is not there
    // at all, and that is the only time it should appear.
    if (simAvailable_) {
        simParams_.seed = seed;
        regrow();
    } else {
        buildSyntheticRoots(seed);
    }
}

// A recursive branching structure standing in for CPlantBox roots: capsule
// segments that taper and split, grown both up (a canopy) and down (roots).
void RootScene::buildSyntheticRoots(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::uniform_real_distribution<float> U01(0.f, 1.f);

    std::vector<float> nodes;   // 3/node
    std::vector<int>   segs;    // 2/seg
    std::vector<float> radii;   // 1/seg

    auto addNode = [&](float x, float y, float z) -> int {
        nodes.push_back(x); nodes.push_back(y); nodes.push_back(z);
        return (int)(nodes.size() / 3) - 1;
    };

    // Recursive branch. dir is a unit-ish vector; length/radius taper with depth.
    struct Frame { int node; float px, py, pz; float dx, dy, dz; float len; float rad; int depth; };
    std::vector<Frame> stack;

    auto seedBranch = [&](float ox, float oy, float oz, float dx, float dy, float dz,
                          float len, float rad, int depth) {
        int n = addNode(ox, oy, oz);
        stack.push_back({n, ox, oy, oz, dx, dy, dz, len, rad, depth});
    };

    // Canopy up, roots down, from a shared base.
    seedBranch(0, 0, 0,  0,  1, 0, 5.5f, 0.9f, 0);
    seedBranch(0, 0, 0,  0, -1, 0, 4.5f, 0.9f, 0);

    const int MAX_DEPTH = 8;
    while (!stack.empty()) {
        Frame fr = stack.back();
        stack.pop_back();
        if (fr.depth > MAX_DEPTH) continue;

        // Normalize direction.
        float dl = std::sqrt(fr.dx*fr.dx + fr.dy*fr.dy + fr.dz*fr.dz);
        if (dl < 1e-5f) dl = 1.f;
        float dx = fr.dx/dl, dy = fr.dy/dl, dz = fr.dz/dl;

        float nx = fr.px + dx * fr.len;
        float ny = fr.py + dy * fr.len;
        float nz = fr.pz + dz * fr.len;
        int child = addNode(nx, ny, nz);
        segs.push_back(fr.node); segs.push_back(child);
        radii.push_back(fr.rad);

        if (fr.depth == MAX_DEPTH) continue;

        // 1-3 children, direction perturbed, tapering length/radius.
        int nChildren = 1 + (U01(rng) < 0.7f ? 1 : 0) + (U01(rng) < 0.3f ? 1 : 0);
        for (int c = 0; c < nChildren; c++) {
            float jitter = 0.55f;
            float cx = dx + U(rng) * jitter;
            float cy = dy + U(rng) * jitter * 0.6f;   // keep vertical bias
            float cz = dz + U(rng) * jitter;
            float childLen = fr.len * (0.78f + 0.12f * U01(rng));
            float childRad = fr.rad * 0.72f;
            stack.push_back({child, nx, ny, nz, cx, cy, cz, childLen, childRad, fr.depth + 1});
        }
    }

    rr_->uploadSegments(nodes, segs, radii);
}

void RootScene::advance(double dt) {
    t_ += dt;
    if (autoOrbit) azimuth += orbitRate * (float)dt;
    rr_->fog.driftTime += (float)dt * rr_->fog.driftSpeed;
    rr_->pulse.time    += (float)dt;
    rr_->wispTime      += (float)dt;

    focusAngle_ += orbitRate * (float)dt;

    // Live growth: advance a few steps, then re-upload geometry + revealed masks.
    if (useSim_ && sim_ && !sim_->done()) {
        for (int i = 0; i < std::max(1, simStepsPerFrame) && !sim_->done(); ++i)
            sim_->step();
        std::vector<float> nodes, radii; std::vector<int> segs;
        sim_->geometry(nodes, segs, radii);
        rr_->uploadSegments(nodes, segs, radii);
        updateBounds(nodes);
        uploadFaceFromMasks();
    }
    applyFraming();
}

id<MTLTexture> RootScene::render(id<MTLCommandBuffer> cb) {
    if (!valid()) return nil;
    float ld = std::sqrt(lightDir[0]*lightDir[0] + lightDir[1]*lightDir[1] + lightDir[2]*lightDir[2]);
    if (ld < 1e-5f) ld = 1.f;
    float L[3] = {lightDir[0]/ld, lightDir[1]/ld, lightDir[2]/ld};
    return rr_->render(cb, azimuth, elevation, radius, target, fov, L);
}
