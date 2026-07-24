#include "root_scene.h"
#include "metal_context.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
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
void appendFaceVertexData(std::vector<float>& out, const Mask& m,
                          const std::vector<float>& fv, const std::vector<int>& ftris,
                          float faceScale, float lightDist, const float color[3]) {
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
            out.push_back(v3[k].x); out.push_back(v3[k].y); out.push_back(v3[k].z);
            out.push_back(fn.x); out.push_back(fn.y); out.push_back(fn.z);
            out.push_back(color[0]); out.push_back(color[1]); out.push_back(color[2]);
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

    buildSyntheticRoots(1u);

    // Canonical face model (shared with the GL sdf_viewer assets).
    loadObj(std::string(SDF_VIEWER_DIR) + "/assets/canonical_face_model.obj",
            faceVerts_, faceTris_);
    normalizeMesh(faceVerts_);
    faceTris_ = cropOvalTris(faceVerts_, faceTris_, 0.72f, 0.98f);
    rebuildFace();
}

void RootScene::rebuildFace() {
    if (!rr_) return;
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
        appendFaceVertexData(data, m, faceVerts_, faceTris_, faceScale, 3.0f, maskColor);
    }
    rr_->uploadFaceMesh(data);
}

void RootScene::ensureSize(int w, int h) {
    if (rr_) rr_->resize(std::max(1, w), std::max(1, h));
}

void RootScene::reseed(uint32_t seed) { buildSyntheticRoots(seed); }

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
}

id<MTLTexture> RootScene::render(id<MTLCommandBuffer> cb) {
    if (!valid()) return nil;
    float ld = std::sqrt(lightDir[0]*lightDir[0] + lightDir[1]*lightDir[1] + lightDir[2]*lightDir[2]);
    if (ld < 1e-5f) ld = 1.f;
    float L[3] = {lightDir[0]/ld, lightDir[1]/ld, lightDir[2]/ld};
    return rr_->render(cb, azimuth, elevation, radius, target, fov, L);
}
