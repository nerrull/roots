#include "root_scene.h"
#include "metal_context.h"

#include <cmath>
#include <random>
#include <vector>

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
