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
#include <memory>
#include <string>

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

    // Regenerate the synthetic root structure with a new random seed.
    void reseed(uint32_t seed);

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

    std::unique_ptr<MetalRootRenderer> rr_;
    double t_ = 0.0;
};
