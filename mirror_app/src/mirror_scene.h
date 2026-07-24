// MirrorScene — the neural-mirror scene: owns a Pond (full demo_panel port),
// advances its clocks, and uploads each frame's low-res image to a Metal texture
// that the compositor presents (linearly upsampled).
#pragma once

#ifndef __OBJC__
#error "mirror_scene.h is ObjC++ only"
#endif

#import <Metal/Metal.h>

#include "pond_state.h"

class MetalContext;

class MirrorScene {
public:
    MirrorScene(const MetalContext& ctx, int seed = 11, int lowW = 480, int lowH = 270);

    // Ensure the low-res render target matches (w, h); recreates it if changed.
    void ensureSize(int w, int h);

    // Advance the ripple / z / transition clocks by dt seconds.
    void advance(double dt);

    // Render the current frame and return the low-res RGBA16F texture.
    id<MTLTexture> render();

    void reseed() { pond_.reseed(); }

    bool valid() const { return tex_ != nil; }
    int  lowW() const { return lw_; }
    int  lowH() const { return lh_; }
    double clock() const { return t_; }

    mirror::PondParams& params() { return params_; }

private:
    void makeTexture();

    const MetalContext& ctx_;
    mirror::Pond pond_;
    mirror::PondParams params_;
    double t_ = 0.0;
    int lw_, lh_;
    id<MTLTexture> tex_ = nil;
};
