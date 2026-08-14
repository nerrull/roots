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

    // The frame render() just produced, as CPU-side RGB floats in [0,1]
    // (lowH*lowW*3, row-major). The face-mask texturing path needs pixels
    // rather than a texture: the mask is coloured per vertex by projecting the
    // fitted mesh into this image and sampling it, so the mask ends up wearing
    // the network's reconstruction of the face rather than the camera's.
    //
    // Cheap: MLX arrays live in unified memory, so at this point the data is
    // already CPU-addressable and this is a copy, not a GPU readback.
    const std::vector<float>& lastImageRGB() const { return cpu_rgb_; }

    void reseed() { pond_.reseed(); }

    // --- live fitting -------------------------------------------------------
    // Exposed straight through: the scene owns the Pond, and fitting is a
    // property of the network rather than of the render target.
    mirror::Pond& pond() { return pond_; }
    // Runs `steps` optimiser steps. Returns the last loss, or -1 if not
    // fitting. Called once per frame from the app loop, before render().
    float fitSteps(int steps, float lr);
    float lastLoss() const { return last_loss_; }

    bool valid() const { return tex_ != nil; }
    int  lowW() const { return lw_; }
    int  lowH() const { return lh_; }
    double clock() const { return t_; }

    mirror::PondParams& params() { return params_; }

private:
    void makeTexture();
    float last_loss_ = -1.f;

    const MetalContext& ctx_;
    mirror::Pond pond_;
    mirror::PondParams params_;
    double t_ = 0.0;
    int lw_, lh_;
    std::vector<float> cpu_rgb_;
    id<MTLTexture> tex_ = nil;
};
