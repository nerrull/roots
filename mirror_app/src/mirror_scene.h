// MirrorScene — the neural-mirror scene. Runs the MLX fused-MLP "pond" render
// each frame and hands back a low-res Metal texture (linearly upsampled at
// present time). This is the C++ analogue of mlx_fused_mlp/demo_pond.
#pragma once

#ifndef __OBJC__
#error "mirror_scene.h is ObjC++ only"
#endif

#import <Metal/Metal.h>
#include <string>

#include "mlp_forward.h"
#include "mirror_render.h"

class MetalContext;

class MirrorScene {
public:
    // assetDir holds pond_weights.f32 / pond_weights.meta (from gen_pond_weights.py).
    MirrorScene(const MetalContext& ctx, const std::string& assetDir,
                int lowW = 480, int lowH = 270);

    // Ensure the low-res render target matches (w, h); recreates it if changed.
    void ensureSize(int w, int h);

    // Animate to time t (seconds) and return the current low-res RGBA16F texture.
    id<MTLTexture> render(double t);

    bool valid() const { return tex_ != nil; }
    int lowW() const { return lw_; }
    int lowH() const { return lh_; }

    float speed = 1.2f;      // ripple propagation speed (demo_pond default)
    float ringFreq = 3.0f;
    float decay = 1.6f;
    float warp = 0.0f;       // domain-warp refraction strength
    float core = 0.0f;       // core-radius damping (0 = off)
    float z = 0.0f;          // latent channel 2
    float zCos = 0.0f;       // quadrature latent (channel 6)
    bool  animateZ = false;  // drive (z, zCos) around the latent circle
    float zSpeed = 0.3f;

private:
    void makeTexture();

    const MetalContext& ctx_;
    mirror::MLPConfig cfg_;
    mirror::mx::array weights_;
    std::vector<mirror::RippleSource> drops_;   // (cx, cy, rate, phase)
    int lw_, lh_;
    id<MTLTexture> tex_ = nil;
};
