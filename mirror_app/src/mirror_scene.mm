#include "mirror_scene.h"
#include "metal_context.h"
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>

namespace mx = mirror::mx;

MirrorScene::MirrorScene(const MetalContext& ctx, int seed, int lowW, int lowH)
    : ctx_(ctx), pond_(seed), lw_(lowW), lh_(lowH) {
    makeTexture();
}

void MirrorScene::makeTexture() {
    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:lw_ height:lh_ mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;   // CPU-writable (unified memory)
    tex_ = [ctx_.device() newTextureWithDescriptor:td];
}

void MirrorScene::ensureSize(int w, int h) {
    w = std::max(2, w);
    h = std::max(2, h);
    if (w == lw_ && h == lh_ && tex_) return;
    lw_ = w; lh_ = h;
    makeTexture();
}

void MirrorScene::advance(double dt) {
    if (!params_.paused) {
        t_ += dt * params_.time_scale;
        params_.z += dt * params_.z_rate;
    }
    if (params_.trans_auto)
        params_.transition = std::min(1.0f, std::max(0.0f, params_.transition + (float)dt * params_.trans_rate));
}

float MirrorScene::fitSteps(int steps, float lr) {
    if (!pond_.fitting()) return -1.f;
    for (int i = 0; i < steps; ++i) last_loss_ = pond_.fitStep(lr, params_);
    return last_loss_;
}

id<MTLTexture> MirrorScene::render() {
    auto img = pond_.render(lh_, lw_, t_, params_);   // (lh, lw, 3) fp32 [0,1]

    // Keep a CPU copy before the fp16 cast: the texturing path wants the same
    // pixels the display shows, at full precision, and evaluating `img` here
    // costs nothing because the concatenate below forces it anyway.
    {
        auto rgbc = mx::contiguous(img);
        mx::eval(rgbc);
        const float* src = rgbc.data<float>();
        cpu_rgb_.assign(src, src + size_t(lh_) * lw_ * 3);
    }

    auto rgb16 = mx::astype(img, mx::float16);
    auto alpha = mx::ones({lh_, lw_, 1}, mx::float16);
    auto rgba = mx::contiguous(mx::concatenate({rgb16, alpha}, 2));
    mx::eval(rgba);

    const void* ptr = rgba.data<uint16_t>();   // fp16 bytes, unified memory
    [tex_ replaceRegion:MTLRegionMake2D(0, 0, lw_, lh_)
            mipmapLevel:0
              withBytes:ptr
            bytesPerRow:(NSUInteger)lw_ * 4 * sizeof(uint16_t)];
    return tex_;
}
