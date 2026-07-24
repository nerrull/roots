#include "mirror_scene.h"
#include "metal_context.h"
#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <vector>

namespace mx = mirror::mx;

namespace {

std::map<std::string, int> read_meta(const std::string& path) {
    std::ifstream f(path);
    std::map<std::string, int> m;
    std::string k; int v;
    while (f >> k >> v) m[k] = v;
    return m;
}

std::vector<float> read_f32(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    std::vector<float> v(count);
    if (f) f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
    return v;
}

}  // namespace

MirrorScene::MirrorScene(const MetalContext& ctx, const std::string& assetDir,
                         int lowW, int lowH)
    : ctx_(ctx), weights_(mx::zeros({1})), lw_(lowW), lh_(lowH) {
    std::string dir = assetDir;
    if (!dir.empty() && dir.back() != '/') dir += '/';

    auto meta = read_meta(dir + "pond_weights.meta");
    cfg_.in_dim = meta.count("in_dim") ? meta["in_dim"] : 8;
    cfg_.hidden_dim = meta.count("hidden_dim") ? meta["hidden_dim"] : 64;
    cfg_.out_dim = meta.count("out_dim") ? meta["out_dim"] : 3;
    cfg_.num_layers = meta.count("num_layers") ? meta["num_layers"] : 6;
    cfg_.activation = static_cast<mirror::Act>(meta.count("act") ? meta["act"] : 1);
    cfg_.out_activation = static_cast<mirror::Act>(meta.count("out_act") ? meta["out_act"] : 2);

    const int total = cfg_.total_weights();
    auto wv = read_f32(dir + "pond_weights.f32", (size_t)total);
    weights_ = mx::array(wv.data(), {total}, mx::float32);

    // Fixed raindrop sites (cx, cy, rate, phase) — a hand-picked spread.
    drops_ = {
        {-0.55f,  0.35f, 0.23f, 0.0f},
        { 0.60f,  0.10f, 0.31f, 1.7f},
        {-0.20f, -0.55f, 0.17f, 3.1f},
        { 0.35f, -0.30f, 0.41f, 4.6f},
    };

    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:lw_ height:lh_ mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;   // CPU-writable (unified memory)
    tex_ = [ctx_.device() newTextureWithDescriptor:td];
}

id<MTLTexture> MirrorScene::render(double t) {
    // pond_sources: pulsing fixed drops + one orbiting source.
    const float phase = 2.0f * (float)M_PI * speed * (float)t;
    std::vector<mirror::RippleSource> sources;
    sources.reserve(drops_.size() + 1);
    for (const auto& d : drops_) {
        float amp = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * d[2] * (float)t + d[3]);
        sources.push_back({d[0], d[1], phase, amp});
    }
    const float asp = (float)lw_ / (float)lh_;
    sources.push_back({0.6f * asp * std::cos(0.5f * (float)t),
                       0.6f * std::sin(0.5f * (float)t), phase, 1.0f});

    auto rgba = mirror::render_pond_lowres(weights_, cfg_, lh_, lw_, sources,
                                           ringFreq, decay, /*z=*/0.f, /*z_cos=*/0.f);
    rgba = mx::contiguous(rgba);
    mx::eval(rgba);

    const void* ptr = rgba.data<uint16_t>();   // fp16 bytes, unified memory
    [tex_ replaceRegion:MTLRegionMake2D(0, 0, lw_, lh_)
            mipmapLevel:0
              withBytes:ptr
            bytesPerRow:(NSUInteger)lw_ * 4 * sizeof(uint16_t)];
    return tex_;
}
