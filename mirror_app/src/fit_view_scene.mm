#include "fit_view_scene.h"
#include "metal_context.h"

#import <simd/simd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace {

struct Params {
    simd_float2 mask_size;
    float mask_tint;
    float bg_dim;
    float mesh_alpha;
    float pad[3];
};

}  // namespace

FitViewScene::FitViewScene(const MetalContext& ctx, const std::string& shaderPath,
                           int w, int h)
    : ctx_(ctx), w_(std::max(2, w)), h_(std::max(2, h)) {
    id<MTLLibrary> lib = ctx_.newLibraryFromFile(shaderPath);
    if (!lib) {
        fprintf(stderr, "FitViewScene: no library from %s\n", shaderPath.c_str());
        return;
    }

    auto make = [&](NSString* vs, NSString* fs, bool blend) -> id<MTLRenderPipelineState> {
        MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
        d.vertexFunction = [lib newFunctionWithName:vs];
        d.fragmentFunction = [lib newFunctionWithName:fs];
        d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
        if (blend) {
            d.colorAttachments[0].blendingEnabled = YES;
            d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            d.colorAttachments[0].destinationRGBBlendFactor =
                MTLBlendFactorOneMinusSourceAlpha;
        }
        NSError* err = nil;
        id<MTLRenderPipelineState> p =
            [ctx_.device() newRenderPipelineStateWithDescriptor:d error:&err];
        if (!p) NSLog(@"FitViewScene: pipeline failed: %@", err);
        return p;
    };

    pipe_bg_   = make(@"v_bg", @"f_bg", false);
    pipe_mesh_ = make(@"v_mesh", @"f_mesh", true);
    pipe_line_ = make(@"v_mesh", @"f_flat", false);
    makeTexture();
}

void FitViewScene::makeTexture() {
    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:w_ height:h_ mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    // Shared rather than private: on unified memory it costs nothing, and it
    // keeps the target readable with getBytes, which is what the headless
    // check reads back. A private target would render identically on screen
    // and crash the moment anything tried to look at it.
    td.storageMode = MTLStorageModeShared;
    tex_ = [ctx_.device() newTextureWithDescriptor:td];
}

void FitViewScene::ensureSize(int w, int h) {
    w = std::max(2, w);
    h = std::max(2, h);
    if (w == w_ && h == h_ && tex_) return;
    w_ = w; h_ = h;
    makeTexture();
}

void FitViewScene::setMask(const std::vector<unsigned char>& mask, int w, int h) {
    if (w <= 0 || h <= 0 || mask.size() != size_t(w) * h) { mask_w_ = 0; return; }
    if (!mask_ || mask_w_ != w || mask_h_ != h) {
        MTLTextureDescriptor* td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                               width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeManaged;
        mask_ = [ctx_.device() newTextureWithDescriptor:td];
        mask_w_ = w; mask_h_ = h;
    }
    // The mask is 0/1 bytes; R8Unorm reads 1 as 1/255, so scale on the way in
    // rather than making the shader compensate for the storage format.
    static std::vector<unsigned char> scaled;
    scaled.resize(mask.size());
    for (size_t i = 0; i < mask.size(); ++i) scaled[i] = mask[i] ? 255 : 0;
    [mask_ replaceRegion:MTLRegionMake2D(0, 0, w, h)
             mipmapLevel:0
               withBytes:scaled.data()
             bytesPerRow:size_t(w)];
}

void FitViewScene::setMesh(const std::vector<float>& uv, const std::vector<float>& rgb,
                           const std::vector<int>& tris) {
    const size_t n = uv.size() / 2;
    if (n == 0) { vert_count_ = 0; return; }

    interleaved_.resize(n * 5);
    const bool have_rgb = rgb.size() >= n * 3;
    for (size_t i = 0; i < n; ++i) {
        float* o = &interleaved_[i * 5];
        o[0] = uv[i * 2];
        o[1] = uv[i * 2 + 1];
        // No sampled colour yet is a state worth seeing rather than hiding: a
        // flat grey mask means the texture pass has not run, which is a
        // different problem from one whose colours are simply in the wrong
        // place.
        o[2] = have_rgb ? rgb[i * 3]     : 0.5f;
        o[3] = have_rgb ? rgb[i * 3 + 1] : 0.5f;
        o[4] = have_rgb ? rgb[i * 3 + 2] : 0.5f;
    }
    const size_t bytes = interleaved_.size() * sizeof(float);
    if (!vbuf_ || vbuf_.length < bytes) {
        vbuf_ = [ctx_.device() newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
    }
    memcpy(vbuf_.contents, interleaved_.data(), bytes);
    vert_count_ = int(n);

    if (tris.empty()) return;   // topology is uploaded once and then reused

    std::vector<uint32_t> idx(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) idx[i] = uint32_t(tris[i]);
    const size_t ibytes = idx.size() * sizeof(uint32_t);
    ibuf_ = [ctx_.device() newBufferWithBytes:idx.data()
                                       length:ibytes
                                      options:MTLResourceStorageModeShared];
    index_count_ = int(idx.size());

    // Wireframe as an explicit line-index buffer: Metal has no line-fill
    // polygon mode, so the three edges of each triangle are expanded once here
    // rather than every frame.
    std::vector<uint32_t> lines;
    lines.reserve(tris.size() * 2);
    for (size_t t = 0; t + 2 < tris.size(); t += 3) {
        const uint32_t a = uint32_t(tris[t]), b = uint32_t(tris[t + 1]),
                       c = uint32_t(tris[t + 2]);
        lines.push_back(a); lines.push_back(b);
        lines.push_back(b); lines.push_back(c);
        lines.push_back(c); lines.push_back(a);
    }
    lbuf_ = [ctx_.device() newBufferWithBytes:lines.data()
                                       length:lines.size() * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];
    line_count_ = int(lines.size());
}

id<MTLTexture> FitViewScene::render(id<MTLCommandBuffer> cb) {
    if (!valid() || !tex_) return nil;

    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = tex_;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.03, 0.03, 0.04, 1.0);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    Params u{};
    u.mask_size = simd_make_float2(float(mask_w_), float(mask_h_));
    u.mask_tint = maskTint;
    u.bg_dim = bgDim;
    u.mesh_alpha = meshAlpha;

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];

    if (bg_) {
        [enc setRenderPipelineState:pipe_bg_];
        [enc setFragmentTexture:bg_ atIndex:0];
        [enc setFragmentTexture:(mask_ ? mask_ : bg_) atIndex:1];
        [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }

    if (showMesh && hasMesh()) {
        [enc setRenderPipelineState:(wireframe ? pipe_line_ : pipe_mesh_)];
        [enc setVertexBuffer:vbuf_ offset:0 atIndex:0];
        [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
        if (wireframe && lbuf_) {
            [enc drawIndexedPrimitives:MTLPrimitiveTypeLine
                            indexCount:line_count_
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:lbuf_
                     indexBufferOffset:0];
        } else if (ibuf_) {
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:index_count_
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:ibuf_
                     indexBufferOffset:0];
        }
    }

    [enc endEncoding];
    return tex_;
}
