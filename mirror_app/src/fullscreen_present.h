// FullscreenPresent — draws a texture across the whole target via present.metal.
// Used to blit a scene's offscreen color texture onto the drawable.
#pragma once

#ifndef __OBJC__
#error "fullscreen_present.h is ObjC++ only"
#endif

#import <Metal/Metal.h>
#include <string>

class MetalContext;

class FullscreenPresent {
public:
    // shaderPath: path to present.metal. colorFormat: the target's pixel format.
    FullscreenPresent(const MetalContext& ctx, const std::string& shaderPath,
                      MTLPixelFormat colorFormat);

    // Draw `tex` over the full viewport into an already-open render encoder.
    void encode(id<MTLRenderCommandEncoder> enc, id<MTLTexture> tex) const;

    bool valid() const { return pipeline_ != nil; }

private:
    id<MTLRenderPipelineState> pipeline_ = nil;
};
