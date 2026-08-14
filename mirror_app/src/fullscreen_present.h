// FullscreenPresent — draws a texture across the whole target via present.metal.
// Used to blit a scene's offscreen color texture onto the drawable.
#pragma once

#ifndef __OBJC__
#error "fullscreen_present.h is ObjC++ only"
#endif

#import <Metal/Metal.h>
#include <string>

#include "text_overlay.h"

class MetalContext;

class FullscreenPresent {
public:
    // shaderPath: path to present.metal. colorFormat: the target's pixel format.
    FullscreenPresent(const MetalContext& ctx, const std::string& shaderPath,
                      MTLPixelFormat colorFormat);

    // Draw `tex` over the full viewport into an already-open render encoder,
    // optionally compositing a text distance field over it (see text_overlay.h).
    // `sdf` may be nil when the overlay is disabled -- a 1x1 stand-in is bound in
    // its place, because the fragment function samples the slot unconditionally
    // and an unbound texture there is undefined behaviour rather than a no-op.
    void encode(id<MTLRenderCommandEncoder> enc, id<MTLTexture> tex,
                id<MTLTexture> sdf, const mirror::TextUniforms& text) const;

    // Without any overlay.
    void encode(id<MTLRenderCommandEncoder> enc, id<MTLTexture> tex) const {
        encode(enc, tex, nil, mirror::TextUniforms{});
    }

    bool valid() const { return pipeline_ != nil; }

private:
    id<MTLRenderPipelineState> pipeline_ = nil;
    id<MTLTexture> dummy_ = nil;
};
