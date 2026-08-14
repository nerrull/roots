#include "fullscreen_present.h"
#include "metal_context.h"
#import <Foundation/Foundation.h>
#include <cstdio>

FullscreenPresent::FullscreenPresent(const MetalContext& ctx,
                                     const std::string& shaderPath,
                                     MTLPixelFormat colorFormat) {
    id<MTLLibrary> lib = ctx.newLibraryFromFile(shaderPath);
    if (!lib) { fprintf(stderr, "FullscreenPresent: no library from %s\n", shaderPath.c_str()); return; }

    MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
    d.vertexFunction = [lib newFunctionWithName:@"present_vs"];
    d.fragmentFunction = [lib newFunctionWithName:@"present_fs"];
    d.colorAttachments[0].pixelFormat = colorFormat;

    NSError* err = nil;
    pipeline_ = [ctx.device() newRenderPipelineStateWithDescriptor:d error:&err];
    if (!pipeline_) NSLog(@"FullscreenPresent: pipeline failed: %@", err);

    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                     width:1 height:1 mipmapped:NO];
    dd.usage = MTLTextureUsageShaderRead;
    dd.storageMode = MTLStorageModeShared;
    dummy_ = [ctx.device() newTextureWithDescriptor:dd];
    const unsigned char zero = 0;
    [dummy_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
              mipmapLevel:0 withBytes:&zero bytesPerRow:1];
}

void FullscreenPresent::encode(id<MTLRenderCommandEncoder> enc, id<MTLTexture> tex,
                               id<MTLTexture> sdf,
                               const mirror::TextUniforms& text) const {
    if (!pipeline_ || !tex) return;
    [enc setRenderPipelineState:pipeline_];
    [enc setFragmentTexture:tex atIndex:0];
    [enc setFragmentTexture:(sdf ? sdf : dummy_) atIndex:1];
    [enc setFragmentBytes:&text length:sizeof(text) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
}
