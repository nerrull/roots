#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "BubbleRenderer.h"
#include <vector>
#include <cmath>
#include <cstdio>

// Layout must mirror the Metal struct exactly.
// Metal float3 occupies 16 bytes (12 data + 4 implicit pad) in a struct,
// so we add explicit padding to keep the C++ layout identical.
struct Uniforms {
    float camPos[3];       // offset  0
    float _p0;             // offset 12
    float camDir[3];       // offset 16
    float _p1;             // offset 28
    float camRight[3];     // offset 32
    float _p2;             // offset 44
    float camUp[3];        // offset 48
    float _p3;             // offset 60
    float time;            // offset 64
    float filmThickness;   // offset 68
    float bubbleRadius;    // offset 72
    float lightAzimuth;    // offset 76
    float lightElevation;  // offset 80
    float aspectRatio;     // offset 84
    float fovTan;          // offset 88
    float wobbleFreq;      // offset 92
    float wobbleAmp;       // offset 96
    float _p4;             // offset 100
};

struct BubbleRenderer::Impl {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLTexture>              outTex;
    id<MTLBuffer>               uniforms;
    std::vector<uint8_t>        pixels;
};

// ---------------------------------------------------------------------------

BubbleRenderer::BubbleRenderer(int w, int h) {
    _impl = std::make_unique<Impl>();

    _impl->device = MTLCreateSystemDefaultDevice();
    if (!_impl->device) { fprintf(stderr, "[bubble] no Metal device\n"); return; }

    _impl->queue = [_impl->device newCommandQueue];

    NSError* err  = nil;
    NSString* path = @METALLIB_PATH;
    id<MTLLibrary> lib = [_impl->device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!lib) {
        fprintf(stderr, "[bubble] metallib load failed: %s\n",
                err.localizedDescription.UTF8String);
        return;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"bubble_kernel"];
    if (!fn) { fprintf(stderr, "[bubble] kernel not found\n"); return; }

    _impl->pipeline = [_impl->device newComputePipelineStateWithFunction:fn error:&err];
    if (!_impl->pipeline) {
        fprintf(stderr, "[bubble] pipeline error: %s\n",
                err.localizedDescription.UTF8String);
        return;
    }

    _impl->uniforms = [_impl->device newBufferWithLength:sizeof(Uniforms)
                                                 options:MTLResourceStorageModeShared];

    // OpenGL texture — will be (re)allocated in resize()
    glGenTextures(1, &_glTex);
    glBindTexture(GL_TEXTURE_2D, _glTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    resize(w, h);
}

BubbleRenderer::~BubbleRenderer() {
    if (_glTex) glDeleteTextures(1, &_glTex);
}

void BubbleRenderer::resize(int w, int h) {
    if (w == _w && h == _h && _impl->outTex) return;
    _w = w; _h = h;

    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:w
                                                          height:h
                                                       mipmapped:NO];
    desc.usage       = MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModeShared;   // CPU-readable (unified mem on Apple Silicon)
    _impl->outTex    = [_impl->device newTextureWithDescriptor:desc];

    _impl->pixels.resize(w * h * 4);

    glBindTexture(GL_TEXTURE_2D, _glTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void BubbleRenderer::render(float time, float azimuth, float elevation,
                            float orbitRadius, const Params& p)
{
    if (!_impl->pipeline) return;

    // Build orbit camera
    float ce = cosf(elevation), se = sinf(elevation);
    float ca = cosf(azimuth),   sa = sinf(azimuth);
    float cx = orbitRadius * ce * sa;
    float cy = orbitRadius * se;
    float cz = orbitRadius * ce * ca;
    float len = sqrtf(cx*cx + cy*cy + cz*cz);

    float dir[3]   = { -cx/len, -cy/len, -cz/len };
    float world[3] = { 0, 1, 0 };

    // right = normalize(camDir × worldUp)
    float right[3] = {
        dir[1]*world[2] - dir[2]*world[1],
        dir[2]*world[0] - dir[0]*world[2],
        dir[0]*world[1] - dir[1]*world[0]
    };
    float rLen = sqrtf(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if (rLen > 1e-6f) { right[0]/=rLen; right[1]/=rLen; right[2]/=rLen; }

    // up = cross(right, dir)
    float up[3] = {
        right[1]*dir[2] - right[2]*dir[1],
        right[2]*dir[0] - right[0]*dir[2],
        right[0]*dir[1] - right[1]*dir[0]
    };

    Uniforms* u   = (Uniforms*)_impl->uniforms.contents;
    u->camPos[0]  = cx;     u->camPos[1]  = cy;     u->camPos[2]  = cz;
    u->camDir[0]  = dir[0]; u->camDir[1]  = dir[1]; u->camDir[2]  = dir[2];
    u->camRight[0]= right[0];u->camRight[1]=right[1];u->camRight[2]=right[2];
    u->camUp[0]   = up[0];  u->camUp[1]   = up[1];  u->camUp[2]   = up[2];
    u->time            = time;
    u->filmThickness   = p.filmThickness;
    u->bubbleRadius    = p.bubbleRadius;
    u->lightAzimuth    = p.lightAzimuth;
    u->lightElevation  = p.lightElevation;
    u->aspectRatio     = (float)_w / (float)_h;
    u->fovTan          = tanf(0.4363f);  // tan(25°) — comfortable FOV
    u->wobbleFreq      = p.wobbleFreq;
    u->wobbleAmp       = p.wobbleAmp;

    // Dispatch compute
    id<MTLCommandBuffer>      cmd = [_impl->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:_impl->pipeline];
    [enc setTexture:_impl->outTex atIndex:0];
    [enc setBuffer:_impl->uniforms offset:0 atIndex:0];

    MTLSize tg   = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake((_w + 15)/16, (_h + 15)/16, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Read back from shared-memory texture (zero-copy on Apple Silicon)
    [_impl->outTex getBytes:_impl->pixels.data()
                bytesPerRow:_w * 4
                 fromRegion:MTLRegionMake2D(0, 0, _w, _h)
                mipmapLevel:0];

    // Upload to OpenGL for ImGui
    glBindTexture(GL_TEXTURE_2D, _glTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _w, _h,
                    GL_RGBA, GL_UNSIGNED_BYTE, _impl->pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}
