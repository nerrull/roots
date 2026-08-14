// FitViewScene — a flat look at what the fit is actually doing.
//
// The mirror, the roots and the transition all composite the fit into something
// where a misplaced mask still reads as an effect. This scene refuses to do
// that: the fit's output, the training mask and the fitted mesh are drawn flat
// and stacked, in the one space they are all defined in (normalised frame
// coordinates), so that anything out of register looks wrong rather than
// interesting.
//
// The mesh is drawn wearing the colours sampled for it. That makes the texture
// mapping self-checking -- the mask's colours are lifted from the layer beneath
// it, so if the two do not agree, the sampling is reading the wrong pixels.
#pragma once
#ifndef __OBJC__
#error "fit_view_scene.h is ObjC++ only"
#endif

#import <Metal/Metal.h>

#include <string>
#include <vector>

class MetalContext;

class FitViewScene {
public:
    FitViewScene(const MetalContext& ctx, const std::string& shaderPath,
                 int w = 1280, int h = 720);

    bool valid() const { return pipe_bg_ != nil; }
    void ensureSize(int w, int h);

    // The layer under everything: whatever the fit produced.
    void setBackground(id<MTLTexture> tex) { bg_ = tex; }

    // The training mask at its own (fit-grid) resolution. Uploaded as R8 and
    // sampled nearest -- it is the exact pixel set the gradient sees, and a
    // filtered edge would imply a softness the training does not have.
    void setMask(const std::vector<unsigned char>& mask, int w, int h);
    void clearMask() { mask_w_ = 0; }

    // The fitted mesh: `uv` is 2 floats per vertex in normalised frame
    // coordinates (y-down), `rgb` 3 floats per vertex, `tris` triples of
    // vertex indices. Triangles are only re-uploaded when non-empty, since the
    // topology never changes after the first frame.
    void setMesh(const std::vector<float>& uv, const std::vector<float>& rgb,
                 const std::vector<int>& tris);
    void clearMesh() { vert_count_ = 0; }
    bool hasMesh() const { return vert_count_ > 0 && index_count_ > 0; }

    id<MTLTexture> render(id<MTLCommandBuffer> cb);
    id<MTLTexture> texture() const { return tex_; }

    // Display knobs, driven straight from the UI.
    float maskTint  = 0.6f;
    float bgDim     = 1.0f;
    float meshAlpha = 1.0f;
    bool  showMesh  = true;
    bool  wireframe = false;

private:
    void makeTexture();

    const MetalContext& ctx_;
    int w_ = 0, h_ = 0;

    id<MTLTexture> tex_  = nil;
    id<MTLTexture> bg_   = nil;
    id<MTLTexture> mask_ = nil;
    int mask_w_ = 0, mask_h_ = 0;

    id<MTLRenderPipelineState> pipe_bg_   = nil;
    id<MTLRenderPipelineState> pipe_mesh_ = nil;
    id<MTLRenderPipelineState> pipe_line_ = nil;

    id<MTLBuffer> vbuf_ = nil;
    id<MTLBuffer> ibuf_ = nil;
    id<MTLBuffer> lbuf_ = nil;          // wireframe line indices
    int vert_count_ = 0, index_count_ = 0, line_count_ = 0;
    std::vector<float> interleaved_;    // scratch: uv + rgb per vertex
};
