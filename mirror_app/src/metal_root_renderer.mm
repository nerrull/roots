#include "metal_root_renderer.h"
#include "metal_context.h"

#import <Foundation/Foundation.h>
#include <simd/simd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Baked tiling 3D value-noise fBm (verbatim port of RootRenderer's CPU bake).
// ---------------------------------------------------------------------------
static float latticeHash(int x, int y, int z) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u
               + (uint32_t)z * 1274126177u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (float)((h ^ (h >> 16)) & 0xFFFFFFu) / (float)0x1000000u;
}

static float tilingValueNoise(float px, float py, float pz, int per) {
    int ix = (int)std::floor(px), iy = (int)std::floor(py), iz = (int)std::floor(pz);
    float fx = px - ix, fy = py - iy, fz = pz - iz;
    float ux = fx * fx * (3.f - 2.f * fx);
    float uy = fy * fy * (3.f - 2.f * fy);
    float uz = fz * fz * (3.f - 2.f * fz);
    auto wrap = [per](int v) { return ((v % per) + per) % per; };
    float c[2][2][2];
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++)
                c[dz][dy][dx] = latticeHash(wrap(ix + dx), wrap(iy + dy), wrap(iz + dz));
    auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
    float x00 = lerp(c[0][0][0], c[0][0][1], ux), x10 = lerp(c[0][1][0], c[0][1][1], ux);
    float x01 = lerp(c[1][0][0], c[1][0][1], ux), x11 = lerp(c[1][1][0], c[1][1][1], ux);
    return lerp(lerp(x00, x10, uy), lerp(x01, x11, uy), uz);
}

// ---------------------------------------------------------------------------
// tiny vec helpers
// ---------------------------------------------------------------------------
namespace {
struct V3 { float x, y, z; };
inline float dot3(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline V3 cross3(V3 a, V3 b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
inline V3 norm3(V3 v) { float l = std::sqrt(dot3(v, v)); if (l < 1e-7f) l = 1.f; return {v.x/l, v.y/l, v.z/l}; }

// Per-node arc-length from each root base, for the travelling-pulse effect
// (shared by uploadSegments and addInstance). hopOffset pre-seeds each hop's
// base node so consecutive masks' pulse trains are phase-shifted.
std::vector<float> computeNodeDist(const float* nodesXYZ, const std::vector<int>& segs,
                                   int nNodes, float hopOffset) {
    std::vector<float> dist(std::max(1, nNodes), 0.f);
    if (nNodes <= 0) return dist;
    const int nSeg = (int)(segs.size() / 2);
    std::vector<char> isChild(nNodes, 0);
    for (int s = 0; s < nSeg; s++) {
        int cy = segs[2*s + 1];
        if (cy >= 0 && cy < nNodes) isChild[cy] = 1;
    }
    if (hopOffset != 0.0f) {
        float hopBase = 0.f;
        for (int i = 0; i < nNodes; i++)
            if (!isChild[i]) { dist[i] = hopBase; hopBase += hopOffset; }
    }
    for (int s = 0; s < nSeg; s++) {
        int pi = segs[2*s], ci = segs[2*s + 1];
        if (pi < 0 || ci < 0 || pi >= nNodes || ci >= nNodes) continue;
        float dx = nodesXYZ[3*ci]   - nodesXYZ[3*pi];
        float dy = nodesXYZ[3*ci+1] - nodesXYZ[3*pi+1];
        float dz = nodesXYZ[3*ci+2] - nodesXYZ[3*pi+2];
        dist[ci] = dist[pi] + std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    return dist;
}
}

// ===========================================================================

MetalRootRenderer::MetalRootRenderer(const MetalContext& ctx, const std::string& shaderDir,
                                     const std::string& sharedHeaderPath, int w, int h)
    : device_(ctx.device()), w_(w), h_(h) {
    // Pipelines: each MSL pass compiled with root_shared.h prepended.
    id<MTLLibrary> geomLib = ctx.newLibraryFromFiles({sharedHeaderPath, shaderDir + "/root_geom.metal"});
    id<MTLLibrary> faceLib = ctx.newLibraryFromFiles({sharedHeaderPath, shaderDir + "/root_face.metal"});
    id<MTLLibrary> leafLib = ctx.newLibraryFromFiles({sharedHeaderPath, shaderDir + "/root_leaf.metal"});
    id<MTLLibrary> fogLib  = ctx.newLibraryFromFiles({sharedHeaderPath, shaderDir + "/root_fog.metal"});
    id<MTLLibrary> aoLib   = ctx.newLibraryFromFiles({sharedHeaderPath, shaderDir + "/root_ao.metal"});
    id<MTLLibrary> blmLib  = ctx.newLibraryFromFiles({sharedHeaderPath, shaderDir + "/root_bloom.metal"});
    id<MTLLibrary> postLib = ctx.newLibraryFromFiles({sharedHeaderPath, shaderDir + "/root_post.metal"});
    if (!geomLib || !faceLib || !leafLib || !fogLib || !aoLib || !blmLib || !postLib) {
        fprintf(stderr, "MetalRootRenderer: shader compile failed\n"); return;
    }

    NSError* err = nil;
    {
        MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
        d.vertexFunction   = [geomLib newFunctionWithName:@"root_geom_vs"];
        d.fragmentFunction = [geomLib newFunctionWithName:@"root_geom_fs"];
        d.colorAttachments[0].pixelFormat = kColorFmt;
        d.depthAttachmentPixelFormat = kDepthFmt;
        geomPipe_ = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
        if (!geomPipe_) { NSLog(@"geom pipeline failed: %@", err); return; }
    }
    {
        MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
        d.vertexFunction   = [faceLib newFunctionWithName:@"root_face_vs"];
        d.fragmentFunction = [faceLib newFunctionWithName:@"root_face_fs"];
        d.colorAttachments[0].pixelFormat = kColorFmt;
        d.depthAttachmentPixelFormat = kDepthFmt;
        facePipe_ = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
        if (!facePipe_) { NSLog(@"face pipeline failed: %@", err); return; }
    }
    {
        MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
        d.vertexFunction   = [leafLib newFunctionWithName:@"root_leaf_vs"];
        d.fragmentFunction = [leafLib newFunctionWithName:@"root_leaf_fs"];
        d.colorAttachments[0].pixelFormat = kColorFmt;
        d.depthAttachmentPixelFormat = kDepthFmt;
        leafPipe_ = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
        if (!leafPipe_) { NSLog(@"leaf pipeline failed: %@", err); return; }
    }
    {
        MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
        d.vertexFunction   = [fogLib newFunctionWithName:@"root_fog_vs"];
        d.fragmentFunction = [fogLib newFunctionWithName:@"root_fog_fs"];
        d.colorAttachments[0].pixelFormat = kColorFmt;
        fogPipe_ = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
        if (!fogPipe_) { NSLog(@"fog pipeline failed: %@", err); return; }
    }
    // The post chain: all fullscreen, all single-attachment, no depth.
    auto makePost = [&](id<MTLLibrary> lib, const char* vs, const char* fs,
                        MTLPixelFormat fmt, bool additive) -> id<MTLRenderPipelineState> {
        MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
        d.vertexFunction   = [lib newFunctionWithName:[NSString stringWithUTF8String:vs]];
        d.fragmentFunction = [lib newFunctionWithName:[NSString stringWithUTF8String:fs]];
        d.colorAttachments[0].pixelFormat = fmt;
        if (additive) {
            // The bloom up-chain adds each level into the one above it, so the
            // blend state is what actually accumulates the glow.
            d.colorAttachments[0].blendingEnabled = YES;
            d.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
            d.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
            d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
        }
        NSError* e = nil;
        id<MTLRenderPipelineState> p = [device_ newRenderPipelineStateWithDescriptor:d error:&e];
        if (!p) NSLog(@"%s pipeline failed: %@", fs, e);
        return p;
    };
    {
        MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
        d.vertexFunction   = [fogLib newFunctionWithName:@"root_fog_vs"];
        d.fragmentFunction = [fogLib newFunctionWithName:@"root_fogvol_fs"];
        d.colorAttachments[0].pixelFormat = kColorFmt;
        fogVolPipe_ = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
        if (!fogVolPipe_) { NSLog(@"fog volumetric pipeline failed: %@", err); return; }
    }
    aoPipe_        = makePost(aoLib,   "root_ao_vs",    "root_ao_fs",         kAOFmt,    false);
    aoBlurPipe_    = makePost(aoLib,   "root_ao_vs",    "root_ao_blur_fs",    kAOFmt,    false);
    bloomDownPipe_ = makePost(blmLib,  "root_bloom_vs", "root_bloom_down_fs", kColorFmt, false);
    bloomUpPipe_   = makePost(blmLib,  "root_bloom_vs", "root_bloom_up_fs",   kColorFmt, true);
    postPipe_      = makePost(postLib, "root_post_vs",  "root_post_fs",       kColorFmt, false);
    if (!aoPipe_ || !aoBlurPipe_ || !bloomDownPipe_ || !bloomUpPipe_ || !postPipe_) return;

    {
        MTLDepthStencilDescriptor* dd = [[MTLDepthStencilDescriptor alloc] init];
        dd.depthCompareFunction = MTLCompareFunctionLess;
        dd.depthWriteEnabled = YES;
        depthState_ = [device_ newDepthStencilStateWithDescriptor:dd];
    }

    buildTargets();
    buildNoiseTexture();

    // Seed empty buffers so render() before uploadSegments is safe.
    uploadSegments({}, {}, {});
}

void MetalRootRenderer::buildTargets() {
    auto make2D = [&](MTLPixelFormat fmt, int w, int h, MTLStorageMode store) -> id<MTLTexture> {
        MTLTextureDescriptor* td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                               width:std::max(1, w)
                                                              height:std::max(1, h)
                                                           mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = store;
        return [device_ newTextureWithDescriptor:td];
    };

    const int ss = std::max(1, std::min(post.ssaa, 4));
    sw_ = std::max(1, w_ * ss);
    sh_ = std::max(1, h_ * ss);

    rootColorTex_ = make2D(kColorFmt, sw_, sh_, MTLStorageModePrivate);
    rootDepthTex_ = make2D(kDepthFmt, sw_, sh_, MTLStorageModePrivate);

    // Both possible outputs are Shared: whichever one render() returns has to be
    // readable by the headless capture paths without a blit, and which one that
    // is depends on a runtime flag.
    fogColorTex_ = make2D(kColorFmt, sw_, sh_, MTLStorageModeShared);
    postTex_     = make2D(kColorFmt, w_,  h_,  MTLStorageModeShared);

    const int ds = std::max(1, std::min(ao.downscale, 4));
    aoTex_     = make2D(kAOFmt, sw_ / ds, sh_ / ds, MTLStorageModePrivate);
    aoBlurTex_ = make2D(kAOFmt, sw_ / ds, sh_ / ds, MTLStorageModePrivate);

    // Sized from the OUTPUT resolution, not the supersampled scene resolution:
    // the integral's quality has nothing to do with how finely the geometry is
    // sampled, and tying it to the scene grid would silently quadruple its cost
    // the moment supersampling was turned on.
    const int fds = std::max(1, std::min(fog.downscale, 4));
    fogVolTex_ = make2D(kColorFmt, w_ / fds, h_ / fds, MTLStorageModePrivate);

    // Bloom levels halve from the scene resolution. Stop before any dimension
    // reaches zero rather than trusting the requested count -- at 320x240 with
    // no supersampling, five levels would ask for a 10x7 target and then a 5x3.
    bloomMips_.clear();
    int bw = sw_, bh = sh_;
    const int levels = std::max(1, std::min(post.bloomLevels, 8));
    for (int i = 0; i < levels; ++i) {
        bw /= 2; bh /= 2;
        if (bw < 4 || bh < 4) break;
        bloomMips_.push_back(make2D(kColorFmt, bw, bh, MTLStorageModePrivate));
    }

    builtSsaa_ = ss;
    builtAoDs_ = ds;
    builtFogDs_ = fds;
    builtBloomLevels_ = (int)bloomMips_.size();
    outTex_ = post.enabled ? postTex_ : fogColorTex_;
}

// The targets depend on runtime settings (supersample factor, AO downscale,
// bloom depth), all of which are live UI sliders. Rebuilding on every frame
// would thrash allocation; rebuilding never would silently ignore the slider.
void MetalRootRenderer::ensureTargets() {
    const int ss = std::max(1, std::min(post.ssaa, 4));
    const int ds = std::max(1, std::min(ao.downscale, 4));
    const int fds = std::max(1, std::min(fog.downscale, 4));
    const int lv = std::max(1, std::min(post.bloomLevels, 8));
    if (ss != builtSsaa_ || ds != builtAoDs_ || fds != builtFogDs_ ||
        (lv != builtBloomLevels_ && lv < 8 && builtBloomLevels_ < lv))
        buildTargets();
    outTex_ = post.enabled ? postTex_ : fogColorTex_;
}

void MetalRootRenderer::setTranche(int level) {
    tranche_ = std::max(0, std::min(level, 3));
    const bool t1 = tranche_ >= 1, t2 = tranche_ >= 2, t3 = tranche_ >= 3;

    // Tranche 1: the image-formation fundamentals.
    post.enabled = t1;
    post.tonemap = t1;
    env.hemiStrength = t1 ? 1.0f : 0.0f;
    // Needs a rebuildFace() to take effect; RootScene reads it when it builds
    // the mesh, and every caller of setTranche does so before growing.
    face.smoothNormals = t1;
    face.lightIntensity = t1 ? 1.8f : 3.2f;

    // Tranche 2: what the light does once it arrives.
    post.ssaa      = t2 ? 2 : 1;
    ao.enabled     = t2;
    env.envSpec    = t2 ? 0.6f  : 0.0f;
    env.rimStrength= t2 ? 0.10f : 0.0f;
    env.sssWrap    = t2 ? 0.55f : 0.0f;
    env.sssTrans   = t2 ? 0.35f : 0.0f;
    detail.strength = t2 ? 0.55f : 0.0f;
    detail.rough    = t2 ? 0.45f : 0.0f;
    detail.tint     = t2 ? 0.14f : 0.0f;
    // Cook-Torrance rather than the ad-hoc Phong lobe. Phong's specular here is
    // a fixed-width white streak with no Fresnel and no roughness, applied
    // identically to every tube -- which is most of what made the roots read as
    // extruded plastic, and no amount of surface detail fixes a highlight that
    // is the wrong shape to begin with.
    shaderMode = t2 ? ShaderMode::PBR : ShaderMode::Phong;
    pbr.roughness = t2 ? 0.80f : 0.70f;

    // Tranche 3: the camera and the film.
    post.bloom     = t3;
    post.dof       = t3;
    post.vignette  = t3 ? 0.22f : 0.0f;
    post.grain     = t3 ? 0.030f : 0.0f;
    post.dither    = t3;
    post.fogDither = t3 ? 1.0f : 0.0f;

    ensureTargets();
}

void MetalRootRenderer::buildNoiseTexture() {
    const int N = 128;
    std::vector<unsigned char> vox((size_t)N * N * N);
    const float basePer = ROOT_NOISE_TILE_PERIOD;
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                float px = (float)x / N * basePer;
                float py = (float)y / N * basePer;
                float pz = (float)z / N * basePer;
                float v = 0.5000f * tilingValueNoise(px,     py,     pz,     (int)basePer)
                        + 0.2500f * tilingValueNoise(px*2,   py*2,   pz*2,   (int)basePer*2)
                        + 0.1250f * tilingValueNoise(px*4,   py*4,   pz*4,   (int)basePer*4)
                        + 0.0625f * tilingValueNoise(px*8,   py*8,   pz*8,   (int)basePer*8);
                v *= 1.0f / 0.9375f;
                vox[((size_t)z * N + y) * N + x] = (unsigned char)(v * 255.0f + 0.5f);
            }
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType3D;
    td.pixelFormat = MTLPixelFormatR8Unorm;
    td.width = N; td.height = N; td.depth = N;
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    noiseTex_ = [device_ newTextureWithDescriptor:td];
    [noiseTex_ replaceRegion:MTLRegionMake3D(0, 0, 0, N, N, N)
                 mipmapLevel:0
                       slice:0
                   withBytes:vox.data()
                 bytesPerRow:N
               bytesPerImage:(NSUInteger)N * N];
}

void MetalRootRenderer::resize(int w, int h) {
    if (w < 1 || h < 1 || (w == w_ && h == h_)) return;
    w_ = w; h_ = h;
    buildTargets();
}

// Encode one fullscreen-triangle pass. Every stage of the post chain is the same
// shape -- one pipeline, one uniform block, up to two source textures, one
// colour attachment -- and writing that out five times was five chances to bind
// a texture to the wrong index.
namespace {
struct FSPass {
    id<MTLCommandBuffer> cb;
    id<MTLRenderPipelineState> pipe;
    id<MTLTexture> dst;
    const void* uniforms; size_t uniformBytes;
    id<MTLTexture> tex0 = nil, tex1 = nil, tex2 = nil, tex3 = nil;
    bool load = false;   // keep the destination's contents (the bloom up-chain)
};
void encodeFS(const FSPass& p) {
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = p.dst;
    rp.colorAttachments[0].loadAction = p.load ? MTLLoadActionLoad : MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = [p.cb renderCommandEncoderWithDescriptor:rp];
    [e setRenderPipelineState:p.pipe];
    [e setFragmentBytes:p.uniforms length:p.uniformBytes atIndex:0];
    if (p.tex0) [e setFragmentTexture:p.tex0 atIndex:0];
    if (p.tex1) [e setFragmentTexture:p.tex1 atIndex:1];
    if (p.tex2) [e setFragmentTexture:p.tex2 atIndex:2];
    if (p.tex3) [e setFragmentTexture:p.tex3 atIndex:3];
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [e endEncoding];
}
}  // namespace

id<MTLBuffer> MetalRootRenderer::makeBuffer(const void* data, size_t bytes) {
    if (bytes == 0) bytes = 4;   // Metal rejects zero-length buffers
    id<MTLBuffer> b = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (data) memcpy(b.contents, data, bytes);
    return b;
}

void MetalRootRenderer::uploadSegments(const std::vector<float>& nodesXYZ,
                                       const std::vector<int>&   segs,
                                       const std::vector<float>& radii,
                                       const std::vector<int>*   groups,
                                       const std::vector<int>*   prims,
                                       const std::vector<float>* frames,
                                       const std::vector<float>* aux) {
    const int nSeg   = (int)(segs.size() / 2);
    const int nNodes = (int)(nodesXYZ.size() / 3);
    segCount_ = nSeg;

    // nodes: packed 3 floats/node (>=1 for a valid buffer).
    std::vector<float> nodeData = nodesXYZ;
    if (nodeData.empty()) nodeData.assign(3, 0.f);

    std::vector<int> segData = segs;
    if (segData.empty()) segData.assign(2, 0);

    std::vector<float> radData = radii;
    if (radData.empty()) radData.push_back(0.f);

    // Per-node arc-length from root base (pulse effect), incl. hopOffset seeding.
    std::vector<float> distData =
        computeNodeDist(nodesXYZ.empty() ? nodeData.data() : nodesXYZ.data(),
                        segs, nNodes, pulse.hopOffset);

    const size_t segCap = std::max<size_t>(1, nSeg);

    std::vector<int> grpData;
    if (groups && !groups->empty()) grpData = *groups;
    grpData.resize(segCap, 0);

    std::vector<int> primData;
    if (prims && !prims->empty()) primData = *prims;
    primData.resize(segCap, 0);

    std::vector<float> frameData;
    if (frames && !frames->empty()) frameData = *frames;
    frameData.resize(segCap * 4, 0.f);

    std::vector<float> auxData;
    if (aux && !aux->empty()) auxData = *aux;
    else {
        auxData.resize(segCap * 4);
        for (size_t i = 0; i < auxData.size(); i += 4) {
            auxData[i] = 0.f; auxData[i+1] = 1.f; auxData[i+2] = 0.f; auxData[i+3] = 0.f;
        }
    }
    auxData.resize(segCap * 4, 0.f);

    nodeBuf_  = makeBuffer(nodeData.data(),  nodeData.size()  * sizeof(float));
    segBuf_   = makeBuffer(segData.data(),   segData.size()   * sizeof(int));
    radBuf_   = makeBuffer(radData.data(),   radData.size()   * sizeof(float));
    distBuf_  = makeBuffer(distData.data(),  distData.size()  * sizeof(float));
    grpBuf_   = makeBuffer(grpData.data(),   grpData.size()   * sizeof(int));
    primBuf_  = makeBuffer(primData.data(),  primData.size()  * sizeof(int));
    frameBuf_ = makeBuffer(frameData.data(), frameData.size() * sizeof(float));
    auxBuf_   = makeBuffer(auxData.data(),   auxData.size()   * sizeof(float));
}

void MetalRootRenderer::uploadFaceMesh(const std::vector<float>& interleaved) {
    faceVertCount_ = (int)(interleaved.size() / 12);
    faceBuf_ = faceVertCount_ > 0
        ? makeBuffer(interleaved.data(), interleaved.size() * sizeof(float))
        : nil;
}

void MetalRootRenderer::uploadLeafMesh(const std::vector<float>& interleaved) {
    leafVertCount_ = (int)(interleaved.size() / 12);
    leafBuf_ = leafVertCount_ > 0
        ? makeBuffer(interleaved.data(), interleaved.size() * sizeof(float))
        : nil;
}

// Constant per-segment attributes (prim=0, grp=0, frame=0, aux=(0,1,0,0)) that
// capsule instances share; grown to the largest instance's LOD0 segment count.
void MetalRootRenderer::ensureDefaults(int segCount) {
    if (segCount <= defCap_) return;
    int n = std::max(1, segCount);
    std::vector<int>   zeros(n, 0);
    std::vector<float> frame((size_t)n * 4, 0.f);
    std::vector<float> aux((size_t)n * 4);
    for (int i = 0; i < n; i++) { aux[4*i]=0.f; aux[4*i+1]=1.f; aux[4*i+2]=0.f; aux[4*i+3]=0.f; }
    defPrim_  = makeBuffer(zeros.data(), (size_t)n * sizeof(int));
    defGrp_   = makeBuffer(zeros.data(), (size_t)n * sizeof(int));
    defFrame_ = makeBuffer(frame.data(), frame.size() * sizeof(float));
    defAux_   = makeBuffer(aux.data(),   aux.size()   * sizeof(float));
    defCap_ = n;
}

int MetalRootRenderer::addInstance(const std::vector<float>& nodesXYZ,
                                   const std::vector<int>&   segs,
                                   const std::vector<float>& radii,
                                   const InstancePlacement&  place) {
    const int nNodes = (int)(nodesXYZ.size() / 3);
    const int nSeg   = (int)(segs.size() / 2);
    if (nNodes == 0 || nSeg == 0) return -1;

    // Bake nodes to world space (scale, yaw about Y, translate).
    const float s = place.scale, cy = cosf(place.rotYaw), sy = sinf(place.rotYaw);
    std::vector<float> wnodes((size_t)nNodes * 3);
    float lo[3], hi[3];
    for (int i = 0; i < nNodes; i++) {
        float lx = nodesXYZ[3*i] * s, ly = nodesXYZ[3*i+1] * s, lz = nodesXYZ[3*i+2] * s;
        float wx = lx * cy + lz * sy + place.translate[0];
        float wy = ly + place.translate[1];
        float wz = -lx * sy + lz * cy + place.translate[2];
        wnodes[3*i] = wx; wnodes[3*i+1] = wy; wnodes[3*i+2] = wz;
        if (i == 0) { lo[0]=hi[0]=wx; lo[1]=hi[1]=wy; lo[2]=hi[2]=wz; }
        else {
            lo[0]=std::min(lo[0],wx); hi[0]=std::max(hi[0],wx);
            lo[1]=std::min(lo[1],wy); hi[1]=std::max(hi[1],wy);
            lo[2]=std::min(lo[2],wz); hi[2]=std::max(hi[2],wz);
        }
    }

    Instance inst;
    inst.center[0] = 0.5f*(lo[0]+hi[0]);
    inst.center[1] = 0.5f*(lo[1]+hi[1]);
    inst.center[2] = 0.5f*(lo[2]+hi[2]);
    float rad = 0.f;
    for (int i = 0; i < nNodes; i++) {
        float dx=wnodes[3*i]-inst.center[0], dy=wnodes[3*i+1]-inst.center[1], dz=wnodes[3*i+2]-inst.center[2];
        rad = std::max(rad, std::sqrt(dx*dx+dy*dy+dz*dz));
    }
    inst.radius = rad;

    inst.node = makeBuffer(wnodes.data(), wnodes.size() * sizeof(float));
    std::vector<float> distData = computeNodeDist(wnodes.data(), segs, nNodes, pulse.hopOffset);
    inst.dist = makeBuffer(distData.data(), distData.size() * sizeof(float));

    // LOD thresholds by radius percentile: coarser LODs drop the thinnest
    // laterals first (invisible once the system is small on screen).
    std::vector<float> sortedR = radii;
    std::sort(sortedR.begin(), sortedR.end());
    auto pct = [&](float f) -> float {
        if (sortedR.empty()) return 0.f;
        int idx = std::min((int)sortedR.size() - 1, std::max(0, (int)(f * sortedR.size())));
        return sortedR[idx];
    };
    const float thr[4] = { -1.f, pct(0.40f), pct(0.65f), pct(0.82f) };

    for (int k = 0; k < 4; k++) {
        std::vector<int>   lseg;
        std::vector<float> lrad;
        lseg.reserve(nSeg * 2);
        for (int j = 0; j < nSeg; j++) {
            float r = (j < (int)radii.size()) ? radii[j] : 0.f;
            if (r >= thr[k]) {
                lseg.push_back(segs[2*j]); lseg.push_back(segs[2*j+1]);
                lrad.push_back(r);
            }
        }
        InstanceLod lod;
        lod.segCount = (int)lrad.size();
        lod.seg = makeBuffer(lseg.empty() ? nullptr : lseg.data(),
                             std::max<size_t>(1, lseg.size()) * sizeof(int));
        lod.rad = makeBuffer(lrad.empty() ? nullptr : lrad.data(),
                             std::max<size_t>(1, lrad.size()) * sizeof(float));
        inst.lods.push_back(lod);
    }

    ensureDefaults(inst.lods[0].segCount);
    instances_.push_back(inst);
    return (int)instances_.size() - 1;
}

void MetalRootRenderer::clearInstances() { instances_.clear(); }

id<MTLTexture> MetalRootRenderer::render(id<MTLCommandBuffer> cb,
                                         float azimuth, float elevation, float radius,
                                         const float target3[3], float fov,
                                         const float lightDir3[3]) {
    if (!valid()) return nil;
    ensureTargets();

    // --- Camera (verbatim port of RootRenderer::render) ---
    float cosEl = cosf(elevation), sinEl = sinf(elevation);
    float cosAz = cosf(azimuth),   sinAz = sinf(azimuth);
    float ex = target3[0] + radius * cosEl * sinAz;
    float ey = target3[1] + radius * sinEl;
    float ez = target3[2] + radius * cosEl * cosAz;

    V3 fwd = norm3({target3[0]-ex, target3[1]-ey, target3[2]-ez});
    V3 wup = {0.f, 1.f, 0.f};
    V3 rawRgt = cross3(fwd, wup);
    V3 rgt = (dot3(rawRgt, rawRgt) > 1e-8f) ? norm3(rawRgt) : V3{1.f, 0.f, 0.f};
    V3 up  = cross3(rgt, fwd);

    const float nearZ = 0.01f, farZ = 500.0f;
    float f   = 1.0f / tanf(fov);
    float asp = (float)sw_ / (float)sh_;   // == w_/h_; the scene passes run supersampled
    float fn  = -(farZ + nearZ) / (farZ - nearZ);
    float fn2 = -2.0f * farZ * nearZ / (farZ - nearZ);

    simd_float4x4 proj = simd_matrix(
        (simd_float4){f/asp, 0, 0, 0},
        (simd_float4){0, f, 0, 0},
        (simd_float4){0, 0, fn, -1},
        (simd_float4){0, 0, fn2, 0});

    float dre = dot3(rgt, {ex, ey, ez});
    float due = dot3(up,  {ex, ey, ez});
    float dfe = dot3(fwd, {ex, ey, ez});

    simd_float4x4 view = simd_matrix(
        (simd_float4){rgt.x,  up.x, -fwd.x, 0},
        (simd_float4){rgt.y,  up.y, -fwd.y, 0},
        (simd_float4){rgt.z,  up.z, -fwd.z, 0},
        (simd_float4){-dre,  -due,   dfe,   1});

    simd_float4x4 vp = simd_mul(proj, view);
    simd_float3x3 cam = simd_matrix(
        (simd_float3){rgt.x, rgt.y, rgt.z},
        (simd_float3){up.x,  up.y,  up.z},
        (simd_float3){fwd.x, fwd.y, fwd.z});

    // --- Animate wisps ---
    int active = std::min(std::max(wispCount, 0), MAX_WISPS);
    std::vector<RootWisp> wispBuf(std::max(1, active));
    for (int i = 0; i < active; i++) {
        const auto& w = wisps[i];
        float t = wispTime;
        float wx = w.basePos[0] + w.driftRadius * sinf(w.driftSpeed * t        + w.phase[0]);
        float wy = w.basePos[1] + w.driftRadius * cosf(w.driftSpeed * t * 0.7f + w.phase[1]);
        float wz = w.basePos[2] + w.driftRadius * sinf(w.driftSpeed * t * 1.3f + w.phase[2]);
        wispBuf[i].pos   = (simd_float4){wx, wy, wz, w.intensity};
        wispBuf[i].color = (simd_float4){w.color[0], w.color[1], w.color[2], 0.f};
    }

    // --- Geometry uniforms ---
    RootGeomU gu = {};
    gu.viewProj = vp;
    gu.cam = cam;
    gu.eye = (simd_float4){ex, ey, ez, 0};
    gu.baseColor  = (simd_float4){mat.baseColor[0], mat.baseColor[1], mat.baseColor[2], 0};
    gu.baseColor2 = (simd_float4){mat.baseColor2[0], mat.baseColor2[1], mat.baseColor2[2], 0};
    gu.specColor  = (simd_float4){mat.specColor[0], mat.specColor[1], mat.specColor[2], 0};
    gu.lightDir   = (simd_float4){lightDir3[0], lightDir3[1], lightDir3[2], 0};
    gu.pulseColor = (simd_float4){pulse.color[0], pulse.color[1], pulse.color[2], 0};
    gu.res = (simd_float2){(float)sw_, (float)sh_};
    gu.fov = fov;
    gu.skyColor    = (simd_float4){env.skyColor[0], env.skyColor[1], env.skyColor[2], 0};
    gu.groundColor = (simd_float4){env.groundColor[0], env.groundColor[1], env.groundColor[2], 0};
    gu.sssTint     = (simd_float4){env.sssTint[0], env.sssTint[1], env.sssTint[2], 0};
    gu.hemiStrength = env.hemiStrength;
    gu.envSpec      = env.envSpec;
    gu.rimStrength  = env.rimStrength;
    gu.sssWrap      = env.sssWrap;
    gu.sssTrans     = env.sssTrans;
    gu.sssPower     = env.sssPower;
    gu.detailStrength = detail.strength;
    gu.detailScale    = detail.scale;
    gu.detailStretch  = detail.stretch;
    gu.detailRough    = detail.rough;
    gu.detailTint     = detail.tint;
    gu.keyColor = (simd_float4){env.keyColor[0] * env.keyIntensity,
                                env.keyColor[1] * env.keyIntensity,
                                env.keyColor[2] * env.keyIntensity, 0};
    gu.radiusScale = radiusScale; gu.radiusMin = radiusMin; gu.radiusMax = radiusMax;
    gu.ambient = mat.ambient; gu.diffuse = mat.diffuse; gu.shininess = mat.shininess;
    gu.colorNoiseScale = mat.colorNoiseScale; gu.colorNoiseStrength = mat.colorNoiseStrength;
    gu.metallic = pbr.metallic; gu.roughness = pbr.roughness;
    gu.pulseSpeed = pulse.speed; gu.pulseSpacing = pulse.spacing; gu.pulseWidth = pulse.width;
    gu.pulseIntensity = pulse.intensity; gu.pulseTime = pulse.time;
    gu.shaderMode = (int)shaderMode;
    gu.wispCount = active;
    gu.pulseEnabled = pulse.enabled ? 1 : 0;
    gu.cullPx = subpixelCull ? 0.75f : 0.0f;
    int pc = paletteCount < 0 ? 0 : (paletteCount > MAX_GROUPS ? MAX_GROUPS : paletteCount);
    gu.paletteCount = pc;
    for (int i = 0; i < pc; i++) {
        gu.palette[i] = (simd_float4){palette[i][0], palette[i][1], palette[i][2], 0};
        bool hasTip = i < paletteTipCount;
        gu.paletteTip[i] = hasTip
            ? (simd_float4){paletteTip[i][0], paletteTip[i][1], paletteTip[i][2], 0}
            : gu.palette[i];
    }

    // --- Pass 1: geometry ---
    bool invert = (shaderMode == ShaderMode::Invert);
    MTLRenderPassDescriptor* gp = [MTLRenderPassDescriptor renderPassDescriptor];
    gp.colorAttachments[0].texture = rootColorTex_;
    gp.colorAttachments[0].loadAction = MTLLoadActionClear;
    gp.colorAttachments[0].clearColor =
        // Alpha 0: the background has no environment term for the AO to
        // attenuate (see root_geom.metal's alpha convention).
        MTLClearColorMake(invert ? 0.0 : 0.12, invert ? 0.0 : 0.08, invert ? 0.0 : 0.05, 0.0);
    gp.colorAttachments[0].storeAction = MTLStoreActionStore;
    gp.depthAttachment.texture = rootDepthTex_;
    gp.depthAttachment.loadAction = MTLLoadActionClear;
    gp.depthAttachment.clearDepth = 1.0;
    gp.depthAttachment.storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> ge = [cb renderCommandEncoderWithDescriptor:gp];
    [ge setRenderPipelineState:geomPipe_];
    [ge setDepthStencilState:depthState_];
    [ge setCullMode:MTLCullModeNone];
    [ge setVertexBytes:&gu length:sizeof(gu) atIndex:8];
    [ge setFragmentBytes:&gu length:sizeof(gu) atIndex:8];
    [ge setFragmentBytes:wispBuf.data() length:wispBuf.size() * sizeof(RootWisp) atIndex:9];
    [ge setFragmentTexture:noiseTex_ atIndex:0];

    lastVisibleInstances = 0; lastCulledInstances = 0; lastDrawnSegments = 0;

    // Bind one capsule set's buffers and draw it (6 verts x segc instances).
    auto drawSet = [&](id<MTLBuffer> node, id<MTLBuffer> seg, id<MTLBuffer> rad,
                       id<MTLBuffer> dist, id<MTLBuffer> grp, id<MTLBuffer> prim,
                       id<MTLBuffer> frame, id<MTLBuffer> aux, int segc) {
        if (segc <= 0) return;
        [ge setVertexBuffer:node  offset:0 atIndex:0];
        [ge setVertexBuffer:seg   offset:0 atIndex:1];
        [ge setVertexBuffer:rad   offset:0 atIndex:2];
        [ge setVertexBuffer:prim  offset:0 atIndex:5];
        [ge setVertexBuffer:frame offset:0 atIndex:6];
        [ge setFragmentBuffer:node  offset:0 atIndex:0];
        [ge setFragmentBuffer:seg   offset:0 atIndex:1];
        [ge setFragmentBuffer:rad   offset:0 atIndex:2];
        [ge setFragmentBuffer:dist  offset:0 atIndex:3];
        [ge setFragmentBuffer:grp   offset:0 atIndex:4];
        [ge setFragmentBuffer:prim  offset:0 atIndex:5];
        [ge setFragmentBuffer:frame offset:0 atIndex:6];
        [ge setFragmentBuffer:aux   offset:0 atIndex:7];
        [ge drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6
             instanceCount:(NSUInteger)segc];
        lastDrawnSegments += segc;
    };

    // The live/dynamic system (re-uploaded each frame) draws in full, unculled.
    if (segCount_ > 0) {
        drawSet(nodeBuf_, segBuf_, radBuf_, distBuf_, grpBuf_, primBuf_, frameBuf_, auxBuf_, segCount_);
        lastVisibleInstances++;
    }

    // Cached instances: frustum-cull whole systems, pick a LOD by projected size.
    if (!instances_.empty()) {
        // Six frustum planes from vp (Gribb-Hartmann). vp is column-major, so
        // clip-space row i = (col0[i], col1[i], col2[i], col3[i]).
        simd_float4 r0 = {vp.columns[0].x, vp.columns[1].x, vp.columns[2].x, vp.columns[3].x};
        simd_float4 r1 = {vp.columns[0].y, vp.columns[1].y, vp.columns[2].y, vp.columns[3].y};
        simd_float4 r2 = {vp.columns[0].z, vp.columns[1].z, vp.columns[2].z, vp.columns[3].z};
        simd_float4 r3 = {vp.columns[0].w, vp.columns[1].w, vp.columns[2].w, vp.columns[3].w};
        simd_float4 planes[6] = { r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2 };
        for (int k = 0; k < 6; k++) {
            float l = simd_length(planes[k].xyz);
            if (l > 1e-8f) planes[k] /= l;
        }
        simd_float3 eye = {ex, ey, ez};
        float pxScale = f * 0.5f * (float)h_;   // NDC radius r*f/d -> output pixels
        const float lodThresh[3] = {300.f, 120.f, 45.f};   // px boundaries between LODs

        for (auto& inst : instances_) {
            simd_float3 c = {inst.center[0], inst.center[1], inst.center[2]};
            if (cullInstances) {
                bool out = false;
                for (int k = 0; k < 6; k++)
                    if (simd_dot(planes[k].xyz, c) + planes[k].w < -inst.radius) { out = true; break; }
                if (out) { lastCulledInstances++; continue; }
            }
            float dist = simd_length(c - eye);
            if (dist < 1e-3f) dist = 1e-3f;
            float screenPx = inst.radius * pxScale / dist;
            if (screenPx < instanceCullPx) { lastCulledInstances++; continue; }

            int nl = (int)inst.lods.size();
            int lod = 0;
            for (int k = 0; k < 3 && k < nl - 1; k++)
                if (screenPx < lodThresh[k] * lodBias) lod = k + 1;
            const InstanceLod& L = inst.lods[lod];
            drawSet(inst.node, L.seg, L.rad, inst.dist,
                    defGrp_, defPrim_, defFrame_, defAux_, L.segCount);
            lastVisibleInstances++;
        }
    }

    // Face mid-geometry pass: mask triangles into the same colour+depth target,
    // depth-composited against the capsules (matches RootRenderer's midGeometryHook).
    if (faceVertCount_ > 0 && facePipe_) {
        RootFaceU ffu = {};
        ffu.viewProj = vp;
        ffu.eye = gu.eye;
        ffu.lightDir = gu.lightDir;
        ffu.veinColor = (simd_float4){face.veinColor[0], face.veinColor[1], face.veinColor[2], 0};
        ffu.lightIntensity = face.lightIntensity;
        ffu.lightFalloff = face.lightFalloff;
        ffu.specStrength = face.specStrength;
        ffu.veinScale = face.veinScale;
        ffu.veinStrength = face.veinStrength;
        ffu.roughness = face.roughness;
        ffu.metallic = face.metallic;
        ffu.reliefStrength = face.reliefStrength;
        ffu.reliefScale = face.reliefScale;
        ffu.keyColor = gu.keyColor;
        ffu.spotLightDist = face.spotLightDist;
        // 90 degrees or wider means "no cone"; the shader takes < -1 as the
        // disable sentinel so it can skip the work entirely.
        if (face.spotOuterDeg >= 89.9f) {
            ffu.spotCosOuter = -2.0f; ffu.spotCosInner = -2.0f;
        } else {
            ffu.spotCosOuter = std::cos(face.spotOuterDeg * 3.14159265f / 180.f);
            ffu.spotCosInner = std::cos(std::min(face.spotInnerDeg,
                                                 face.spotOuterDeg - 1.f)
                                        * 3.14159265f / 180.f);
        }
        ffu.skyColor = gu.skyColor;
        ffu.groundColor = gu.groundColor;
        ffu.sssTint = gu.sssTint;
        ffu.hemiStrength = env.hemiStrength;
        ffu.envSpec = env.envSpec;
        ffu.rimStrength = env.rimStrength;
        ffu.sssWrap = env.sssWrap;
        ffu.sssTrans = env.sssTrans;
        ffu.sssPower = env.sssPower;
        [ge setRenderPipelineState:facePipe_];
        [ge setDepthStencilState:depthState_];
        [ge setCullMode:MTLCullModeNone];
        [ge setVertexBuffer:faceBuf_ offset:0 atIndex:0];
        [ge setVertexBytes:&ffu length:sizeof(ffu) atIndex:1];
        [ge setFragmentBytes:&ffu length:sizeof(ffu) atIndex:1];
        [ge drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0
                vertexCount:(NSUInteger)faceVertCount_];
    }

    // Leaf mid-geometry pass: meshed leaves, same targets and depth convention
    // as the face pass, shaded as thin translucent lamina instead of stone.
    if (leafVertCount_ > 0 && leafPipe_) {
        RootLeafU lu = {};
        lu.viewProj     = vp;
        lu.eye          = gu.eye;
        lu.lightDir     = gu.lightDir;
        lu.skyColor     = gu.skyColor;
        lu.groundColor  = gu.groundColor;
        lu.sssTint      = gu.sssTint;
        lu.diffuse      = leaf.diffuse;
        lu.specStrength = leaf.specStrength;
        lu.roughness    = leaf.roughness;
        lu.hemiStrength = env.hemiStrength;
        lu.rimStrength  = env.rimStrength;
        lu.sssTrans     = leaf.sssTrans;
        lu.sssPower     = leaf.sssPower;
        [ge setRenderPipelineState:leafPipe_];
        [ge setDepthStencilState:depthState_];
        [ge setCullMode:MTLCullModeNone];
        [ge setVertexBuffer:leafBuf_ offset:0 atIndex:0];
        [ge setVertexBytes:&lu length:sizeof(lu) atIndex:1];
        [ge setFragmentBytes:&lu length:sizeof(lu) atIndex:1];
        [ge drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0
                vertexCount:(NSUInteger)leafVertCount_];
    }
    [ge endEncoding];

    // --- Pass 2: ambient occlusion (depth only, half resolution) ---
    // Runs before the fog because the fog pass is what applies it, and it reads
    // the geometry pass's depth, which is finished by now.
    const bool aoOn = ao.enabled && aoPipe_ && aoBlurPipe_;
    if (aoOn) {
        const int ds = std::max(1, builtAoDs_);
        RootAOU au = {};
        au.cam = cam;
        au.eye = gu.eye;
        au.res = (simd_float2){(float)(sw_ / ds), (float)(sh_ / ds)};
        au.fov = fov; au.nearZ = nearZ; au.farZ = farZ;
        au.radius = ao.radius; au.intensity = ao.intensity; au.bias = ao.bias;
        au.samples = std::max(1, ao.samples);
        au.blurDir = 0;
        encodeFS({cb, aoPipe_, aoTex_, &au, sizeof(au), rootDepthTex_});
        // Separable bilateral blur, ping-ponging so neither pass reads the
        // texture it is writing.
        encodeFS({cb, aoBlurPipe_, aoBlurTex_, &au, sizeof(au), aoTex_, rootDepthTex_});
        au.blurDir = 1;
        encodeFS({cb, aoBlurPipe_, aoTex_, &au, sizeof(au), aoBlurTex_, rootDepthTex_});
    }

    // --- Pass 3: fog ---
    RootFogU fu = {};
    fu.cam = cam;
    fu.eye = gu.eye;
    fu.fogColor = (simd_float4){fog.color[0], fog.color[1], fog.color[2], 0};
    fu.res = gu.res;
    fu.fov = fov; fu.nearZ = nearZ; fu.farZ = farZ;
    fu.aoEnabled = aoOn ? 1 : 0;
    fu.fogDither = post.fogDither;
    fu.fogDensity = (fog.enabled && fog.visibility > 1e-3f) ? 1.0f / fog.visibility : 0.0f;
    fu.fogHeightRef = fog.heightRef;
    fu.fogHeightScale = fog.heightScale;
    fu.fogNoiseScale = fog.noiseScale;
    fu.fogNoiseStrength = fog.noiseStrength;
    fu.fogNoiseContrast = fog.noiseContrast;
    fu.fogStart = fog.startAuto ? radius * fog.startFrac : fog.startDist;
    fu.fogSteps = fog.steps;
    fu.fogScatter = fog.scatter;
    fu.fogAnisotropy = fog.anisotropy;
    fu.lightDir = gu.lightDir;
    fu.keyColor = gu.keyColor;
    // Two advection vectors that are deliberately not parallel and not
    // commensurate in speed: with one, or with two that differ only in
    // magnitude, the field still resolves into a single sliding direction.
    // Both carry a Y component, which the original had none of at all.
    {
        const float t = fog.driftTime;
        fu.fogDrift0 = (simd_float4){ t * 0.050f,  t * 0.021f, t * 0.033f, 0.f};
        fu.fogDrift1 = (simd_float4){-t * 0.027f,  t * 0.044f, t * 0.012f, 0.f};
    }
    fu.wispGlowStrength = wispGlowStrength;
    fu.axisLength = overlay.axisLength; fu.gridSpacing = overlay.gridSpacing;
    fu.showAxes = overlay.showAxes ? 1 : 0; fu.showGrid = overlay.showGrid ? 1 : 0;
    fu.wispCount = active;

    // --- Pass 3a: the volumetric integral, at scene resolution / fog.downscale.
    // Its own uniform copy differs only in `res`, which is what the ray
    // reconstruction divides by; the aspect is unchanged, so the rays it builds
    // are the same rays the composite reconstructs.
    {
        RootFogU vu = fu;
        vu.res = (simd_float2){(float)(w_ / builtFogDs_), (float)(h_ / builtFogDs_)};
        encodeFS({cb, fogVolPipe_, fogVolTex_, &vu, sizeof(vu), rootDepthTex_, noiseTex_});
    }

    // --- Pass 3b: composite ---
    MTLRenderPassDescriptor* fp = [MTLRenderPassDescriptor renderPassDescriptor];
    fp.colorAttachments[0].texture = fogColorTex_;
    fp.colorAttachments[0].loadAction = MTLLoadActionClear;
    fp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    fp.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> fe = [cb renderCommandEncoderWithDescriptor:fp];
    [fe setRenderPipelineState:fogPipe_];
    [fe setFragmentBytes:&fu length:sizeof(fu) atIndex:0];
    [fe setFragmentBytes:wispBuf.data() length:wispBuf.size() * sizeof(RootWisp) atIndex:1];
    [fe setFragmentTexture:rootColorTex_ atIndex:0];
    [fe setFragmentTexture:rootDepthTex_ atIndex:1];
    [fe setFragmentTexture:noiseTex_ atIndex:2];
    // Bound whether or not AO ran: an unbound texture read is undefined even
    // behind a branch the shader never takes.
    [fe setFragmentTexture:aoTex_ atIndex:3];
    [fe setFragmentTexture:fogVolTex_ atIndex:4];
    [fe drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [fe endEncoding];

    if (!post.enabled) { outTex_ = fogColorTex_; return fogColorTex_; }

    // --- Pass 4: bloom chain ---
    const bool bloomOn = post.bloom && !bloomMips_.empty()
                      && bloomDownPipe_ && bloomUpPipe_;
    if (bloomOn) {
        // Down: scene -> mip0 -> mip1 -> ... Each step halves, and the shader
        // assumes exactly that, so the source's texel size is all it needs.
        for (size_t i = 0; i < bloomMips_.size(); ++i) {
            id<MTLTexture> src = (i == 0) ? fogColorTex_ : bloomMips_[i - 1];
            RootBloomU bu = {};
            bu.srcTexel = (simd_float2){1.0f / (float)src.width, 1.0f / (float)src.height};
            bu.threshold = post.bloomThreshold;
            bu.radius = post.bloomRadius;
            bu.prefilter = (i == 0) ? 1 : 0;
            encodeFS({cb, bloomDownPipe_, bloomMips_[i], &bu, sizeof(bu), src});
        }
        // Up: each level is tent-filtered and *added* into the level above it
        // (loadAction Load + additive blend), so mip0 ends up holding the sum of
        // every scale of glow.
        for (size_t i = bloomMips_.size() - 1; i > 0; --i) {
            id<MTLTexture> src = bloomMips_[i];
            RootBloomU bu = {};
            bu.srcTexel = (simd_float2){1.0f / (float)src.width, 1.0f / (float)src.height};
            bu.radius = post.bloomRadius;
            encodeFS({cb, bloomUpPipe_, bloomMips_[i - 1], &bu, sizeof(bu), src,
                      nil, nil, nil, /*load=*/true});
        }
    }

    // --- Pass 5: composite ---
    RootPostU pu = {};
    pu.res = (simd_float2){(float)w_, (float)h_};
    pu.srcTexel = (simd_float2){1.0f / (float)sw_, 1.0f / (float)sh_};
    pu.ssaa = builtSsaa_;
    pu.tonemap = post.tonemap ? 1 : 0;
    pu.bloomOn = bloomOn ? 1 : 0;
    pu.dofOn = post.dof ? 1 : 0;
    pu.ditherOn = post.dither ? 1 : 0;
    pu.exposure = post.exposure;
    pu.bloomIntensity = post.bloomIntensity;
    // A focus distance of 0 means "wherever the camera is looking", which for an
    // orbit camera is its own radius. Hard-coding a distance instead would have
    // the subject drift out of focus every time the framing changed.
    pu.dofFocus = (post.dofFocus > 0.0f) ? post.dofFocus : radius;
    pu.dofRange = post.dofRange;
    pu.dofStrength = post.dofStrength;
    pu.vignette = post.vignette;
    pu.grain = post.grain;
    pu.time = postTime;
    pu.nearZ = nearZ; pu.farZ = farZ;
    pu.caStrength = post.caStrength;
    pu.streak = post.streak;
    pu.streakLength = post.streakLength;
    pu.streakTint = (simd_float4){post.streakTint[0], post.streakTint[1], post.streakTint[2], 0};
    pu.halation = post.halation;
    pu.halationTint = (simd_float4){post.halationTint[0], post.halationTint[1], post.halationTint[2], 0};
    pu.contrast = post.contrast;
    pu.saturation = post.saturation;
    pu.lift   = (simd_float4){post.lift[0], post.lift[1], post.lift[2], 0};
    pu.gammaC = (simd_float4){post.gammaC[0], post.gammaC[1], post.gammaC[2], 0};
    pu.gainC  = (simd_float4){post.gain[0], post.gain[1], post.gain[2], 0};
    pu.shadowTint    = (simd_float4){post.shadowTint[0], post.shadowTint[1], post.shadowTint[2], 0};
    pu.highlightTint = (simd_float4){post.highlightTint[0], post.highlightTint[1], post.highlightTint[2], 0};
    pu.toneBalance = post.toneBalance;
    pu.grainSize = post.grainSize;
    pu.grainChroma = post.grainChroma;
    pu.splitStrength = post.splitStrength;
    pu.distortK1 = post.distortK1;
    pu.distortK2 = post.distortK2;
    pu.distortZoom = post.distortZoom;
    // Halation and the streak want a *wider* spread than the bloom composite
    // does, so they read a deeper level of the same chain rather than paying for
    // a second blur. Clamped to what the chain actually has: at small render
    // sizes buildTargets stops before the requested depth.
    id<MTLTexture> haloTex = fogColorTex_;
    if (bloomOn) {
        const int hm = std::max(0, std::min(post.halationMip, (int)bloomMips_.size() - 1));
        haloTex = bloomMips_[hm];
    }
    encodeFS({cb, postPipe_, postTex_, &pu, sizeof(pu),
              fogColorTex_, bloomOn ? bloomMips_[0] : fogColorTex_, rootDepthTex_,
              haloTex});

    outTex_ = postTex_;
    return postTex_;
}
