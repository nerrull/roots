#include "metal_root_renderer.h"
#include "metal_context.h"

#import <Foundation/Foundation.h>
#include <simd/simd.h>

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
}

// ===========================================================================

MetalRootRenderer::MetalRootRenderer(const MetalContext& ctx, const std::string& shaderDir,
                                     const std::string& sharedHeaderPath, int w, int h)
    : device_(ctx.device()), w_(w), h_(h) {
    // Pipelines: each MSL pass compiled with root_shared.h prepended.
    id<MTLLibrary> geomLib = ctx.newLibraryFromFiles({sharedHeaderPath, shaderDir + "/root_geom.metal"});
    id<MTLLibrary> fogLib  = ctx.newLibraryFromFiles({sharedHeaderPath, shaderDir + "/root_fog.metal"});
    if (!geomLib || !fogLib) { fprintf(stderr, "MetalRootRenderer: shader compile failed\n"); return; }

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
        d.vertexFunction   = [fogLib newFunctionWithName:@"root_fog_vs"];
        d.fragmentFunction = [fogLib newFunctionWithName:@"root_fog_fs"];
        d.colorAttachments[0].pixelFormat = kColorFmt;
        fogPipe_ = [device_ newRenderPipelineStateWithDescriptor:d error:&err];
        if (!fogPipe_) { NSLog(@"fog pipeline failed: %@", err); return; }
    }
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
    auto make2D = [&](MTLPixelFormat fmt, MTLTextureUsage usage, MTLStorageMode store) -> id<MTLTexture> {
        MTLTextureDescriptor* td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                               width:std::max(1, w_)
                                                              height:std::max(1, h_)
                                                           mipmapped:NO];
        td.usage = usage;
        td.storageMode = store;
        return [device_ newTextureWithDescriptor:td];
    };
    const MTLTextureUsage rt = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    rootColorTex_ = make2D(kColorFmt, rt, MTLStorageModePrivate);
    rootDepthTex_ = make2D(kDepthFmt, rt, MTLStorageModePrivate);
    // Shared so the final image can be read back headlessly (--roottest) and
    // presented without a copy; it's the last texture in the chain.
    fogColorTex_  = make2D(kColorFmt, rt, MTLStorageModeShared);
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
    std::vector<float> distData(std::max(1, nNodes), 0.f);
    if (nNodes > 0) {
        std::vector<char> isChild(nNodes, 0);
        for (int s = 0; s < nSeg; s++) {
            int cy = segs[2*s + 1];
            if (cy >= 0 && cy < nNodes) isChild[cy] = 1;
        }
        if (pulse.hopOffset != 0.0f) {
            float hopBase = 0.f;
            for (int i = 0; i < nNodes; i++)
                if (!isChild[i]) { distData[i] = hopBase; hopBase += pulse.hopOffset; }
        }
        for (int s = 0; s < nSeg; s++) {
            int pi = segs[2*s], ci = segs[2*s + 1];
            if (pi < 0 || ci < 0 || pi >= nNodes || ci >= nNodes) continue;
            float dx = nodesXYZ[3*ci]   - nodesXYZ[3*pi];
            float dy = nodesXYZ[3*ci+1] - nodesXYZ[3*pi+1];
            float dz = nodesXYZ[3*ci+2] - nodesXYZ[3*pi+2];
            distData[ci] = distData[pi] + std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    }

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

id<MTLTexture> MetalRootRenderer::render(id<MTLCommandBuffer> cb,
                                         float azimuth, float elevation, float radius,
                                         const float target3[3], float fov,
                                         const float lightDir3[3]) {
    if (!valid()) return nil;

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
    float asp = (float)w_ / (float)h_;
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
    gu.res = (simd_float2){(float)w_, (float)h_};
    gu.fov = fov;
    gu.radiusScale = radiusScale; gu.radiusMin = radiusMin; gu.radiusMax = radiusMax;
    gu.ambient = mat.ambient; gu.diffuse = mat.diffuse; gu.shininess = mat.shininess;
    gu.colorNoiseScale = mat.colorNoiseScale; gu.colorNoiseStrength = mat.colorNoiseStrength;
    gu.metallic = pbr.metallic; gu.roughness = pbr.roughness;
    gu.pulseSpeed = pulse.speed; gu.pulseSpacing = pulse.spacing; gu.pulseWidth = pulse.width;
    gu.pulseIntensity = pulse.intensity; gu.pulseTime = pulse.time;
    gu.shaderMode = (int)shaderMode;
    gu.wispCount = active;
    gu.pulseEnabled = pulse.enabled ? 1 : 0;
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
        MTLClearColorMake(invert ? 0.0 : 0.12, invert ? 0.0 : 0.08, invert ? 0.0 : 0.05, 1.0);
    gp.colorAttachments[0].storeAction = MTLStoreActionStore;
    gp.depthAttachment.texture = rootDepthTex_;
    gp.depthAttachment.loadAction = MTLLoadActionClear;
    gp.depthAttachment.clearDepth = 1.0;
    gp.depthAttachment.storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> ge = [cb renderCommandEncoderWithDescriptor:gp];
    if (segCount_ > 0) {
        [ge setRenderPipelineState:geomPipe_];
        [ge setDepthStencilState:depthState_];
        [ge setCullMode:MTLCullModeNone];
        // vertex stage
        [ge setVertexBuffer:nodeBuf_ offset:0 atIndex:0];
        [ge setVertexBuffer:segBuf_  offset:0 atIndex:1];
        [ge setVertexBuffer:radBuf_  offset:0 atIndex:2];
        [ge setVertexBuffer:primBuf_ offset:0 atIndex:5];
        [ge setVertexBuffer:frameBuf_ offset:0 atIndex:6];
        [ge setVertexBytes:&gu length:sizeof(gu) atIndex:8];
        // fragment stage
        [ge setFragmentBuffer:nodeBuf_ offset:0 atIndex:0];
        [ge setFragmentBuffer:segBuf_  offset:0 atIndex:1];
        [ge setFragmentBuffer:radBuf_  offset:0 atIndex:2];
        [ge setFragmentBuffer:distBuf_ offset:0 atIndex:3];
        [ge setFragmentBuffer:grpBuf_  offset:0 atIndex:4];
        [ge setFragmentBuffer:primBuf_ offset:0 atIndex:5];
        [ge setFragmentBuffer:frameBuf_ offset:0 atIndex:6];
        [ge setFragmentBuffer:auxBuf_  offset:0 atIndex:7];
        [ge setFragmentBytes:&gu length:sizeof(gu) atIndex:8];
        [ge setFragmentBytes:wispBuf.data() length:wispBuf.size() * sizeof(RootWisp) atIndex:9];
        [ge setFragmentTexture:noiseTex_ atIndex:0];
        [ge drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6
             instanceCount:(NSUInteger)segCount_];
    }
    [ge endEncoding];

    // --- Pass 2: fog ---
    RootFogU fu = {};
    fu.cam = cam;
    fu.eye = gu.eye;
    fu.fogColor = (simd_float4){fog.color[0], fog.color[1], fog.color[2], 0};
    fu.res = gu.res;
    fu.fov = fov; fu.nearZ = nearZ; fu.farZ = farZ;
    fu.fogDensity = fog.density; fu.fogFalloff = fog.falloff;
    fu.fogNoiseScale = fog.noiseScale; fu.fogNoiseStrength = fog.noiseStrength;
    fu.fogTime = fog.driftTime; fu.fogRefDist = fog.refDist;
    fu.wispGlowStrength = wispGlowStrength;
    fu.axisLength = overlay.axisLength; fu.gridSpacing = overlay.gridSpacing;
    fu.showAxes = overlay.showAxes ? 1 : 0; fu.showGrid = overlay.showGrid ? 1 : 0;
    fu.wispCount = active;

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
    [fe drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [fe endEncoding];

    return fogColorTex_;
}
