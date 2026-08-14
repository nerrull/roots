#include "transition_scene.h"
#include "metal_context.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

// Front-on camera. The sheet is sized so a *flat* sheet at z=0 exactly fills the
// frustum cross-section, which is the precondition for the swap: the flat cloth
// and the fullscreen pond quad then cover identical pixels.
constexpr float CAM_D = 3.0f;
constexpr float CAM_FOV = 45.0f * float(M_PI) / 180.0f;
constexpr simd_float4 LIGHT = {0.35f, 0.55f, 0.75f, 0.5f};

struct Vertex { simd_float3 pos; simd_float3 nrm; simd_float2 uv; };
struct Uniforms { simd_float4x4 mvp; simd_float4x4 model; simd_float4 lightDir; simd_float4 baseColor; };
struct EmergeU { simd_float4 p; simd_float4 light; };
struct ReliefU { simd_float4x4 mvp; simd_float4 p; };

simd_float4x4 perspective(float fovy, float aspect, float zn, float zf) {
    float f = 1.0f / std::tan(fovy * 0.5f);
    return simd_matrix(simd_make_float4(f / aspect, 0, 0, 0), simd_make_float4(0, f, 0, 0),
                       simd_make_float4(0, 0, zf / (zn - zf), -1),
                       simd_make_float4(0, 0, (zn * zf) / (zn - zf), 0));
}
simd_float4x4 lookAt(simd_float3 eye, simd_float3 c, simd_float3 up) {
    simd_float3 z = simd_normalize(eye - c), x = simd_normalize(simd_cross(up, z)), y = simd_cross(z, x);
    return simd_matrix(simd_make_float4(x.x, y.x, z.x, 0), simd_make_float4(x.y, y.y, z.y, 0),
                       simd_make_float4(x.z, y.z, z.z, 0),
                       simd_make_float4(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1));
}
simd_float4x4 frontVP(float aspect) {
    return simd_mul(perspective(CAM_FOV, aspect, 0.05f, 50.0f),
                    lookAt(simd_make_float3(0, 0, CAM_D), simd_make_float3(0, 0, 0),
                           simd_make_float3(0, 1, 0)));
}

}  // namespace

struct TransitionScene::Impl {
    const MetalContext& ctx;
    int w = 0, h = 0;

    id<MTLRenderPipelineState> psoMain = nil, psoFS = nil, psoRelief = nil;
    id<MTLDepthStencilState> dss = nil, dssFS = nil;
    id<MTLTexture> colorTex = nil, depthTex = nil, reliefTex = nil, reliefDepth = nil;
    id<MTLTexture> pondTex = nil;
    id<MTLBuffer> clothVB = nil, clothIB = nil, faceVB = nil, faceIB = nil;
    size_t clothIdx = 0, faceIdx = 0;

    Cloth cloth;
    std::vector<Vertex> clothVerts;
    std::vector<float> faceModelVerts;     // normalised, placed
    std::vector<int>   faceTris;
    bool haveFace = false;
    // Fixed normalisation captured from the first mesh: an expression changes
    // the mesh extent, and re-deriving it per frame would pump the face's size.
    bool normSet = false;
    float centre[3] = {0, 0, 0}, scale = 1.0f;
    float faceZMin = 0.f, faceZRange = 1.f;

    double t = 0.0;
    float prevE = 0.f, dE = 0.f;
    float sheetHalf = CAM_D * std::tan(CAM_FOV * 0.5f);

    explicit Impl(const MetalContext& c) : ctx(c) {}

    bool buildPipelines(const std::string& shaderDir);
    void makeTargets(int W, int H);
    void rebuildCloth();
    void uploadFace();
    void packCloth();
};

bool TransitionScene::Impl::buildPipelines(const std::string& shaderDir) {
    id<MTLLibrary> lib = ctx.newLibraryFromFile(shaderDir + "/transition.metal");
    if (!lib) return false;

    auto make = [&](const char* vs, const char* fs, MTLPixelFormat fmt,
                    bool depth, bool blend) -> id<MTLRenderPipelineState> {
        MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [lib newFunctionWithName:[NSString stringWithUTF8String:vs]];
        d.fragmentFunction = [lib newFunctionWithName:[NSString stringWithUTF8String:fs]];
        if (!d.vertexFunction || !d.fragmentFunction) {
            std::fprintf(stderr, "transition: missing %s/%s\n", vs, fs);
            return nil;
        }
        d.colorAttachments[0].pixelFormat = fmt;
        if (blend) {
            d.colorAttachments[0].blendingEnabled = YES;
            d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;
        }
        if (depth) d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        NSError* err = nil;
        id<MTLRenderPipelineState> p = [ctx.device() newRenderPipelineStateWithDescriptor:d
                                                                                    error:&err];
        if (!p) std::fprintf(stderr, "transition: pipeline %s/%s: %s\n", vs, fs,
                             err.localizedDescription.UTF8String);
        return p;
    };

    // RGBA16Float + Shared matches the rest of the app: the compositor samples
    // it and the headless shot path reads it back with getBytes, which a Private
    // texture cannot serve.
    psoMain   = make("v_main",   "f_main",   MTLPixelFormatRGBA16Float, true,  false);
    psoFS     = make("v_fs",     "f_emerge", MTLPixelFormatRGBA16Float, false, true);
    psoRelief = make("v_relief", "f_relief", MTLPixelFormatRGBA8Unorm, true,  false);

    MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = MTLCompareFunctionLess;
    dd.depthWriteEnabled = YES;
    dss = [ctx.device() newDepthStencilStateWithDescriptor:dd];

    MTLDepthStencilDescriptor* df = [MTLDepthStencilDescriptor new];
    df.depthCompareFunction = MTLCompareFunctionAlways;
    df.depthWriteEnabled = NO;
    dssFS = [ctx.device() newDepthStencilStateWithDescriptor:df];

    return psoMain && psoFS && psoRelief;
}

void TransitionScene::Impl::makeTargets(int W, int H) {
    if (W == w && H == h && colorTex) return;
    w = W; h = H;
    auto tex = [&](MTLPixelFormat fmt, MTLTextureUsage usage, bool shared = false) {
        MTLTextureDescriptor* td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                               width:W height:H mipmapped:NO];
        td.usage = usage;
        td.storageMode = shared ? MTLStorageModeShared : MTLStorageModePrivate;
        return [ctx.device() newTextureWithDescriptor:td];
    };
    const MTLTextureUsage rt = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    colorTex    = tex(MTLPixelFormatRGBA16Float, rt, /*shared=*/true);
    depthTex    = tex(MTLPixelFormatDepth32Float, MTLTextureUsageRenderTarget);
    reliefTex   = tex(MTLPixelFormatRGBA8Unorm,  rt);
    reliefDepth = tex(MTLPixelFormatDepth32Float, MTLTextureUsageRenderTarget);
}

void TransitionScene::Impl::rebuildCloth() {
    // Solid sheet, no hole: the opaque face occludes the centre, so carving one
    // only creates a rim to misalign. The interior ring is pinned to the mask's
    // back rim.
    const float hrx = 0.42f, hry = 0.55f;
    cloth.build(64, 64, 2 * sheetHalf, 2 * sheetHalf, hrx, hry, /*carve=*/false);
    clothVerts.assign(cloth.pos.size(), Vertex{});

    // Re-pin onto the fitted face's own silhouette. cloth_cpp could use a fixed
    // ellipse because its face was a baked asset; a fitted face is a different
    // shape per person, so the ring is projected onto the actual mesh extent.
    if (haveFace && !faceModelVerts.empty()) {
        std::vector<int> pins = cloth.pinIndices();
        std::vector<simd_float3> targets;
        targets.reserve(pins.size());
        // Mesh silhouette radius per angle, from the placed vertices.
        const int NB = 64;
        std::vector<float> rad(NB, 0.f);
        for (size_t i = 0; i + 2 < faceModelVerts.size(); i += 3) {
            const float x = faceModelVerts[i], y = faceModelVerts[i + 1];
            float a = std::atan2(y, x);
            if (a < 0) a += 2.f * float(M_PI);
            const int b = std::min(NB - 1, int(a / (2.f * float(M_PI)) * NB));
            rad[b] = std::max(rad[b], std::sqrt(x * x + y * y));
        }
        for (int b = 0; b < NB; ++b)      // fill any empty angular bin
            if (rad[b] <= 0.f) rad[b] = (rad[(b + NB - 1) % NB] + rad[(b + 1) % NB]) * 0.5f;

        for (int k : pins) {
            const simd_float3 p = cloth.pin[size_t(k)];
            float a = std::atan2(p.y, p.x);
            if (a < 0) a += 2.f * float(M_PI);
            const int b = std::min(NB - 1, int(a / (2.f * float(M_PI)) * NB));
            const float r = rad[b] > 0.f ? rad[b] : 0.45f;
            targets.push_back(simd_make_float3(std::cos(a) * r, std::sin(a) * r, -0.02f));
        }
        cloth.pinTo(targets);
    }
    // Normals before the first render, not just inside the sim step. build()
    // leaves them zeroed, and a zero normal normalises to garbage -- the flat
    // sheet then shades nothing like the pond it is supposed to be
    // indistinguishable from, and the crossfade blends two different images.
    // This is the same brightness-match trap the shader notes describe, entered
    // from the other side.
    cloth.computeNormals();
    packCloth();
}

void TransitionScene::Impl::packCloth() {
    // uv = screen-planar of the REST (flat) grid, y-flipped to the pond's screen
    // orientation: at the flat swap frame this equals the fullscreen pond, and it
    // stays locked to the surface as the sheet falls.
    for (int j = 0; j < cloth.ny; ++j)
        for (int i = 0; i < cloth.nx; ++i) {
            const int k = cloth.idx(i, j);
            clothVerts[size_t(k)].pos = cloth.pos[size_t(k)];
            clothVerts[size_t(k)].nrm = cloth.nrm[size_t(k)];
            clothVerts[size_t(k)].uv =
                simd_make_float2(i / float(cloth.nx - 1), 1.0f - j / float(cloth.ny - 1));
        }
    const size_t bytes = clothVerts.size() * sizeof(Vertex);
    if (!clothVB || clothVB.length < bytes)
        clothVB = [ctx.device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    std::memcpy(clothVB.contents, clothVerts.data(), bytes);

    if (!clothIB || clothIdx != cloth.tris.size()) {
        clothIdx = cloth.tris.size();
        if (clothIdx)
            clothIB = [ctx.device() newBufferWithBytes:cloth.tris.data()
                                                length:clothIdx * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
    }
}

void TransitionScene::Impl::uploadFace() {
    if (faceModelVerts.empty() || faceTris.empty()) { faceIdx = 0; return; }
    const size_t n = faceModelVerts.size() / 3;
    std::vector<Vertex> v(n);
    for (size_t i = 0; i < n; ++i) {
        const float x = faceModelVerts[i * 3], y = faceModelVerts[i * 3 + 1],
                    z = faceModelVerts[i * 3 + 2];
        v[i].pos = simd_make_float3(x, y, z);
        // Screen-planar uv, matching the emerge pass and the cloth, so the pond
        // pattern is continuous across the swap.
        v[i].uv = simd_make_float2(0.5f + x / (2 * sheetHalf), 0.5f - y / (2 * sheetHalf));
        v[i].nrm = simd_make_float3(0, 0, 1);
    }
    std::vector<simd_float3> nn(n, simd_make_float3(0, 0, 0));
    for (size_t t = 0; t + 2 < faceTris.size(); t += 3) {
        const int a = faceTris[t], b = faceTris[t + 1], c = faceTris[t + 2];
        const simd_float3 fn = simd_cross(v[b].pos - v[a].pos, v[c].pos - v[a].pos);
        nn[a] += fn; nn[b] += fn; nn[c] += fn;
    }
    for (size_t i = 0; i < n; ++i) {
        const float l = simd_length(nn[i]);
        v[i].nrm = l > 1e-8f ? nn[i] / l : simd_make_float3(0, 0, 1);
    }

    faceVB = [ctx.device() newBufferWithBytes:v.data() length:n * sizeof(Vertex)
                                      options:MTLResourceStorageModeShared];
    std::vector<uint32_t> idx(faceTris.begin(), faceTris.end());
    faceIdx = idx.size();
    faceIB = [ctx.device() newBufferWithBytes:idx.data() length:faceIdx * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];

    faceZMin = v[0].pos.z;
    float zmax = v[0].pos.z;
    for (size_t i = 0; i < n; ++i) {
        faceZMin = std::min(faceZMin, v[i].pos.z);
        zmax = std::max(zmax, v[i].pos.z);
    }
    faceZRange = std::max(1e-4f, zmax - faceZMin);
}

// ---------------------------------------------------------------------------

TransitionScene::TransitionScene(const MetalContext& ctx, int w, int h)
    : impl_(new Impl(ctx)) {
    if (!impl_->buildPipelines(std::string(MIRROR_APP_SHADER_DIR))) return;
    impl_->makeTargets(std::max(2, w), std::max(2, h));
    impl_->rebuildCloth();
}

TransitionScene::~TransitionScene() = default;

bool TransitionScene::valid() const { return impl_ && impl_->psoMain && impl_->psoFS; }
int  TransitionScene::width() const { return impl_->w; }
int  TransitionScene::height() const { return impl_->h; }
void TransitionScene::ensureSize(int w, int h) { impl_->makeTargets(std::max(2, w), std::max(2, h)); }
const Cloth& TransitionScene::cloth() const { return impl_->cloth; }
bool TransitionScene::hasFace() const { return impl_->haveFace; }
double TransitionScene::clock() const { return impl_->t; }
void TransitionScene::setPondTexture(id<MTLTexture> pond) { impl_->pondTex = pond; }

void TransitionScene::setFaceMesh(const std::vector<float>& verts, const std::vector<int>& tris) {
    if (verts.size() < 9) return;
    if (!tris.empty()) impl_->faceTris = tris;
    if (impl_->faceTris.empty()) return;

    if (!impl_->normSet) {
        const size_t n = verts.size() / 3;
        double cx = 0, cy = 0, cz = 0;
        for (size_t i = 0; i < n; ++i) {
            cx += verts[i * 3]; cy += verts[i * 3 + 1]; cz += verts[i * 3 + 2];
        }
        impl_->centre[0] = float(cx / n);
        impl_->centre[1] = float(cy / n);
        impl_->centre[2] = float(cz / n);
        float m = 1e-9f;
        for (size_t i = 0; i < n; ++i)
            m = std::max({m, std::fabs(verts[i * 3] - impl_->centre[0]),
                             std::fabs(verts[i * 3 + 1] - impl_->centre[1])});
        // 0.55 puts the fitted face at roughly the extent cloth_cpp's baked one
        // occupied (FACE_HY = 0.55), so the tuned look carries over.
        impl_->scale = 0.55f / m;
        impl_->normSet = true;
    }

    const size_t n = verts.size() / 3;
    impl_->faceModelVerts.resize(n * 3);
    const float s = impl_->scale * faceScale;
    for (size_t i = 0; i < n; ++i) {
        impl_->faceModelVerts[i * 3]     = (verts[i * 3]     - impl_->centre[0]) * s;
        impl_->faceModelVerts[i * 3 + 1] = (verts[i * 3 + 1] - impl_->centre[1]) * s;
        impl_->faceModelVerts[i * 3 + 2] = (verts[i * 3 + 2] - impl_->centre[2]) * s;
    }
    const bool first = !impl_->haveFace;
    impl_->haveFace = true;
    impl_->uploadFace();
    if (first) impl_->rebuildCloth();   // re-pin onto the real silhouette
}

void TransitionScene::restart() {
    impl_->t = 0.0;
    impl_->prevE = 0.f;
    impl_->dE = 0.f;
    impl_->rebuildCloth();
}

float TransitionScene::emergence() const {
    const float t = float(impl_->t);
    if (t < timing.hold) return 0.f;
    return std::min(1.0f, (t - timing.hold) / std::max(1e-3f, timing.emerge));
}

bool TransitionScene::done() const {
    return float(impl_->t) > timing.hold + timing.emerge + timing.settle + 4.0f;
}

const char* TransitionScene::phaseName() const {
    const float t = float(impl_->t);
    if (t < timing.hold) return "hold";
    if (t < timing.hold + timing.emerge) return "emerge";
    if (t < timing.hold + timing.emerge + timing.settle) return "settle";
    return "fall";
}

void TransitionScene::advance(double dt) {
    const float e0 = emergence();
    impl_->t += dt;
    const float e1 = emergence();
    // Emergence velocity, for the velocity-driven refraction. Normalised by dt
    // so the look does not change with framerate.
    impl_->dE = dt > 1e-6 ? float((e1 - e0) / dt) : 0.f;
    impl_->prevE = e1;

    // Gravity starts only once the crossfade has finished. While it runs, the
    // 3D cloth is flat and static and therefore pixel-identical to the pond it
    // is fading in over -- which is the whole point of crossfading. Letting the
    // sheet start falling on the swap frame (as the prototype did, with a hard
    // cut) means the blend is between two *different* images and the seam is
    // exactly as visible as it was before.
    const float swapT = timing.hold + timing.emerge + timing.settle;
    if (float(impl_->t) >= swapT + timing.fade && showCloth) {
        impl_->cloth.gravity = simd_make_float3(0.f, -gravityDown, -gravityBack);
        impl_->cloth.iterations = iterations;
        const int ss = std::max(1, substeps);
        for (int i = 0; i < ss; ++i) impl_->cloth.step(float(dt) / float(ss));
        impl_->cloth.computeNormals();
        impl_->packCloth();
    }
}

id<MTLTexture> TransitionScene::render(id<MTLCommandBuffer> cb) {
    if (!valid()) return nil;
    Impl& I = *impl_;
    const float aspect = float(I.w) / float(std::max(1, I.h));
    const simd_float4x4 vp = frontVP(aspect);
    const float e = emergence();
    const float swapT = timing.hold + timing.emerge + timing.settle;

    // --- the face relief G-buffer, rendered from the real mesh -------------
    // This is the replacement for cloth_cpp's baked face.bin: same
    // (normal.xy, height, coverage) layout, produced live from the fitted
    // geometry, so it carries expression and head pose and has no shell seam.
    if (I.haveFace && I.faceIdx) {
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = I.reliefTex;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 0.5, 0.0, 0.0);
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.depthAttachment.texture = I.reliefDepth;
        rp.depthAttachment.loadAction = MTLLoadActionClear;
        rp.depthAttachment.clearDepth = 1.0;
        rp.depthAttachment.storeAction = MTLStoreActionDontCare;

        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:I.psoRelief];
        [enc setDepthStencilState:I.dss];
        [enc setCullMode:MTLCullModeNone];
        ReliefU ru;
        ru.mvp = vp;
        ru.p = simd_make_float4(1.0f / I.faceZRange, I.faceZMin, 0, 0);
        [enc setVertexBuffer:I.faceVB offset:0 atIndex:0];
        [enc setVertexBytes:&ru length:sizeof(ru) atIndex:1];
        [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:I.faceIdx
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:I.faceIB
                 indexBufferOffset:0];
        [enc endEncoding];
    }

    // --- the main pass ------------------------------------------------------
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = I.colorTex;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0.02, 0.02, 0.03, 1.0);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.depthAttachment.texture = I.depthTex;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = 1.0;
    rp.depthAttachment.storeAction = MTLStoreActionDontCare;

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];

    // Crossfade across the swap rather than cutting. cloth_cpp left this as a
    // TODO ("a 2-3 frame crossfade would hide any residual mismatch"); with a
    // live pond and a live face the assets change constantly, so it matters more
    // here than it did there.
    float emergeAlpha = 1.f;
    if (float(I.t) > swapT) {
        const float k = (float(I.t) - swapT) / std::max(1e-3f, timing.fade);
        emergeAlpha = std::max(0.f, 1.f - k);
    }

    // 3D pass first when the cloth is live, so the crossfading flat pass lands
    // on top of it.
    const bool cloth3D = float(I.t) >= swapT && showCloth;
    if (cloth3D) {
        [enc setRenderPipelineState:I.psoMain];
        [enc setDepthStencilState:I.dss];
        [enc setCullMode:MTLCullModeNone];
        [enc setTriangleFillMode:wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
        [enc setFragmentTexture:I.pondTex atIndex:0];

        Uniforms u;
        u.mvp = vp;
        u.model = matrix_identity_float4x4;
        u.lightDir = simd_make_float4(LIGHT.x, LIGHT.y, LIGHT.z, 0.0f);
        u.baseColor = simd_make_float4(1, 1, 1, 1);

        if (I.clothIdx) {
            [enc setVertexBuffer:I.clothVB offset:0 atIndex:0];
            [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
            [enc setFragmentBytes:&u length:sizeof(u) atIndex:1];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:I.clothIdx
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:I.clothIB
                     indexBufferOffset:0];
        }
        if (I.haveFace && I.faceIdx) {
            [enc setVertexBuffer:I.faceVB offset:0 atIndex:0];
            [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
            [enc setFragmentBytes:&u length:sizeof(u) atIndex:1];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:I.faceIdx
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:I.faceIB
                     indexBufferOffset:0];
        }
    }

    if (emergeAlpha > 0.001f && I.pondTex) {
        [enc setRenderPipelineState:I.psoFS];
        [enc setDepthStencilState:I.dssFS];
        EmergeU u;
        // Velocity-driven refraction: cloth_cpp's was a single constant, so the
        // distortion peaked at the timeline midpoint regardless of how fast the
        // face was actually moving through the film. Scaling by d(emergence)/dt
        // ties it to the motion, which is what the effect is depicting.
        const float refr = refract * (1.0f + refractVel * std::fabs(I.dE));
        u.p = simd_make_float4(e, refr, emergeAlpha, 0.f);
        u.light = LIGHT;
        [enc setFragmentBytes:&u length:sizeof(u) atIndex:0];
        [enc setFragmentTexture:I.pondTex atIndex:0];
        [enc setFragmentTexture:I.reliefTex atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }

    [enc endEncoding];
    return I.colorTex;
}
