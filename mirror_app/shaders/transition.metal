// The pond -> face hydro-dip transition.
//
// Ported from neuromirror/cloth_cpp/src/shaders.metal, with one structural
// change: the face relief is **rendered from the real fitted mesh every frame**
// rather than sampled from a baked heightmap texture.
//
// cloth_cpp baked the neutral ICT mask into face.bin (height + coverage on a
// grid) and displaced a grid by it. That shell is what produced its two open
// artifacts -- a seam line around the silhouette and a boxy neck edge -- and it
// also locked the effect to one neutral face, front-on. The combined app has the
// actual fitted mesh (Maxine/NVF topology, this person's identity, this frame's
// expression and head pose), so `f_relief` rasterises that geometry into the
// same (normal.xy, height, coverage) G-buffer the emerge pass already wanted.
// Same consumer, live and watertight producer.
//
// Four phases on one timeline, one locked front-on camera:
//   1 hold     flat pond fills the frame
//   2 emerge   the face rises, refracting and embossing the pond
//   3 swap     the flat pond becomes a 3D cloth over the same pixels
//   4 fall     the cloth falls away behind the face
#include <metal_stdlib>
using namespace metal;

struct Vertex {                 // matches the C++ Vertex (simd_float3 x2 + float2)
    float3 pos;
    float3 nrm;
    float2 uv;
};

struct Uniforms {
    float4x4 mvp;
    float4x4 model;
    float4 lightDir;            // xyz = light dir, w = mode (0/1 textured, 2 solid)
    float4 baseColor;
};

struct VOut {
    float4 clip [[position]];
    float3 wnrm;
    float2 uv;
};

// ---- the face relief G-buffer ----------------------------------------------
// Rasterise the fitted mesh to (normal.xy encoded, height, coverage) -- the
// exact layout cloth_cpp's baked face.bin had, so the emerge pass is unchanged.

struct ReliefU {
    float4x4 mvp;
    float4 p;      // x = 1/depthRange for height normalisation, y = zMin
};

struct ROut {
    float4 clip [[position]];
    float3 nrm;
    float  height;
};

vertex ROut v_relief(uint vid [[vertex_id]],
                     device const Vertex* verts [[buffer(0)]],
                     constant ReliefU& u [[buffer(1)]]) {
    ROut o;
    float3 p = verts[vid].pos;
    o.clip = u.mvp * float4(p, 1.0);
    o.nrm = verts[vid].nrm;
    o.height = saturate((p.z - u.p.y) * u.p.x);
    return o;
}

fragment float4 f_relief(ROut in [[stage_in]]) {
    float3 N = normalize(in.nrm);
    if (N.z < 0.0) N = -N;                    // front-on view: face the camera
    // RG = encoded normal.xy, B = height, A = coverage. Coverage is 1 wherever
    // the mesh actually drew, which is what makes this watertight: there is no
    // interpolated alpha ramp at the silhouette to produce a seam.
    return float4(N.xy * 0.5 + 0.5, in.height, 1.0);
}

// ---- phases 1-2: fullscreen pond, embossed/refracted by the emerging face ---

struct EmergeU {
    // x = emergence 0..1, y = refract scale, z = swap crossfade 0..1, w = unused
    float4 p;
    float4 light;    // xyz = light dir
};

struct FSOut { float4 clip [[position]]; float2 uv; };

vertex FSOut v_fs(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);      // fullscreen triangle
    FSOut o; o.clip = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(p.x, 1.0 - p.y);
    return o;
}

fragment float4 f_emerge(FSOut in [[stage_in]],
                         constant EmergeU& u [[buffer(0)]],
                         texture2d<float> pond [[texture(0)]],
                         texture2d<float> relief [[texture(1)]]) {
    constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);
    float e = u.p.x;

    // The relief is rendered by the same camera as everything else, so it is
    // already in screen space -- no face-region remap, which is what cloth_cpp
    // needed (u.p.zw) because its relief was an unprojected baked square.
    float4 fs = relief.sample(smp, in.uv);
    float2 nxy = fs.rg * 2.0 - 1.0;
    float nz = sqrt(max(0.0, 1.0 - dot(nxy, nxy)));
    float3 N = float3(nxy, nz);
    float a = fs.a * e;

    // Refract the pond while emerging, settling to 0 at full emergence so the
    // final face is undistorted -- identical to the 3D face at the swap.
    float refr = u.p.y * fs.a * (e * (1.0 - e) * 4.0);
    float2 ruv = in.uv + nxy * refr;
    float3 base = pond.sample(smp, ruv).rgb;

    // Shade EXACTLY like f_main (0.30 ambient + 0.85 diffuse, no spec) blended
    // from the flat pond, so film brightness and the emerged face both match the
    // 3D cloth and face after the swap.
    float3 L = normalize(u.light.xyz);
    float ndl = max(0.0, dot(N, L));
    float flatShade = 0.30 + 0.85 * max(0.0, L.z);
    float shade = mix(flatShade, 0.30 + 0.85 * ndl, a);
    return float4(base * shade, u.p.z);
}

// ---- phases 3-4: 3D cloth + face --------------------------------------------

vertex VOut v_main(uint vid [[vertex_id]],
                   device const Vertex* verts [[buffer(0)]],
                   constant Uniforms& u [[buffer(1)]]) {
    VOut o;
    float3 p = verts[vid].pos;
    o.clip = u.mvp * float4(p, 1.0);
    o.wnrm = (u.model * float4(verts[vid].nrm, 0.0)).xyz;
    o.uv = verts[vid].uv;
    return o;
}

fragment float4 f_main(VOut in [[stage_in]],
                       constant Uniforms& u [[buffer(1)]],
                       texture2d<float> tex [[texture(0)]]) {
    // Two-sided, viewer-facing normal (camera is fixed front-on). A flat sheet
    // then reads N=(0,0,1) and shades like the emerge pass's flat pond, so the
    // swap has no brightness jump. Winding otherwise leaves the flat cloth
    // normal at -z (ndl=0), which crushes it to ambient -- a ~3x drop.
    float3 N = normalize(in.wnrm);
    if (N.z < 0.0) N = -N;
    float3 L = normalize(u.lightDir.xyz);
    float ndl = max(0.0, dot(N, L));
    float shade = 0.30 + 0.85 * ndl;

    float mode = u.lightDir.w;
    float3 base;
    if (mode < 1.5) {
        constexpr sampler smp(coord::normalized, address::clamp_to_edge, filter::linear);
        base = tex.sample(smp, in.uv).rgb;           // the neural pond film / skin
    } else {
        base = u.baseColor.rgb;
    }
    return float4(base * shade, 1.0);
}
