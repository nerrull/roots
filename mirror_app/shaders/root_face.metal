// root_face.metal — Metal port of sdf_viewer's face.vert + face.frag.
//
// The mask-mesh mid-geometry pass: triangle meshes (the face masks) rasterized
// into the SAME colour + depth targets as the root capsules, between the capsule
// pass and the fog pass, so they depth-composite against the sphere-traced roots
// and are included in the fog. Marble veining + a per-face point light. Vertices
// arrive interleaved (pos3, normal3, color3, lightPos3) exactly as the GL VBO.
//
// Depth: the capsule pass writes custom depth = (clip.z/clip.w)*0.5+0.5 (GL
// convention). These triangles use the hardware rasterizer's depth, so the vertex
// shader remaps GL clip-z [-1,1] into Metal's [0,1] (z = (z+w)/2) — after the
// perspective divide that yields the identical value, so the two passes composite.
#include <metal_stdlib>
using namespace metal;

struct FaceVertex {
    packed_float3 pos;
    packed_float3 normal;
    packed_float3 color;
    packed_float3 lightPos;
};

struct FaceVOut {
    float4 pos [[position]];
    float3 worldPos;
    float3 normal;
    float3 color;
    float3 lightPos;
};

vertex FaceVOut root_face_vs(uint vid [[vertex_id]],
                             device const FaceVertex* verts [[buffer(0)]],
                             constant RootFaceU&      U     [[buffer(1)]]) {
    FaceVertex v = verts[vid];
    FaceVOut o;
    o.worldPos = float3(v.pos);
    o.normal   = float3(v.normal);
    o.color    = float3(v.color);
    o.lightPos = float3(v.lightPos);
    float4 c = U.viewProj * float4(float3(v.pos), 1.0);
    c.z = (c.z + c.w) * 0.5;   // GL [-1,1] clip-z -> Metal [0,1], matches capsule depth
    o.pos = c;
    return o;
}

// cheap 3-octave value noise for marble turbulence (verbatim from face.frag)
static float _hash(float3 p) {
    p = fract(p * float3(443.897, 397.297, 491.187));
    p += dot(p, p.zyx + 19.19);
    return fract((p.x + p.y) * p.z);
}
static float _noise(float3 p) {
    float3 i = floor(p), f = fract(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(_hash(i),                   _hash(i+float3(1,0,0)), u.x),
            mix(_hash(i+float3(0,1,0)),     _hash(i+float3(1,1,0)), u.x), u.y),
        mix(mix(_hash(i+float3(0,0,1)),     _hash(i+float3(1,0,1)), u.x),
            mix(_hash(i+float3(0,1,1)),     _hash(i+float3(1,1,1)), u.x), u.y), u.z);
}
static float _turbulence(float3 p) {
    return abs(_noise(p) - 0.5) * 0.5000
         + abs(_noise(p * 2.03) - 0.5) * 0.2500
         + abs(_noise(p * 4.07) - 0.5) * 0.1250;
}

fragment float4 root_face_fs(FaceVOut in [[stage_in]],
                             constant RootFaceU& U [[buffer(1)]]) {
    float3 n = normalize(in.normal);
    float3 v = normalize(U.eye.xyz - in.worldPos);

    float t = _turbulence(in.worldPos * U.veinScale);
    float vein = sin((in.worldPos.x + in.worldPos.y * 0.4 + t * 12.0) * U.veinScale * 3.0);
    vein = smoothstep(0.15, 0.85, vein * 0.5 + 0.5);
    float3 baseColor = mix(in.color, U.veinColor.xyz, vein * U.veinStrength);

    float3 l = normalize(U.lightDir.xyz);
    float lam = abs(dot(n, l));
    float3 col = baseColor * (0.04 + 0.06 * lam);

    float3 toLight = in.lightPos - in.worldPos;
    float dist = length(toLight);
    float3 ldir = toLight / max(dist, 0.001);
    float atten = 1.0 / (1.0 + U.lightFalloff * dist * dist);
    float ndotl = max(dot(n, ldir), 0.0);
    float3 h = normalize(ldir + v);
    float spec = pow(max(dot(n, h), 0.0), 60.0) * U.specStrength;
    float fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.25;
    col += baseColor * ndotl * atten * U.lightIntensity
         + float3(spec) * atten
         + float3(fresnel) * atten;

    return float4(col, 1.0);
}
