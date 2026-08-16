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

// Cook-Torrance, the same model the roots use. The mask used to run a bare
// pow(NdotH, 60) lobe, which gives one hard highlight dot of a fixed size no
// matter how the surface is angled -- the classic tell of a shader that has a
// specular exponent instead of a roughness.
static constant float PI = 3.14159265359;
static float D_GGX(float NdotH, float a2) {
    const float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
static float G_Smith(float NdotV, float NdotL, float k) {
    const float gv = NdotV / (NdotV * (1.0 - k) + k);
    const float gl = NdotL / (NdotL * (1.0 - k) + k);
    return gv * gl;
}
static float3 F_Schlick(float c, float3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - c, 0.0, 1.0), 5.0);
}
static float3 ggx(float3 N, float3 V, float3 L, float3 albedo,
                  float metallic, float roughness) {
    const float NdotL = max(dot(N, L), 0.0);
    if (NdotL < 1e-4) return float3(0.0);
    const float3 H = normalize(V + L);
    const float NdotH = max(dot(N, H), 0.0);
    const float NdotV = max(dot(N, V), 1e-4);
    const float VdotH = max(dot(V, H), 0.0);
    const float a = roughness * roughness;
    const float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    const float3 F0 = mix(float3(0.04), albedo, metallic);
    const float3 F = F_Schlick(VdotH, F0);
    const float3 spec = (D_GGX(NdotH, a * a) * G_Smith(NdotV, NdotL, k) * F)
                      / max(4.0 * NdotV * NdotL, 1e-3);
    return ((1.0 - F) * (1.0 - metallic) * albedo / PI + spec) * NdotL;
}

static float3 hemiAmbient(float3 n, float3 sky, float3 ground) {
    return mix(ground, sky, n.y * 0.5 + 0.5);
}

fragment float4 root_face_fs(FaceVOut in [[stage_in]],
                             constant RootFaceU& U [[buffer(1)]]) {
    float3 n = normalize(in.normal);
    const float3 v = normalize(U.eye.xyz - in.worldPos);
    const float3 P = in.worldPos;

    const float t = _turbulence(P * U.veinScale);
    float vein = sin((P.x + P.y * 0.4 + t * 12.0) * U.veinScale * 3.0);
    vein = smoothstep(0.15, 0.85, vein * 0.5 + 0.5);
    const float3 baseColor = mix(in.color, U.veinColor.xyz, vein * U.veinStrength);

    // --- surface relief ------------------------------------------------------
    // Perturb the shading normal by the gradient of a turbulence field, so the
    // stone has a fine grain instead of being an ideal smooth solid. The mask
    // mesh is a few thousand triangles and cannot carry this as geometry;
    // sampling a field three more times can. Finite differences rather than an
    // analytic gradient because the field is a sum of three value-noise octaves
    // and its derivative is not worth deriving.
    //
    // Sampled at reliefScale times the vein frequency, and NOT at the vein
    // frequency itself: a mask is around four world units across and veinScale
    // is 0.6, so a gradient taken at the vein's own scale is very nearly
    // constant over the whole head -- it tilts the entire face by a fixed amount
    // and reads as nothing at all. Relief has to run at the scale of the surface
    // it is meant to roughen, which is one to two orders finer than the pattern.
    if (U.reliefStrength > 0.0 && U.reliefScale > 0.0) {
        const float rs = U.veinScale * U.reliefScale;
        const float rt = _turbulence(P * rs);
        const float e = 0.25 / max(rs, 1e-3);
        const float3 g = float3(_turbulence((P + float3(e, 0, 0)) * rs) - rt,
                                _turbulence((P + float3(0, e, 0)) * rs) - rt,
                                _turbulence((P + float3(0, 0, e)) * rs) - rt);
        // Only the component across the surface bends the normal; pushing along
        // it would just scale the normal and change nothing.
        const float3 gt = g - n * dot(g, n);
        n = normalize(n - gt * (U.reliefStrength / max(e, 1e-4)));
    }

    // Veined stone is not uniformly polished: the vein mineral takes a different
    // finish from the matrix, and varying roughness with the same field is what
    // keeps the highlight from sliding across the face as one unbroken sheet.
    const float rough = clamp(U.roughness * mix(1.0, 0.55, vein * U.veinStrength),
                              0.04, 1.0);

    const float3 sky = U.skyColor.xyz, ground = U.groundColor.xyz;
    float3 indirect = baseColor * hemiAmbient(n, sky, ground) * U.hemiStrength;
    if (U.envSpec > 0.0) {
        const float NdotV = max(dot(n, v), 0.0);
        const float3 R = normalize(mix(reflect(-v, n), n, rough * rough));
        indirect += hemiAmbient(R, sky, ground)
                  * F_Schlick(NdotV, mix(float3(0.04), baseColor, U.metallic))
                  * U.envSpec * (1.0 - rough * 0.8);
    }
    if (U.rimStrength > 0.0)
        indirect += sky * (pow(1.0 - max(dot(n, v), 0.0), 3.0) * U.rimStrength);

    // The directional key, kept from the original as a soft two-sided fill.
    const float3 l = normalize(U.lightDir.xyz);
    float3 col = indirect + U.keyColor.xyz * baseColor * (0.04 + 0.06 * abs(dot(n, l)));

    // The mask's own light, sitting just in front of its face -- now a spotlight
    // rather than a bare point. A point light a few units off a face lights the
    // brow, the nose and the surrounding roots equally, which is why the masks
    // read as self-illuminated objects sitting in the tangle rather than as
    // objects someone has aimed a lamp at. A cone falls off towards the edges of
    // the face and leaves the roots around it to the key and the environment.
    //
    // The cone axis is the mask's own facing, recovered from the light's offset:
    // the mesh builder places the light at (face origin + normal * lightDist),
    // so the direction from the light back to the face is the axis, to within
    // the width of the face itself.
    const float3 toLight = in.lightPos - P;
    const float dist = length(toLight);
    const float3 ldir = toLight / max(dist, 0.001);
    float atten = 1.0 / (1.0 + U.lightFalloff * dist * dist);
    if (U.spotCosOuter > -1.0 && U.spotLightDist > 1e-4) {
        // The cone angle without needing the axis. The mesh builder puts the
        // light exactly spotLightDist along the mask's normal from the face
        // plane, so for a point r out from the axis the light is
        // sqrt(spotLightDist^2 + r^2) away -- which makes spotLightDist/dist the
        // cosine of the angle off-axis directly. The alternative was carrying the
        // mask's plane normal through as a fourth vertex attribute; the shading
        // normal cannot stand in for it, because that is the thing that varies
        // across a face and the axis is the thing that must not.
        const float axisCos = clamp(U.spotLightDist / max(dist, 1e-4), 0.0, 1.0);
        atten *= smoothstep(U.spotCosOuter, U.spotCosInner, axisCos);
    }
    float3 direct = ggx(n, v, ldir, baseColor, U.metallic, rough)
                  * (U.lightIntensity * U.specStrength * atten);

    // Skin and stone both pass light a short way through the surface before it
    // comes back out; the wrap term is what stops the terminator from cutting a
    // hard line across a cheekbone.
    if (U.sssWrap > 0.0 || U.sssTrans > 0.0) {
        const float w = max(U.sssWrap, 0.0);
        const float wrapped = max((dot(n, ldir) + w) / ((1.0 + w) * (1.0 + w)), 0.0);
        const float lam = max(dot(n, ldir), 0.0);
        float3 sss = baseColor * max(wrapped - lam, 0.0) * U.sssTint.xyz;
        if (U.sssTrans > 0.0)
            sss += baseColor * U.sssTint.xyz
                 * (pow(saturate(dot(v, -ldir)), max(U.sssPower, 1.0)) * U.sssTrans);
        direct += sss * (U.lightIntensity * atten / PI);
    }
    col += direct;

    const float lumT = dot(col,      float3(0.2126, 0.7152, 0.0722));
    const float lumI = dot(indirect, float3(0.2126, 0.7152, 0.0722));
    return float4(col, saturate(lumI / max(lumT, 1e-5)));
}
