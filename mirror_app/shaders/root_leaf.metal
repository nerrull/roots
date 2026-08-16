// root_leaf.metal — mid-geometry pass for MESHED leaves.
//
// Leaves used to be drawn on the capsule/blade path, as a blade SDF whose
// outline was carved from a union of capsules. That gives every lobe a circular
// tip and the whole margin a rounded, constant-thickness edge — it reads as a
// moulded plastic part. The mesh (see sdf_viewer/LeafMesh.h) instead thins to a
// knife edge at the margin and swells only over the veins, which is geometry the
// SDF path cannot express, so leaves get rasterized here.
//
// Same contract as root_face.metal: triangles into the SAME colour + depth
// targets as the capsules, between the capsule pass and the fog pass, so they
// depth-composite against sphere-traced geometry and are fogged with it. The
// vertex layout matches the shared mid-geometry vertex (pos3, normal3, colour3,
// param3); for a leaf, param is (s along the midrib, t across, vein proximity).
//
// Depth: the capsule pass writes (clip.z/clip.w)*0.5+0.5 (GL convention), so the
// vertex shader remaps GL clip-z [-1,1] into Metal's [0,1] exactly as the face
// pass does, and the two composite.
#include <metal_stdlib>
using namespace metal;

struct LeafVertex {
    packed_float3 pos;
    packed_float3 normal;
    packed_float3 color;
    packed_float3 param;      // (s, t, vein)
};

struct LeafVOut {
    float4 pos [[position]];
    float3 worldPos;
    float3 normal;
    float3 color;
    float3 param;
};

vertex LeafVOut root_leaf_vs(uint vid [[vertex_id]],
                             device const LeafVertex* verts [[buffer(0)]],
                             constant RootLeafU&      U     [[buffer(1)]]) {
    LeafVertex v = verts[vid];
    LeafVOut o;
    o.worldPos = float3(v.pos);
    o.normal   = float3(v.normal);
    o.color    = float3(v.color);
    o.param    = float3(v.param);
    float4 c = U.viewProj * float4(float3(v.pos), 1.0);
    c.z = (c.z + c.w) * 0.5;   // GL [-1,1] clip-z -> Metal [0,1]
    o.pos = c;
    return o;
}

static float3 F_Schlick_l(float c, float3 F0) { return F0 + (1.0 - F0) * pow(1.0 - c, 5.0); }

fragment float4 root_leaf_fs(LeafVOut in [[stage_in]],
                             constant RootLeafU& U [[buffer(1)]]) {
    float3 n = normalize(in.normal);
    const float3 v = normalize(U.eye.xyz - in.worldPos);
    // Two-sided: the pass draws with culling off so both faces of the blade are
    // rasterized, and a leaf seen from below must still shade.
    if (dot(n, v) < 0.0) n = -n;

    const float3 l = normalize(U.lightDir.xyz);
    const float vein = clamp(in.param.z, 0.0, 1.0);

    float3 albedo = in.color;

    // Diffuse + a broad, weak specular. Leaves are matte with a slight sheen;
    // the veins sit a touch rougher than the lamina.
    const float rough = clamp(U.roughness + 0.18 * vein, 0.05, 1.0);
    const float NdotL = max(dot(n, l), 0.0);
    const float3 h = normalize(v + l);
    const float NdotH = max(dot(n, h), 0.0);
    const float spec = pow(NdotH, max(2.0, 64.0 * (1.0 - rough))) * U.specStrength * (1.0 - 0.5 * vein);

    // Hemispheric ambient, so an unlit underside is not pure black.
    const float3 amb = mix(U.groundColor.xyz, U.skyColor.xyz, n.y * 0.5 + 0.5) * U.hemiStrength;

    // Translucency: a leaf lit from behind glows. Cheap wrap-around term driven
    // by how much the light comes through the far side.
    const float back = pow(clamp(dot(-n, l) * 0.5 + 0.5, 0.0, 1.0), U.sssPower);
    const float3 trans = U.sssTint.xyz * albedo * (back * U.sssTrans);

    float3 c = albedo * (amb + NdotL * U.diffuse) + float3(spec) + trans;

    // Rim light picks the leaf's silhouette out of the fog.
    const float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * U.rimStrength;
    c += albedo * rim;

    return float4(c, 1.0);
}
