// root_ao.metal — screen-space ambient occlusion over the geometry pass's depth
// buffer, plus the separable bilateral blur that cleans it up.
//
// Why this scene wants AO more than most: the roots are a dense tangle of thin
// tubes that pass in front of and behind one another constantly, and with a
// single directional light and a constant ambient term there is nothing in the
// image that says which tube is in front. Contact darkening where they cross,
// and where a mask sits inside the tangle, is what turns a pile of separately
// shaded cylinders into one connected body.
//
// Normals are reconstructed from depth rather than written out by the geometry
// pass. A G-buffer normal target would be more accurate, but the geometry pass
// is already fragment-bound on the sphere-tracer and a second colour attachment
// costs it bandwidth on every one of those fragments -- whereas this pass runs
// at quarter the pixels. The reconstruction uses the closest-neighbour variant
// (pick the horizontal and vertical neighbour whose depth is nearest the
// centre), which is what keeps silhouettes on thin tubes from producing normals
// that face nowhere.
#include <metal_stdlib>
using namespace metal;

struct AOVOut { float4 pos [[position]]; };

vertex AOVOut root_ao_vs(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    AOVOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

static float ign(float2 p) {
    return fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715))));
}

// GL-convention depth (see root_geom.metal) -> positive view-space distance.
static float linearDepth(float d, float nearZ, float farZ) {
    const float ndcZ = d * 2.0 - 1.0;
    return (2.0 * nearZ * farZ) / (farZ + nearZ - ndcZ * (farZ - nearZ));
}

// View-space position for a uv, reconstructed the same way root_fog.metal
// reconstructs its ray, so the two passes agree about where a pixel is.
static float3 viewPos(float2 uv, float d, constant RootAOU& U) {
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
    ndc.x *= U.res.x / U.res.y;
    const float3 dirV = float3(ndc * tan(U.fov), 1.0);   // view space, +Z forward
    const float ze = linearDepth(d, U.nearZ, U.farZ);
    return dirV * ze;   // dirV.z == 1, so this lands exactly at distance ze
}

fragment float4 root_ao_fs(AOVOut in [[stage_in]],
                           constant RootAOU& U       [[buffer(0)]],
                           depth2d<float>    depthTex [[texture(0)]]) {
    constexpr sampler pt(mag_filter::nearest, min_filter::nearest,
                         address::clamp_to_edge);

    const float2 uv = in.pos.xy / U.res;
    const float2 texel = 1.0 / U.res;
    const float dC = depthTex.sample(pt, uv);
    if (dC >= 0.9999) return float4(1.0);   // background: nothing to occlude

    const float3 P = viewPos(uv, dC, U);

    // Closest-neighbour derivatives.
    const float dL = depthTex.sample(pt, uv - float2(texel.x, 0.0));
    const float dR = depthTex.sample(pt, uv + float2(texel.x, 0.0));
    const float dU = depthTex.sample(pt, uv - float2(0.0, texel.y));
    const float dD = depthTex.sample(pt, uv + float2(0.0, texel.y));
    const float3 pL = viewPos(uv - float2(texel.x, 0.0), dL, U);
    const float3 pR = viewPos(uv + float2(texel.x, 0.0), dR, U);
    const float3 pU = viewPos(uv - float2(0.0, texel.y), dU, U);
    const float3 pD = viewPos(uv + float2(0.0, texel.y), dD, U);
    const float3 ddx = (abs(dR - dC) < abs(dC - dL)) ? (pR - P) : (P - pL);
    const float3 ddy = (abs(dD - dC) < abs(dC - dU)) ? (pD - P) : (P - pU);
    float3 N = cross(ddx, ddy);
    const float nl = length(N);
    if (nl < 1e-9) return float4(1.0);
    N /= nl;
    if (N.z > 0.0) N = -N;   // face the camera (view space looks down +Z here)

    // Cosine-weighted hemisphere taps around N, spun by a per-pixel angle so the
    // sample pattern becomes noise the blur can remove rather than a fixed
    // rosette baked into the image.
    const float rot = ign(in.pos.xy) * 6.2831853;
    const float3 up = (abs(N.z) < 0.9) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    const float3 T = normalize(cross(up, N));
    const float3 B = cross(N, T);

    const int   n = max(U.samples, 1);
    const float invN = 1.0 / float(n);
    float occ = 0.0;
    for (int i = 0; i < n; ++i) {
        // Sunflower spiral in the disc, lifted onto the hemisphere. Radius
        // scaled by i/n as well, so taps cluster near the centre where contact
        // shadowing actually lives.
        const float fi = (float(i) + 0.5) * invN;
        const float ang = rot + fi * 6.2831853 * 3.883;   // ~golden-angle turns
        const float rad = sqrt(fi);
        const float3 dir = normalize(T * (cos(ang) * rad) + B * (sin(ang) * rad)
                                     + N * sqrt(max(1.0 - rad * rad, 0.0)));
        const float3 S = P + dir * (U.radius * (0.35 + 0.65 * fi));

        // Project the sample back to the screen and read what the depth buffer
        // actually has there.
        const float2 sNdc = float2(S.x / (S.z * tan(U.fov)) * (U.res.y / U.res.x),
                                   S.y / (S.z * tan(U.fov)));
        const float2 sUv = float2(sNdc.x * 0.5 + 0.5, 0.5 - sNdc.y * 0.5);
        if (any(sUv < 0.0) || any(sUv > 1.0)) continue;
        const float sD = depthTex.sample(pt, sUv);
        if (sD >= 0.9999) continue;
        const float sceneZ = linearDepth(sD, U.nearZ, U.farZ);

        // Occluded if the scene is in front of the sample, but only within
        // `radius` -- without the range check a distant tube seen past a near
        // one paints a black halo around the near one's silhouette.
        const float diff = S.z - sceneZ;
        if (diff > U.bias) {
            occ += smoothstep(1.0, 0.0, (diff - U.bias) / U.radius);
        }
    }
    const float ao = saturate(1.0 - occ * invN * U.intensity);
    return float4(ao, 0.0, 0.0, 1.0);
}

// Separable cross-bilateral blur: averages the noisy AO but refuses to average
// across a depth discontinuity, so the occlusion stays attached to the tube that
// cast it instead of bleeding onto whatever is behind it.
fragment float4 root_ao_blur_fs(AOVOut in [[stage_in]],
                                constant RootAOU& U        [[buffer(0)]],
                                texture2d<float>  aoTex    [[texture(0)]],
                                depth2d<float>    depthTex [[texture(1)]]) {
    constexpr sampler pt(mag_filter::nearest, min_filter::nearest,
                         address::clamp_to_edge);
    const float2 uv = in.pos.xy / U.res;
    const float2 texel = 1.0 / U.res;
    const float2 step = (U.blurDir == 0) ? float2(texel.x, 0.0) : float2(0.0, texel.y);

    const float zC = linearDepth(depthTex.sample(pt, uv), U.nearZ, U.farZ);
    // A depth tolerance proportional to distance: a 5cm gap is a hard edge up
    // close and nothing at all across the room.
    const float tol = max(zC * 0.02, 0.05);

    float sum = 0.0, wsum = 0.0;
    for (int i = -3; i <= 3; ++i) {
        const float2 s = uv + step * float(i);
        const float z = linearDepth(depthTex.sample(pt, s), U.nearZ, U.farZ);
        const float wG = exp(-float(i * i) / 8.0);
        const float wD = (abs(z - zC) < tol) ? 1.0 : 0.0;
        const float w = wG * wD;
        sum += aoTex.sample(pt, s).r * w;
        wsum += w;
    }
    return float4(wsum > 0.0 ? sum / wsum : aoTex.sample(pt, uv).r, 0.0, 0.0, 1.0);
}
