// root_fog.metal — Metal port of sdf_viewer's fog.vert + fog.frag.
//
// Fullscreen pass: reads the geometry pass's colour + depth, applies an FXAA-lite
// smooth, volumetric noise fog (camera-stable option), optional axes/grid
// overlays, and soft wisp glow. root_shared.h (host-prepended) supplies RootFogU /
// RootWisp. The ray/uv reconstruction uses the fragment's pixel position so it
// matches the geometry pass's NDC exactly (Metal textures are row-0-at-top).
#include <metal_stdlib>
using namespace metal;

static float  glmod(float x, float y)   { return x - y * floor(x / y); }
static float3 glmod3(float3 p, float s) { return p - s * floor(p / s); }

struct FogVOut { float4 pos [[position]]; };

vertex FogVOut root_fog_vs(uint vid [[vertex_id]]) {
    // Fullscreen triangle in clip space.
    float2 p = float2((vid << 1) & 2, vid & 2);
    FogVOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

static float fogFbm(float3 p, texture3d<float> noiseTex, sampler smp) {
    return noiseTex.sample(smp, p * (1.0 / ROOT_NOISE_TILE_PERIOD)).r;
}

static float fogDensity(float3 pos, constant RootFogU& U,
                        texture3d<float> noiseTex, sampler smp) {
    float base = U.fogDensity * exp(-U.fogFalloff * pos.y);
    if (U.fogNoiseStrength < 0.001) return base;
    float3 np = pos * U.fogNoiseScale + float3(U.fogTime * 0.5, 0.0, U.fogTime * 0.3);
    float n = fogFbm(np, noiseTex, smp);
    return base * mix(1.0, n * 2.0, U.fogNoiseStrength);
}

static float marchFog(float3 ro, float3 rd, float tMax, constant RootFogU& U,
                     texture3d<float> noiseTex, sampler smp) {
    if (U.fogDensity <= 0.0) return 0.0;
    const int N = 8;
    float marchT = min(tMax, 200.0);
    float stepSize = marchT / float(N);
    float tau = 0.0;
    for (int i = 0; i < N; i++)
        tau += fogDensity(ro + rd * ((float(i) + 0.5) * stepSize), U, noiseTex, smp) * stepSize;
    return tau;
}

static float gridSDF(float3 p, float s) {
    float3 q = min(glmod3(p, s), s - glmod3(p, s));
    return min(length(q.yz), min(length(q.xz), length(q.xy)));
}

static float rayCapsule(float3 ro, float3 rd, float3 a, float3 b, float r) {
    float3 ba = b - a, oa = ro - a;
    float baba = dot(ba, ba);
    float bard = dot(ba, rd);
    float baoa = dot(ba, oa);
    float A = baba - bard * bard;
    float B = baba * dot(rd, oa) - baoa * bard;
    float C = baba * dot(oa, oa) - baoa * baoa - r * r * baba;
    float h = B * B - A * C;
    if (h < 0.0) return -1.0;
    float t = (-B - sqrt(h)) / A;
    float y = baoa + t * bard;
    if (y > 0.0 && y < baba && t > 0.0) return t;
    float3 oc = y <= 0.0 ? oa : ro - b;
    float B2 = dot(rd, oc), C2 = dot(oc, oc) - r * r;
    float h2 = B2 * B2 - C2;
    if (h2 < 0.0) return -1.0;
    t = -B2 - sqrt(h2);
    return t > 0.0 ? t : -1.0;
}

fragment float4 root_fog_fs(FogVOut in [[stage_in]],
                            constant RootFogU&    U        [[buffer(0)]],
                            device const RootWisp* wisps    [[buffer(1)]],
                            texture2d<float>      colorTex [[texture(0)]],
                            depth2d<float>        depthTex [[texture(1)]],
                            texture3d<float>      noiseTex [[texture(2)]]) {
    constexpr sampler linSmp(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    constexpr sampler noiseSmp(mag_filter::linear, min_filter::linear,
                               address::repeat, mip_filter::linear);

    float2 uv = in.pos.xy / U.res;    // row 0 at top (Metal), matches geometry write

    float3 sceneColor  = colorTex.sample(linSmp, uv).rgb;
    float  depthSample = depthTex.sample(linSmp, uv);

    // FXAA-lite (see fog.frag rationale)
    {
        float2 texel = 1.0 / U.res;
        float3 nW = colorTex.sample(linSmp, uv + float2(-texel.x, 0.0)).rgb;
        float3 nE = colorTex.sample(linSmp, uv + float2( texel.x, 0.0)).rgb;
        float3 nN = colorTex.sample(linSmp, uv + float2(0.0,  texel.y)).rgb;
        float3 nS = colorTex.sample(linSmp, uv + float2(0.0, -texel.y)).rgb;
        float3 avg = (nW + nE + nN + nS) * 0.25;
        float contrast = length(sceneColor - avg);
        float blend = smoothstep(0.04, 0.22, contrast) * 0.65;
        sceneColor = mix(sceneColor, avg, blend);
    }

    // Reconstruct the world ray (matches root_geom_vs's NDC).
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
    ndc.x *= U.res.x / U.res.y;
    float3 rd = normalize(U.cam * float3(ndc * tan(U.fov), 1.0));
    float3 ro = U.eye.xyz;

    float ndcZ = depthSample * 2.0 - 1.0;
    float ze   = (2.0 * U.nearZ * U.farZ) / (U.farZ + U.nearZ - ndcZ * (U.farZ - U.nearZ));
    float hitT = ze / max(dot(rd, U.cam[2]), 1e-4);

    // Axes
    float tAxis = 1e9;
    float3 axisColor = float3(0.0);
    if (U.showAxes == 1) {
        const float R = 0.15;
        float tx = rayCapsule(ro, rd, float3(0.0), float3(U.axisLength, 0.0, 0.0), R);
        float ty = rayCapsule(ro, rd, float3(0.0), float3(0.0, U.axisLength, 0.0), R);
        float tz = rayCapsule(ro, rd, float3(0.0), float3(0.0, 0.0, U.axisLength), R);
        if (tx > 0.0 && tx < tAxis) { tAxis = tx; axisColor = float3(0.85, 0.15, 0.15); }
        if (ty > 0.0 && ty < tAxis) { tAxis = ty; axisColor = float3(0.15, 0.85, 0.15); }
        if (tz > 0.0 && tz < tAxis) { tAxis = tz; axisColor = float3(0.15, 0.35, 0.85); }
    }

    // Grid
    float tGrid = 1e9;
    if (U.showGrid == 1) {
        float t = 0.001;
        float limit = min(hitT, 80.0);
        float thresh = U.gridSpacing * 0.012;
        const int GSTEPS = 64;
        for (int i = 0; i < GSTEPS; i++) {
            if (t >= limit) break;
            float d = gridSDF(ro + rd * t, U.gridSpacing);
            if (d < thresh) { tGrid = t; break; }
            t += max(d * 0.8, 0.002);
        }
    }

    float tHit = hitT;
    float3 hitColor = sceneColor;
    if (tGrid < tHit) { tHit = tGrid; hitColor = float3(0.28, 0.30, 0.36); }
    if (tAxis < tHit) { tHit = tAxis; hitColor = axisColor; }

    float tau = marchFog(ro, rd, tHit, U, noiseTex, noiseSmp);
    tau = max(tau - U.fogDensity * U.fogRefDist, 0.0);
    float3 foggedColor = mix(U.fogColor.xyz, hitColor, exp(-tau));

    // Wisp glow
    float3 wispGlow = float3(0.0);
    const float GLOW_R = 5.0;
    const float GLOW_CUTOFF2 = (4.0 * GLOW_R) * (4.0 * GLOW_R);
    float tauPerT = tau / max(min(tHit, 200.0), 1e-3);
    for (int wi = 0; wi < U.wispCount; wi++) {
        float3 oc = ro - wisps[wi].pos.xyz;
        float bo = dot(rd, oc);
        float tc = max(0.0, -bo);
        float3 cp = (ro + rd * tc) - wisps[wi].pos.xyz;
        float d2 = dot(cp, cp);
        if (d2 > GLOW_CUTOFF2) continue;
        float glow = wisps[wi].pos.w * exp(-d2 / (GLOW_R * GLOW_R));
        float tauWisp = tauPerT * min(tc, 200.0);
        glow *= exp(-tauWisp) * step(tc, tHit + 0.5);
        wispGlow += wisps[wi].color.xyz * glow;
    }

    float glowGate = clamp(U.fogDensity / 0.015, 0.0, 1.0) * U.wispGlowStrength;
    return float4(foggedColor + wispGlow * 0.15 * glowGate, 1.0);
}
