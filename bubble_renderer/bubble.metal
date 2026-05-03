#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float3 camPos;        // offset  0, size 16
    float3 camDir;        // offset 16, size 16
    float3 camRight;      // offset 32, size 16
    float3 camUp;         // offset 48, size 16
    float  time;          // offset 64
    float  filmThickness; // offset 68  (nanometres)
    float  bubbleRadius;  // offset 72
    float  lightAzimuth;  // offset 76
    float  lightElevation;// offset 80
    float  aspectRatio;   // offset 84
    float  fovTan;        // offset 88  (tan of half-FOV)
    float  _pad;          // offset 92
};

// ---------- noise ----------

float hash3(float3 p) {
    p = fract(p * float3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

float valueNoise(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    float3 s = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(hash3(i + float3(0,0,0)), hash3(i + float3(1,0,0)), s.x),
            mix(hash3(i + float3(0,1,0)), hash3(i + float3(1,1,0)), s.x), s.y),
        mix(mix(hash3(i + float3(0,0,1)), hash3(i + float3(1,0,1)), s.x),
            mix(hash3(i + float3(0,1,1)), hash3(i + float3(1,1,1)), s.x), s.y),
        s.z);
}

float filmNoise(float3 p, float t) {
    float3 drift = float3(0.07, -0.04, 0.05) * t;
    float n  = valueNoise(p * 1.2 + drift)       * 0.65
             + valueNoise(p * 2.8 + drift * 1.5) * 0.35;
    return n;
}

// ---------- thin-film interference ----------

float3 thinFilm(float cosTheta, float thickness) {
    constexpr float n = 1.45;
    float sinT2 = (1.0 - cosTheta * cosTheta) / (n * n);
    float cosT  = sqrt(max(0.0, 1.0 - sinT2));
    float opd   = 2.0 * n * thickness * cosT;
    float3 phi  = 2.0 * M_PI_F * opd / float3(700.0, 546.0, 435.0);
    phi += M_PI_F;
    return saturate(0.5 + 0.5 * cos(phi));
}

float fresnel(float cosTheta, float F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// ---------- SDF ----------

// Domain-warped sphere: wobbles organically over time.
float sdBubble(float3 p, float r, float time) {
    float3 drift = float3(0.13, -0.07, 0.09) * time;
    float3 warp  = float3(
        valueNoise(p * 1.8 + drift),
        valueNoise(p * 1.8 + drift + float3(3.7, 0.0, 0.0)),
        valueNoise(p * 1.8 + drift + float3(0.0, 7.3, 0.0))
    ) * 2.0 - 1.0;
    return length(p + warp * r * 0.08) - r;
}

float smin(float a, float b, float k) {
    float h = saturate(0.5 + 0.5 * (b - a) / k);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Scene SDF: one bubble for now; add more with smin() to blend.
float sdScene(float3 p, float r, float time) {
    float d = sdBubble(p, r, time);
    // float d2 = sdBubble(p - float3(r * 1.4, 0, 0), r * 0.9, time);
    // d = smin(d, d2, r * 0.4);
    return d;
}

float3 sceneNormal(float3 p, float r, float time) {
    const float e = 0.003;
    return normalize(float3(
        sdScene(p + float3(e,0,0), r, time) - sdScene(p - float3(e,0,0), r, time),
        sdScene(p + float3(0,e,0), r, time) - sdScene(p - float3(0,e,0), r, time),
        sdScene(p + float3(0,0,e), r, time) - sdScene(p - float3(0,0,e), r, time)
    ));
}

// ---------- rotating cube ----------

float3x3 cubeRotation(float t) {
    float ay = t * 0.7, ax = t * 0.4;
    float cy = cos(ay), sy = sin(ay);
    float cx = cos(ax), sx = sin(ax);
    float3x3 Ry = float3x3(float3(cy,0,-sy), float3(0,1,0), float3(sy,0,cy));
    float3x3 Rx = float3x3(float3(1,0,0), float3(0,cx,sx), float3(0,-sx,cx));
    return Ry * Rx;
}

float cubeHit(float3 ro, float3 rd, float hs, float3x3 rotInv,
              thread float3& outNormalLocal) {
    float3 roL = rotInv * ro;
    float3 rdL = rotInv * rd;

    float3 invD = 1.0 / rdL;
    float3 t0   = (-hs - roL) * invD;
    float3 t1   = ( hs - roL) * invD;
    float3 tmi  = min(t0, t1);
    float3 tma  = max(t0, t1);

    float tNear = max(max(tmi.x, tmi.y), tmi.z);
    float tFar  = min(min(tma.x, tma.y), tma.z);

    if (tFar < 0.001 || tNear > tFar) return -1.0;

    float tHit = tNear > 0.001 ? tNear : tFar;
    if (tHit < 0.001) return -1.0;

    float3 pL  = roL + tHit * rdL;
    float3 absP = abs(pL) / hs;
    if      (absP.x > absP.y && absP.x > absP.z) outNormalLocal = float3(sign(pL.x), 0, 0);
    else if (absP.y > absP.z)                     outNormalLocal = float3(0, sign(pL.y), 0);
    else                                           outNormalLocal = float3(0, 0, sign(pL.z));

    return tHit;
}

float3 faceColor(float3 nLocal) {
    if (nLocal.x >  0.5) return float3(0.90, 0.20, 0.20);
    if (nLocal.x < -0.5) return float3(1.00, 0.55, 0.10);
    if (nLocal.y >  0.5) return float3(0.20, 0.80, 0.25);
    if (nLocal.y < -0.5) return float3(0.20, 0.30, 0.90);
    if (nLocal.z >  0.5) return float3(0.90, 0.85, 0.15);
                         return float3(0.75, 0.20, 0.80);
}

float3 surfaceIrid(float cosNV, float3 noisePos,
                   float filmThickness, float bubbleRadius, float time) {
    float noise = filmNoise(noisePos * bubbleRadius, time);
    float thick = filmThickness + (noise - 0.5) * 300.0;
    return (thinFilm(cosNV, thick) + thinFilm(cosNV, thick * 0.96 + 8.0)) * 0.5;
}

// ---------- kernel ----------

kernel void bubble_kernel(
    texture2d<float, access::write> outTex [[texture(0)]],
    constant Uniforms&              u      [[buffer(0)]],
    uint2                           tid    [[thread_position_in_grid]])
{
    uint texW = outTex.get_width();
    uint texH = outTex.get_height();
    if (tid.x >= texW || tid.y >= texH) return;

    float2 uv  = (float2(tid) + 0.5) / float2(texW, texH) * 2.0 - 1.0;
    uv.y       = -uv.y;
    uv.x      *= u.aspectRatio;
    float3 rd  = normalize(u.camDir + u.fovTan * (uv.x * u.camRight + uv.y * u.camUp));
    float3 ro  = u.camPos;

    float3 bgTop = float3(0.005, 0.005, 0.015);
    float3 bgBot = float3(0.02,  0.01,  0.04);
    float3 bg    = mix(bgTop, bgBot, uv.y * 0.5 + 0.5);

    float tMax = u.bubbleRadius * 4.0;

    // ---- March front surface (outside → in) ----
    float tFront = -1.0;
    {
        float t = 0.0;
        for (int i = 0; i < 96; i++) {
            float d = sdScene(ro + t * rd, u.bubbleRadius, u.time);
            if (d < 0.0005) { tFront = t; break; }
            t += d;
            if (t > tMax) break;
        }
    }

    if (tFront < 0.0) { outTex.write(float4(bg, 1.0), tid); return; }

    // ---- March back surface (inside → out, negate SDF) ----
    float tBack = -1.0;
    {
        float t = tFront + 0.01;
        float limit = tFront + u.bubbleRadius * 3.0;
        for (int i = 0; i < 64; i++) {
            float d = -sdScene(ro + t * rd, u.bubbleRadius, u.time);
            if (d < 0.0005) { tBack = t; break; }
            t += max(d, 0.002 * u.bubbleRadius);
            if (t > limit) break;
        }
    }

    float3 V = -rd;
    float3 L = normalize(float3(
        cos(u.lightElevation) * sin(u.lightAzimuth),
        sin(u.lightElevation),
        cos(u.lightElevation) * cos(u.lightAzimuth)
    ));

    // ---- Back surface ----
    float3 irid2 = float3(0.0);
    float  F2 = 0.0, spec2 = 0.0;
    if (tBack > 0.0) {
        float3 P2    = ro + tBack * rd;
        float3 N2    = -sceneNormal(P2, u.bubbleRadius, u.time); // inward
        float cosNV2 = max(0.0, dot(N2, V));
        irid2 = surfaceIrid(cosNV2, normalize(P2), u.filmThickness, u.bubbleRadius, u.time);
        F2    = fresnel(cosNV2, 0.025);
        float3 H2 = normalize(V + L);
        spec2 = pow(max(0.0, dot(N2, H2)), 80.0);
    }

    // ---- Front surface ----
    float3 P1    = ro + tFront * rd;
    float3 N1    = sceneNormal(P1, u.bubbleRadius, u.time);
    float cosNV1 = max(0.0, dot(N1, V));
    float3 irid1 = surfaceIrid(cosNV1, normalize(P1), u.filmThickness, u.bubbleRadius, u.time);
    float  F1    = fresnel(cosNV1, 0.025);
    float3 H1    = normalize(V + L);
    float  spec1 = pow(max(0.0, dot(N1, H1)), 120.0);
    float  edge1 = pow(1.0 - cosNV1, 2.5);

    // ---- Refract background through front surface ----
    float3 bgRd = rd;
    float3 refr = refract(rd, N1, 1.0 / 1.05);
    if (any(refr != float3(0.0))) bgRd = refr;
    float bgV  = dot(normalize(bgRd), float3(0, 1, 0)) * 0.5 + 0.5;
    float3 bgR = mix(bgTop, bgBot, bgV);

    // ---- Rotating cube inside the bubble ----
    float3x3 rot    = cubeRotation(u.time);
    float3x3 rotInv = transpose(rot);
    float cubeHalf  = u.bubbleRadius * 0.38;

    float3 nLocal;
    float tC = cubeHit(ro, rd, cubeHalf, rotInv, nLocal);

    float tBackSafe = tBack > 0.0 ? tBack : tMax;
    bool cubeVisible = tC > tFront + 0.001 && tC < tBackSafe - 0.001;

    float3 innerColor = bgR;
    if (cubeVisible) {
        float3 nW   = rot * nLocal;
        float  diff = max(0.0, dot(nW, L));
        float3 Hc   = normalize(V + L);
        float  spec = pow(max(0.0, dot(nW, Hc)), 60.0);
        float3 base = faceColor(nLocal);
        innerColor  = base * (0.12 + 0.88 * diff) + float3(spec) * 0.45;
    }

    // ---- Composite back → front ----
    float transF2 = 1.0 - F2 * 0.85;
    float transF1 = 1.0 - F1 * 0.85;

    float3 color = innerColor * transF2 * transF1;
    color += (irid2 * F2 * 3.0 + float3(spec2) * 0.25) * (1.0 - F1 * 0.5);
    color += irid1 * F1 * 4.0
           + irid1 * edge1 * 1.2
           + float3(spec1) * 0.7;

    outTex.write(float4(color, 1.0), tid);
}
