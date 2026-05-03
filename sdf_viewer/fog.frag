#version 410 core

uniform sampler2D u_colorTex;
uniform sampler2D u_depthTex;

uniform vec3  u_eye;
uniform mat3  u_cam;
uniform float u_fov;
uniform vec2  u_res;
uniform float u_near;
uniform float u_far;

uniform vec3  u_fogColor;
uniform float u_fogDensity;
uniform float u_fogFalloff;
uniform float u_fogNoiseScale;
uniform float u_fogNoiseStrength;
uniform float u_fogTime;
uniform int   u_noiseType;      // 0 = value, 1 = simplex

uniform int   u_showAxes;
uniform float u_axisLength;
uniform int   u_showGrid;
uniform float u_gridSpacing;

uniform int   u_wispCount;
uniform vec3  u_wispPos[4];
uniform vec3  u_wispColor[4];
uniform float u_wispIntensity[4];

in vec2  v_uv;
out vec4 fragColor;

// ---------------------------------------------------------------------------
// Value noise — trilinear, smoothstep-filtered
// ---------------------------------------------------------------------------
float _hash(vec3 p) {
    p  = fract(p * vec3(443.897, 397.297, 491.187));
    p += dot(p, p.zyx + 19.19);
    return fract((p.x + p.y) * p.z);
}

float _valueNoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(_hash(i),              _hash(i+vec3(1,0,0)), u.x),
            mix(_hash(i+vec3(0,1,0)),  _hash(i+vec3(1,1,0)), u.x), u.y),
        mix(mix(_hash(i+vec3(0,0,1)),  _hash(i+vec3(1,0,1)), u.x),
            mix(_hash(i+vec3(0,1,1)),  _hash(i+vec3(1,1,1)), u.x), u.y), u.z);
}

// 4-octave fBm, returns [0, 0.9375]
float _valueFbm(vec3 p) {
    return 0.5000 * _valueNoise(p)
         + 0.2500 * _valueNoise(p * 2.03)
         + 0.1250 * _valueNoise(p * 4.07)
         + 0.0625 * _valueNoise(p * 8.11);
}

// ---------------------------------------------------------------------------
// Simplex noise — Gustavson / McEwan
// ---------------------------------------------------------------------------
vec3 _m289v3(vec3 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 _m289v4(vec4 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 _perm(vec4 x)   { return _m289v4(((x * 34.0) + 10.0) * x); }
vec4 _tis(vec4 r)    { return 1.79284291400159 - 0.85373472095314 * r; }

float _snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g  = step(x0.yzx, x0.xyz);
    vec3 l  = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - 0.5;
    i = _m289v3(i);
    vec4 p = _perm(_perm(_perm(
        i.z + vec4(0.0, i1.z, i2.z, 1.0)) +
        i.y + vec4(0.0, i1.y, i2.y, 1.0)) +
        i.x + vec4(0.0, i1.x, i2.x, 1.0));
    const vec3 ns = vec3(0.285714, -0.928571, 0.142857);
    vec4 j  = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 xs = x_ * ns.x + vec4(ns.y);
    vec4 ys = y_ * ns.x + vec4(ns.y);
    vec4 h  = 1.0 - abs(xs) - abs(ys);
    vec4 b0 = vec4(xs.xy, ys.xy);
    vec4 b1 = vec4(xs.zw, ys.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = _tis(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

// 4-octave fBm, returns [-0.9375, 0.9375]
float _simplexFbm(vec3 p) {
    return 0.5000 * _snoise(p)
         + 0.2500 * _snoise(p * 2.03)
         + 0.1250 * _snoise(p * 4.07)
         + 0.0625 * _snoise(p * 8.11);
}

// Both variants normalised to [0, 1], mean ≈ 0.5
float fogFbm(vec3 p) {
    if (u_noiseType == 0)
        return _valueFbm(p) * (1.0 / 0.9375);
    else
        return 0.5 + _simplexFbm(p) * (0.5 / 0.9375);
}

// ---------------------------------------------------------------------------
// Fog density: sigma(y) = density * exp(-falloff * y), noise-modulated
// ---------------------------------------------------------------------------
float fogDensity(vec3 pos) {
    float base = u_fogDensity * exp(-u_fogFalloff * pos.z);
    if (u_fogNoiseStrength < 0.001) return base;
    vec3  np  = pos * u_fogNoiseScale + vec3(u_fogTime * 0.5, 0.0, u_fogTime * 0.3);
    float n   = fogFbm(np);                      // [0, 1], mean 0.5
    return base * mix(1.0, n * 2.0, u_fogNoiseStrength);
}

float marchFog(vec3 ro, vec3 rd, float tMax) {
    const int N    = 16;
    float marchT   = min(tMax, 200.0);
    float stepSize = marchT / float(N);
    float tau = 0.0;
    for (int i = 0; i < N; i++)
        tau += fogDensity(ro + rd * ((float(i) + 0.5) * stepSize)) * stepSize;
    return tau;
}

// ---------------------------------------------------------------------------
// 3D grid SDF
// Infinite lattice of lines along all three axes at multiples of u_gridSpacing.
// Exact Lipschitz-1 SDF: safe for sphere tracing.
// ---------------------------------------------------------------------------
float gridSDF(vec3 p) {
    float s = u_gridSpacing;
    vec3  q = min(mod(p, s), s - mod(p, s));   // dist to nearest plane in each axis
    return min(length(q.yz), min(length(q.xz), length(q.xy)));
}

// ---------------------------------------------------------------------------
// Analytical ray-capsule intersection — returns first positive t, or -1.
// Based on Inigo Quilez's formula.
// ---------------------------------------------------------------------------
float rayCapsule(vec3 ro, vec3 rd, vec3 a, vec3 b, float r) {
    vec3  ba   = b - a, oa = ro - a;
    float baba = dot(ba, ba);
    float bard = dot(ba, rd);
    float baoa = dot(ba, oa);
    float A    = baba - bard * bard;
    float B    = baba * dot(rd, oa) - baoa * bard;
    float C    = baba * dot(oa, oa) - baoa * baoa - r * r * baba;
    float h    = B * B - A * C;
    if (h < 0.0) return -1.0;
    float t = (-B - sqrt(h)) / A;
    float y = baoa + t * bard;
    if (y > 0.0 && y < baba && t > 0.0) return t;     // cylinder body
    vec3  oc = y <= 0.0 ? oa : ro - b;                 // pick nearest cap
    float B2 = dot(rd, oc), C2 = dot(oc, oc) - r * r;
    float h2 = B2 * B2 - C2;
    if (h2 < 0.0) return -1.0;
    t = -B2 - sqrt(h2);
    return t > 0.0 ? t : -1.0;
}

// ---------------------------------------------------------------------------
void main() {
    vec3  sceneColor  = texture(u_colorTex, v_uv).rgb;
    float depthSample = texture(u_depthTex,  v_uv).r;

    // Reconstruct world-space ray (mirrors the geometry shader)
    vec2 ndc = v_uv * 2.0 - 1.0;
    ndc.x *= u_res.x / u_res.y;
    vec3 rd = normalize(u_cam * vec3(ndc * tan(u_fov), 1.0));
    vec3 ro = u_eye;

    // Linearise depth buffer → eye-space depth → ray distance
    float ndcZ = depthSample * 2.0 - 1.0;
    float ze   = (2.0 * u_near * u_far) / (u_far + u_near - ndcZ * (u_far - u_near));
    float hitT = ze / max(dot(rd, u_cam[2]), 1e-4);

    // ---- Axes (analytical ray-capsule) ----
    float tAxis     = 1e9;
    vec3  axisColor = vec3(0.0);
    if (u_showAxes == 1) {
        const float R = 0.15;
        float tx = rayCapsule(ro, rd, vec3(0.0), vec3(u_axisLength, 0.0,          0.0         ), R);
        float ty = rayCapsule(ro, rd, vec3(0.0), vec3(0.0,          u_axisLength, 0.0         ), R);
        float tz = rayCapsule(ro, rd, vec3(0.0), vec3(0.0,          0.0,          u_axisLength), R);
        if (tx > 0.0 && tx < tAxis) { tAxis = tx; axisColor = vec3(0.85, 0.15, 0.15); }
        if (ty > 0.0 && ty < tAxis) { tAxis = ty; axisColor = vec3(0.15, 0.85, 0.15); }
        if (tz > 0.0 && tz < tAxis) { tAxis = tz; axisColor = vec3(0.15, 0.35, 0.85); }
    }

    // ---- 3D grid (sphere-traced) ----
    float tGrid  = 1e9;
    if (u_showGrid == 1) {
        float t      = 0.001;
        float limit  = min(hitT, 80.0);
        float thresh = u_gridSpacing * 0.012;
        const int GSTEPS = 64;
        for (int i = 0; i < GSTEPS; i++) {
            if (t >= limit) break;
            float d = gridSDF(ro + rd * t);
            if (d < thresh) { tGrid = t; break; }
            t += max(d * 0.8, 0.002);
        }
    }

    // ---- Composite: nearest among scene / grid / axes ----
    float tHit    = hitT;
    vec3  hitColor = sceneColor;

    if (tGrid < tHit) { tHit = tGrid; hitColor = vec3(0.28, 0.30, 0.36); }
    if (tAxis < tHit) { tHit = tAxis; hitColor = axisColor; }

    // Apply fog from camera to the nearest surface
    float tau = marchFog(ro, rd, tHit);
    vec3  foggedColor = mix(u_fogColor, hitColor, exp(-tau));

    // Wisp glow — soft emissive blobs floating in the atmosphere
    vec3 wispGlow = vec3(0.0);
    const float GLOW_R = 5.0;  // glow radius in world units
    for (int wi = 0; wi < u_wispCount; wi++) {
        vec3  oc  = ro - u_wispPos[wi];
        float b   = dot(rd, oc);
        float tc  = max(0.0, -b);                  // closest t along ray, clamped to front
        vec3  cp  = (ro + rd * tc) - u_wispPos[wi];
        float d2  = dot(cp, cp);
        float glow = u_wispIntensity[wi] * exp(-d2 / (GLOW_R * GLOW_R));
        // Attenuate by fog between camera and wisp, and occlude behind geometry
        float tauWisp = marchFog(ro, rd, tc);
        glow *= exp(-tauWisp) * step(tc, tHit + 0.5);
        wispGlow += u_wispColor[wi] * glow;
    }

    fragColor = vec4(foggedColor + wispGlow * 0.15, 1.0);
}
