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
uniform float u_fogRefDist;         // >0: camera->target distance whose mean
                                    // optical depth is subtracted out, so fog
                                    // encodes relative depth within the scene
                                    // instead of absolute camera distance
uniform float u_wispGlowStrength;   // scales the atmospheric face-light glow
uniform int   u_noiseType;      // 0 = value, 1 = simplex

uniform int   u_showAxes;
uniform float u_axisLength;
uniform int   u_showGrid;
uniform float u_gridSpacing;

uniform int   u_wispCount;
uniform vec3  u_wispPos[50];
uniform vec3  u_wispColor[50];
uniform float u_wispIntensity[50];

uniform sampler3D u_noiseTex;   // baked tiling 4-octave value fBm, [0,1]

in vec2  v_uv;
out vec4 fragColor;

// ---------------------------------------------------------------------------
// fBm via one trilinear fetch into a baked tiling 3D texture.
//
// This used to be a fully procedural 4-octave fBm -- 32 hash evaluations per
// call, called 16x/pixel by marchFog(): profiling showed this single shader
// was ~96% of total frame time at 4K (~199ms of ~208ms). The baked texture
// is the same noise statistically (see RootRenderer::buildNoiseTexture) at
// ~1/500th the ALU. u_noiseType (value vs simplex) is intentionally ignored
// now: both were normalized to the same [0,1] range/mean, and fog is far too
// soft for their character difference to read -- not worth a second texture.
// ---------------------------------------------------------------------------
const float NOISE_TILE_PERIOD = 8.0;   // must match RootRenderer.cpp

float fogFbm(vec3 p) {
    return texture(u_noiseTex, p * (1.0 / NOISE_TILE_PERIOD)).r;
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
    if (u_fogDensity <= 0.0) return 0.0;   // invert mode zeroes density -- skip the march
    // 8 steps, down from 16: even 16 undersampled the noise wavelength over a
    // 200-unit ray, and fog this soft shows no banding at 8 -- but it halves
    // what profiling showed is the remaining per-pixel cost of this pass
    // (the 3D-texture fetch per step).
    const int N    = 8;
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

    // Cheap edge-aware smoothing (FXAA-lite): the root capsules are sphere-
    // traced with a hard discard at the silhouette, not rasterized triangle
    // edges, so hardware MSAA wouldn't help even if we had it (GL 4.1 has no
    // compute shaders to build a proper AA solution either). This runs at
    // u_res -- the INTERNAL render resolution (renderScale, can be well below
    // the window's real size) -- so it's cheap regardless of final output
    // size, and matters more than usual here since upscaling a lower-res
    // render otherwise just blows up its jaggies rather than smoothing them.
    {
        vec2 texel = 1.0 / u_res;
        vec3 nW = texture(u_colorTex, v_uv + vec2(-texel.x, 0.0)).rgb;
        vec3 nE = texture(u_colorTex, v_uv + vec2( texel.x, 0.0)).rgb;
        vec3 nN = texture(u_colorTex, v_uv + vec2(0.0,  texel.y)).rgb;
        vec3 nS = texture(u_colorTex, v_uv + vec2(0.0, -texel.y)).rgb;
        vec3 avg = (nW + nE + nN + nS) * 0.25;
        float contrast = length(sceneColor - avg);
        float blend = smoothstep(0.04, 0.22, contrast) * 0.65;
        sceneColor = mix(sceneColor, avg, blend);
    }

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
    // Camera-stable fog: subtract the mean optical depth accumulated over the
    // camera->target distance, so the subject keeps a constant brightness as
    // the camera zooms/orbits and fog only differentiates depth WITHIN the
    // scene. (The noise modulation is zero-mean, so density * refDist is the
    // right baseline; clamping at 0 just means nothing nearer than the focus
    // plane gets artificially brightened.) u_fogRefDist = 0 disables this.
    tau = max(tau - u_fogDensity * u_fogRefDist, 0.0);
    vec3  foggedColor = mix(u_fogColor, hitColor, exp(-tau));

    // Wisp glow — soft emissive blobs floating in the atmosphere
    vec3 wispGlow = vec3(0.0);
    const float GLOW_R = 5.0;  // glow radius in world units
    // Distance cutoff before the expensive part -- the Gaussian glow term
    // already decays hard within a few multiples of GLOW_R, so skip wisps
    // clearly too far to matter.
    const float GLOW_CUTOFF2 = (4.0 * GLOW_R) * (4.0 * GLOW_R);
    // Fog attenuation to each wisp reuses the scene march's optical depth,
    // scaled by how far along the ray the wisp sits, instead of re-running a
    // full 16-step marchFog() per wisp (which profiling showed multiplying
    // the already-dominant fog cost wherever several face lights were in
    // range). Exact only for homogeneous fog, but the noise modulation is
    // zero-mean over 16 samples and this term only gates a soft glow.
    float tauPerT = tau / max(min(tHit, 200.0), 1e-3);
    for (int wi = 0; wi < u_wispCount; wi++) {
        vec3  oc  = ro - u_wispPos[wi];
        float b   = dot(rd, oc);
        float tc  = max(0.0, -b);                  // closest t along ray, clamped to front
        vec3  cp  = (ro + rd * tc) - u_wispPos[wi];
        float d2  = dot(cp, cp);
        if (d2 > GLOW_CUTOFF2) continue;
        float glow = u_wispIntensity[wi] * exp(-d2 / (GLOW_R * GLOW_R));
        // Attenuate by fog between camera and wisp, and occlude behind geometry
        float tauWisp = tauPerT * min(tc, 200.0);
        glow *= exp(-tauWisp) * step(tc, tHit + 0.5);
        wispGlow += u_wispColor[wi] * glow;
    }

    // The glow blobs read as fog-scattered light -- with no fog there's
    // nothing to scatter, so gate them by density (normalized to the default
    // density 0.015 so the long-standing look is unchanged there) instead of
    // unconditionally adding a "light volume" even in perfectly clear air.
    float glowGate = clamp(u_fogDensity / 0.015, 0.0, 1.0) * u_wispGlowStrength;
    fragColor = vec4(foggedColor + wispGlow * 0.15 * glowGate, 1.0);
}
