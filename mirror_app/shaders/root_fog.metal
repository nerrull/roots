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

// Extinction at a point, in "per world unit".
//
// Two things here used to make the fog read as a slab of moving wallpaper
// rather than as air:
//
//  * The height falloff was exp(-falloff * pos.y) -- an exponential about the
//    *world origin*, while the piece sits eight units up. So the fog's gradient
//    had no relationship to the thing being lit, and the visible band of it
//    landed wherever the scene happened to be. It now pivots about a reference
//    height that the scene supplies.
//
//  * The noise was a single octave translated by (t*0.5, 0, t*0.3) -- a rigid
//    translation with no Y component at all. A rigid translation IS a plane of
//    motion, and the eye locks onto one immediately; every feature in the fog
//    slid along the same horizontal line. Two octaves now advect along
//    deliberately non-parallel directions, so no single velocity explains what
//    the field is doing and the motion reads as churn instead of as a conveyor.
static float fogDensity(float3 pos, constant RootFogU& U,
                        texture3d<float> noiseTex, sampler smp) {
    const float h = (pos.y - U.fogHeightRef) / max(U.fogHeightScale, 1e-3);
    const float base = U.fogDensity * exp(-h);
    if (U.fogNoiseStrength < 0.001) return base;

    const float3 p0 = pos * U.fogNoiseScale + U.fogDrift0.xyz;
    const float3 p1 = pos * (U.fogNoiseScale * 2.17) + U.fogDrift1.xyz;
    const float n = fogFbm(p0, noiseTex, smp) * 0.65 + fogFbm(p1, noiseTex, smp) * 0.35;

    // Centred on the field's own mean, so strength changes the *contrast* of the
    // fog and not its average density. The old form multiplied by mix(1, n*2, s),
    // which coupled the two: asking for more texture also asked for more fog,
    // and the only way to get visible structure was to make the scene opaque.
    const float d = 1.0 + (n - 0.5) * 2.0 * U.fogNoiseStrength * U.fogNoiseContrast;
    return base * max(d, 0.0);
}

// `jitter` in [0,1) offsets the first sample within its step. Eight fixed steps
// put every pixel's samples on the same eight planes in depth, and the fog's
// noise then reads as eight nested shells rather than as a volume -- visible as
// banding wherever the density gradient is shallow, which here is everywhere.
// Offsetting per pixel converts that structured error into high-frequency noise,
// which the eye reads as grain and the final dither absorbs.
// Integrate extinction from fogStart out to the hit.
//
// Starting at fogStart, rather than integrating the whole ray and then
// subtracting a constant afterwards, is the substantive change. The old code
// did `tau -= fogDensity * fogRefDist`, which made the amount of fog removed
// scale with the density -- so the density slider moved the near clearing and
// the far haze at once, in opposite visual directions, and neither one could be
// set without disturbing the other. A start distance is also just what the
// quantity means: the air between the lens and the subject is clear.
// Henyey-Greenstein phase function: how much light scattering off the medium
// goes forward (g > 0) versus back (g < 0). This is the whole reason fog has a
// direction to it -- mist is bright where you look towards the sun and dull
// where you look away, and without a phase term fog is just a grey wash that
// gets denser with distance.
static float phaseHG(float cosTheta, float g) {
    const float g2 = g * g;
    const float d = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * 3.14159265 * max(d * sqrt(max(d, 1e-4)), 1e-4));
}

// Front-to-back single-scatter integration. Returns transmittance in .a and the
// light scattered into the ray in .rgb.
//
// The previous version returned optical depth alone and the caller did
// mix(fogColor, sceneColor, exp(-tau)) -- pure extinction towards a constant.
// That is why the fog was so hard to place: it could only ever *remove*
// contrast, it had no relationship to where the lights were, and against a
// background already painted fogColor it was invisible by construction, so the
// only way to see it at all was to crank it until the scene disappeared. With
// an in-scatter term the fog is lit -- it picks up the key's colour, it brightens
// towards the light and stays dark away from it, and the near-to-far gradient
// that reads as depth appears at densities that leave the subject perfectly
// legible.
//
// Starting at t0 rather than integrating the whole ray and subtracting a
// constant afterwards also matters: the old `tau -= density * refDist` scaled
// the near clearing with the density, so one slider moved the clearing and the
// far haze in opposite directions at once.
static float4 marchFog(float3 ro, float3 rd, float t0, float t1, float jitter,
                       constant RootFogU& U,
                       texture3d<float> noiseTex, sampler smp) {
    if (U.fogDensity <= 0.0) return float4(0.0, 0.0, 0.0, 1.0);
    t1 = min(t1, 400.0);
    if (t1 <= t0) return float4(0.0, 0.0, 0.0, 1.0);

    const int N = clamp(U.fogSteps, 4, 32);
    const float stepSize = (t1 - t0) / float(N);

    // Light travels along -lightDir (lightDir points from a surface towards the
    // light), so a view ray parallel to that is looking straight down the beam.
    const float cosT = dot(rd, -normalize(U.lightDir.xyz));
    // Normalised so that isotropic scattering is exactly 1: the value is then
    // "how many times more light comes from this direction than from an average
    // one", which is the number the anisotropy slider should be moving. At
    // g = 0.55 this peaks near 8 looking into the beam and drops to 0.2 looking
    // away -- a 40:1 swing, which is the whole effect.
    const float ph = phaseHG(cosT, clamp(U.fogAnisotropy, -0.95, 0.95)) * 12.566370;

    float3 scat = float3(0.0);
    float T = 1.0;
    for (int i = 0; i < 32; i++) {
        if (i >= N) break;
        const float3 p = ro + rd * (t0 + (float(i) + jitter) * stepSize);
        const float sigma = fogDensity(p, U, noiseTex, smp);
        const float dT = exp(-sigma * stepSize);
        // Energy removed from the beam over this step, of which fogScatter is
        // the fraction that scatters rather than being absorbed.
        // Ambient in-scatter is fogColor and is NOT scaled by fogScatter, so
        // "the colour the fog reaches at infinity" stays exactly fogColor and
        // the scatter slider only adds the directional component on top. Scaling
        // both would make the scatter control double as a brightness control for
        // the whole background, which is not what anyone reaches for it to do.
        scat += T * (1.0 - dT)
              * (U.fogColor.xyz + U.fogScatter * ph * U.keyColor.xyz);
        T *= dT;
        if (T < 0.002) break;
    }
    return float4(scat, T);
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

// ---------------------------------------------------------------------------
// Volumetric pass: the single-scatter integral alone, at a fraction of the
// scene resolution.
//
// This is separate from the composite for a measured reason. The march costs
// fogSteps x 2 trilinear fetches into a 128^3 volume *per pixel*, and the scene
// passes run supersampled -- at 2x that is 8.3M pixels, and the whole fog pass
// went from about 2 ms to 46. Most of that is not arithmetic but cache: the old
// noise scale of 0.05 had the entire scene sampling one small corner of the
// noise volume, so every fetch hit cache. At a scale where the noise actually
// has structure across the piece, the samples spread over the volume and the
// fetches start missing.
//
// The integral is smooth, though -- it is a low-frequency quantity with no
// silhouettes of its own -- so it does not need the supersampled grid, or even
// the output grid. Computing it at fogDownscale and letting the composite
// sample it bilinearly costs nothing visible and gives back the whole factor.
fragment float4 root_fogvol_fs(FogVOut in [[stage_in]],
                               constant RootFogU&  U        [[buffer(0)]],
                               depth2d<float>      depthTex [[texture(0)]],
                               texture3d<float>    noiseTex [[texture(1)]]) {
    constexpr sampler ptSmp(mag_filter::nearest, min_filter::nearest,
                            address::clamp_to_edge);
    constexpr sampler noiseSmp(mag_filter::linear, min_filter::linear,
                               address::repeat, mip_filter::linear);

    // U.res is the *volumetric* buffer's size here; the aspect is the same as
    // the scene's, so the reconstructed rays match the composite's exactly.
    const float2 uv = in.pos.xy / U.res;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
    ndc.x *= U.res.x / U.res.y;
    const float3 rd = normalize(U.cam * float3(ndc * tan(U.fov), 1.0));
    const float3 ro = U.eye.xyz;

    // Point-sampled: a linear fetch here would average depths across a
    // silhouette and put the fog at a distance nothing in the scene is at.
    const float d = depthTex.sample(ptSmp, uv);
    const float ndcZ = d * 2.0 - 1.0;
    const float ze = (2.0 * U.nearZ * U.farZ) /
                     (U.farZ + U.nearZ - ndcZ * (U.farZ - U.nearZ));
    const float hitT = ze / max(dot(rd, U.cam[2]), 1e-4);

    const float jitter = (U.fogDither > 0.0)
        ? mix(0.5, fract(52.9829189 * fract(dot(in.pos.xy,
                float2(0.06711056, 0.00583715)))), U.fogDither)
        : 0.5;

    return marchFog(ro, rd, U.fogStart, hitT, jitter, U, noiseTex, noiseSmp);
}

fragment float4 root_fog_fs(FogVOut in [[stage_in]],
                            constant RootFogU&    U        [[buffer(0)]],
                            device const RootWisp* wisps    [[buffer(1)]],
                            texture2d<float>      colorTex [[texture(0)]],
                            depth2d<float>        depthTex [[texture(1)]],
                            texture3d<float>      noiseTex [[texture(2)]],
                            texture2d<float>      aoTex    [[texture(3)]],
                            texture2d<float>      volTex   [[texture(4)]]) {
    constexpr sampler linSmp(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    constexpr sampler noiseSmp(mag_filter::linear, min_filter::linear,
                               address::repeat, mip_filter::linear);

    float2 uv = in.pos.xy / U.res;    // row 0 at top (Metal), matches geometry write

    float4 sceneSample = colorTex.sample(linSmp, uv);
    float3 sceneColor  = sceneSample.rgb;
    float  depthSample = depthTex.sample(linSmp, uv);

    // Ambient occlusion, applied to the environment share the geometry pass
    // recorded in alpha (see root_geom.metal) rather than to the whole pixel.
    // Multiplying the finished colour would darken the key light and the
    // travelling pulses too, and a pulse that dims when it passes behind
    // another root is the sort of wrongness that is hard to name but easy to
    // see. The AO buffer is half-resolution and sampled linearly; its own
    // bilateral blur has already stopped it from crossing depth edges, so the
    // upsample cannot drag occlusion onto a surface that is not occluded.
    if (U.aoEnabled == 1) {
        const float ao = aoTex.sample(linSmp, uv).r;
        sceneColor *= mix(1.0, ao, sceneSample.a);
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

    // The integral comes from the low-resolution volumetric pass, sampled
    // bilinearly. The overlays below (axes, grid) are debug geometry the
    // volumetric pass does not know about, so they take the fog of whatever is
    // behind them -- which is correct enough for a debug overlay.
    const float4 fogRes = volTex.sample(linSmp, uv);
    const float T = fogRes.a;
    // What is left of the scene through the medium, plus what the medium
    // scattered into the ray on the way. No mix() towards a constant: the
    // "colour of the fog at infinity" is now something the integration arrives
    // at rather than something asserted.
    float3 foggedColor = hitColor * T + fogRes.rgb;
    const float tau = -log(max(T, 1e-4));   // the wisp attenuation still wants it

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
