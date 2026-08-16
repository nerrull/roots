#version 410 core

uniform samplerBuffer  u_nodes;
uniform isamplerBuffer u_segments;
uniform samplerBuffer  u_radii;
uniform vec3           u_eye;
uniform mat3           u_cam;
uniform float          u_fov;
uniform vec2           u_res;
uniform mat4           u_viewProj;

uniform vec3  u_baseColor;
uniform float u_ambient;
uniform float u_diffuse;
uniform vec3  u_specColor;
uniform float u_shininess;
uniform vec3  u_lightDir;

uniform int   u_shaderMode;   // 0 = Phong, 1 = PBR
uniform float u_metallic;
uniform float u_roughness;

uniform int   u_wispCount;
uniform vec3  u_wispPos[50];
uniform vec3  u_wispColor[50];
uniform float u_wispIntensity[50];

uniform float u_radiusScale;
uniform float u_radiusMin;
uniform float u_radiusMax;

uniform vec3  u_baseColor2;        // second color, blended in by noise
uniform float u_colorNoiseScale;   // spatial frequency of the mottling
uniform float u_colorNoiseStrength;

// Per-segment palette groups: instead of one RGB per capsule, each capsule
// carries a small int index into u_palette. u_paletteCount == 0 disables it
// and falls back to u_baseColor / u_baseColor2. The darker mottle color is
// derived from the group color, so the noise breakup still works per group.
uniform isamplerBuffer u_groups;
uniform vec3  u_palette[8];
uniform int   u_paletteCount;

// Per-segment primitive type + extra frame data, so the same instanced pass
// can draw more than capsules. type 0 = capsule (a,b,radius; analytic). type 1
// = blade: a flattened, tapering, cupped/curling surface for leaves & petals,
// sphere-marched. u_primFrame.xyz is the blade's half-width vector (a direction
// perpendicular to the a->b axis, length = half width in cm); .w is a curl
// amount (cross-section cupping + lengthwise droop).
uniform isamplerBuffer u_primType;
uniform samplerBuffer  u_primFrame;

// Per-segment aux for curved petal strips + colour gradients:
//   .xy = (s0, s1) the segment's fractional span of the WHOLE petal, so a
//         chain of blade segments shares one continuous outline (no pinch at
//         joints) -- a single-segment petal uses (0,1).
//   .zw = (grad0, grad1) the gradient parameter at the segment's base/tip; the
//         albedo lerps u_palette -> u_paletteTip by this, giving base->tip (and,
//         via per-petal bias, centre->rim) colour gradients.
uniform samplerBuffer u_primAux;
uniform vec3 u_paletteTip[8];

uniform sampler3D u_noiseTex;   // baked tiling fBm (shared with fog pass)

uniform samplerBuffer u_nodeDist;   // per-node arc-length from root base
uniform int   u_pulseEnabled;
uniform vec3  u_pulseColor;
uniform float u_pulseSpeed;
uniform float u_pulseSpacing;
uniform float u_pulseWidth;
uniform float u_pulseIntensity;
uniform float u_pulseTime;

flat in int  v_segIdx;
in vec2      v_ndc;

out vec4 fragColor;

const float PI = 3.14159265359;

// root-surface color mottling: one fetch into the baked tiling fBm texture
// (see RootRenderer::buildNoiseTexture) instead of the old 3-octave
// procedural value noise (24 hash evaluations per shaded fragment -- and
// with gl_FragDepth writes disabling early-z, every overlapping capsule's
// fragments pay full shading, so per-fragment ALU here is multiplied by
// overdraw). Same statistics, slightly different octave weights -- reads
// identically as organic mottling.
const float NOISE_TILE_PERIOD = 8.0;   // must match RootRenderer.cpp

float _rfbm(vec3 p) {
    return texture(u_noiseTex, p * (1.0 / NOISE_TILE_PERIOD)).r;
}

// Analytic ray-capsule intersection (Inigo Quilez) -- returns first positive
// t, or -1. Replaces a 20-step sphere-trace loop: the march cost 20 SDF
// evaluations per fragment per instance (multiplied by heavy overdraw in a
// dense root mass), converged only to within HIT_EPS, and could even miss
// thin capsules at glancing angles. One quadratic solve is exact and ~10x
// less ALU.
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
    float t = (-B - sqrt(h)) / max(A, 1e-8);
    float y = baoa + t * bard;
    if (y > 0.0 && y < baba && t > 0.0) return t;     // cylinder body
    vec3  oc = y <= 0.0 ? oa : ro - b;                 // pick nearest cap
    float B2 = dot(rd, oc), C2 = dot(oc, oc) - r * r;
    float h2 = B2 * B2 - C2;
    if (h2 < 0.0) return -1.0;
    t = -B2 - sqrt(h2);
    return t > 0.0 ? t : -1.0;
}

vec3 capsuleNormal(vec3 p, vec3 a, vec3 b) {
    vec3  ab = b - a;
    float t  = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return normalize(p - (a + t * ab));
}

// ---------------------------------------------------------------------------
// Pinnately lobed leaf outline (chrysanthemum).
//
// A width-versus-position profile -- hw(s), which is what the blade outline uses
// for petals -- can only ever carve bumps that stick out PERPENDICULAR to the
// midrib, symmetric about each station along it. A chrysanthemum leaf is nothing
// like that: its lobes angle FORWARD toward the tip, the sinuses between them
// are deep and narrow, and every lobe carries its own teeth. So the leaf gets a
// real 2D silhouette instead: a union of oriented, tapering, toothed lobes in
// the blade's own (along, across) plane, with the across coordinate folded to
// |v| so one set of lobes serves both margins.
// ---------------------------------------------------------------------------

// One lobe: a 2D capsule a->b whose radius lerps r0->r1, nibbled by `teeth`
// marginal serrations at `tf` cycles along its length.
float sdLobe2(vec2 p, vec2 a, vec2 b, float r0, float r1, float teeth, float tf) {
    vec2  pa = p - a, ba = b - a;
    float h  = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
    float r  = mix(r0, r1, h) * (1.0 - teeth * (0.5 + 0.5 * cos(6.2831853 * h * tf)));
    return length(pa - ba * h) - r;
}

// Distance from a point to the leaf's venation: the midrib, plus one vein
// running out into each lateral lobe. Used to raise a slight rib along them.
float sdSeg2(vec2 p, vec2 a, vec2 b) {
    vec2  pa = p - a, ba = b - a;
    float h  = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
    return length(pa - ba * h);
}

float leafVeinDist(vec2 p, float L, float W) {
    float d = sdSeg2(p, vec2(0.02 * L, 0.0), vec2(0.97 * L, 0.0));   // midrib
    for (int i = 0; i < 3; ++i) {
        float f   = float(i);
        vec2  a   = vec2((0.15 + 0.225 * f) * L, 0.0);
        float len = W * (0.46 + 0.30 * sin(3.14159265 * (f + 0.85) / 3.0));
        float ang = radians(58.0 - 7.0 * f);
        d = min(d, sdSeg2(p, a, a + vec2(cos(ang), sin(ang)) * len * 0.88));
    }
    return d;
}

// L = blade length, W = half width. Returns <0 inside the leaf.
float sdLeafOutline(vec2 p, float L, float W) {
    // Central body: a wedge widening from the cuneate base toward the tip. It has
    // to stay substantial -- the sinuses cut roughly half way in on a real leaf,
    // and a thin body turns the lobes into the arms of a starfish.
    float d = sdLobe2(p, vec2(0.02 * L, 0.0), vec2(0.46 * L, 0.0),
                      W * 0.09, W * 0.34, 0.0, 1.0);
    d = min(d, sdLobe2(p, vec2(0.46 * L, 0.0), vec2(0.78 * L, 0.0),
                       W * 0.34, W * 0.30, 0.0, 1.0));
    // Terminal lobe, tapering to a blunt point.
    d = min(d, sdLobe2(p, vec2(0.60 * L, 0.0), vec2(0.99 * L, 0.0),
                       W * 0.30, W * 0.07, 0.16, 3.0));
    // Three pairs of lateral lobes, each swept forward and separated from its
    // neighbours by a deep narrow sinus. Broad and blunt with their own marginal
    // teeth, smallest at the base -- not tapering spikes.
    for (int i = 0; i < 3; ++i) {
        float f    = float(i);
        vec2  a    = vec2((0.15 + 0.225 * f) * L, 0.0);
        float len  = W * (0.46 + 0.30 * sin(3.14159265 * (f + 0.85) / 3.0));
        float ang  = radians(58.0 - 7.0 * f);          // forward sweep
        vec2  b    = a + vec2(cos(ang), sin(ang)) * len;
        d = min(d, sdLobe2(p, a, b, W * 0.30, W * 0.13, 0.22, 2.5));
    }
    return d;
}

// ---------------------------------------------------------------------------
// Blade primitive: a leaf/petal surface. A thin slab in the (axis, width)
// plane, clipped to a tapering lanceolate outline, its mid-surface cupped
// across the width and curled/drooped along its length -- so it reads as an
// organic blade, not a cylinder. Signed distance (approx, Lipschitz enough to
// sphere-march). Frame: a=base, ax=unit axis, L=length, wdir=unit width axis,
// nrm=surface normal axis, hw0=half width, th=half thickness, curl=curvature.
// oExp/tipTaper shape the outline: low oExp + no taper = a broad, round-tipped
// spoon petal; higher oExp + taper = a pointed lanceolate leaf.
float sdBlade(vec3 pt, vec3 a, vec3 ax, float L, vec3 wdir, vec3 nrm,
              float hw0, float th, float curl, float oExp, float tipTaper,
              float s0, float s1, float latCup, float lobe) {
    vec3  d = pt - a;
    float u = dot(d, ax);            // along the blade
    float v = dot(d, wdir);          // across the blade
    float w = dot(d, nrm);           // off the surface
    float sLocal = clamp(u / L, 0.0, 1.0);
    float s = mix(s0, s1, sLocal);   // fraction of the WHOLE petal (strip-aware)
    if (lobe > 0.0) {
        // Leaf: the outline comes from the pinnate silhouette above, so the
        // half-width profile below does not apply. The cup is a shallow parabolic
        // channel rather than the rolled arc petals use -- a leaf never cups past
        // vertical, and a height field leaves the silhouette exact.
        float bend = curl * 0.6 * L * s * s;
        float cupD = latCup * 0.45 * (v * v) / max(hw0, 1e-4) * (1.0 - 0.5 * s);
        float wS2  = w - bend - cupD;
        float thk2 = th * (1.0 - 0.35 * s);
        // Venation: thicken the slab slightly along the midrib and lobe veins, so
        // they stand as ribs. Cheap, and it is what stops a leaf reading as a
        // paper cut-out.
        vec2  p2   = vec2(u, abs(v));
        float dv   = leafVeinDist(p2, L, hw0) / max(0.055 * hw0, 1e-4);
        thk2 *= 1.0 + 0.55 * exp(-dv * dv);
        float dSil = sdLeafOutline(p2, L, hw0);
        vec2  q2   = vec2(dSil, abs(wS2) - thk2);
        return min(max(q2.x, q2.y), 0.0) + length(max(q2, 0.0)) - 0.01;
    }
    float prof = pow(sin(3.14159265 * s), oExp) * (1.0 - tipTaper * s);
    float hw   = hw0 * max(prof, 0.015);
    float bend = curl * 0.6 * L * s * s;          // lengthwise curl/droop toward the tip
    float wS   = w - bend;
    float thk  = th * (1.0 - 0.5 * s);

    // Cross-section: instead of a parabolic height field (which can't curl past
    // vertical), ROLL the flat strip [-hw,hw] onto a circular arc whose half-
    // wrap angle is |latCup| (radians); sign = cup direction. latCup ~= PI rolls
    // it into a full tube; small values give a gentle round cup. Deeper near the
    // base (cupProfile). The cross-section SDF is the distance to that arc shell.
    float cupProfile = 1.0 - 0.6 * s;
    float theta = latCup * cupProfile;            // half-wrap angle
    float dCross;
    if (abs(theta) < 0.03) {                       // ~flat: plain slab
        vec2 q = vec2(abs(v) - hw, abs(wS) - thk);
        dCross = min(max(q.x, q.y), 0.0) + length(max(q, 0.0));
    } else {
        float sg = sign(theta), th_ = abs(theta);
        float R  = hw / th_;                       // arc radius (arc length hw -> angle th_)
        vec2  C  = vec2(0.0, sg * R);              // arc centre
        vec2  pc = vec2(v, wS) - C;
        float rho = length(pc);
        float phi = atan(pc.x, -sg * pc.y);        // 0 at strip centre, +/-th_ at edges
        if (abs(phi) <= th_) {
            dCross = abs(rho - R) - thk;            // on the arc: radial shell distance
        } else {                                   // past an edge: distance to endpoint
            float pe = clamp(phi, -th_, th_);
            vec2  E  = C + R * vec2(sin(pe), -sg * cos(pe));
            dCross = length(vec2(v, wS) - E) - thk;
        }
    }
    // Extrude the cross-section along the length, capped at [0,L].
    float dLen = max(-u, u - L);
    float outside = length(max(vec2(dLen, dCross), 0.0));
    float inside  = min(max(dLen, dCross), 0.0);
    return outside + inside - 0.01;
}

// ---------------------------------------------------------------------------
// PBR — Cook-Torrance BRDF (GGX + Smith + Schlick)
// ---------------------------------------------------------------------------
float D_GGX(float NdotH, float a2) {
    float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float G_SchlickGGX(float NdotX, float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}

vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Returns radiance contribution for a single light (no distance falloff — caller scales).
vec3 pbrBRDF(vec3 N, vec3 V, vec3 L, vec3 albedo, float metallic, float roughness) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL < 0.0001) return vec3(0.0);
    vec3  H     = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.0001);
    float VdotH = max(dot(V, H), 0.0);
    float a     = roughness * roughness;
    float a2    = a * a;
    float k     = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float D     = D_GGX(NdotH, a2);
    float G     = G_SchlickGGX(NdotV, k) * G_SchlickGGX(NdotL, k);
    vec3  F0    = mix(vec3(0.04), albedo, metallic);
    vec3  F     = F_Schlick(VdotH, F0);
    vec3  spec  = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
    vec3  kD    = (1.0 - F) * (1.0 - metallic);
    return (kD * albedo / PI + spec) * NdotL;
}

void main() {
    vec2 ndc = v_ndc;
    ndc.x *= u_res.x / u_res.y;
    vec3 rd = normalize(u_cam * vec3(ndc * tan(u_fov), 1.0));
    vec3 ro = u_eye;

    ivec2 seg = texelFetch(u_segments, v_segIdx).xy;
    vec3  a   = texelFetch(u_nodes, seg.x).xyz;
    vec3  b   = texelFetch(u_nodes, seg.y).xyz;
    float r   = texelFetch(u_radii, v_segIdx).r * u_radiusScale;
    if (u_radiusMin > 0.0) r = max(r, u_radiusMin);
    if (u_radiusMax > 0.0) r = min(r, u_radiusMax);

    const float MAX_DIST = 200.0;
    int  prim = texelFetch(u_primType, v_segIdx).r;
    vec4 aux  = texelFetch(u_primAux, v_segIdx);   // (s0, s1, grad0, grad1)

    vec3  p, n;
    float gradT = 0.0;      // 0 = base colour, 1 = tip colour
    if (prim == 0) {
        // --- Capsule: exact analytic intersection ---
        float hit = rayCapsule(ro, rd, a, b, r);
        if (hit < 0.0 || hit > MAX_DIST) discard;
        p = ro + rd * hit;
        n = capsuleNormal(p, a, b);
        vec3 ab = b - a;
        float al = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
        gradT = clamp(mix(aux.x, aux.y, al) + aux.w, 0.0, 1.0);   // fraction + bias
    } else {
        // --- Blade (leaf/petal): sphere-march the SDF ---
        vec3  ax = b - a;
        float L  = max(length(ax), 1e-4);
        ax /= L;
        vec4  fr   = texelFetch(u_primFrame, v_segIdx);
        vec3  W    = fr.xyz;
        float hw0  = max(length(W), 1e-4);
        vec3  nrm  = normalize(cross(ax, W));
        vec3  wdir = normalize(cross(nrm, ax));
        float th   = max(r, 0.02);
        float curl = fr.w;
        // prim 1 = pointed lanceolate blade; prim 2 = broad round-tipped spoon
        // petal; prim 3 = deeply-lobed, serrated leaf (lobe carved in sdBlade).
        float oExp = (prim == 2) ? 0.32 : (prim == 3 ? 0.70 : 0.6);
        float tipT = (prim == 2) ? 0.0  : (prim == 3 ? 0.30 : 0.28);
        float lobe = (prim == 3) ? 1.0  : 0.0;
        float s0 = aux.x, s1 = aux.y, latCup = aux.z;
        float t = 0.0;
        bool  hitB = false;
        // The bend term makes s (hence the field) depend on position along the
        // axis, so sdBlade is not a unit-Lipschitz distance -- its gradient can
        // exceed 1 where curl/cup are strong, and overshooting the thin surface
        // showed up as hard "clipped" cuts at high curl. Adaptive step: a
        // near-full step is safe away from the surface; damp to 0.6x only
        // within a few blade-thicknesses, where the overshoot actually bites.
        float near2 = th * 4.0 + 0.05;
        for (int i = 0; i < 72; ++i) {
            vec3  pt = ro + rd * t;
            float dd = sdBlade(pt, a, ax, L, wdir, nrm, hw0, th, curl, oExp, tipT, s0, s1, latCup, lobe);
            if (dd < 0.004) { hitB = true; break; }
            t += dd * (dd > near2 ? 0.95 : 0.6);
            if (t > MAX_DIST) break;
        }
        if (!hitB) discard;
        p = ro + rd * t;
        // Tetrahedron normal: 4 SDF taps instead of 6.
        const vec2 k = vec2(1.0, -1.0);
        float he = 0.01;
        n = normalize(
            k.xyy * sdBlade(p + k.xyy * he, a, ax, L, wdir, nrm, hw0, th, curl, oExp, tipT, s0, s1, latCup, lobe) +
            k.yyx * sdBlade(p + k.yyx * he, a, ax, L, wdir, nrm, hw0, th, curl, oExp, tipT, s0, s1, latCup, lobe) +
            k.yxy * sdBlade(p + k.yxy * he, a, ax, L, wdir, nrm, hw0, th, curl, oExp, tipT, s0, s1, latCup, lobe) +
            k.xxx * sdBlade(p + k.xxx * he, a, ax, L, wdir, nrm, hw0, th, curl, oExp, tipT, s0, s1, latCup, lobe));
        // Two-sided: face the viewer so back-lit petals still shade.
        if (dot(n, rd) > 0.0) n = -n;
        float al = clamp(dot(p - a, ax) / L, 0.0, 1.0);
        gradT = clamp(mix(s0, s1, al) + aux.w, 0.0, 1.0);   // petal fraction + bias
    }
    vec3 V = -rd;

    vec4 clipP = u_viewProj * vec4(p, 1.0);
    gl_FragDepth = (clipP.z / clipP.w) * 0.5 + 0.5;

    vec3 color;

    if (u_shaderMode == 2) {
        // --- Invert/XOR silhouette: solid white on hit. Combined with
        // glLogicOp(GL_XOR) and depth testing relaxed to GL_ALWAYS (set by
        // the caller), every capsule whose analytic surface covers a pixel
        // toggles it, instead of only the nearest one winning. ---
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    // Mottled albedo: blend base/second color by world-space noise instead of
    // a single flat color -- the single biggest lever against the uniform
    // "3D pipes screensaver" look, since it breaks up the material itself
    // rather than just the lighting on top of it.
    vec3 bc = u_baseColor, bc2 = u_baseColor2;
    if (u_paletteCount > 0) {
        int g = texelFetch(u_groups, v_segIdx).r;
        g  = clamp(g, 0, u_paletteCount - 1);
        bc = mix(u_palette[g], u_paletteTip[g], gradT);   // base->tip gradient
        bc2 = bc * 0.5;              // darker variant of the same hue for the mottle
    }
    float cn = _rfbm(p * u_colorNoiseScale);
    vec3 albedo = mix(bc, bc2, smoothstep(0.3, 0.7, cn) * u_colorNoiseStrength);

    if (u_shaderMode == 0) {
        // --- Phong ---
        float diff = max(dot(n, u_lightDir), 0.0);
        vec3  h    = normalize(u_lightDir + V);
        float spec = pow(max(dot(n, h), 0.0), u_shininess);
        color = albedo * (u_ambient + u_diffuse * diff) + u_specColor * spec;

        const float WISP_CUTOFF2 = 400.0;   // see PBR branch below for why
        for (int wi = 0; wi < u_wispCount; wi++) {
            vec3  lv    = u_wispPos[wi] - p;
            float dist2 = dot(lv, lv);
            if (dist2 > WISP_CUTOFF2) continue;
            vec3  ldir  = lv * inversesqrt(dist2);
            float att   = u_wispIntensity[wi] / (1.0 + dist2 * 0.008);
            float wd    = max(dot(n, ldir), 0.0);
            vec3  wh    = normalize(ldir + V);
            float ws    = pow(max(dot(n, wh), 0.0), u_shininess);
            color += att * u_wispColor[wi] * (albedo * u_diffuse * wd + u_specColor * ws);
        }
    } else {
        // --- PBR ---
        color = albedo * u_ambient;
        color += pbrBRDF(n, V, u_lightDir, albedo, u_metallic, u_roughness);

        // Cheap Blinn-Phong approximation for wisp/accent lights instead of
        // full Cook-Torrance -- with up to MAX_WISPS simultaneous point
        // lights now feeding in from every revealed face, a full pbrBRDF()
        // per wisp per pixel (several pow/sqrt each) was the actual cause of
        // a severe framerate regression once mask count grew past a handful.
        // Distance cutoff: the old falloff (1/(1+dist2*0.008)) never actually
        // reached zero within this scene's scale (~20-60 units across), so
        // every wisp was paying its full cost -- inversesqrt, two pow calls --
        // for every pixel regardless of how far away it was, and contributing
        // a barely-visible sliver of light to roots nowhere near it. A hard
        // radius cutoff is the cheap substitute for real spatial partitioning
        // here (no compute shaders on GL 4.1/macOS to build one properly):
        // skip the light entirely, before any of that math, past ~20 units.
        const float WISP_CUTOFF2 = 400.0;   // 20 units, matches typical mask spacing
        for (int wi = 0; wi < u_wispCount; wi++) {
            vec3  lv    = u_wispPos[wi] - p;
            float dist2 = dot(lv, lv);
            if (dist2 > WISP_CUTOFF2) continue;
            vec3  ldir  = lv * inversesqrt(dist2);
            float att   = u_wispIntensity[wi] / (1.0 + dist2 * 0.008);
            float ndotl = max(dot(n, ldir), 0.0);
            vec3  wh    = normalize(ldir + V);
            float wspec = pow(max(dot(n, wh), 0.0), 32.0) * (0.2 + 0.8 * u_metallic);
            color += att * u_wispColor[wi] * (albedo * ndotl * (1.0 - u_metallic) + vec3(wspec));
        }
    }

    // --- travelling light pulses ------------------------------------------
    // An additive emissive band scrolling along each root's arc-length from its
    // base. The per-node distance is interpolated to the exact hit point, then
    // folded into a repeating [0,spacing) phase offset by time*speed so the
    // crest crawls outward along the tube. A smooth triangular falloff over
    // u_pulseWidth keeps the band soft rather than a hard ring.
    if (u_pulseEnabled == 1 && u_pulseSpacing > 0.0) {
        vec3  ab   = b - a;
        float tpar = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
        float da   = texelFetch(u_nodeDist, seg.x).r;
        float db   = texelFetch(u_nodeDist, seg.y).r;
        float s    = mix(da, db, tpar);
        // signed distance to the nearest pulse crest along the root
        float phase = s - u_pulseTime * u_pulseSpeed;
        float d     = abs(mod(phase, u_pulseSpacing) - 0.5 * u_pulseSpacing);
        d           = 0.5 * u_pulseSpacing - d;                 // 0 at crest, grows away
        float band  = clamp(1.0 - d / max(u_pulseWidth, 1e-4), 0.0, 1.0);
        band        = band * band * (3.0 - 2.0 * band);          // smoothstep ease
        color += u_pulseColor * (band * u_pulseIntensity);
    }

    fragColor = vec4(color, 1.0);
}
