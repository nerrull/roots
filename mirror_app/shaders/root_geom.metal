// root_geom.metal — Metal port of sdf_viewer's shader.vert + shader.frag.
//
// Instanced pass: 6 vertices/segment build a screen-space bounding quad from the
// projected capsule/blade endpoints; the fragment analytically intersects the
// capsule (or sphere-marches the blade SDF), shades Phong/PBR, adds wisps + a
// travelling pulse, and writes real scene depth so the fog pass and any mid
// geometry composite against it. root_shared.h (host-prepended) supplies RootGeomU
// / RootWisp and the tiling-noise period. NOTE: Invert (XOR) mode has no Metal
// equivalent (no fragment logic ops); it falls back to solid white — see
// metal_root_renderer for the documented divergence.
#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265359;

// GLSL mod(): always the sign of y (fmod takes the sign of x). The pulse phase
// goes negative, so this must match GL exactly.
static float glmod(float x, float y) { return x - y * floor(x / y); }

// per-node arc-length, groups, prim type/frame/aux, nodes/segs/radii are bound as
// plain device buffers (Metal has no texture-buffer objects — a straight buffer
// read replaces every texelFetch).

struct GeomVOut {
    float4 pos [[position]];
    float2 ndc;
    int    segIdx [[flat]];
};

struct GeomFOut {
    float4 color [[color(0)]];
    float  depth [[depth(any)]];
};

// ---------------------------------------------------------------------------
// Vertex: conservative screen-space bounding quad for segment gl_InstanceID.
// ---------------------------------------------------------------------------
vertex GeomVOut root_geom_vs(uint vid       [[vertex_id]],
                             uint iid       [[instance_id]],
                             device const packed_float3* nodes   [[buffer(0)]],
                             device const int2*          segs    [[buffer(1)]],
                             device const float*         radii   [[buffer(2)]],
                             device const int*           primT   [[buffer(5)]],
                             device const float4*        primF   [[buffer(6)]],
                             constant RootGeomU&         U       [[buffer(8)]]) {
    int i = int(iid);
    GeomVOut o;
    o.segIdx = i;

    int2  seg = segs[i];
    float3 a  = float3(nodes[seg.x]);
    float3 b  = float3(nodes[seg.y]);
    float  r  = radii[i] * U.radiusScale;
    if (U.radiusMin > 0.0) r = max(r, U.radiusMin);
    if (U.radiusMax > 0.0) r = min(r, U.radiusMax);

    if (primT[i] != 0) {
        float4 fr   = primF[i];
        float  hw   = length(fr.xyz);
        float  curl = abs(fr.w);
        float  L    = distance(a, b);
        float  reach = curl * 0.6 * L + 2.0 * hw;
        r = max(r, hw + reach) * 1.7;
    }

    float4 ca = U.viewProj * float4(a, 1.0);
    float4 cb = U.viewProj * float4(b, 1.0);

    if (ca.w <= 0.0 && cb.w <= 0.0) {
        o.pos = float4(0.0, 0.0, 0.0, -1.0);
        o.ndc = float2(0.0);
        return o;
    }

    float2 ndcA = ca.xy / max(ca.w, 0.001);
    float2 ndcB = cb.xy / max(cb.w, 0.001);

    float aspect = U.res.x / U.res.y;
    float f      = 1.0 / tan(U.fov);
    float minW   = max(min(ca.w, cb.w), 0.001);
    float sry    = r * f / minW;
    float srx    = sry / aspect;

    // Sub-pixel cull: if the capsule's projected radius is smaller than cullPx
    // pixels, it can't contribute a visible fragment — degenerate the quad so no
    // fragments (hence no ray-capsule intersections) are launched for it.
    if (U.cullPx > 0.0 && sry * U.res.y * 0.5 < U.cullPx) {
        o.pos = float4(0.0, 0.0, 0.0, -1.0);
        o.ndc = float2(0.0);
        return o;
    }

    float2 lo = min(ndcA, ndcB) - float2(srx, sry);
    float2 hi = max(ndcA, ndcB) + float2(srx, sry);

    float2 corners[6];
    corners[0] = float2(lo.x, lo.y);
    corners[1] = float2(hi.x, lo.y);
    corners[2] = float2(lo.x, hi.y);
    corners[3] = float2(hi.x, lo.y);
    corners[4] = float2(hi.x, hi.y);
    corners[5] = float2(lo.x, hi.y);

    float2 p = corners[vid];
    o.ndc = p;
    o.pos = float4(p, 0.0, 1.0);
    return o;
}

// ---------------------------------------------------------------------------
// Baked tiling fBm — one trilinear fetch (matches RootRenderer::buildNoiseTexture).
// ---------------------------------------------------------------------------
static float rfbm(float3 p, texture3d<float> noiseTex, sampler smp) {
    return noiseTex.sample(smp, p * (1.0 / ROOT_NOISE_TILE_PERIOD)).r;
}

// Analytic ray-capsule intersection (Inigo Quilez). First positive t, or -1.
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
    float t = (-B - sqrt(h)) / max(A, 1e-8);
    float y = baoa + t * bard;
    if (y > 0.0 && y < baba && t > 0.0) return t;
    float3 oc = y <= 0.0 ? oa : ro - b;
    float B2 = dot(rd, oc), C2 = dot(oc, oc) - r * r;
    float h2 = B2 * B2 - C2;
    if (h2 < 0.0) return -1.0;
    t = -B2 - sqrt(h2);
    return t > 0.0 ? t : -1.0;
}

static float3 capsuleNormal(float3 p, float3 a, float3 b) {
    float3 ab = b - a;
    float t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return normalize(p - (a + t * ab));
}

// Blade primitive SDF (leaf/petal). Verbatim port of sdBlade.
static float sdBlade(float3 pt, float3 a, float3 ax, float L, float3 wdir, float3 nrm,
                     float hw0, float th, float curl, float oExp, float tipTaper,
                     float s0, float s1, float latCup, float lobe) {
    float3 d = pt - a;
    float u = dot(d, ax);
    float v = dot(d, wdir);
    float w = dot(d, nrm);
    float sLocal = clamp(u / L, 0.0, 1.0);
    float s = mix(s0, s1, sLocal);
    float prof = pow(sin(3.14159265 * s), oExp) * (1.0 - tipTaper * s);
    if (lobe > 0.0) {
        float lobes = 0.5 + 0.5 * cos(6.2831853 * (s * 3.5 - 0.15));
        float teeth = 0.5 + 0.5 * cos(6.2831853 * s * 11.0);
        prof *= 1.0 - lobe * (0.5 * lobes + 0.12 * teeth);
    }
    float hw   = hw0 * max(prof, 0.015);
    float bend = curl * 0.6 * L * s * s;
    float wS   = w - bend;
    float thk  = th * (1.0 - 0.5 * s);

    float cupProfile = 1.0 - 0.6 * s;
    float theta = latCup * cupProfile;
    float dCross;
    if (abs(theta) < 0.03) {
        float2 q = float2(abs(v) - hw, abs(wS) - thk);
        dCross = min(max(q.x, q.y), 0.0) + length(max(q, 0.0));
    } else {
        float sg = sign(theta), th_ = abs(theta);
        float R  = hw / th_;
        float2 C = float2(0.0, sg * R);
        float2 pc = float2(v, wS) - C;
        float rho = length(pc);
        float phi = atan2(pc.x, -sg * pc.y);
        if (abs(phi) <= th_) {
            dCross = abs(rho - R) - thk;
        } else {
            float pe = clamp(phi, -th_, th_);
            float2 E = C + R * float2(sin(pe), -sg * cos(pe));
            dCross = length(float2(v, wS) - E) - thk;
        }
    }
    float dLen = max(-u, u - L);
    float outside = length(max(float2(dLen, dCross), 0.0));
    float inside  = min(max(dLen, dCross), 0.0);
    return outside + inside - 0.01;
}

// PBR — Cook-Torrance
static float D_GGX(float NdotH, float a2) {
    float d = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
static float G_SchlickGGX(float NdotX, float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}
static float3 F_Schlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
// --- surface detail ---------------------------------------------------------
// A root is not a tube. It is a bundle of fibres running lengthwise, and the
// single thing that most makes these capsules read as extruded plastic is that
// their surface is perfectly, uniformly smooth: a swept cylinder with a constant
// albedo gives one broad even highlight down its whole length, which is exactly
// what moulded plastic looks like and exactly what nothing organic looks like.
//
// The fix is anisotropic. Sampling isotropic noise would give the roots a
// gravelly, sandpapered look -- detail with no direction, which reads as dirt on
// the surface rather than as structure of it. Compressing the sample
// coordinate's axial component makes the field vary slowly along the root and
// quickly around it, so the features come out as fibres running the length of
// the root, which is what the eye is looking for.
static float detailField(float3 x, float3 ax, float scale, float stretch,
                         texture3d<float> tex, sampler smp) {
    const float axial = dot(x, ax);
    const float3 perp = x - ax * axial;
    const float3 q = perp * scale + ax * (axial * scale / max(stretch, 1e-3));
    return tex.sample(smp, q * (1.0 / ROOT_NOISE_TILE_PERIOD)).r;
}

// Gradient of that field, projected into the surface, applied to the normal.
// Returns the field value too, since the shading also wants it for the
// roughness break-up and paying for a fifth fetch to get it again is silly.
static float3 detailNormal(float3 p, float3 n, float3 ax, float scale, float stretch,
                           float strength, texture3d<float> tex, sampler smp,
                           thread float& fieldOut) {
    const float f0 = detailField(p, ax, scale, stretch, tex, smp);
    fieldOut = f0;
    if (strength <= 0.0) return n;
    // One noise cell across, so the difference is a real slope and not the
    // aliased leftovers of one sampled far finer than the lattice.
    const float e = 1.0 / max(scale, 1e-3);
    const float3 g = float3(
        detailField(p + float3(e, 0, 0), ax, scale, stretch, tex, smp) - f0,
        detailField(p + float3(0, e, 0), ax, scale, stretch, tex, smp) - f0,
        detailField(p + float3(0, 0, e), ax, scale, stretch, tex, smp) - f0) / e;
    // Only the in-surface component tilts the normal; the along-normal part
    // would just rescale it.
    const float3 gt = g - n * dot(g, n);
    return normalize(n - gt * strength);
}

// Per-segment tint jitter. Without it every root in a group carries exactly the
// same albedo, and a hundred identical objects is a strong cue that they were
// generated rather than grown -- the world-space colour noise cannot supply this
// because it is continuous, so two roots crossing at a point get the same value.
static float3 segmentTint(int si, float amount) {
    uint h = uint(si) * 2654435761u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    const float r0 = float(h & 0xFFFFu) / 65535.0;
    const float r1 = float((h >> 16) & 0xFFFFu) / 65535.0;
    // Brightness carries most of it; a small warm/cool lean stops the variation
    // from reading as a single dimmer being turned up and down.
    return float3(1.0 + (r0 - 0.5) * 2.0 * amount,
                  1.0 + (r0 - 0.5) * 2.0 * amount * 0.85,
                  1.0 + (r0 - 0.5) * 2.0 * amount * 0.65)
         * (1.0 + (r1 - 0.5) * amount * 0.35);
}

// --- environment ------------------------------------------------------------
// A two-colour hemisphere in place of an environment probe. This is the single
// cheapest thing that stops a surface reading as plastic: a constant ambient
// term lights every facing equally, so the unlit side of a tube is a flat patch
// of colour and the eye gets no shape information from it at all. Keying the
// ambient on n.y gives the underside a different colour from the top, which is
// what every real outdoor surface does.
static float3 hemiAmbient(float3 n, float3 sky, float3 ground) {
    return mix(ground, sky, n.y * 0.5 + 0.5);
}

// Roughness-blurred sky reflection. Not a real prefiltered probe -- there is no
// probe -- but the reflection vector still picks up the sky/ground split, and
// with a Fresnel in front of it that is enough to read as a glancing sheen
// rather than as a single hard highlight dot.
static float3 envSpecular(float3 n, float3 V, float roughness, float3 sky, float3 ground) {
    const float3 R = reflect(-V, n);
    // Pull the reflection towards the normal as roughness rises: a rough surface
    // gathers from a wide cone, and the cone's mean sits between R and N.
    const float3 Rr = normalize(mix(R, n, roughness * roughness));
    return hemiAmbient(Rr, sky, ground);
}

// Wrapped diffuse plus back-lit transmission. Roots and leaves are thin, damp
// and translucent; a Lambert term terminates at NdotL = 0 with a hard line,
// which is exactly the edge the scene is trying to lose. Wrapping pushes the
// terminator around the tube, and the transmission lobe puts light through the
// far side when the camera is looking towards the source.
static float3 organicDiffuse(float3 N, float3 V, float3 L, float3 albedo,
                             float wrap, float trans, float power, float3 tint) {
    const float w = max(wrap, 0.0);
    const float wrapped = max((dot(N, L) + w) / ((1.0 + w) * (1.0 + w)), 0.0);
    float3 c = albedo * wrapped;
    if (trans > 0.0) {
        // Light leaving the surface roughly opposite the incoming direction,
        // seen when V looks back along it.
        const float back = pow(saturate(dot(V, -L)), max(power, 1.0));
        c += albedo * tint * (back * trans);
    }
    return c;
}

static float3 pbrBRDF(float3 N, float3 V, float3 L, float3 albedo, float metallic, float roughness) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL < 0.0001) return float3(0.0);
    float3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.0001);
    float VdotH = max(dot(V, H), 0.0);
    float a = roughness * roughness;
    float a2 = a * a;
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float D = D_GGX(NdotH, a2);
    float G = G_SchlickGGX(NdotV, k) * G_SchlickGGX(NdotL, k);
    float3 F0 = mix(float3(0.04), albedo, metallic);
    float3 F  = F_Schlick(VdotH, F0);
    float3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
    float3 kD = (1.0 - F) * (1.0 - metallic);
    return (kD * albedo / PI + spec) * NdotL;
}

fragment GeomFOut root_geom_fs(GeomVOut in [[stage_in]],
                               device const packed_float3* nodes  [[buffer(0)]],
                               device const int2*          segs   [[buffer(1)]],
                               device const float*         radii  [[buffer(2)]],
                               device const float*         nodeD  [[buffer(3)]],
                               device const int*           groups [[buffer(4)]],
                               device const int*           primT  [[buffer(5)]],
                               device const float4*        primF  [[buffer(6)]],
                               device const float4*        primA  [[buffer(7)]],
                               constant RootGeomU&         U      [[buffer(8)]],
                               device const RootWisp*      wisps  [[buffer(9)]],
                               texture3d<float>            noiseTex [[texture(0)]]) {
    constexpr sampler noiseSmp(mag_filter::linear, min_filter::linear,
                               address::repeat, mip_filter::linear);

    float2 ndc = in.ndc;
    ndc.x *= U.res.x / U.res.y;
    float3 rd = normalize(U.cam * float3(ndc * tan(U.fov), 1.0));
    float3 ro = U.eye.xyz;

    int    si  = in.segIdx;
    int2   seg = segs[si];
    float3 a   = float3(nodes[seg.x]);
    float3 b   = float3(nodes[seg.y]);
    float  r   = radii[si] * U.radiusScale;
    if (U.radiusMin > 0.0) r = max(r, U.radiusMin);
    if (U.radiusMax > 0.0) r = min(r, U.radiusMax);

    const float MAX_DIST = 200.0;
    int   prim = primT[si];
    float4 aux = primA[si];

    float3 p, n;
    float  gradT = 0.0;
    float3 axis  = normalize(b - a);   // the direction the surface detail runs
    if (prim == 0) {
        float hit = rayCapsule(ro, rd, a, b, r);
        if (hit < 0.0 || hit > MAX_DIST) discard_fragment();
        p = ro + rd * hit;
        n = capsuleNormal(p, a, b);
        float3 ab = b - a;
        float al = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
        gradT = clamp(mix(aux.x, aux.y, al) + aux.w, 0.0, 1.0);
    } else {
        float3 ax = b - a;
        float L = max(length(ax), 1e-4);
        ax /= L;
        float4 fr  = primF[si];
        float3 W   = fr.xyz;
        float  hw0 = max(length(W), 1e-4);
        float3 nrm = normalize(cross(ax, W));
        float3 wdir = normalize(cross(nrm, ax));
        float  th  = max(r, 0.02);
        float  curl = fr.w;
        float  oExp = (prim == 2) ? 0.32 : (prim == 3 ? 0.70 : 0.6);
        float  tipT = (prim == 2) ? 0.0  : (prim == 3 ? 0.30 : 0.28);
        float  lobe = (prim == 3) ? 1.0  : 0.0;
        float  s0 = aux.x, s1 = aux.y, latCup = aux.z;
        float  t = 0.0;
        bool   hitB = false;
        float  near2 = th * 4.0 + 0.05;
        for (int i = 0; i < 72; ++i) {
            float3 pt = ro + rd * t;
            float dd = sdBlade(pt, a, ax, L, wdir, nrm, hw0, th, curl, oExp, tipT, s0, s1, latCup, lobe);
            if (dd < 0.004) { hitB = true; break; }
            t += dd * (dd > near2 ? 0.95 : 0.6);
            if (t > MAX_DIST) break;
        }
        if (!hitB) discard_fragment();
        p = ro + rd * t;
        const float2 k = float2(1.0, -1.0);
        float he = 0.01;
        n = normalize(
            k.xyy * sdBlade(p + k.xyy * he, a, ax, L, wdir, nrm, hw0, th, curl, oExp, tipT, s0, s1, latCup, lobe) +
            k.yyx * sdBlade(p + k.yyx * he, a, ax, L, wdir, nrm, hw0, th, curl, oExp, tipT, s0, s1, latCup, lobe) +
            k.yxy * sdBlade(p + k.yxy * he, a, ax, L, wdir, nrm, hw0, th, curl, oExp, tipT, s0, s1, latCup, lobe) +
            k.xxx * sdBlade(p + k.xxx * he, a, ax, L, wdir, nrm, hw0, th, curl, oExp, tipT, s0, s1, latCup, lobe));
        if (dot(n, rd) > 0.0) n = -n;
        float al = clamp(dot(p - a, ax) / L, 0.0, 1.0);
        gradT = clamp(mix(s0, s1, al) + aux.w, 0.0, 1.0);
    }
    float3 V = -rd;

    float4 clipP = U.viewProj * float4(p, 1.0);
    float depth  = (clipP.z / clipP.w) * 0.5 + 0.5;

    GeomFOut fo;
    fo.depth = depth;

    if (U.shaderMode == 2) {
        // Invert/XOR silhouette has no Metal fragment-logic-op equivalent; the
        // best faithful stand-in is a flat white silhouette (nearest-depth wins).
        // Alpha 0: the silhouette carries no environment term, so there is no
        // ambient share for the AO to attenuate.
        fo.color = float4(1.0, 1.0, 1.0, 0.0);
        return fo;
    }

    float3 bc = U.baseColor.xyz, bc2 = U.baseColor2.xyz;
    if (U.paletteCount > 0) {
        int g = groups[si];
        g = clamp(g, 0, U.paletteCount - 1);
        bc  = mix(U.palette[g].xyz, U.paletteTip[g].xyz, gradT);
        bc2 = bc * 0.5;
    }
    float cn = rfbm(p * U.colorNoiseScale, noiseTex, noiseSmp);
    float3 albedo = mix(bc, bc2, smoothstep(0.3, 0.7, cn) * U.colorNoiseStrength);
    if (U.detailTint > 0.0) albedo *= segmentTint(si, U.detailTint);

    // Fibre detail: tilt the normal and record the field for the break-up below.
    float detail = 0.5;
    if (U.detailStrength > 0.0 || U.detailRough > 0.0)
        n = detailNormal(p, n, axis, U.detailScale, U.detailStretch,
                         U.detailStrength, noiseTex, noiseSmp, detail);

    // Fibres also catch the light unevenly along their length. Modulating the
    // specular response by the same field is what turns one continuous highlight
    // running the length of a root into a broken, strand-wise one -- and an
    // unbroken specular band down a cylinder is most of what "plastic" means.
    const float detailSpec = 1.0 + (detail - 0.5) * 2.0 * U.detailRough;
    // Darken where the fibres dip. A crevice is shaded by its own walls, which
    // nothing else in this shader accounts for at this scale.
    albedo *= 1.0 - U.detailRough * 0.35 * (1.0 - detail);

    // The indirect (environment) contribution, kept separate from the direct
    // light so the fog pass can attenuate it by the screen-space AO without
    // also darkening the key light -- occlusion occludes the sky, not the sun.
    const float3 sky = U.skyColor.xyz, ground = U.groundColor.xyz;
    float3 indirect = albedo * U.ambient;
    if (U.hemiStrength > 0.0)
        indirect += albedo * hemiAmbient(n, sky, ground) * U.hemiStrength;
    if (U.envSpec > 0.0) {
        const float NdotV = max(dot(n, V), 0.0);
        const float rough = (U.shaderMode == 1) ? U.roughness : 0.5;
        const float3 F0 = mix(float3(0.04), albedo, (U.shaderMode == 1) ? U.metallic : 0.0);
        indirect += envSpecular(n, V, rough, sky, ground)
                  * F_Schlick(NdotV, F0) * U.envSpec * (1.0 - rough * 0.8);
    }
    if (U.rimStrength > 0.0) {
        const float rim = pow(1.0 - max(dot(n, V), 0.0), 3.0);
        indirect += sky * (rim * U.rimStrength);
    }

    float3 color;
    if (U.shaderMode == 0) {
        float3 h = normalize(U.lightDir.xyz + V);
        float spec = pow(max(dot(n, h), 0.0), U.shininess);
        color = indirect
              + U.keyColor.xyz * U.diffuse
                * organicDiffuse(n, V, U.lightDir.xyz, albedo,
                                 U.sssWrap, U.sssTrans, U.sssPower, U.sssTint.xyz)
              + U.keyColor.xyz * U.specColor.xyz * (spec * detailSpec);

        const float WISP_CUTOFF2 = 400.0;
        for (int wi = 0; wi < U.wispCount; wi++) {
            float3 lv = wisps[wi].pos.xyz - p;
            float dist2 = dot(lv, lv);
            if (dist2 > WISP_CUTOFF2) continue;
            float3 ldir = lv * rsqrt(dist2);
            float att = wisps[wi].pos.w / (1.0 + dist2 * 0.008);
            float wd = max(dot(n, ldir), 0.0);
            float3 wh = normalize(ldir + V);
            float ws = pow(max(dot(n, wh), 0.0), U.shininess);
            color += att * wisps[wi].color.xyz * (albedo * U.diffuse * wd + U.specColor.xyz * ws);
        }
    } else {
        color = indirect;
        // The fibre field varies the finish as well as the form: a raised fibre
        // is polished by whatever the root grew against, a groove is not.
        const float rough = clamp(U.roughness * (2.0 - detailSpec), 0.04, 1.0);
        color += U.keyColor.xyz
               * pbrBRDF(n, V, U.lightDir.xyz, albedo, U.metallic, rough);
        // The BRDF's own NdotL already terminated at the horizon; the wrap and
        // transmission are added on top as the part of the response a
        // single-scatter microfacet model does not have.
        if (U.sssWrap > 0.0 || U.sssTrans > 0.0) {
            const float3 lam = albedo * max(dot(n, U.lightDir.xyz), 0.0);
            const float3 org = organicDiffuse(n, V, U.lightDir.xyz, albedo,
                                              U.sssWrap, U.sssTrans, U.sssPower,
                                              U.sssTint.xyz);
            color += U.keyColor.xyz * max(org - lam, 0.0) * (1.0 - U.metallic) * (1.0 / PI);
        }

        const float WISP_CUTOFF2 = 400.0;
        for (int wi = 0; wi < U.wispCount; wi++) {
            float3 lv = wisps[wi].pos.xyz - p;
            float dist2 = dot(lv, lv);
            if (dist2 > WISP_CUTOFF2) continue;
            float3 ldir = lv * rsqrt(dist2);
            float att = wisps[wi].pos.w / (1.0 + dist2 * 0.008);
            float ndotl = max(dot(n, ldir), 0.0);
            float3 wh = normalize(ldir + V);
            float wspec = pow(max(dot(n, wh), 0.0), 32.0) * (0.2 + 0.8 * U.metallic);
            color += att * wisps[wi].color.xyz * (albedo * ndotl * (1.0 - U.metallic) + float3(wspec));
        }
    }

    // travelling light pulses
    if (U.pulseEnabled == 1 && U.pulseSpacing > 0.0) {
        float3 ab = b - a;
        float tpar = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
        float da = nodeD[seg.x];
        float db = nodeD[seg.y];
        float s = mix(da, db, tpar);
        float phase = s - U.pulseTime * U.pulseSpeed;
        float d = abs(glmod(phase, U.pulseSpacing) - 0.5 * U.pulseSpacing);
        d = 0.5 * U.pulseSpacing - d;
        float band = clamp(1.0 - d / max(U.pulseWidth, 1e-4), 0.0, 1.0);
        band = band * band * (3.0 - 2.0 * band);
        color += U.pulseColor.xyz * (band * U.pulseIntensity);
    }

    // Alpha carries the fraction of this pixel's radiance that came from the
    // environment rather than from a light. The fog pass multiplies exactly
    // that share by the screen-space AO, so occlusion darkens the sky term and
    // leaves the key light and the pulses alone -- which is the difference
    // between AO that grounds the geometry and AO that just looks like dirt.
    // One spare channel in an attachment we already pay for, instead of a
    // second render target on the pass that can least afford one.
    const float lumT = dot(color,    float3(0.2126, 0.7152, 0.0722));
    const float lumI = dot(indirect, float3(0.2126, 0.7152, 0.0722));
    fo.color = float4(color, saturate(lumI / max(lumT, 1e-5)));
    return fo;
}
