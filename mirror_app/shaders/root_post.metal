// root_post.metal — the final composite for the root scene.
//
// Everything between "the fog pass produced a linear HDR image" and "this is a
// display-referred image ready for the drawable" happens here, in one pass, in
// this order: chromatic aberration + supersample resolve, depth-of-field,
// bloom, halation, anamorphic streak, exposure, vignette, filmic tonemap,
// print grade, sRGB encode, grain, dither.
//
// The ordering is not arbitrary. Resolve and DoF are scene-referred operations
// on linear radiance and have to precede the curve; bloom is added in linear
// because that is where light actually adds; vignette multiplies exposure so it
// belongs before the curve too. Grain, by contrast, is a *display* artefact and
// is applied after the curve, or it would be swallowed in the shadows and
// exaggerated in the highlights. The dither is last of all, because its whole
// job is to break up the quantisation the 8-bit drawable is about to do.
//
// NOTE: this pass is what makes the root chain output display-referred. The
// headless capture paths must not re-apply a gamma on top of it -- see
// MetalRootRenderer::outputIsEncoded().
#include <metal_stdlib>
using namespace metal;

struct PostVOut { float4 pos [[position]]; };

vertex PostVOut root_post_vs(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    PostVOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// Interleaved gradient noise (Jorge Jimenez). One dot product and a fract, and
// it decorrelates far better across neighbouring pixels than a hash does --
// which is the whole point for both the grain and the dither.
static float ign(float2 p) {
    return fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715))));
}

// ACES filmic, Stephen Hill's RRT+ODT fit. Chosen over a Reinhard curve because
// the shoulder desaturates towards white the way film does; Reinhard holds
// saturation into the clip and the bright wisps come out as flat colour blobs.
static constant float3x3 kACESIn = float3x3(
    float3(0.59719, 0.07600, 0.02840),
    float3(0.35458, 0.90834, 0.13383),
    float3(0.04823, 0.01566, 0.83777));
static constant float3x3 kACESOut = float3x3(
    float3( 1.60475, -0.10208, -0.00327),
    float3(-0.53108,  1.10813, -0.07276),
    float3(-0.07367, -0.00605,  1.07602));

static float3 acesFitted(float3 c) {
    c = kACESIn * c;
    float3 a = c * (c + 0.0245786) - 0.000090537;
    float3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
    c = kACESOut * (a / b);
    return clamp(c, 0.0, 1.0);
}

// Piecewise sRGB, not pow(1/2.2). The two diverge most in the darkest stop, and
// the fog floor of this scene lives exactly there.
static float3 srgbEncode(float3 c) {
    c = clamp(c, 0.0, 1.0);
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

// Depth-of-field gather kernel: a centre tap and two full rings of nine, spun
// per pixel. The tap count is not arbitrary. This scene's defocused content is
// thin bright roots on a dark ground, which is the worst case for a sparse
// gather -- each tap lands either on a root or off it, so a sparse kernel prints
// the kernel's own shape as a string of beads along every out-of-focus root
// rather than blurring it. Nineteen taps over a radius capped at seven pixels
// keeps the sample spacing under a root's width, which is the condition for the
// gather to read as blur instead of as pattern.
#define DOF_TAPS 19
static constant float2 kDisk[DOF_TAPS] = {
    float2( 0.000,  0.000),
    // inner ring, r = 0.55
    float2( 0.550,  0.000), float2( 0.421,  0.354), float2( 0.096,  0.542),
    float2(-0.275,  0.476), float2(-0.517,  0.188), float2(-0.517, -0.188),
    float2(-0.275, -0.476), float2( 0.096, -0.542), float2( 0.421, -0.354),
    // outer ring, r = 1.0, rotated half a step so the two rings interleave
    float2( 0.985,  0.174), float2( 0.643,  0.766), float2( 0.087,  0.996),
    float2(-0.500,  0.866), float2(-0.906,  0.423), float2(-0.940, -0.342),
    float2(-0.574, -0.819), float2( 0.000, -1.000), float2( 0.574, -0.819)
};

// The supersample box, factored out so chromatic aberration can run it once per
// channel at three slightly different magnifications.
static float3 resolveScene(texture2d<float> tex, sampler smp, float2 uv,
                           constant RootPostU& U) {
    if (U.ssaa <= 1) return tex.sample(smp, uv).rgb;
    const float inv = 1.0 / float(U.ssaa);
    float3 acc = float3(0.0);
    for (int y = 0; y < 4; ++y) {
        if (y >= U.ssaa) break;
        for (int x = 0; x < 4; ++x) {
            if (x >= U.ssaa) break;
            const float2 o = (float2(x, y) + 0.5) * inv - 0.5;
            acc += tex.sample(smp, uv + o * U.srcTexel).rgb;
        }
    }
    return acc / float(U.ssaa * U.ssaa);
}

// ASC-CDL-style grade: gain scales, lift offsets, gamma bends the middle. The
// order matters and this is the standard one -- lift last would drag the
// highlights with it.
static float3 gradeCDL(float3 c, float3 lift, float3 gammaC, float3 gain) {
    c = c * gain + lift;
    return pow(max(c, 0.0), max(gammaC, 1e-3));
}

fragment float4 root_post_fs(PostVOut in [[stage_in]],
                             constant RootPostU&  U         [[buffer(0)]],
                             texture2d<float>     sceneTex   [[texture(0)]],
                             texture2d<float>     bloomTex   [[texture(1)]],
                             depth2d<float>       depthTex   [[texture(2)]],
                             texture2d<float>     haloTex    [[texture(3)]]) {
    constexpr sampler linSmp(mag_filter::linear, min_filter::linear,
                             address::clamp_to_edge);

    float2 uv = in.pos.xy / U.res;

    // --- lens distortion -----------------------------------------------------
    // A short lens does not just see more, it bends what it sees: straight lines
    // bow outward. Without this a "wide angle" setting is only a narrower crop
    // of the same rectilinear projection, which reads as stepping backwards
    // rather than as changing lens. Applied to the sample coordinate, so every
    // later read -- scene, depth, bloom -- stays registered to it for free.
    //
    // Radial, worked in an aspect-corrected space so the distortion is circular
    // rather than elliptical, then zoomed to keep the corners inside the source.
    if (U.distortK1 != 0.0 || U.distortK2 != 0.0) {
        const float aspect = U.res.x / max(U.res.y, 1.0);
        float2 c = (uv - 0.5) * float2(aspect, 1.0);
        const float r2 = dot(c, c);
        c *= 1.0 + U.distortK1 * r2 + U.distortK2 * r2 * r2;
        c *= U.distortZoom;
        uv = 0.5 + c / float2(aspect, 1.0);
    }

    // --- supersample resolve -------------------------------------------------
    // A box filter over the SSAA grid. The scene is rendered at ssaa x the
    // output, and the capsule pass's coverage is binary (it comes from a
    // discard), so this box *is* the anti-aliasing -- there is no hardware
    // coverage to resolve and nothing cleverer to reconstruct.
    float3 scene;
    if (U.caStrength > 0.0) {
        // Chromatic aberration: a real lens focuses the three wavelengths at
        // slightly different magnifications, so the channels are sampled from
        // uv scaled about the centre rather than offset. Scaling (not
        // translating) is what makes it vanish at the optical axis and grow
        // towards the corners, which is the whole signature of the effect --
        // a constant offset just looks like a misregistered print.
        const float2 c = uv - 0.5;
        // Per-pixel magnification difference, converted to a uv scale.
        const float k = U.caStrength / max(length(U.res) * 0.5, 1.0);
        scene = float3(resolveScene(sceneTex, linSmp, 0.5 + c * (1.0 + k), U).r,
                       resolveScene(sceneTex, linSmp, uv, U).g,
                       resolveScene(sceneTex, linSmp, 0.5 + c * (1.0 - k), U).b);
    } else {
        scene = resolveScene(sceneTex, linSmp, uv, U);
    }

    // --- depth of field ------------------------------------------------------
    // Deliberately a gather on the resolved image rather than a scatter or a
    // separate blurred target: the intent is to take the hard edge off the
    // tangle at the near and far extremes, not to simulate a lens. The circle
    // of confusion is signed-distance-from-focus normalised by dofRange, so
    // near and far defocus symmetrically.
    if (U.dofOn == 1 && U.dofStrength > 0.0) {
        const float d = depthTex.sample(linSmp, uv);
        // Depth is written GL-style (see root_geom.metal), so undo that before
        // the standard perspective un-projection.
        const float ndcZ = d * 2.0 - 1.0;
        const float ze = (2.0 * U.nearZ * U.farZ) /
                         (U.farZ + U.nearZ - ndcZ * (U.farZ - U.nearZ));
        float coc = saturate(abs(ze - U.dofFocus) / max(U.dofRange, 1e-3));
        coc = coc * coc * U.dofStrength;   // squared: the focal plane stays wide
        if (coc > 0.002) {
            // Rotate the disk per pixel so what is left of the ring structure
            // becomes noise rather than a repeated rosette.
            const float ang = ign(in.pos.xy) * 6.2831853;
            const float ca = cos(ang), sa = sin(ang);
            const float2 r = coc * 7.0 / U.res;   // 7 px at full defocus
            float3 acc = float3(0.0);
            for (int i = 0; i < DOF_TAPS; ++i) {
                const float2 k = kDisk[i];
                const float2 o = float2(k.x * ca - k.y * sa, k.x * sa + k.y * ca);
                acc += sceneTex.sample(linSmp, uv + o * r).rgb;
            }
            scene = mix(scene, acc * (1.0 / float(DOF_TAPS)), coc);
        }
    }

    // --- bloom ---------------------------------------------------------------
    if (U.bloomOn == 1) {
        scene += bloomTex.sample(linSmp, uv).rgb * U.bloomIntensity;

        // Halation. In film, light that makes it through the emulsion reflects
        // off the base and re-exposes the layers from behind -- and because the
        // red-sensitive layer is deepest, the halo it leaves is warm. It is the
        // single most recognisable difference between film and a clean digital
        // capture, and it is not the same thing as bloom: bloom is symmetric and
        // white and lives in the lens, halation is coloured and sits *only*
        // around what is genuinely bright. Reusing a deeper, already-thresholded
        // mip gets both properties for one sample.
        // x4: the halo mip is the fourth downsample of an already-thresholded
        // image, so a thin bright root has been averaged over 16x16 pixels by
        // the time it arrives and its peak is a fraction of what it started as.
        // Without the gain a 0..1 slider spans "invisible" to "barely one code
        // value", which is not a control.
        if (U.halation > 0.0)
            scene += haloTex.sample(linSmp, uv).rgb * U.halationTint.xyz
                   * (U.halation * 4.0);

        // Anamorphic streak. A cylindrical front element focuses horizontally
        // and vertically at different rates, so highlights smear along one axis.
        // Taken from the halo mip because it is small -- a wide horizontal reach
        // there costs a handful of taps rather than hundreds at output scale.
        if (U.streak > 0.0) {
            const float2 t = float2(U.streakLength / max(U.res.x, 1.0), 0.0);
            float3 st = float3(0.0);
            float wsum = 0.0;
            for (int i = -6; i <= 6; ++i) {
                // Triangular falloff: a box reads as a hard bar, a Gaussian this
                // wide needs more taps than the effect is worth.
                const float w = 1.0 - abs(float(i)) / 7.0;
                st += haloTex.sample(linSmp, uv + t * float(i)).rgb * w;
                wsum += w;
            }
            scene += st / wsum * U.streakTint.xyz * (U.streak * 4.0);
        }
    }

    // --- exposure and vignette (both scene-referred, both before the curve) ---
    float3 col = scene * U.exposure;
    if (U.vignette > 0.0) {
        float2 q = (uv - 0.5) * float2(U.res.x / U.res.y, 1.0);
        col *= mix(1.0, smoothstep(1.25, 0.35, length(q)), U.vignette);
    }

    col = (U.tonemap == 1) ? acesFitted(col) : clamp(col, 0.0, 1.0);

    // --- print grade ---------------------------------------------------------
    // After the curve, deliberately. These are the controls a colourist reaches
    // for and they are defined on a display-referred image: "lift the shadows"
    // means something specific about the black point of the picture you are
    // looking at, and applied in linear it would be an exposure change instead.
    {
        // Contrast about mid-grey. 0.4358 is 0.5 in sRGB taken back to linear --
        // pivoting on 0.5 of the *encoded* signal, which is where the eye reads
        // the middle of the picture.
        if (U.contrast != 1.0)
            col = clamp((col - 0.4358) * U.contrast + 0.4358, 0.0, 4.0);

        if (U.saturation != 1.0) {
            const float l = dot(col, float3(0.2126, 0.7152, 0.0722));
            col = clamp(mix(float3(l), col, U.saturation), 0.0, 4.0);
        }

        // Split toning: shadows one way, highlights the other. The classic
        // teal-and-orange, but here the useful version is the opposite of the
        // scene's own light -- the key is warm, so cooling the shadows is what
        // separates the tangle from the ground.
        if (U.toneBalance >= 0.0 && U.splitStrength > 0.0) {
            const float l = dot(col, float3(0.2126, 0.7152, 0.0722));
            const float hi = smoothstep(U.toneBalance - 0.25, U.toneBalance + 0.25, l);
            // The tints are stored as full-strength colours and scaled towards
            // neutral here, so one slider takes the whole effect from off to
            // full without having to edit two swatches in step.
            const float3 sh = mix(float3(1.0), U.shadowTint.xyz, U.splitStrength);
            const float3 hl = mix(float3(1.0), U.highlightTint.xyz, U.splitStrength);
            col *= mix(sh, hl, hi);
        }

        col = gradeCDL(col, U.lift.xyz, U.gammaC.xyz, U.gainC.xyz);
        col = clamp(col, 0.0, 1.0);
    }

    col = srgbEncode(col);

    // --- grain ---------------------------------------------------------------
    // Weighted towards the midtones by 1 - |2L-1|: grain in the deep shadows
    // reads as sensor noise and grain in the highlights reads as a broken
    // gradient, and neither is the intent.
    if (U.grain > 0.0) {
        // Grain has a physical size -- it is silver crystals, not pixels -- so
        // it does not get finer when the render resolution goes up. Quantising
        // the sample coordinate gives it a cell size that stays put in the
        // image, which is what makes it read as film stock rather than as
        // sensor noise.
        const float2 gp = floor(in.pos.xy / max(U.grainSize, 1.0))
                        + float2(U.time * 137.0, U.time * 79.0);
        const float mono = ign(gp) - 0.5;
        // Real dye-cloud grain is not perfectly correlated between layers, so a
        // little per-channel independence stops it looking like a luminance
        // overlay; too much and it reads as chroma noise.
        float3 n = float3(mono);
        if (U.grainChroma > 0.0)
            n = mix(n, float3(ign(gp + 11.0) - 0.5,
                              ign(gp + 23.0) - 0.5,
                              ign(gp + 37.0) - 0.5), U.grainChroma);
        const float l = dot(col, float3(0.2126, 0.7152, 0.0722));
        col = saturate(col + n * U.grain * (1.0 - abs(2.0 * l - 1.0)));
    }

    // --- dither --------------------------------------------------------------
    // Triangular PDF at one 8-bit step: the sum of two uniforms, so the
    // quantisation error is decorrelated from the signal and the fog gradients
    // stop banding. Applied post-encode because the banding is a property of
    // the encoded values, not the linear ones.
    if (U.ditherOn == 1) {
        const float n1 = ign(in.pos.xy);
        const float n2 = ign(in.pos.xy + 17.0);
        col += (n1 + n2 - 1.0) * (1.0 / 255.0);
    }

    return float4(col, 1.0);
}
