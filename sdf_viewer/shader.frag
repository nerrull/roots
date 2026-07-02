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
uniform vec3  u_wispPos[16];
uniform vec3  u_wispColor[16];
uniform float u_wispIntensity[16];

uniform float u_radiusScale;
uniform float u_radiusMin;
uniform float u_radiusMax;

uniform vec3  u_baseColor2;        // second color, blended in by noise
uniform float u_colorNoiseScale;   // spatial frequency of the mottling
uniform float u_colorNoiseStrength;

flat in int  v_segIdx;
in vec2      v_ndc;

out vec4 fragColor;

const float PI = 3.14159265359;

// cheap 3-octave value noise for root-surface color mottling -- breaks the
// flat single-color "3D pipes screensaver" look without needing a texture.
float _rhash(vec3 p) {
    p = fract(p * vec3(443.897, 397.297, 491.187));
    p += dot(p, p.zyx + 19.19);
    return fract((p.x + p.y) * p.z);
}
float _rnoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(_rhash(i),             _rhash(i+vec3(1,0,0)), u.x),
            mix(_rhash(i+vec3(0,1,0)), _rhash(i+vec3(1,1,0)), u.x), u.y),
        mix(mix(_rhash(i+vec3(0,0,1)), _rhash(i+vec3(1,0,1)), u.x),
            mix(_rhash(i+vec3(0,1,1)), _rhash(i+vec3(1,1,1)), u.x), u.y), u.z);
}
float _rfbm(vec3 p) {
    return 0.55 * _rnoise(p) + 0.30 * _rnoise(p * 2.11) + 0.15 * _rnoise(p * 4.37);
}

float capsuleSDF(vec3 p, vec3 a, vec3 b, float r) {
    vec3  ab = b - a;
    vec3  ap = p - a;
    float t  = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    return length(ap - t * ab) - r;
}

vec3 capsuleNormal(vec3 p, vec3 a, vec3 b) {
    vec3  ab = b - a;
    float t  = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return normalize(p - (a + t * ab));
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

    const int   MAX_STEPS = 32;
    const float HIT_EPS   = 0.001;
    const float MAX_DIST  = 200.0;

    float t   = 0.0;
    float hit = -1.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        float d = capsuleSDF(ro + rd * t, a, b, r);
        if (d < HIT_EPS) { hit = t; break; }
        t += d;
        if (t > MAX_DIST) break;
    }

    if (hit < 0.0) discard;

    vec3 p = ro + rd * hit;
    vec3 n = capsuleNormal(p, a, b);
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
    float cn = _rfbm(p * u_colorNoiseScale);
    vec3 albedo = mix(u_baseColor, u_baseColor2, smoothstep(0.3, 0.7, cn) * u_colorNoiseStrength);

    if (u_shaderMode == 0) {
        // --- Phong ---
        float diff = max(dot(n, u_lightDir), 0.0);
        vec3  h    = normalize(u_lightDir + V);
        float spec = pow(max(dot(n, h), 0.0), u_shininess);
        color = albedo * (u_ambient + u_diffuse * diff) + u_specColor * spec;

        for (int wi = 0; wi < u_wispCount; wi++) {
            vec3  lv    = u_wispPos[wi] - p;
            float dist2 = dot(lv, lv);
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

        for (int wi = 0; wi < u_wispCount; wi++) {
            vec3  lv    = u_wispPos[wi] - p;
            float dist2 = dot(lv, lv);
            vec3  ldir  = lv * inversesqrt(dist2);
            float att   = u_wispIntensity[wi] / (1.0 + dist2 * 0.008);
            color += att * u_wispColor[wi] * pbrBRDF(n, V, ldir, albedo, u_metallic, u_roughness);
        }
    }

    fragColor = vec4(color, 1.0);
}
