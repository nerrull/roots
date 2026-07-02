#version 410 core

in vec3 v_worldPos;
in vec3 v_normal;
in vec3 v_color;
in vec3 v_lightPos;

uniform vec3 u_eye;
uniform vec3 u_lightDir;

uniform float u_lightIntensity;   // diffuse strength of the per-face point light
uniform float u_lightFalloff;     // quadratic attenuation coefficient -- lower reaches further
uniform float u_specStrength;

uniform vec3  u_veinColor;
uniform float u_veinScale;        // spatial frequency of the marble turbulence
uniform float u_veinStrength;     // 0 = flat color, 1 = fully veined

out vec4 fragColor;

// cheap 3-octave value noise for marble turbulence (self-contained, no
// shared includes across GLSL files).
float _hash(vec3 p) {
    p = fract(p * vec3(443.897, 397.297, 491.187));
    p += dot(p, p.zyx + 19.19);
    return fract((p.x + p.y) * p.z);
}
float _noise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(_hash(i),             _hash(i+vec3(1,0,0)), u.x),
            mix(_hash(i+vec3(0,1,0)), _hash(i+vec3(1,1,0)), u.x), u.y),
        mix(mix(_hash(i+vec3(0,0,1)), _hash(i+vec3(1,0,1)), u.x),
            mix(_hash(i+vec3(0,1,1)), _hash(i+vec3(1,1,1)), u.x), u.y), u.z);
}
float _turbulence(vec3 p) {
    return abs(_noise(p) - 0.5) * 0.5000
         + abs(_noise(p * 2.03) - 0.5) * 0.2500
         + abs(_noise(p * 4.07) - 0.5) * 0.1250;
}

void main() {
    vec3 n = normalize(v_normal);
    vec3 v = normalize(u_eye - v_worldPos);

    // --- marble veining: classic turbulence-warped sine bands, blended
    // between the base mask color and a vein color. ---
    float t = _turbulence(v_worldPos * u_veinScale);
    float vein = sin((v_worldPos.x + v_worldPos.y * 0.4 + t * 12.0) * u_veinScale * 3.0);
    vein = smoothstep(0.15, 0.85, vein * 0.5 + 0.5);
    vec3 baseColor = mix(v_color, u_veinColor, vein * u_veinStrength);

    // very dim ambient/global directional term -- just enough that unlit
    // parts of the face aren't pure black, so the point light dominates.
    vec3 l = normalize(u_lightDir);
    float lam = abs(dot(n, l));
    vec3 col = baseColor * (0.04 + 0.06 * lam);

    // primary light: a point light floating in front of the face (in the
    // view-clear cylinder, so it's never occluded by the wrapped roots).
    vec3 toLight = v_lightPos - v_worldPos;
    float dist = length(toLight);
    vec3 ldir = toLight / max(dist, 0.001);
    float atten = 1.0 / (1.0 + u_lightFalloff * dist * dist);
    float ndotl = max(dot(n, ldir), 0.0);
    vec3 h = normalize(ldir + v);
    // polished-stone highlight: tight and bright, plus a soft fresnel rim.
    float spec = pow(max(dot(n, h), 0.0), 60.0) * u_specStrength;
    float fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.25;
    col += baseColor * ndotl * atten * u_lightIntensity
         + vec3(spec) * atten
         + vec3(fresnel) * atten;

    fragColor = vec4(col, 1.0);
}
