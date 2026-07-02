#version 410 core

in vec3 v_worldPos;
in vec3 v_normal;
in vec3 v_color;
in vec3 v_lightPos;

uniform vec3 u_eye;
uniform vec3 u_lightDir;

out vec4 fragColor;

void main() {
    vec3 n = normalize(v_normal);
    vec3 v = normalize(u_eye - v_worldPos);

    // very dim ambient/global directional term -- just enough that unlit
    // parts of the face aren't pure black, so the point light dominates and
    // actually reads as a distinct light source instead of a subtle tweak.
    vec3 l = normalize(u_lightDir);
    float lam = abs(dot(n, l));
    vec3 col = v_color * (0.04 + 0.06 * lam);

    // primary light: a point light floating in front of the face (in the
    // view-clear cylinder, so it's never occluded by the wrapped roots).
    // Slow falloff (stays bright out to ~lightDist) and a tight, strong
    // specular glint so it reads as a real spotlight, not a faint highlight.
    vec3 toLight = v_lightPos - v_worldPos;
    float dist = length(toLight);
    vec3 ldir = toLight / max(dist, 0.001);
    float atten = 1.0 / (1.0 + 0.012 * dist * dist);
    float ndotl = max(dot(n, ldir), 0.0);
    vec3 h = normalize(ldir + v);
    float spec = pow(max(dot(n, h), 0.0), 40.0) * 1.2;
    col += v_color * ndotl * atten * 3.2 + vec3(spec) * atten;

    fragColor = vec4(col, 1.0);
}
