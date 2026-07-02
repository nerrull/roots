#version 410 core

in vec3 v_worldPos;
in vec3 v_normal;
in vec3 v_color;

uniform vec3 u_eye;
uniform vec3 u_lightDir;

out vec4 fragColor;

void main() {
    vec3 n = normalize(v_normal);
    vec3 l = normalize(u_lightDir);
    // two-sided lambert (marble reliefs are thin -- normals can face away from camera)
    float lam = abs(dot(n, l));
    vec3 v = normalize(u_eye - v_worldPos);
    vec3 h = normalize(l + v);
    float spec = pow(max(dot(n, h), 0.0), 24.0) * 0.15;
    vec3 col = v_color * (0.30 + 0.75 * lam) + vec3(spec);
    fragColor = vec4(col, 1.0);
}
