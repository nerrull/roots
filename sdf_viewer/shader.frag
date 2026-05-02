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

flat in int  v_segIdx;
in vec2      v_ndc;

out vec4 fragColor;

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

void main() {
    vec2 ndc = v_ndc;
    ndc.x *= u_res.x / u_res.y;
    vec3 rd = normalize(u_cam * vec3(ndc * tan(u_fov), 1.0));
    vec3 ro = u_eye;

    ivec2 seg = texelFetch(u_segments, v_segIdx).xy;
    vec3  a   = texelFetch(u_nodes, seg.x).xyz;
    vec3  b   = texelFetch(u_nodes, seg.y).xyz;
    float r   = texelFetch(u_radii, v_segIdx).r;

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

    vec4 clipP = u_viewProj * vec4(p, 1.0);
    gl_FragDepth = (clipP.z / clipP.w) * 0.5 + 0.5;

    float diff = max(dot(n, u_lightDir), 0.0);
    vec3  h    = normalize(u_lightDir - rd);
    float spec = pow(max(dot(n, h), 0.0), u_shininess);

    fragColor = vec4(u_baseColor * (u_ambient + u_diffuse * diff) + u_specColor * spec, 1.0);
}
