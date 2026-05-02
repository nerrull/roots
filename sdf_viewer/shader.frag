#version 410 core

in  vec2 v_uv;
out vec4 fragColor;

uniform samplerBuffer  u_nodes;     // vec3 per node (GL_RGB32F)
uniform isamplerBuffer u_segments;  // ivec2 per segment (GL_RG32I)
uniform samplerBuffer  u_radii;     // float per segment (GL_R32F)
uniform int            u_segCount;
uniform vec3           u_eye;
uniform mat3           u_cam;       // columns: [right, up, forward]
uniform float          u_fov;       // vertical half-angle in radians
uniform vec2           u_res;

// Capsule SDF for one root segment
float capsuleSDF(vec3 p, vec3 a, vec3 b, float r) {
    vec3 ab = b - a;
    vec3 ap = p - a;
    float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    return length(ap - t * ab) - r;
}

float sceneSDF(vec3 p) {
    float d = 1e9;
    for (int i = 0; i < u_segCount; i++) {
        ivec2 seg = texelFetch(u_segments, i).xy;
        vec3  a   = texelFetch(u_nodes, seg.x).xyz;
        vec3  b   = texelFetch(u_nodes, seg.y).xyz;
        float r   = texelFetch(u_radii, i).r;
        d = min(d, capsuleSDF(p, a, b, r));
    }
    return d;
}

vec3 calcNormal(vec3 p) {
    const float e = 0.01;
    return normalize(vec3(
        sceneSDF(p + vec3(e, 0, 0)) - sceneSDF(p - vec3(e, 0, 0)),
        sceneSDF(p + vec3(0, e, 0)) - sceneSDF(p - vec3(0, e, 0)),
        sceneSDF(p + vec3(0, 0, e)) - sceneSDF(p - vec3(0, 0, e))
    ));
}

void main() {
    if (u_segCount == 0) {
        fragColor = vec4(0.12, 0.08, 0.05, 1.0);
        return;
    }

    // Build perspective ray
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= u_res.x / u_res.y;
    vec3 rd = normalize(u_cam * vec3(uv * tan(u_fov), 1.0));
    vec3 ro = u_eye;

    // Sphere trace
    const int   MAX_STEPS = 96;
    const float HIT_EPS   = 0.001;
    const float MAX_DIST  = 200.0;

    float t   = 0.0;
    float hit = -1.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        float d = sceneSDF(ro + rd * t);
        if (d < HIT_EPS) { hit = t; break; }
        t += d;
        if (t > MAX_DIST) break;
    }

    if (hit > 0.0) {
        vec3 p    = ro + rd * hit;
        vec3 n    = calcNormal(p);
        vec3 ldir = normalize(vec3(-0.5, 1.0, 0.3));
        vec3 base = vec3(0.55, 0.40, 0.20);

        float diff = max(dot(n, ldir), 0.0);
        vec3  h    = normalize(ldir - rd);
        float spec = pow(max(dot(n, h), 0.0), 32.0);

        vec3 color = base * (0.2 + 0.8 * diff) + vec3(0.3) * spec;
        fragColor = vec4(color, 1.0);
    } else {
        fragColor = vec4(0.12, 0.08, 0.05, 1.0);
    }
}
