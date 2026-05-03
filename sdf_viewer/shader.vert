#version 410 core

uniform samplerBuffer  u_nodes;
uniform isamplerBuffer u_segments;
uniform samplerBuffer  u_radii;
uniform mat4           u_viewProj;
uniform float          u_fov;
uniform vec2           u_res;
uniform float          u_radiusScale;

flat out int  v_segIdx;
out vec2      v_ndc;

void main() {
    int i = gl_InstanceID;
    v_segIdx = i;

    ivec2 seg = texelFetch(u_segments, i).xy;
    vec3  a   = texelFetch(u_nodes, seg.x).xyz;
    vec3  b   = texelFetch(u_nodes, seg.y).xyz;
    float r   = texelFetch(u_radii, i).r * u_radiusScale;

    vec4 ca = u_viewProj * vec4(a, 1.0);
    vec4 cb = u_viewProj * vec4(b, 1.0);

    // Both endpoints behind camera — degenerate the quad
    if (ca.w <= 0.0 && cb.w <= 0.0) {
        gl_Position = vec4(0.0, 0.0, 0.0, -1.0);
        v_ndc = vec2(0.0);
        return;
    }

    vec2 ndcA = ca.xy / max(ca.w, 0.001);
    vec2 ndcB = cb.xy / max(cb.w, 0.001);

    // Conservative screen-space bounding box expanded by projected radius.
    // NDC y-expansion: r * (1/tan(fov)) / depth
    // NDC x-expansion: same / aspect
    float aspect = u_res.x / u_res.y;
    float f      = 1.0 / tan(u_fov);
    float minW   = max(min(ca.w, cb.w), 0.001);
    float sry    = r * f / minW;
    float srx    = sry / aspect;

    vec2 lo = min(ndcA, ndcB) - vec2(srx, sry);
    vec2 hi = max(ndcA, ndcB) + vec2(srx, sry);

    // Two triangles, corners selected by vertex ID
    vec2 corners[6];
    corners[0] = vec2(lo.x, lo.y);
    corners[1] = vec2(hi.x, lo.y);
    corners[2] = vec2(lo.x, hi.y);
    corners[3] = vec2(hi.x, lo.y);
    corners[4] = vec2(hi.x, hi.y);
    corners[5] = vec2(lo.x, hi.y);

    vec2 pos    = corners[gl_VertexID];
    v_ndc       = pos;
    gl_Position = vec4(pos, 0.0, 1.0);
}
