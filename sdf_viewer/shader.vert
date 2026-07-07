#version 410 core

uniform samplerBuffer  u_nodes;
uniform isamplerBuffer u_segments;
uniform samplerBuffer  u_radii;
uniform mat4           u_viewProj;
uniform float          u_fov;
uniform vec2           u_res;
uniform float          u_radiusScale;
uniform float          u_radiusMin;   // clamp, 0 = no floor
uniform float          u_radiusMax;   // clamp, 0 = no ceiling

uniform isamplerBuffer u_primType;    // 0 = capsule, 1 = blade
uniform samplerBuffer  u_primFrame;   // blade half-width vector in .xyz, curl in .w
uniform samplerBuffer  u_primAux;     // (s0,s1,latCup,gradBias)

flat out int  v_segIdx;
out vec2      v_ndc;

void main() {
    int i = gl_InstanceID;
    v_segIdx = i;

    ivec2 seg = texelFetch(u_segments, i).xy;
    vec3  a   = texelFetch(u_nodes, seg.x).xyz;
    vec3  b   = texelFetch(u_nodes, seg.y).xyz;
    float r   = texelFetch(u_radii, i).r * u_radiusScale;
    if (u_radiusMin > 0.0) r = max(r, u_radiusMin);
    if (u_radiusMax > 0.0) r = min(r, u_radiusMax);

    // Blades reach out to their half-width AND bend/curl off the a->b axis by
    // up to ~curl*0.6*L (see sdBlade's `bend`) plus the cross-section cup. If
    // the quad only covered the half-width, the curled surface would poke past
    // the AABB and get sliced by a hard straight edge. Include the curl term.
    if (texelFetch(u_primType, i).r != 0) {
        vec4  fr   = texelFetch(u_primFrame, i);
        float hw   = length(fr.xyz);
        float curl = abs(fr.w);
        float L    = distance(a, b);
        // Rolled cross-section stays within ~hw of the axis in the cross plane
        // (a strip of arc-length hw wrapped onto a circle never gets farther
        // than hw from the axis), plus the lengthwise bend ~curl*0.6*L. Cover
        // both generously so no marched surface is sliced by the quad edge.
        float reach = curl * 0.6 * L + 2.0 * hw;
        r = max(r, hw + reach) * 1.7;
    }

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
