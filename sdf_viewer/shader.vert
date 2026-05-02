#version 410 core

out vec2 v_uv;

void main() {
    // Two triangles covering clip space, driven purely by gl_VertexID (no VBO needed)
    vec2 verts[6];
    verts[0] = vec2(-1.0, -1.0);
    verts[1] = vec2( 1.0, -1.0);
    verts[2] = vec2(-1.0,  1.0);
    verts[3] = vec2( 1.0, -1.0);
    verts[4] = vec2( 1.0,  1.0);
    verts[5] = vec2(-1.0,  1.0);

    vec2 p = verts[gl_VertexID];
    gl_Position = vec4(p, 0.0, 1.0);
    v_uv = p * 0.5 + 0.5;
}
