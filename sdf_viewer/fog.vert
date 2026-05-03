#version 410 core

out vec2 v_uv;

void main() {
    // Fullscreen triangle — no VBO needed.
    // VertexID 0,1,2 → NDC (-1,-1), (3,-1), (-1,3)
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    v_uv        = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
