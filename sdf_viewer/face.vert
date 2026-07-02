#version 410 core

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;

uniform mat4 u_viewProj;

out vec3 v_worldPos;
out vec3 v_normal;
out vec3 v_color;

void main() {
    v_worldPos  = a_pos;
    v_normal    = a_normal;
    v_color     = a_color;
    gl_Position = u_viewProj * vec4(a_pos, 1.0);
}
