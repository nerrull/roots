#version 410 core

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;
layout(location = 3) in vec3 a_lightPos;   // per-face point light, floating in front of it

uniform mat4 u_viewProj;

out vec3 v_worldPos;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_lightPos;

void main() {
    v_worldPos  = a_pos;
    v_normal    = a_normal;
    v_color     = a_color;
    v_lightPos  = a_lightPos;
    gl_Position = u_viewProj * vec4(a_pos, 1.0);
}
