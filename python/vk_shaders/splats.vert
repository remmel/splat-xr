#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inScale;
layout(location = 2) in vec4 inRotation;
layout(location = 3) in vec4 inColor;

layout(location = 0) out SplatData {
    vec3 center;
    vec3 scale;
    vec4 rotation;
    vec4 color;
} gs_out;

void main() {
    gs_out.center = inPosition;
    gs_out.scale = inScale;
    gs_out.rotation = inRotation;
    gs_out.color = inColor;
}