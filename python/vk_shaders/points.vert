#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inScale;
layout(location = 2) in vec4 inRotation;
layout(location = 3) in vec4 inColor;

layout(location = 0) out vec4 vColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    float width;
    float height;
    float focal;
} ubo;

void main() {
    gl_Position = ubo.proj * ubo.view * vec4(inPosition, 1.0);
    gl_PointSize = 4.0;

    vColor = inColor;
}