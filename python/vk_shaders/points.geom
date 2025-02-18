#version 450

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in vec4 vColor[];

layout(location = 0) out vec4 gColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    float width;
    float height;
    float focal;
} ubo;

void main() {
    vec4 center = gl_in[0].gl_Position;
    vec2 vp = vec2(ubo.width, ubo.height);
    float size = 10;

    vec2 quad[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );

    for (int i = 0; i < 4; i++) {
        gl_Position = center + vec4(quad[i] * size / vp, 0.0, 0.0);
        gColor = vColor[0];
        EmitVertex();
    }

    EndPrimitive();
}