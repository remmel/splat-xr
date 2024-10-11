// point shader

// language=glsl
export const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;

in int index;

out vec4 vColor;

void main () {
    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
    gl_Position = projection * cam;
    gl_PointSize = 1.0; // Set point size

    uvec4 col = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vColor = vec4((col.w) & 0xffu, (col.w >> 8) & 0xffu, (col.w >> 16) & 0xffu, (col.w >> 24) & 0xffu) / 255.0;
}`.trim();

// language=glsl
export const fragmentShaderSource = `#version 300 es
precision highp float;

in vec4 vColor;

out vec4 fragColor;

void main () {
    fragColor = vColor;
}
`.trim();
