// point shader

// language=glsl
export const vertexShaderSourcePoint = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;

in int index;

out vec4 vColor;

void main () {
    // the 32B <=> 2xRGBA <=> 2x4x16b are stored in 2 units of the texture data
//    uint x = (uint(index) % 1024u) * 2u;
    uint x = (uint(index) & 0x3ffu) << 1; // Extract lower 10 bits and multiply by 2
//    uint y = uint(index) / 1024u;
    uint y = uint(index) >> 10;           // Extract upper bits

    uvec4 cen = texelFetch(u_texture, ivec2(x, y), 0);
    vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
    gl_Position = projection * cam;
    gl_PointSize = 1.0; // Set point size

    uvec4 col = texelFetch(u_texture, ivec2(x | 1u, y), 0);
    vColor = vec4(
        (col.w) & 0xffu, //[0-255] color, vColor needs float
        (col.w >> 8) & 0xffu,
        (col.w >> 16) & 0xffu,
        (col.w >> 24) & 0xffu
    ) / 255.0;
}`.trim();

// language=glsl
export const fragmentShaderSourcePoint = `
#version 300 es
precision highp float;

in vec4 vColor;

out vec4 fragColor;

void main () {
    fragColor = vec4(1.0,1.0,0.0,1.0);
//    fragColor = vColor;
}
`.trim();

// language=glsl
export const vertexShaderSourcePointArrays = `
#version 300 es
precision highp float;

uniform mat4 projection;
uniform mat4 view;

in vec3 a_position;
in vec4 a_color;

out vec4 vColor;

void main() {
    vec4 viewPosition = view * vec4(a_position, 1.0);
    gl_Position = projection * viewPosition;
    gl_PointSize = 1.0;
    vColor = a_color;
}
`.trim();

// language=glsl
export const fragmentShaderSourcePointArray = `
#version 300 es
precision highp float;

in vec4 vColor;
out vec4 fragColor;

void main() {
    fragColor = vColor;
}
`.trim();