// point shader

// language=glsl
export const vertexShaderSourcePoint = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D uTexture;
uniform mat4 uProj, uView;

in int aIndex;

out vec4 vColor;

void main () {
    // the 32B <=> 2xRGBA <=> 2x4x16b are stored in 2 units of the texture data
//    uint x = (uint(aIndex) % 1024u) * 2u;
    uint x = (uint(aIndex) & 0x3ffu) << 1; // Extract lower 10 bits and multiply by 2
//    uint y = uint(aIndex) / 1024u;
    uint y = uint(aIndex) >> 10;           // Extract upper bits

    uvec4 centeru = texelFetch(uTexture, ivec2(x, y), 0); //center position of the splat
    vec4 center = vec4(uintBitsToFloat(centeru.xyz), 1);
    vec4 cam = uView * center;
    gl_Position = uProj * cam;
    gl_PointSize = 1.0; // Set point size

    uvec4 col = texelFetch(uTexture, ivec2(x | 1u, y), 0);
    
    vec4 color = vec4(
        (col.w) & 0xffu, //[0-255] color, vColor needs float
        (col.w >> 8) & 0xffu,
        (col.w >> 16) & 0xffu,
        (col.w >> 24) & 0xffu
    ) / 255.0;
//    color.r = 1.0 - uintBitsToFloat(uint(aIndex)) / 10000.0;
    vColor = color;
    
}`.trim();

// language=glsl
export const fragmentShaderSourcePoint = `
#version 300 es
precision highp float;

in vec4 vColor;

out vec4 fragColor;

void main () {
//    fragColor = vec4(1.0,1.0,0.0,1.0);
    fragColor = vColor;
}
`.trim();

// language=glsl
export const vertexShaderSourcePointArrays = `
#version 300 es
precision highp float;

uniform mat4 uProj;
uniform mat4 uView;

in vec3 aPosition;
in vec4 aColor;

out vec4 vColor;

void main() {
    vec4 viewPosition = uView * vec4(aPosition, 1.0);
    gl_Position = uProj * viewPosition;
    gl_PointSize = 1.0;
    vColor = aColor;
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