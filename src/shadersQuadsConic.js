
const useQuad = 1 // will use rotated quads -  standard way to render splats in opengl
// const useQuad = 0 // will use un-rotated rects (aligned with axis) - to match python
// language=glsl
export const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D uTexture;
uniform mat4 uProj, uView;
uniform vec2 uFocal; //focal in pixel eg [1150, 1150]
uniform vec2 uViewport; //resolution in pixel eg [1920, 1080]

in vec2 aPosition;
in int aIndex;

out vec4 vColor;
out vec3 vConic;
out vec2 vCoordxy;

// [-1, 1] => [0, 1920]
vec2 ndcToPx(vec2 ndc, vec2 vp) {
    return (ndc * 0.5 + 0.5) * vp;
}

vec3 computeCov2D(vec4 cam, vec2 focal, mat3 cov3D, mat4 view) //vec2 tan_fov, 
{
    // why need this? Try remove this later
    /*
    float limx = 1.3f * tan_fov.x;
    float limy = 1.3f * tan_fov.y;
    float txtz = cam.x / cam.z;
    float tytz = cam.y / cam.z;
    cam.x = min(limx, max(-limx, txtz)) * cam.z;
    cam.y = min(limy, max(-limy, tytz)) * cam.z;
    */

    mat3 J = mat3(
        focal.x / cam.z, 0.0f, -(focal.x * cam.x) / (cam.z * cam.z),
        0.0f, -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
        0, 0, 0
    ); 

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov = transpose(T) * cov3D * T;
    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    // cov[0][0] += 0.3f;
    // cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}



void main () {

    // unpack data
    // the 32B <=> 2xRGBA <=> 2x4x16b are stored in 2 units of the texture data
    uint x = (uint(aIndex) & 0x3ffu) << 1; // Extract lower 10 bits and multiply by 2
    uint y = uint(aIndex) >> 10;           // Extract upper bits
    uvec4 centeru = texelFetch(uTexture, ivec2(x, y), 0); //center position of the splat
    vec4 center = vec4(uintBitsToFloat(centeru.xyz), 1);
    
    uvec4 cov = texelFetch(uTexture, ivec2(x | 1u, y), 0);
    vColor = vec4((cov.w) & 0xffu,(cov.w >> 8) & 0xffu,(cov.w >> 16) & 0xffu,(cov.w >> 24) & 0xffu) / 255.0;
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);
    
    // calculate the vertex position
    vec4 cam = uView * center;
    vec4 pos2d = uProj * cam;
    vec2 center_ndc = pos2d.xy / pos2d.w;
    // early culling
    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
   
    vec3 cov2d = computeCov2D(cam, uFocal, Vrk, uView); 
    
    // Invert covariance (EWA algorithm)
    float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
    if (det == 0.0f) gl_Position = vec4(0.f, 0.f, 0.f, 0.f);
    
    float det_inv = 1.f / det;
    vConic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    
    vec2 quadwh_px = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));  // half quad height and width
    vec2 quadwh_ndc = quadwh_px / uViewport * 2.f;  // in ndc space
    
    gl_Position = vec4(center_ndc + aPosition * quadwh_ndc, 0.0, 1.0); //[-1, 1]
    vCoordxy = aPosition * quadwh_px;
}
`.trim();


// language=glsl
export const fragmentShaderSource = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec3 vConic;
in vec2 vCoordxy;  // local coordinate in quad, unit in pixel

out vec4 fragColor;

void main()
{
    float power = -0.5f * (vConic.x * vCoordxy.x * vCoordxy.x + vConic.z * vCoordxy.y * vCoordxy.y) - vConic.y * vCoordxy.x * vCoordxy.y;
    if (power > 0.f) discard;
    float a = min(0.99f, vColor.a * exp(power));
    if (a < 1.f / 255.f) discard;
    fragColor = vec4(vColor.rgb * a, a);
}
`.trim();
