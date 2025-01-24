
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
out vec2 vPosition;
out vec2 vCenter;
out vec2 rectSize_px;
out vec2 vMajorAxis;
out vec2 vMinorAxis;

// [-1, 1] => [0, 1920]
vec2 ndcToPx(vec2 ndc, vec2 vp) {
    return (ndc * 0.5 + 0.5) * vp;
}

void main () {

    // the 32B <=> 2xRGBA <=> 2x4x16b are stored in 2 units of the texture data
//    uint x = (uint(aIndex) % 1024u) * 2u;
    uint x = (uint(aIndex) & 0x3ffu) << 1; // Extract lower 10 bits and multiply by 2
//    uint y = uint(aIndex) / 1024u;
    uint y = uint(aIndex) >> 10;           // Extract upper bits
    uvec4 centeru = texelFetch(uTexture, ivec2(x, y), 0); //center position of the splat
    vec4 center = vec4(uintBitsToFloat(centeru.xyz), 1);
    vec4 cam = uView * center;
    vec4 pos2d = uProj * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(uTexture, ivec2(x | 1u, y), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        uFocal.x / cam.z, 0., -(uFocal.x * cam.x) / (cam.z * cam.z),
        0., -uFocal.y / cam.z, (uFocal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(uView)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector; //in pixel
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4(
        (cov.w) & 0xffu,
        (cov.w >> 8) & 0xffu,
        (cov.w >> 16) & 0xffu,
        (cov.w >> 24) & 0xffu
    ) / 255.0;

    vCenter = vec2(pos2d) / pos2d.w; //[-1,1]

    vPosition = aPosition;
    
    // pos0 are [-1,1]
    vec2 axisSumPx = abs(majorAxis) + abs(minorAxis);
    vec2 axisSum01 = (abs(majorAxis) + abs(minorAxis))/uViewport;
    vec2 minRect = vCenter - 4.0 * axisSum01, maxRect = vCenter + 4.0 * axisSum01;
    vec2 minRect_px = ndcToPx(minRect,uViewport), maxRect_px = ndcToPx(maxRect, uViewport);
    rectSize_px = maxRect_px - minRect_px;
    
    #if ${useQuad}
        // to use with default rasterisation pipeline
        gl_Position = vec4(vCenter + 4.0 * (aPosition.x * majorAxis + aPosition.y * minorAxis) / uViewport, 0.0, 1.0);
    #else
        // generating fragment as unrotated rectangle, it has impact on the localPos
        gl_Position = vec4(vCenter + 4.0 * (aPosition * axisSumPx) / uViewport, 0.0, 1.0);
    #endif

    vMajorAxis = majorAxis;
    vMinorAxis = minorAxis;
}
`.trim();

// language=glsl
export const fragmentShaderSource = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vPosition; //[-1,1], but was before [-2,-2]
in vec2 vCenter; //[-1, 1] window-relative
in vec2 rectSize_px;
in vec2 vMajorAxis;
in vec2 vMinorAxis;

//gl_FragCoord // eg [0-1919, 0-1079] current pixel position
uniform vec2 uViewport;

layout(location = 0) out vec4 fragColor;

// [-1, 1] => [0, 1919]
vec2 ndcToPx(vec2 ndc, vec2 vp) {
    return (ndc * 0.5 + 0.5) * vp;
}

void main () {
    vec2 centerPx = ndcToPx(vCenter, uViewport);
    vec2 delta_px = gl_FragCoord.xy - centerPx;

#if ${useQuad}
    float A = dot(vPosition*2.0, vPosition*2.0);
#else
    //vec2 localPos = (delta_px / rectSize_px) * 4.0; //[-1, 1] local-relative - not rotated (rect)
    // TODO some optimization to discard points outside quad (which is calculated in the vertex)
    vec2 localPos = vec2(
        dot(delta_px, normalize(vMajorAxis)) / length(vMajorAxis),
        dot(delta_px, normalize(vMinorAxis)) / length(vMinorAxis)); // [-1, 1] - rotated (rect)

    float A = dot(localPos, localPos);
#endif
    
    if (A > 4.0) discard;
    float B = exp(-A) * vColor.a;
    if(B < 1.0/255.0) discard;
    fragColor = vec4(vColor.rgb * B, B);
}
`.trim();
