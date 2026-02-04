#version 450

#ifdef VERT // VERTEX SHADER ****************************************************

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

#endif // VERT

#ifdef GEOM // GEOMETRY SHADER ****************************************************

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in SplatData {
    vec3 center;
    vec3 scale;
    vec4 rotation;
    vec4 color;
} gs_in[];

layout(location = 0) out vec4 gColor;
layout(location = 1) out vec2 gPosition;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec2 viewport;
    vec2 focal;
} ubo;

mat3 quat_to_mat3(vec4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return mat3(
        1. - 2. * (y * y + z * z),  2. * (x * y + w * z),       2. * (x * z - w * y),
        2. * (x * y - w * z),       1. - 2. * (x * x + z * z),  2. * (y * z + w * x),
        2. * (x * z + w * y),       2. * (y * z - w * x),       1. - 2. * (x * x + y * y)
    );
}

mat3 computeCov3D(vec4 quaternion, vec3 scale) {
    mat3 R = quat_to_mat3(quaternion);
    mat3 S = mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    mat3 M = R * S;
    mat3 cov3d = M * transpose(M);
    return cov3d;
}

vec3 computeCov2D(vec4 cam, mat3 Vrk, mat4 view, vec2 focal) {
    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z),
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;
    return vec3(cov2d[0][0], cov2d[0][1], cov2d[1][1]); //a,b,d
}

void main() {
    vec4 center = vec4(gs_in[0].center, 1.0);
    vec3 scale = gs_in[0].scale;
    vec4 rotation = gs_in[0].rotation;
    vec4 color = gs_in[0].color;

    vec4 cam = ubo.view * center;
    vec4 pos2d = ubo.proj * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    mat3 Vrk = 4.0 * computeCov3D(rotation, scale);

    vec3 cov = computeCov2D(cam, Vrk, ubo.view, ubo.focal);

    float mid = (cov.x + cov.z) / 2.0;
    float radius = length(vec2((cov.x - cov.z) / 2.0, cov.y));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov.y, lambda1 - cov.x));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector; //in pixel
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    gColor = color;
    vec2 vCenter = vec2(pos2d) / pos2d.w;

    vec2 quad[4] = vec2[](vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0), vec2( 1.0,  1.0));

    for (int i = 0; i < 4; i++) {
        gColor = color; //it must be in the loop when using my RTX :s
        gPosition = quad[i] * 2.0;
        gl_Position = vec4(vCenter + (gPosition.x * majorAxis + gPosition.y * minorAxis) / ubo.viewport, 0.0, 1.0);
        gl_Position.y = -gl_Position.y; //flip y for vulkan
        EmitVertex();
    }

    EndPrimitive();
}

#endif // GEOM


#ifdef FRAG // FRAGMENT SHADER ****************************************************

layout(location = 0) in vec4 gColor;
layout(location = 1) in vec2 gPosition;

layout(location = 0) out vec4 fragColor;

void main() {
  float A = -dot(gPosition, gPosition);
  if (A < -4.0) discard;
  float B = exp(A) * gColor.a;
  fragColor = vec4(B * gColor.rgb, B);
}

#endif // FRAG