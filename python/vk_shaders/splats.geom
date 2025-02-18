#version 450

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
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return mat3(
        1.0 - 2.0 * (z * z + w * w),
        2.0 * (y * z + x * w),
        2.0 * (y * w - x * z),

        2.0 * (y * z - x * w),
        1.0 - 2.0 * (y * y + w * w),
        2.0 * (z * w + x * y),

        2.0 * (y * w + x * z),
        2.0 * (z * w - x * y),
        1.0 - 2.0 * (y * y + z * z)
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


    mat3 J = mat3(
        ubo.focal.x / cam.z, 0., -(ubo.focal.x * cam.x) / (cam.z * cam.z),
        0., -ubo.focal.y / cam.z, (ubo.focal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(ubo.view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector; //in pixel
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    gColor = color;
    vec2 vCenter = vec2(pos2d) / pos2d.w;

    vec2 quad[4] = vec2[](
        vec2(-2.0, -2.0),
        vec2( 2.0, -2.0),
        vec2(-2.0,  2.0),
        vec2( 2.0,  2.0)
    );

    for (int i = 0; i < 4; i++) {
        gColor = color; //it must be in the loop when using my RTX :s
        gPosition = quad[i];
        gl_Position = vec4(vCenter + (quad[i].x * majorAxis + quad[i].y * minorAxis) / ubo.viewport, 0.0, 1.0);
        gl_Position.y = -gl_Position.y; //flip y for vulkan
        EmitVertex();
    }

    EndPrimitive();
}