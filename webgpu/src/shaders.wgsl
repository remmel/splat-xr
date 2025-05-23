struct Uniforms {
    view : mat4x4f, //modelView = view * model
    proj: mat4x4f,
    focal : vec2f, //in px
    viewport : vec2f, //in px
}
@binding(0) @group(0) var<uniform> u : Uniforms;

struct SplatSSBO {
    center: vec3f,
    _padding0: f32,
    cov3d_packed: vec3u,
    color_packed: u32,
}
@binding(1) @group(0) var<storage> splats: array<SplatSSBO>;
@binding(2) @group(0) var<storage> splatOrder: array<u32>;

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) vPosition: vec2f, //position relative to the fragment
    @location(1) vColor: vec4f,
}

// Quad vertex positions for a triangle strip
// Arranged as a unit quad centered at origin
var<private> quadPositions: array<vec2f, 4> = array<vec2f, 4>(
    vec2f(-1., -1.),
    vec2f(1., -1.),
    vec2f(-1., 1.),
    vec2f(1., 1.)
);

struct Splat {
    center: vec4f,
    cov3d: mat3x3f,
    color: vec4f,
}

fn unpackSplatSSBO(s:SplatSSBO) -> Splat {
    let u1 = unpack2x16float(s.cov3d_packed.x);
    let u2 = unpack2x16float(s.cov3d_packed.y);
    let u3 = unpack2x16float(s.cov3d_packed.z);
    let Vrk = mat3x3f(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);
    return Splat(
        vec4f(s.center, 1.0),
        mat3x3f(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y),
        unpack4x8unorm(s.color_packed)
    );
}

@vertex
fn vertex_main(
    @builtin(vertex_index) vertexIndex: u32, //[0-3]
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    var output : VertexOutput;
    output.position = vec4(0.0, 0.0, 2.0, 1.0);

    let splatIdx = splatOrder[instanceIndex];
    let s = unpackSplatSSBO(splats[splatIdx]);

    let aPosition = quadPositions[vertexIndex];

    let mvp = u.proj * u.view;

    let cam:vec4f = u.view * s.center;
    let pos2d:vec4f = u.proj * cam;
    let clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) { return output; }

    let _J = mat3x3f(
        u.focal.x / cam.z, 0., -(u.focal.x * cam.x) / (cam.z * cam.z),
        0., -u.focal.y / cam.z, (u.focal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    let view3x3f = mat3x3f(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
    let _T = transpose(view3x3f) * _J;
    let cov2d_mat = transpose(_T) * s.cov3d * _T;
    let cov2d = vec3(cov2d_mat[0][0], cov2d_mat[1][1], cov2d_mat[0][1]);

    let mid = (cov2d.x + cov2d.y) / 2.0;
    let radius = length(vec2((cov2d.x - cov2d.y) / 2.0, cov2d.z));
    let lambda1 = mid + radius;
    let lambda2 = mid - radius;

    if(lambda2 < 0.0) { return output; }

    let diagonalVector = normalize(vec2(cov2d.z, lambda1 - cov2d.x));
    let majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector; //in pixel
    let minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    let vCenter = pos2d.xyz / pos2d.w; //[-1,1]
    let quadLen = 2.0;

    output.position = vec4(vCenter.xy + 2.0 * quadLen * (aPosition.x * majorAxis + aPosition.y * minorAxis) / u.viewport, 0.0, 1.0);
//    output.position = vec4f(pos2d.xy + aPosition * pos2d.w * 50.0 / u.viewport, pos2d.z, pos2d.w);
    output.vPosition = aPosition * quadLen;
    output.vColor = s.color;
    return output;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4f {
    let _A = dot(in.vPosition, in.vPosition);
    if (_A > 4.0) { discard; }
    let _B = exp(-_A) * in.vColor.a;
    return vec4(in.vColor.rgb * _B, _B);
}
