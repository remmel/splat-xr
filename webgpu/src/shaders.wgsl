struct Uniforms {
    mvp : mat4x4f,
    viewport : vec2f,
}
@binding(0) @group(0) var<uniform> u : Uniforms;

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) fragPosition: vec4f,
    @location(1) fragColor: vec4f,
}

// Quad vertex positions for a triangle strip
// Arranged as a unit quad centered at origin
var<private> quadPositions: array<vec2f, 4> = array<vec2f, 4>(
    vec2f(-0.5, -0.5),
    vec2f(0.5, -0.5),
    vec2f(-0.5, 0.5),
    vec2f(0.5, 0.5)
);

@vertex
fn vertex_main(
    @location(0) center : vec4f, //splat position, do not confuse with position of the vertex
    @location(1) color : vec4f,
    @builtin(vertex_index) vertexIndex: u32, //[0-3]
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    var output : VertexOutput;

    let quadOffset = quadPositions[vertexIndex];

    var center2d:vec4f = u.mvp * center;

    let pointSize = 50.0;
    let ndcSize = pointSize / u.viewport;
    let offsetXY = quadOffset * ndcSize * center2d.w;

    output.position = vec4f(center2d.xy + offsetXY.xy, center2d.z, center2d.w);
    output.fragPosition = 0.5 * (center + vec4(1.0, 1.0, 1.0, 1.0));
    output.fragColor = color;
    return output;
}

@fragment
fn fragment_main(
    @location(0) fragPosition: vec4f,
    @location(1) fragColor: vec4f
) -> @location(0) vec4f {
    return fragColor;
}
