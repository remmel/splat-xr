struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) fragPosition: vec4f,
  @location(1) fragColor: vec4f,
}

@vertex
fn vertex_main(
  @location(0) position : vec4f,
  @location(1) color : vec4f
) -> VertexOutput {
  var output : VertexOutput;
  output.position = uniforms.modelViewProjectionMatrix * position;
  output.fragPosition = 0.5 * (position + vec4(1.0, 1.0, 1.0, 1.0));
  output.fragColor = color;
  return output;
}

@fragment
fn fragment_main(
  // fragUV input removed
  @location(0) fragPosition: vec4f,
  @location(1) fragColor: vec4f
) -> @location(0) vec4f {
  return fragColor;
}