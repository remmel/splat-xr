struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) fragPosition: vec4f,
}

@vertex
fn vertex_main(
  @location(0) position : vec4f,
  @location(1) uv : vec2f
) -> VertexOutput {
  var output : VertexOutput;
  output.position = uniforms.modelViewProjectionMatrix * position;
  output.fragUV = uv;
  // Preserving the original logic for fragPosition
  output.fragPosition = 0.5 * (position + vec4(1.0, 1.0, 1.0, 1.0));
  return output;
}

@fragment
fn fragment_main(
  @location(0) fragUV: vec2f,
  @location(1) fragPosition: vec4f
) -> @location(0) vec4f {
  return fragPosition;
}