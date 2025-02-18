#version 450

layout(location = 0) in vec4 gColor;
layout(location = 1) in vec2 gPosition;

layout(location = 0) out vec4 fragColor;

void main() {
  float A = -dot(gPosition, gPosition);
  if (A < -4.0) discard;
  float B = exp(A) * gColor.a;
  fragColor = vec4(B * gColor.rgb, B);
}