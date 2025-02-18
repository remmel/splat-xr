#version 450

layout(location = 0) in vec4 gColor;

layout(location = 0) out vec4 FragColor;

void main() {
  FragColor = gColor.rgba;
}