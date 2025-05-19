Migration of the antimatter webgl splat viewer into webgpu on my iGPU  
⚠️ -35% perf on MQ3 web browser (flat, no XR) vs webgl

|           | WebGL | WebGPU |
|-----------|-------|--------|
| iGPU      | 32    | 29     |
| RTX 3060m | 120   | 120    |
| MQ3 Flat  | 39    | 25     |
| MQ3 VR    | 11    | n/a    |


TODOs
1. Upload index instead of ordered splats data
1. Add xr support when it will be supported in Meta Quest Browser  
https://immersive-web.github.io/webxr-samples/webgpu/  
https://github.com/immersive-web/webxr-samples/blob/main/webgpu/vr-barebones.html