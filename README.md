# splat

The objective of that repository is for learning purposes duplicate the antimatter webgl viewer.
Implementations using:
- Opengl (using geometry shader)
- python numpy
- python loop


To run the js webgl viewer:
```shell
npm install
npm run dev
```

To run the python viewers:
```shell
cd python
conda env create -f environment.yml
conda activate splat-render
python main.py
```

You might need to
- call with prefix `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python main.py` or
- add as environment variables `__NV_PRIME_RENDER_OFFLOAD=1;__GLX_VENDOR_LIBRARY_NAME=nvidia`


# Technicals explanations

## Benchmarking pre-multiplying and blending order
How can we explain that pre-multiplying the rgb * a in the fragment boost the fps? Test done on my iGPU, webgl, conic.
  - `gl.blendFunc(gl.ONE_MINUS_DST_ALPHA, gl.ONE)` - front to back - 45fps - `fragColor = vec4(vColor.rgb * a, a)` - `dst.rgba = src.rgba * (1-dst.a) + dst.rgba * 1`
  - `gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)` - back to front - 38fps - `fragColor = vec4(vColor.rgb * a, a)` - `dst.rgba = src.rgba * 1 + dst.rgba * (1-src.a)`
  - `gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)` - back to front - 34fps - `fragColor = vec4(vColor.rgb, a)` - `dst.rgba = src.rgba * src.a + dst.rgba * (1-src.a)`
  - as a reference the antimatter version (not conic) - 53fps
