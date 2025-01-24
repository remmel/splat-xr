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
python main.py
```

You might need to
- call with prefix `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python main.py` or
- add as environment variables `__NV_PRIME_RENDER_OFFLOAD=1;__GLX_VENDOR_LIBRARY_NAME=nvidia`