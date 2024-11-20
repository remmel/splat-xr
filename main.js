// import {fragmentShaderSourcePoint as fragmentShaderSource, vertexShaderSourcePoint as vertexShaderSource} from "./shadersPoints.js"; //pointcloud
import {fragmentShaderSource, vertexShaderSource} from "./shadersQuads.js";
import {createProgram, invert4, multiply4, rotate4} from "./utils.js";

let camera = {fy: 1150, fx: 1150}

function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.2;
    const zfar = 200;
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, -(2 * fy) / height, 0, 0],
        [0, 0, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

async function main() {


    let wbuffer;
    // 6*4 + 4 + 4 = 8*4
    // XYZ - Position (Float32)
    // XYZ - Scale (Float32)
    // RGBA - colors (uint8)
    // IJKL - quaternion/rot (uint8)
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    let lastProj = [];
    let depthIndex = new Uint32Array();
    let lastVertexCount = 0;

    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);

    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];

        var sign = (f >> 31) & 0x0001;
        var exp = (f >> 23) & 0x00ff;
        var frac = f & 0x007fffff;

        var newExp;
        if (exp == 0) {
            newExp = 0;
        } else if (exp < 113) {
            newExp = 0;
            frac |= 0x00800000;
            frac = frac >> (113 - exp);
            if (frac & 0x01000000) {
                newExp = 1;
                frac = 0;
            }
        } else if (exp < 142) {
            newExp = exp - 112;
        } else {
            newExp = 31;
            frac = 0;
        }

        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    function packHalf2x16(x, y) {
        return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
    }

    // pack buffer data into texture
    function generateTexture() {
        if (!wbuffer) return;
        const buffer_f32 = new Float32Array(wbuffer);
        const buffer_u8 = new Uint8Array(wbuffer);
        const buffer_u32 = new Uint32Array(wbuffer);

        var texwidth = 1024 * 2; // Set to your desired width
        var texheight = Math.ceil((2 * vertexCount) / texwidth); // Set to your desired height
        console.log("w", texwidth, "h", texheight, "texture units", texwidth*texheight*4 / 8, "nb splats", buffer_u32.length / 8, "lost", texwidth*texheight*4 / 8 - buffer_u32.length / 8)
        var texdata_u32 = new Uint32Array(texwidth * texheight * 4); // 4 components per pixel (RGBA) => RGBA32UI (4x32b)
        var texdata_u8 = new Uint8Array(texdata_u32.buffer);
        var texdata_f32 = new Float32Array(texdata_u32.buffer);

        // Here we convert from a .splat file buffer into a texture
        // With a little bit more foresight perhaps this texture file
        // should have been the native format as it'd be very easy to
        // load it into webgl.
        for (let i = 0; i < vertexCount; i++) {
            // x, y, z - Float32 - 3x4Bytes <=> 3x32b
            texdata_f32[8 * i + 0] = buffer_f32[8 * i + 0];
            texdata_f32[8 * i + 1] = buffer_f32[8 * i + 1];
            texdata_f32[8 * i + 2] = buffer_f32[8 * i + 2];

            // r, g, b, a - uint8 - 4*1B <=> 4x8b
            // texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
            texdata_u8[32 * i + 28 + 0] = buffer_u8[32 * i + 24 + 0];
            texdata_u8[32 * i + 28 + 1] = buffer_u8[32 * i + 24 + 1];
            texdata_u8[32 * i + 28 + 2] = buffer_u8[32 * i + 24 + 2];
            texdata_u8[32 * i + 28 + 3] = buffer_u8[32 * i + 24 + 3];

            // quaternions
            let scale = [
                buffer_f32[8 * i + 3 + 0],
                buffer_f32[8 * i + 3 + 1],
                buffer_f32[8 * i + 3 + 2],
            ];
            let rot = [
                (buffer_u8[32 * i + 28 + 0] - 128) / 128,
                (buffer_u8[32 * i + 28 + 1] - 128) / 128,
                (buffer_u8[32 * i + 28 + 2] - 128) / 128,
                (buffer_u8[32 * i + 28 + 3] - 128) / 128,
            ];

            // Compute the matrix product of S and R (M = S * R)
            const M = [
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),

                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),

                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
            ].map((k, i) => k * scale[Math.floor(i / 3)]);

            const sigma = [
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
            ];

            //uint32 - 3x4B <=>3x32b
            texdata_u32[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
            texdata_u32[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
            texdata_u32[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
        }

        setGpuTexturedata(texdata_u32, texwidth, texheight)
    }

    function runSort(viewProj) {
        if (!wbuffer) return;
        const buffer_f32 = new Float32Array(wbuffer);
        if (lastVertexCount === vertexCount) {
            let dot =
                lastProj[2] * viewProj[2] +
                lastProj[6] * viewProj[6] +
                lastProj[10] * viewProj[10];
            if (Math.abs(dot - 1) < 0.1) {
                return;
            }
        } else {
            generateTexture();
            lastVertexCount = vertexCount;
        }

        console.time("sort");
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) {
            let depth =
                ((viewProj[2] * buffer_f32[8 * i + 0] +
                        viewProj[6] * buffer_f32[8 * i + 1] +
                        viewProj[10] * buffer_f32[8 * i + 2]) *
                    4096) |
                0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }

        // This is a 16 bit single-pass counting sort
        let depthInv = (256 * 256) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);
        for (let i = 0; i < vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        depthIndex = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i;

        console.timeEnd("sort");

        lastProj = viewProj;

        setGpuDepthIndex(depthIndex)
    }

let worldTransform = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 3, 1
];

let viewMatrix = worldTransform;



    // const url = 'dataset/train.splat'
    const url = 'dataset/gs_Emma_26fev_converted_by_kwok.splat'
    // const url = new URLSearchParams(location.search).get("url") ?? 'https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat'

    let xrSession = null;
    let xrReferenceSpace = null;

    const req = await fetch(url, {
        mode: "cors", // no-cors, *cors, same-origin
        credentials: "omit", // include, *same-origin, omit
    });
    console.log(req);
    if (req.status != 200)
        throw new Error(req.status + " Unable to load " + req.url);

    const downsample = 1

    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");

    let projectionMatrix;

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });

    const program = createProgram(gl, vertexShaderSource, fragmentShaderSource)
    gl.useProgram(program);

    gl.disable(gl.DEPTH_TEST); // Disable depth testing

    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    const u_projection = gl.getUniformLocation(program, "uProj");
    const u_viewport = gl.getUniformLocation(program, "uViewport");
    const u_focal = gl.getUniformLocation(program, "uFocal");
    const u_view = gl.getUniformLocation(program, "uView");

    // positions
    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "aPosition");
    gl.enableVertexAttribArray(a_position);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    var u_texture = gl.getUniformLocation(program, "uTexture");
    gl.uniform1i(u_texture, 0);

    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "aIndex");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0);
    gl.vertexAttribDivisor(a_index, 1);

    const resize = () => {
        gl.uniform2fv(u_focal, new Float32Array([camera.fx, camera.fy]));
        projectionMatrix = getProjectionMatrix(camera.fx, camera.fy, innerWidth, innerHeight);
        gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));
        gl.canvas.width = Math.round(innerWidth / downsample);
        gl.canvas.height = Math.round(innerHeight / downsample);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
    };

    window.addEventListener("resize", resize);
    resize();


    const setGpuTexturedata = (texdata, texwidth, texheight) => { //conversion from data to texture finished
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, texwidth, texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
    }

    const setGpuDepthIndex = (depthIndex) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
    }


    let vertexCount = 0;

    let lastFrame = 0;
    let avgFps = 0;

    const onFrame = (now) => {
        const viewProj = multiply4(projectionMatrix, viewMatrix);
        runSort(viewProj)


        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        if (vertexCount > 0) {
            document.getElementById("spinner").style.display = "none";
            gl.uniformMatrix4fv(u_view, false, viewMatrix);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
            // gl.drawArraysInstanced(gl.POINTS, 0, 1, vertexCount) //pointcloud
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            document.getElementById("spinner").style.display = "";
        }

        fps.innerText = Math.round(avgFps) + " fps";
        lastFrame = now;
        requestAnimationFrame(onFrame);
    };

    let worldXRTransform = [
        -1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, 1, 0,
        2, -1, 4, 1
    ];
    worldXRTransform = rotate4(invert4(worldXRTransform), -90 * Math.PI / 180, 0, 1, 0);

    function onXRFrame(time, frame) {
        const session = frame.session;
        session.requestAnimationFrame(onXRFrame);

        const pose = frame.getViewerPose(xrReferenceSpace);
        if (!pose) return

        const glLayer = session.renderState.baseLayer;

        gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);


        for (const view of pose.views) {
            const viewport = glLayer.getViewport(view);
            gl.viewport(viewport.x * downsample, viewport.y * downsample, viewport.width * downsample, viewport.height * downsample);
            let projectionMatrix = view.projectionMatrix;
            gl.uniform2fv(u_viewport, new Float32Array([viewport.width, viewport.height]));

            // Update projection and view matrices based on VR pose
            gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
            gl.uniformMatrix4fv(u_view, false, view.transform.inverse.matrix);
            gl.uniform2fv(u_focal, new Float32Array([(projectionMatrix[0] * viewport.width) / 2, -(projectionMatrix[5] * viewport.height) / 2]));

            const transformedView = multiply4(view.transform.inverse.matrix, worldXRTransform);
            gl.uniformMatrix4fv(u_view, false, transformedView);

            // Render the scene
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
            // gl.drawArraysInstanced(gl.POINTS, 0, 1, vertexCount) //pointcloud

            const viewProj = multiply4(projectionMatrix, transformedView);
            // worker.postMessage({ view: viewProj });
            runSort(viewProj)
        }
    }



    const splatData = new Uint8Array(await req.arrayBuffer())
    vertexCount = splatData.length / rowLength
    console.log(vertexCount, downsample);
    wbuffer = splatData.buffer

    onFrame();

    async function startXR() {
        xrSession = await navigator.xr.requestSession('immersive-vr', {optionalFeatures: ['local-floor']})
        xrSession.addEventListener('end', onXRSessionEnded);
        xrReferenceSpace = await xrSession.requestReferenceSpace('local-floor');
        await gl.makeXRCompatible();
        xrSession.updateRenderState({
            baseLayer: new XRWebGLLayer(xrSession, gl, {
                // framebufferScaleFactor: 0.75,
                fixedFoveation: 1.0,
                // antialias: true,
                // depth: true,
                // ignoreDepthValues: true
                // foveationLevel
            })
        });
        xrSession.requestAnimationFrame(onXRFrame);
    }

    function onXRSessionEnded() {
        xrSession = null;
    }

    if (navigator.xr) {
        navigator.xr.isSessionSupported('immersive-vr').then(supported => {
            if (supported) {
                document.getElementById("enter-vr").disabled = false
                document.getElementById('enter-vr').addEventListener('click', () => {
                    startXR();
                });
            }
        });
    }

}

main()