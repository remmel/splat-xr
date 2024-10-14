import {fragmentShaderSourcePoint, vertexShaderSourcePoint} from "./shadersPoints.js";
import {fragmentShaderSourceQuads, vertexShaderSourceQuads} from "./shadersQuads.js";

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

function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

let handleSorted = null
let handlePacked = null


    let wbuffer;
    let wvertexCount = 0;
    let wviewProj;
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
        var texheight = Math.ceil((2 * wvertexCount) / texwidth); // Set to your desired height
        console.log("w", texwidth, "h", texheight, "texture units", texwidth*texheight*4 / 8, "nb splats", buffer_u32.length / 8, "lost", texwidth*texheight*4 / 8 - buffer_u32.length / 8)
        var texdata_u32 = new Uint32Array(texwidth * texheight * 4); // 4 components per pixel (RGBA) => RGBA32UI (4x32b)
        var texdata_u8 = new Uint8Array(texdata_u32.buffer);
        var texdata_f32 = new Float32Array(texdata_u32.buffer);

        // Here we convert from a .splat file buffer into a texture
        // With a little bit more foresight perhaps this texture file
        // should have been the native format as it'd be very easy to
        // load it into webgl.
        for (let i = 0; i < wvertexCount; i++) {
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

        handlePacked(texdata_u32, texwidth, texheight)
    }

    function runSort(viewProj) {
        if (!wbuffer) return;
        const buffer_f32 = new Float32Array(wbuffer);
        if (lastVertexCount === wvertexCount) {
            let dot =
                lastProj[2] * viewProj[2] +
                lastProj[6] * viewProj[6] +
                lastProj[10] * viewProj[10];
            if (Math.abs(dot - 1) < 0.01) {
                return;
            }
        } else {
            generateTexture();
            lastVertexCount = wvertexCount;
        }

        console.time("sort");
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(wvertexCount);
        for (let i = 0; i < wvertexCount; i++) {
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
        for (let i = 0; i < wvertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        depthIndex = new Uint32Array(wvertexCount);
        for (let i = 0; i < wvertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i;

        console.timeEnd("sort");

        lastProj = viewProj;

        handleSorted(depthIndex, wviewProj, wvertexCount)
    }

    function wSetData(_buffer, _vertexCount) {
        wbuffer = _buffer
        wvertexCount = _vertexCount
    }

    function wSort(_viewProj) {
        wviewProj = _viewProj
        runSort(wviewProj)
    }

const vertexShaderSource = vertexShaderSourceQuads

const fragmentShaderSource = fragmentShaderSourceQuads

let worldTransform = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 3, 1
];

let viewMatrix = worldTransform;

async function main() {

    // const url = 'https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat'
    const url = 'dataset/train/train.splat'

    const req = await fetch(url, {
        mode: "cors", // no-cors, *cors, same-origin
        credentials: "omit", // include, *same-origin, omit
    });
    console.log(req);
    if (req.status != 200)
        throw new Error(req.status + " Unable to load " + req.url);

    const rowLength = 3 * 4 + 3 * 4 + 4 + 4; //34

    const downsample = 1

    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");

    let projectionMatrix;

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(vertexShader));

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(fragmentShader));

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.error(gl.getProgramInfoLog(program));

    gl.disable(gl.DEPTH_TEST); // Disable depth testing

    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");

    // positions
    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    var u_textureLocation = gl.getUniformLocation(program, "u_texture");
    gl.uniform1i(u_textureLocation, 0);

    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
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


    handlePacked = (texdata, texwidth, texheight) => { //conversion from data to texture finished
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, texwidth, texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
    }

    handleSorted = (depthIndex, wviewProj, wvertexCount) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
        vertexCount = wvertexCount
    }


    let vertexCount = 0;

    let lastFrame = 0;
    let avgFps = 0;

    const frame = (now) => {
        const viewProj = multiply4(projectionMatrix, viewMatrix);
        wSort(viewProj)


        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        if (vertexCount > 0) {
            document.getElementById("spinner").style.display = "none";
            gl.uniformMatrix4fv(u_view, false, viewMatrix);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
            // gl.drawArraysInstanced(gl.POINTS, 0, 1, vertexCount)
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            document.getElementById("spinner").style.display = "";
        }

        fps.innerText = Math.round(avgFps) + " fps";
        lastFrame = now;
        requestAnimationFrame(frame);
    };



    const splatData = new Uint8Array(await req.arrayBuffer())
    vertexCount = splatData.length / rowLength
    console.log(vertexCount, downsample);
    wSetData(splatData.buffer, vertexCount)

    frame();

}

main()