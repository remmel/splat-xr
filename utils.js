function getViewMatrix(camera) {
    const R = camera.rotation.flat();
    const t = camera.position;
    const camToWorld = [
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [
            -t[0] * R[0] - t[1] * R[3] - t[2] * R[6],
            -t[0] * R[1] - t[1] * R[4] - t[2] * R[7],
            -t[0] * R[2] - t[1] * R[5] - t[2] * R[8],
            1,
        ],
    ].flat();
    return camToWorld;
}

export function multiply4(a, b) {
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

export function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

export function rotate4(a, rad, x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16),
    ];
}

export function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

export function getPerspectiveRH_ZO(fx, fy, width, height, znear = 0.2, zfar = 200) {
    return [
        (2 * fx) / width, 0, 0, 0,
        0, (2 * fy) / height, 0, 0,
        0, 0, zfar/(znear-zfar), -1,
        0, 0, -(zfar*znear)/(zfar-znear), 0
    ];
}

export function getPerspectiveRH_NO(fx, fy, width, height, znear = 0.2, zfar = 200) {
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, (2 * fy) / height, 0, 0],
        [0, 0, -(zfar + znear)/(zfar - znear), -1],
        [0, 0, -(2 * zfar * znear)/(zfar - znear), 0]
    ].flat();
}

/**
 * Original getProjectionMatrix, is getPerspectiveLH_ZO with flipped Y.
 * We should use instead getPerspectiveRH_NO :
 * - RH (Right Hand)
 * - NO (Negative One to One) [-1;1]
 */
export function getProjectionMatrix(fx, fy, width, height) {
    let proj = getPerspectiveLH_ZO(fx, fy, width, height)
    proj[5] *= -1 //flip Y
    return proj
}

export function getPerspectiveLH_ZO(fx, fy, width, height, znear = 0.2, zfar = 200) {
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, (2 * fy) / height, 0, 0],
        [0, 0, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

export function getPerspectiveLH_NO(fx, fy, width, height, znear = 0.2, zfar = 200) {
    return [
        (2 * fx) / width, 0, 0, 0,
        0, (2 * fy) / height, 0, 0,
        0, 0, (zfar + znear) / (zfar - znear), 1,
        0, 0, -(2 * zfar * znear) / (zfar - znear), 0
    ];
}

export function animateCarrouselMouvement(_view) {
    // carrousel movement
    let inv = invert4(_view)
    const t = Math.sin(Date.now() / 1000)
    inv = translate4(inv, .5 * t, 0, 0.5 * (1 - Math.cos(t)))
    inv = rotate4(inv, -0.1 * t, 0, 1, 0)
    const view = invert4(inv)
    return view
}

// carrousel movement
// let inv = invert4(worldTransform) //defaultViewMatrix);
// const t = Math.sin(Date.now() / 1000);
// inv = translate4(inv, .5 * t, 0, 0.5 * (1 - Math.cos(t)));
// inv = rotate4(inv, -0.1 * t, 0, 1, 0);
// view = invert4(inv);


function compileShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const error = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error(`Shader compilation failed: ${error}`);
    }

    return shader;
}

export function createProgram(gl, vertexShaderSource, fragmentShaderSource) {
    let vertexShader, fragmentShader;

    try {
        vertexShader = compileShader(gl, vertexShaderSource, gl.VERTEX_SHADER);
        fragmentShader = compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER);

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const error = gl.getProgramInfoLog(program);
            gl.deleteProgram(program);
            throw new Error(`Program linking failed: ${error}`);
        }

        // Clean up shaders
        gl.deleteShader(vertexShader);
        gl.deleteShader(fragmentShader);

        return program;
    } catch (error) {
        if (vertexShader) gl.deleteShader(vertexShader);
        if (fragmentShader) gl.deleteShader(fragmentShader);
        throw error;
    }
}
export class Fps {
    constructor(el) {
        this.el = el
        this.lastLog = 0 //in ms
        this.nbFrames = 0 //nb of frames since lastLog
    }

    /**
     * Log in htmlElement or in console.log the framerate.
     * Must be called at every frame in order to also count the number of frames
     * It will write only every 1s
     */
    log(inElement = true, inConsole = true) { //log every seconds
        this.nbFrames++
        const now = performance.now()
        if(now - this.lastLog > 1000) { //1s has been elapsed
            if(inConsole)
                console.log(this.nbFrames + ' fps')
            if(inElement)
                this.el.innerText = this.nbFrames + ' fps'
            this.lastLog = now
            this.nbFrames = 0
        }
    }
}
