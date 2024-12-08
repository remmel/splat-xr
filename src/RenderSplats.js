// import {fragmentShaderSourcePoint as fragmentShaderSource, vertexShaderSourcePoint as vertexShaderSource} from "./shadersPoints.js"; //pointcloud
import { fragmentShaderSource, vertexShaderSource } from "./shadersQuads.js"
import { createProgram, multiply4, packHalf2x16 } from "./utils.js"

export class RenderSplats {

    vertexCount = 0
    lastVertexCount = 0

    /** @var ArrayBuffer contains the splats data fetched */
    splatsBuffer = null
    constructor(gl) {

        this.vao = gl.createVertexArray()
        gl.bindVertexArray(this.vao)

        const program = this.program = createProgram(gl, vertexShaderSource, fragmentShaderSource)
        // gl.useProgram(program)

        this.uProjLoc = gl.getUniformLocation(program, "uProj")
        this.uViewportLoc = gl.getUniformLocation(program, "uViewport")
        this.uFocalLoc = gl.getUniformLocation(program, "uFocal")
        this.uViewLoc = gl.getUniformLocation(program, "uView")

        // positions
        const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2])
        const vertexBuffer = gl.createBuffer()
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
        gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW)
        const aPositionLoc = gl.getAttribLocation(program, "aPosition")
        gl.enableVertexAttribArray(aPositionLoc)
        gl.vertexAttribPointer(aPositionLoc, 2, gl.FLOAT, false, 0, 0)

        this.texture = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, this.texture)
        // const uTextureLoc = gl.getUniformLocation(program, "uTexture")
        // gl.uniform1i(uTextureLoc, 0)

        this.indexBuffer = gl.createBuffer()
        const aIndexLoc = gl.getAttribLocation(program, "aIndex")
        gl.enableVertexAttribArray(aIndexLoc)
        gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer)
        gl.vertexAttribIPointer(aIndexLoc, 1, gl.INT, false, 0)
        gl.vertexAttribDivisor(aIndexLoc, 1)

        this.gl = gl

        const w = gl.canvas.width, h = gl.canvas.height
        this.fbo = gl.createFramebuffer()
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo)

        const textureColor0 = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, textureColor0)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, textureColor0, 0)

        // gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, w, h, 0); //TO DO WHAT?

        const textureCount = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, textureCount)
        // gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null) // default

        // using `fragCount` KO: blending seems to be  ignored (color OK)
        // gl.texImage2D(gl.TEXTURE_2D, 0,
        //     gl.R32I, //internalformat
        //     w, h, 0,
        //     gl.RED_INTEGER,//format
        //     gl.INT, //type
        //     null);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, null)

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, textureCount, 0)

        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    }

    async fetch(url) {
        const req = await fetch(url, {
            mode: "cors", // no-cors, *cors, same-origin
            credentials: "omit", // include, *same-origin, omit
        })
        console.log(req)
        if (req.status != 200)
            throw new Error(req.status + " Unable to load " + req.url)

        const splatData = new Uint8Array(await req.arrayBuffer())
        // 6*4 + 4 + 4 = 8*4
        // XYZ - Position (Float32)
        // XYZ - Scale (Float32)
        // RGBA - colors (uint8)
        // IJKL - quaternion/rot (uint8)
        const rowLength = 3 * 4 + 3 * 4 + 4 + 4
        this.vertexCount = splatData.length / rowLength
        console.log('vertexCount', this.vertexCount)
        console.log(splatData.buffer)
        this.splatsBuffer = splatData.buffer

        this.generateTexture()

    }

    // pack buffer data into texture
    generateTexture() {
        const buffer_f32 = new Float32Array(this.splatsBuffer)
        const buffer_u8 = new Uint8Array(this.splatsBuffer)
        const buffer_u32 = new Uint32Array(this.splatsBuffer)

        var texwidth = 1024 * 2 // Set to your desired width
        var texheight = Math.ceil((2 * this.vertexCount) / texwidth) // Set to your desired height
        console.log("w", texwidth, "h", texheight, "texture units", texwidth*texheight*4 / 8, "nb splats", buffer_u32.length / 8, "lost", texwidth*texheight*4 / 8 - buffer_u32.length / 8)
        var texdata_u32 = new Uint32Array(texwidth * texheight * 4) // 4 components per pixel (RGBA) => RGBA32UI (4x32b)
        var texdata_u8 = new Uint8Array(texdata_u32.buffer)
        var texdata_f32 = new Float32Array(texdata_u32.buffer)

        // Here we convert from a .splat file buffer into a texture
        // With a little bit more foresight perhaps this texture file
        // should have been the native format as it'd be very easy to
        // load it into webgl.
        for (let i = 0; i < this.vertexCount; i++) {
            // x, y, z - Float32 - 3x4Bytes <=> 3x32b
            texdata_f32[8 * i + 0] = buffer_f32[8 * i + 0]
            texdata_f32[8 * i + 1] = buffer_f32[8 * i + 1]
            texdata_f32[8 * i + 2] = buffer_f32[8 * i + 2]

            // r, g, b, a - uint8 - 4*1B <=> 4x8b
            texdata_u32[8*i+7] = buffer_u32[8 * i + 6]

            // quaternions
            let scale = [
                buffer_f32[8 * i + 3 + 0],
                buffer_f32[8 * i + 3 + 1],
                buffer_f32[8 * i + 3 + 2],
            ]
            let rot = [
                (buffer_u8[32 * i + 28 + 0] - 128) / 128,
                (buffer_u8[32 * i + 28 + 1] - 128) / 128,
                (buffer_u8[32 * i + 28 + 2] - 128) / 128,
                (buffer_u8[32 * i + 28 + 3] - 128) / 128,
            ]

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
            ].map((k, i) => k * scale[Math.floor(i / 3)])

            const sigma = [
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
            ]

            //uint32 - 3x4B <=>3x32b
            texdata_u32[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1])
            texdata_u32[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3])
            texdata_u32[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5])
        }

        this.setGpuTexturedata(texdata_u32, texwidth, texheight)
    }

    setGpuTexturedata(texdata, texwidth, texheight) { //conversion from data to texture finished
        const gl = this.gl
        gl.bindTexture(gl.TEXTURE_2D, this.texture)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, texwidth, texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata)
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, this.texture)
    }

    setGpuDepthIndex(depthIndex) {
        const gl = this.gl
        gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer)
        gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW)
    }

    draw(view, viewport, proj) {
        const gl = this.gl
        const w = gl.canvas.width, h = gl.canvas.height

        gl.disable(gl.DEPTH_TEST)

        // Enable blending
        gl.enable(gl.BLEND)
        // gl.blendFunc(gl.ONE_MINUS_DST_ALPHA, gl.ONE) // antimatter
        // gl.blendFunc(gl.ONE, gl.ONE_MINUS_DST_ALPHA)
        gl.blendFunc(gl.ONE, gl.ONE)

        // gl.clearBufferfv(gl.COLOR, 0, [0.0, 0.0, 0.0, 0.0])
        // gl.clearBufferuiv(gl.COLOR, 1, new Uint32Array([0, 0, 0, 0]));


        gl.useProgram(this.program)
        gl.bindVertexArray(this.vao)
        gl.uniformMatrix4fv(this.uProjLoc, false, proj) //fixed
        gl.uniformMatrix4fv(this.uViewLoc, false, view) //fixed
        gl.uniform2fv(this.uViewportLoc, new Float32Array([viewport.width, viewport.height])) //fixed
        gl.uniform2fv(this.uFocalLoc, new Float32Array([
            (proj[0] * viewport.width) / 2,
            -(proj[5] * viewport.height) / 2
        ]))

        const viewProj = multiply4(proj, view)
        this.runSort(viewProj)
        gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, this.vertexCount)
        // gl.drawArraysInstanced(gl.POINTS, 0, 1, this.vertexCount) //pointcloud

        gl.readBuffer(gl.COLOR_ATTACHMENT0);
        const pixels = new Uint8Array(4);
        gl.readPixels(w/2, h/2, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        console.log(`COLOR_ATTACHMENT0:`, pixels);

        // Now blit to the canvas (default framebuffer), using attachment just read `gl.readBuffer`
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.fbo);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
        gl.blitFramebuffer(
            0, 0, w, h,  // source rectangle
            0, 0, w, h,  // destination rectangle
            gl.COLOR_BUFFER_BIT,               // mask
            gl.NEAREST                         // filter
        );


        // attachment1 referring to `fragCount`

        gl.readBuffer(gl.COLOR_ATTACHMENT1)

        const pixel_1_f32 = new Float32Array(1)
        gl.readPixels(w/2, h/2, 1, 1, gl.RED, gl.FLOAT, pixel_1_f32)
        console.log(`COLOR_ATTACHMENT1:`, pixel_1_f32);

        // const pixel_1_i32 = new Int32Array(1)
        // gl.readPixels(w/2, h/2, 1, 1, gl.RED_INTEGER, gl.INT, pixel_1_i32)
        // console.log(`COLOR_ATTACHMENT1:`, pixel_1_i32);

        const pixels_1_f32 = new Float32Array(w * h)
        gl.readPixels(0, 0, w, h, gl.RED, gl.FLOAT, pixels_1_f32);
        let maxR = -Infinity
        for(let i = 0; i < pixels_1_f32.length; i++) maxR = Math.max(maxR, pixels_1_f32[i])
        console.log('Max number of time a single pixel has been updated (layers):', maxR)
    }

    runSort(viewProj) {
        const [x, y, z] = [viewProj[2], viewProj[6], viewProj[10]]
        if (this.lastVertexCount === this.vertexCount) {
            const [lastX, lastY, lastZ] = [this.lastProj[2], this.lastProj[6], this.lastProj[10]]
            let dot = lastX * x + lastY * y + lastZ * z
            if (Math.abs(dot - 1) < 0.01) {
                return
            }
        } else {
            this.generateTexture()
            this.lastVertexCount = this.vertexCount
        }

        const depthIndex = this.sort(viewProj[2], viewProj[6], viewProj[10])
        this.lastProj = viewProj
        this.setGpuDepthIndex(depthIndex)
    }

    sort(x,y,z) {
        console.time("sort")
        const buffer_f32 = new Float32Array(this.splatsBuffer)
        let maxDepth = -Infinity
        let minDepth = Infinity
        let sizeList = new Int32Array(this.vertexCount)
        for (let i = 0; i < this.vertexCount; i++) {
            let depth = ((x * buffer_f32[8 * i + 0] + y * buffer_f32[8 * i + 1] + z * buffer_f32[8 * i + 2]) * 4096) | 0
            sizeList[i] = depth
            if (depth > maxDepth) maxDepth = depth
            if (depth < minDepth) minDepth = depth
        }

        // This is a 16 bit single-pass counting sort
        let depthInv = (256 * 256) / (maxDepth - minDepth)
        let counts0 = new Uint32Array(256 * 256)
        for (let i = 0; i < this.vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0
            counts0[sizeList[i]]++
        }
        let starts0 = new Uint32Array(256 * 256)
        for (let i = 1; i < 256 * 256; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1]
        let depthIndex = new Uint32Array(this.vertexCount)
        for (let i = 0; i < this.vertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i

        console.timeEnd("sort")
        return depthIndex
    }
}
