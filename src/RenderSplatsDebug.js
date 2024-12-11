import { RenderSplats } from "./RenderSplats.js";

/**
 * RenderSplat using 2 buffers:
 * - one for the html canvas (untouched)
 * - one for getting values from the shader
 * It uses new framebuffer, and copy (blit) the gl.COLOR_ATTACHMENT0 to the default framebuffer;
 * so the values of what we see (canvas) is different from the values we read (array)
 * Here a float (f32) out array value is used
 */
export class RenderSplatsDebug extends RenderSplats{
    constructor(gl) {
        super(gl)
        this.gl = gl
        this.initExtension()
        this.initFramebuffer()
    }

    initExtension() {
        const ext = this.gl.getExtension('EXT_color_buffer_float')
        if (!ext) console.error('Floating point textures `EXT_color_buffer_float` not supported')
    }

    initFramebuffer() {
        const gl = this.gl, w = gl.canvas.width, h = gl.canvas.height
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

        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1])

        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) console.error('Framebuffer is not complete')
    }

    draw(view, viewport, proj) {
        super.draw(view, viewport, proj)
        this.renderFramebuffer()
    }

    renderFramebuffer(){
        const gl = this.gl, w = gl.canvas.width, h = gl.canvas.height
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
}
