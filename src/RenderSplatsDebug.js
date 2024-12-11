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

        // canvas color texture
        const textureColor = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, textureColor)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, textureColor, 0)

        // debug texture
        const textureDebug = gl.createTexture()
        gl.bindTexture(gl.TEXTURE_2D, textureDebug)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, null)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, textureDebug, 0)

        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1])

        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) console.error('Framebuffer is not complete')
    }

    draw(view, viewport, proj) {
        super.draw(view, viewport, proj)
        this.renderFramebuffer()
    }

    renderFramebuffer(){
        const gl = this.gl, w = gl.canvas.width, h = gl.canvas.height

        // attachement0 referring to `fragColor` / canvas
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


        // attachment1 referring to `fragDebug`
        gl.readBuffer(gl.COLOR_ATTACHMENT1)

        const pixel_1_f32 = new Float32Array(1)
        gl.readPixels(w/2, h/2, 1, 1, gl.RED, gl.FLOAT, pixel_1_f32)
        console.log(`COLOR_ATTACHMENT1:`, pixel_1_f32);

        const pixels_1_f32 = new Float32Array(w * h)
        gl.readPixels(0, 0, w, h, gl.RED, gl.FLOAT, pixels_1_f32);
        let maxR = maxArray(pixels_1_f32)
        console.log('Max number of time a single pixel has been updated (layers):', maxR)
    }
}

function maxArray(arr) {
    //handle huge array and avoiding `Uncaught (in promise) RangeError: Maximum call stack size exceeded`
    let maxVal = -Infinity
    for(let i = 0; i < arr.length; i++) maxVal = Math.max(maxVal, arr[i])
    return maxVal
}
