import { RenderSplats } from "./RenderSplats.js";
import { RenderSplatsDebug } from "./RenderSplatsDebug.js";
import { animateCarrouselMouvement, Fps, getProjectionMatrix, invert4, multiply4, rotate4 } from "./utils.js";

async function main() {

    let worldTransform = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 3, 1
    ]

    let view = worldTransform



    // const url = 'dataset/train.splat'
    const url = 'tmp/gs_Emma_26fev_converted_by_kwok.splat'
    // const url = 'tmp/gs_Emma_26fev_low.splat'
    // const url = new URLSearchParams(location.search).get("url") ?? 'https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat'

    let xrSession = null;
    let xrReferenceSpace = null;

    const canvas = document.getElementById("canvas");
    const fps = new Fps(document.getElementById("fps"))

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });

    // console.log("canvas size before", gl.canvas.width, gl.canvas.height) //why is it 300x150?!?
    const w = 1000, h = 1000, fx=1000, fy = 1000 //for benchmark purposes
    // const ds = 1 //downscale
    // const w = innerWidth / ds, h = innerHeight / ds, fx = 1150 / ds, fy = 1150 / ds
    gl.canvas.width = w
    gl.canvas.height = h
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)

    const renderSplats = new RenderSplatsDebug(gl)
    await renderSplats.fetch(url)
    document.getElementById("spinner").style.display = "none"

    const onFrame = (now) => {
        const viewport = {width: w, height: h};
        let proj = getProjectionMatrix(fx, fy, w, h)

        fps.log(true, false)
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        // const view = animateCarrouselMouvement(worldTransform)

        renderSplats.draw(view, viewport, proj)

        // requestAnimationFrame(onFrame);
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

        for (const xrview of pose.views) {
            const xrviewport = glLayer.getViewport(xrview); // {x: 0, y: 0, width: 1680, height: 1760}
            gl.viewport(xrviewport.x, xrviewport.y , xrviewport.width, xrviewport.height)
            const proj = xrview.projectionMatrix
            const view = multiply4(xrview.transform.inverse.matrix, worldXRTransform)
            renderSplats.draw(view, xrviewport, proj)
        }
    }

    onFrame();

    async function startXR() {
        xrSession = await navigator.xr.requestSession('immersive-vr', {optionalFeatures: ['local-floor']})
        xrSession.addEventListener('end', onXRSessionEnded);
        xrReferenceSpace = await xrSession.requestReferenceSpace('local-floor');
        await gl.makeXRCompatible();
        xrSession.updateRenderState({
            baseLayer: new XRWebGLLayer(xrSession, gl, {
                // framebufferScaleFactor: 0.50, //similar as downscaling the viewport?
                // fixedFoveation: 1.0,
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
                document.getElementById('enter-vr').addEventListener('click', () => startXR())
            }
        });
    }

}

main()
