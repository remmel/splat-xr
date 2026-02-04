import { mat4, vec3 } from 'wgpu-matrix';
import { quitIfWebGPUNotAvailable } from './util';
import {RenderSplat} from "./RenderSplat.ts";
import {Interactions} from "../../web/src/Interactions.js"
import {Fps} from "./utils";

const canvas = document.querySelector('canvas') as HTMLCanvasElement;
// const enterVRButton = document.getElementById('enter-vr') as HTMLButtonElement;

const fps = new Fps(document.getElementById("fps"))

const adapter = await navigator.gpu?.requestAdapter({
    featureLevel: 'compatibility',
});

if(!adapter) throw new Error('adapter is null');
const device = await adapter.requestDevice({
    requiredLimits: {
        maxStorageBuffersInVertexStage: adapter.limits.maxStorageBuffersInVertexStage,
    },
});
quitIfWebGPUNotAvailable(adapter, device);
console.log(adapter.info)

const context = canvas.getContext('webgpu') as GPUCanvasContext;

const w = 1000, h = 1000, fx=1000, fy = 1000;

canvas.width = w //canvas.clientWidth * window.devicePixelRatio;
canvas.height = h //canvas.clientHeight * window.devicePixelRatio;
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
const depthFormat: GPUTextureFormat = 'depth24plus';

context.configure({device, format: presentationFormat});

const depthTexture = device.createTexture({
    size: [w, h],
    format: depthFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [{
        view: undefined, // Will be set in frame()
        clearValue: [0.0, 0.0, 0.0, 0.0],
        loadOp: 'clear',
        storeOp: 'store',
    },],
    depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
    },
};

const renderSplat = new RenderSplat(device, presentationFormat, depthFormat);
// const url = 'ds/tmp/gs_garden_mipnerf360_vr.splat'
// const url = 'ds/tmp/gs_Emma_26fev_converted_by_kwok.splat'
const url = 'ds/axis.splat'
renderSplat.fetch(url)


const viewport = {w, h} //{w:canvas.width, h:canvas.height};
// const aspect = canvas.width / canvas.height;
// const proj = mat4.perspective((2 * Math.PI) / 5, aspect, 1, 100.0);

const proj = mat4.create(
    2, 0, 0, 0,
    0, -2, 0, 0,
    0, 0, 1.0010010010010009, 1,
    0, 0, -0.20020020020020018, 0
)

let model = mat4.identity();

const view = mat4.create(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0.1, 3, 1
)

let interactions = new Interactions(view, canvas)

// Standard rendering loop (non-VR)
function frame() {
    fps.log(true, false)
    const colorAttachment = renderPassDescriptor.colorAttachments[0] as GPURenderPassColorAttachment;
    colorAttachment.view = context.getCurrentTexture().createView();
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    renderSplat.draw(passEncoder, model, interactions.viewMatrix, proj, viewport);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(frame);
}

// Start regular rendering
requestAnimationFrame(frame);
