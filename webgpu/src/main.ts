import { mat4, vec3 } from 'wgpu-matrix';
import type {Mat4Arg} from "wgpu-matrix";

import shadersWGSL from './shaders.wgsl?raw';
import { quitIfWebGPUNotAvailable } from './util';

const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const adapter = await navigator.gpu?.requestAdapter({
    featureLevel: 'compatibility',
});

if(!adapter) throw new Error('adapter is null');
const device = await adapter.requestDevice();
quitIfWebGPUNotAvailable(adapter, device);

const context = canvas.getContext('webgpu') as GPUCanvasContext;

const devicePixelRatio = window.devicePixelRatio;
canvas.width = canvas.clientWidth * devicePixelRatio;
canvas.height = canvas.clientHeight * devicePixelRatio;
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
const depthFormat: GPUTextureFormat = 'depth24plus';

context.configure({
    device,
    format: presentationFormat,
});

const UNIFORM_BUFFER_SIZE = 4 * 16 + 4 * 4; // 4x4 matrix + vec2f (aligned to 16 bytes), bytes

// RenderCube class
class RenderCube {
    private pipeline: GPURenderPipeline;
    private vertexBuffer: GPUBuffer;
    private count : number = 0; //number of splats (NOT number of vertices)
    private buffer: Float32Array = new Float32Array(0);
    // offsets in bytes
    private bufferOffsets = {
        'position': 0,
        'color': 4*4,
        'stride': 8*4
    }

    private uniformBuffer: GPUBuffer;
    private uniformBindGroup: GPUBindGroup;
    private mvpMatrix = mat4.create(); // Stores the final MVP for this cube
    private uniformData = new Float32Array(UNIFORM_BUFFER_SIZE / 4); // Float32Array to hold all uniform data

    constructor(
        device: GPUDevice,
        presentationFormat: GPUTextureFormat,
        depthFormat: GPUTextureFormat,
    ) {
        this.uniformBuffer = device.createBuffer({
            size: UNIFORM_BUFFER_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const uniformGroupLayout = device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: {type: 'uniform'},
            },],
        });

        this.uniformBindGroup = device.createBindGroup({
            layout: uniformGroupLayout,
            entries: [{
                binding: 0,
                resource: {buffer: this.uniformBuffer},
            },],
        });

        const shaderModule = device.createShaderModule({
            code: shadersWGSL,
        });

        this.pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [uniformGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vertex_main',
                buffers: [{
                    arrayStride: this.bufferOffsets.stride,
                    attributes: [
                        {shaderLocation: 0, offset: this.bufferOffsets.position, format: 'float32x4'}, // position
                        {shaderLocation: 1, offset: this.bufferOffsets.color, format: 'float32x4'}, // color
                    ],
                    stepMode: 'instance', // Use instance mode to repeat vertex data for each quad
                }],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragment_main',
                targets: [{ format: presentationFormat }],
            },
            primitive: {
                topology: 'triangle-strip',
                stripIndexFormat: 'uint32',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: depthFormat,
            },
        });

        this.fetch("aaa")
    }

    async fetch(url:string) {
        this.buffer = new Float32Array([
            // Position (x, y, z, w)   Color (r, g, b, a)
            -1, -1,  1, 1,   1, 0, 0, 1, // Red
            1, -1,  1, 1,   0, 1, 0, 1, // Green
            1,  1,  1, 1,   0, 0, 1, 1, // Blue
            -1,  1,  1, 1,   1, 1, 0, 1, // Yellow
            -1, -1, -1, 1,   0, 1, 1, 1, // Cyan
            1, -1, -1, 1,   1, 0, 1, 1, // Magenta
            1,  1, -1, 1,   1, 1, 1, 1, // White
            -1,  1, -1, 1,   0.2, 0.2, 0.2, 1, // Dark Gray
        ]);

        this.vertexBuffer = device.createBuffer({
            size: this.buffer.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(this.buffer);
        this.vertexBuffer.unmap();
        this.count = this.buffer.byteLength / this.bufferOffsets.stride;
    }
    public draw(passEncoder: GPURenderPassEncoder, device: GPUDevice, model: Mat4Arg, view: Mat4Arg, proj: Mat4Arg, viewport: {w: number, h: number}): void {
        if(this.count ===0) return;

        // update uniform
        const modelViewMatrix = mat4.create();
        mat4.multiply(view, model, modelViewMatrix);
        mat4.multiply(proj, modelViewMatrix, this.mvpMatrix);

        this.uniformData.set(this.mvpMatrix, 0);
        this.uniformData.set([viewport.w, viewport.h], 16);

        device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            this.uniformData
        );

        //draw
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.draw(4, this.count);
    }
}

const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: depthFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

const renderPassDescriptor: GPURenderPassDescriptor = {
    colorAttachments: [{
        view: undefined, // Will be set in frame()
        clearValue: [0.0, 0.0, 0.0, 1.0],
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

const renderCube = new RenderCube(device, presentationFormat, depthFormat);


// const w = 1000, h = 1000, fx=1000, fy = 1000;

const viewport = {w:canvas.width, h:canvas.height}; //TODO providate that to the shader
const aspect = canvas.width / canvas.height;
const proj = mat4.perspective((2 * Math.PI) / 5, aspect, 1, 100.0);


let modelCube = mat4.identity();
function updateAnimateCube(now: number, model: Mat4Arg) {
    mat4.identity(model);
    mat4.rotate(model, vec3.fromValues(Math.sin(now), Math.cos(now), 0), 1, model);
}

const view = mat4.identity();
mat4.translate(view, vec3.fromValues(0, 0, -4), view); // Move camera back

function frame() {
    const now = Date.now() / 1000;
    updateAnimateCube(now, modelCube)

    const colorAttachment = renderPassDescriptor.colorAttachments[0] as GPURenderPassColorAttachment;
    colorAttachment.view = context.getCurrentTexture().createView();
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    renderCube.draw(passEncoder, device, modelCube, view, proj, viewport);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
