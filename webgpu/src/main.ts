import { mat4, vec3 } from 'wgpu-matrix';
import type {Mat4Arg} from "wgpu-matrix";
import {packHalf2x16} from './utils.js'

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
    private bufferGpu_f32: Float32Array = new Float32Array(0);
    // offsets in bytes
    private bufferOffsets = {
        'position': 0,
        'color': 4*7,
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
                        {shaderLocation: 0, offset: this.bufferOffsets.position, format: 'float32x3'}, // position
                        {shaderLocation: 1, offset: this.bufferOffsets.color, format: 'unorm8x4'}, // color
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
    }

    async fetch(url:string) {
        const req = await fetch(url, {mode: "cors", credentials: "omit"})
        console.log(req)
        if (req.status != 200)
            throw new Error(req.status + " Unable to load " + req.url)

        const bufferFile = await req.arrayBuffer();
        const bufferFile_u8 = new Uint8Array(bufferFile), bufferFile_f32 = new Float32Array(bufferFile),
            bufferFile_u32 = new Uint32Array(bufferFile)
        // 6*4 + 4 + 4 = 8*4
        // XYZ - Position (Float32)
        // XYZ - Scale (Float32)
        // RGBA - colors (uint8)
        // IJKL - quaternion/rot (uint8)
        const fileStride = 3 * 4 + 3 * 4 + 4 + 4 //in Bytes
        const count = bufferFile_u8.length / fileStride
        console.log('vertexCount', count)

        this.bufferGpu_f32 = new Float32Array(8*count);
        const bufferGpu_u32 = new Uint32Array(this.bufferGpu_f32.buffer)
        for (let i = 0; i < count; i++) {
            // center : x, y, z - Float32 - 3x4Bytes <=> 3x32b
            this.bufferGpu_f32[8*i+0] = bufferFile_f32[8 * i + 0];
            this.bufferGpu_f32[8*i+1] = bufferFile_f32[8 * i + 1];
            this.bufferGpu_f32[8*i+2] = bufferFile_f32[8 * i + 2];

            // color : r, g, b, a - uint8 - 4*1B <=> 4x8b
            bufferGpu_u32[8*i+7] = bufferFile_u32[8 * i + 6]

            // quaternions
            let scale = [
                bufferFile_f32[8 * i + 3 + 0],
                bufferFile_f32[8 * i + 3 + 1],
                bufferFile_f32[8 * i + 3 + 2],
            ]
            let rot = [
                (bufferFile_u8[32 * i + 28 + 0] - 128) / 128,
                (bufferFile_u8[32 * i + 28 + 1] - 128) / 128,
                (bufferFile_u8[32 * i + 28 + 2] - 128) / 128,
                (bufferFile_u8[32 * i + 28 + 3] - 128) / 128,
            ]

            // Compute the matrix product of S and R (M = S * R)
            const [qw, qx, qy, qz] = rot
            const M = [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy + qw * qz),
                2.0 * (qx * qz - qw * qy),

                2.0 * (qx * qy - qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz + qw * qx),

                2.0 * (qx * qz + qw * qy),
                2.0 * (qy * qz - qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
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
            const c = 1
            bufferGpu_u32[8 * i + 4] = packHalf2x16(c * sigma[0], c * sigma[1])
            bufferGpu_u32[8 * i + 5] = packHalf2x16(c * sigma[2], c * sigma[3])
            bufferGpu_u32[8 * i + 6] = packHalf2x16(c * sigma[4], c * sigma[5])
        }

        this.vertexBuffer = device.createBuffer({
            size: this.bufferGpu_f32.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(this.bufferGpu_f32);
        this.vertexBuffer.unmap();
        this.count = this.bufferGpu_f32.byteLength / this.bufferOffsets.stride;
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
renderCube.fetch('tmp/axis.splat')


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
