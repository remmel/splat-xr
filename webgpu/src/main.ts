import { mat4, vec3 } from 'wgpu-matrix';
import type {Mat4Arg} from "wgpu-matrix";

import {
    cubeVertexArray,
    cubeVertexSize,
    cubePositionOffset,
    cubeVertexCount,
    cubeColorOffset,
} from './cube';

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

const UNIFORM_BUFFER_SIZE = 4 * 16; // 4x4 matrix, bytes

// RenderCube class
class RenderCube {
    private pipeline: GPURenderPipeline;
    private vertexBuffer: GPUBuffer;
    private vertexCount: number;

    private uniformBuffer: GPUBuffer;
    private uniformBindGroup: GPUBindGroup;
    private modelMatrix = mat4.identity();
    private mvpMatrix = mat4.create(); // Stores the final MVP for this cube

    constructor(
        device: GPUDevice,
        presentationFormat: GPUTextureFormat,
        depthFormat: GPUTextureFormat
    ) {
        this.vertexCount = cubeVertexCount;

        this.vertexBuffer = device.createBuffer({
            size: cubeVertexArray.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(cubeVertexArray);
        this.vertexBuffer.unmap();

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
                buffers: [
                    {
                        arrayStride: cubeVertexSize,
                        attributes: [
                            { shaderLocation: 0, offset: cubePositionOffset, format: 'float32x4' }, // position
                            { shaderLocation: 1, offset: cubeColorOffset, format: 'float32x4' }, // color
                        ],
                    },
                ],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragment_main',
                targets: [{ format: presentationFormat }],
            },
            primitive: {
                topology: 'point-list',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: depthFormat,
            },
        });
    }

    public update(now: number, projectionMatrix: Mat4Arg, cameraViewMatrix: Mat4Arg, device: GPUDevice): void {
        // Model transform (rotation specific to the cube)
        mat4.identity(this.modelMatrix);
        // Apply cube-specific transformations, e.g., rotation around Y axis and then X axis based on time
        mat4.rotate(this.modelMatrix, vec3.fromValues(Math.sin(now), Math.cos(now), 0), 1, this.modelMatrix);


        // Calculate ModelView matrix: V_camera * M_model
        const modelViewMatrix = mat4.create();
        mat4.multiply(cameraViewMatrix, this.modelMatrix, modelViewMatrix);

        // Calculate MVP: P * (V_camera * M_model)
        mat4.multiply(projectionMatrix, modelViewMatrix, this.mvpMatrix);

        device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            this.mvpMatrix.buffer,
            this.mvpMatrix.byteOffset,
            this.mvpMatrix.byteLength
        );
    }

    public draw(passEncoder: GPURenderPassEncoder): void {
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.draw(this.vertexCount);
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

const aspect = canvas.width / canvas.height;
const projectionMatrix = mat4.perspective((2 * Math.PI) / 5, aspect, 1, 100.0);

function frame() {
    const now = Date.now() / 1000;

    // Camera view matrix
    const cameraViewMatrix = mat4.identity();
    mat4.translate(cameraViewMatrix, vec3.fromValues(0, 0, -4), cameraViewMatrix); // Move camera back

    // Update the cube's state and uniforms
    renderCube.update(now, projectionMatrix, cameraViewMatrix, device);

    const colorAttachment = renderPassDescriptor.colorAttachments[0] as GPURenderPassColorAttachment;
    colorAttachment.view = context
        .getCurrentTexture()
        .createView();

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

    renderCube.draw(passEncoder);

    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);