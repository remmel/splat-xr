import {packHalf2x16} from "./utils";
import {mat4, type Mat4Arg} from "wgpu-matrix";
import shadersWGSL from './shaders.wgsl?raw';

export class RenderSplat {
    private pipeline: GPURenderPipeline;
    private vertexBuffer: GPUBuffer | null = null;
    private sortedVertexBuffer: GPUBuffer | null = null;
    // private indexBuffer: GPUBuffer | null = null;
    private count : number = 0; //number of splats (NOT number of vertices)
    private lastCount: number = 0;
    private bufferGpu_f32: Float32Array = new Float32Array(0);
    private bufferGpuOffset = { // offsets in bytes
        'center': 0, //f32*3
        'scale': 4 * 3,//u8*4
        'cov2d': 4 * 4, //f32*3
        'color': 4 * 7, //u8*4
        'stride': 4 * 8
    }

    private uniformOffset = { // offsets in floats
        'modelView': 0, //mat4x4f
        'proj': 16, //mat4x4f
        'focal': 16 + 16, //vec2f
        'vp': 16 + 16 + 2, //vec2f
        'length': 16 + 16 + 2 + 2,
    }

    private uniformBuffer: GPUBuffer;
    private uniformBindGroup: GPUBindGroup;
    private uniformData = new Float32Array(this.uniformOffset.length);
    private device: GPUDevice;

    private lastProj: Mat4Arg | null = null;

    constructor(
        device: GPUDevice,
        presentationFormat: GPUTextureFormat,
        depthFormat: GPUTextureFormat,
    ) {
        this.device = device;

        this.uniformBuffer = device.createBuffer({
            size: this.uniformOffset.length * 4,
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

        // Create static index buffer for quad vertices (triangle strip: 0,1,2,3)
        // const quadIndices = new Uint32Array([0, 1, 2, 3]);
        // this.indexBuffer = device.createBuffer({
        //     size: quadIndices.byteLength,
        //     usage: GPUBufferUsage.INDEX,
        //     mappedAtCreation: true,
        // });
        // new Uint32Array(this.indexBuffer.getMappedRange()).set(quadIndices);
        // this.indexBuffer.unmap();

        this.pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [uniformGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vertex_main',
                buffers: [{
                    arrayStride: this.bufferGpuOffset.stride,
                    attributes: [
                        {shaderLocation: 0, offset: this.bufferGpuOffset.center, format: 'float32x3'},
                        {shaderLocation: 1, offset: this.bufferGpuOffset.scale, format: 'unorm8x4'},
                        {shaderLocation: 2, offset: this.bufferGpuOffset.cov2d, format: 'uint32x3'},
                        {shaderLocation: 3, offset: this.bufferGpuOffset.color, format: 'unorm8x4'},
                    ],
                    stepMode: 'instance', // Use instance mode to repeat vertex data for each quad
                }],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragment_main',
                targets: [{
                    format: presentationFormat,
                    blend: {
                        //f2b
                        color: {srcFactor: 'one-minus-dst-alpha', dstFactor: 'one', operation: 'add'},
                        alpha: {srcFactor: 'one-minus-dst-alpha', dstFactor: 'one', operation: 'add'}
                        //b2f
                        // color: {srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add'},
                        // alpha: {srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add'}
                    }
                }],
            },
            primitive: {
                topology: 'triangle-strip',
                stripIndexFormat: 'uint32',
            },
            depthStencil: {
                depthWriteEnabled: false,
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
        this.bufferGpu_f32 = new Float32Array(8 * count);
        const bufferGpu_u32 = new Uint32Array(this.bufferGpu_f32.buffer)
        for (let i = 0; i < count; i++) {
            // center : x, y, z - Float32 - 3x4Bytes <=> 3x32b
            this.bufferGpu_f32[8 * i + 0] = bufferFile_f32[8 * i + 0];
            this.bufferGpu_f32[8 * i + 1] = bufferFile_f32[8 * i + 1];
            this.bufferGpu_f32[8 * i + 2] = bufferFile_f32[8 * i + 2];

            // color : r, g, b, a - uint8 - 4*1B <=> 4x8b
            bufferGpu_u32[8 * i + 7] = bufferFile_u32[8 * i + 6]

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

        this.vertexBuffer = this.device.createBuffer({
            size: this.bufferGpu_f32.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(this.bufferGpu_f32);
        this.vertexBuffer.unmap();

        this.sortedVertexBuffer = this.device.createBuffer({
            size: this.bufferGpu_f32.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        this.count = count
    }

    public draw(passEncoder: GPURenderPassEncoder, model: Mat4Arg, view: Mat4Arg, proj: Mat4Arg, viewport: {w: number, h: number}): void {
        if(this.count ===0) return;

        // update uniform
        const modelView= mat4.create();
        mat4.multiply(view, model, modelView);

        const focal = [(proj[0] * viewport.w) / 2, -(proj[5] * viewport.h) / 2];

        this.uniformData.set(modelView, this.uniformOffset.modelView);
        this.uniformData.set(proj, this.uniformOffset.proj);
        this.uniformData.set(focal, this.uniformOffset.focal);
        this.uniformData.set([viewport.w, viewport.h], this.uniformOffset.vp);

        this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);

        const mvp= mat4.create();
        mat4.multiply(proj, modelView, mvp);
        this.runSort(mvp) // updates the sorted vertex buffer

        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.setVertexBuffer(0, this.sortedVertexBuffer);
        // passEncoder.setIndexBuffer(this.indexBuffer, 'uint32');
        passEncoder.draw(4, this.count);
    }

    private runSort(mvp: Mat4Arg) {
        const [x, y, z] = [mvp[2], mvp[6], mvp[10]];
        if (this.lastCount === this.count && this.lastProj) {
            const [lastX, lastY, lastZ] = [this.lastProj[2], this.lastProj[6], this.lastProj[10]];
            let dot = lastX * x + lastY * y + lastZ * z;
            if (Math.abs(dot - 1) < 0.01) {
                return;
            }
        }

        const splatOrder = this.sort(x, y, z);

        // TODO Provide sorted ids instead of sorted splats
        const sortedData = new Float32Array(8 * this.count);
        for (let i = 0; i < this.count; i++) {
            const srcIndex = splatOrder[i];
            const srcOffset = srcIndex * 8;
            const dstOffset = i * 8;
            for (let j = 0; j < 8; j++) {
                sortedData[dstOffset + j] = this.bufferGpu_f32[srcOffset + j];
            }
        }

        this.device.queue.writeBuffer(this.sortedVertexBuffer, 0, sortedData);

        this.lastProj = mvp;
        this.lastCount = this.count;

        console.log('sortedVertexBuffer #0:', splatOrder[0]);
    }

    private sort(x: number, y: number, z: number): Uint32Array {
        const buffer_f32 = this.bufferGpu_f32;
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(this.count);

        // Calculate depths and find min/max
        for (let i = 0; i < this.count; i++) {
            let depth = ((x * buffer_f32[8 * i + 0] + y * buffer_f32[8 * i + 1] + z * buffer_f32[8 * i + 2]) * 4096) | 0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }

        // 16-bit single-pass counting sort
        let depthInv = (256 * 256 - 1) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);

        // Normalize depths and count occurrences
        for (let i = 0; i < this.count; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }

        // Calculate start positions
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++) {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        // Build final index array - this represents the ORDER of splats to draw
        let depthIndex = new Uint32Array(this.count);
        for (let i = 0; i < this.count; i++) {
            depthIndex[starts0[sizeList[i]]++] = i;
        }

        return depthIndex;
    }
}