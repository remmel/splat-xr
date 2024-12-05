import { createProgram } from "./utils.js";

export class RenderCube {
    constructor(gl) {
        // language=glsl
        const vertexShaderSource = `#version 300 es
        precision highp float;
        
        uniform mat4 uProj;
        uniform mat4 uView;
        uniform mat4 uModel;
        
        in vec3 aPosition;
        in vec3 aColor;
        
        out vec3 vColor;
        
        void main() {
            gl_Position = uProj * uView * uModel * vec4(aPosition, 1.0);
            vColor = aColor;
        }`;

        // language=glsl
        const fragmentShaderSource = `#version 300 es
        precision highp float;
        
        in vec3 vColor;
        out vec4 fragColor;
        
        void main() {
            fragColor = vec4(vColor, 1.0);
        }`

        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao)
        const program = this.program = createProgram(gl, vertexShaderSource, fragmentShaderSource)
        this.gl = gl;

        // Get uniform locations
        this.uProjLoc = gl.getUniformLocation(program, "uProj");
        this.uViewLoc = gl.getUniformLocation(program, "uView");
        this.uModelLoc = gl.getUniformLocation(program, "uModel");

        // Create cube vertices and colors
        const vertices = new Float32Array([
            // Front face
            -0.5, -0.5,  0.5,
            0.5, -0.5,  0.5,
            0.5,  0.5,  0.5,
            -0.5,  0.5,  0.5,
            // Back face
            -0.5, -0.5, -0.5,
            -0.5,  0.5, -0.5,
            0.5,  0.5, -0.5,
            0.5, -0.5, -0.5,
            // Top face
            -0.5,  0.5, -0.5,
            -0.5,  0.5,  0.5,
            0.5,  0.5,  0.5,
            0.5,  0.5, -0.5,
            // Bottom face
            -0.5, -0.5, -0.5,
            0.5, -0.5, -0.5,
            0.5, -0.5,  0.5,
            -0.5, -0.5,  0.5,
            // Right face
            0.5, -0.5, -0.5,
            0.5,  0.5, -0.5,
            0.5,  0.5,  0.5,
            0.5, -0.5,  0.5,
            // Left face
            -0.5, -0.5, -0.5,
            -0.5, -0.5,  0.5,
            -0.5,  0.5,  0.5,
            -0.5,  0.5, -0.5,
        ]);

        const colors = new Float32Array([
            // Front face (red)
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            // Back face (green)
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            // Top face (blue)
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            // Bottom face (yellow)
            1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            // Right face (magenta)
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
            // Left face (cyan)
            0.0, 1.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 1.0, 1.0,
        ]);

        const indices = new Uint16Array([
            0,  1,  2,    0,  2,  3,  // Front
            4,  5,  6,    4,  6,  7,  // Back
            8,  9,  10,   8,  10, 11, // Top
            12, 13, 14,   12, 14, 15, // Bottom
            16, 17, 18,   16, 18, 19, // Right
            20, 21, 22,   20, 22, 23  // Left
        ]);

        // Create and bind vertex buffer
        const vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
        const aPositionLoc = gl.getAttribLocation(program, "aPosition");
        gl.enableVertexAttribArray(aPositionLoc);
        gl.vertexAttribPointer(aPositionLoc, 3, gl.FLOAT, false, 0, 0);

        // Create and bind color buffer
        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
        const aColorLoc = gl.getAttribLocation(program, "aColor");
        gl.enableVertexAttribArray(aColorLoc);
        gl.vertexAttribPointer(aColorLoc, 3, gl.FLOAT, false, 0, 0);

        // Create and bind index buffer
        const indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

        this.indexCount = indices.length;
        this.rotation = 0;
    }

    draw(view, viewport, proj) {
        const gl = this.gl;

        // Use the cube shader program
        gl.useProgram(this.program);
        gl.bindVertexArray(this.vao);

        // Enable depth testing for 3D rendering
        gl.enable(gl.DEPTH_TEST);

        // Set uniforms
        gl.uniformMatrix4fv(this.uProjLoc, false, proj);
        gl.uniformMatrix4fv(this.uViewLoc, false, view);

        // Update rotation
        this.rotation += 0.01;

        // Create model matrix for rotation
        const model = [
            Math.cos(this.rotation), 0, Math.sin(this.rotation), 0,
            0, 1, 0, 0,
            -Math.sin(this.rotation), 0, Math.cos(this.rotation), 0,
            0, 0, 0, 1
        ];

        gl.uniformMatrix4fv(this.uModelLoc, false, model);

        // Draw the cube
        gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0);

        // Cleanup
        // gl.disable(gl.DEPTH_TEST);
    }
}
