import OpenGL.GL as gl
# from OpenGL.GL import *
import numpy as np
from OpenGL.GL import shaders

import struct

from utils import load_splat_file, timer, Glfw


def pack_half2x16(a: float, b: float) -> np.uint32:
    """Pack two float32 values into a uint32 as two float16s."""
    return (np.uint32(struct.unpack('H', np.float16(a).tobytes())[0]) |
            (np.uint32(struct.unpack('H', np.float16(b).tobytes())[0]) << 16))


def pack_half2x16_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized version of pack_half2x16 that operates on numpy arrays."""
    # Convert float32 arrays to float16 arrays
    a_f16 = a.astype(np.float16)
    b_f16 = b.astype(np.float16)

    # View the float16 arrays as uint16
    a_bits = a_f16.view(np.uint16)
    b_bits = b_f16.view(np.uint16)

    # Combine the bits using vectorized operations
    return a_bits.astype(np.uint32) | (b_bits.astype(np.uint32) << 16)


vertex_shader = """
#version 430
precision highp float;
precision highp int;

layout(binding = 0) uniform usampler2D uTexture;
uniform mat4 uProj;
uniform mat4 uView;
uniform vec2 uFocal;     // focal in pixel eg [1150, 1150]
uniform vec2 uViewport;  // resolution in pixel eg [1920, 1080]

layout(location = 0) in vec2 aPosition;
layout(location = 1) in int aIndex;

out vec4 vColor;
out vec2 vPosition;

void main() {
    // Extract x and y coordinates for texture lookup
    uint x = (uint(aIndex) & 0x3ffu) << 1;  // Extract lower 10 bits and multiply by 2
    uint y = uint(aIndex) >> 10;            // Extract upper bits

    uvec4 centeru = texelFetch(uTexture, ivec2(x, y), 0);
    vec4 center = vec4(uintBitsToFloat(centeru.xyz), 1);
    vec4 cam = uView * center;
    vec4 pos2d = uProj * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip ||
        pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(uTexture, ivec2(x | 1u, y), 0);
    vec2 u1 = unpackHalf2x16(cov.x);
    vec2 u2 = unpackHalf2x16(cov.y);
    vec2 u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = 4.0 * mat3(
        u1.x, u1.y, u2.x,
        u1.y, u2.y, u3.x,
        u2.x, u3.x, u3.y
    );

    mat3 J = mat3(
        uFocal.x / cam.z, 0., -(uFocal.x * cam.x) / (cam.z * cam.z),
        0., -uFocal.y / cam.z, (uFocal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(uView)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius;
    float lambda2 = mid - radius;

    if(lambda2 < 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) *
                    vec2(diagonalVector.y, -diagonalVector.x);

    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4(
        float(cov.w & 0xffu),
        float((cov.w >> 8) & 0xffu),
        float((cov.w >> 16) & 0xffu),
        float((cov.w >> 24) & 0xffu)
    ) / 255.0;

    vPosition = aPosition;
    vec2 vCenter = vec2(pos2d) / pos2d.w;

    gl_Position = vec4(
        vCenter + (aPosition.x * majorAxis + aPosition.y * minorAxis) / uViewport,
        0.0,
        1.0
    );
}
"""

fragment_shader = """
#version 430
precision highp float;

in vec4 vColor;
in vec2 vPosition;
in float vDebugValue;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out float fragDebug;

void main() {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(vColor.rgb * B, B);
    fragDebug = vDebugValue;
}
"""


class SplatsRendererGl:
    """"
    This is a port of the antimatter webgl2 code. Data are store in texture.
    """
    def __init__(self, splat_file_path, width, height):
        # self.positions, self.scales, self.rots, self.colors = load_splat_file(splat_file_path)
        self.width, self.height = width, height

        self.glfw = Glfw(width, height)

        self.setup_shaders()
        texture_data = self.load_and_generate_texture(splat_file_path)
        self.setup_buffers(texture_data)

    def setup_shaders(self):
        vertex_shader_c = shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER)
        fragment_shader_c = shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vertex_shader_c, fragment_shader_c)

        # Get uniform locations
        self.uProjLoc = gl.glGetUniformLocation(self.program, "uProj")
        self.uViewLoc = gl.glGetUniformLocation(self.program, "uView")
        self.uFocalLoc = gl.glGetUniformLocation(self.program, "uFocal")
        self.uViewportLoc = gl.glGetUniformLocation(self.program, "uViewport")

    def generate_texture(self, splat_buffer: np.ndarray, vertex_count: int) -> np.ndarray:
        # Convert splat file to gpu (texture) ready data format
        # Keep that slow loop for comparison purpose with js antimatter code. Here it takes ~35s for 1M splats
        # The vectorize version takes 0.82-1s whereas the js one 0.65-1s.

        # Create different views of the buffer
        buffer_f32 = np.frombuffer(splat_buffer, dtype=np.float32)
        buffer_u8 = np.frombuffer(splat_buffer, dtype=np.uint8)
        buffer_u32 = np.frombuffer(splat_buffer, dtype=np.uint32)

        # Set texture dimensions
        tex_width = 1024 * 2
        tex_height = np.ceil((2 * vertex_count) / tex_width).astype(int)

        print(f"w: {tex_width}, h: {tex_height}")
        print(f"texture units: {tex_width * tex_height * 4 / 8}")
        print(f"nb splats: {len(buffer_u32) / 8}")
        print(f"lost: {tex_width * tex_height * 4 / 8 - len(buffer_u32) / 8}")

        # Create texture data arrays
        texdata_u32 = np.zeros(tex_width * tex_height * 4, dtype=np.uint32)
        texdata_u8 = texdata_u32.view(np.uint8)
        texdata_f32 = texdata_u32.view(np.float32)

        for i in range(vertex_count):
            # Copy positions (x, y, z)
            texdata_f32[8 * i:8 * i + 3] = buffer_f32[8 * i:8 * i + 3]

            # Copy color (rgba)
            texdata_u32[8 * i + 7] = buffer_u32[8 * i + 6]

            # Extract scale and rotation
            scale = buffer_f32[8 * i + 3:8 * i + 6]
            rot = (buffer_u8[32 * i + 28:32 * i + 32].astype(np.float32) - 128) / 128

            # Compute rotation matrix from quaternion
            qw, qx, qy, qz = rot
            M = np.array([
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy + qw * qz),
                2.0 * (qx * qz - qw * qy),

                2.0 * (qx * qy - qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz + qw * qx),

                2.0 * (qx * qz + qw * qy),
                2.0 * (qy * qz - qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ])

            # Scale each row of the rotation matrix
            M = M * np.repeat(scale, 3)

            # Compute covariance matrix
            sigma = np.array([
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
            ])

            # Pack covariance into texture
            c = 1.0  # Scale factor
            texdata_u32[8 * i + 4] = pack_half2x16(c * sigma[0], c * sigma[1])
            texdata_u32[8 * i + 5] = pack_half2x16(c * sigma[2], c * sigma[3])
            texdata_u32[8 * i + 6] = pack_half2x16(c * sigma[4], c * sigma[5])


        return texdata_f32.reshape(tex_height, tex_width, 4)


    def generate_texture_vectorized(self, splat_buffer: np.ndarray, vertex_count: int) -> np.ndarray:
        # Create different views of the buffer
        buffer_f32 = np.frombuffer(splat_buffer, dtype=np.float32)
        buffer_u8 = np.frombuffer(splat_buffer, dtype=np.uint8)
        buffer_u32 = np.frombuffer(splat_buffer, dtype=np.uint32)

        # Set texture dimensions
        tex_width = 1024 * 2
        tex_height = np.ceil((2 * vertex_count) / tex_width).astype(int)

        print(f"w: {tex_width}, h: {tex_height}")
        print(f"texture units: {tex_width * tex_height * 4 / 8}")
        print(f"nb splats: {len(buffer_u32) / 8}")
        print(f"lost: {tex_width * tex_height * 4 / 8 - len(buffer_u32) / 8}")

        # Create texture data arrays
        texdata_u32 = np.zeros(tex_width * tex_height * 4, dtype=np.uint32)
        texdata_f32 = texdata_u32.view(np.float32)

        # Vectorized copy of positions (x, y, z)
        positions_idx = np.arange(vertex_count) * 8
        texdata_f32[positions_idx[:, None] + np.arange(3)] = buffer_f32[positions_idx[:, None] + np.arange(3)]

        # Vectorized copy of colors (rgba)
        texdata_u32[positions_idx + 7] = buffer_u32[positions_idx + 6]

        # Extract scales and rotations for all vertices at once
        scales = buffer_f32[positions_idx[:, None] + np.arange(3, 6)].reshape(-1, 3)
        rots = (buffer_u8.reshape(-1, 32)[:, 28:32].astype(np.float32) - 128) / 128

        # Vectorized quaternion to rotation matrix conversion
        qw, qx, qy, qz = rots.T

        # Compute rotation matrices for all vertices at once
        M = np.zeros((vertex_count, 9))
        M[:, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
        M[:, 1] = 2.0 * (qx * qy + qw * qz)
        M[:, 2] = 2.0 * (qx * qz - qw * qy)
        M[:, 3] = 2.0 * (qx * qy - qw * qz)
        M[:, 4] = 1.0 - 2.0 * (qx * qx + qz * qz)
        M[:, 5] = 2.0 * (qy * qz + qw * qx)
        M[:, 6] = 2.0 * (qx * qz + qw * qy)
        M[:, 7] = 2.0 * (qy * qz - qw * qx)
        M[:, 8] = 1.0 - 2.0 * (qx * qx + qy * qy)

        # Scale each row of all rotation matrices at once
        M = M * np.repeat(scales, 3, axis=1)

        # Compute covariance matrices for all vertices at once
        sigma = np.zeros((vertex_count, 6))
        sigma[:, 0] = M[:, 0] * M[:, 0] + M[:, 3] * M[:, 3] + M[:, 6] * M[:, 6]
        sigma[:, 1] = M[:, 0] * M[:, 1] + M[:, 3] * M[:, 4] + M[:, 6] * M[:, 7]
        sigma[:, 2] = M[:, 0] * M[:, 2] + M[:, 3] * M[:, 5] + M[:, 6] * M[:, 8]
        sigma[:, 3] = M[:, 1] * M[:, 1] + M[:, 4] * M[:, 4] + M[:, 7] * M[:, 7]
        sigma[:, 4] = M[:, 1] * M[:, 2] + M[:, 4] * M[:, 5] + M[:, 7] * M[:, 8]
        sigma[:, 5] = M[:, 2] * M[:, 2] + M[:, 5] * M[:, 5] + M[:, 8] * M[:, 8]

        # Scale factor for covariance
        c = 1.0
        sigma *= c

        # Pack covariance matrices using vectorized operations
        texdata_u32[positions_idx + 4] = pack_half2x16_vectorized(sigma[:, 0], sigma[:, 1])
        texdata_u32[positions_idx + 5] = pack_half2x16_vectorized(sigma[:, 2], sigma[:, 3])
        texdata_u32[positions_idx + 6] = pack_half2x16_vectorized(sigma[:, 4], sigma[:, 5])

        return texdata_f32.reshape(tex_height, tex_width, 4)

    # Example usage:
    def load_and_generate_texture(self, filepath: str) -> np.ndarray:
        with open(filepath, 'rb') as f:
            data = f.read()

        splat_buffer = np.frombuffer(data, dtype=np.uint8)
        self.vertex_count = len(splat_buffer) // 32  # 32 bytes per vertex

        # duplicated load the data
        positions, scales, rots, colors = load_splat_file(filepath)
        self.positions = positions # used to sort

        with timer("generate texture"):
            splats_data_array = self.generate_texture_vectorized(splat_buffer, self.vertex_count)
        return splats_data_array

    def setup_buffers(self, splats_data_array):
        # Create and bind vertex array object
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        # Vertex - quad position
        triangle_vertices = np.array([-2, -2, 2, -2, 2, 2, -2, 2], dtype=np.float32)
        self.vertexBuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertexBuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, triangle_vertices.nbytes, triangle_vertices, gl.GL_STATIC_DRAW)

        aPositionLoc = gl.glGetAttribLocation(self.program, "aPosition")
        gl.glEnableVertexAttribArray(aPositionLoc)
        gl.glVertexAttribPointer(aPositionLoc, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # indexBuffer - provides the index of the splat to handle
        self.indexBuffer = gl.glGenBuffers(1)
        aIndexLoc = gl.glGetAttribLocation(self.program, "aIndex")
        gl.glEnableVertexAttribArray(aIndexLoc)
        gl.glVertexAttribIPointer(aIndexLoc, 1, gl.GL_UNSIGNED_INT, 0, None)
        gl.glVertexAttribDivisor(aIndexLoc, 1)
        # self.sort()

        # Create, bind texture, and upload data
        self.input_splat_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.input_splat_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        [tex_h, tex_w, _] = splats_data_array.shape
        # Upload texture data
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32UI, tex_w, tex_h, 0, gl.GL_RGBA_INTEGER, gl.GL_UNSIGNED_INT, splats_data_array.tobytes())

        gl.glClearColor(0, 0, 0, 0)


    def draw(self, view, proj, w, h, f):
        assert view.shape == (4, 4); assert proj.shape == (4, 4)

        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE_MINUS_DST_ALPHA, gl.GL_ONE) #antimatter
        # gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)  # c++
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(self.program)

        gl.glUniformMatrix4fv(self.uProjLoc, 1, gl.GL_FALSE, proj)
        gl.glUniformMatrix4fv(self.uViewLoc, 1, gl.GL_FALSE, view)
        gl.glUniform2f(self.uFocalLoc, f, f)
        gl.glUniform2f(self.uViewportLoc, w, h)

        gl.glBindVertexArray(self.vao)
        # with timer("glDraw"):
        gl.glDrawArraysInstanced(gl.GL_TRIANGLE_FAN, 0, 4, self.vertex_count)
        # glFinish()

    def sort(self, viewProj=None):
        if viewProj is None:
            indices = np.arange(len(self.positions), dtype=np.uint32)
        else:
            positions_v4 = np.hstack([self.positions, np.ones((len(self.positions), 1))])
            cam = (viewProj @ positions_v4.T).T
            depths = cam[:, 2]
            indices = np.argsort(depths).astype(np.uint32)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.indexBuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        aIndexLoc = gl.glGetAttribLocation(self.program, "aIndex")
        # why should I repeat glVertexAttribIPointer ? (in cpp, webgl2 not needed)
        gl.glVertexAttribIPointer(aIndexLoc, 1, gl.GL_UNSIGNED_INT, 0, None)

    def loop(self, view, proj, w, h, f):
        while self.glfw.opened():
            self.draw(view, proj, w, h, f)

    def readImage(self):
        pixels = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 4)
        image = np.flipud(image)  # Flip because OpenGL has bottom-left origin
        image = image[:, :, :3]  # Remove alpha channel
        return image
