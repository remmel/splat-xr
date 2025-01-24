import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
from utils import load_splat_file, Glfw, timer

glsl = str

vertex_shader: glsl = """
#version 330 core
layout (location = 0) in vec3 center;     // 3D position
layout (location = 1) in vec3 scale;      // Scale factors
layout (location = 2) in vec4 rotation;   // Quaternion rotation
layout (location = 3) in vec4 color;      // RGBA color
layout (location = 4) in vec2 majorAxis;  // Major axis for 2D gaussian
layout (location = 5) in vec2 minorAxis;  // Minor axis for 2D gaussian
layout (location = 6) in vec2 vCenter;    // Screen-space center position

out VS_OUT {
    vec4 color;
    vec2 majorAxis;
    vec2 minorAxis;
    vec2 vCenter;
} vs_out;

void main() {
    vs_out.color = color;
    vs_out.majorAxis = majorAxis;
    vs_out.minorAxis = minorAxis;
    vs_out.vCenter = vCenter;
}
"""

geometry_shader: glsl = """
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform vec2 uViewport;

in VS_OUT {
    vec4 color;
    vec2 majorAxis;
    vec2 minorAxis;
    vec2 vCenter;
} gs_in[];

out vec4 gColor;
out vec2 gPosition;

void main() {
    vec4 color = gs_in[0].color;
    vec2 majorAxis = gs_in[0].majorAxis;
    vec2 minorAxis = gs_in[0].minorAxis;
    vec2 vCenter = gs_in[0].vCenter;

    gColor = color;

    vec2 quad[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );

    for (int i = 0; i < 4; i++) {
        vec2 quadi = quad[i];
        gPosition = quadi * 2.0;
        
        gl_Position = vec4(
            vCenter.x + 2.0 * (quadi.x * majorAxis.x + quadi.y * minorAxis.x) / uViewport.x,
            vCenter.y + 2.0 * (quadi.x * majorAxis.y + quadi.y * minorAxis.y) / uViewport.y,
            0.0, 1.0);
        EmitVertex();
    }

    EndPrimitive();
}
"""

fragment_shader: glsl = """
#version 330 core
in vec4 gColor;
in vec2 gPosition;
out vec4 fragColor;

void main() {
    float A = dot(gPosition, gPosition);
    if (A > 4.0) discard;
    float B = exp(-A) * gColor.a;
    fragColor = vec4(B * gColor.rgb, B);
}
"""


class SplatsRendererGlNoVertexSh:
    """
        Created to check if the 2d splats are properly calculated
        - Projection to 2d in numpy python (instead of vertex shader - passthrough)
        - Standard rasterisation in opengl (fragment shader)
    """
    def __init__(self, splat_file_path: str, width: int, height: int):
        self.width, self.height = width, height
        self.points = load_splat_file(splat_file_path)
        self.glfw = Glfw(width, height)
        self.program = self.create_shader_program()
        self.setup_buffers()

        # Get uniform locations
        self.uViewLoc = glGetUniformLocation(self.program, "uView")
        self.uProjLoc = glGetUniformLocation(self.program, "uProj")
        self.uViewportLoc = glGetUniformLocation(self.program, "uViewport")

        # Setup blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE)  # antimatter - front to back
        glClearColor(0, 0, 0, 0)

    def compute_jacobian(self, cam_points, f):
        x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
        fx, fy = f, f

        # Compute Jacobian matrices for all points
        J = np.zeros((len(cam_points), 3, 3))
        J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = fx / z, 0, -(fx * x) / (z * z)
        J[:, 1, 0], J[:, 1, 1], J[:, 1, 2] = 0, -fy / z, (fy * y) / (z * z)
        J[:, 2, 0], J[:, 2, 1], J[:, 2, 2] = 0, 0, 0

        return J

    def compute_cov2d(self, view, vrk, J):
        T = view[:3, :3].T @ J.transpose(0, 2, 1) # should I transpose the view, try with view rotated
        cov2d = T.transpose(0, 2, 1) @ vrk @ T

        cov2d_batch = cov2d[:, :2, :2]  # Take only the 2x2 part
        eigenvals, eigenvecs = np.linalg.eigh(cov2d_batch)
        lambda2, lambda1 = eigenvals[:, 0, np.newaxis], eigenvals[:, 1, np.newaxis]

        # Calculate major and minor axes
        major_axis = np.minimum(np.sqrt(2 * lambda1), 1024) * eigenvecs[:, :, 1]
        minor_axis = np.minimum(np.sqrt(2 * lambda2), 1024) * eigenvecs[:, :, 0]
        # minor_axis = np.minimum(np.sqrt(2 * lambda2), 1024) * np.column_stack([eigenvecs[:, 1, 1],-eigenvecs[:, 0, 1]])

        # Filter out degenerate gaussians
        valid = lambda2.squeeze() >= 0

        return major_axis, minor_axis, valid

    def compute_cov3d(self, scales, rots):
        w, x, y, z = rots.T
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        ]).transpose(2, 0, 1)
        S = scales.reshape(-1, 3, 1) * np.eye(3)
        M = R @ S
        return M @ M.transpose(0, 2, 1)

    def setup_buffers(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # [(loc, nb of floats)]
        attrs = [
            (0, 3),  # Position
            (1, 3),  # Scale
            (2, 4),  # Rotation
            (3, 4),  # Color
            (4, 2),  # Major axis
            (5, 2),  # Minor axis
            (6, 2),  # vCenter
        ]

        stride = sum(size for loc, size in attrs) * sizeof(GLfloat)
        offset = 0

        for loc, size in attrs:
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
            offset += size * sizeof(GLfloat)

    def create_shader_program(self):
        vert_shader = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
        geom_shader = shaders.compileShader(geometry_shader, GL_GEOMETRY_SHADER)
        frag_shader = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert_shader, geom_shader, frag_shader)

    def draw(self, view: np.array, proj: np.array, w: int, h: int, f: int):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        num_splats_visible = self.setVbo(view.T, proj.T, w, h, f)

        glUniformMatrix4fv(self.uViewLoc, 1, GL_FALSE, view)
        glUniformMatrix4fv(self.uProjLoc, 1, GL_FALSE, proj)
        glUniform2f(self.uViewportLoc, w, h)

        glBindVertexArray(self.vao)
        # with timer("glDraw"):
        glDrawArrays(GL_POINTS, 0, num_splats_visible)
        # glFinish()

    def sort(self, viewProj):
        # sort done in render
        pass

    def setVbo(self, view, proj, w, h, f):
        positions, scales, rots, colors = self.points

        positions_v4 = np.hstack([positions, np.ones((len(positions), 1))])
        cam = (view @ positions_v4.T).T
        pos2d = (proj @ cam.T).T

        clip = 1.2 * pos2d[:, 3]
        in_frustum = ~(
                (pos2d[:, 2] < -clip) |
                (pos2d[:, 0] < -clip) |
                (pos2d[:, 0] > clip) |
                (pos2d[:, 1] < -clip) |
                (pos2d[:, 1] > clip)
        )
        print(f"{np.sum(in_frustum):,}/{len(in_frustum):,} in frustum")

        # Keep only visible
        positions, scales, rots, colors = positions[in_frustum], scales[in_frustum], rots[in_frustum], colors[in_frustum]
        pos2d, cam = pos2d[in_frustum], cam[in_frustum]

        vrks = 4 * self.compute_cov3d(scales, rots)
        J = self.compute_jacobian(cam, f)
        major_axis, minor_axis, valid = self.compute_cov2d(view, vrks, J)

        vCenter = pos2d[:, :2] / pos2d[:, 3:]

        vertex_data = np.hstack([
            positions,
            scales,
            rots,
            colors,
            major_axis,
            minor_axis,
            vCenter,
        ]).astype(np.float32)

        indices = np.argsort(cam[:, 2])
        vertex_data = vertex_data[indices]  # sort items

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)

        return len(vertex_data)

    def readImage(self) -> np.array:
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 4)
        image = np.flipud(image)  # flip upside down
        image = image[:, :, :3]  # remove alpha channel
        return image

    def loop(self, view, proj, w, h, f):
        while self.glfw.opened():
            self.draw(view, proj, w, h, f)