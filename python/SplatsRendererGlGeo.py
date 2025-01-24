import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders

from utils import load_splat_file, Glfw, timer

glsl = str

vertex_shader: glsl = """
#version 330 core
layout (location = 0) in vec3 center;    // 3D position
layout (location = 1) in vec3 scale;     // Scale factors
layout (location = 2) in vec4 rotation;  // Quaternion rotation
layout (location = 3) in vec4 color;     // RGBA color

out VS_OUT {
    vec3 center;
    vec3 scale;
    vec4 rotation;
    vec4 color;
} vs_out;

void main() {
    vs_out.center = center;
    vs_out.scale = scale;
    vs_out.rotation = rotation;
    vs_out.color = color;
}
"""

geometry_shader: glsl = """
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 uProj, uView;
uniform vec2 uFocal, uViewport;

in VS_OUT {
    vec3 center;
    vec3 scale;
    vec4 rotation;
    vec4 color;
} gs_in[];

out vec4 gColor;
out vec2 gPosition;

mat3 quat_to_mat3(vec4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return mat3(
        1.0 - 2.0 * (z * z + w * w),
        2.0 * (y * z + x * w),
        2.0 * (y * w - x * z),

        2.0 * (y * z - x * w),
        1.0 - 2.0 * (y * y + w * w),
        2.0 * (z * w + x * y),

        2.0 * (y * w + x * z),
        2.0 * (z * w - x * y),
        1.0 - 2.0 * (y * y + z * z)
    );
}

mat3 computeCov3D(vec4 quaternion, vec3 scale) {
    mat3 R = quat_to_mat3(quaternion);
    mat3 S = mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    mat3 M = R * S;
    mat3 cov3d = M * transpose(M);
    return cov3d;
}

void main() {
    vec4 center = vec4(gs_in[0].center, 1.0);
    vec3 scale = gs_in[0].scale;
    vec4 rotation = gs_in[0].rotation;
    vec4 color = gs_in[0].color;

    vec4 cam = uView * center;
    vec4 pos2d = uProj * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    mat3 Vrk = 4.0 * computeCov3D(rotation, scale);

    mat3 J = mat3(
        uFocal.x / cam.z, 0., -(uFocal.x * cam.x) / (cam.z * cam.z),
        0., -uFocal.y / cam.z, (uFocal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(uView)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector; //in pixel
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    gColor = color;
    vec2 vCenter = vec2(pos2d) / pos2d.w;

    vec2 quad[4] = vec2[](
        vec2(-2.0, -2.0),
        vec2( 2.0, -2.0),
        vec2(-2.0,  2.0),
        vec2( 2.0,  2.0)
    );

    for (int i = 0; i < 4; i++) {
        gPosition = quad[i];
        gl_Position = vec4(vCenter + (quad[i].x * majorAxis + quad[i].y * minorAxis) / uViewport, 0.0, 1.0);
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
    float A = -dot(gPosition, gPosition);
    if (A < -4.0) discard;
    float B = exp(A) * gColor.a;
    fragColor = vec4(B * gColor.rgb, B);
}
"""


class SplatsRendererGlGeo:
    """
        Use standard opengl pipeline : vertex -> geometry -> fragment
        At geometry step, it transforms a point into a quad
        cov3d could be preprocessed instead of being recalculated at every frame
    """
    def __init__(self, splat_file_path: str, width: int, height: int):
        self.width, self.height = width, height

        # Load splat data
        self.points = load_splat_file(splat_file_path)

        self.glfw = Glfw(width, height)

        self.program = self.create_shader_program()

        # Create and setup buffers
        self.setup_buffers()

        # Get uniform locations
        self.uViewLoc = glGetUniformLocation(self.program, "uView")
        self.uProjLoc = glGetUniformLocation(self.program, "uProj")
        self.uFocalLoc = glGetUniformLocation(self.program, "uFocal")
        self.uViewportLoc = glGetUniformLocation(self.program, "uViewport")

        # Setup blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_ONE) # antimatter - front to back
        # glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)  # c++ - back to front - np.argsort(-depths)
        glClearColor(0,0,0,0)

    def setup_buffers(self):
        positions, scales, rotations, colors = self.points

        self.positions = positions #for sort

        # Combine all data into a single array
        vertex_data = np.hstack([
            positions,  # 3 floats
            scales,  # 3 floats
            rotations,  # 4 floats (quaternion)
            colors  # 4 floats (RGBA)
        ]).astype(np.float32)

        # Create and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Create and bind VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        attributes = [
            ("center", 3),  # position/center: 3 floats
            ("scale", 3),  # scale: 3 floats
            ("rotation", 4),  # rotation: 4 floats
            ("color", 4)  # color: 4 floats
        ]

        stride = sum(count for _, count in attributes) * sizeof(GLfloat)
        offset = 0

        for index, (attr_name, count) in enumerate(attributes):
            loc = glGetAttribLocation(self.program, attr_name)
            glVertexAttribPointer(loc, count, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
            glEnableVertexAttribArray(index)
            offset += count * sizeof(GLfloat)

        # to store order, see #sort
        self.ebo = glGenBuffers(1)

    def create_shader_program(self):
        vert_shader = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
        geom_shader = shaders.compileShader(geometry_shader, GL_GEOMETRY_SHADER)
        frag_shader = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vert_shader, geom_shader, frag_shader)

    def draw(self, view: np.array, proj: np.array, w: int, h: int, f: int):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        glUniformMatrix4fv(self.uViewLoc, 1, GL_FALSE, view)
        glUniformMatrix4fv(self.uProjLoc, 1, GL_FALSE, proj)
        glUniform2f(self.uFocalLoc, f, f)
        glUniform2f(self.uViewportLoc, w, h)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        # with timer("glDraw"):
        glDrawElements(GL_POINTS, len(self.positions), GL_UNSIGNED_INT, None)
        # glFinish()

    def sort(self, viewProj):
        if viewProj is None:
            indices = np.arange(len(self.positions), dtype=np.uint32)
        else:
            positions_v4 = np.hstack([self.positions, np.ones((len(self.positions), 1))])
            cam = (viewProj @ positions_v4.T).T
            depths = cam[:, 2]
            indices = np.argsort(depths).astype(np.uint32)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)
        assert indices.dtype == np.uint32

    def readImage(self) -> np.array:
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 4)
        image = np.flipud(image) # upside down
        image = image[:, :, :3] # remove alpha
        return image

    def loop(self, view, proj, w, h, f):
        while self.glfw.opened():
            self.draw(view, proj, w, h, f)


