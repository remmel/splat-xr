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
out vec3 vConic;
out vec2 vCoordxy;

mat3 quat_to_mat3(vec4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return mat3(
        1. - 2. * (y * y + z * z),  2. * (x * y + w * z),       2. * (x * z - w * y),
        2. * (x * y - w * z),       1. - 2. * (x * x + z * z),  2. * (y * z + w * x),
        2. * (x * z + w * y),       2. * (y * z - w * x),       1. - 2. * (x * x + y * y)
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
    return M * transpose(M);
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

    mat3 Vrk = computeCov3D(rotation, scale);

    mat3 J = mat3(
        uFocal.x / cam.z, 0., -(uFocal.x * cam.x) / (cam.z * cam.z),
        0., -uFocal.y / cam.z, (uFocal.y * cam.y) / (cam.z * cam.z),
        0., 0., 0.
    );

    mat3 T = transpose(mat3(uView)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    vec3 cov = vec3(cov2d[0][0], cov2d[0][1], cov2d[1][1]);

    float det = cov.x * cov.z - cov.y * cov.y;

    if (det == 0.0)
        return;

    float det_inv = 1.0 / det;
    vConic = vec3(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);

    vec2 quadwh_scr = vec2(3.0 * sqrt(cov.x), 3.0 * sqrt(cov.z));
    vec2 quadwh_ndc = quadwh_scr / uViewport * 2.0;
    vec2 center_ndc = vec2(pos2d) / pos2d.w;

    gColor = color;

    vec2 quads[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0)
    );

    for (int i = 0; i < 4; i++) {
        vec2 quad = quads[i];
        gl_Position = vec4(center_ndc + quad * quadwh_ndc, 0.0, 1.0f);
        vCoordxy = quad * quadwh_scr;
        EmitVertex();
    }

    EndPrimitive();
}
"""

fragment_shader: glsl = """
#version 330 core
in vec4 gColor;
in vec3 vConic;
in vec2 vCoordxy;   // local coordinate in quad, unit in pixel
out vec4 fragColor;

void main() {
    float power = -0.5 * (vConic.x * vCoordxy.x * vCoordxy.x + vConic.z * vCoordxy.y * vCoordxy.y) - vConic.y * vCoordxy.x * vCoordxy.y;
    if (power > 0.0) discard;
    float alpha = min(0.99, gColor.a * exp(power));
    if (alpha < 1.0 / 255.0) discard;
    fragColor = vec4(gColor.rgb * alpha, alpha);
}
"""


class SplatsRendererGlGeoConic:
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


