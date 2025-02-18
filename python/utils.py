import glfw
import numpy as np
import glm
from contextlib import contextmanager
import time


def create_projection_matrix(fx, fy, width, height):
    return np.array([
        [2*fx/width, 0, 0, 0],
        [0, -2*fy/height, 0, 0],
        [0, 0, -1, -1],
        [0, 0, -1, 0]
    ], dtype=np.float32)


def glm_to_numpy(glm_matrix):
    return np.array([glm_matrix[i][j] for i in range(4) for j in range(4)]).reshape(4,4)

def numpy_to_glm(np_matrix):
    return glm.mat4(*[np_matrix[i][j] for i in range(4) for j in range(4)])

@contextmanager
def timer(name):
    start = time.time()
    yield
    secs = time.time() - start
    print(f"{name}: {secs:.3f} seconds - {1/secs:.0f} fps")


class FPSCounter:
    def __init__(self):
        self.last_time = glfw.get_time()
        self.frame_count = 0

    def update(self):
        current_time = glfw.get_time()
        self.frame_count += 1

        if current_time - self.last_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_time)
            print(f"FPS: {int(fps)}")
            self.frame_count = 0
            self.last_time = current_time


def load_splat_file(file_path):
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    row_length = 3 * 4 + 3 * 4 + 4 + 4  # position + scale + rgba + quaternion
    vertex_count = len(data) // row_length
    data = data.reshape(vertex_count, row_length)

    # Extract positions
    positions = np.frombuffer(data[:, :12].tobytes(), dtype=np.float32).reshape(-1, 3)

    # Extract scales and rotations
    scales = np.frombuffer(data[:, 12:24].tobytes(), dtype=np.float32).reshape(-1, 3)
    rots = (data[:, 28:32].astype(np.float32) - 128) / 128

    # Extract colors
    colors = data[:, 24:28].astype(np.float32) / 255.0

    return positions, scales, rots, colors


class Glfw:
    def __init__(self, width: int, height: int, title: str = "GLFW Window", visible: bool = True):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        if not visible:
            glfw.window_hint(glfw.VISIBLE, False)

        glfw.window_hint(glfw.RESIZABLE, False) #keep it unresizable for benchmark purposes

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        self.fps_counter = FPSCounter()
        self.glfw = glfw

    def opened(self) -> bool:
        if glfw.window_should_close(self.window):
            glfw.terminate()
            return False

        self.fps_counter.update()
        glfw.poll_events()
        glfw.swap_buffers(self.window)
        return True


def remove_alpha(img, bg_is_white=False):
    bg_color = bg_is_white # black=0, white=1
    alpha = img[:, :, 3:4] #(h,w,1) instead of (h,w) <=> img[:, :, 3][:, :, np.newaxis]
    rgb = img[:, :, :3]
    return rgb * alpha + bg_color * (1 - alpha)

def remove_alpha_uint8(img, bg_is_white=False):
    return (remove_alpha(img / 255.0, bg_is_white) * 255.0).astype(np.uint8)
