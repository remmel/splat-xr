from dataclasses import dataclass
import numpy as np
import glm
from contextlib import contextmanager
import time


@dataclass
class Viewport:
    width: int
    height: int

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
    print(f"{name}: {time.time() - start:.4f} seconds")