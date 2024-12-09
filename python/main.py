import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glm

from python.PointsRenderer_ import PointsRenderer
from python.SplatsRenderer import SplatsRenderer
from python.utils import Viewport, create_projection_matrix, glm_to_numpy

if __name__ == "__main__":
    renderer = SplatsRenderer("../public/tmp/gs_Emma_26fev_low.splat")
    # renderer = PointsRenderer("../public/tmp/gs_Emma_26fev_low.splat")
    view_matrix = np.eye(4)  # Identity matrix for testing
    w, h, f = 1000, 1000, 1000
    viewport = Viewport(width=w, height=h)

    # view_matrix = np.array([ #//perspectiveLH_NO
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 3],
    #     [0, 0, 0, 1]
    # ])

    view_matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, -3],
        [0, 0, 0, 1]
    ])

    proj1 = create_projection_matrix(f, f, viewport.width, viewport.height)

    # proj2 = glm.perspective(f, w/h, 0.2, 200.0)
    # proj = glm_to_numpy(proj2)
    # proj3 = (np.eye(4) * proj2).T
    # proj4 = glm_to_numpy(proj2).T

    # Render the point cloud
    image = renderer.render(view_matrix, proj1, viewport, f)
    Image.fromarray(image).save('tmp/result.png')

    plt.imshow(image)
    plt.axis('off')
    plt.show()
