import math

import numpy as np
import matplotlib

from SplatsRendererGlGeo import SplatsRendererGlGeo
from SplatsRendererGl import SplatsRendererGl
from SplatsRendererGlGeoConic import SplatsRendererGlGeoConic
from SplatsRendererGlNoVertexSh import SplatsRendererGlNoVertexSh
from SplatsRendererVkGeo import SplatsRendererVkGeo

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
import glm

from SplatsRenderer import SplatsRenderer
from SplatsRendererLoop import SplatsRendererLoop
from utils import create_projection_matrix, timer


def get_view_proj_using_glm(w, h, f):
    view_matrix = np.eye(4)  # Identity matrix for testing

    # view_matrix = np.array([ #//perspectiveLH_NO
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 3],
    #     [0, 0, 0, 1]
    # ])

    # view_matrix = np.array([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, -1, -3],
    #     [0, 0, 0, 1]
    # ])

    proj1 = create_projection_matrix(f, f, w, h)

    # proj2 = glm.perspective(f, w/h, 0.2, 200.0)
    # proj = glm_to_numpy(proj2)
    # proj3 = (np.eye(4) * proj2).T
    # proj4 = glm_to_numpy(proj2).T

def get_view_proj_matrix_antimatter():
    view_antimatter = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0.1, 3, 1]
    ])

    assert np.allclose(view_antimatter, np.array(glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.1, 3.0))).T)

    # view_glm = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.1, 3.0))
    # view_glm = glm.rotate(view_glm, glm.radians(-30.0), glm.vec3(0.0, 1.0, 0.0))
    # view_antimatter = np.array(view_glm).T

    proj_antimatter = np.array([
        [2., 0., 0., 0., ],
        [0., -2., 0., 0., ],
        [0., 0., 1.0010010010010009, 1.],
        [0., 0., -0.20020020020020018, 0., ]
    ])

    wfov = math.atan((w/2)/f) * 2
    print(f"Camera {w}x{h} wfov={math.degrees(wfov):.1f}°")
    # proj2 = glm.perspectiveLH_NO(fov, w / float(h), 0.2, 200.0)
    # flip y

    return view_antimatter, proj_antimatter


if __name__ == "__main__":

    splatAxis = "../web/public/ds/axis.splat"
    splatTrain = "../web/public/ds/train.splat" # https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat


    w, h, f = 1000, 1000, 1000

    # choose which splat file to use
    splat, fn = splatAxis, 'axis'
    # splat, fn = splatTrain, 'train'

    # choose which render to use
    # renderer, fn_render, og = SplatsRendererLoop(splat), "loop", False
    # renderer, fn_render, og = SplatsRenderer(splat), "vect", False
    # renderer, fn_render, og = SplatsRendererGl(splat, w, h), "gl", True
    # renderer, fn_render, og = SplatsRendererGlGeo(splat, w, h), "glgeo", True
    # renderer, fn_render, og = SplatsRendererGlGeoConic(splat, w, h), "glgeoconic", True
    # renderer, fn_render, og = SplatsRendererGlNoVertexSh(splat, w, h), "glnovertex", True
    renderer, fn_render, og = SplatsRendererVkGeo(splat, w, h), "vkgeo", True

    output = f"test/{fn}_{fn_render}.png"

    view, proj = get_view_proj_matrix_antimatter()

    # TODO keep only one version
    if not og:
        # Calculation renderers
        with timer("render"):
            image = renderer.render(view.T, proj.T, w, h, f)
        print(f"Rendering {renderer.__class__.__name__} in {output}")
        Image.fromarray(image).save(output)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    else:
        # OpenGL/Vk renderers
        loop = False # display and save once, or loop
        renderer.sort(view @ proj)
        if not loop:
            with timer('draw'):
                renderer.draw(view, proj, w, h, f)
            with timer("read"):
                image = renderer.readImage()

            print(f"Rendering {renderer.__class__.__name__} in {output}")
            Image.fromarray(image).save(output)
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        else:
            renderer.loop(view, proj, w, h, f)

