import math

import numpy as np
import matplotlib
from pyglm.glm import inverse, perspective, radians, translate, mat4, vec3, rotate

from SplatsRendererGlGeo import SplatsRendererGlGeo
from SplatsRendererGl import SplatsRendererGl
from SplatsRendererGlGeoConic import SplatsRendererGlGeoConic
from SplatsRendererGlNoVertexSh import SplatsRendererGlNoVertexSh
from SplatsRendererVkGeo import SplatsRendererVkGeo

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
from pyglm import glm

from SplatsRendererNp import SplatsRendererNp
from SplatsRendererLoop import SplatsRendererLoop
from utils import create_projection_matrix, timer


def get_view_proj_matrix_antimatter():
    view_antimatter = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, .1, 3, 1]
    ])

    assert np.allclose(view_antimatter, np.array(translate(mat4(1.0), vec3(0.0, 0.1, 3.0))).T)

    proj_antimatter = np.array([
        [2., 0., 0., 0., ],
        [0., -2., 0., 0., ],
        [0., 0., 1.0010010010010009, 1.],
        [0., 0., -0.20020020020020018, 0., ]
    ])

    wfov = math.atan((w/2)/f) * 2
    print(f"Camera {w}x{h} wfov={math.degrees(wfov):.1f}°")
    # proj_antimatter = np.array(glm.perspectiveLH_NO(wfov, w / float(h), 0.2, 200.0)).T
    # flip y

    return view_antimatter, proj_antimatter

#def get_view_proj_matrix():

def get_view_proj_matrix_garden():
    # correct view, (need to invert -f.y) eg glUniform2f(self.uFocalLoc, f, -f)
    proj = perspective(radians(80.0), w / h, 0.2, 200.0)

    # view = glm.lookAt(vec3(0,1.5,1), vec3(0,1,0), vec3(0,1,0))
    view = inverse(translate(mat4(1.0), vec3(0.0, 1.5, 0.0)))
    model = rotate(mat4(1), radians(180), vec3(0, 0, 1))

    fx = proj[0][0] * w / 2
    # fy = -proj[1][1] * h / 2

    # because of antimatter different order, and I don't want to change Jacobian calculation everywhere (keep it like antimatter)
    proj = glm.scale(glm.mat4(1.0), glm.vec3(1.0, -1.0, 1.0)) * proj #proj[1][1] *= -1 #flip y and more..
    view = glm.scale(glm.mat4(1.0), glm.vec3(1.0, -1.0, 1.0)) * view

    return np.array(view * model).T, np.array(proj).T, fx



if __name__ == "__main__":

    splatAxis = "../web/public/ds/axis.splat"
    splatTrain = "../web/public/ds/tmp/train.splat" # https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat
    splatGarden = "../web/public/ds/tmp/gs_garden_mipnerf360_vr.splat" # https://www.metalograms.com/ftp/gs/gs_garden_mipnerf360_vr.splat

    w, h, f = 1024, 1024, 1000

    # choose which splat file to use
    # splat, fn = splatAxis, 'axis'
    # splat, fn = splatTrain, 'train'
    splat, fn = splatGarden, 'garden'

    # choose which render to use
    # renderer, fn_render, og = SplatsRendererLoop(splat), "loop", False
    # renderer, fn_render, og = SplatsRendererNp(splat), "np", False
    # renderer, fn_render, og = SplatsRendererGl(splat, w, h), "gl", True
    # renderer, fn_render, og = SplatsRendererGlGeo(splat, w, h), "glgeo", True
    # renderer, fn_render, og = SplatsRendererGlGeoConic(splat, w, h), "glgeoconic", True
    # renderer, fn_render, og = SplatsRendererGlNoVertexSh(splat, w, h), "glnovertex", True
    renderer, fn_render, og = SplatsRendererVkGeo(splat, w, h), "vkgeo", True

    output = f"test/{fn}_{fn_render}.png"

    # view, proj = get_view_proj_matrix_antimatter()
    view, proj, f = get_view_proj_matrix_garden()

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

