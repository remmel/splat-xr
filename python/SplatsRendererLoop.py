import numpy as np

from utils import load_splat_file


class SplatsRendererLoop:
    def __init__(self, splat_file_path):
        self.points = load_splat_file(splat_file_path)

    def compute_cov3d_one(self, scale, rot):
        # Convert quaternion to rotation matrix
        qw, qx, qy, qz = rot
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]
        ])

        S = np.diag(scale)
        M = R @ S
        res = M @ M.T
        return res

    def compute_cov3d(self, scales, rots):
        qw, qx, qy, qz = rots.T
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]
        ]).transpose(2, 0, 1)

        S = scales.reshape(-1, 3, 1) * np.eye(3)
        M = R @ S
        res = M @ M.transpose(0, 2, 1)
        return res

    def render(self, view, proj, w, h, f):
        positions, scales, rots, colors = self.points
        uViewport = np.array([w, h])

        # Transform all points - (proj @ cam.T).T[70802, :] == (proj @ cam[70802, :])
        positions_v4 = np.hstack([positions, np.ones((len(positions), 1))])
        cam = (view @ positions_v4.T).T #cam = uView * center
        pos2d = (proj @ cam.T).T

        # frustum culling optimization
        clip = 1.2 * pos2d[:, 3]
        in_frustum = ~(
                (pos2d[:, 2] < -clip) |  # z < -clip
                (pos2d[:, 0] < -clip) |  # x < -clip
                (pos2d[:, 0] > clip) |  # x > clip
                (pos2d[:, 1] < -clip) |  # y < -clip
                (pos2d[:, 1] > clip)  # y > clip
        )
        # in_frustum[:],in_frustum[928310] = False, True #to try only x-th splat

        # positions, scales, rots, colors = positions[in_frustum], scales[in_frustum], rots[in_frustum], colors[in_frustum]
        # pos2d, cam = pos2d[in_frustum], cam[in_frustum]

        depths = cam[:, 2]

        vrk = 1 * self.compute_cov3d(scales, rots) #(n,3,3)}
        focal = np.array([f, f])

        depths_sq = depths * depths
        fx, fy = focal
        # Quicker to "bake" view in J like before?
        J = np.zeros((len(depths), 2, 3))
        J[:, 0, 0] = fx / depths
        J[:, 0, 2] = -fx * positions[:, 0] / depths_sq
        J[:, 1, 1] = -fy / depths
        J[:, 1, 2] = fy * positions[:, 1] / depths_sq

        T = view[:3, :3].T @ J.transpose(0, 2, 1)
        cov2d = T.transpose(0, 2, 1) @ vrk @ T

        # Compute eigenvalues and vectors for all points
        lambdas, diagonalVecs = np.linalg.eigh(cov2d)

        # Compute axes
        major_axes = np.minimum(np.sqrt(2 * lambdas[:, 1, None]), 1024) * diagonalVecs[:, :, 1]
        minor_axes = np.minimum(np.sqrt(2 * lambdas[:, 0, None]), 1024) * diagonalVecs[:, :, 0]

        center_ndc = pos2d[:, :2] / pos2d[:, 3:4]
        center_px = ndc_to_px(center_ndc, uViewport)

        img_rgba = np.zeros((h, w, 4), dtype=np.float32)


        indices = np.argsort(depths) # Sort by depth
        # indices = np.arange(0, len(depths))
        # indices = np.arange(0, len(depths))[::-1] #reversed
        #indices[0] == 100053

        axes_01 = (np.abs(major_axes) + np.abs(minor_axes)) / uViewport
        rect_min_ndc, rect_max_ndc = center_ndc - 4 * axes_01, center_ndc + 4 * axes_01
        rect_min_px, rect_max_px = ndc_to_px(rect_min_ndc, uViewport), ndc_to_px(rect_max_ndc, uViewport)
        # rect_min_px, rect_max_px = get_rect(center_ndc, major_axes, minor_axes, uViewport, 4) #gl_Positions

        for idx in indices:
            min_x, min_y = rect_min_px[idx, :]
            max_x, max_y = rect_max_px[idx, :]

            if (not in_frustum[idx]): continue

            valid_rect = min_x > 0 and min_y > 0 and max_x < w and max_y < h
            if not valid_rect: continue  # TODO handle that better, to avoid loosing edge splats

            major_len, minor_len = np.linalg.norm(major_axes[idx]), np.linalg.norm(minor_axes[idx])
            major_normlzd, minor_normlzd = major_axes[idx] / major_len, minor_axes[idx] / minor_len

            for x in range(int(min_x), int(max_x)):
                for y in range(int(min_y), int(max_y)):

                    # not rotated rect - for learning purposes
                    # dx = (x - center_px[idx, 0]) / (max_x - min_x) * 4
                    # dy = (y - center_px[idx, 1]) / (max_y - min_y) * 4

                    # rotated rect
                    delta_x_px, delta_y_px = x - center_px[idx, 0], y - center_px[idx, 1]
                    dx = (delta_x_px * major_normlzd[0] + delta_y_px * major_normlzd[1]) / major_len
                    dy = (delta_x_px * minor_normlzd[0] + delta_y_px * minor_normlzd[1]) / minor_len

                    A = dx**2+dy**2 #vPosition = np.array([dx, dy]) -np.dot(vPosition, vPosition)
                    if A > 4.0: continue
                    B = np.exp(-A) * colors[idx][3]
                    src_rgba = np.array([*colors[idx][:3] * B, B])

                    # Alpha blending
                    # gl.blendFunc(gl.ONE_MINUS_DST_ALPHA, gl.ONE)
                    # src_rgba * (1-dst_a) + dst_rgba * 1
                    if(img_rgba[y, x, 3] >= 1):
                        continue
                    img_rgba[y, x] = src_rgba * (1 - img_rgba[y, x,3]) + img_rgba[y,x] * 1
                    # img_rgba[y, x] = np.array([1.0,0.0,0.0, 1.0])

        img_rgba = np.flipud(img_rgba) # upside down
        img_rgb = img_rgba[:, :, :3]
        return (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)


# reproduce the antimatter code / opengl pipeline
def get_rect(center_f, major_axes, minor_axes, uViewport, qs = 2):
    a_positions = np.array([[-qs, -qs], [qs, -qs], [qs, qs], [-qs, qs]])  # quad
    nb_splats = center_f.shape[0]
    nb_quads = len(a_positions)  # 4
    nb_xy = center_f.shape[1]  # tuple x,y
    gl_Positions = np.zeros((nb_splats, nb_quads, nb_xy))  # (99,4,2)
    for idx, a_pos in enumerate(a_positions):
        gl_Position = (center_f
                       + a_pos[0] * major_axes / uViewport
                       + a_pos[1] * minor_axes / uViewport)  # (99,2)
        gl_Position_px = ((gl_Position + 1) * uViewport / 2).astype(int)
        gl_Positions[:, idx, :] = gl_Position_px
    rect_min, rect_max = gl_Positions.min(axis=1).astype(int), gl_Positions.max(axis=1).astype(int)
    return rect_min, rect_max

def ndc_to_px(f, viewport):
    return (f + 1) * viewport / 2