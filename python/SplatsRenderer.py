import numpy as np
import scipy

from python.utils import Viewport, timer


class SplatsRenderer:
    def __init__(self, splat_file_path):
        self.points = self.load_splat_file(splat_file_path)

    def load_splat_file(self, file_path):
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
        return res * 4

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

    def render(self, view, proj, viewport: Viewport, focal_length):
        positions, scales, rots, colors = self.points
        uViewport = np.array([viewport.width, viewport.height]) # TODO remove viewport class

        # Transform all points - (proj @ cam.T).T[70802, :] == (proj @ cam[70802, :])
        positions_v4 = np.hstack([positions, np.ones((len(positions), 1))])
        cam = (view @ positions_v4.T).T #cam = uView * center
        pos2d = (proj @ cam.T).T

        # Add frustum culling optimization
        clip = 1.2 * pos2d[:, 3]
        in_frustum = ~(
                (pos2d[:, 2] < -clip) |  # z < -clip
                (pos2d[:, 0] < -clip) |  # x < -clip
                (pos2d[:, 0] > clip) |  # x > clip
                (pos2d[:, 1] < -clip) |  # y < -clip
                (pos2d[:, 1] > clip)  # y > clip
        )

        # Update arrays to only include valid points
        # positions = positions[in_frustum]
        # scales = scales[in_frustum]
        # rots = rots[in_frustum]
        # colors = colors[in_frustum]
        # pos2d = pos2d[in_frustum]
        # cam = cam[in_frustum]

        depths = cam[:, 2]

        vrk = 4 * self.compute_cov3d(scales, rots) #(n,3,3)}
        focal = np.array([focal_length, focal_length])

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
        minor_axes = -1 * np.minimum(np.sqrt(2 * lambdas[:, 0, None]), 1024) * diagonalVecs[:, :, 0] #FIXME different calculation

        center_f = pos2d[:, :2] / pos2d[:, 3:4] # position in screen coords
        means2D = center_px_all = ((center_f + 1) * uViewport / 2).astype(int)

        # a_pos = np.array([2.0,2.0])

        # Setup rendering buffers
        image = np.zeros((viewport.height, viewport.width, 3), dtype=np.float32)
        alpha = np.zeros((viewport.height, viewport.width), dtype=np.float32)

        # Sort by depth
        indices = np.argsort(-depths)
        #indices[0] == 100053
        # radii = get_radius(cov2d)
        # (rect_min0, rect_max0) = get_rect(center_px_all, radii, viewport.width, viewport.height)

        # gl_Position = (center_f[idx_expanded] +  # Shape: (N, 1, 2)
        #                a_pos_expanded[..., 0] * major_axes[idx_expanded] / uViewport +  # Shape: (N, 4, 2)
        #                a_pos_expanded[..., 1] * minor_axes[idx_expanded] / uViewport)

        a_positions = np.array([[-2,-2], [2,-2], [2,2], [-2,2]])

        nb_splats = center_f.shape[0]
        nb_quads = len(a_positions) #4
        nb_xy = center_f.shape[1]# tuple x,y
        gl_Positions = np.zeros((nb_splats, nb_quads, nb_xy)) #(99,4,2)
        rect_min = rect_max = np.zeros((center_f.shape[0], nb_xy)) #(99,2)
        for i in np.arange(len(a_positions)):
            gl_Position = (center_f
                           + a_positions[i][0] * major_axes / uViewport
                           + a_positions[i][1] * minor_axes / uViewport) #(99,2)
            gl_Position_px = ((gl_Position + 1) * uViewport / 2).astype(int)
            gl_Positions[:, i, :] = gl_Position_px

        rect_min = gl_Positions.min(axis=1).astype(int) #(99,2)
        rect_max = gl_Positions.max(axis=1).astype(int) #(99,2)


        # for idx in indices:
        #     for gl_Position_px in gl_Positions[idx, :]:
        #         # gl_Position_px = ((gl_Position + 1) * uViewport / 2).astype(int)
        #         x,y=gl_Position_px
        #         image[int(x),int(y)] = np.array(colors[idx][:3])

        for idx in indices:
            min_x, min_y = rect_min[idx, :]
            max_x, max_y = rect_max[idx, :]

            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    dx = (x - center_px_all[idx, 0]) / ((max_x - min_x)/2) * 2 #*2 because quad2 ?
                    dy = (y - center_px_all[idx, 1]) / ((max_y - min_y)/2) * 2

                    vPosition = np.array([dx, dy])

                    A = -np.dot(vPosition, vPosition) #-(dx**2+dy**2)
                    if A < -4.0: continue
                    B = np.exp(A) * colors[idx][3]

                    color_ = np.array(colors[idx][:3]) * B
                    alpha_ = B



                    # Blend ONE_MINUS_DST_ALPHA, ONE
                    dst_alpha = alpha[x, y]
                    image[x,y] = image[x,y] + color_ * (1- dst_alpha)
                    alpha[x,y] = alpha[x,y] + alpha_

                    # if (x == 462 and y == 462):
                    #     print(image[x,y])




        # print(1)




            # rect_min_x, rect_min_y = rect_min[idx, :]
            # rect_max_x, rect_max_y = rect_max[idx, :]
            # image[int(rect_min_x), int(rect_min_y)] = np.array(colors[idx][:3])
            # image[int(rect_max_x), int(rect_max_y)] = np.array(colors[idx][:3])






            # center_px = center_px_all[idx, :]
            # radius = int(np.max([np.linalg.norm(major_axes[idx]), np.linalg.norm(minor_axes[idx])]))
            #
            # y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            # pos = np.column_stack((x.flatten(), y.flatten()))
            #
            # transformed_pos = pos @ np.column_stack(
            #     (major_axes[idx] / viewport.width, minor_axes[idx] / viewport.height))
            # gaussian = np.exp(-np.sum(transformed_pos ** 2, axis=1))
            #
            # valid_mask = gaussian > 0.05
            # positions = center_px + pos[valid_mask]
            # valid_px = (positions[:, 0] >= 0) & (positions[:, 0] < viewport.width) & \
            #            (positions[:, 1] >= 0) & (positions[:, 1] < viewport.height)
            #
            # if np.any(valid_px):
            #     positions = positions[valid_px]
            #     gaussian = gaussian[valid_mask][valid_px]
            #     color = colors[idx]
            #
            #     for p, g in zip(positions, gaussian):
            #         x, y = p
            #         a = g * color[3]
            #         curr_alpha = alpha[y, x]
            #         # new_alpha = curr_alpha + a * (1 - curr_alpha)
            #         # if new_alpha > 0:
            #         #     image[y, x] = (image[y, x] * curr_alpha + color[:3] * a * (1 - curr_alpha)) / new_alpha
            #         # alpha[y, x] = new_alpha
            #         image[y, x] = np.array(color[:3])

        return (image * 255).astype(np.uint8)


def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + np.sqrt(np.clip(mid**2-det, a_min=0.1, a_max=None))
    lambda2 = mid - np.sqrt(np.clip(mid**2-det, a_min=0.1, a_max=None))
    return np.ceil(3.0 * np.sqrt(np.maximum(lambda1, lambda2)))