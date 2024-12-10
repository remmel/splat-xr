import numpy as np

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

    def compute_covariance(self, scale, rot):
        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = rot
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]
        ])

        S = np.diag(scale)
        M = R @ S
        return M @ M.T

    def compute_covariances(self, scales, rots):
        qx, qy, qz, qw = rots.T

        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]
        ]).transpose(2, 0, 1)

        S = scales.reshape(-1, 3, 1) * np.eye(3)
        M = R @ S
        return M @ M.transpose(0, 2, 1)

    def compute_cov2d(self, J, cov3d):
        # J: (N,2,3), cov3d: (N,3,3)
        temp = np.einsum('nij,njk->nik', J, cov3d)  # (N,2,3)
        return np.einsum('nij,nkj->nik', temp, J)  # (N,2,2)

    def render(self, view, proj, viewport: Viewport, focal_length):
        positions, scales, rots, colors = self.points

        # Transform all points
        points_homo = np.hstack([positions, np.ones((len(positions), 1))])
        cam_points = (view @ points_homo.T).T
        clip_points = (proj @ cam_points.T).T
        w = clip_points[:, 3:4]
        screen_points = clip_points[:, :2] / w
        depths = cam_points[:, 2]

        cov3d_all = self.compute_covariances(scales, rots)
        focal = np.array([focal_length, focal_length])

        # Compute Jacobians for all points
        depths_sq = depths * depths
        fx, fy = focal
        J = np.zeros((len(depths), 2, 3))
        J[:, 0, 0] = fx / depths
        J[:, 0, 2] = -fx * cam_points[:, 0] / depths_sq
        J[:, 1, 1] = -fy / depths
        J[:, 1, 2] = fy * cam_points[:, 1] / depths_sq

        # Compute 2D covariances
        # with timer("Operation"):
        #     cov2d_all = np.array([J[i] @ cov3d_all[i] @ J[i].T for i in range(len(J))])
        # with timer("Operation"):
        # np.all(cov2d_all == cov2d_all2)

        cov2d_all = self.compute_cov2d(J, cov3d_all)


        # Compute eigenvalues and vectors for all points
        eigvals_all, eigvecs_all = np.linalg.eigh(cov2d_all)

        # Compute axes
        major_axes = np.minimum(np.sqrt(2 * eigvals_all[:, 1, None]), 1024) * eigvecs_all[:, :, 1]
        minor_axes = np.minimum(np.sqrt(2 * eigvals_all[:, 0, None]), 1024) * eigvecs_all[:, :, 0]

        # Setup rendering buffers
        image = np.zeros((viewport.height, viewport.width, 3), dtype=np.float32)
        alpha = np.zeros((viewport.height, viewport.width), dtype=np.float32)

        # Sort by depth
        indices = np.argsort(-depths)
        #indices[0] == 100053

        for idx in indices:
            center = ((screen_points[idx] + 1) * viewport.width / 2).astype(int)
            radius = int(np.max([np.linalg.norm(major_axes[idx]), np.linalg.norm(minor_axes[idx])]))

            y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            pos = np.column_stack((x.flatten(), y.flatten()))

            transformed_pos = pos @ np.column_stack(
                (major_axes[idx] / viewport.width, minor_axes[idx] / viewport.height))
            gaussian = np.exp(-np.sum(transformed_pos ** 2, axis=1))

            valid_mask = gaussian > 0.01
            positions = center + pos[valid_mask]
            valid_px = (positions[:, 0] >= 0) & (positions[:, 0] < viewport.width) & \
                       (positions[:, 1] >= 0) & (positions[:, 1] < viewport.height)

            if np.any(valid_px):
                positions = positions[valid_px]
                gaussian = gaussian[valid_mask][valid_px]
                color = colors[idx]

                for p, g in zip(positions, gaussian):
                    x, y = p
                    a = g * color[3]
                    curr_alpha = alpha[y, x]
                    new_alpha = curr_alpha + a * (1 - curr_alpha)
                    if new_alpha > 0:
                        image[y, x] = (image[y, x] * curr_alpha + color[:3] * a * (1 - curr_alpha)) / new_alpha
                    alpha[y, x] = new_alpha

        return (image * 255).astype(np.uint8)