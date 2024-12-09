import numpy as np

from python.utils import Viewport


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

    def render(self, view_matrix, proj, viewport: Viewport, focal_length):
        positions, scales, rots, colors = self.points

        # Transform to camera space
        points_homo = np.hstack([positions, np.ones((len(positions), 1))])
        cam_points = (view_matrix @ points_homo.T).T

        # Project to screen space
        clip_points = (proj @ cam_points.T).T
        w = clip_points[:, 3:4]
        screen_points = clip_points[:, :2] / w
        depths = cam_points[:, 2]

        # Create image buffers
        image = np.zeros((viewport.height, viewport.width, 3), dtype=np.float32)
        alpha = np.zeros((viewport.height, viewport.width), dtype=np.float32)

        # Sort by depth
        indices = np.argsort(-depths)

        # Compute focal vector
        focal = np.array([focal_length, focal_length])

        for idx in indices:
            # Compute 2D covariance
            cov3d = self.compute_covariance(scales[idx], rots[idx])
            depth = depths[idx]

            J = np.array([
                [focal[0] / depth, 0, -focal[0] * cam_points[idx, 0] / (depth * depth)],
                [0, -focal[1] / depth, focal[1] * cam_points[idx, 1] / (depth * depth)]
            ])

            cov2d = J @ cov3d @ J.T

            # Compute eigenvalues and vectors
            eigvals, eigvecs = np.linalg.eigh(cov2d)

            # Compute major and minor axes
            major_axis = np.minimum(np.sqrt(2 * eigvals[1]), 1024) * eigvecs[:, 1]
            minor_axis = np.minimum(np.sqrt(2 * eigvals[0]), 1024) * eigvecs[:, 0]

            # Generate gaussian splat
            center = ((screen_points[idx] + 1) * viewport.width / 2).astype(int)
            radius = int(np.max([np.linalg.norm(major_axis), np.linalg.norm(minor_axis)]))

            y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            pos = np.column_stack((x.flatten(), y.flatten()))

            # Transform positions to eigen space
            transformed_pos = pos @ np.column_stack((major_axis / viewport.width, minor_axis / viewport.height))
            gaussian = np.exp(-np.sum(transformed_pos ** 2, axis=1))

            valid_mask = gaussian > 0.01
            positions = center + pos[valid_mask]
            valid_px = (positions[:, 0] >= 0) & (positions[:, 0] < viewport.width) & \
                       (positions[:, 1] >= 0) & (positions[:, 1] < viewport.height)

            if np.any(valid_px):
                positions = positions[valid_px]
                gaussian = gaussian[valid_mask][valid_px]
                color = colors[idx]

                # Alpha blending
                for p, g in zip(positions, gaussian):
                    x, y = p
                    a = g * color[3]
                    curr_alpha = alpha[y, x]
                    new_alpha = curr_alpha + a * (1 - curr_alpha)
                    if new_alpha > 0:
                        image[y, x] = (image[y, x] * curr_alpha + color[:3] * a * (1 - curr_alpha)) / new_alpha #antimatter gl.ONE_MINUS_DST_ALPHA, gl.ONE
                        # image[y, x] = image[y, x] * (1 - curr_alpha) + color[:3] * curr_alpha # gl.ONE, gl.ONE_MINUS_DST_ALPHA
                    alpha[y, x] = new_alpha

        return (image * 255).astype(np.uint8)