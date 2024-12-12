import numpy as np
from python.utils import create_projection_matrix, Viewport


class PointsRenderer:
    def __init__(self, splat_file_path):
        self.points = self.load_splat_file(splat_file_path)

    def load_splat_file(self, file_path):
        with open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)

        row_length = 3*4 + 3*4 + 4 + 4  # position + scale + rgba + quaternion
        vertex_count = len(data) // row_length

        # Reshape data for easier access
        data = data.reshape(vertex_count, row_length)

        # Extract positions (xyz)
        positions = np.frombuffer(data[:, :12].tobytes(), dtype=np.float32)
        positions = positions.reshape(-1, 3)

        # Extract colors (rgba)
        colors = data[:, 24:28].astype(np.float32) / 255.0

        return np.hstack([positions, colors])

    def render(self, view_matrix, proj, viewport: Viewport, f):
        # Transform points
        points = self.points[:, :3]  # xyz coordinates
        colors = self.points[:, 3:]  # rgba values

        # Apply view transform
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        view_points = (view_matrix @ points_homo.T).T[:, :3]

        # Project to screen space
        clip_points = (proj @ np.hstack([view_points, np.ones((len(view_points), 1))]).T).T
        w = clip_points[:, 3:4]
        screen_points = clip_points[:, :3] / w

        # Convert to pixel coordinates
        pixel_x = ((screen_points[:, 0] + 1) * viewport.width / 2).astype(int)
        pixel_y = ((screen_points[:, 1] + 1) * viewport.height / 2).astype(int)

        # Sort points by depth
        depth = view_points[:, 2]
        sorted_indices = np.argsort(-depth)

        # Create image array
        image = np.zeros((viewport.height, viewport.width, 3), dtype=np.float32)
        alpha = np.zeros((viewport.height, viewport.width), dtype=np.float32)

        # Filter valid pixels
        valid = (pixel_x >= 0) & (pixel_x < viewport.width) & \
                (pixel_y >= 0) & (pixel_y < viewport.height)

        # Render points from back to front
        for idx in sorted_indices[valid[sorted_indices]]:
            x, y = pixel_x[idx], pixel_y[idx]
            color = colors[idx, :3]
            a = colors[idx, 3]

            # Blend colors
            current_alpha = alpha[y, x]
            new_alpha = current_alpha + a * (1 - current_alpha)
            if new_alpha > 0:
                image[y, x] = (image[y, x] * current_alpha + color * a * (1 - current_alpha)) / new_alpha
            alpha[y, x] = new_alpha

        return (image * 255).astype(np.uint8)