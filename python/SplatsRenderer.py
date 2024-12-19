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
        print(f"{np.sum(in_frustum):,}/{len(in_frustum):,} in frustum")

        # Update arrays to only include valid points
        positions = positions[in_frustum]
        scales = scales[in_frustum]
        rots = rots[in_frustum]
        colors = colors[in_frustum]
        pos2d = pos2d[in_frustum]
        cam = cam[in_frustum]

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
        minor_axes = np.minimum(np.sqrt(2 * lambdas[:, 0, None]), 1024) * diagonalVecs[:, :, 0] #FIXME different calculation

        center_f = pos2d[:, :2] / pos2d[:, 3:4] # position in screen coords
        center_px = ((center_f + 1) * uViewport / 2).astype(int)

        img_rgba = np.zeros((viewport.height, viewport.width, 4), dtype=np.float32) # Setup rendering buffers

        indices = np.argsort(depths) # Sort by depth
        # indices = np.arange(0, len(depths))
        #indices[0] == 100053

        # calculating the rectangle (quad min & max from gl_Position) - see get_rect
        axes_f = (abs(major_axes) + abs(minor_axes)) / uViewport #from px to [0,1]
        rect_min = (((center_f + -2 * axes_f)+1)*uViewport/2).astype(int) #in px, -2 is quad min
        rect_max = (((center_f + +2 * axes_f)+1)*uViewport/2).astype(int)

        for idx in indices:
            min_x, min_y = rect_min[idx, :]
            max_x, max_y = rect_max[idx, :]

            # if(not in_frustum[idx]): continue

            valid_rect = min_x > 0 and min_y > 0 and max_x < viewport.width and max_y < viewport.height
            if not valid_rect: continue #TODO handle that better, to avoid loosing edge splats

            quad_shape = (max_y-min_y, max_x-min_x)

            x_coords, y_coords = np.meshgrid(
                np.arange(min_x, max_x),
                np.arange(min_y, max_y),
            )

            dx = (x_coords - center_px[idx, 0]) / ((max_x - min_x) / 2) * 2
            dy = (y_coords - center_px[idx, 1]) / ((max_y - min_y) / 2) * 2

            A = -(dx**2 + dy**2)
            mask = A >= -4.0

            B = np.zeros_like(A)
            B[mask] = np.exp(A[mask]) * colors[idx][3]

            # Calculate color and alpha
            src_rgba = np.dstack((colors[idx][:3] * B[:, :, np.newaxis], B))

            # frag_rgb[:, :] = [1, 0, 0]
            # frag_alpha[:, :] = 1

            # Get current values for the region
            region_rgba = img_rgba[min_y:max_y, min_x:max_x] #dst

            blend_mask = (region_rgba[:,:,3] < 1) & mask # Create mask for non-saturated pixels (alpha)

            # Update only valid pixels
            if np.any(blend_mask):
                x_idcs, y_idcs = np.where(blend_mask)

                # Perform blending for valid pixels
                # gl.blendFunc(gl.ONE_MINUS_DST_ALPHA, gl.ONE)
                # dst  = src * (1-dst_a) + dst * 1 <=> dst += src * (1-dst_a)
                region_rgba[x_idcs, y_idcs] += src_rgba[x_idcs, y_idcs] * (1 - region_rgba[x_idcs, y_idcs, 3, np.newaxis])

            # Update the original arrays
            img_rgba[min_y:max_y, min_x:max_x] = region_rgba

        alpha_mask = img_rgba[:,:, 3] > 0
        img_rgba[alpha_mask, :3] /= img_rgba[alpha_mask, 3:4] # unpremult: img.rgb /= img.alpha
        img_rgba = img_rgba[::-1, :] # image origin was top-left so flip y axis
        return (np.clip(img_rgba, 0, 1) * 255).astype(np.uint8)
