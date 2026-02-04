import numpy as np
from tqdm import tqdm

from utils import load_splat_file, remove_alpha


class SplatsRendererNp:
    def __init__(self, splat_file_path):
        self.points = load_splat_file(splat_file_path)

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

    def compute_jacobian(self, cam_points, f):
        x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
        fx, fy = f, f

        # Compute Jacobian matrices for all points
        J = np.zeros((len(cam_points), 3, 3))
        J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = fx / z, 0, -(fx * x) / (z * z)
        J[:, 1, 0], J[:, 1, 1], J[:, 1, 2] = 0, -fy / z, (fy * y) / (z * z)
        J[:, 2, 0], J[:, 2, 1], J[:, 2, 2] = 0, 0, 0

        return J

    def compute_cov2d(self, view, vrk, J):
        """Compute 2D covariance matrices for all points."""
        T = view[:3, :3].T @ J.transpose(0, 2, 1)  # should I transpose the view, try with view rotated
        cov2d = T.transpose(0, 2, 1) @ vrk @ T

        cov2d_batch = cov2d[:, :2, :2]  # Take only the 2x2 part
        eigenvals, eigenvecs = np.linalg.eigh(cov2d_batch)
        lambda2, lambda1 = eigenvals[:, 0, np.newaxis], eigenvals[:, 1, np.newaxis]

        # Calculate major and minor axes
        major_axis = np.minimum(np.sqrt(2 * lambda1), 1024) * eigenvecs[:, :, 1]
        minor_axis = np.minimum(np.sqrt(2 * lambda2), 1024) * eigenvecs[:, :, 0]
        # minor_axis = np.minimum(np.sqrt(2 * lambda2), 1024) * np.column_stack([eigenvecs[:, 1, 1],-eigenvecs[:, 0, 1]])

        # Filter out degenerate gaussians
        valid = lambda2.squeeze() >= 0

        return major_axis, minor_axis, valid

    def render(self, view, proj, w, h, f):
        assert view.shape == (4, 4);
        assert proj.shape == (4, 4)

        positions, scales, rots, colors = self.points

        # Transform all points - (proj @ cam.T).T[70802, :] == (proj @ cam[70802, :])
        positions_v4 = np.hstack([positions, np.ones((len(positions), 1))])
        cam = (view @ positions_v4.T).T  # cam = uView * center
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
        # in_frustum[np.random.rand(in_frustum.shape[0]) < 0.995] = False #keep only 0.5% of the splats
        # in_frustum[:] = False
        # in_frustum[928310] = True
        print(f"{np.sum(in_frustum):,}/{len(in_frustum):,} in frustum")

        # Update arrays to only include valid points
        positions, scales, rots, colors = positions[in_frustum], scales[in_frustum], rots[in_frustum], colors[in_frustum]
        pos2d, cam = pos2d[in_frustum], cam[in_frustum]

        vrks = self.compute_cov3d(scales, rots)  # (n,3,3)}

        J = self.compute_jacobian(cam, f)

        major_axis, minor_axis, valid = self.compute_cov2d(view, vrks, J)

        uViewport = np.array([w, h])
        center_ndc = pos2d[:, :2] / pos2d[:, 3:4]  # position in screen coords
        center_px = ndc_to_px(center_ndc, uViewport).astype(int)

        img_rgba = np.zeros((h, w, 4), dtype=np.float32)  # Setup rendering buffers

        debug_contribution_map = np.zeros((h, w), dtype=np.float32)
        debug_contribution_map_ellipsis = np.zeros((h, w), dtype=np.float32)
        debug_contribution_map_rect_aabb = np.zeros((h, w), dtype=np.float32) #not squared like Inria
        debug_contribution_map_unvisible = np.zeros((h, w), dtype=np.float32)
        # TODO add contribution_map_rect_obb and contribution_map_rect_aabb_squared

        indices = np.argsort(pos2d[:, 2])  # Sort by depth
        # indices = np.arange(0, len(depths))
        # indices[0] == 100053

        # calculating the rectangle (quad min & max from gl_Position) - see get_rect
        axes_01 = (abs(major_axis) + abs(minor_axis)) / uViewport  # from px to [0,1]
        rect_min_ndc = center_ndc + -4 * axes_01  # [-1,1]
        rect_max_ndc = center_ndc + +4 * axes_01  # [-1,1]
        rect_min_px, rect_max_px = ndc_to_px(rect_min_ndc, uViewport).astype(int), ndc_to_px(rect_max_ndc, uViewport).astype(int)
        rect_min_px, rect_max_px = np.maximum(0, rect_min_px), np.minimum(uViewport, rect_max_px)
        rect_size_px = rect_max_px - rect_min_px

        for idx in tqdm(indices, desc="rendering by splat"):
            min_x_px, min_y_px = rect_min_px[idx, :]
            max_x_px, max_y_px = rect_max_px[idx, :]

            # discard splat which rectangle is outside the screen (but center inside frustum * 1.2)
            if min_x_px >= max_x_px or min_y_px >= max_y_px:
                continue

            rect_shape = (max_y_px - min_y_px, max_x_px - min_x_px)  # (rect_size_px[idx, 1], rect_size_px[idx, 0])

            x_coords, y_coords = np.meshgrid(
                np.arange(min_x_px, max_x_px),
                np.arange(min_y_px, max_y_px),
            )

            delta_px = np.stack([
                x_coords - center_px[idx, 0],
                y_coords - center_px[idx, 1]
            ], axis=-1)  # (rect_h_px, rect_w_px, 2)

            # not rotated rect <=> localPos = (delta_px / rect_size_px[idx]) * 4 # [-2,2]
            # dx = (x_coords - center_px[idx, 0]) / (max_x_px - min_x_px) * 4
            # dy = (y_coords - center_px[idx, 1]) / (max_y_px - min_y_px) * 4

            # rotated rect
            major_axis_length, minor_axis_length = np.linalg.norm(major_axis[idx]), np.linalg.norm(minor_axis[idx])
            major_axis_normalized, minor_axis_normalized = major_axis[idx] / major_axis_length, minor_axis[idx] / minor_axis_length
            dx = (delta_px[:, :, 0] * major_axis_normalized[0] + delta_px[:, :, 1] * major_axis_normalized[1]) / major_axis_length
            dy = (delta_px[:, :, 0] * minor_axis_normalized[0] + delta_px[:, :, 1] * minor_axis_normalized[1]) / minor_axis_length

            assert dx.shape == rect_shape and dy.shape == rect_shape
            A = dx ** 2 + dy ** 2
            mask = A <= 4.0

            B = np.zeros_like(A)
            B[mask] = np.exp(-A[mask]) * colors[idx][3]

            # Calculate color and alpha
            src_rgba = np.dstack((colors[idx][:3] * B[:, :, np.newaxis], B))
            # src_rgba = np.dstack((np.array([1.0,0,0]) * np.ones_like(B[:, :, np.newaxis]),  np.ones_like(B)))

            # frag_rgb[:, :] = [1, 0, 0]
            # frag_alpha[:, :] = 1

            # Get current values for the region
            region_rgba = img_rgba[min_y_px:max_y_px, min_x_px:max_x_px]  # dst

            blend_mask = (region_rgba[:, :, 3] < 1) & mask  # Create mask for non-saturated pixels (alpha)

            # Update only valid pixels
            if np.any(blend_mask):
                x_idcs, y_idcs = np.where(blend_mask)

                # Perform blending for valid pixels
                # gl.blendFunc(gl.ONE_MINUS_DST_ALPHA, gl.ONE) #antimatter
                # dst  = src * (1-dst_a) + dst * 1 <=> dst += src * (1-dst_a)
                region_rgba[x_idcs, y_idcs] += src_rgba[x_idcs, y_idcs] * (1 - region_rgba[x_idcs, y_idcs, 3, np.newaxis])

            # Update the original arrays
            # region_rgba = np.broadcast_to(np.append(np.random.rand(3), 1.0), region_rgba.shape) #put random color for debug purpose
            img_rgba[min_y_px:max_y_px, min_x_px:max_x_px] = region_rgba

            #pixels contribution
            debug_contribution_map[min_y_px:max_y_px, min_x_px:max_x_px] += blend_mask #356
            debug_contribution_map_rect_aabb[min_y_px:max_y_px, min_x_px:max_x_px]+= 1.0 #1093
            debug_contribution_map_ellipsis[min_y_px:max_y_px, min_x_px:max_x_px] += mask #A <= 4.0 #397
            debug_contribution_map_unvisible[min_y_px:max_y_px, min_x_px:max_x_px] += ~blend_mask #1006

        img_rgba = np.flipud(img_rgba)  # upside down
        img_rgb = img_rgba[:, :, :3]
        return (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)


def ndc_to_px(f, viewport):
    return (f + 1) * viewport / 2