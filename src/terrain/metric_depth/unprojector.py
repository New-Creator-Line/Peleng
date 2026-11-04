"""
3D Metric Depth Unprojection and Multi-View Voxel Fusion Module.

Implements:
1. Pinhole inverse projection:
       P(u, v) = C_c + D(u, v) * R_c * K_c^{-1} * [u, v, 1]^T
   mapping depth pixels to Unreal Engine centimeters (UE5_cm).
2. Outlier rejection clipping to the Peleng bounding box:
       X in [-40000.0, 20000.0] cm
       Y in [-30000.0, 30000.0] cm
       Z in [0.0, 5000.0] cm
3. Multi-camera voxel grid fusion (step 100 cm) with centroid and RGB blending.
4. Fast statistical / density outlier rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.camera_model import CameraModel


DEFAULT_BOUNDS = {
    "x_min": -40000.0,
    "x_max": 20000.0,
    "y_min": -30000.0,
    "y_max": 30000.0,
    "z_min": 0.0,
    "z_max": 5000.0,
}


def filter_points_by_bounds(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    bounds: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Removes points lying outside the valid spatial bounding envelope.
    """
    if len(points) == 0:
        return points, colors

    b = bounds if bounds is not None else DEFAULT_BOUNDS
    x_min, x_max = b.get("x_min", -40000.0), b.get("x_max", 20000.0)
    y_min, y_max = b.get("y_min", -30000.0), b.get("y_max", 30000.0)
    z_min, z_max = b.get("z_min", 0.0), b.get("z_max", 5000.0)

    px = points[:, 0]
    py = points[:, 1]
    pz = points[:, 2]

    in_bounds = (
        (px >= x_min)
        & (px <= x_max)
        & (py >= y_min)
        & (py <= y_max)
        & (pz >= z_min)
        & (pz <= z_max)
        & np.isfinite(px)
        & np.isfinite(py)
        & np.isfinite(pz)
    )

    filtered_pts = points[in_bounds]
    filtered_cols = colors[in_bounds] if colors is not None else None
    return filtered_pts, filtered_cols


def unproject_depth_map(
    depth_map: np.ndarray,
    camera_model: CameraModel,
    rgb_frame: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unprojects a 2D depth map D(u, v) into 3D world coordinates:
        P(u, v) = C_c + D(u, v) * R_c * K_c^{-1} * [u, v, 1]^T
    where R_c = camera_model.R_cam.T (transforming camera ray to world ray).
    Returns:
        points_3d: (N, 3) float64 in UE5_cm
        colors_rgb: (N, 3) uint8 [0, 255]
    """
    h_d, w_d = depth_map.shape
    fx_orig, fy_orig, cx_orig, cy_orig = camera_model.get_intrinsics()

    scale_x = w_d / float(camera_model.image_width_px)
    scale_y = h_d / float(camera_model.image_height_px)
    fx = fx_orig * scale_x
    fy = fy_orig * scale_y
    cx = cx_orig * scale_x
    cy = cy_orig * scale_y

    u_coords = np.linspace(0, w_d - 1, w_d, dtype=np.float64)
    v_coords = np.linspace(0, h_d - 1, h_d, dtype=np.float64)
    uu, vv = np.meshgrid(u_coords, v_coords)

    # Ray in camera coordinates
    rx = (uu - cx) / fx
    ry = (vv - cy) / fy
    rz = np.ones_like(rx)
    ray_cam = np.stack([rx, ry, rz], axis=-1)  # (H, W, 3)

    # Ray in world coordinates (ray_cam @ R_cam is equivalent to R_cam.T @ ray_cam)
    ray_world = ray_cam @ camera_model.R_cam  # (H, W, 3)

    # 3D World Points: P = C_c + D(u, v) * ray_world
    points_3d_grid = camera_model.loc + depth_map[..., None] * ray_world  # (H, W, 3)

    # Selection mask
    active_mask = (depth_map > 1.0) & np.isfinite(depth_map)
    if valid_mask is not None:
        active_mask = active_mask & valid_mask

    flat_pts = points_3d_grid[active_mask]  # (N, 3)

    # Colors
    if rgb_frame is not None:
        if rgb_frame.shape[:2] != (h_d, w_d):
            from PIL import Image

            pil_rgb = Image.fromarray(rgb_frame).resize((w_d, h_d), Image.Resampling.BILINEAR)
            rgb_arr = np.asarray(pil_rgb, dtype=np.uint8)
        else:
            rgb_arr = rgb_frame.astype(np.uint8)
        flat_cols = rgb_arr[active_mask]
    else:
        flat_cols = np.full((len(flat_pts), 3), 128, dtype=np.uint8)

    return flat_pts, flat_cols


def fuse_voxel_grid(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel_size_cm: float = 100.0,
    min_points_per_voxel: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuses 3D points into a regular voxel grid of resolution `voxel_size_cm`.
    Computes centroid and blended color for each occupied voxel.
    """
    if len(points) == 0:
        empty_pts = np.empty((0, 3), dtype=np.float64)
        empty_cols = np.empty((0, 3), dtype=np.uint8)
        return empty_pts, empty_cols

    voxel_indices = np.floor(points / voxel_size_cm).astype(np.int64)
    _, inv_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    counts = np.bincount(inv_indices)
    valid_voxels = counts >= min_points_per_voxel

    x_sum = np.bincount(inv_indices, weights=points[:, 0])
    y_sum = np.bincount(inv_indices, weights=points[:, 1])
    z_sum = np.bincount(inv_indices, weights=points[:, 2])

    fused_x = (x_sum / counts)[valid_voxels]
    fused_y = (y_sum / counts)[valid_voxels]
    fused_z = (z_sum / counts)[valid_voxels]
    fused_pts = np.column_stack([fused_x, fused_y, fused_z])

    if colors is not None and len(colors) == len(points):
        cols_float = colors.astype(np.float64)
        r_sum = np.bincount(inv_indices, weights=cols_float[:, 0])
        g_sum = np.bincount(inv_indices, weights=cols_float[:, 1])
        b_sum = np.bincount(inv_indices, weights=cols_float[:, 2])

        fused_r = np.clip((r_sum / counts)[valid_voxels], 0, 255).astype(np.uint8)
        fused_g = np.clip((g_sum / counts)[valid_voxels], 0, 255).astype(np.uint8)
        fused_b = np.clip((b_sum / counts)[valid_voxels], 0, 255).astype(np.uint8)
        fused_cols = np.column_stack([fused_r, fused_g, fused_b])
    else:
        fused_cols = np.full((len(fused_pts), 3), 128, dtype=np.uint8)

    return fused_pts, fused_cols


class MetricDepthUnprojector:
    """
    Handles unprojection of multi-view depth maps, coordinate clipping,
    and unified 3D voxel fusion.
    """

    def __init__(
        self,
        voxel_size_cm: float = 100.0,
        bounds: Optional[Dict[str, float]] = None,
        min_points_per_voxel: int = 1,
    ) -> None:
        self.voxel_size_cm = float(voxel_size_cm)
        self.bounds = bounds if bounds is not None else DEFAULT_BOUNDS
        self.min_points_per_voxel = int(min_points_per_voxel)

    def process_camera(
        self,
        camera_model: CameraModel,
        depth_map: np.ndarray,
        rgb_frame: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unprojects a single camera's depth map and filters against scene bounds.
        """
        pts, cols = unproject_depth_map(depth_map, camera_model, rgb_frame, valid_mask)
        pts_filtered, cols_filtered = filter_points_by_bounds(pts, cols, self.bounds)
        if cols_filtered is None:
            cols_filtered = np.full((len(pts_filtered), 3), 128, dtype=np.uint8)
        return pts_filtered, cols_filtered

    def unproject_and_fuse(
        self,
        depth_dict: Dict[int, np.ndarray],
        cameras: Dict[int, CameraModel],
        rgb_dict: Optional[Dict[int, np.ndarray]] = None,
        valid_masks: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unprojects all cameras, filters bounding box, and fuses into unified voxel grid.
        Returns:
            vertices: (N, 3) float64
            colors: (N, 3) uint8 [0, 255]
        """
        all_pts: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []

        for cam_id, depth_map in depth_dict.items():
            if cam_id not in cameras:
                continue
            model = cameras[cam_id]
            rgb = rgb_dict.get(cam_id) if rgb_dict else None
            mask = valid_masks.get(cam_id) if valid_masks else None

            pts, cols = self.process_camera(model, depth_map, rgb, mask)
            if len(pts) > 0:
                all_pts.append(pts)
                all_cols.append(cols)

        if not all_pts:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)

        concat_pts = np.vstack(all_pts)
        concat_cols = np.vstack(all_cols)

        # Unified voxel grid downsampling
        fused_pts, fused_cols = fuse_voxel_grid(
            concat_pts,
            concat_cols,
            voxel_size_cm=self.voxel_size_cm,
            min_points_per_voxel=self.min_points_per_voxel,
        )

        return fused_pts, fused_cols
