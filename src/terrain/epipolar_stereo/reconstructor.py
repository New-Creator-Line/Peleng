"""
Multi-View Triangulation and Bounding Box Filtering Module (Task 014: Method 1).

Implements:
1. Linear Direct Linear Transform (DLT) triangulation via NumPy SVD.
2. Reprojection error calculation and threshold filtering (err <= 3.5 px).
3. Unreal Engine 5 spatial envelope filtering:
   X in [-40000.0, 20000.0] cm
   Y in [-30000.0, 30000.0] cm
   Z in [0.0, 5000.0] cm
4. Assembly into standard TerrainPayload dictionary conforming to terrain_method_contract.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.camera_model import CameraModel
from src.terrain.epipolar_stereo.matcher import StereoMatch


# Spatial bounding envelope from terrain_method_contract.md (UE5_cm)
BOUNDS_X_MIN: float = -40000.0
BOUNDS_X_MAX: float = 20000.0
BOUNDS_Y_MIN: float = -30000.0
BOUNDS_Y_MAX: float = 30000.0
BOUNDS_Z_MIN: float = 0.0
BOUNDS_Z_MAX: float = 5000.0


@dataclass(frozen=True)
class TriangulatedPoint:
    """Represents a validated 3D landmark reconstructed from two or more cameras."""
    point_3d_ue: np.ndarray  # (3,) in UE5 centimeters
    reprojection_error_px: float
    ncc: float
    color_rgb: Tuple[int, int, int]


def dlt_triangulate_point(
    projection_matrices: Sequence[np.ndarray],
    points_2d: Sequence[Tuple[float, float]],
) -> Tuple[Optional[np.ndarray], float]:
    """
    Triangulates a 3D point from 2 or more camera views using Direct Linear Transform (DLT) via SVD.
    
    Args:
        projection_matrices: List of 3x4 projection matrices.
        points_2d: List of (u, v) pixel observations in matching order.
        
    Returns:
        (point_3d, mean_reprojection_error) or (None, inf) if triangulation fails.
    """
    n_views = len(projection_matrices)
    if n_views < 2 or len(points_2d) < 2 or n_views != len(points_2d):
        return None, float("inf")

    # Construct 2M x 4 linear system A @ X = 0
    A = np.zeros((2 * n_views, 4), dtype=np.float64)
    for i, (P, (u, v)) in enumerate(zip(projection_matrices, points_2d)):
        A[2 * i] = u * P[2, :] - P[0, :]
        A[2 * i + 1] = v * P[2, :] - P[1, :]

    # Solve via Singular Value Decomposition
    try:
        _, _, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None, float("inf")

    X_hom = vh[-1]
    w = float(X_hom[3])
    if abs(w) < 1e-9:
        return None, float("inf")

    point_3d = X_hom[:3] / w

    # Calculate mean reprojection error across all views
    total_err = 0.0
    valid_cams = 0
    hom_pt = np.append(point_3d, 1.0)

    for P, (u, v) in zip(projection_matrices, points_2d):
        proj_h = P @ hom_pt
        z_cam = float(proj_h[2])
        if z_cam <= 1e-4:
            # Point is behind or on the camera projection plane
            return None, float("inf")

        u_proj = float(proj_h[0] / z_cam)
        v_proj = float(proj_h[1] / z_cam)
        err = float(np.hypot(u_proj - u, v_proj - v))
        total_err += err
        valid_cams += 1

    if valid_cams == 0:
        return None, float("inf")

    mean_err = total_err / valid_cams
    return point_3d, mean_err


def is_point_within_bounds(
    pt_3d: np.ndarray,
    x_min: float = BOUNDS_X_MIN,
    x_max: float = BOUNDS_X_MAX,
    y_min: float = BOUNDS_Y_MIN,
    y_max: float = BOUNDS_Y_MAX,
    z_min: float = BOUNDS_Z_MIN,
    z_max: float = BOUNDS_Z_MAX,
) -> bool:
    """Checks whether a 3D point lies strictly within the Peleng UE5 spatial envelope."""
    x, y, z = float(pt_3d[0]), float(pt_3d[1]), float(pt_3d[2])
    return (
        (x_min <= x <= x_max)
        and (y_min <= y <= y_max)
        and (z_min <= z <= z_max)
    )


class EpipolarReconstructor:
    """Reconstructs 3D terrain points from stereo feature matches and builds TerrainPayload."""

    def __init__(
        self,
        max_reprojection_error_px: float = 3.5,
        x_bounds: Tuple[float, float] = (BOUNDS_X_MIN, BOUNDS_X_MAX),
        y_bounds: Tuple[float, float] = (BOUNDS_Y_MIN, BOUNDS_Y_MAX),
        z_bounds: Tuple[float, float] = (BOUNDS_Z_MIN, BOUNDS_Z_MAX),
    ) -> None:
        self.max_reprojection_error = float(max_reprojection_error_px)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds

    def reconstruct_matches(
        self,
        matches: Sequence[StereoMatch],
        cameras: Dict[int, CameraModel],
    ) -> List[TriangulatedPoint]:
        """
        Triangulates a sequence of StereoMatch items and filters out high-error or out-of-bounds points.
        
        Args:
            matches: Verified stereo correspondences.
            cameras: Mapping from cam_id to CameraModel.
            
        Returns:
            List of valid TriangulatedPoint objects.
        """
        valid_points: List[TriangulatedPoint] = []

        for m in matches:
            if m.cam_i not in cameras or m.cam_j not in cameras:
                continue

            P_i = cameras[m.cam_i].get_projection_matrix()
            P_j = cameras[m.cam_j].get_projection_matrix()

            pt_3d, err = dlt_triangulate_point([P_i, P_j], [m.pt_i, m.pt_j])
            if pt_3d is None or err > self.max_reprojection_error:
                continue

            if not is_point_within_bounds(
                pt_3d,
                x_min=self.x_bounds[0],
                x_max=self.x_bounds[1],
                y_min=self.y_bounds[0],
                y_max=self.y_bounds[1],
                z_min=self.z_bounds[0],
                z_max=self.z_bounds[1],
            ):
                continue

            valid_points.append(
                TriangulatedPoint(
                    point_3d_ue=pt_3d.astype(np.float32),
                    reprojection_error_px=err,
                    ncc=m.ncc,
                    color_rgb=m.color_rgb,
                )
            )

        return valid_points

    def create_terrain_payload(
        self,
        triangulated_points: Sequence[TriangulatedPoint],
        runtime_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Constructs standard TerrainPayload dictionary conforming to terrain_method_contract.md.
        
        Args:
            triangulated_points: Triangulated and validated 3D points.
            runtime_ms: Total pipeline execution runtime in milliseconds.
            
        Returns:
            Dict conforming to TerrainPayload.
        """
        n_points = len(triangulated_points)

        if n_points > 0:
            vertices = np.array([p.point_3d_ue for p in triangulated_points], dtype=np.float32)
            colors = np.array([p.color_rgb for p in triangulated_points], dtype=np.uint8)
            z_min = round(float(np.min(vertices[:, 2])), 2)
            z_max = round(float(np.max(vertices[:, 2])), 2)
        else:
            vertices = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.uint8)
            z_min = 0.0
            z_max = 0.0

        payload: Dict[str, Any] = {
            "method_name": "epipolar_stereo",
            "vertices": vertices,
            "faces": None,
            "colors": colors,
            "point_cloud": vertices,
            "density": "sparse_landmarks",
            "stats": {
                "vertex_count": int(n_points),
                "face_count": 0,
                "runtime_ms": round(float(runtime_ms), 2),
                "z_min_cm": z_min,
                "z_max_cm": z_max,
            },
        }
        return payload
