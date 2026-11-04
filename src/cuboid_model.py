"""
Module 4: 3D Rigid Cuboid Geometry, Uncertainty Ellipsoids & Camera Projections.
Implements the core geometric foundation for the Peleng 3D multi-camera tracking system:
- TruckCuboid3D: 8-vertex oriented 3D cuboid with dynamic dimensions (L, W, H) in UE5 coordinates.
- UncertaintyEllipsoid3D: 95% confidence 3D ellipsoid via spectral decomposition of P_XYZ.
- project_cuboid_to_camera_bbox: 3D-to-2D projection, wireframe extraction, and circumscribed 2D BBox.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from src.camera_model import CameraModel


@dataclass(frozen=True)
class TruckCuboid3D:
    """
    Oriented 3D Bounding Box for vehicle in Unreal Engine 5 coordinates.
    All positional and dimensional units are centimeters (cm).
    - x, y, z: coordinates of the center of the bottom base on ground plane.
    - yaw: heading angle around UE5 vertical Z-axis in radians.
    - length: physical length along vehicle local forward axis (cm).
    - width: physical width along vehicle local lateral axis (cm).
    - height: physical height along vertical Z axis (cm).
    """
    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float

    @classmethod
    def from_state_10d(cls, state: np.ndarray) -> "TruckCuboid3D":
        """Builds cuboid from 10D UKF state vector [X, Y, Z, v, yaw, omega, vz, L, W, H]^T."""
        st = np.asarray(state, dtype=np.float64).flatten()
        return cls(
            x=float(st[0]),
            y=float(st[1]),
            z=float(st[2]),
            yaw=float(st[4]),
            length=max(10.0, float(st[7])),
            width=max(10.0, float(st[8])),
            height=max(10.0, float(st[9])),
        )

    def get_local_vertices(self) -> np.ndarray:
        """
        Returns 8 local vertices relative to the bottom base center (0, 0, 0):
        Corners:
          0: [-L/2, -W/2, 0]  (Rear-Left-Bottom)
          1: [+L/2, -W/2, 0]  (Front-Left-Bottom)
          2: [+L/2, +W/2, 0]  (Front-Right-Bottom)
          3: [-L/2, +W/2, 0]  (Rear-Right-Bottom)
          4: [-L/2, -W/2, H]  (Rear-Left-Top)
          5: [+L/2, -W/2, H]  (Front-Left-Top)
          6: [+L/2, +W/2, H]  (Front-Right-Top)
          7: [-L/2, +W/2, H]  (Rear-Right-Top)
        """
        dx = self.length / 2.0
        dy = self.width / 2.0
        h = self.height
        return np.array([
            [-dx, -dy, 0.0],
            [+dx, -dy, 0.0],
            [+dx, +dy, 0.0],
            [-dx, +dy, 0.0],
            [-dx, -dy, h],
            [+dx, -dy, h],
            [+dx, +dy, h],
            [-dx, +dy, h],
        ], dtype=np.float64)

    def get_world_vertices(self) -> np.ndarray:
        """Transforms 8 local vertices to world UE5 coordinates rotated by yaw."""
        local_v = self.get_local_vertices()
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        rot_z = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        world_v = (local_v @ rot_z.T) + np.array([self.x, self.y, self.z], dtype=np.float64)
        return world_v

    def get_centroid(self) -> np.ndarray:
        """Returns the geometric 3D center of the vehicle body: [X, Y, Z + H/2]^T."""
        return np.array([self.x, self.y, self.z + self.height / 2.0], dtype=np.float64)

    @staticmethod
    def get_edges() -> List[Tuple[int, int]]:
        """Returns 12 vertex index pairs defining the cuboid wireframe edges."""
        return [
            # Bottom base
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top roof
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical pillars
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

    @staticmethod
    def get_front_face_indices() -> List[int]:
        """Returns indices of the 4 front vertices (cabin face): [1, 2, 6, 5]."""
        return [1, 2, 6, 5]


@dataclass(frozen=True)
class UncertaintyEllipsoid3D:
    """
    3D Uncertainty Ellipsoid computed from 3x3 position covariance matrix P_XYZ.
    - center: [X, Y, Z] centroid in world coordinates.
    - radii: sorted semi-axes lengths [r1, r2, r3] in cm corresponding to 95% confidence.
    - axes: (3, 3) rotation matrix where columns are unit eigenvectors of principal axes.
    - volume: volume of the ellipsoid in cm^3: (4/3) * pi * r1 * r2 * r3.
    - cov: (3, 3) regularized covariance matrix.
    - inv_cov: (3, 3) inverse covariance matrix for Mahalanobis distance.
    - chi2_val: critical value (default: 7.815 for 3 DOF at p=0.05).
    """
    center: np.ndarray
    radii: np.ndarray
    axes: np.ndarray
    volume: float
    cov: np.ndarray
    inv_cov: np.ndarray
    chi2_val: float = 7.815

    def mahalanobis_distance_sq(self, point: np.ndarray) -> float:
        """Calculates squared Mahalanobis distance d_M^2 = (p - mu)^T * P^-1 * (p - mu)."""
        diff = np.asarray(point, dtype=np.float64).flatten() - self.center.flatten()
        return float(diff.T @ self.inv_cov @ diff)

    def is_inside_95(self, point: np.ndarray) -> bool:
        """Returns True if point lies inside the 95% confidence ellipsoid (d_M^2 <= chi2_val)."""
        return self.mahalanobis_distance_sq(point) <= self.chi2_val


def compute_uncertainty_ellipsoid_3d(
    cov_xyz: np.ndarray,
    center: Optional[np.ndarray] = None,
    chi2_val: float = 7.815,
) -> UncertaintyEllipsoid3D:
    """
    Performs spectral decomposition of 3x3 covariance matrix P_XYZ to compute
    the 3D confidence ellipsoid semi-axes, orientation, and Mahalanobis metric.
    Includes robust symmetrization and eigenvalue clipping to ensure numerical stability.
    """
    cov = np.asarray(cov_xyz, dtype=np.float64)
    if cov.shape != (3, 3):
        raise ValueError(f"Expected 3x3 covariance matrix, got shape {cov.shape}")

    # 1. Enforce strict symmetry
    cov_sym = 0.5 * (cov + cov.T)

    # 2. Add diagonal regularization if necessary
    cov_reg = cov_sym + np.eye(3) * 1e-6

    # 3. Eigendecomposition (eigh guaranteed real for symmetric matrix)
    eigvals, eigvecs = np.linalg.eigh(cov_reg)

    # 4. Clip negative or near-zero eigenvalues to positive threshold
    eigvals_clipped = np.maximum(eigvals, 1e-6)

    # 5. Semi-axes radii r_i = sqrt(chi2 * lambda_i)
    radii = np.sqrt(eigvals_clipped * chi2_val)

    # 6. Inverse covariance matrix for Mahalanobis distance
    try:
        inv_cov = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        inv_cov = np.diag(1.0 / eigvals_clipped)

    volume = float((4.0 / 3.0) * math.pi * np.prod(radii))
    mu = np.zeros(3, dtype=np.float64) if center is None else np.asarray(center, dtype=np.float64)

    return UncertaintyEllipsoid3D(
        center=mu,
        radii=radii,
        axes=eigvecs,
        volume=volume,
        cov=cov_reg,
        inv_cov=inv_cov,
        chi2_val=chi2_val,
    )


def project_cuboid_to_camera_bbox(
    cuboid: TruckCuboid3D,
    cam_model: CameraModel,
    img_w: int = 1920,
    img_h: int = 1080,
    min_z_cam: float = 1.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[Tuple[Tuple[float, float], Tuple[float, float]]]]]:
    """
    Projects 3D cuboid onto camera sensor.
    Returns:
      - bbox_2d: np.ndarray [x_min, y_min, x_max, y_max] strictly circumscribing visible vertices.
      - pts_2d: np.ndarray of shape (8, 2) containing projected pixel coordinates.
      - edges_2d: List of 12 line segments [((u1, v1), (u2, v2)), ...] for wireframe HUD rendering.
    Returns None if all or critical cuboid vertices lie behind the camera (z_cam <= min_z_cam).
    """
    world_verts = cuboid.get_world_vertices()  # (8, 3)

    # Transform to camera frame: X_cam = R_cam * X_world + t_cam
    cam_pts = (world_verts @ cam_model.R_cam.T) + cam_model.t_cam.reshape(1, 3)

    # Reject if cuboid is behind the camera sensor plane
    if np.any(cam_pts[:, 2] <= min_z_cam):
        # Check if entire cuboid is behind camera
        if np.all(cam_pts[:, 2] <= min_z_cam):
            return None
        # If partially clipped, ensure center is ahead
        centroid = cuboid.get_centroid()
        center_cam = cam_model.R_cam @ centroid + cam_model.t_cam
        if center_cam[2] <= min_z_cam:
            return None

    # Project 3D points via camera intrinsics
    # Homogeneous coordinates: uv_h = K * pt_cam
    uv_h = cam_pts @ cam_model.K.T
    depths = np.maximum(uv_h[:, 2:], 1e-6)
    pts_2d = uv_h[:, :2] / depths  # (8, 2)

    # Check for NaN / Inf
    if np.isnan(pts_2d).any() or np.isinf(pts_2d).any():
        return None

    # Compute circumscribed 2D bounding box clamped to image borders
    x_min = float(np.clip(np.min(pts_2d[:, 0]), 0, img_w - 1))
    y_min = float(np.clip(np.min(pts_2d[:, 1]), 0, img_h - 1))
    x_max = float(np.clip(np.max(pts_2d[:, 0]), 0, img_w - 1))
    y_max = float(np.clip(np.max(pts_2d[:, 1]), 0, img_h - 1))

    # Reject degenerate boxes
    if (x_max - x_min) < 1.0 or (y_max - y_min) < 1.0:
        return None

    bbox_2d = np.array([x_min, y_min, x_max, y_max], dtype=np.float64)

    # Build 2D wireframe edges
    edges_indices = cuboid.get_edges()
    edges_2d = []
    for i1, i2 in edges_indices:
        p1 = (float(pts_2d[i1, 0]), float(pts_2d[i1, 1]))
        p2 = (float(pts_2d[i2, 0]), float(pts_2d[i2, 1]))
        edges_2d.append((p1, p2))

    return bbox_2d, pts_2d, edges_2d
