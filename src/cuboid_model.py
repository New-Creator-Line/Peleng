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
    - x, y, z: coordinates of the cabin / front axle tracking pivot (SM_AA_track) on chassis/ground plane.
    - yaw: heading angle around UE5 vertical Z-axis in radians.
    - length: physical length along vehicle local forward axis (cm).
    - width: physical width along vehicle local lateral axis (cm).
    - height: physical height along vertical Z axis (cm).
    - d_front: distance from cabin pivot to front bumper along +X_local (default: 150.0 cm).
    - z_offset: vertical base offset relative to state Z (default: 0.0 cm, as state Z is aligned to chassis).
    - use_cabin_pivot: if True (default), local X spans [d_front - L, +d_front]; if False, [-L/2, +L/2].
    """
    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float
    d_front: float = 150.0
    z_offset: float = 0.0
    use_cabin_pivot: bool = True

    @classmethod
    def from_state_10d(
        cls,
        state: np.ndarray,
        d_front: float = 150.0,
        z_offset: float = 0.0,
        use_cabin_pivot: bool = True,
    ) -> "TruckCuboid3D":
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
            d_front=float(d_front),
            z_offset=float(z_offset),
            use_cabin_pivot=bool(use_cabin_pivot),
        )

    def get_local_vertices(self) -> np.ndarray:
        """
        Returns 8 local vertices relative to the cabin tracking pivot (0, 0, 0):
        When use_cabin_pivot is True:
          x_front = +d_front (~150 cm), x_rear = d_front - L
        Corners:
          0: [x_rear,  -W/2, z_bot]  (Rear-Left-Bottom)
          1: [x_front, -W/2, z_bot]  (Front-Left-Bottom)
          2: [x_front, +W/2, z_bot]  (Front-Right-Bottom)
          3: [x_rear,  +W/2, z_bot]  (Rear-Right-Bottom)
          4: [x_rear,  -W/2, z_top]  (Rear-Left-Top)
          5: [x_front, -W/2, z_top]  (Front-Left-Top)
          6: [x_front, +W/2, z_top]  (Front-Right-Top)
          7: [x_rear,  +W/2, z_top]  (Rear-Right-Top)
        """
        if self.use_cabin_pivot:
            x_front = float(self.d_front)
            x_rear = float(self.d_front - self.length)
        else:
            x_front = self.length / 2.0
            x_rear = -self.length / 2.0
        dy = self.width / 2.0
        z_bot = float(self.z_offset)
        z_top = float(self.z_offset + self.height)
        return np.array([
            [x_rear,  -dy, z_bot],
            [x_front, -dy, z_bot],
            [x_front, +dy, z_bot],
            [x_rear,  +dy, z_bot],
            [x_rear,  -dy, z_top],
            [x_front, -dy, z_top],
            [x_front, +dy, z_top],
            [x_rear,  +dy, z_top],
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
        """Returns the geometric 3D center of the vehicle body in world UE5 coordinates."""
        x_local_c = (self.d_front - self.length / 2.0) if self.use_cabin_pivot else 0.0
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return np.array([
            self.x + x_local_c * c,
            self.y + x_local_c * s,
            self.z + self.z_offset + self.height / 2.0,
        ], dtype=np.float64)

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

    # 4. Clip eigenvalues to positive threshold
    eigvals_clipped = np.maximum(eigvals, 1e-6)

    # 5. Guaranteed positive semi-definite covariance & inverse reconstruction
    cov_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    inv_cov_psd = eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T

    # 6. Semi-axes radii r_i = sqrt(chi2 * lambda_i)
    radii = np.sqrt(eigvals_clipped * chi2_val)

    volume = float((4.0 / 3.0) * math.pi * np.prod(radii))
    mu = np.zeros(3, dtype=np.float64) if center is None else np.asarray(center, dtype=np.float64)

    return UncertaintyEllipsoid3D(
        center=mu,
        radii=radii,
        axes=eigvecs,
        volume=volume,
        cov=cov_psd,
        inv_cov=inv_cov_psd,
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

    # Check visibility: reject if center or too many vertices are behind near sensor plane
    visible_mask = cam_pts[:, 2] > min_z_cam
    if not np.all(visible_mask):
        centroid = cuboid.get_centroid()
        center_cam = cam_model.R_cam @ centroid + cam_model.t_cam
        if center_cam[2] <= min_z_cam or np.count_nonzero(visible_mask) < 4:
            return None

    # Project only valid points in front of near plane to prevent 1/depth explosion
    uv_h = cam_pts @ cam_model.K.T
    pts_2d = np.full((8, 2), np.nan, dtype=np.float64)
    for i in range(8):
        if visible_mask[i]:
            z = uv_h[i, 2]
            pts_2d[i, 0] = uv_h[i, 0] / z
            pts_2d[i, 1] = uv_h[i, 1] / z

    valid_pts = pts_2d[visible_mask]
    if len(valid_pts) == 0 or np.isnan(valid_pts).any() or np.isinf(valid_pts).any():
        return None

    # Compute circumscribed 2D bounding box from visible vertices clamped to image borders
    x_min = float(np.clip(np.min(valid_pts[:, 0]), 0, img_w - 1))
    y_min = float(np.clip(np.min(valid_pts[:, 1]), 0, img_h - 1))
    x_max = float(np.clip(np.max(valid_pts[:, 0]), 0, img_w - 1))
    y_max = float(np.clip(np.max(valid_pts[:, 1]), 0, img_h - 1))

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


def draw_cuboid_wireframe_hud(
    frame: np.ndarray,
    cuboid: TruckCuboid3D,
    cam_model: CameraModel,
    edge_color: Tuple[int, int, int] = (0, 230, 255),
    front_face_color: Tuple[int, int, int] = (0, 100, 255),
    thickness: int = 2,
    draw_vertices: bool = True,
) -> bool:
    """
    Renders 3D rigid cuboid wireframe onto camera image frame.
    Highlights front cabin face for clear heading orientation.
    """
    proj = project_cuboid_to_camera_bbox(cuboid, cam_model, img_w=frame.shape[1], img_h=frame.shape[0])
    if proj is None:
        return False

    bbox_2d, pts_2d, edges_2d = proj

    # 1. Fill front cabin face with translucent overlay (bounded ROI blending)
    front_indices = cuboid.get_front_face_indices()
    if all(not np.isnan(pts_2d[i]).any() for i in front_indices):
        front_pts = np.asarray([pts_2d[i] for i in front_indices], dtype=np.int32).reshape((-1, 1, 2))
        fx1 = max(0, int(np.min(front_pts[:, 0, 0])))
        fy1 = max(0, int(np.min(front_pts[:, 0, 1])))
        fx2 = min(frame.shape[1], int(np.max(front_pts[:, 0, 0])) + 1)
        fy2 = min(frame.shape[0], int(np.max(front_pts[:, 0, 1])) + 1)
        if fx2 > fx1 and fy2 > fy1:
            roi = frame[fy1:fy2, fx1:fx2]
            roi_overlay = roi.copy()
            pts_roi = front_pts - np.array([[[fx1, fy1]]], dtype=np.int32)
            cv2.fillPoly(roi_overlay, [pts_roi], front_face_color)
            cv2.addWeighted(roi_overlay, 0.35, roi, 0.65, 0, roi)

    # 2. Draw wireframe edges
    for p1, p2 in edges_2d:
        if math.isnan(p1[0]) or math.isnan(p1[1]) or math.isnan(p2[0]) or math.isnan(p2[1]):
            continue
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        pt2 = (int(round(p2[0])), int(round(p2[1])))
        cv2.line(frame, pt1, pt2, edge_color, thickness, lineType=cv2.LINE_AA)

    # 3. Draw vertices
    if draw_vertices:
        for p in pts_2d:
            if not np.isnan(p).any():
                cv2.circle(frame, (int(round(p[0])), int(round(p[1]))), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    return True


def draw_uncertainty_ellipsoid_hud(
    frame: np.ndarray,
    ellipsoid: UncertaintyEllipsoid3D,
    cam_model: CameraModel,
    color: Tuple[int, int, int] = (180, 50, 240),
    thickness: int = 1,
    num_samples: int = 36,
) -> bool:
    """
    Projects 3D principal meridian circles of 95% confidence ellipsoid onto camera frame.
    """
    center = ellipsoid.center
    radii = ellipsoid.radii
    axes = ellipsoid.axes

    angles = np.linspace(0, 2.0 * math.pi, num_samples, endpoint=True)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    meridians = [
        np.column_stack([radii[0] * cos_a, radii[1] * sin_a, np.zeros_like(cos_a)]),
        np.column_stack([radii[0] * cos_a, np.zeros_like(cos_a), radii[2] * sin_a]),
        np.column_stack([np.zeros_like(cos_a), radii[1] * cos_a, radii[2] * sin_a]),
    ]

    for circle in meridians:
        world_pts = (circle @ axes.T) + center
        cam_pts = (world_pts @ cam_model.R_cam.T) + cam_model.t_cam.reshape(1, 3)
        if np.any(cam_pts[:, 2] <= 1.0):
            continue
        uv_h = cam_pts @ cam_model.K.T
        depths = np.maximum(uv_h[:, 2:], 1e-6)
        pts_2d = uv_h[:, :2] / depths

        pts_int = np.asarray(np.round(pts_2d), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_int], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    return True

