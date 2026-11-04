"""
Multi-view oriented-box fitting.

For every frame we search the box pose (center X, Y, Z and yaw) of a truck with known
dimensions such that the 2D extents of its projected 3D box match the detected bounding
boxes in all cameras simultaneously (robust non-linear least squares).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from src.camera_model import CameraModel
from src.truck_model import TruckModel


@dataclass
class BoxObservation:
    cam_id: int
    model: CameraModel
    box: np.ndarray          # (x1, y1, x2, y2) pixels
    conf: float
    image_w: int = 1920
    image_h: int = 1080
    border_px: float = 4.0

    def edge_mask(self) -> np.ndarray:
        """Edges touching the image border are truncated -> unreliable."""
        x1, y1, x2, y2 = self.box
        return np.array(
            [
                x1 > self.border_px,
                y1 > self.border_px,
                x2 < self.image_w - self.border_px,
                y2 < self.image_h - self.border_px,
            ]
        )


@dataclass
class BoxFitResult:
    center: np.ndarray       # box center (3,)
    yaw: float               # box yaw (rad), defined modulo pi
    cov_pos: np.ndarray      # (3, 3) position covariance
    yaw_std: float
    rms_px: float            # RMS edge residual (pixels) of inlier edges
    n_cams: int
    n_edges: int


def project_box_extents(model: CameraModel, corners: np.ndarray) -> Optional[np.ndarray]:
    """Projects 3D corners and returns the 2D extent (x1, y1, x2, y2) or None if behind camera."""
    pc = corners @ model.R_cam.T + model.t_cam
    z = pc[:, 2]
    if np.any(z <= 1.0):
        return None
    uv = (pc @ model.K.T)[:, :2] / z[:, None]
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


class MultiViewBoxFitter:
    def __init__(
        self,
        truck: TruckModel,
        pixel_sigma: float = 6.0,
        relative_sigma: float = 0.03,
        robust_scale: float = 2.0,
    ) -> None:
        self.truck = truck
        self.pixel_sigma = float(pixel_sigma)
        self.relative_sigma = float(relative_sigma)
        self.robust_scale = float(robust_scale)

    def _edge_sigma(self, obs: BoxObservation) -> float:
        size = max(obs.box[2] - obs.box[0], obs.box[3] - obs.box[1])
        return float(np.hypot(self.pixel_sigma, self.relative_sigma * size))

    def _residuals(
        self,
        theta: np.ndarray,
        observations: Sequence[BoxObservation],
        prior: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        center, yaw = theta[:3], theta[3]
        corners = self.truck.box_corners(center, yaw)
        res: List[float] = []
        for obs in observations:
            mask = obs.edge_mask()
            ext = project_box_extents(obs.model, corners)
            sigma = self._edge_sigma(obs)
            if ext is None:
                res.extend([50.0] * int(mask.sum()))
                continue
            d = (ext - obs.box) / sigma
            res.extend(d[mask].tolist())
        if prior is not None:
            mean, std = prior
            dp = (theta - mean) / std
            dp[3] = (np.angle(np.exp(1j * (theta[3] - mean[3])))) / std[3]
            res.extend(dp.tolist())
        return np.asarray(res, dtype=np.float64)

    def fit(
        self,
        observations: Sequence[BoxObservation],
        init_center: np.ndarray,
        init_yaw: float,
        prior: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        yaw_candidates: Optional[Sequence[float]] = None,
    ) -> Optional[BoxFitResult]:
        if not observations:
            return None
        n_edges = int(sum(o.edge_mask().sum() for o in observations))
        n_prior = 4 if prior is not None else 0
        if n_edges + n_prior < 4:
            return None

        yaws = list(yaw_candidates) if yaw_candidates is not None else [init_yaw]
        best = None
        for y0 in yaws:
            x0 = np.array([*np.asarray(init_center, dtype=np.float64), y0])
            try:
                sol = least_squares(
                    self._residuals,
                    x0,
                    args=(observations, prior),
                    loss="soft_l1",
                    f_scale=self.robust_scale,
                    x_scale=np.array([200.0, 200.0, 50.0, 0.2]),
                    max_nfev=60,
                )
            except ValueError:
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        if best is None:
            return None

        theta = best.x
        J = best.jac
        try:
            cov = np.linalg.inv(J.T @ J + 1e-9 * np.eye(4))
        except np.linalg.LinAlgError:
            return None
        r_edges = self._residuals(theta, observations, None)
        inl = np.abs(r_edges) < 3.0
        sig = np.mean([self._edge_sigma(o) for o in observations])
        rms_px = float(np.sqrt(np.mean((r_edges[inl] * sig) ** 2))) if inl.any() else float("inf")
        chi2 = float(np.mean(np.minimum(r_edges ** 2, 9.0))) if len(r_edges) else 1.0
        cov = cov * max(1.0, chi2)

        yaw = float(np.angle(np.exp(1j * theta[3])))
        return BoxFitResult(
            center=theta[:3].copy(),
            yaw=yaw,
            cov_pos=cov[:3, :3],
            yaw_std=float(np.sqrt(max(cov[3, 3], 1e-8))),
            rms_px=rms_px,
            n_cams=len(observations),
            n_edges=n_edges,
        )

    def predicted_extents(
        self, models: Dict[int, CameraModel], center: np.ndarray, yaw: float
    ) -> Dict[int, Optional[np.ndarray]]:
        corners = self.truck.box_corners(center, yaw)
        return {cid: project_box_extents(m, corners) for cid, m in models.items()}
