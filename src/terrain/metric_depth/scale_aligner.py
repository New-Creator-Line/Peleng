"""
Metric Scale & Shift Alignment Module.

Solves and applies affine calibration (s, t) to map relative/unscaled depth maps
to true Unreal Engine centimeters (UE5_cm):
    D_metric(u, v) = s * D_raw(u, v) + t

Uses:
- Inter-camera baseline distances (e.g. Cam 3 to Cam 5 = 18180.0 cm).
- Ground truth object/truck trajectory anchors when available.
- Valley floor geometric plane references.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.camera_model import CameraModel


@dataclass
class AnchorPoint:
    u: float  # pixel x (in full or scaled resolution)
    v: float  # pixel y (in full or scaled resolution)
    target_metric_depth: float  # ground-truth z_cam in centimeters
    source: str  # 'inter_camera' | 'ground_truth_track' | 'ground_plane'


@dataclass
class ScaleCalibrationResult:
    cam_id: int
    scale_s: float
    shift_t: float
    num_anchors: int
    residual_rmse: float
    aligned_depth: np.ndarray


def solve_affine_scale_shift(
    predicted_depths: np.ndarray,
    target_depths: np.ndarray,
    allow_shift: bool = True,
    min_scale: float = 1e-4,
) -> Tuple[float, float, float]:
    """
    Solves min_{s, t} sum (s * d_i + t - D*_i)^2 subject to s > 0.
    Returns (scale_s, shift_t, rmse).
    """
    p = np.asarray(predicted_depths, dtype=np.float64).ravel()
    tgt = np.asarray(target_depths, dtype=np.float64).ravel()

    valid = (p > 1e-4) & (tgt > 1e-4) & np.isfinite(p) & np.isfinite(tgt)
    p = p[valid]
    tgt = tgt[valid]

    if len(p) == 0:
        return 1.0, 0.0, 0.0

    if len(p) == 1 or not allow_shift:
        # 1-parameter scale: s = (p . tgt) / (p . p)
        denom = float(np.dot(p, p))
        s = float(np.dot(p, tgt) / max(denom, 1e-8)) if denom > 1e-8 else 1.0
        s = max(s, min_scale)
        residuals = s * p - tgt
        rmse = float(np.sqrt(np.mean(residuals**2)))
        return s, 0.0, rmse

    # 2-parameter least squares
    a_mat = np.column_stack([p, np.ones_like(p)])
    try:
        sol, residuals, rank, _ = np.linalg.lstsq(a_mat, tgt, rcond=None)
        s, t = float(sol[0]), float(sol[1])
        if s <= min_scale or np.isnan(s) or np.isnan(t):
            # Degenerate slope, fallback to positive ratio
            s = float(np.median(tgt / np.maximum(p, 1e-4)))
            s = max(s, min_scale)
            t = 0.0
    except Exception:
        s = float(np.median(tgt / np.maximum(p, 1e-4)))
        s = max(s, min_scale)
        t = 0.0

    err = s * p + t - tgt
    rmse = float(np.sqrt(np.mean(err**2)))
    return s, t, rmse


class ScaleAligner:
    """
    Extracts geometric & baseline anchors across the camera rig and computes
    optimal affine scale/shift parameters to align depth maps to centimeters.
    """

    def __init__(self, calib_dir: Optional[str] = None) -> None:
        self.calib_dir = calib_dir
        self._track_df: Optional[pd.DataFrame] = None
        self._load_ground_truth_track()

    def _load_ground_truth_track(self) -> None:
        """Loads object_transform_SM_AA_track.csv if available."""
        if not self.calib_dir:
            return
        track_path = os.path.join(self.calib_dir, "object_transform_SM_AA_track.csv")
        if os.path.isfile(track_path):
            try:
                df = pd.read_csv(track_path, sep=";", encoding="cp1251")
                col_map = {c: str(c).strip() for c in df.columns}
                df = df.rename(columns=col_map)
                self._track_df = df
            except Exception:
                pass

    def get_inter_camera_anchors(
        self,
        cam_id: int,
        camera_model: CameraModel,
        all_cameras: Dict[int, CameraModel],
        depth_shape: Tuple[int, int] = (180, 320),
    ) -> List[AnchorPoint]:
        """
        Discovers mutual visibility anchors where camera cam_id observes another camera.
        For example: Cam 3 sees Cam 5 at optical depth ~18180.0 cm.
        """
        h_d, w_d = depth_shape
        scale_u = w_d / float(camera_model.image_width_px)
        scale_v = h_d / float(camera_model.image_height_px)

        anchors: List[AnchorPoint] = []
        for other_id, other_model in all_cameras.items():
            if other_id == cam_id:
                continue

            u, v, z_cam = camera_model.project_point(other_model.loc)
            if z_cam > 100.0:
                # Check within image bounds
                if 0 <= u < camera_model.image_width_px and 0 <= v < camera_model.image_height_px:
                    u_scaled = u * scale_u
                    v_scaled = v * scale_v
                    anchors.append(
                        AnchorPoint(
                            u=u_scaled,
                            v=v_scaled,
                            target_metric_depth=z_cam,
                            source=f"inter_cam_to_{other_id}",
                        )
                    )
        return anchors

    def get_trajectory_anchor(
        self,
        camera_model: CameraModel,
        frame_idx: int = 0,
        depth_shape: Tuple[int, int] = (180, 320),
    ) -> Optional[AnchorPoint]:
        """
        Extracts ground truth truck anchor at frame_idx if visible.
        """
        if self._track_df is None:
            return None

        sub = self._track_df[self._track_df["Frame"] == frame_idx]
        if sub.empty:
            return None

        try:
            row = sub.iloc[0]
            loc_truck = np.array([float(row["Loc_X"]), float(row["Loc_Y"]), float(row["Loc_Z"])])
            u, v, z_cam = camera_model.project_point(loc_truck)
            if z_cam > 100.0:
                if 0 <= u < camera_model.image_width_px and 0 <= v < camera_model.image_height_px:
                    h_d, w_d = depth_shape
                    u_s = u * (w_d / float(camera_model.image_width_px))
                    v_s = v * (h_d / float(camera_model.image_height_px))
                    return AnchorPoint(
                        u=u_s,
                        v=v_s,
                        target_metric_depth=z_cam,
                        source="ground_truth_track",
                    )
        except Exception:
            pass
        return None

    def get_ground_plane_anchors(
        self,
        camera_model: CameraModel,
        depth_shape: Tuple[int, int] = (180, 320),
        nominal_ground_z: float = 400.0,
        num_samples: int = 9,
    ) -> List[AnchorPoint]:
        """
        Generates reference ground plane anchors across the visible lower half of the frame.
        """
        h_d, w_d = depth_shape
        fx, fy, cx, cy = camera_model.get_intrinsics()
        scale_x = w_d / float(camera_model.image_width_px)
        scale_y = h_d / float(camera_model.image_height_px)
        fx_s, fy_s = fx * scale_x, fy * scale_y
        cx_s, cy_s = cx * scale_x, cy * scale_y

        anchors: List[AnchorPoint] = []
        u_samples = np.linspace(w_d * 0.2, w_d * 0.8, int(math.sqrt(num_samples)))
        v_samples = np.linspace(h_d * 0.5, h_d * 0.9, int(math.sqrt(num_samples)))

        for v in v_samples:
            for u in u_samples:
                ray_cam = np.array([(u - cx_s) / fx_s, (v - cy_s) / fy_s, 1.0], dtype=np.float64)
                ray_world = camera_model.R_cam.T @ ray_cam
                if ray_world[2] < -0.01:
                    lam = (nominal_ground_z - camera_model.loc[2]) / ray_world[2]
                    if 100.0 < lam < 250000.0:
                        anchors.append(
                            AnchorPoint(
                                u=float(u),
                                v=float(v),
                                target_metric_depth=lam,
                                source="ground_plane",
                            )
                        )
        return anchors

    def calibrate_depth_map(
        self,
        cam_id: int,
        raw_depth: np.ndarray,
        camera_model: CameraModel,
        all_cameras: Optional[Dict[int, CameraModel]] = None,
        frame_idx: int = 0,
        additional_anchors: Optional[List[AnchorPoint]] = None,
    ) -> ScaleCalibrationResult:
        """
        Fits (scale_s, shift_t) to align raw_depth to world centimeters (UE5_cm).
        """
        h, w = raw_depth.shape
        anchors: List[AnchorPoint] = []

        # 1. Mutual camera visibility anchors
        if all_cameras:
            anchors.extend(
                self.get_inter_camera_anchors(cam_id, camera_model, all_cameras, depth_shape=(h, w))
            )

        # 2. Trajectory anchor
        truck_anchor = self.get_trajectory_anchor(camera_model, frame_idx, depth_shape=(h, w))
        if truck_anchor is not None:
            anchors.append(truck_anchor)

        # 3. Additional user/test anchors
        if additional_anchors:
            anchors.extend(additional_anchors)

        # 4. Ground plane anchors (supplementary anchors to guarantee well-conditioned fit)
        if len(anchors) < 2:
            ground_anchors = self.get_ground_plane_anchors(
                camera_model, depth_shape=(h, w), num_samples=9
            )
            anchors.extend(ground_anchors)

        # Sample raw depth at anchor coordinates
        pred_vals: List[float] = []
        target_vals: List[float] = []

        for anc in anchors:
            u_idx = int(round(anc.u))
            v_idx = int(round(anc.v))
            if 0 <= u_idx < w and 0 <= v_idx < h:
                # 3x3 local patch median to avoid pixel noise
                u_min, u_max = max(0, u_idx - 1), min(w, u_idx + 2)
                v_min, v_max = max(0, v_idx - 1), min(h, v_idx + 2)
                patch = raw_depth[v_min:v_max, u_min:u_max]
                val = float(np.median(patch))
                if val > 1e-3:
                    pred_vals.append(val)
                    target_vals.append(anc.target_metric_depth)

        if len(pred_vals) >= 2:
            s, t, rmse = solve_affine_scale_shift(
                np.array(pred_vals), np.array(target_vals), allow_shift=True
            )
        elif len(pred_vals) == 1:
            s, t, rmse = solve_affine_scale_shift(
                np.array(pred_vals), np.array(target_vals), allow_shift=False
            )
        else:
            # Fallback based on scale inspection
            med = float(np.median(raw_depth[raw_depth > 1.0])) if np.any(raw_depth > 1.0) else 1.0
            if med < 200.0:  # in meters
                s, t, rmse = 100.0, 0.0, 0.0
            else:
                s, t, rmse = 1.0, 0.0, 0.0

        aligned = np.clip(s * raw_depth + t, 0.0, 300000.0)
        return ScaleCalibrationResult(
            cam_id=cam_id,
            scale_s=s,
            shift_t=t,
            num_anchors=len(pred_vals),
            residual_rmse=rmse,
            aligned_depth=aligned,
        )

    def calibrate_all(
        self,
        depth_dict: Dict[int, np.ndarray],
        cameras: Dict[int, CameraModel],
        frame_idx: int = 0,
    ) -> Dict[int, ScaleCalibrationResult]:
        """
        Calibrates all cameras simultaneously.
        """
        results: Dict[int, ScaleCalibrationResult] = {}
        for cam_id, depth_map in depth_dict.items():
            model = cameras[cam_id]
            res = self.calibrate_depth_map(
                cam_id=cam_id,
                raw_depth=depth_map,
                camera_model=model,
                all_cameras=cameras,
                frame_idx=frame_idx,
            )
            results[cam_id] = res
        return results
