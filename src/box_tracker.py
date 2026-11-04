"""
Truck tracker based on multi-view oriented-box fitting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.box_fitter import BoxObservation, MultiViewBoxFitter, box_iou
from src.camera_model import CameraTimeline
from src.tracker_3d import ConstantVelocityKF
from src.triangulator import MultiViewTriangulator
from src.truck_model import TruckModel


@dataclass
class BoxTrackerConfig:
    dt: float = 1.0 / 30.0
    conf_threshold: float = 0.3
    image_w: int = 1920
    image_h: int = 1080
    accel_std_xy: float = 600.0
    accel_std_z: float = 150.0
    yaw_accel_std: float = 1.5
    min_pos_meas_std: float = 15.0
    min_yaw_meas_std: float = 0.03
    prior_inflation: float = 3.0
    max_rms_px: float = 45.0
    gate_chi2: float = 40.0
    assoc_min_iou: float = 0.05
    init_max_rms_px: float = 30.0
    min_speed_for_heading: float = 150.0


@dataclass
class TrackStats:
    measured_multi: int = 0
    measured_single: int = 0
    predicted_only: int = 0
    rejected: int = 0
    first_frame: Optional[int] = None


def _det_box(o: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            o["u"] - o["w"] / 2.0,
            o["v"] - o["h"] / 2.0,
            o["u"] + o["w"] / 2.0,
            o["v"] + o["h"] / 2.0,
        ],
        dtype=np.float64,
    )


class BoxTruckTracker:
    def __init__(
        self,
        cameras: Dict[int, CameraTimeline],
        truck: TruckModel,
        config: Optional[BoxTrackerConfig] = None,
    ) -> None:
        self.cameras = cameras
        self.truck = truck
        self.cfg = config or BoxTrackerConfig()
        self.fitter = MultiViewBoxFitter(truck)
        self.triangulator = MultiViewTriangulator(reprojection_error_threshold_px=80.0)
        c = self.cfg
        self.kf_pos = ConstantVelocityKF(3, c.dt, [c.accel_std_xy, c.accel_std_xy, c.accel_std_z])
        self.kf_yaw = ConstantVelocityKF(1, c.dt, c.yaw_accel_std)
        self.stats = TrackStats()
        self._heading_sign = 1.0
        self._frames: List[int] = []

    def _make_obs(self, frame: int, o: Dict[str, float]) -> BoxObservation:
        return BoxObservation(
            cam_id=int(o["cam_id"]),
            model=self.cameras[int(o["cam_id"])].get_model(frame),
            box=_det_box(o),
            conf=float(o["conf"]),
            image_w=self.cfg.image_w,
            image_h=self.cfg.image_h,
        )

    def _candidates(self, frame: int, dets: List[Dict[str, float]]) -> Dict[int, List[Dict[str, float]]]:
        by_cam: Dict[int, List[Dict[str, float]]] = {}
        for o in sorted(dets, key=lambda d: d["conf"], reverse=True):
            if o["conf"] < self.cfg.conf_threshold or int(o["cam_id"]) not in self.cameras:
                continue
            by_cam.setdefault(int(o["cam_id"]), []).append(o)
        return by_cam

    def _associate(self, frame: int, by_cam, center: np.ndarray, yaw: float) -> List[BoxObservation]:
        models = {cid: self.cameras[cid].get_model(frame) for cid in by_cam}
        pred = self.fitter.predicted_extents(models, center, yaw)
        obs: List[BoxObservation] = []
        for cid, dets in by_cam.items():
            ext = pred.get(cid)
            if ext is None:
                continue
            best, best_iou = None, self.cfg.assoc_min_iou
            for o in dets:
                iou = box_iou(ext, _det_box(o))
                if iou > best_iou:
                    best, best_iou = o, iou
            if best is not None:
                obs.append(self._make_obs(frame, best))
        return obs

    def _try_initialize(self, frame: int, by_cam) -> Optional[Tuple[np.ndarray, float, object]]:
        cam_ids = list(by_cam.keys())
        if len(cam_ids) < 2:
            return None
        best = None
        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                o1, o2 = by_cam[cam_ids[i]][0], by_cam[cam_ids[j]][0]
                P1 = self.cameras[cam_ids[i]].get_model(frame).get_projection_matrix()
                P2 = self.cameras[cam_ids[j]].get_model(frame).get_projection_matrix()
                pt, _ = self.triangulator.triangulate([P1, P2], [(o1["u"], o1["v"]), (o2["u"], o2["v"])])
                if pt is None:
                    continue
                obs = [self._make_obs(frame, by_cam[c][0]) for c in cam_ids]
                res = self.fitter.fit(obs, pt, 0.0, yaw_candidates=np.linspace(0, np.pi, 6, endpoint=False))
                if res is None:
                    continue
                consistent = self._associate(frame, by_cam, res.center, res.yaw)
                if len(consistent) >= 2:
                    res2 = self.fitter.fit(consistent, res.center, res.yaw)
                    if res2 is not None:
                        res = res2
                if res.rms_px < self.cfg.init_max_rms_px and (best is None or res.rms_px < best.rms_px):
                    best = res
        if best is None:
            return None
        return best.center, best.yaw, best

    def run(self, grouped_detections: Dict[int, List[Dict[str, float]]], n_frames: int) -> pd.DataFrame:
        c = self.cfg
        rows = []
        for frame in range(n_frames):
            by_cam = self._candidates(frame, grouped_detections.get(frame, []))

            if not self.kf_pos.is_initialized:
                init = self._try_initialize(frame, by_cam)
                if init is None:
                    rows.append({"Frame": frame, "status": "none", "n_cams": 0, "rms_px": np.nan})
                    continue
                center, yaw, res = init
                self.kf_pos.initialize(
                    center,
                    res.cov_pos + np.eye(3) * c.min_pos_meas_std ** 2,
                    vel=np.zeros(3),
                    vel_std=[1500.0, 1500.0, 200.0],
                )
                self.kf_yaw.initialize(
                    [yaw],
                    [[res.yaw_std ** 2 + c.min_yaw_meas_std ** 2]],
                    vel=[0.0],
                    vel_std=0.5,
                )
                self.stats.first_frame = frame
                self.stats.measured_multi += 1
                self._frames.append(frame)
                rows.append({"Frame": frame, "status": "init", "n_cams": res.n_cams, "rms_px": res.rms_px})
                continue

            c_pred = self.kf_pos.begin_step()
            y_pred = float(self.kf_yaw.begin_step()[0])
            P_pos = self.kf_pos.P[:3, :3]
            P_yaw = float(self.kf_yaw.P[0, 0])

            obs = self._associate(frame, by_cam, c_pred, y_pred)
            status, n_cams, rms = "predicted", 0, np.nan
            z_pos = R_pos = z_yaw = R_yaw = None
            if obs:
                prior_std = np.concatenate(
                    [
                        np.maximum(np.sqrt(np.diag(P_pos)) * c.prior_inflation, 30.0),
                        [max(np.sqrt(P_yaw) * c.prior_inflation, 0.05)],
                    ]
                )
                prior = (np.array([*c_pred, y_pred]), prior_std)
                res = self.fitter.fit(obs, c_pred, y_pred, prior=prior)
                if res is not None and res.rms_px <= c.max_rms_px:
                    R = res.cov_pos + np.eye(3) * c.min_pos_meas_std ** 2
                    if self.kf_pos.innovation_mahalanobis(res.center, R) <= c.gate_chi2:
                        z_pos, R_pos = res.center, R
                        dy = np.angle(np.exp(1j * 2.0 * (res.yaw - y_pred))) / 2.0
                        z_yaw = np.array([y_pred + dy])
                        R_yaw = np.array([[res.yaw_std ** 2 + c.min_yaw_meas_std ** 2]])
                        status = "multi" if res.n_cams >= 2 else "single"
                        n_cams, rms = res.n_cams, res.rms_px
                    else:
                        self.stats.rejected += 1
                else:
                    self.stats.rejected += 1

            self.kf_pos.end_step(z_pos, R_pos)
            self.kf_yaw.end_step(z_yaw, R_yaw)
            if status == "multi":
                self.stats.measured_multi += 1
            elif status == "single":
                self.stats.measured_single += 1
            else:
                self.stats.predicted_only += 1
            self._frames.append(frame)
            rows.append({"Frame": frame, "status": status, "n_cams": n_cams, "rms_px": rms})

        diag = pd.DataFrame(rows)
        return self._finalize(diag, n_frames)

    def _pivots(self, pos_states: np.ndarray, yaw_states: np.ndarray) -> np.ndarray:
        out = np.zeros((len(pos_states), 3))
        sign = 1.0
        for k in range(len(pos_states)):
            v = pos_states[k, 3:5]
            if np.linalg.norm(v) > self.cfg.min_speed_for_heading:
                h = np.array([np.cos(yaw_states[k, 0]), np.sin(yaw_states[k, 0])])
                sign = 1.0 if h @ v >= 0 else -1.0
                break
        for k in range(len(pos_states)):
            yaw = yaw_states[k, 0]
            v = pos_states[k, 3:5]
            h = np.array([np.cos(yaw), np.sin(yaw)])
            if np.linalg.norm(v) > self.cfg.min_speed_for_heading:
                sign = 1.0 if h @ v >= 0 else -1.0
            heading = yaw if sign > 0 else yaw + np.pi
            out[k] = self.truck.pivot_from_center(pos_states[k, :3], heading)
        return out

    def _finalize(self, diag: pd.DataFrame, n_frames: int) -> pd.DataFrame:
        frames = np.array(self._frames, dtype=int)
        out = pd.DataFrame({"Frame": np.arange(n_frames)})
        cols = ["X_online", "Y_online", "Z_online", "X_offline", "Y_offline", "Z_offline"]
        if len(frames) == 0:
            for col in cols:
                out[col] = np.nan
            return out.merge(diag, on="Frame", how="left")

        online = self._pivots(self.kf_pos.filtered_states(), self.kf_yaw.filtered_states())
        offline = self._pivots(self.kf_pos.smooth(), self.kf_yaw.smooth())
        tmp = pd.DataFrame(np.hstack([online, offline]), columns=cols)
        tmp["Frame"] = frames
        out = out.merge(tmp, on="Frame", how="left")
        out[cols] = out[cols].bfill().ffill()
        return out.merge(diag, on="Frame", how="left")
