from dataclasses import dataclass
import glob
import math
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CameraParams:
    cam_id: int
    loc: np.ndarray  # (3,) Loc_X, Loc_Y, Loc_Z in UE world
    rot: np.ndarray  # (3,) Roll, Pitch, Yaw in degrees
    focal_length_mm: float = 35.0
    sensor_width_mm: float = 23.76
    sensor_height_mm: float = 13.365
    image_width_px: int = 1920
    image_height_px: int = 1080


def get_ue_camera_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Computes camera rotation matrix from Unreal Engine FRotator (Roll, Pitch, Yaw).
    Official Unreal Engine FRotationMatrix definition:
      Forward = [ cp*cy, cp*sy, sp ]
      Right   = [ sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, -sr*cp ]
      Up      = [ -(cr*sp*cy + sr*sy), -(cr*sp*sy - sr*cy), cr*cp ]

    In OpenCV camera coordinates:
      X_cv = Right
      Y_cv = Down = -Up
      Z_cv = Forward (optical axis)
    """
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    sr, cr = math.sin(r), math.cos(r)

    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float64)
    right   = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp], dtype=np.float64)
    up      = np.array([-(cr * sp * cy + sr * sy), -(cr * sp * sy - sr * cy), cr * cp], dtype=np.float64)

    R_cam = np.vstack([right, -up, forward])
    return R_cam


class CameraModel:
    def __init__(
        self,
        cam_id: int,
        loc: np.ndarray,
        rot: np.ndarray,
        focal_length_mm: float = 35.0,
        sensor_width_mm: float = 23.76,
        sensor_height_mm: float = 13.365,
        image_width_px: int = 1920,
        image_height_px: int = 1080,
    ) -> None:
        self.cam_id = cam_id
        self.loc = np.asarray(loc, dtype=np.float64)
        self.rot = np.asarray(rot, dtype=np.float64)
        self.focal_length_mm = float(focal_length_mm)
        self.sensor_width_mm = float(sensor_width_mm)
        self.sensor_height_mm = float(sensor_height_mm)
        self.image_width_px = int(image_width_px)
        self.image_height_px = int(image_height_px)

        self._compute_matrices()

    def _compute_matrices(self) -> None:
        fx = self.focal_length_mm * self.image_width_px / self.sensor_width_mm
        fy = self.focal_length_mm * self.image_height_px / self.sensor_height_mm
        cx = self.image_width_px / 2.0
        cy = self.image_height_px / 2.0

        self.K = np.array([
            [fx,  0.0, cx ],
            [0.0, fy,  cy ],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.R_cam = get_ue_camera_rotation_matrix(self.rot[0], self.rot[1], self.rot[2])
        self.t_cam = -self.R_cam @ self.loc

        Rt = np.hstack([self.R_cam, self.t_cam.reshape(3, 1)])
        self.P = self.K @ Rt

    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        return self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

    def get_projection_matrix(self) -> np.ndarray:
        return self.P

    def project_point(self, pt_3d_ue: np.ndarray) -> Tuple[float, float, float]:
        """Projects a 3D point from UE coordinates to (u, v, z_cam)."""
        pt = np.asarray(pt_3d_ue, dtype=np.float64)
        pt_cam = self.R_cam @ pt + self.t_cam
        z_cam = float(pt_cam[2])

        if z_cam <= 1e-6:
            return -1.0, -1.0, z_cam

        uv_h = self.K @ pt_cam
        u = float(uv_h[0] / uv_h[2])
        v = float(uv_h[1] / uv_h[2])
        return u, v, z_cam

    def project_ray_to_z(self, u: float, v: float, target_z: float = 450.0) -> Optional[np.ndarray]:
        """Intersects a 2D pixel ray (u, v) with a horizontal world plane Z = target_z."""
        fx, fy, cx, cy = self.get_intrinsics()
        ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        ray_world = self.R_cam.T @ ray_cam
        if abs(ray_world[2]) < 1e-6:
            return None
        lam = (target_z - self.loc[2]) / ray_world[2]
        if lam <= 0:
            return None
        return self.loc + lam * ray_world


class CameraTimeline:
    """Manages camera calibration parameters over all video frames."""
    def __init__(self, cam_id: int) -> None:
        self.cam_id = cam_id
        self.frames: Dict[int, CameraModel] = {}
        self.default_model: Optional[CameraModel] = None

    def add_frame_model(self, frame_idx: int, model: CameraModel) -> None:
        self.frames[frame_idx] = model
        if self.default_model is None:
            self.default_model = model

    def get_model(self, frame_idx: int) -> CameraModel:
        if frame_idx in self.frames:
            return self.frames[frame_idx]
        if self.default_model is not None:
            return self.default_model
        raise ValueError(f"No camera model available for cam {self.cam_id}")


def _parse_float(val: object) -> float:
    return float(str(val).strip().replace(',', '.'))


def _extract_optics_from_csv(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Extracts focal_length_mm and sensor dimensions (sw, sh) from the camera CSV header/first row.
    Header format: Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;60
    First row format: ...;;;Размеры матрицы;23,76*13,365
    """
    focal_length = 35.0
    sw, sh = 23.76, 13.365

    cols = [str(c).strip() for c in df.columns]
    for idx, col_name in enumerate(cols):
        if col_name.lower().startswith("focal length") and idx + 1 < len(cols):
            try:
                focal_length = _parse_float(cols[idx + 1])
            except ValueError:
                pass
            # Check first row of that same column for sensor dimensions "23,76*13,365"
            if not df.empty:
                raw_sensor = str(df.iloc[0, idx + 1]).strip()
                sensor_match = re.search(r"(\d+[.,]?\d*)\s*\*\s*(\d+[.,]?\d*)", raw_sensor)
                if sensor_match:
                    try:
                        sw = _parse_float(sensor_match.group(1))
                        sh = _parse_float(sensor_match.group(2))
                    except ValueError:
                        pass
            break

    return focal_length, sw, sh


def load_camera_csv(csv_path: str, cam_id: int) -> CameraTimeline:
    timeline = CameraTimeline(cam_id)
    df = pd.read_csv(csv_path, sep=';', encoding='cp1251')

    focal_length, sw, sh = _extract_optics_from_csv(df)

    col_map = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=col_map)

    for _, row in df.iterrows():
        try:
            frame_idx = int(_parse_float(row['Frame']))
            loc = np.array([
                _parse_float(row['Loc_X']),
                _parse_float(row['Loc_Y']),
                _parse_float(row['Loc_Z']),
            ])
            rot = np.array([
                _parse_float(row['Rot_Roll']),
                _parse_float(row['Rot_Pitch']),
                _parse_float(row['Rot_Yaw']),
            ])

            model = CameraModel(
                cam_id=cam_id,
                loc=loc,
                rot=rot,
                focal_length_mm=focal_length,
                sensor_width_mm=sw,
                sensor_height_mm=sh,
            )
            timeline.add_frame_model(frame_idx, model)
        except (ValueError, KeyError):
            continue

    return timeline


def load_all_cameras(calib_dir: str) -> Dict[int, CameraTimeline]:
    cameras: Dict[int, CameraTimeline] = {}
    pattern = os.path.join(calib_dir, "object_transform_CineCameraActor*.csv")
    files = glob.glob(pattern)

    for fpath in files:
        match = re.search(r"CineCameraActor(\d+)\.csv", fpath)
        if match:
            cam_id = int(match.group(1))
            cameras[cam_id] = load_camera_csv(fpath, cam_id)

    return cameras
