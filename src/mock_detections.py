import math
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.camera_model import CameraTimeline, load_all_cameras
from src.evaluator import load_ground_truth


def generate_mock_detections(
    calib_dir: str,
    gt_path: str,
    output_csv: str,
    noise_std_px: float = 1.0,
    min_confidence: float = 0.85,
) -> pd.DataFrame:
    """
    Generates realistic 2D bounding boxes (u, v, w, h, conf) for all active cameras
    by projecting ground truth 3D positions with synthetic Gaussian noise and field-of-view filtering.
    Format: Frame;Cam_ID;u;v;w;h;conf
    """
    cameras = load_all_cameras(calib_dir)
    gt_df = load_ground_truth(gt_path)

    records = []

    for _, row in gt_df.iterrows():
        frame_idx = int(row['Frame'])
        pt_3d = np.array([float(row['Loc_X']), float(row['Loc_Y']), float(row['Loc_Z'])])

        for cam_id, timeline in cameras.items():
            model = timeline.get_model(frame_idx)
            u, v, z_cam = model.project_point(pt_3d)

            # Check if point is in front of camera and inside 1920x1080 frame
            if z_cam > 10.0 and (0 <= u < 1920) and (0 <= v < 1080):
                # Apply realistic pixel jitter
                u_noisy = float(u + np.random.normal(0, noise_std_px))
                v_noisy = float(v + np.random.normal(0, noise_std_px))

                # Approximate truck bbox size based on distance
                # Real truck ~8m long (800 units in UE), distance ~15000-40000 units
                scale = 2828.0 / max(100.0, z_cam)
                box_w = float(np.clip(700.0 * scale, 15.0, 300.0))
                box_h = float(np.clip(350.0 * scale, 10.0, 150.0))
                conf = float(np.clip(min_confidence + np.random.uniform(0.0, 0.14), 0.0, 0.99))

                records.append({
                    "Frame": frame_idx,
                    "Cam_ID": cam_id,
                    "u": round(u_noisy, 2),
                    "v": round(v_noisy, 2),
                    "w": round(box_w, 2),
                    "h": round(box_h, 2),
                    "conf": round(conf, 4),
                })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, sep=';', index=False)
    return df


def load_detections(detections_csv_path: str) -> Dict[int, List[Dict[str, float]]]:
    """
    Loads 2D detections and groups them by frame.
    Returns: {frame_idx: [{'cam_id': int, 'u': float, 'v': float, 'w': float, 'h': float, 'conf': float}]}
    """
    df = pd.read_csv(detections_csv_path, sep=';')
    grouped: Dict[int, List[Dict[str, float]]] = {}

    for _, row in df.iterrows():
        frame_idx = int(row['Frame'])
        if frame_idx not in grouped:
            grouped[frame_idx] = []
        grouped[frame_idx].append({
            "cam_id": int(row['Cam_ID']),
            "u": float(row['u']),
            "v": float(row['v']),
            "w": float(row['w']),
            "h": float(row['h']),
            "conf": float(row['conf']),
        })

    return grouped
