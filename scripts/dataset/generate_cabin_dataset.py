"""
Dataset generator: extracts frames from multi-camera videos and projects
exact 3D truck cabin ground-truth into 2D YOLO bounding boxes.
"""

import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from src.camera_model import load_all_cameras, CameraModel
from src.evaluator import load_ground_truth
from src.cuboid_ukf import cuboid_to_camera_bbox
from src.truck_model import TruckModel


def generate_cabin_dataset(
    videos_dir: str = "хакатон",
    output_dir: str = "data/cabin_dataset",
    step: int = 3,  # every 3rd frame (~1660 total images)
    val_ratio: float = 0.15,
):
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

    gt_path = glob.glob(os.path.join(videos_dir, "object_transform_SM_AA_track.csv"))[0]
    gt_df = load_ground_truth(gt_path)
    cams = load_all_cameras(videos_dir)

    gx = gt_df["Loc_X"].values
    gy = gt_df["Loc_Y"].values
    gz = gt_df["Loc_Z"].values
    vx = np.gradient(gx)
    vy = np.gradient(gy)
    yaws = np.arctan2(vy, vx)

    cabin_truck = TruckModel(length=420.0, width=320.0, height=360.0, center_forward=0.0, center_up=180.0)

    # Find video files
    video_paths: Dict[int, str] = {}
    for cam_id in range(1, 6):
        pattern = os.path.join(videos_dir, f"*{cam_id}.mp4")
        matches = glob.glob(pattern)
        if matches:
            video_paths[cam_id] = matches[0]

    print(f"[INFO] Found {len(video_paths)} videos. Generating cabin dataset at '{output_dir}'...")

    total_samples = 0
    train_count = 0
    val_count = 0

    for cam_id, vpath in video_paths.items():
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Processing Camera {cam_id}: {os.path.basename(vpath)} ({total_f} frames)...")

        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or f_idx >= len(gt_df):
                break

            if f_idx % step == 0:
                yaw = yaws[f_idx]
                state = np.array([gx[f_idx], gy[f_idx], gz[f_idx], 0, yaw, 0, 0, 420.0, 320.0, 360.0])
                m = cams[cam_id].get_model(f_idx)
                cab_bbox = cuboid_to_camera_bbox(state, m, img_w=1920, img_h=1080, truck=cabin_truck)

                # Check if visible and sufficiently sized in camera view
                if cab_bbox is not None:
                    x1, y1, x2, y2 = cab_bbox
                    w_box = x2 - x1
                    h_box = y2 - y1

                    # Only keep detections with plausible size and within frame
                    if w_box >= 12.0 and h_box >= 12.0:
                        is_val = (f_idx % 20 == 0) or ((f_idx // step) % int(1.0 / val_ratio) == 0)
                        split = "val" if is_val else "train"

                        img_name = f"cam{cam_id}_f{f_idx:04d}.jpg"
                        lbl_name = f"cam{cam_id}_f{f_idx:04d}.txt"

                        img_dst = os.path.join(output_dir, "images", split, img_name)
                        lbl_dst = os.path.join(output_dir, "labels", split, lbl_name)

                        # Write image
                        cv2.imwrite(img_dst, frame)

                        # Write YOLO label: class x_center y_center width height
                        cx = ((x1 + x2) / 2.0) / 1920.0
                        cy = ((y1 + y2) / 2.0) / 1080.0
                        nw = w_box / 1920.0
                        nh = h_box / 1080.0

                        with open(lbl_dst, "w") as f_lbl:
                            f_lbl.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                        total_samples += 1
                        if is_val:
                            val_count += 1
                        else:
                            train_count += 1

            f_idx += 1
        cap.release()

    # Create data.yaml
    yaml_content = f"""path: {os.path.abspath(output_dir).replace('\\\\', '/')}
train: images/train
val: images/val

nc: 1
names: ['cabin']
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)

    print(f"\n[DONE] Cabin dataset generated:")
    print(f"       Total images: {total_samples} (Train: {train_count}, Val: {val_count})")
    print(f"       YAML config: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    generate_cabin_dataset(step=4)
