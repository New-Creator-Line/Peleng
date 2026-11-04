"""
Inference script to detect truck cabin across all 5 cameras using fine-tuned model.
Outputs results/detections/detections_2d_cabin.csv.
"""

import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO


def run_cabin_detection(
    weights_path: str = "weights/cabin/peleng_cabin_best.pt",
    videos_dir: str = "хакатон",
    output_csv: str = "results/detections/detections_2d_cabin.csv",
    conf: float = 0.25,
    imgsz: int = 960,
    device: str = "0" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print(f"[INFO] Loading cabin detector from: {weights_path}")
    model = YOLO(weights_path)

    # Discover videos
    video_paths = {}
    for cam_id in range(1, 6):
        pattern = os.path.join(videos_dir, f"*{cam_id}.mp4")
        matches = glob.glob(pattern)
        if matches:
            video_paths[cam_id] = matches[0]

    print(f"[INFO] Running cabin detection on {len(video_paths)} cameras at imgsz={imgsz} on {device}...")
    records = []

    for cam_id in sorted(video_paths.keys()):
        vpath = video_paths[cam_id]
        cap = cv2.VideoCapture(vpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  [Camera {cam_id}] Processing {os.path.basename(vpath)} ({total_frames} frames)...")

        f_idx = 0
        batch_frames = []
        batch_indices = []
        batch_size = 8

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            batch_frames.append(frame)
            batch_indices.append(f_idx)

            if len(batch_frames) >= batch_size:
                results = model.predict(batch_frames, conf=conf, imgsz=imgsz, device=device, verbose=False)
                for f_num, res in zip(batch_indices, results):
                    if len(res.boxes) > 0:
                        # Take the detection with highest confidence
                        best_idx = int(torch.argmax(res.boxes.conf).item())
                        box = res.boxes.xyxy[best_idx].cpu().numpy()
                        c = float(res.boxes.conf[best_idx].item())
                        x1, y1, x2, y2 = box
                        u = (x1 + x2) / 2.0
                        v = (y1 + y2) / 2.0
                        w = x2 - x1
                        h = y2 - y1
                        is_edge = int(x1 <= 5.0 or x2 >= 1915.0 or y1 <= 5.0 or y2 >= 1075.0)
                        records.append({
                            "Frame": f_num,
                            "Cam_ID": cam_id,
                            "u": round(float(u), 2),
                            "v": round(float(v), 2),
                            "w": round(float(w), 2),
                            "h": round(float(h), 2),
                            "conf": round(float(c), 4),
                            "is_occluded": is_edge,
                            "occlusion_score": 0.5 if is_edge else 0.0,
                        })
                batch_frames = []
                batch_indices = []

            f_idx += 1

        if batch_frames:
            results = model.predict(batch_frames, conf=conf, imgsz=imgsz, device=device, verbose=False)
            for f_num, res in zip(batch_indices, results):
                if len(res.boxes) > 0:
                    best_idx = int(torch.argmax(res.boxes.conf).item())
                    box = res.boxes.xyxy[best_idx].cpu().numpy()
                    c = float(res.boxes.conf[best_idx].item())
                    x1, y1, x2, y2 = box
                    u = (x1 + x2) / 2.0
                    v = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1
                    is_edge = int(x1 <= 5.0 or x2 >= 1915.0 or y1 <= 5.0 or y2 >= 1075.0)
                    records.append({
                        "Frame": f_num,
                        "Cam_ID": cam_id,
                        "u": round(float(u), 2),
                        "v": round(float(v), 2),
                        "w": round(float(w), 2),
                        "h": round(float(h), 2),
                        "conf": round(float(c), 4),
                        "is_occluded": is_edge,
                        "occlusion_score": 0.5 if is_edge else 0.0,
                    })

        cap.release()

    df = pd.DataFrame(records)
    df.sort_values(by=["Frame", "Cam_ID"], inplace=True)
    df.to_csv(output_csv, sep=";", index=False)
    print(f"\n[DONE] Saved {len(df)} cabin detections to: '{output_csv}'")
    return output_csv


if __name__ == "__main__":
    run_cabin_detection()

