"""
Peleng YOLOv8 Detector Module.
Performs 2D object detection using fine-tuned YOLOv8 weights and exports
detections formatted for 3D multi-view triangulation and tracking.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class YOLOv8Detector:
    """
    Detector wrapper around fine-tuned YOLOv8 models.
    Supports single image, batch images, or full video directory processing.
    """

    def __init__(
        self,
        weights_path: str = "weights/peleng_yolov8x_best.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            # Fallback to pretrained or default if fine-tuned weights not found yet
            fallback = Path("yolov8x.pt")
            if not self.weights_path.exists() and not fallback.exists():
                pass
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(str(self.weights_path))

    def detect_frame(
        self,
        image: Union[np.ndarray, str, Path],
        target_class_id: int = 0,
    ) -> List[Dict[str, float]]:
        """
        Runs detection on a single frame (numpy array or file path).
        Returns list of detections: [{'u': center_x, 'v': center_y, 'w': width, 'h': height, 'conf': score}]
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, float]] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            if cls != target_class_id:
                continue

            conf = float(boxes.conf[i].item())
            # xywh format in pixels
            xywh = boxes.xywh[i].cpu().numpy()
            u, v, w, h = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])

            detections.append({
                "u": round(u, 2),
                "v": round(v, 2),
                "w": round(w, 2),
                "h": round(h, 2),
                "conf": round(conf, 4),
            })

        return detections

    def process_camera_frames(
        self,
        frames_by_cam: Dict[int, Dict[int, str]],
        output_csv: str = "results/detections_2d_yolo.csv",
    ) -> pd.DataFrame:
        """
        Processes dictionary of camera frames: {cam_id: {frame_idx: image_path}}
        Saves output formatted as: Frame;Cam_ID;u;v;w;h;conf
        """
        records: List[Dict[str, Any]] = []

        all_cam_ids = sorted(frames_by_cam.keys())
        for cam_id in all_cam_ids:
            cam_frames = frames_by_cam[cam_id]
            for frame_idx in sorted(cam_frames.keys()):
                img_path = cam_frames[frame_idx]
                dets = self.detect_frame(img_path)
                for det in dets:
                    records.append({
                        "Frame": frame_idx,
                        "Cam_ID": cam_id,
                        "u": det["u"],
                        "v": det["v"],
                        "w": det["w"],
                        "h": det["h"],
                        "conf": det["conf"],
                    })

        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, sep=";", index=False)
        return df
