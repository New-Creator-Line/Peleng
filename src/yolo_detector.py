"""
Peleng Multi-Model Detector Module.
Supports fine-tuned YOLOv8 and RT-DETR (Real-Time DEtection TRansformer) models.
Includes occlusion detection & aspect ratio analysis for robust multi-view 3D tracking.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from ultralytics import RTDETR, YOLO


def resolve_weight_path(weights_path: Union[str, Path]) -> str:
    """
    Resolves weight file location, searching in canonical weights/pretrained
    or weights/finetuned directories if the file is not at the exact path.
    """
    path = Path(weights_path)
    if path.exists():
        return str(path)

    candidate_pretrained = Path("weights/pretrained") / path.name
    if candidate_pretrained.exists():
        return str(candidate_pretrained)

    candidate_finetuned = Path("weights/finetuned") / path.name
    if candidate_finetuned.exists():
        return str(candidate_finetuned)

    return str(weights_path)


def load_detection_model(weights_path: Union[str, Path]) -> Union[YOLO, RTDETR]:
    """
    Factory loading either RT-DETR or YOLO model depending on weight filename or model type.
    """
    resolved_path = resolve_weight_path(weights_path)
    path_str = str(resolved_path).lower()
    if "rtdetr" in path_str:
        return RTDETR(resolved_path)
    return YOLO(resolved_path)


def estimate_occlusion_status(
    u: float,
    v: float,
    w: float,
    h: float,
    cam_id: Optional[int] = None,
    img_w: int = 1920,
    img_h: int = 1080,
    expected_aspect_ratio: float = 2.4,
) -> Tuple[bool, float]:
    """
    Analyzes 2D bounding box geometry to detect partial occlusion:
    1. Edge clipping: bounding box touches or is near image borders (< 15 px).
    2. Aspect ratio distortion: w / h drops significantly below typical unoccluded side-view ratio.

    Returns:
        (is_occluded: bool, occlusion_score: float in [0.0, 1.0])
    """
    if h <= 0.0 or w <= 0.0:
        return True, 1.0

    x1 = u - w / 2.0
    x2 = u + w / 2.0
    y1 = v - h / 2.0
    y2 = v + h / 2.0

    # 1. Edge occlusion check (truck entering or leaving camera FOV)
    margin = 15.0
    is_edge_clipped = (
        x1 <= margin or x2 >= (img_w - margin) or
        y1 <= margin or y2 >= (img_h - margin)
    )
    if is_edge_clipped:
        return True, 0.90

    # 2. Aspect ratio occlusion check
    aspect_ratio = w / h
    # For lateral/ground cameras (1 to 4), unoccluded aspect ratio is typically 2.0 - 3.2
    if cam_id is None or cam_id in (1, 2, 3, 4):
        if aspect_ratio < 1.55:
            # Drop in width while height remains ~ constant implies horizontal truncation
            score = float(np.clip(1.0 - (aspect_ratio / expected_aspect_ratio), 0.0, 1.0))
            return True, score

    return False, 0.0


class PelengDetector:
    """
    Universal detector wrapper supporting both YOLOv8 and RT-DETR.
    Provides single frame detection, batch processing, and occlusion estimation.
    """

    def __init__(
        self,
        weights_path: str = "weights/peleng_rtdetr_best.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            # Fallback search order: yolo26s-seg -> rtdetr best -> yolov8x best -> base models
            for cand in [
                Path("yolo26s-seg.pt"),
                Path("weights/peleng_rtdetr_best.pt"),
                Path("weights/peleng_yolov8x_best.pt"),
                Path("rtdetr-l.pt"),
                Path("yolov8x.pt"),
            ]:
                if cand.exists():
                    self.weights_path = cand
                    break

        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = load_detection_model(self.weights_path)

    def detect_frame(
        self,
        image: Union[np.ndarray, str, Path],
        target_class_id: int = 0,
        cam_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs detection on a single frame.
        Returns list of detections with:
        {'u', 'v', 'w', 'h', 'conf', 'aspect_ratio', 'is_occluded', 'occlusion_score'}
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
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
            xywh = boxes.xywh[i].cpu().numpy()
            u, v, w, h = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])

            is_occ, occ_score = estimate_occlusion_status(u, v, w, h, cam_id=cam_id)
            ar = round(w / max(1e-3, h), 3)

            detections.append({
                "u": round(u, 2),
                "v": round(v, 2),
                "w": round(w, 2),
                "h": round(h, 2),
                "conf": round(conf, 4),
                "aspect_ratio": ar,
                "is_occluded": is_occ,
                "occlusion_score": round(occ_score, 3),
            })

        return detections

    def process_camera_frames(
        self,
        frames_by_cam: Dict[int, Dict[int, str]],
        output_csv: str = "results/detections/detections_2d_rtdetr.csv",
    ) -> pd.DataFrame:
        """
        Processes dictionary of camera frames: {cam_id: {frame_idx: image_path}}
        Saves output formatted as: Frame;Cam_ID;u;v;w;h;conf;is_occluded;occlusion_score
        """
        records: List[Dict[str, Any]] = []

        all_cam_ids = sorted(frames_by_cam.keys())
        for cam_id in all_cam_ids:
            cam_frames = frames_by_cam[cam_id]
            for frame_idx in sorted(cam_frames.keys()):
                img_path = cam_frames[frame_idx]
                dets = self.detect_frame(img_path, cam_id=cam_id)
                for det in dets:
                    records.append({
                        "Frame": frame_idx,
                        "Cam_ID": cam_id,
                        "u": det["u"],
                        "v": det["v"],
                        "w": det["w"],
                        "h": det["h"],
                        "conf": det["conf"],
                        "is_occluded": 1 if det["is_occluded"] else 0,
                        "occlusion_score": det["occlusion_score"],
                    })

        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, sep=";", index=False)
        return df


# Backwards compatibility alias
YOLOv8Detector = PelengDetector
