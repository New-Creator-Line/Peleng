"""
Module 1: 2D Detection and Instance Segmentation (YOLO26-seg / YOLO11-seg).
Implements Section 3 of plan_YOLO_Kalman.md:
- Supports native YOLO26-seg (yolo26s-seg.pt, yolo26x-seg.pt, etc.)
- Spatial moments center of mass (cv2.moments): u_c = M10/M00, v_c = M01/M00
- Visible bounds w_px = x_max - x_min, h_px = y_max - y_min
- Edge truncation detection is_truncated: x_min <= 2 or x_max >= W_img - 3
- Batch multi-camera inference for N camera streams
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class DetectionResult:
    cam_id: Optional[int]
    u: float               # Centroid x (from moments)
    v: float               # Centroid y (from moments)
    w: float               # Visible width in pixels
    h: float               # Visible height in pixels
    conf: float            # Detection confidence
    polygon: np.ndarray    # (N, 2) Polygon coordinates in pixels
    is_truncated: bool     # True if object touches image boundaries
    class_id: int          # Target class ID
    class_name: str        # Class label (e.g. 'truck')


def compute_mask_moments(
    polygon: np.ndarray,
    img_w: int = 1920,
    img_h: int = 1080,
    edge_margin_px: float = 3.0,
) -> Tuple[float, float, float, float, bool]:
    """
    Computes exact mask center of mass and bounding dimensions via spatial moments:
        u_c = M_10 / M_00,  v_c = M_01 / M_00
    Also flags whether the mask touches image borders (is_truncated).
    """
    pts = np.asarray(polygon, dtype=np.float32)
    if len(pts) < 3:
        # Fallback for degenerate polygons
        if len(pts) > 0:
            x_min, y_min = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
            x_max, y_max = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
            u_c, v_c = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
            w_px, h_px = max(1.0, x_max - x_min), max(1.0, y_max - y_min)
            is_trunc = (x_min <= edge_margin_px or x_max >= (img_w - edge_margin_px) or
                        y_min <= edge_margin_px or y_max >= (img_h - edge_margin_px))
            return u_c, v_c, w_px, h_px, is_trunc
        return 0.0, 0.0, 0.0, 0.0, True

    x_min, y_min = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x_max, y_max = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    w_px = max(1.0, x_max - x_min)
    h_px = max(1.0, y_max - y_min)

    # Edge truncation flag: truck entering/leaving FOV
    is_truncated = bool(
        x_min <= edge_margin_px or x_max >= (img_w - edge_margin_px) or
        y_min <= edge_margin_px or y_max >= (img_h - edge_margin_px)
    )

    # Calculate spatial moments using cv2.moments on contour
    moments = cv2.moments(pts)
    m00 = moments.get("m00", 0.0)
    if abs(m00) > 1e-4:
        u_c = float(moments["m10"] / m00)
        v_c = float(moments["m01"] / m00)
    else:
        # Fallback to polygon mean if contour area is degenerate
        u_c = float(np.mean(pts[:, 0]))
        v_c = float(np.mean(pts[:, 1]))

    return u_c, v_c, w_px, h_px, is_truncated


def suppress_nested_detections(
    detections: List[DetectionResult],
    iomin_threshold: float = 0.50,
) -> List[DetectionResult]:
    """
    Suppresses nested / duplicate detections where a smaller sub-part (e.g. cabin)
    is geometrically enclosed inside a larger bounding box (e.g. whole truck).

    If IoMin = Area(intersection) / min(Area(A), Area(B)) >= iomin_threshold,
    the smaller box is suppressed, the larger enclosing box is retained,
    and the retained box's confidence is set to max(conf_large, conf_small).
    """
    if len(detections) <= 1:
        return detections

    # Calculate bounding box coordinates and areas for each detection
    boxes = []
    for d in detections:
        x1 = d.u - d.w / 2.0
        y1 = d.v - d.h / 2.0
        x2 = d.u + d.w / 2.0
        y2 = d.v + d.h / 2.0
        area = max(1.0, d.w * d.h)
        boxes.append([x1, y1, x2, y2, area, d.conf])

    boxes_arr = np.array(boxes, dtype=np.float64)
    # Sort by area descending (largest box first)
    order = list(np.argsort(-boxes_arr[:, 4]))

    retained_indices: List[int] = []
    updated_confs = {i: detections[i].conf for i in range(len(detections))}

    while len(order) > 0:
        curr_idx = order[0]
        retained_indices.append(curr_idx)
        if len(order) == 1:
            break

        remaining = order[1:]
        surviving = []
        b_curr = boxes_arr[curr_idx]

        for rem_idx in remaining:
            b_rem = boxes_arr[rem_idx]
            # Compute axis-aligned intersection
            ix1 = max(b_curr[0], b_rem[0])
            iy1 = max(b_curr[1], b_rem[1])
            ix2 = min(b_curr[2], b_rem[2])
            iy2 = min(b_curr[3], b_rem[3])
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            min_area = min(b_curr[4], b_rem[4])
            iomin = inter / min_area if min_area > 0 else 0.0

            if iomin >= iomin_threshold:
                # b_rem is geometrically nested inside b_curr!
                # Suppress b_rem, absorb its confidence if higher
                updated_confs[curr_idx] = max(updated_confs[curr_idx], detections[rem_idx].conf)
            else:
                surviving.append(rem_idx)

        order = surviving

    # Reconstruct filtered DetectionResult list maintaining original order
    cleaned: List[DetectionResult] = []
    for idx in sorted(retained_indices):
        orig = detections[idx]
        new_conf = round(float(updated_confs[idx]), 4)
        if new_conf != orig.conf:
            cleaned.append(DetectionResult(
                cam_id=orig.cam_id,
                u=orig.u,
                v=orig.v,
                w=orig.w,
                h=orig.h,
                conf=new_conf,
                polygon=orig.polygon,
                is_truncated=orig.is_truncated,
                class_id=orig.class_id,
                class_name=orig.class_name,
            ))
        else:
            cleaned.append(orig)

    return cleaned


class YOLO26SegmentationDetector:
    """
    Production detector using YOLO26-seg (with fallback to YOLO11-seg).
    Optimized for multi-camera batch inference (batch=N) in TensorRT / PyTorch.
    """

    def __init__(
        self,
        model_name: str = "yolo26x-seg.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_class_names: Optional[List[str]] = None,
        filter_nested: bool = True,
        iomin_threshold: float = 0.50,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.filter_nested = filter_nested
        self.iomin_threshold = iomin_threshold

        # Target class resolution
        self.target_class_names = [c.lower() for c in (target_class_names or ["truck"])]

        # Load model using Ultralytics
        self.model = self._load_model(model_name)
        self.target_class_ids = self._resolve_target_class_ids()

    def _load_model(self, model_path_or_name: str) -> YOLO:
        """
        Loads the YOLO26-seg model.
        Falls back to other sizes or yolo11s-seg if requested.
        """
        path = Path(model_path_or_name)
        if path.exists():
            return YOLO(str(path))

        # Check weights directory
        for cand in [
            Path("weights") / path.name,
            Path("weights/pretrained") / path.name,
            Path("weights/finetuned") / path.name,
        ]:
            if cand.exists():
                return YOLO(str(cand))

        # Direct Ultralytics download by name (e.g. yolo26s-seg.pt, yolo11s-seg.pt)
        return YOLO(model_path_or_name)

    def _resolve_target_class_ids(self) -> List[int]:
        """Maps class names (e.g. 'truck') to integer class IDs."""
        ids: List[int] = []
        if not hasattr(self.model, "names") or not self.model.names:
            return [0]  # Default single class

        for idx, name in self.model.names.items():
            if str(name).lower() in self.target_class_names:
                ids.append(int(idx))

        if not ids:
            # If target name not found in model, assume class 0 for custom-trained single-class model
            ids = [0]
        return ids

    def detect_batch(
        self,
        frames: List[np.ndarray],
        cam_ids: Optional[List[int]] = None,
    ) -> List[List[DetectionResult]]:
        """
        Runs batch inference across N camera frames in a single forward pass (batch=N).
        """
        if not frames:
            return []

        n_cams = len(frames)
        if cam_ids is None:
            cam_ids = list(range(1, n_cams + 1))

        # Run inference
        results = self.model.predict(
            source=frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        batch_detections: List[List[DetectionResult]] = []

        for cam_idx, (frame, r) in enumerate(zip(frames, results)):
            cam_id = cam_ids[cam_idx]
            img_h, img_w = frame.shape[:2]
            frame_dets: List[DetectionResult] = []

            if r.boxes is not None and len(r.boxes) > 0:
                has_masks = r.masks is not None and len(r.masks) > 0

                for i in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[i].item())
                    if cls_id not in self.target_class_ids:
                        continue

                    conf = float(r.boxes.conf[i].item())
                    cls_name = self.model.names.get(cls_id, str(cls_id))

                    if has_masks and i < len(r.masks.xy):
                        poly = r.masks.xy[i]
                        u_c, v_c, w_px, h_px, is_trunc = compute_mask_moments(
                            poly, img_w=img_w, img_h=img_h
                        )
                    else:
                        # Fallback if no polygon mask available (box mode)
                        xywh = r.boxes.xywh[i].cpu().numpy()
                        u_c, v_c, w_px, h_px = float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
                        xyxy = r.boxes.xyxy[i].cpu().numpy()
                        is_trunc = bool(
                            xyxy[0] <= 3.0 or xyxy[2] >= (img_w - 3.0) or
                            xyxy[1] <= 3.0 or xyxy[3] >= (img_h - 3.0)
                        )
                        poly = np.array([
                            [xyxy[0], xyxy[1]],
                            [xyxy[2], xyxy[1]],
                            [xyxy[2], xyxy[3]],
                            [xyxy[0], xyxy[3]],
                        ], dtype=np.float32)

                    frame_dets.append(DetectionResult(
                        cam_id=cam_id,
                        u=round(u_c, 2),
                        v=round(v_c, 2),
                        w=round(w_px, 2),
                        h=round(h_px, 2),
                        conf=round(conf, 4),
                        polygon=poly,
                        is_truncated=is_trunc,
                        class_id=cls_id,
                        class_name=cls_name,
                    ))

                # Filter nested sub-parts (e.g. cabin detected inside full truck)
                if self.filter_nested and len(frame_dets) > 1:
                    frame_dets = suppress_nested_detections(
                        frame_dets, iomin_threshold=self.iomin_threshold
                    )

            batch_detections.append(frame_dets)

        return batch_detections

    def detect_frame(
        self,
        frame: np.ndarray,
        cam_id: Optional[int] = None,
    ) -> List[DetectionResult]:
        """Runs detection on a single frame."""
        batch_res = self.detect_batch([frame], cam_ids=[cam_id] if cam_id is not None else None)
        return batch_res[0] if batch_res else []
