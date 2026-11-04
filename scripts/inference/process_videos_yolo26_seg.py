"""
Process Multi-Camera Video Streams with Fine-Tuned YOLO26x-seg (Instance Segmentation)
for an arbitrary number of cameras (N >= 1).

Features:
1. Dynamic discovery of N >= 1 camera streams (`discover_camera_videos`).
2. High-precision instance mask overlays (alpha-blended polygon with sharp contour) + EMA box stabilization.
3. Exact spatial centroid (u_c, v_c) calculated via cv2.moments on mask polygon.
4. Edge truncation detection (is_truncated flag for truck entering/leaving FOV).
5. Individual annotated videos for all N cameras: results/videos/camera_{i}_yolo26_seg.mp4
6. Synchronized N-Camera Multi-View Dashboard: results/videos/multiview_dashboard_yolo26_seg.mp4
7. Unified 2D Detections CSV + automatic 3D UKF + Sliding-Window RTS (lag=30) trajectory output.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Auto-configure TensorRT DLL search paths on Windows
for trt_cand in [r"C:\TensorRT\TensorRT-11.4.0.106\bin", r"C:\TensorRT\TensorRT-11.4.0.106\lib"]:
    if os.path.isdir(trt_cand) and trt_cand not in os.environ.get("PATH", ""):
        os.environ["PATH"] = trt_cand + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(trt_cand)
            except Exception:
                pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.async_capture import (
    AsyncMultiCameraCapture,
    NVDEC_STATUS,
    assemble_multiview_grid,
    compute_dashboard_grid_dims,
    discover_camera_videos,
)
from src.async_writer import AsyncMultiVideoWriter, transcode_to_nvenc_parallel

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = None

from src.detector import compute_mask_moments, suppress_nested_detections, DetectionResult

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("YOLO26VideoProcessor")


def draw_styled_segmentation(
    frame: np.ndarray,
    polygon: np.ndarray,
    box: Tuple[float, float, float, float],
    conf: float,
    u_c: float,
    v_c: float,
    is_trunc: bool,
    cam_id: int,
    alpha: float = 0.40,
) -> None:
    """Draws alpha-blended polygon mask, glowing centroid, bounding box, and badges."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

    # Color palette
    if is_trunc:
        mask_color = (0, 140, 255)    # Amber for truncated / entering boundary
        contour_color = (0, 200, 255)
    elif conf > 0.70:
        mask_color = (255, 120, 0)    # Azure / Cyan for strong detection
        contour_color = (255, 200, 50)
    else:
        mask_color = (0, 220, 100)    # Emerald green
        contour_color = (50, 255, 150)

    # 1. Alpha-blended mask (bounded ROI to eliminate 6MB full-frame copy & blend)
    if len(polygon) >= 3:
        pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        px1 = max(0, min(int(np.min(pts[:, 0, 0])), x1))
        py1 = max(0, min(int(np.min(pts[:, 0, 1])), y1))
        px2 = min(w_img, max(int(np.max(pts[:, 0, 0])) + 1, x2 + 1))
        py2 = min(h_img, max(int(np.max(pts[:, 0, 1])) + 1, y2 + 1))
        if px2 > px1 and py2 > py1:
            roi = frame[py1:py2, px1:px2]
            roi_overlay = roi.copy()
            pts_roi = pts - np.array([[[px1, py1]]], dtype=np.int32)
            cv2.fillPoly(roi_overlay, [pts_roi], mask_color)
            cv2.addWeighted(roi_overlay, alpha, roi, 1.0 - alpha, 0, roi)
        cv2.polylines(frame, [pts], isClosed=True, color=contour_color, thickness=2, lineType=cv2.LINE_AA)

    # 2. Bounding box with corner accents
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    corner_len = min(20, max(5, (x2 - x1) // 5))
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), contour_color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), contour_color, 3)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), contour_color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), contour_color, 3)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), contour_color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), contour_color, 3)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), contour_color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), contour_color, 3)

    # 3. Exact centroid crosshair (moments)
    u_int, v_int = int(round(u_c)), int(round(v_c))
    if 0 <= u_int < w_img and 0 <= v_int < h_img:
        cv2.circle(frame, (u_int, v_int), 7, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (u_int, v_int), 9, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(frame, (u_int - 15, v_int), (u_int + 15, v_int), (0, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(frame, (u_int, v_int - 15), (u_int, v_int + 15), (0, 255, 255), 1, lineType=cv2.LINE_AA)

    # 4. Label badge
    badge_text = f"TRUCK {conf:.1%} | ({u_c:.0f}, {v_c:.0f})"
    if is_trunc:
        badge_text += " [EDGE]"
    (tw, th), bl = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    by1 = max(0, y1 - th - 10)
    by2 = by1 + th + 10
    bx2 = min(w_img - 1, x1 + tw + 14)
    cv2.rectangle(frame, (x1, by1), (bx2, by2), (20, 24, 30), -1)
    cv2.rectangle(frame, (x1, by1), (bx2, by2), contour_color, 1)
    cv2.putText(frame, badge_text, (x1 + 6, by2 - bl - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def draw_hud_header(frame: np.ndarray, cam_id: int, frame_idx: int, total_frames: int, n_dets: int) -> None:
    """Draws sleek telemetry HUD at top of each camera frame."""
    h_bar = 44
    overlay = frame[:h_bar, :].copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h_bar), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.85, frame[:h_bar, :], 0.15, 0, frame[:h_bar, :])

    # Camera label & status
    status_str = "TRACKING" if n_dets > 0 else "SEARCHING"
    status_col = (0, 255, 120) if n_dets > 0 else (120, 120, 140)

    cv2.putText(frame, f"CAM {cam_id}", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (125, 24), 6, status_col, -1, lineType=cv2.LINE_AA)
    cv2.putText(frame, status_str, (140, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 1, lineType=cv2.LINE_AA)

    # Frame counter and model name
    right_str = f"FRAME: {frame_idx:04d}/{total_frames:04d} | YOLO26x-seg (COCO + Peleng fine-tuned)"
    (rw, _), _ = cv2.getTextSize(right_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, right_str, (frame.shape[1] - rw - 18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 210, 220), 1, lineType=cv2.LINE_AA)


def draw_telemetry_tile(
    tile_w: int,
    tile_h: int,
    frame_idx: int,
    total_frames: int,
    cam_stats: Dict[int, Optional[Dict[str, Any]]],
    fps: float,
) -> np.ndarray:
    """Draws telemetry tile for the synchronized N-camera multi-view dashboard."""
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tile[:] = (20, 24, 32)

    # Title header
    cv2.rectangle(tile, (0, 0), (tile_w, 40), (28, 35, 48), -1)
    cv2.putText(tile, "PELENG MULTI-CAMERA FUSION STATUS", (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, lineType=cv2.LINE_AA)

    # Live frame info
    cv2.putText(tile, f"SYNC TIME: {frame_idx / fps:.2f}s  |  FRAME: {frame_idx:04d}/{total_frames:04d}", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Camera visibility table (dynamic N cameras)
    cam_ids = sorted(cam_stats.keys())
    num_cams = len(cam_ids)
    active_cams = sum(1 for c, st in cam_stats.items() if st is not None)
    cv2.putText(
        tile,
        f"CAMERAS DETECTING TARGET: {active_cams} / {num_cams}",
        (18, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 150) if active_cams >= 2 else (0, 165, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    y_offset = 140
    row_step = min(32, max(16, (tile_h - 195) // max(1, num_cams)))
    for c_id in cam_ids:
        if y_offset > tile_h - 50:
            break
        st = cam_stats.get(c_id)
        if st is not None:
            c_text = f"CAM {c_id}: VISIBLE | Conf: {st['conf']:.1%} | ({st['u']:.0f}, {st['v']:.0f})"
            dot_color = (0, 255, 100)
        else:
            c_text = f"CAM {c_id}: BLIND / OCCLUDED"
            dot_color = (80, 80, 95)

        cv2.circle(tile, (25, y_offset - 5), 5, dot_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(tile, c_text, (42, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 235, 240) if st else (120, 125, 135), 1, lineType=cv2.LINE_AA)
        y_offset += row_step

    # Triangulation readiness banner
    y_offset = max(y_offset + 12, tile_h - 30)
    if active_cams >= 2:
        tri_text = "3D TRIANGULATION: READY (Rays >= 2)"
        tri_col = (0, 255, 120)
    else:
        tri_text = "3D TRIANGULATION: PREDICTION ONLY (Kalman/Prior)"
        tri_col = (0, 140, 255)

    cv2.rectangle(tile, (15, y_offset - 20), (tile_w - 15, y_offset + 15), (28, 35, 48), -1)
    cv2.putText(tile, tri_text, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tri_col, 2, lineType=cv2.LINE_AA)

    return tile


def _predict_batch_dynamic(
    model: YOLO,
    frames_list: List[np.ndarray],
    is_trt_engine: bool,
    conf_threshold: float,
    imgsz: int,
    device: str,
    trt_fixed_batch: int = 5,
) -> List[Any]:
    """
    Runs GPU batch inference for an arbitrary number of cameras N >= 1.
    If using a TensorRT `.engine` compiled with a fixed batch size (e.g., 5),
    automatically chunks/pads the frame list so any N works seamlessly.
    """
    if not is_trt_engine:
        return list(
            model.predict(
                source=frames_list,
                conf=conf_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
        )

    all_results: List[Any] = []
    n_frames = len(frames_list)
    for start_idx in range(0, n_frames, trt_fixed_batch):
        chunk = frames_list[start_idx : start_idx + trt_fixed_batch]
        orig_len = len(chunk)
        if orig_len < trt_fixed_batch:
            dummy = np.zeros_like(chunk[0])
            chunk = chunk + [dummy] * (trt_fixed_batch - orig_len)
        chunk_res = model.predict(
            source=chunk,
            conf=conf_threshold,
            imgsz=imgsz,
            device=device,
            rect=False,
            verbose=False,
        )
        all_results.extend(chunk_res[:orig_len])
    return all_results


def process_videos_yolo26_seg(
    weights_path: str = "weights/finetuned/peleng_yolo26x_seg_clean_polish_best_fp16.engine",
    videos_dir: str = "data/raw/hackathon",
    output_dir: str = "results/videos",
    csv_output: str = "results/detections/detections_2d_yolo26_seg.csv",
    trajectory_csv: Optional[str] = "results/trajectories/output_trajectory_ukf.csv",
    conf_threshold: float = 0.25,
    imgsz: int = 1024,
    device: str = "0",
    max_frames: Optional[int] = None,
    save_videos: bool = True,
    only_dashboard: bool = False,
    run_3d_pipeline: bool = True,
    rts_lag: int = 30,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(csv_output).parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve weights path
    wt_path = Path(weights_path)
    if not wt_path.exists():
        for cand in [
            Path("weights/finetuned/peleng_yolo26x_seg_clean_polish_best_fp16.engine"),
            Path("weights/finetuned/peleng_yolo26x_seg_clean_best.engine"),
            Path("weights/finetuned/peleng_yolo26x_seg_clean_polish_best.pt"),
            Path("weights/finetuned/peleng_yolo26x_seg_roboflow_best.pt"),
            Path("runs/segment/runs/segment/peleng_yolo26x_seg_roboflow/weights/best.pt"),
        ]:
            if cand.exists():
                wt_path = cand
                break
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

    logger.info("Loading fine-tuned YOLO26x-seg from: %s", wt_path.resolve())
    is_trt_engine = wt_path.suffix == ".engine"
    if is_trt_engine:
        model = YOLO(str(wt_path), task="segment")
    else:
        model = YOLO(str(wt_path))

    # Optional 3D Trajectory for HUD overlay
    cam_models = None
    traj_dict = None
    if trajectory_csv and Path(trajectory_csv).exists():
        try:
            from src.camera_model import load_all_cameras
            from src.cuboid_model import TruckCuboid3D, draw_cuboid_wireframe_hud, compute_uncertainty_ellipsoid_3d, draw_uncertainty_ellipsoid_hud
            cam_models = load_all_cameras(videos_dir)
            t_df = pd.read_csv(trajectory_csv, sep=';')
            traj_dict = {int(r["Frame"]): r for _, r in t_df.iterrows()}
            logger.info("Loaded 3D trajectory with %d frames for 3D HUD rendering", len(traj_dict))
        except Exception as e:
            logger.warning("Could not load 3D trajectory for HUD rendering: %s", e)

    # 2. Dynamically discover all N >= 1 camera video streams in videos_dir
    cam_videos = discover_camera_videos(videos_dir)
    cam_ids = sorted(cam_videos.keys())
    num_cams = len(cam_ids)
    logger.info("Discovered %d camera video stream(s) in '%s': %s", num_cams, videos_dir, cam_ids)
    for c_id in cam_ids:
        logger.info("  Camera %d: %s", c_id, Path(cam_videos[c_id]).name)

    # 3. Open hardware-accelerated NVDEC capture streams
    logger.info(
        "NVDEC Video Codec SDK %s Status: Backend=%s | H.264=%s | NVDEC Engines=%s",
        NVDEC_STATUS.get("sdk_version", "13.1.24"),
        NVDEC_STATUS.get("backend", "NVIDIA_NVDEC_H264_CUVID"),
        NVDEC_STATUS.get("h264_supported", True),
        NVDEC_STATUS.get("num_nvdec_engines", 1),
    )
    multi_cap = AsyncMultiCameraCapture(cam_videos, queue_size=4, use_nvdec=True)

    total_frames = multi_cap.total_frames
    fps = multi_cap.fps or 30.0
    w_orig = multi_cap.width
    h_orig = multi_cap.height
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    logger.info(
        "Streams opened (%d cameras): %dx%d @ %.1f FPS | Total Frames to Process: %d",
        num_cams, w_orig, h_orig, fps, total_frames,
    )

    # 4. Initialize Asynchronous Multi-Stream Video Writers
    out_paths: Dict[int, Path] = {}
    tile_w, tile_h = 640, 360
    dashboard_path = out_dir / "multiview_dashboard_yolo26_seg.mp4"
    async_writer: Optional[AsyncMultiVideoWriter] = None

    if save_videos:
        async_writer = AsyncMultiVideoWriter(queue_size=16)
        if not only_dashboard:
            for c_id in cam_ids:
                p = out_dir / f"camera_{c_id}_yolo26_seg.mp4"
                async_writer.add_stream(c_id, p, fps, (w_orig, h_orig))
                out_paths[c_id] = p

        # Dynamic Multi-view dashboard grid for N cameras + 1 telemetry tile
        _, _, dashboard_w, dashboard_h = compute_dashboard_grid_dims(
            num_cams, include_telemetry=True, tile_w=tile_w, tile_h=tile_h
        )
        async_writer.add_stream("dashboard", dashboard_path, fps, (dashboard_w, dashboard_h))

    detection_records: List[Dict[str, Any]] = []
    # Causal EMA state per camera for 2D bounding-box anti-shake visualization
    box_ema_state: Dict[int, np.ndarray] = {}
    ema_alpha = 0.35
    t_start = time.perf_counter()

    logger.info("Processing %d synchronized camera stream(s)...", num_cams)

    try:
        for f_idx in range(total_frames):
            raw_frames = multi_cap.read_sync(timeout=3.0)
            if raw_frames is None or len(raw_frames) < num_cams:
                logger.warning("Stream ended early at frame %d", f_idx)
                break

            # Batch predict on all N camera frames
            frames_list = [raw_frames[c_id] for c_id in cam_ids]
            results = _predict_batch_dynamic(
                model=model,
                frames_list=frames_list,
                is_trt_engine=is_trt_engine,
                conf_threshold=conf_threshold,
                imgsz=imgsz,
                device=device,
            )

            cam_stats: Dict[int, Optional[Dict[str, Any]]] = {c: None for c in cam_ids}
            annotated_tiles: Dict[int, np.ndarray] = {}

            for idx, c_id in enumerate(cam_ids):
                frame = raw_frames[c_id]
                res = results[idx]
                n_dets = 0
                cam_candidates: List[DetectionResult] = []

                if res.boxes is not None and len(res.boxes) > 0:
                    for b_idx in range(len(res.boxes)):
                        cls_id = int(res.boxes.cls[b_idx].item())
                        if cls_id != 0:
                            continue

                        conf = float(res.boxes.conf[b_idx].item())
                        xyxy = res.boxes.xyxy[b_idx].cpu().numpy().tolist()

                        # Get polygon
                        poly: np.ndarray
                        if res.masks is not None and len(res.masks) > b_idx:
                            poly = res.masks.xy[b_idx]
                        else:
                            # Fallback polygon from box
                            bx1, by1, bx2, by2 = xyxy
                            poly = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)

                        u_c, v_c, w_px, h_px, is_trunc = compute_mask_moments(poly, img_w=w_orig, img_h=h_orig)

                        cam_candidates.append(DetectionResult(
                            cam_id=c_id,
                            u=round(float(u_c), 2),
                            v=round(float(v_c), 2),
                            w=round(float(w_px), 2),
                            h=round(float(h_px), 2),
                            conf=round(float(conf), 4),
                            polygon=poly,
                            is_truncated=is_trunc,
                            class_id=cls_id,
                            class_name="truck",
                        ))

                # Suppress nested sub-parts (e.g. smaller cabin box inside full truck box)
                surviving_dets = suppress_nested_detections(cam_candidates, iomin_threshold=0.50)

                if not surviving_dets:
                    box_ema_state.pop(c_id, None)

                for det in surviving_dets:
                    raw_box = np.array([
                        max(0.0, det.u - det.w / 2.0),
                        max(0.0, det.v - det.h / 2.0),
                        min(float(w_orig - 1), det.u + det.w / 2.0),
                        min(float(h_orig - 1), det.v + det.h / 2.0),
                    ], dtype=np.float64)

                    if c_id in box_ema_state and len(surviving_dets) == 1:
                        smooth_box = ema_alpha * raw_box + (1.0 - ema_alpha) * box_ema_state[c_id]
                        box_ema_state[c_id] = smooth_box
                    else:
                        smooth_box = raw_box
                        if len(surviving_dets) == 1:
                            box_ema_state[c_id] = raw_box

                    bx1, by1, bx2, by2 = [float(v) for v in smooth_box]

                    # Draw styling
                    draw_styled_segmentation(frame, det.polygon, (bx1, by1, bx2, by2), det.conf, det.u, det.v, det.is_truncated, cam_id=c_id)

                    detection_records.append({
                        "Frame": f_idx,
                        "Cam_ID": c_id,
                        "u": det.u,
                        "v": det.v,
                        "w": det.w,
                        "h": det.h,
                        "conf": det.conf,
                        "is_truncated": 1 if det.is_truncated else 0,
                        "num_poly_pts": len(det.polygon),
                    })
                    cam_stats[c_id] = {
                        "conf": det.conf,
                        "u": det.u,
                        "v": det.v,
                        "is_trunc": det.is_truncated,
                    }
                    n_dets += 1

                if save_videos:
                    # Draw 3D Cuboid & Uncertainty Ellipsoid HUD if available
                    if traj_dict is not None and cam_models is not None and f_idx in traj_dict and c_id in cam_models:
                        tr = traj_dict[f_idx]
                        yaw = float(tr.get("Yaw", 0.0))
                        length = float(tr.get("Length", 629.0))
                        width = float(tr.get("Width", 207.0))
                        height = float(tr.get("Height", 212.0))
                        cuboid = TruckCuboid3D(
                            x=float(tr["Loc_X"]),
                            y=float(tr["Loc_Y"]),
                            z=float(tr["Loc_Z"]),
                            yaw=yaw,
                            length=length,
                            width=width,
                            height=height,
                        )
                        cam_m = cam_models[c_id].get_model(f_idx)
                        draw_cuboid_wireframe_hud(frame, cuboid, cam_m)
                        if "Sigma_X" in tr and "Sigma_Y" in tr and "Sigma_Z" in tr:
                            cov = np.diag([float(tr["Sigma_X"]) ** 2, float(tr["Sigma_Y"]) ** 2, float(tr["Sigma_Z"]) ** 2])
                            ellipsoid = compute_uncertainty_ellipsoid_3d(cov, center=cuboid.get_centroid())
                            draw_uncertainty_ellipsoid_hud(frame, ellipsoid, cam_m)

                    # Draw top HUD
                    draw_hud_header(frame, c_id, f_idx, total_frames, n_dets)

                    # Write individual full-res video asynchronously
                    if not only_dashboard and async_writer is not None:
                        async_writer.write(c_id, frame)

                    # Downscale for dashboard tile
                    tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                    annotated_tiles[c_id] = tile

            if save_videos and async_writer is not None:
                # Build Telemetry Status Pane & assemble N-camera grid dynamically
                telemetry_tile = draw_telemetry_tile(tile_w, tile_h, f_idx, total_frames, cam_stats, fps)
                dashboard_frame = assemble_multiview_grid(annotated_tiles, telemetry_tile, tile_w=tile_w, tile_h=tile_h)
                async_writer.write("dashboard", dashboard_frame)

            if (f_idx + 1) % 100 == 0 or (f_idx + 1) == total_frames:
                elapsed = time.perf_counter() - t_start
                fps_proc = (f_idx + 1) / elapsed
                logger.info(
                    "Processed frame %d/%d (%.1f%%) | Speed: %.1f sync FPS (%d streams parallel)",
                    f_idx + 1, total_frames, (f_idx + 1) / total_frames * 100, fps_proc, num_cams,
                )

    finally:
        multi_cap.close()
        if async_writer is not None:
            async_writer.close()

    # Hardware NVENC Post-Transcode via NVIDIA Video Codec SDK
    if save_videos and FFMPEG_EXE is not None:
        logger.info("Hardware Encoding (NVIDIA NVENC / Video Codec SDK): Finalizing H.264 streams in parallel...")
        videos_to_transcode = []
        if not only_dashboard:
            videos_to_transcode.extend(list(out_paths.values()))
        if dashboard_path.exists():
            videos_to_transcode.append(dashboard_path)

        transcode_to_nvenc_parallel(videos_to_transcode, preset="p2", bitrate="5M", max_workers=min(8, num_cams + 1))

    total_time = time.perf_counter() - t_start
    logger.info(
        "Video processing complete in %.2f s (Average speed: %.1f stream FPS across %d cameras)",
        total_time, (total_frames * num_cams) / total_time if total_time else 0.0, num_cams,
    )

    # Save CSV with semicolon delimiter compatible with main.py pipeline
    df = pd.DataFrame(detection_records)
    if max_frames is None or not Path(csv_output).exists():
        df.to_csv(csv_output, sep=';', index=False)
        logger.info("Saved %d total detections across %d cameras to: %s", len(df), num_cams, csv_output)
    else:
        logger.info("Preview run (max_frames=%d): preserved existing full %s", max_frames, csv_output)

    # Automatically run 3D triangulation + 10D UKF + Sliding-Window RTS (lag=30) to produce final 3D results
    if run_3d_pipeline and max_frames is None and len(df) > 0:
        try:
            from main import run_pipeline
            out_traj = trajectory_csv or "results/trajectories/output_trajectory_ukf.csv"
            logger.info("Running 3D Multi-Camera Triangulation & UKF (RTS sliding window = %d frames)...", rts_lag)
            run_pipeline(
                calib_dir=videos_dir,
                detections_csv=csv_output,
                output_csv=out_traj,
                rts_lag=rts_lag,
            )
        except Exception as e:
            logger.warning("Could not auto-run 3D pipeline after video detection: %s", e)

    return {
        "num_cameras": num_cams,
        "camera_ids": cam_ids,
        "total_frames": total_frames,
        "total_time_s": total_time,
        "dashboard_path": str(dashboard_path),
        "individual_videos": {c: str(p) for c, p in out_paths.items()},
        "csv_path": csv_output,
        "total_detections": len(df),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Process N multi-camera videos with YOLO26x-seg")
    default_weights = "weights/finetuned/peleng_yolo26x_seg_clean_polish_best_fp16.engine"
    if not Path(default_weights).exists():
        default_weights = "weights/finetuned/peleng_yolo26x_seg_clean_best.engine"
    if not Path(default_weights).exists():
        default_weights = "weights/finetuned/peleng_yolo26x_seg_clean_best.onnx"
    if not Path(default_weights).exists():
        default_weights = "weights/finetuned/peleng_yolo26x_seg_clean_polish_best.pt"
    if not Path(default_weights).exists():
        default_weights = "weights/finetuned/peleng_yolo26x_seg_roboflow_best.pt"
    parser.add_argument("--weights", default=default_weights)
    parser.add_argument("--videos-dir", default="data/raw/hackathon")
    parser.add_argument("--output-dir", default="results/videos")
    parser.add_argument("--csv-output", default="results/detections/detections_2d_yolo26_seg.csv")
    parser.add_argument("--trajectory", default="results/trajectories/output_trajectory_ukf.csv", help="3D trajectory CSV for HUD wireframe overlay and output")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames to process for quick preview")
    parser.add_argument("--skip-videos", action="store_true", help="Skip video generation, only extract 2D detections CSV & 3D trajectory")
    parser.add_argument("--only-dashboard", action="store_true", help="Only generate multi-view dashboard video, skip individual camera videos")
    parser.add_argument("--no-3d", action="store_false", dest="run_3d", help="Skip automatic 3D trajectory reconstruction after 2D extraction")
    parser.add_argument("--rts-lag", type=int, default=30, help="Sliding-window RTS lag in frames (default: 30)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_videos_yolo26_seg(
        weights_path=args.weights,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        csv_output=args.csv_output,
        trajectory_csv=args.trajectory,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        max_frames=args.max_frames,
        save_videos=not args.skip_videos,
        only_dashboard=args.only_dashboard,
        run_3d_pipeline=args.run_3d,
        rts_lag=args.rts_lag,
    )
