"""
Process Multi-Camera Video Streams with Fine-Tuned YOLO26x-seg (Instance Segmentation).

Features:
1. High-precision instance mask overlays (alpha-blended polygon with sharp contour).
2. Exact spatial centroid (u_c, v_c) calculated via cv2.moments on mask polygon.
3. Edge truncation detection (is_truncated flag for truck entering/leaving FOV).
4. Individual annotated videos for Cameras 1 to 5: results/videos/camera_{i}_yolo26_seg.mp4
5. Synchronized 5-Camera Multi-View Dashboard: results/videos/multiview_dashboard_yolo26_seg.mp4
6. Unified 2D Detections CSV: results/detections/detections_2d_yolo26_seg.csv
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

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

    # 1. Alpha-blended mask
    if len(polygon) >= 3:
        pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], mask_color)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
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
    """Draws 6th telemetry tile for the synchronized multi-view dashboard."""
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tile[:] = (20, 24, 32)

    # Title header
    cv2.rectangle(tile, (0, 0), (tile_w, 40), (28, 35, 48), -1)
    cv2.putText(tile, "PELENG MULTI-CAMERA FUSION STATUS", (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, lineType=cv2.LINE_AA)

    # Live frame info
    cv2.putText(tile, f"SYNC TIME: {frame_idx / fps:.2f}s  |  FRAME: {frame_idx:04d}/{total_frames:04d}", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Camera visibility table
    active_cams = sum(1 for c, st in cam_stats.items() if st is not None)
    cv2.putText(tile, f"CAMERAS DETECTING TARGET: {active_cams} / 5", (18, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150) if active_cams >= 2 else (0, 165, 255), 2, lineType=cv2.LINE_AA)

    y_offset = 145
    for c_id in range(1, 6):
        st = cam_stats.get(c_id)
        if st is not None:
            c_text = f"CAM {c_id}: VISIBLE | Conf: {st['conf']:.1%} | ({st['u']:.0f}, {st['v']:.0f})"
            dot_color = (0, 255, 100)
        else:
            c_text = f"CAM {c_id}: BLIND / OCCLUDED"
            dot_color = (80, 80, 95)

        cv2.circle(tile, (25, y_offset - 5), 5, dot_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(tile, c_text, (42, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 235, 240) if st else (120, 125, 135), 1, lineType=cv2.LINE_AA)
        y_offset += 32

    # Triangulation readiness banner
    y_offset += 15
    if active_cams >= 2:
        tri_text = "3D TRIANGULATION: READY (Rays >= 2)"
        tri_col = (0, 255, 120)
    else:
        tri_text = "3D TRIANGULATION: PREDICTION ONLY (Kalman/Prior)"
        tri_col = (0, 140, 255)

    cv2.rectangle(tile, (15, y_offset - 20), (tile_w - 15, y_offset + 15), (28, 35, 48), -1)
    cv2.putText(tile, tri_text, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tri_col, 2, lineType=cv2.LINE_AA)

    return tile


def process_videos_yolo26_seg(
    weights_path: str = "weights/finetuned/peleng_yolo26x_seg_roboflow_best.pt",
    videos_dir: str = "data/raw/hackathon",
    output_dir: str = "results/videos",
    csv_output: str = "results/detections/detections_2d_yolo26_seg.csv",
    trajectory_csv: Optional[str] = "results/trajectories/output_trajectory_ukf.csv",
    conf_threshold: float = 0.25,
    imgsz: int = 1024,
    device: str = "0",
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(csv_output).parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve weights path
    wt_path = Path(weights_path)
    if not wt_path.exists():
        fallback = Path("runs/segment/runs/segment/peleng_yolo26x_seg_roboflow/weights/best.pt")
        if fallback.exists():
            wt_path = fallback
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path} or {fallback}")

    logger.info("Loading fine-tuned YOLO26x-seg from: %s", wt_path.resolve())
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

    # 2. Map video files
    cam_videos: Dict[int, str] = {}
    v_dir = Path(videos_dir)
    for c_id in range(1, 6):
        matches = list(v_dir.glob(f"*Actor{c_id}.mp4"))
        if not matches:
            raise FileNotFoundError(f"Video for Camera {c_id} not found in {videos_dir}")
        cam_videos[c_id] = str(matches[0])
        logger.info("Camera %d: %s", c_id, matches[0].name)

    # 3. Open capture streams
    caps: Dict[int, cv2.VideoCapture] = {}
    for c_id, path in cam_videos.items():
        caps[c_id] = cv2.VideoCapture(path)

    total_frames = int(caps[1].get(cv2.CAP_PROP_FRAME_COUNT))
    fps = caps[1].get(cv2.CAP_PROP_FPS) or 30.0
    w_orig = int(caps[1].get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(caps[1].get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    logger.info("Streams opened: %dx%d @ %.1f FPS | Total Frames to Process: %d", w_orig, h_orig, fps, total_frames)

    # 4. Initialize Video Writers
    # H.264 / mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers: Dict[int, cv2.VideoWriter] = {}
    out_paths: Dict[int, Path] = {}
    for c_id in range(1, 6):
        p = out_dir / f"camera_{c_id}_yolo26_seg.mp4"
        writers[c_id] = cv2.VideoWriter(str(p), fourcc, fps, (w_orig, h_orig))
        out_paths[c_id] = p

    # Multi-view dashboard: 3 cols x 2 rows (each tile 640x360 -> total 1920x720)
    tile_w, tile_h = 640, 360
    dashboard_w, dashboard_h = 1920, 720
    dashboard_path = out_dir / "multiview_dashboard_yolo26_seg.mp4"
    dashboard_writer = cv2.VideoWriter(str(dashboard_path), fourcc, fps, (dashboard_w, dashboard_h))

    detection_records: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    logger.info("Processing synchronized multi-camera streams...")

    try:
        for f_idx in range(total_frames):
            raw_frames: Dict[int, np.ndarray] = {}
            for c_id in range(1, 6):
                ret, frame = caps[c_id].read()
                if not ret or frame is None:
                    break
                raw_frames[c_id] = frame

            if len(raw_frames) < 5:
                logger.warning("Stream ended early at frame %d", f_idx)
                break

            # Batch predict on all 5 camera frames in a single GPU forward pass
            frames_list = [raw_frames[c_id] for c_id in range(1, 6)]
            results = model.predict(
                source=frames_list,
                conf=conf_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            cam_stats: Dict[int, Optional[Dict[str, Any]]] = {c: None for c in range(1, 6)}
            annotated_tiles: Dict[int, np.ndarray] = {}

            for idx, c_id in enumerate(range(1, 6)):
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

                for det in surviving_dets:
                    bx1 = max(0.0, det.u - det.w / 2.0)
                    by1 = max(0.0, det.v - det.h / 2.0)
                    bx2 = min(float(w_orig - 1), det.u + det.w / 2.0)
                    by2 = min(float(h_orig - 1), det.v + det.h / 2.0)

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

                # Write individual full-res video
                writers[c_id].write(frame)

                # Downscale for dashboard tile
                tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                annotated_tiles[c_id] = tile

            # Build 6th tile (Telemetry Status Pane)
            telemetry_tile = draw_telemetry_tile(tile_w, tile_h, f_idx, total_frames, cam_stats, fps)

            # Assemble 3x2 grid:
            # Row 1: Cam 1 | Cam 2 | Cam 3
            # Row 2: Cam 4 | Cam 5 | Telemetry
            row1 = np.hstack([annotated_tiles[1], annotated_tiles[2], annotated_tiles[3]])
            row2 = np.hstack([annotated_tiles[4], annotated_tiles[5], telemetry_tile])
            dashboard_frame = np.vstack([row1, row2])

            dashboard_writer.write(dashboard_frame)

            if (f_idx + 1) % 100 == 0 or (f_idx + 1) == total_frames:
                elapsed = time.perf_counter() - t_start
                fps_proc = (f_idx + 1) / elapsed
                logger.info("Processed frame %d/%d (%.1f%%) | Speed: %.1f FPS (5 streams parallel)",
                            f_idx + 1, total_frames, (f_idx + 1) / total_frames * 100, fps_proc)

    finally:
        for cap in caps.values():
            cap.release()
        for w in writers.values():
            w.release()
        dashboard_writer.release()

    total_time = time.perf_counter() - t_start
    logger.info("Video processing complete in %.2f s (Average speed: %.1f FPS)", total_frames / total_time * 5 if total_time else 0, total_frames / total_time if total_time else 0)

    # Save CSV with semicolon delimiter compatible with main.py pipeline
    if max_frames is None or not Path(csv_output).exists():
        df = pd.DataFrame(detection_records)
        df.to_csv(csv_output, sep=';', index=False)
        logger.info("Saved %d total detections to: %s", len(df), csv_output)
    else:
        logger.info("Preview run (max_frames=%d): preserved existing full %s", max_frames, csv_output)

    return {
        "total_frames": total_frames,
        "total_time_s": total_time,
        "dashboard_path": str(dashboard_path),
        "individual_videos": {c: str(p) for c, p in out_paths.items()},
        "csv_path": csv_output,
        "total_detections": len(df),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Process multi-camera videos with YOLO26x-seg")
    parser.add_argument("--weights", default="weights/finetuned/peleng_yolo26x_seg_roboflow_best.pt")
    parser.add_argument("--videos-dir", default="data/raw/hackathon")
    parser.add_argument("--output-dir", default="results/videos")
    parser.add_argument("--csv-output", default="results/detections/detections_2d_yolo26_seg.csv")
    parser.add_argument("--trajectory", default="results/trajectories/output_trajectory_ukf.csv", help="3D trajectory CSV for HUD wireframe overlay")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames to process for quick preview")
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
    )
