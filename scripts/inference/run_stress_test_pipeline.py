"""
Stress-Test Pipeline Runner for Peleng 3D Vision.
Processes degraded / corrupted video streams from stress-test folders:
1. Runs YOLOv8x-seg (YOLO26-seg) instance segmentation across all 5 camera streams.
2. Generates 2D detection CSV with occlusion tagging.
3. Reconstructs 3D trajectory using C++20 core and Kalman filter.
4. Evaluates accuracy against ground truth (MAE / RMSE).
5. Renders individual camera feeds + synchronized multi-view dashboard with telemetry.
6. Saves all artifacts into the target result folder.
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from src.camera_model import CameraTimeline, load_all_cameras
from src.evaluator import TrajectoryEvaluator, load_ground_truth
from src.yolo_detector import estimate_occlusion_status

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("StressTestPipeline")


def discover_camera_videos(videos_dir: Path) -> Dict[int, Path]:
    """Finds all 5 camera video streams matching Cam{id} or Actor{id}."""
    cam_videos: Dict[int, Path] = {}
    for p in sorted(videos_dir.glob("*.mp4")):
        m = re.search(r"(?:Actor|Cam|Camera)[^\d]*(\d+)", p.name, re.IGNORECASE)
        if m:
            cam_videos[int(m.group(1))] = p
    if len(cam_videos) < 5:
        logger.warning("Found %d camera videos in %s: %s", len(cam_videos), videos_dir, cam_videos)
    return dict(sorted(cam_videos.items()))


def draw_hud(frame: np.ndarray, cam_id: int, frame_idx: int, total_frames: int, n_dets: int):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (450, 80), (20, 25, 35), -1)
    cv2.rectangle(frame, (10, 10), (450, 80), (60, 75, 95), 1)
    cv2.putText(frame, f"PELENG 3D [YOLO26-SEG] | CAM {cam_id}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
    cv2.putText(frame, f"Frame: {frame_idx:04d}/{total_frames:04d} | Dets: {n_dets}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)


def make_dashboard_tile(frame: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    return cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)


def run_stress_test(
    video_dir: str,
    calib_dir: str = "хакатон",
    result_dir: Optional[str] = None,
    weights_path: str = "yolov8x-seg.pt",
    conf_threshold: float = 0.15,
    imgsz: int = 960,
    device: str = "0" if torch.cuda.is_available() else "cpu",
):
    v_path = Path(video_dir).resolve()
    if not v_path.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    if result_dir is None:
        out_dir = v_path / "result"
    else:
        out_dir = Path(result_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("   PELENG STRESS-TEST PIPELINE (YOLO26-SEG)")
    logger.info("   Video Source: %s", v_path)
    logger.info("   Output Dir:   %s", out_dir)
    logger.info("=" * 65)

    # 1. Discover videos
    cam_videos = discover_camera_videos(v_path)
    if not cam_videos:
        raise FileNotFoundError(f"No MP4 videos found in {v_path}")
    cam_ids = sorted(cam_videos.keys())
    logger.info("Discovered %d camera streams: %s", len(cam_ids), cam_videos)

    # 2. Check video metadata
    test_cap = cv2.VideoCapture(str(cam_videos[cam_ids[0]]))
    total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = test_cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_orig = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    test_cap.release()
    logger.info("Video format: %dx%d @ %.1f FPS | Total Frames: %d", w_orig, h_orig, fps, total_frames)

    # 3. Load YOLO26-seg model
    logger.info("Loading YOLO26-seg model from: %s on %s...", weights_path, device)
    model = YOLO(weights_path)
    truck_classes = [k for k, name in model.names.items() if str(name).lower() in ("truck", "car", "vehicle", "bus")]
    if not truck_classes:
        truck_classes = [0]
    logger.info("Target classes: %s (names: %s)", truck_classes, [model.names[c] for c in truck_classes])

    # 4. PASS 1: Run Batch Detection across all frames
    csv_detections = out_dir / "detections_2d.csv"
    detection_records: List[Dict[str, Any]] = []

    caps = {c_id: cv2.VideoCapture(str(cam_videos[c_id])) for c_id in cam_ids}
    logger.info("[1/4] Running YOLO26-seg Inference on degraded streams...")
    t_det_start = time.perf_counter()

    for f_idx in range(total_frames):
        frames = {}
        for c_id in cam_ids:
            ret, frame = caps[c_id].read()
            if not ret or frame is None:
                break
            frames[c_id] = frame

        if len(frames) < len(cam_ids):
            break

        # Batch forward pass on all cameras
        frames_list = [frames[c_id] for c_id in cam_ids]
        results = model.predict(
            source=frames_list,
            conf=conf_threshold,
            classes=truck_classes,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        for idx, c_id in enumerate(cam_ids):
            res = results[idx]
            boxes = res.boxes

            if boxes is not None and len(boxes) > 0:
                for b_idx in range(len(boxes)):
                    conf = float(boxes.conf[b_idx].item())
                    xyxy = boxes.xyxy[b_idx].cpu().numpy().tolist()
                    xywh = boxes.xywh[b_idx].cpu().numpy().tolist()
                    u, v, w, h = xywh
                    is_occ, occ_score = estimate_occlusion_status(u, v, w, h, cam_id=c_id, img_w=w_orig, img_h=h_orig)
                    vis_occ = (
                        xyxy[0] <= 15.0 or xyxy[2] >= w_orig - 15.0 or
                        xyxy[1] <= 15.0 or xyxy[3] >= h_orig - 15.0 or
                        (w / max(1.0, h) < 0.85)
                    )
                    detection_records.append({
                        "Frame": f_idx,
                        "Cam_ID": c_id,
                        "u": round(float(u), 2),
                        "v": round(float(v), 2),
                        "w": round(float(w), 2),
                        "h": round(float(h), 2),
                        "conf": round(float(conf), 4),
                        "is_occluded": 1 if (is_occ or vis_occ) else 0,
                        "occlusion_score": round(float(occ_score), 3),
                    })

        if (f_idx + 1) % 100 == 0 or (f_idx + 1) == total_frames:
            logger.info("Inference: %d/%d frames (%d total detections)", f_idx + 1, total_frames, len(detection_records))

    for cap in caps.values():
        cap.release()

    det_elapsed = time.perf_counter() - t_det_start
    logger.info("Inference completed in %.2fs (%.1f FPS)", det_elapsed, (total_frames * len(cam_ids)) / max(1e-5, det_elapsed))

    # Save 2D detections CSV
    df_dets = pd.DataFrame(detection_records)
    df_dets.to_csv(csv_detections, sep=";", index=False)
    logger.info("Saved 2D detections to: %s", csv_detections)

    # 5. PASS 2: Run 3D Reconstruction Pipeline via main.py C++ engine
    logger.info("[2/4] Running 3D Reconstruction Pipeline with C++ Core...")
    import main

    out_traj_csv = out_dir / "output_trajectory.csv"
    out_metrics_json = out_dir / "metrics_summary.json"
    out_plot_png = out_dir / "trajectory_comparison.png"

    # Resolve GT path
    gt_candidates = [
        Path(calib_dir) / "object_transform_SM_AA_track.csv",
        Path("хакатон") / "object_transform_SM_AA_track.csv",
        Path("data/raw/hackathon") / "object_transform_SM_AA_track.csv",
    ]
    gt_path = next((str(p) for p in gt_candidates if p.exists()), None)

    traj_df = main.run_pipeline(
        calib_dir=calib_dir,
        detections_csv=str(csv_detections),
        gt_csv=gt_path,
        output_csv=str(out_traj_csv),
        metrics_json=str(out_metrics_json),
        plot_path=str(out_plot_png),
        plot=True,
        conf_threshold=conf_threshold,
        tracker_type="linear",
    )
    logger.info("3D Trajectory saved to: %s", out_traj_csv)

    # 6. PASS 3: Render Video Outputs (Streaming memory-safe)
    logger.info("[3/4] Rendering annotated multi-camera videos and dashboard...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {
        c_id: cv2.VideoWriter(str(out_dir / f"camera_{c_id}_detected.mp4"), fourcc, fps, (w_orig, h_orig))
        for c_id in cam_ids
    }
    tile_w, tile_h = 640, 360
    dashboard_writer = cv2.VideoWriter(str(out_dir / "multiview_dashboard.mp4"), fourcc, fps, (1920, 720))

    # Reopen capture streams for video generation
    caps = {c_id: cv2.VideoCapture(str(cam_videos[c_id])) for c_id in cam_ids}

    # Group detections by frame and camera for O(1) lookup
    dets_lookup = {}
    for d in detection_records:
        dets_lookup.setdefault((d["Frame"], d["Cam_ID"]), []).append(d)

    # Load 3D trajectory dictionary
    traj_dict = {int(r["Frame"]): (float(r["Loc_X"]), float(r["Loc_Y"]), float(r["Loc_Z"])) for _, r in traj_df.iterrows()}
    cameras = load_all_cameras(calib_dir)

    for f_idx in range(total_frames):
        grid_tiles = []
        for c_id in cam_ids:
            ret, frame = caps[c_id].read()
            if not ret or frame is None:
                frame = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)

            cam_frame_dets = dets_lookup.get((f_idx, c_id), [])

            # Draw detections
            for d in cam_frame_dets:
                u, v, w, h = d["u"], d["v"], d["w"], d["h"]
                x1, y1 = int(u - w / 2.0), int(v - h / 2.0)
                x2, y2 = int(u + w / 2.0), int(v + h / 2.0)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_orig - 1, x2), min(h_orig - 1, y2)
                is_occ = bool(d["is_occluded"])
                col = (0, 165, 255) if is_occ else (0, 255, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)

                # Label
                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + 160, y1), col, -1)
                lbl = f"yolo26-seg {d['conf']:.2f}"
                cv2.putText(frame, lbl, (x1 + 5, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # Draw projected 3D trajectory position on frame
            if f_idx in traj_dict and c_id in cameras:
                cam_m = cameras[c_id].get_model(f_idx)
                res = cam_m.project_point(np.array(traj_dict[f_idx]))
                if res is not None:
                    u_p, v_p, z_c = res
                    if z_c > 0 and 0 <= u_p < w_orig and 0 <= v_p < h_orig:
                        cu, cv_pt = int(u_p), int(v_p)
                        cv2.circle(frame, (cu, cv_pt), 8, (0, 240, 255), -1, lineType=cv2.LINE_AA)
                        cv2.circle(frame, (cu, cv_pt), 14, (0, 240, 255), 2, lineType=cv2.LINE_AA)
                        cv2.putText(frame, "3D Cabin", (cu + 12, cv_pt + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 2)

            draw_hud(frame, c_id, f_idx, total_frames, len(cam_frame_dets))
            writers[c_id].write(frame)
            grid_tiles.append(make_dashboard_tile(frame, tile_w, tile_h))

        # Dashboard layout: 3 cols x 2 rows (Tile 6 is telemetry card)
        telemetry_tile = np.full((tile_h, tile_w, 3), (25, 30, 42), dtype=np.uint8)
        cv2.putText(telemetry_tile, "PELENG TELEMETRY DASHBOARD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)
        cv2.putText(telemetry_tile, f"Degradation: 06_HEAVY_BLUR", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(telemetry_tile, f"Model: YOLO26-seg (Segmentation)", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 1)
        cv2.putText(telemetry_tile, f"Frame: {f_idx:04d} / {total_frames:04d}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        if f_idx in traj_dict:
            tx, ty, tz = traj_dict[f_idx]
            cv2.putText(telemetry_tile, f"3D Pos X: {tx:10.1f} cm", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 200, 100), 1)
            cv2.putText(telemetry_tile, f"3D Pos Y: {ty:10.1f} cm", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 200, 100), 1)
            cv2.putText(telemetry_tile, f"3D Pos Z: {tz:10.1f} cm", (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 200, 100), 1)
        cv2.putText(telemetry_tile, "Core: Native C++20 Jacobi SVD", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 150), 1)
        grid_tiles.append(telemetry_tile)

        # Assemble 2 rows of 3 columns
        row1 = np.hstack([grid_tiles[0], grid_tiles[1], grid_tiles[2]])
        row2 = np.hstack([grid_tiles[3], grid_tiles[4], grid_tiles[5]])
        dashboard_frame = np.vstack([row1, row2])
        dashboard_writer.write(dashboard_frame)

        if (f_idx + 1) % 150 == 0 or (f_idx + 1) == total_frames:
            logger.info("Video rendering: %d/%d frames", f_idx + 1, total_frames)

    for w in writers.values():
        w.release()
    dashboard_writer.release()
    for cap in caps.values():
        cap.release()

    logger.info("[4/4] Pipeline completed successfully!")
    logger.info("All deliverables saved in: %s", out_dir)


def main_cli():
    parser = argparse.ArgumentParser(description="Run Peleng Stress-Test Pipeline")
    parser.add_argument("--video-dir", required=True, help="Path to stress test video folder")
    parser.add_argument("--calib-dir", default="хакатон", help="Path to camera calibration folder")
    parser.add_argument("--result-dir", default=None, help="Output directory (defaults to <video-dir>/result)")
    parser.add_argument("--weights", default="yolov8x-seg.pt", help="Path to YOLO segmentation model weights")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image resolution")
    args = parser.parse_args()

    run_stress_test(
        video_dir=args.video_dir,
        calib_dir=args.calib_dir,
        result_dir=args.result_dir,
        weights_path=args.weights,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main_cli()
