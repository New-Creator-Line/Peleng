"""
Process Multi-Camera Video Streams with Fine-Tuned YOLOv8x.

Generates:
1. Individual annotated videos for Cameras 1 to 5: results/videos/camera_{i}_yolo.mp4
2. Synchronized 5-Camera Multi-View Dashboard: results/videos/multiview_dashboard_yolo.mp4
3. Unified 2D Detections CSV: results/detections/detections_2d_yolo_real.csv
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from src.yolo_detector import load_detection_model, estimate_occlusion_status

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VideoProcessor")


def draw_detection(
    frame: np.ndarray,
    box: Tuple[float, float, float, float],
    conf: float,
    cam_id: int,
    frame_idx: int,
    label: str = "truck",
    is_occluded: bool = False,
) -> None:
    """Draws styled bounding box and tracking HUD on frame."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

    # Box outline
    color = (0, 140, 255) if is_occluded else (0, 255, 128)  # Amber if occluded, emerald if clear
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Center point (u, v)
    u_center = int((x1 + x2) / 2)
    v_center = int((y1 + y2) / 2)
    cv2.circle(frame, (u_center, v_center), 5, (0, 0, 255), -1)
    cv2.circle(frame, (u_center, v_center), 7, (255, 255, 255), 1)

    # Label badge with confidence
    occ_tag = " [OCCLUDED]" if is_occluded else ""
    caption = f"{label.upper()} {conf:.1%}{occ_tag}"
    (w_text, h_text), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    badge_y1 = max(0, y1 - h_text - 8)
    badge_y2 = badge_y1 + h_text + 8
    badge_x2 = x1 + w_text + 12
    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), (0, 200, 100), -1)
    cv2.putText(
        frame,
        caption,
        (x1 + 6, badge_y2 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    cam_id: int,
    frame_idx: int,
    total_frames: int,
    num_dets: int,
) -> None:
    """Draws top HUD status banner on camera frame."""
    hud_bg = frame[:48, :]
    overlay = hud_bg.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 48), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.75, hud_bg, 0.25, 0, hud_bg)

    hud_text = (
        f"CAMERA {cam_id} | FRAME: {frame_idx:04d}/{total_frames:04d} | "
        f"DETECTIONS: {num_dets} | YOLOv8x EXTRA-LARGE"
    )
    cv2.putText(
        frame,
        hud_text,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def create_info_card(
    frame_idx: int,
    total_frames: int,
    cam_detections: Dict[int, int],
    fps: float,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    """Creates a telemetry dashboard card for the 6th slot in multi-view grid."""
    card = np.full((height, width, 3), 20, dtype=np.uint8)

    # Header
    cv2.rectangle(card, (0, 0), (width, 50), (35, 45, 60), -1)
    cv2.putText(
        card,
        "PELENG 3D VISION DASHBOARD",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 200),
        2,
        cv2.LINE_AA,
    )

    # Telemetry metrics
    y_pos = 90
    spacing = 38
    metrics = [
        ("Model", "YOLOv8x Fine-Tuned (Truck)"),
        ("Sync Frame", f"{frame_idx:04d} / {total_frames:04d}"),
        ("Process FPS", f"{fps:.1f} FPS"),
        ("Cams with Target", f"{sum(1 for v in cam_detections.values() if v > 0)} / 5"),
    ]

    for label, val in metrics:
        cv2.putText(
            card,
            f"{label}:",
            (25, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (160, 170, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            card,
            val,
            (220, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_pos += spacing

    # Camera status indicators
    y_pos += 10
    cv2.putText(
        card,
        "Camera Streams Status:",
        (25, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (130, 140, 150),
        1,
        cv2.LINE_AA,
    )
    y_pos += 30

    x_indicator = 30
    for c_id in range(1, 6):
        has_det = cam_detections.get(c_id, 0) > 0
        dot_color = (0, 255, 128) if has_det else (100, 100, 100)
        cv2.circle(card, (x_indicator + 12, y_pos), 8, dot_color, -1)
        cv2.putText(
            card,
            f"C{c_id}",
            (x_indicator + 26, y_pos + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        x_indicator += 115

    return card


def process_all_videos(
    weights_path: str = "weights/peleng_yolov8x_best.pt",
    videos_dir: str = "хакатон",
    output_dir: str = "results/videos",
    csv_output: str = "results/detections/detections_2d_yolo_real.csv",
    conf_threshold: float = 0.35,
    imgsz: int = 960,
    device: str = "0",
) -> Dict[str, Any]:
    """
    Simultaneously processes all 5 synchronized camera streams:
    - Generates 5 camera output MP4 videos
    - Generates 1 combined Multi-View Grid MP4 video
    - Saves unified 2D detections CSV
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_output).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading YOLOv8 model from: {weights_path}")
    model = load_detection_model(weights_path)

    # Map video files to camera IDs
    cam_videos: Dict[int, str] = {}
    v_dir = Path(videos_dir)
    for c_id in range(1, 6):
        matches = list(v_dir.glob(f"*Actor{c_id}.mp4"))
        if not matches:
            raise FileNotFoundError(f"Video for Camera {c_id} not found in {videos_dir}")
        cam_videos[c_id] = str(matches[0])
        logger.info(f"Camera {c_id} video: {matches[0].name}")

    # Open video capture handles
    caps: Dict[int, cv2.VideoCapture] = {}
    for c_id, path in cam_videos.items():
        caps[c_id] = cv2.VideoCapture(path)

    # Read video metadata from Camera 1
    total_frames = int(caps[1].get(cv2.CAP_PROP_FRAME_COUNT))
    fps = caps[1].get(cv2.CAP_PROP_FPS) or 30.0
    w_orig = int(caps[1].get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(caps[1].get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video parameters: {w_orig}x{h_orig} @ {fps} FPS, Total Frames: {total_frames}")

    # Initialize video writers for individual cameras
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers: Dict[int, cv2.VideoWriter] = {}
    output_video_paths: Dict[int, str] = {}
    for c_id in range(1, 6):
        out_path = Path(output_dir) / f"camera_{c_id}_detected.mp4"
        writers[c_id] = cv2.VideoWriter(str(out_path), fourcc, fps, (w_orig, h_orig))
        output_video_paths[c_id] = str(out_path)

    # Initialize video writer for multi-view dashboard (3 columns x 2 rows, each tile 640x360 -> total 1920x720)
    grid_w, grid_h = 1920, 720
    dashboard_path = Path(output_dir) / "multiview_dashboard_yolo.mp4"
    grid_writer = cv2.VideoWriter(str(dashboard_path), fourcc, fps, (grid_w, grid_h))

    detection_records: List[Dict[str, Any]] = []

    t_start = time.perf_counter()
    logger.info(f"Processing {total_frames} frames across 5 camera streams...")

    try:
        for frame_idx in range(total_frames):
            raw_frames: Dict[int, np.ndarray] = {}
            for c_id in range(1, 6):
                ret, frame = caps[c_id].read()
                if not ret or frame is None:
                    break
                raw_frames[c_id] = frame

            if len(raw_frames) < 5:
                logger.warning(f"Stream ended early at frame {frame_idx}")
                break

            # Batch predict on all 5 frames for GPU efficiency
            frames_list = [raw_frames[c_id] for c_id in range(1, 6)]
            results = model.predict(
                source=frames_list,
                conf=conf_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            cam_det_counts: Dict[int, int] = {}
            grid_tiles: Dict[int, np.ndarray] = {}

            for idx, c_id in enumerate(range(1, 6)):
                frame = raw_frames[c_id]
                res = results[idx]
                boxes = res.boxes

                n_dets = 0
                if boxes is not None and len(boxes) > 0:
                    for b_idx in range(len(boxes)):
                        cls_id = int(boxes.cls[b_idx].item())
                        if cls_id != 0:
                            continue

                        conf = float(boxes.conf[b_idx].item())
                        xyxy = boxes.xyxy[b_idx].cpu().numpy().tolist()
                        xywh = boxes.xywh[b_idx].cpu().numpy().tolist()

                        u, v, w, h = xywh
                        is_occ, occ_score = estimate_occlusion_status(float(u), float(v), float(w), float(h), cam_id=c_id)
                        draw_detection(frame, tuple(xyxy), conf, c_id, frame_idx, is_occluded=is_occ)

                        detection_records.append({
                            "Frame": frame_idx,
                            "Cam_ID": c_id,
                            "u": round(float(u), 2),
                            "v": round(float(v), 2),
                            "w": round(float(w), 2),
                            "h": round(float(h), 2),
                            "conf": round(float(conf), 4),
                            "is_occluded": 1 if is_occ else 0,
                            "occlusion_score": round(occ_score, 3),
                        })
                        n_dets += 1

                cam_det_counts[c_id] = n_dets
                draw_hud(frame, c_id, frame_idx, total_frames, n_dets)

                # Write individual full-HD camera frame
                writers[c_id].write(frame)

                # Downsample tile for dashboard grid (640x360)
                grid_tiles[c_id] = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)

            # Build 2x3 dashboard grid
            # Row 1: Cam 1, Cam 2, Cam 3
            # Row 2: Cam 4, Cam 5, Telemetry Card
            elapsed = max(0.001, time.perf_counter() - t_start)
            current_fps = (frame_idx + 1) * 5 / elapsed

            info_tile = create_info_card(frame_idx, total_frames, cam_det_counts, current_fps, 640, 360)
            row_top = np.hstack([grid_tiles[1], grid_tiles[2], grid_tiles[3]])
            row_bot = np.hstack([grid_tiles[4], grid_tiles[5], info_tile])
            dashboard_frame = np.vstack([row_top, row_bot])

            grid_writer.write(dashboard_frame)

            if (frame_idx + 1) % 100 == 0 or frame_idx == total_frames - 1:
                logger.info(
                    f"Processed {frame_idx + 1}/{total_frames} frames "
                    f"({(frame_idx + 1) / total_frames:.1%}) | "
                    f"Overall: {current_fps:.1f} FPS | "
                    f"Total Detections: {len(detection_records)}"
                )

    finally:
        for c_id, cap in caps.items():
            cap.release()
        for c_id, writer in writers.items():
            writer.release()
        grid_writer.release()

    total_time = time.perf_counter() - t_start
    logger.info(f"Video processing complete in {total_time:.2f}s ({total_frames * 5 / total_time:.1f} FPS)!")

    # Save 2D detections CSV
    df = pd.DataFrame(detection_records)
    df.to_csv(csv_output, sep=";", index=False)
    logger.info(f"Saved {len(df)} 2D detections to: {csv_output}")

    return {
        "individual_videos": output_video_paths,
        "dashboard_video": str(dashboard_path),
        "detections_csv": csv_output,
        "total_detections": len(df),
        "total_frames_processed": total_frames,
        "processing_time_sec": round(total_time, 2),
    }


def parse_args() -> argparse.Namespace:
    def resolve_path(preferred: str, fallback: str) -> str:
        if os.path.exists(preferred):
            return preferred
        if os.path.exists(fallback):
            return fallback
        return preferred

    default_weights = resolve_path("weights/finetuned/peleng_yolov8x_best.pt", "weights/peleng_yolov8x_best.pt")
    default_videos = resolve_path("data/raw/hackathon", "хакатон")

    parser = argparse.ArgumentParser(description="Process multi-camera videos with YOLOv8x")
    parser.add_argument("--weights", type=str, default=default_weights)
    parser.add_argument("--videos_dir", type=str, default=default_videos)
    parser.add_argument("--output_dir", type=str, default="results/videos")
    parser.add_argument("--csv", type=str, default="results/detections/detections_2d_yolo_real.csv")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("  PELENG MULTI-CAMERA VIDEO GENERATOR (YOLOv8x)")
    print("=" * 60)
    res = process_all_videos(
        weights_path=args.weights,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        csv_output=args.csv,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    print("\n--- Generation Summary ---")
    print(f"Cameras 1-5 videos saved to: {args.output_dir}/")
    print(f"Multi-view Dashboard: {res['dashboard_video']}")
    print(f"Detections CSV: {res['detections_csv']} ({res['total_detections']} detections)")
