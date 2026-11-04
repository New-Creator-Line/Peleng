"""
Process Multi-Camera Video Streams with Fine-Tuned YOLO / RT-DETR.

Supports an arbitrary number of cameras N discovered automatically from *Actor{id}.mp4:
1. Individual annotated videos for each camera: results/videos/camera_{id}_detected.mp4
2. Synchronized Multi-View Dashboard: results/videos/multiview_dashboard_yolo.mp4
3. Unified 2D Detections CSV: results/detections/detections_2d_yolo_real.csv
"""

import argparse
import logging
import math
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
from src.camera_model import CameraTimeline, load_all_cameras
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
    proj_3d_uv: Optional[Tuple[float, float]] = None,
) -> None:
    """Draws styled bounding box, cabin 3D projection crosshair, and tracking HUD on frame."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

    if label == "cabin":
        color = (0, 230, 255)  # Vibrant cyan/gold for cabin
    else:
        color = (0, 165, 255) if is_occluded else (0, 255, 128)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    u_center = int((x1 + x2) / 2)
    v_center = int((y1 + y2) / 2)
    cv2.circle(frame, (u_center, v_center), 5, (0, 0, 255), -1)
    cv2.circle(frame, (u_center, v_center), 7, (255, 255, 255), 1)

    if proj_3d_uv is not None:
        up, vp = int(round(proj_3d_uv[0])), int(round(proj_3d_uv[1]))
        # Draw 3D cabin projection only when it falls within the visible truck box region
        if max(0, x1 - 25) <= up <= min(w_img - 1, x2 + 25) and max(0, y1 - 25) <= vp <= min(h_img - 1, y2 + 25):
            cv2.line(frame, (u_center, v_center), (up, vp), (255, 220, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (up, vp), 6, (255, 220, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (up, vp), 8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (up - 12, vp), (up + 12, vp), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (up, vp - 12), (up, vp + 12), (255, 255, 255), 1, cv2.LINE_AA)

    occ_tag = " [EDGE/OCC]" if is_occluded else ""
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
        f"DETECTIONS: {num_dets} | PELENG 3D VISION"
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
    pos_3d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Creates a telemetry dashboard card for the multi-view grid."""
    card = np.full((height, width, 3), 20, dtype=np.uint8)
    cam_ids = sorted(cam_detections.keys())
    num_cams = len(cam_ids)

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

    y_pos = 84
    spacing = 34
    xyz_str = f"X={pos_3d[0]:.0f} Y={pos_3d[1]:.0f} Z={pos_3d[2]:.0f}" if pos_3d is not None else "Computing..."
    metrics = [
        ("Model", "Peleng Fine-Tuned Detector"),
        ("Sync Frame", f"{frame_idx:04d} / {total_frames:04d}"),
        ("Process FPS", f"{fps:.1f} FPS"),
        ("Cams with Target", f"{sum(1 for v in cam_detections.values() if v > 0)} / {num_cams}"),
        ("3D Cabin (UE)", xyz_str),
    ]

    for label, val in metrics:
        cv2.putText(card, f"{label}:", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 170, 180), 1, cv2.LINE_AA)
        cv2.putText(card, val, (220, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y_pos += spacing

    y_pos += 10
    cv2.putText(card, "Camera Streams Status:", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 140, 150), 1, cv2.LINE_AA)
    y_pos += 30

    x_indicator = 30
    step_x = max(60, min(115, (width - 60) // max(1, num_cams)))
    for c_id in cam_ids:
        has_det = cam_detections.get(c_id, 0) > 0
        dot_color = (0, 255, 128) if has_det else (100, 100, 100)
        cv2.circle(card, (x_indicator + 12, y_pos), 8, dot_color, -1)
        cv2.putText(card, f"C{c_id}", (x_indicator + 26, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        x_indicator += step_x

    return card


def discover_camera_videos(videos_dir: str) -> Dict[int, str]:
    """Discovers all camera MP4 streams (*Actor{id}.mp4) in videos_dir for arbitrary N cameras."""
    v_dir = Path(videos_dir)
    cam_videos: Dict[int, str] = {}
    for mp4_path in sorted(v_dir.glob("*Actor*.mp4")):
        m = re.search(r"Actor(\d+)\.mp4$", mp4_path.name, re.IGNORECASE)
        if m:
            cam_videos[int(m.group(1))] = str(mp4_path)
    if not cam_videos:
        raise FileNotFoundError(f"No camera videos (*Actor*.mp4) found in {videos_dir}")
    return dict(sorted(cam_videos.items()))


def process_all_videos(
    weights_path: str = "weights/peleng_yolov8x_best.pt",
    videos_dir: str = "хакатон",
    output_dir: str = "results/videos",
    csv_output: str = "results/detections/detections_2d_yolo_real.csv",
    conf_threshold: float = 0.35,
    imgsz: int = 960,
    device: str = "0",
    detections_csv: Optional[str] = None,
    trajectory_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simultaneously processes all synchronized camera streams (arbitrary N cameras):
    - Generates N camera output MP4 videos
    - Generates 1 combined Multi-View Grid MP4 video
    - Saves unified 2D detections CSV
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_output).parent.mkdir(parents=True, exist_ok=True)

    model = None
    precomputed_map: Optional[Dict[Tuple[int, int], List[Dict[str, Any]]]] = None
    if detections_csv and os.path.exists(detections_csv):
        logger.info(f"Using precomputed 2D detections from: {detections_csv}")
        sep = ';' if ';' in open(detections_csv, 'r', encoding='utf-8').readline() else ','
        df_dets = pd.read_csv(detections_csv, sep=sep)
        col_m = {c.lower(): c for c in df_dets.columns}
        f_col = col_m.get('frame', 'Frame')
        cam_col = col_m.get('cam_id', col_m.get('camera', 'Cam_ID'))
        u_col = col_m.get('u', col_m.get('x_center', 'u'))
        v_col = col_m.get('v', col_m.get('y_center', 'v'))
        w_col = col_m.get('w', col_m.get('width', 'w'))
        h_col = col_m.get('h', col_m.get('height', 'h'))
        cf_col = col_m.get('conf', col_m.get('confidence', 'conf'))
        tr_col = col_m.get('is_truncated', col_m.get('is_occluded', None))
        precomputed_map = {}
        for _, row in df_dets.iterrows():
            f_i = int(row[f_col])
            c_i = int(row[cam_col])
            u_v, v_v = float(row[u_col]), float(row[v_col])
            w_v, h_v = float(row[w_col]), float(row[h_col])
            cf_v = float(row[cf_col]) if cf_col in row else 0.90
            tr_v = bool(row[tr_col]) if tr_col and tr_col in row else False
            precomputed_map.setdefault((f_i, c_i), []).append({
                'u': u_v, 'v': v_v, 'w': w_v, 'h': h_v, 'conf': cf_v, 'is_occluded': tr_v
            })
    else:
        logger.info(f"Loading detection model from: {weights_path}")
        model = load_detection_model(weights_path)

    cam_videos = discover_camera_videos(videos_dir)
    cam_ids = sorted(cam_videos.keys())
    num_cams = len(cam_ids)
    for c_id, path in cam_videos.items():
        logger.info(f"Camera {c_id} video: {Path(path).name}")

    # Load camera calibrations and precomputed 3D trajectory (if available) to project 3D cabin center
    cameras: Dict[int, CameraTimeline] = {}
    try:
        cameras = load_all_cameras(videos_dir)
    except Exception:
        pass
    traj_3d_map: Dict[int, np.ndarray] = {}
    resolved_traj = trajectory_csv or ("results/trajectories/output_trajectory_yolo26_seg.csv" if os.path.exists("results/trajectories/output_trajectory_yolo26_seg.csv") else "results/trajectories/output_trajectory_yolo.csv")
    traj_csv = Path(resolved_traj)
    if traj_csv.exists():
        try:
            t_df = pd.read_csv(traj_csv, sep=";")
            for _, r in t_df.iterrows():
                traj_3d_map[int(r["Frame"])] = np.array([float(r["Loc_X"]), float(r["Loc_Y"]), float(r["Loc_Z"])])
        except Exception:
            pass

    caps: Dict[int, cv2.VideoCapture] = {c_id: cv2.VideoCapture(path) for c_id, path in cam_videos.items()}

    first_cap = caps[cam_ids[0]]
    total_frames = int(first_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_orig = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video parameters: {w_orig}x{h_orig} @ {fps} FPS, Total Frames: {total_frames}, Cameras: {num_cams}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers: Dict[int, cv2.VideoWriter] = {}
    output_video_paths: Dict[int, str] = {}
    for c_id in cam_ids:
        out_path = Path(output_dir) / f"camera_{c_id}_detected.mp4"
        writers[c_id] = cv2.VideoWriter(str(out_path), fourcc, fps, (w_orig, h_orig))
        output_video_paths[c_id] = str(out_path)

    tile_w, tile_h = 640, 360
    total_slots = num_cams + 1
    grid_cols = min(3, total_slots)
    grid_rows = int(math.ceil(total_slots / grid_cols))
    grid_w, grid_h = grid_cols * tile_w, grid_rows * tile_h
    dashboard_path = Path(output_dir) / "multiview_dashboard_yolo.mp4"
    grid_writer = cv2.VideoWriter(str(dashboard_path), fourcc, fps, (grid_w, grid_h))

    detection_records: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    logger.info(f"Processing {total_frames} frames across {num_cams} camera streams...")

    try:
        for frame_idx in range(total_frames):
            raw_frames: Dict[int, np.ndarray] = {}
            for c_id in cam_ids:
                ret, frame = caps[c_id].read()
                if not ret or frame is None:
                    break
                raw_frames[c_id] = frame

            if len(raw_frames) < num_cams:
                logger.warning(f"Stream ended early at frame {frame_idx}")
                break

            cam_det_counts: Dict[int, int] = {}
            grid_tiles: List[np.ndarray] = []

            if precomputed_map is not None:
                for idx, c_id in enumerate(cam_ids):
                    frame = raw_frames[c_id]
                    h_f, w_f = frame.shape[:2]
                    dets_for_cam = precomputed_map.get((frame_idx, c_id), [])
                    n_dets = 0
                    for d in dets_for_cam:
                        u, v, w, h = d['u'], d['v'], d['w'], d['h']
                        conf = d['conf']
                        vis_occ = bool(d.get('is_occluded', False))
                        xyxy = [u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0]
                        proj_uv = None
                        if frame_idx in traj_3d_map and c_id in cameras:
                            up, vp, zc = cameras[c_id].get_model(frame_idx).project_point(traj_3d_map[frame_idx])
                            if zc > 0:
                                proj_uv = (up, vp)
                        label = d.get('label') or ("cabin" if "cabin" in (detections_csv or "").lower() else "truck")
                        draw_detection(frame, tuple(xyxy), conf, c_id, frame_idx, label=label, is_occluded=vis_occ, proj_3d_uv=proj_uv)
                        detection_records.append({
                            "Frame": frame_idx, "Cam_ID": c_id,
                            "u": round(float(u), 2), "v": round(float(v), 2),
                            "w": round(float(w), 2), "h": round(float(h), 2),
                            "conf": round(float(conf), 4), "is_occluded": 1 if vis_occ else 0,
                            "occlusion_score": 1.0 if vis_occ else 0.0,
                        })
                        n_dets += 1
                    cam_det_counts[c_id] = n_dets
                    draw_hud(frame, c_id, frame_idx, total_frames, n_dets)
                    writers[c_id].write(frame)
                    grid_tiles.append(cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA))
            else:
                frames_list = [raw_frames[c_id] for c_id in cam_ids]
                results = model.predict(
                    source=frames_list,
                    conf=conf_threshold,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                )
                for idx, c_id in enumerate(cam_ids):
                    frame = raw_frames[c_id]
                    h_f, w_f = frame.shape[:2]
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
                            is_occ, occ_score = estimate_occlusion_status(
                                float(u), float(v), float(w), float(h), cam_id=c_id, img_w=w_f, img_h=h_f
                            )
                            vis_occ = (
                                xyxy[0] <= 15.0
                                or xyxy[2] >= w_f - 15.0
                                or xyxy[1] <= 15.0
                                or xyxy[3] >= h_f - 15.0
                                or (w / max(1.0, h) < 0.85)
                            )
                            proj_uv = None
                            if frame_idx in traj_3d_map and c_id in cameras:
                                up, vp, zc = cameras[c_id].get_model(frame_idx).project_point(traj_3d_map[frame_idx])
                                if zc > 0:
                                    proj_uv = (up, vp)
                            label = "cabin" if "cabin" in weights_path.lower() else "truck"
                            draw_detection(frame, tuple(xyxy), conf, c_id, frame_idx, label=label, is_occluded=vis_occ, proj_3d_uv=proj_uv)

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
                    writers[c_id].write(frame)
                    grid_tiles.append(cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA))



            elapsed = max(0.001, time.perf_counter() - t_start)
            current_fps = (frame_idx + 1) * num_cams / elapsed

            info_tile = create_info_card(
                frame_idx, total_frames, cam_det_counts, current_fps, tile_w, tile_h, pos_3d=traj_3d_map.get(frame_idx)
            )
            grid_tiles.append(info_tile)
            while len(grid_tiles) < grid_rows * grid_cols:
                grid_tiles.append(np.full((tile_h, tile_w, 3), 15, dtype=np.uint8))

            rows_img = [
                np.hstack(grid_tiles[r * grid_cols : (r + 1) * grid_cols])
                for r in range(grid_rows)
            ]
            dashboard_frame = np.vstack(rows_img)
            grid_writer.write(dashboard_frame)

            if (frame_idx + 1) % 100 == 0 or frame_idx == total_frames - 1:
                logger.info(
                    f"Processed {frame_idx + 1}/{total_frames} frames "
                    f"({(frame_idx + 1) / total_frames:.1%}) | "
                    f"Overall: {current_fps:.1f} FPS | "
                    f"Total Detections: {len(detection_records)}"
                )

    finally:
        for cap in caps.values():
            cap.release()
        for writer in writers.values():
            writer.release()
        grid_writer.release()

    total_time = time.perf_counter() - t_start
    logger.info(f"Video processing complete in {total_time:.2f}s ({total_frames * num_cams / total_time:.1f} FPS)!")

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

    default_weights = resolve_path(
        "runs/detect/train-3/weights/best.pt",
        resolve_path("weights/finetuned/peleng_yolov8x_best.pt", "weights/peleng_yolov8x_best.pt"),
    )
    default_videos = resolve_path("хакатон", "data/raw/hackathon")

    parser = argparse.ArgumentParser(description="Process multi-camera videos with YOLO/RT-DETR")
    parser.add_argument("--weights", type=str, default=default_weights)
    parser.add_argument("--videos_dir", type=str, default=default_videos)
    parser.add_argument("--output_dir", type=str, default="results/videos")
    parser.add_argument("--csv", type=str, default="results/detections/detections_2d_yolo_real.csv")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detections-csv", type=str, default=None, help="Precomputed detections CSV to render directly")
    parser.add_argument("--trajectory-csv", type=str, default=None, help="Precomputed 3D trajectory CSV to project onto videos")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("  PELENG MULTI-CAMERA VIDEO GENERATOR")
    print("=" * 60)
    res = process_all_videos(
        weights_path=args.weights,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        csv_output=args.csv,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        detections_csv=args.detections_csv,
        trajectory_csv=args.trajectory_csv,
    )
    print("\n--- Generation Summary ---")
    print(f"Camera videos saved to: {args.output_dir}/")
    print(f"Multi-view Dashboard: {res['dashboard_video']}")
    print(f"Detections CSV: {res['detections_csv']} ({res['total_detections']} detections)")
