"""
Process Multi-Camera Video Streams for Scenario 2 (9 Cameras)
using the user's fine-tuned models:
1. Truck Instance Segmentation: weights/finetuned/peleng_yolo26x_seg_clean_polish_best*.engine
2. Cabin Detector: weights/cabin/peleng_cabin_best.pt

GPU-first pipeline (CPU only draws overlays):
- NVDEC: each camera decoded by its own ffmpeg h264_cuvid process (hardware decoder).
- TensorRT / CUDA inference, batched across cameras.
- NVENC: frames piped straight into h264_nvenc (no CPU mp4v encode, no transcode pass).
- Decode, inference and render/encode run concurrently (render thread + decode prefetch threads).
- Only 2D boxes and center points / trail dots (NO 3D boxes).
- Outputs: 3x3 dashboard (1920x1080) + optional individual camera videos + CSV detections.
"""

import argparse
import collections
import logging
import os
import queue
import sys
import threading
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from src.detector import compute_mask_moments, suppress_nested_detections, DetectionResult
from src.ffmpeg_pipe import FFmpegNvdecReader, FFmpegNvencWriter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Scen2Processor")

N_CAMS = 9
TILE_W, TILE_H = 640, 360  # 3x3 grid -> 1920x1080


def draw_styled_box(
    frame: np.ndarray,
    box: Tuple[float, float, float, float],
    conf: float,
    u_c: float,
    v_c: float,
    label: str,
    color: Tuple[int, int, int],
    point_color: Tuple[int, int, int],
    is_occluded: bool = False,
) -> None:
    """Draws 2D bounding box with corner accents, center point, and badge (NO 3D BOX)."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

    corner_len = min(24, max(6, (x2 - x1) // 5))
    accent_color = (255, 255, 255)
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (px, py), (px + dx * corner_len, py), accent_color, 3)
        cv2.line(frame, (px, py), (px, py + dy * corner_len), accent_color, 3)

    u_int, v_int = int(round(u_c)), int(round(v_c))
    if 0 <= u_int < w_img and 0 <= v_int < h_img:
        cv2.circle(frame, (u_int, v_int), 6, point_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (u_int, v_int), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(frame, (u_int - 12, v_int), (u_int + 12, v_int), (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(frame, (u_int, v_int - 12), (u_int, v_int + 12), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    occ_str = " [EDGE]" if is_occluded else ""
    badge_text = f"{label.upper()} {conf:.1%}{occ_str}"
    (tw, th), bl = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    by1 = max(0, y1 - th - 8)
    by2 = by1 + th + 8
    bx2 = min(w_img - 1, x1 + tw + 12)
    cv2.rectangle(frame, (x1, by1), (bx2, by2), (20, 24, 30), -1)
    cv2.rectangle(frame, (x1, by1), (bx2, by2), color, 1)
    cv2.putText(frame, badge_text, (x1 + 6, by2 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, lineType=cv2.LINE_AA)


def draw_hud_header(
    frame: np.ndarray,
    cam_id: int,
    frame_idx: int,
    total_frames: int,
    has_truck: bool,
    has_cabin: bool,
) -> None:
    """Draws telemetry header banner at the top of each camera frame."""
    h_bar = 42
    overlay = frame[:h_bar, :].copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h_bar), (16, 20, 28), -1)
    cv2.addWeighted(overlay, 0.85, frame[:h_bar, :], 0.15, 0, frame[:h_bar, :])

    if has_cabin:
        status_str, status_col = "TRACKING (CABIN PIVOT)", (255, 200, 0)
    elif has_truck:
        status_str, status_col = "TRACKING (TRUCK BODY)", (0, 255, 128)
    else:
        status_str, status_col = "SEARCHING", (130, 135, 145)

    cv2.putText(frame, f"CAM {cam_id}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (120, 22), 6, status_col, -1, lineType=cv2.LINE_AA)
    cv2.putText(frame, status_str, (134, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_col, 1, lineType=cv2.LINE_AA)

    info_str = f"FRAME: {frame_idx:04d}/{total_frames:04d} | SCENARIO 2 (TEST SET)"
    (rw, _), _ = cv2.getTextSize(info_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(frame, info_str, (frame.shape[1] - rw - 16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (200, 210, 220), 1, lineType=cv2.LINE_AA)


def draw_motion_trail(frame: np.ndarray, trail: List[Tuple[float, float]],
                      point_color: Tuple[int, int, int] = (0, 230, 255)) -> None:
    """Draws subtle fading trail dots of recent center points."""
    n_pts = len(trail)
    for i, (u_pt, v_pt) in enumerate(trail):
        radius = max(2, int(round(2 + 3 * (i + 1) / n_pts)))
        cv2.circle(frame, (int(round(u_pt)), int(round(v_pt))), radius, point_color, -1, lineType=cv2.LINE_AA)


def parse_camera(
    res_c: Any,
    res_t: Any,
    c_id: int,
    w_orig: int,
    h_orig: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Extracts best cabin and best (NMS-suppressed) truck detection of one camera."""
    best_cabin: Optional[Dict[str, Any]] = None
    best_truck: Optional[Dict[str, Any]] = None

    if res_c.boxes is not None and len(res_c.boxes) > 0:
        confs = res_c.boxes.conf
        bi = int(torch.argmax(confs).item())
        cx1, cy1, cx2, cy2 = res_c.boxes.xyxy[bi].cpu().numpy().tolist()
        best_cabin = {
            "u": round((cx1 + cx2) / 2.0, 2),
            "v": round((cy1 + cy2) / 2.0, 2),
            "w": round(cx2 - cx1, 2),
            "h": round(cy2 - cy1, 2),
            "conf": round(float(confs[bi].item()), 4),
            "box": (cx1, cy1, cx2, cy2),
            "is_occluded": int(cx1 <= 10.0 or cx2 >= w_orig - 10.0 or cy1 <= 10.0 or cy2 >= h_orig - 10.0),
        }

    if res_t.boxes is not None and len(res_t.boxes) > 0:
        cands: List[DetectionResult] = []
        cls_all = res_t.boxes.cls.cpu().numpy()
        conf_all = res_t.boxes.conf.cpu().numpy()
        xyxy_all = res_t.boxes.xyxy.cpu().numpy()
        has_masks = res_t.masks is not None
        for b_i in range(len(cls_all)):
            if int(cls_all[b_i]) != 0:
                continue
            poly = res_t.masks.xy[b_i] if (has_masks and len(res_t.masks) > b_i) else None
            if poly is None or len(poly) < 3:
                bx1, by1, bx2, by2 = xyxy_all[b_i].tolist()
                poly = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)
            tu, tv, tw, th, is_edge = compute_mask_moments(poly, img_w=w_orig, img_h=h_orig)
            cands.append(DetectionResult(
                cam_id=c_id, u=round(float(tu), 2), v=round(float(tv), 2),
                w=round(float(tw), 2), h=round(float(th), 2), conf=round(float(conf_all[b_i]), 4),
                polygon=poly, is_truncated=is_edge, class_id=0, class_name="truck",
            ))
        survivors = suppress_nested_detections(cands, iomin_threshold=0.50)
        if survivors:
            t = max(survivors, key=lambda d: d.conf)
            best_truck = {
                "u": t.u, "v": t.v, "w": t.w, "h": t.h, "conf": t.conf,
                "box": (max(0.0, t.u - t.w / 2.0), max(0.0, t.v - t.h / 2.0),
                        min(float(w_orig - 1), t.u + t.w / 2.0), min(float(h_orig - 1), t.v + t.h / 2.0)),
                "is_occluded": 1 if t.is_truncated else 0,
            }
    return best_cabin, best_truck


def _render_loop(
    q: "queue.Queue",
    total_frames: int,
    dash_writer: Optional[FFmpegNvencWriter],
    cam_writers: Dict[int, FFmpegNvencWriter],
    errors: List[BaseException],
) -> None:
    """Background render thread: draws 2D overlays and pushes frames into NVENC pipes."""
    trails: Dict[int, collections.deque] = {c: collections.deque(maxlen=20) for c in range(1, N_CAMS + 1)}
    try:
        while True:
            item = q.get()
            if item is None:
                break
            f_idx, frames, dets = item
            tiles: List[np.ndarray] = []
            for c_id in range(1, N_CAMS + 1):
                frame = frames[c_id]
                cabin, truck = dets[c_id]
                ref = cabin or truck
                if ref is not None:
                    trails[c_id].append((ref["u"], ref["v"]))
                if len(trails[c_id]) > 1:
                    draw_motion_trail(frame, list(trails[c_id]))
                if truck is not None:
                    draw_styled_box(frame, truck["box"], truck["conf"], truck["u"], truck["v"], "truck",
                                    (0, 220, 100), (0, 255, 128), bool(truck["is_occluded"]))
                if cabin is not None:
                    draw_styled_box(frame, cabin["box"], cabin["conf"], cabin["u"], cabin["v"], "cabin",
                                    (255, 180, 0), (0, 0, 255), bool(cabin["is_occluded"]))
                draw_hud_header(frame, c_id, f_idx, total_frames, truck is not None, cabin is not None)
                if c_id in cam_writers:
                    cam_writers[c_id].write(frame)
                tiles.append(cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA))
            if dash_writer is not None:
                grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6]), np.hstack(tiles[6:9])])
                dash_writer.write(grid)
    except BaseException as e:  # surface render errors to the main thread
        errors.append(e)
        while True:  # drain so the producer never blocks forever
            try:
                if q.get(timeout=0.5) is None:
                    break
            except queue.Empty:
                break


def process_scen_2_dataset(
    videos_dir: str = "data/raw/scen_2",
    yolo_weights: str = "weights/finetuned/peleng_yolo26x_seg_clean_polish_best_fp16.engine",
    cabin_weights: str = "weights/cabin/peleng_cabin_best.pt",
    output_videos_dir: str = "results/scen_2/videos",
    detections_dir: str = "results/scen_2/detections",
    conf_truck: float = 0.25,
    conf_cabin: float = 0.25,
    imgsz_truck: int = 1024,
    imgsz_cabin: int = 960,
    truck_batch: int = 5,
    device: str = "0",
    max_frames: Optional[int] = None,
    save_videos: bool = True,
    individual: bool = False,
    nvdec: bool = True,
) -> Dict[str, Any]:
    """Runs dual-model detection and renders 9-camera videos fully on GPU codecs."""
    v_dir = Path(videos_dir)
    out_vid_dir = Path(output_videos_dir)
    out_det_dir = Path(detections_dir)
    out_vid_dir.mkdir(parents=True, exist_ok=True)
    out_det_dir.mkdir(parents=True, exist_ok=True)

    cam_videos: Dict[int, str] = {}
    for c_id in range(1, N_CAMS + 1):
        matches = list(v_dir.glob(f"*Actor{c_id}.mp4"))
        if not matches:
            raise FileNotFoundError(f"Video for Camera {c_id} not found in {videos_dir}")
        cam_videos[c_id] = str(matches[0])

    logger.info("Loading truck model: %s", yolo_weights)
    model_truck = YOLO(yolo_weights, task="segment")
    logger.info("Loading cabin model: %s", cabin_weights)
    model_cabin = YOLO(cabin_weights)

    probe = cv2.VideoCapture(cam_videos[1])
    w_orig = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    logger.info("%dx%d @ %.1f FPS | frames: %d | NVDEC=%s", w_orig, h_orig, fps, total_frames, nvdec)

    readers = {c: FFmpegNvdecReader(p, w_orig, h_orig, max_frames=total_frames, use_nvdec=nvdec)
               for c, p in cam_videos.items()}

    dashboard_path = out_vid_dir / "multiview_dashboard_scen_2.mp4"
    dash_writer = FFmpegNvencWriter(dashboard_path, TILE_W * 3, TILE_H * 3, fps) if save_videos else None
    cam_writers: Dict[int, FFmpegNvencWriter] = {}
    if save_videos and individual:
        cam_writers = {c: FFmpegNvencWriter(out_vid_dir / f"camera_{c}_detected.mp4", w_orig, h_orig, fps)
                       for c in range(1, N_CAMS + 1)}

    render_q: "queue.Queue" = queue.Queue(maxsize=6)
    render_errors: List[BaseException] = []
    render_thread: Optional[threading.Thread] = None
    if save_videos:
        render_thread = threading.Thread(
            target=_render_loop, args=(render_q, total_frames, dash_writer, cam_writers, render_errors), daemon=True)
        render_thread.start()

    rec_h: List[Dict[str, Any]] = []
    rec_t: List[Dict[str, Any]] = []
    rec_c: List[Dict[str, Any]] = []
    is_engine = str(yolo_weights).endswith(".engine")

    t_start = time.perf_counter()
    done_frames = 0
    try:
        for f_idx in range(total_frames):
            frames: Dict[int, np.ndarray] = {}
            for c_id, rd in readers.items():
                fr = rd.read(timeout=15.0)
                if fr is None:
                    break
                frames[c_id] = fr
            if len(frames) < N_CAMS:
                logger.warning("Stream ended early at frame %d", f_idx)
                break
            if render_errors:
                raise RuntimeError("Render thread failed") from render_errors[0]

            frames_list = [frames[c] for c in range(1, N_CAMS + 1)]

            res_cabin = model_cabin.predict(source=frames_list, conf=conf_cabin, imgsz=imgsz_cabin,
                                            device=device, verbose=False)
            res_truck: List[Any] = []
            for i in range(0, N_CAMS, truck_batch):
                kw: Dict[str, Any] = {"source": frames_list[i:i + truck_batch], "conf": conf_truck,
                                      "imgsz": imgsz_truck, "device": device, "verbose": False}
                if is_engine:
                    kw["rect"] = False
                res_truck.extend(model_truck.predict(**kw))

            dets: Dict[int, Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = {}
            for idx, c_id in enumerate(range(1, N_CAMS + 1)):
                cabin, truck = parse_camera(res_cabin[idx], res_truck[idx], c_id, w_orig, h_orig)
                dets[c_id] = (cabin, truck)
                if cabin is not None:
                    row = {k: cabin[k] for k in ("u", "v", "w", "h", "conf", "is_occluded")}
                    rec_c.append({"Frame": f_idx, "Cam_ID": c_id, **row})
                    rec_h.append({"Frame": f_idx, "Cam_ID": c_id, **row, "label": "cabin"})
                if truck is not None:
                    row = {k: truck[k] for k in ("u", "v", "w", "h", "conf", "is_occluded")}
                    rec_t.append({"Frame": f_idx, "Cam_ID": c_id, **row})
                    if cabin is None:
                        rec_h.append({"Frame": f_idx, "Cam_ID": c_id, **row, "label": "truck"})

            if render_thread is not None:
                render_q.put((f_idx, frames, dets))
            done_frames = f_idx + 1

            if done_frames % 50 == 0 or done_frames == total_frames:
                el = time.perf_counter() - t_start
                logger.info("Processed %d/%d (%.1f%%) | %.2f FPS | ETA %.1f min",
                            done_frames, total_frames, done_frames / total_frames * 100,
                            done_frames / el, (total_frames - done_frames) / (done_frames / el) / 60)
    finally:
        for rd in readers.values():
            rd.close()
        if render_thread is not None:
            render_q.put(None)
            render_thread.join()
        if dash_writer is not None:
            dash_writer.close()
        for w in cam_writers.values():
            w.close()

    if render_errors:
        raise RuntimeError("Render thread failed") from render_errors[0]

    total_time = time.perf_counter() - t_start
    logger.info("Finished %d frames in %.1f s (%.2f FPS)", done_frames, total_time, done_frames / max(total_time, 1e-3))

    csvs = {}
    for name, rows in (("hybrid", rec_h), ("truck", rec_t), ("cabin", rec_c)):
        path = out_det_dir / f"detections_2d_{name}.csv"
        pd.DataFrame(rows).to_csv(path, sep=";", index=False)
        csvs[name] = str(path)
        logger.info("Saved %d %s detections to %s", len(rows), name, path)

    return {"total_frames": done_frames, "total_time_s": total_time,
            "dashboard_path": str(dashboard_path), "csvs": csvs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process Scenario 2 with fine-tuned YOLO26x-seg & cabin detector")
    parser.add_argument("--videos-dir", default="data/raw/scen_2")
    parser.add_argument("--yolo-weights", default="weights/finetuned/peleng_yolo26x_seg_clean_polish_best_fp16.engine",
                        help="Truck seg model; pass ..._polish_best.engine for the FP32 engine (≈3x slower)")
    parser.add_argument("--cabin-weights", default="weights/cabin/peleng_cabin_best.pt")
    parser.add_argument("--output-videos-dir", default="results/scen_2/videos")
    parser.add_argument("--detections-dir", default="results/scen_2/detections")
    parser.add_argument("--conf-truck", type=float, default=0.25)
    parser.add_argument("--conf-cabin", type=float, default=0.25)
    parser.add_argument("--imgsz-truck", type=int, default=1024)
    parser.add_argument("--imgsz-cabin", type=int, default=960)
    parser.add_argument("--truck-batch", type=int, default=5, help="Max engine batch (FP16 engine profile is 1..5)")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--skip-videos", action="store_true", help="Only write CSV detections")
    parser.add_argument("--individual", action="store_true", help="Also write 9 full-HD per-camera videos (slower)")
    parser.add_argument("--no-nvdec", action="store_true", help="Decode on CPU instead of NVDEC")
    return parser.parse_args()


if __name__ == "__main__":
    a = parse_args()
    process_scen_2_dataset(
        videos_dir=a.videos_dir, yolo_weights=a.yolo_weights, cabin_weights=a.cabin_weights,
        output_videos_dir=a.output_videos_dir, detections_dir=a.detections_dir,
        conf_truck=a.conf_truck, conf_cabin=a.conf_cabin, imgsz_truck=a.imgsz_truck,
        imgsz_cabin=a.imgsz_cabin, truck_batch=a.truck_batch, device=a.device,
        max_frames=a.max_frames, save_videos=not a.skip_videos, individual=a.individual,
        nvdec=not a.no_nvdec,
    )
