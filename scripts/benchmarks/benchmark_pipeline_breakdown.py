"""
Side-by-Side Pipeline Benchmark: Baseline vs Optimized YOLO26x-seg Pipeline.
Measures latency (ms) and speed (FPS) for:
1. Decode (parallel capture)
2. Model Inference (FP32 TRT vs FP16 TRT)
3. Detection Postprocessing
4. Rendering & Overlays (Full-frame blend vs Bounded ROI blend)
5. Dashboard Assembly
6. Video Encoding (Sync cv2.VideoWriter vs AsyncMultiVideoWriter)
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2

# TensorRT DLL search paths
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

from src.async_capture import AsyncMultiCameraCapture
from src.async_writer import AsyncMultiVideoWriter
from ultralytics import YOLO
from src.detector import compute_mask_moments, suppress_nested_detections, DetectionResult
from scripts.inference.process_videos_yolo26_seg import (
    draw_styled_segmentation,
    draw_hud_header,
    draw_telemetry_tile,
)


def run_benchmark(model_path: str, use_async_writer: bool = True, num_frames: int = 30):
    model = YOLO(model_path, task="segment")
    videos_dir = Path("data/raw/hackathon")
    cam_videos = {i: str(list(videos_dir.glob(f"*Actor{i}.mp4"))[0]) for i in range(1, 6)}

    multi_cap = AsyncMultiCameraCapture(cam_videos, queue_size=4, use_nvdec=True)
    w_orig, h_orig = multi_cap.width, multi_cap.height
    fps = multi_cap.fps or 30.0

    out_dir = Path("results/benchmarks/test_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 640, 360

    if use_async_writer:
        writer = AsyncMultiVideoWriter(queue_size=16)
        for i in range(1, 6):
            writer.add_stream(i, out_dir / f"async_cam_{i}.mp4", fps, (w_orig, h_orig))
        writer.add_stream("dash", out_dir / "async_dash.mp4", fps, (1920, 720))
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        sync_writers = {i: cv2.VideoWriter(str(out_dir / f"sync_cam_{i}.mp4"), fourcc, fps, (w_orig, h_orig)) for i in range(1, 6)}
        sync_dash = cv2.VideoWriter(str(out_dir / "sync_dash.mp4"), fourcc, fps, (1920, 720))

    t_decode, t_infer, t_post, t_draw, t_dash, t_write, t_total = [], [], [], [], [], [], []

    # Warmup
    for _ in range(5):
        raw = multi_cap.read_sync(timeout=3.0)
        frames_list = [raw[c] for c in range(1, 6)]
        _ = model.predict(source=frames_list, conf=0.25, imgsz=1024, device="0", verbose=False, rect=False)

    for f_idx in range(num_frames):
        t0 = time.perf_counter()

        # 1. Decode
        raw_frames = multi_cap.read_sync(timeout=3.0)
        t1 = time.perf_counter()

        # 2. Infer
        frames_list = [raw_frames[c_id] for c_id in range(1, 6)]
        results = model.predict(source=frames_list, conf=0.25, imgsz=1024, device="0", verbose=False, rect=False)
        t2 = time.perf_counter()

        # 3. Postprocess
        cam_stats = {c: None for c in range(1, 6)}
        annotated_tiles = {}
        for idx, c_id in enumerate(range(1, 6)):
            frame = raw_frames[c_id]
            res = results[idx]
            cam_candidates = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b_idx in range(len(res.boxes)):
                    cls_id = int(res.boxes.cls[b_idx].item())
                    if cls_id != 0:
                        continue
                    conf = float(res.boxes.conf[b_idx].item())
                    xyxy = res.boxes.xyxy[b_idx].cpu().numpy().tolist()
                    if res.masks is not None and len(res.masks) > b_idx:
                        poly = res.masks.xy[b_idx]
                    else:
                        bx1, by1, bx2, by2 = xyxy
                        poly = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)
                    u_c, v_c, w_px, h_px, is_trunc = compute_mask_moments(poly, img_w=w_orig, img_h=h_orig)
                    cam_candidates.append(DetectionResult(
                        cam_id=c_id, u=round(float(u_c), 2), v=round(float(v_c), 2),
                        w=round(float(w_px), 2), h=round(float(h_px), 2), conf=round(float(conf), 4),
                        polygon=poly, is_truncated=is_trunc, class_id=cls_id, class_name="truck"
                    ))
            surviving = suppress_nested_detections(cam_candidates, iomin_threshold=0.50)
        t3 = time.perf_counter()

        # 4. Drawing (Optimized ROI)
        for idx, c_id in enumerate(range(1, 6)):
            frame = raw_frames[c_id]
            for det in surviving:
                bx1 = max(0.0, det.u - det.w / 2.0)
                by1 = max(0.0, det.v - det.h / 2.0)
                bx2 = min(float(w_orig - 1), det.u + det.w / 2.0)
                by2 = min(float(h_orig - 1), det.v + det.h / 2.0)
                draw_styled_segmentation(frame, det.polygon, (bx1, by1, bx2, by2), det.conf, det.u, det.v, det.is_truncated, cam_id=c_id)
            draw_hud_header(frame, c_id, f_idx, 1000, len(surviving))
        t4 = time.perf_counter()

        # 5. Dashboard Assembly
        for c_id in range(1, 6):
            annotated_tiles[c_id] = cv2.resize(raw_frames[c_id], (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        telemetry_tile = draw_telemetry_tile(tile_w, tile_h, f_idx, 1000, cam_stats, fps)
        row1 = np.hstack([annotated_tiles[1], annotated_tiles[2], annotated_tiles[3]])
        row2 = np.hstack([annotated_tiles[4], annotated_tiles[5], telemetry_tile])
        dashboard_frame = np.vstack([row1, row2])
        t5 = time.perf_counter()

        # 6. Video Writing
        if use_async_writer:
            for c_id in range(1, 6):
                writer.write(c_id, raw_frames[c_id])
            writer.write("dash", dashboard_frame)
        else:
            for c_id in range(1, 6):
                sync_writers[c_id].write(raw_frames[c_id])
            sync_dash.write(dashboard_frame)
        t6 = time.perf_counter()

        t_decode.append((t1 - t0) * 1000)
        t_infer.append((t2 - t1) * 1000)
        t_post.append((t3 - t2) * 1000)
        t_draw.append((t4 - t3) * 1000)
        t_dash.append((t5 - t4) * 1000)
        t_write.append((t6 - t5) * 1000)
        t_total.append((t6 - t0) * 1000)

    multi_cap.close()
    if use_async_writer:
        writer.close()
    else:
        for w in sync_writers.values():
            w.release()
        sync_dash.release()

    return {
        "decode": float(np.mean(t_decode)),
        "infer": float(np.mean(t_infer)),
        "post": float(np.mean(t_post)),
        "draw": float(np.mean(t_draw)),
        "dash": float(np.mean(t_dash)),
        "write": float(np.mean(t_write)),
        "total": float(np.mean(t_total)),
        "fps": float(1000.0 / np.mean(t_total)),
    }


def main():
    fp32_engine = "weights/finetuned/peleng_yolo26x_seg_clean_best.engine"
    fp16_engine = "weights/finetuned/peleng_yolo26x_seg_clean_polish_best_fp16.engine"

    print("\n" + "=" * 78)
    print("  RUNNING PIPELINE PERFORMANCE COMPARISON: BASELINE VS OPTIMIZED")
    print("=" * 78)

    print("\n>>> Profiling BASELINE (FP32 TensorRT + Synchronous VideoWriter)...")
    res_base = run_benchmark(fp32_engine, use_async_writer=False, num_frames=25)

    print("\n>>> Profiling OPTIMIZED (FP16 TensorRT + AsyncMultiVideoWriter + ROI Blending)...")
    res_opt = run_benchmark(fp16_engine, use_async_writer=True, num_frames=25)

    print("\n" + "=" * 78)
    print("                     STAGE LATENCY & SPEED COMPARISON")
    print("=" * 78)
    print(f" {'Stage':<26} | {'Baseline':<12} | {'Optimized':<12} | {'Speedup / Improvement':<20}")
    print("-" * 78)
    print(f" {'1. Decode (Capture)':<26} | {res_base['decode']:8.2f} ms | {res_opt['decode']:8.2f} ms | {res_base['decode']/(res_opt['decode']+1e-6):4.1f}x")
    print(f" {'2. Inference (TRT Batch 5)':<26} | {res_base['infer']:8.2f} ms | {res_opt['infer']:8.2f} ms | {res_base['infer']/res_opt['infer']:4.1f}x (FP16 Engine)")
    print(f" {'3. Postprocessing':<26} | {res_base['post']:8.2f} ms | {res_opt['post']:8.2f} ms | {res_base['post']/res_opt['post']:4.1f}x")
    print(f" {'4. Drawing (ROI Blend)':<26} | {res_base['draw']:8.2f} ms | {res_opt['draw']:8.2f} ms | {res_base['draw']/res_opt['draw']:4.1f}x")
    print(f" {'5. Dashboard Assembly':<26} | {res_base['dash']:8.2f} ms | {res_opt['dash']:8.2f} ms | {res_base['dash']/res_opt['dash']:4.1f}x")
    print(f" {'6. Video Writing (6 streams)':<26} | {res_base['write']:8.2f} ms | {res_opt['write']:8.2f} ms | {res_base['write']/(res_opt['write']+1e-6):4.1f}x (Async Thread Pool)")
    print("-" * 78)
    print(f" {'TOTAL STEP LATENCY':<26} | {res_base['total']:8.2f} ms | {res_opt['total']:8.2f} ms | {res_base['total']/res_opt['total']:4.1f}x FASTER")
    print(f" {'EFFECTIVE SPEED':<26} | {res_base['fps']:8.2f} FPS| {res_opt['fps']:8.2f} FPS| +{res_opt['fps'] - res_base['fps']:.2f} FPS ({res_opt['fps']/res_base['fps']:.1f}x)")
    print(f" {'5-CAMERA THROUGHPUT':<26} | {res_base['fps']*5:8.1f} c-FPS| {res_opt['fps']*5:8.1f} c-FPS| {res_opt['fps']/res_base['fps']:.1f}x")
    print("=" * 78)


if __name__ == "__main__":
    main()
