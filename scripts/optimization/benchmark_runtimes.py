"""
Benchmark script comparing YOLO26-seg inference runtimes on NVIDIA GPU:
1. PyTorch FP32
2. PyTorch FP16 (Tensor Cores)
3. ONNX Runtime GPU (CUDAExecutionProvider)
4. TensorRT Engine (.engine, if available)

Outputs structured comparison table with latency (ms), throughput (FPS), and VRAM.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BenchmarkRuntimes")


def benchmark_model(
    model_path: str,
    device: str = "0",
    half: bool = False,
    batch_size: int = 5,
    imgsz: int = 1024,
    n_iters: int = 10,
    warmup_iters: int = 3,
) -> dict:
    """Runs timed benchmark for a given model checkpoint."""
    path = Path(model_path)
    if not path.exists():
        return {"status": "file_not_found"}

    logger.info("Benchmarking: %s (half=%s, batch=%d, imgsz=%d)", path.name, half, batch_size, imgsz)

    # Prepare synthetic 1080p frames matching 5 cameras
    dummy_frames = [
        np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]

    model = YOLO(str(path))

    # Warmup
    for _ in range(warmup_iters):
        model.predict(dummy_frames, device=device, half=half, imgsz=imgsz, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    for _ in range(n_iters):
        model.predict(dummy_frames, device=device, half=half, imgsz=imgsz, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_total = time.perf_counter() - t_start

    latency_per_batch_ms = (t_total / n_iters) * 1000.0
    throughput_fps = (n_iters * batch_size) / t_total

    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "status": "success",
        "name": path.name,
        "latency_batch_ms": latency_per_batch_ms,
        "throughput_fps": throughput_fps,
        "vram_mb": vram_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO26-seg across runtimes")
    parser.add_argument("--pt-weights", default="weights/finetuned/peleng_yolo26x_seg_roboflow_best.pt")
    parser.add_argument("--onnx-model", default="weights/finetuned/peleng_yolo26x_seg_roboflow_best.onnx")
    parser.add_argument("--engine-model", default="weights/finetuned/peleng_yolo26x_seg.engine")
    parser.add_argument("--batch", type=int, default=5, help="Number of synchronized cameras")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    results = []

    # 1. PyTorch FP32
    if Path(args.pt_weights).exists():
        r_fp32 = benchmark_model(args.pt_weights, half=False, batch_size=args.batch, imgsz=args.imgsz, n_iters=args.iters)
        if r_fp32["status"] == "success":
            results.append(("PyTorch FP32", r_fp32))

    # 2. PyTorch FP16
    if Path(args.pt_weights).exists():
        r_fp16 = benchmark_model(args.pt_weights, half=True, batch_size=args.batch, imgsz=args.imgsz, n_iters=args.iters)
        if r_fp16["status"] == "success":
            results.append(("PyTorch FP16 (Tensor Cores)", r_fp16))

    # 3. ONNX Runtime GPU
    if Path(args.onnx_model).exists():
        r_onnx = benchmark_model(args.onnx_model, half=False, batch_size=args.batch, imgsz=args.imgsz, n_iters=args.iters)
        if r_onnx["status"] == "success":
            results.append(("ONNX Runtime GPU", r_onnx))

    # 4. TensorRT Engine
    if Path(args.engine_model).exists():
        r_trt = benchmark_model(args.engine_model, half=True, batch_size=args.batch, imgsz=args.imgsz, n_iters=args.iters)
        if r_trt["status"] == "success":
            results.append(("TensorRT Engine (.engine)", r_trt))

    # Display comparison table
    print("\n" + "=" * 75)
    print(f"  PELENG YOLO RUNTIME BENCHMARK (Batch={args.batch} Cams @ {args.imgsz}x{args.imgsz})")
    print("=" * 75)
    print(f"{'Runtime Engine':<30} | {'5-Cam Latency':<16} | {'Throughput (FPS)':<18} | {'VRAM (MB)':<10}")
    print("-" * 75)
    for label, res in results:
        lat = f"{res['latency_batch_ms']:.1f} ms"
        fps = f"{res['throughput_fps']:.1f} FPS"
        vram = f"{res['vram_mb']:.0f} MB"
        print(f"{label:<30} | {lat:<16} | {fps:<18} | {vram:<10}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
