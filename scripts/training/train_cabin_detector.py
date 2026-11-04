"""
Fine-tuning script for Peleng Cabin Detector using YOLO.
Optimized for NVIDIA RTX 3050 Laptop GPU (4 GB VRAM).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
from ultralytics import YOLO


def train_cabin_detector(
    data_yaml: str = "data/cabin_dataset/data.yaml",
    base_model: str = "weights/peleng_yolov8x_best.pt",
    epochs: int = 15,
    batch_size: int = 4,
    imgsz: int = 640,
    freeze: int = 10,
    device: str = "0" if torch.cuda.is_available() else "cpu",
    output_dir: str = "weights/cabin",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("  PELENG CABIN DETECTOR FINE-TUNING")
    print("=" * 60)
    print(f"Data:       {data_yaml}")
    print(f"Base Model: {base_model}")
    print(f"Epochs:     {epochs}")
    print(f"Batch:      {batch_size}")
    print(f"Img Size:   {imgsz}")
    print(f"Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Freeze:     {freeze} backbone layers")
    print("=" * 60)

    # If the base model weights exist, use them, otherwise use pretrained yolov8s/m
    if os.path.exists(base_model):
        print(f"[INFO] Initializing from existing weights: {base_model}")
        model = YOLO(base_model)
    else:
        print("[INFO] Initializing from pretrained yolov8s.pt...")
        model = YOLO("yolov8s.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        freeze=freeze,
        project="runs/detect",
        name="cabin_detector",
        exist_ok=True,
        workers=2,
        patience=6,
        save=True,
        plots=True,
        val=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        amp=True,  # FP16 mixed precision for fast training on RTX GPU
    )

    best_weight = os.path.join("runs", "detect", "cabin_detector", "weights", "best.pt")
    target_weight = os.path.join(output_dir, "peleng_cabin_best.pt")
    if os.path.exists(best_weight):
        import shutil
        shutil.copy(best_weight, target_weight)
        print(f"\n[SUCCESS] Cabin detector saved to: '{target_weight}'")
    else:
        print(f"[WARNING] Best weights not found at {best_weight}")

    return target_weight


if __name__ == "__main__":
    train_cabin_detector()
