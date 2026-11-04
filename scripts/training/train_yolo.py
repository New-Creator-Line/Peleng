"""
Fine-tuning pipeline for YOLOv8x on Peleng Multi-Camera Dataset.

Usage:
    python train_yolo.py [--model yolov8x.pt] [--epochs 50] [--batch 8] [--imgsz 512] [--device 0]
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from ultralytics import YOLO

# Setup clean logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PelengYOLOTrainer")


@dataclass(frozen=True)
class TrainingConfig:
    data_yaml: str = "data/datasets/yolo_v3/data.yaml"
    model_name: str = "weights/pretrained/yolov8x.pt"
    epochs: int = 50
    batch_size: int = 8
    imgsz: int = 512
    device: str = "0" if torch.cuda.is_available() else "cpu"
    workers: int = 2
    patience: int = 15
    seed: int = 42
    project: str = "runs/detect"
    name: str = "peleng_yolov8x"
    save_weights_dir: str = "weights/finetuned"
    best_weight_name: str = "peleng_yolov8x_best.pt"


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Reproducibility seed set to: {seed}")


def validate_environment(config: TrainingConfig) -> None:
    """Validates presence of dataset YAML, directories, and hardware support."""
    data_path = Path(config.data_yaml).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found at: {data_path}")

    logger.info(f"Using dataset configuration: {data_path}")
    logger.info(f"Target model architecture: {config.model_name}")

    if config.device != "cpu" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"CUDA accelerator active: {gpu_name} ({gpu_mem:.2f} GB VRAM)")
    else:
        logger.warning("CUDA not available or CPU requested. Training will run on CPU.")


def train_yolov8(config: TrainingConfig) -> Dict[str, Any]:
    """
    Executes model fine-tuning with early stopping, validation, and weight preservation.
    """
    set_seed(config.seed)
    validate_environment(config)

    logger.info(f"Loading pretrained weights: {config.model_name}...")
    model = YOLO(config.model_name)

    logger.info("Initiating model training loop:")
    logger.info(f"  - Epochs: {config.epochs}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Image size: {config.imgsz}")
    logger.info(f"  - Early stopping patience: {config.patience}")
    logger.info(f"  - Device: {config.device}")

    # Train model
    results = model.train(
        data=str(Path(config.data_yaml).resolve()),
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.imgsz,
        device=config.device,
        workers=config.workers,
        patience=config.patience,
        project=config.project,
        name=config.name,
        seed=config.seed,
        pretrained=True,
        save=True,
        plots=True,
        exist_ok=True,
    )

    train_save_dir = Path(model.trainer.save_dir)
    logger.info(f"Training run saved to: {train_save_dir}")

    # Validation on Val split
    logger.info("Evaluating fine-tuned model on validation split...")
    val_metrics = model.val(
        data=str(Path(config.data_yaml).resolve()),
        split="val",
        imgsz=config.imgsz,
        batch=config.batch_size,
        device=config.device,
    )

    # Validation on Test split
    logger.info("Evaluating fine-tuned model on test split...")
    test_metrics = model.val(
        data=str(Path(config.data_yaml).resolve()),
        split="test",
        imgsz=config.imgsz,
        batch=config.batch_size,
        device=config.device,
    )

    # Extract metrics summary
    summary = {
        "model": config.model_name,
        "epochs_trained": getattr(model.trainer, "epoch", config.epochs) + 1,
        "val": {
            "mAP50": float(val_metrics.box.map50),
            "mAP50-95": float(val_metrics.box.map),
            "precision": float(val_metrics.box.mp),
            "recall": float(val_metrics.box.mr),
        },
        "test": {
            "mAP50": float(test_metrics.box.map50),
            "mAP50-95": float(test_metrics.box.map),
            "precision": float(test_metrics.box.mp),
            "recall": float(test_metrics.box.mr),
        },
    }

    # Copy best weights to dedicated folder
    best_pt = train_save_dir / "weights" / "best.pt"
    dest_dir = Path(config.save_weights_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_best_pt = dest_dir / config.best_weight_name

    if best_pt.exists():
        shutil.copy2(best_pt, dest_best_pt)
        logger.info(f"Saved best weights to: {dest_best_pt}")
        summary["best_weights_path"] = str(dest_best_pt.resolve())
    else:
        logger.warning(f"best.pt not found at {best_pt}")

    # Also save to results/
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "yolo_metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved metrics summary to: {metrics_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Peleng Dataset")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="Pretrained YOLO model (default: yolov8x.pt)")
    parser.add_argument("--data", type=str, default="dataset_v3/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=512, help="Image resolution (default: 512)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('0' or 'cpu')")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    
    cfg = TrainingConfig(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        patience=args.patience,
    )
    
    print("=" * 60)
    print("  PELENG YOLOv8 FINE-TUNING PIPELINE")
    print("=" * 60)
    summary_metrics = train_yolov8(cfg)
    print("\n--- Training & Validation Summary ---")
    print(json.dumps(summary_metrics, indent=2))
