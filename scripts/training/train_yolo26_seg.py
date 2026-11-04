"""
Training pipeline for YOLO26s-seg (Instance Segmentation)
on the curated Peleng Polygon Dataset.

Implements requirements from plan_YOLO_Kalman.md Section 3.2:
- Target: truck instance segmentation masks
- Online GPU augmentations: random erasing (0.4), mosaic (1.0), mixup (0.1)
- Strict boundary alignment: close_mosaic=15 (turns off mosaic 15 epochs before finish)
- Export to weights/finetuned/peleng_yolo26s_seg_best.pt
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PelengYOLO26Trainer")


@dataclass(frozen=True)
class TrainingConfig:
    data_yaml: str = "data/datasets/My First Project.v1i.yolo26/data.yaml"
    model_name: str = "yolo26x-seg.pt"
    epochs: int = 40
    batch_size: int = 2
    imgsz: int = 1024
    device: str = "0" if torch.cuda.is_available() else "cpu"
    workers: int = 2
    patience: int = 0         # 0 disables early stopping, guaranteeing all epochs run
    seed: int = 42
    lr0: float = 0.001        # Safe initial learning rate for fine-tuning
    close_mosaic: int = 10    # Turn off mosaic 10 epochs before finish for clean polygon boundaries
    erasing: float = 0.4      # Cutout / random erasing to simulate foliage occlusions
    mosaic: float = 1.0
    mixup: float = 0.1
    warmup_epochs: float = 3.0 # Warmup duration in epochs
    cache: str = "ram"        # Fully cache dataset in RAM
    nbs: int = 64             # Nominal batch size for automatic gradient accumulation
    freeze: Optional[int] = 10  # Freeze 10 backbone layers by default to preserve COCO weights & stabilize BN
    project: str = "runs/segment"
    name: str = "peleng_yolo26x_seg_roboflow"
    save_weights_dir: str = "weights/finetuned"
    best_weight_name: str = "peleng_yolo26x_seg_roboflow_best.pt"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_environment(cfg: TrainingConfig) -> None:
    yaml_path = Path(cfg.data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path.resolve()}")

    logger.info("Environment check passed. Device: %s (CUDA available: %s)",
                cfg.device, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s | VRAM: %.2f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))


def train_yolo26_seg(cfg: TrainingConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    validate_environment(cfg)

    logger.info("Initializing %s for instance segmentation...", cfg.model_name)
    model = YOLO(cfg.model_name)

    logger.info("Starting training for %d epochs with batch=%d, imgsz=%d, close_mosaic=%d...",
                cfg.epochs, cfg.batch_size, cfg.imgsz, cfg.close_mosaic)

    train_args = {
        "data": cfg.data_yaml,
        "epochs": cfg.epochs,
        "batch": cfg.batch_size,
        "imgsz": cfg.imgsz,
        "device": cfg.device,
        "workers": cfg.workers,
        "patience": cfg.patience,
        "seed": cfg.seed,
        "lr0": cfg.lr0,
        "close_mosaic": cfg.close_mosaic,
        "erasing": cfg.erasing,
        "mosaic": cfg.mosaic,
        "mixup": cfg.mixup,
        "warmup_epochs": cfg.warmup_epochs,
        "cache": cfg.cache,
        "nbs": cfg.nbs,
        "project": cfg.project,
        "name": cfg.name,
        "exist_ok": True,
        "verbose": True,
    }
    if cfg.freeze is not None:
        train_args["freeze"] = cfg.freeze
    if cfg.mosaic == 0.0:
        # Zero noise / clean polish mode
        train_args["hsv_h"] = 0.0
        train_args["hsv_s"] = 0.0
        train_args["hsv_v"] = 0.0
        train_args["fliplr"] = 0.0
        train_args["auto_augment"] = None

    results = model.train(**train_args)

    # Export best weights
    save_dir = Path(cfg.save_weights_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_pt = Path(model.trainer.best)

    exported_pt = save_dir / cfg.best_weight_name
    if best_pt.exists():
        shutil.copy2(best_pt, exported_pt)
        logger.info("Exported best model weights to: %s", exported_pt.resolve())
    else:
        logger.warning("best.pt not found at %s", best_pt)

    # Save summary metrics
    summary = {
        "config": asdict(cfg),
        "save_dir": str(model.trainer.save_dir),
        "exported_weight": str(exported_pt),
    }

    metrics_file = Path("results/metrics/yolo26_seg_training_summary.json")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training pipeline finished. Summary saved to: %s", metrics_file)
    return summary


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train YOLO26x-seg on Peleng Dataset")
    parser.add_argument("--data", default="data/datasets/My First Project.v1i.yolo26/data.yaml")
    parser.add_argument("--model", default="yolo26x-seg.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--erasing", type=float, default=0.4)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--cache", default="ram", choices=["ram", "disk", "False", "True"])
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size for gradient accumulation")
    parser.add_argument("--freeze", type=int, default=10, help="Number of backbone layers to freeze to stabilize BN")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 to disable)")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--name", default="peleng_yolo26x_seg_roboflow", help="Experiment run name")
    parser.add_argument("--clean-polish", action="store_true", help="Zero-augmentation clean polishing mode")
    args = parser.parse_args()

    cache_val: Any = args.cache
    if cache_val == "True":
        cache_val = True
    elif cache_val == "False":
        cache_val = False

    # Apply clean polish overrides if requested
    epochs = args.epochs
    lr0 = args.lr0
    mosaic = args.mosaic
    mixup = args.mixup
    erasing = args.erasing
    warmup_epochs = args.warmup_epochs
    close_mosaic = args.close_mosaic
    name = args.name

    if args.clean_polish:
        logger.info("Applying CLEAN POLISH presets: zero augmentations, zero warmup spike, ultra-low LR.")
        mosaic = 0.0
        mixup = 0.0
        erasing = 0.0
        warmup_epochs = 0.0
        close_mosaic = 0
        lr0 = 0.00005 if args.lr0 == 0.001 else args.lr0
        epochs = 15 if args.epochs == 40 else args.epochs
        if args.name == "peleng_yolo26x_seg_roboflow":
            name = "peleng_yolo26x_seg_clean_polish"

    best_name = f"{name}_best.pt"
    return TrainingConfig(
        data_yaml=args.data,
        model_name=args.model,
        epochs=epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        close_mosaic=close_mosaic,
        erasing=erasing,
        mosaic=mosaic,
        mixup=mixup,
        warmup_epochs=warmup_epochs,
        cache=cache_val,
        nbs=args.nbs,
        freeze=args.freeze,
        patience=args.patience,
        lr0=lr0,
        name=name,
        best_weight_name=best_name,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_yolo26_seg(cfg)
