"""
Training pipeline for Baidu RT-DETR (Real-Time DEtection TRansformer)
on the Peleng Multi-Camera Dataset (v3).

Features:
- Global multi-scale deformable attention for amodal object reasoning.
- Cutout / Random Erasing augmentations (erasing=0.4) to force robust detection
  under partial occlusions.
- Automatic weight export to weights/peleng_rtdetr_best.pt.
- Structured metric summary export.
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
from ultralytics import RTDETR

# Setup clean logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PelengRTDETRTrainer")


@dataclass(frozen=True)
class RTDETRConfig:
    data_yaml: str = "data/datasets/yolo_v3/data.yaml"
    model_name: str = "weights/pretrained/rtdetr-l.pt"
    epochs: int = 35
    batch_size: int = 4
    imgsz: int = 640
    device: str = "0" if torch.cuda.is_available() else "cpu"
    workers: int = 2
    patience: int = 15
    seed: int = 42
    erasing: float = 0.4      # Random erasing probability to simulate occlusions
    mosaic: float = 1.0       # Mosaic augmentation
    mixup: float = 0.1        # Mixup augmentation
    project: str = "runs"
    name: str = "peleng_rtdetr"
    save_weights_dir: str = "weights/finetuned"
    best_weight_name: str = "peleng_rtdetr_best.pt"


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_environment(config: RTDETRConfig) -> None:
    """Validates presence of dataset YAML, directories, and hardware support."""
    data_path = Path(config.data_yaml).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found at: {data_path}")

    # Verify CUDA device availability if requested
    if config.device != "cpu":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"Using GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")


class PelengRTDETRTrainer:
    """Manages RT-DETR model fine-tuning, evaluation, and artifact saving."""

    def __init__(self, config: RTDETRConfig) -> None:
        self.config = config
        validate_environment(self.config)
        set_seed(self.config.seed)

        logger.info(f"Initializing RT-DETR model with backbone: {self.config.model_name}")
        self.model = RTDETR(self.config.model_name)

    def train(self) -> Dict[str, Any]:
        """Runs the training session with occlusion-resilient augmentations."""
        logger.info("=" * 60)
        logger.info(" Starting RT-DETR Training (Occlusion-Resilient Mode)")
        logger.info(f" Dataset:    {self.config.data_yaml}")
        logger.info(f" Epochs:     {self.config.epochs}")
        logger.info(f" Batch size: {self.config.batch_size}")
        logger.info(f" Image size: {self.config.imgsz}")
        logger.info(f" Erasing:    {self.config.erasing} (simulates partial occlusions)")
        logger.info("=" * 60)

        results = self.model.train(
            data=self.config.data_yaml,
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            imgsz=self.config.imgsz,
            device=self.config.device,
            workers=self.config.workers,
            patience=self.config.patience,
            seed=self.config.seed,
            erasing=self.config.erasing,
            mosaic=self.config.mosaic,
            mixup=self.config.mixup,
            project=self.config.project,
            name=self.config.name,
            exist_ok=True,
            verbose=True,
        )

        return self._collect_metrics(results)

    def _collect_metrics(self, train_results: Any) -> Dict[str, Any]:
        """Collects validation metrics and exports the best weights."""
        metrics_dict: Dict[str, Any] = {
            "config": asdict(self.config),
            "save_dir": str(getattr(train_results, "save_dir", "")),
        }

        # Check train_results.box metrics if available
        if hasattr(train_results, "box"):
            box = train_results.box
            metrics_dict["metrics"] = {
                "mAP50": float(getattr(box, "map50", 0.0)),
                "mAP50-95": float(getattr(box, "map", 0.0)),
                "precision": float(getattr(box, "mp", 0.0)),
                "recall": float(getattr(box, "mr", 0.0)),
            }
            logger.info("Validation Results:")
            logger.info(f"  mAP50:      {metrics_dict['metrics']['mAP50']:.4f}")
            logger.info(f"  mAP50-95:   {metrics_dict['metrics']['mAP50-95']:.4f}")
            logger.info(f"  Precision:  {metrics_dict['metrics']['precision']:.4f}")
            logger.info(f"  Recall:     {metrics_dict['metrics']['recall']:.4f}")

        # Locate and copy best.pt to weights directory
        run_dir = Path(metrics_dict["save_dir"])
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            # Fallback search in project/name
            fallback = Path(self.config.project) / self.config.name / "weights" / "best.pt"
            if fallback.exists():
                best_pt = fallback

        if best_pt.exists():
            target_dir = Path(self.config.save_weights_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / self.config.best_weight_name
            shutil.copy2(best_pt, target_path)
            logger.info(f"Successfully saved best weights to: {target_path}")
            metrics_dict["exported_weight"] = str(target_path)
        else:
            logger.warning(f"best.pt not found at expected path: {best_pt}")

        # Save metrics summary JSON
        summary_path = Path("results/metrics/rtdetr_training_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Metrics saved to: {summary_path}")

        return metrics_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR for Peleng 3D Tracking")
    parser.add_argument("--data", default="dataset/peleng.v3i.yolov8/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="rtdetr-l.pt", help="Base model backbone")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4 for 8GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="Device (0 or cpu)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--erasing", type=float, default=0.4, help="Random erasing prob for occlusion resistance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RTDETRConfig(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        erasing=args.erasing,
    )

    trainer = PelengRTDETRTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
