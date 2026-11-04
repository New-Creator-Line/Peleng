"""
Comparative evaluation and visual benchmarking between two YOLO26 segmentation models:
- Model A: Trained on raw / initial unmerged dataset
- Model B: Trained on unified Roboflow dataset (My First Project.v1i.yolo26)

Produces:
1. Quantitative validation metric table (Box & Mask mAP50, mAP50-95, Precision, Recall).
2. Loss curve comparison if training run logs (results.csv) are available.
3. Side-by-side visual comparison on multi-camera test frames (highlighting occlusions).
4. Markdown summary report saved to results/comparisons/comparison_report.md.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelComparator")


def evaluate_model(model_path: Path, data_yaml: str, device: str = "0") -> Dict[str, float]:
    """Run validation benchmark on target dataset."""
    logger.info("Evaluating %s on %s...", model_path.name, data_yaml)
    model = YOLO(str(model_path))
    metrics = model.val(
        data=data_yaml,
        device=device,
        batch=2,
        imgsz=1024,
        verbose=False,
    )

    box_res = metrics.box
    seg_res = metrics.seg

    return {
        "box_p": float(box_res.p[0]) if len(box_res.p) else float(box_res.mp),
        "box_r": float(box_res.r[0]) if len(box_res.r) else float(box_res.mr),
        "box_map50": float(box_res.map50),
        "box_map50_95": float(box_res.map),
        "mask_p": float(seg_res.p[0]) if len(seg_res.p) else float(seg_res.mp),
        "mask_r": float(seg_res.r[0]) if len(seg_res.r) else float(seg_res.mr),
        "mask_map50": float(seg_res.map50),
        "mask_map50_95": float(seg_res.map),
        "speed_preprocess_ms": float(metrics.speed.get("preprocess", 0.0)),
        "speed_inference_ms": float(metrics.speed.get("inference", 0.0)),
        "speed_postprocess_ms": float(metrics.speed.get("postprocess", 0.0)),
    }


def parse_training_csv(csv_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Extract key metrics from results.csv if it exists."""
    if not csv_path or not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        # Best mask mAP50
        mask_col = [c for c in df.columns if "mAP50(M)" in c or "metrics/mAP50(M)" in c]
        best_epoch = int(df[mask_col[0]].idxmax()) + 1 if mask_col else len(df)
        best_map = float(df[mask_col[0]].max()) if mask_col else 0.0

        return {
            "total_epochs": len(df),
            "best_epoch": best_epoch,
            "best_mask_map50": best_map,
            "final_train_loss": float(df["train/seg_loss"].iloc[-1]) if "train/seg_loss" in df else 0.0,
        }
    except Exception as e:
        logger.warning("Could not parse %s: %s", csv_path, e)
        return None


def generate_side_by_side(
    model_a: YOLO,
    model_b: YOLO,
    img_path: Path,
    label_a: str,
    label_b: str,
    output_path: Path,
    conf: float = 0.25,
) -> None:
    """Generate side-by-side visual comparison image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return

    res_a = model_a.predict(img, conf=conf, verbose=False)[0]
    res_b = model_b.predict(img, conf=conf, verbose=False)[0]

    plot_a = res_a.plot(boxes=True, masks=True)
    plot_b = res_b.plot(boxes=True, masks=True)

    # Add header bars
    h, w, _ = plot_a.shape
    bar_h = 50
    header_a = np.zeros((bar_h, w, 3), dtype=np.uint8)
    header_b = np.zeros((bar_h, w, 3), dtype=np.uint8)

    num_det_a = len(res_a.boxes) if res_a.boxes is not None else 0
    num_det_b = len(res_b.boxes) if res_b.boxes is not None else 0

    text_a = f"{label_a} | Detections: {num_det_a}"
    text_b = f"{label_b} | Detections: {num_det_b}"

    cv2.putText(header_a, text_a, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
    cv2.putText(header_b, text_b, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    col_a = np.vstack([header_a, plot_a])
    col_b = np.vstack([header_b, plot_b])

    # Resize to fit screen cleanly if wide (e.g. 2 x 1920 = 3840 -> scale by 0.5)
    combined = np.hstack([col_a, col_b])
    target_w = 1920
    target_h = int(combined.shape[0] * (target_w / combined.shape[1]))
    resized = cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_AREA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), resized)
    logger.info("Saved visual comparison: %s", output_path)


def run_comparison(
    model_a_path: Path,
    model_b_path: Path,
    label_a: str,
    label_b: str,
    data_yaml: str,
    run_a_dir: Optional[Path],
    run_b_dir: Optional[Path],
    output_dir: Path,
    device: str = "0",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validation benchmark
    logger.info("--- Phase 1: Benchmark Validation on Common Dataset ---")
    metrics_a = evaluate_model(model_a_path, data_yaml, device=device)
    metrics_b = evaluate_model(model_b_path, data_yaml, device=device)

    # 2. Parse training logs if available
    train_a = parse_training_csv(run_a_dir / "results.csv" if run_a_dir else None)
    train_b = parse_training_csv(run_b_dir / "results.csv" if run_b_dir else None)

    # 3. Visual Comparison
    logger.info("--- Phase 2: Generating Visual Side-by-Side Frames ---")
    model_a = YOLO(str(model_a_path))
    model_b = YOLO(str(model_b_path))

    # Pick representative frames across cameras
    test_frames: List[Path] = []
    val_images_dir = Path("data/datasets/My First Project.v1i.yolo26/valid/images")
    if val_images_dir.exists():
        all_val = sorted(val_images_dir.glob("*.jpg"))
        # Pick 1 from each camera if possible
        cams = {}
        for p in all_val:
            cam_key = p.name.split("_")[0]
            if cam_key not in cams:
                cams[cam_key] = p
        test_frames = list(cams.values())[:5]

    for p in test_frames:
        out_img = output_dir / f"compare_{p.stem}.jpg"
        generate_side_by_side(model_a, model_b, p, label_a, label_b, out_img)

    # 4. Generate Markdown Report
    report_file = output_dir / "comparison_report.md"
    diff_box_map50 = (metrics_b["box_map50"] - metrics_a["box_map50"]) * 100
    diff_mask_map50 = (metrics_b["mask_map50"] - metrics_a["mask_map50"]) * 100
    diff_mask_map = (metrics_b["mask_map50_95"] - metrics_a["mask_map50_95"]) * 100

    md = f"""# YOLO26 Segmentation Comparison Report

**Generated:** Auto-benchmark on `{data_yaml}`

| Metric | {label_a} | {label_b} | Delta (B vs A) |
|:---|:---:|:---:|:---:|
| **Weights Path** | `{model_a_path.name}` | `{model_b_path.name}` | - |
| **Mask mAP@50** | **{metrics_a['mask_map50']*100:.2f}%** | **{metrics_b['mask_map50']*100:.2f}%** | **{diff_mask_map50:+.2f}%** |
| **Mask mAP@50-95** | {metrics_a['mask_map50_95']*100:.2f}% | {metrics_b['mask_map50_95']*100:.2f}% | {diff_mask_map:+.2f}% |
| **Mask Precision** | {metrics_a['mask_p']*100:.2f}% | {metrics_b['mask_p']*100:.2f}% | {(metrics_b['mask_p'] - metrics_a['mask_p'])*100:+.2f}% |
| **Mask Recall** | {metrics_a['mask_r']*100:.2f}% | {metrics_b['mask_r']*100:.2f}% | {(metrics_b['mask_r'] - metrics_a['mask_r'])*100:+.2f}% |
| **Box mAP@50** | {metrics_a['box_map50']*100:.2f}% | {metrics_b['box_map50']*100:.2f}% | {diff_box_map50:+.2f}% |
| **Box mAP@50-95** | {metrics_a['box_map50_95']*100:.2f}% | {metrics_b['box_map50_95']*100:.2f}% | {(metrics_b['box_map50_95'] - metrics_a['box_map50_95'])*100:+.2f}% |
| **Inference Latency** | {metrics_a['speed_inference_ms']:.1f} ms | {metrics_b['speed_inference_ms']:.1f} ms | - |

## Training Dynamics

| Parameter | {label_a} | {label_b} |
|:---|:---:|:---:|
| **Epochs Trained** | {train_a['total_epochs'] if train_a else 'N/A'} | {train_b['total_epochs'] if train_b else 'N/A'} |
| **Best Epoch** | {train_a['best_epoch'] if train_a else 'N/A'} | {train_b['best_epoch'] if train_b else 'N/A'} |
| **Best Mask mAP50 (Train log)** | {f"{train_a['best_mask_map50']*100:.2f}%" if train_a else 'N/A'} | {f"{train_b['best_mask_map50']*100:.2f}%" if train_b else 'N/A'} |

## Visual Side-by-Side Images
The side-by-side comparison images have been saved to `{output_dir.as_posix()}`.
"""
    report_file.write_text(md, encoding="utf-8")
    logger.info("Comparison report generated successfully: %s", report_file)
    print("\n" + md)
    return report_file


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two YOLO26-seg models")
    parser.add_argument("--model-a", default="weights/finetuned/peleng_yolo26x_seg_best.pt",
                        help="Path to Model A weights (Old dataset)")
    parser.add_argument("--model-b", default="weights/finetuned/peleng_yolo26x_seg_roboflow_best.pt",
                        help="Path to Model B weights (New Roboflow dataset)")
    parser.add_argument("--label-a", default="Old_Dataset", help="Label for Model A")
    parser.add_argument("--label-b", default="Roboflow_Dataset", help="Label for Model B")
    parser.add_argument("--data", default="data/datasets/My First Project.v1i.yolo26/data.yaml",
                        help="Benchmark validation dataset yaml")
    parser.add_argument("--run-a-dir", default="runs/segment/runs/segment/peleng_yolo26x_seg",
                        help="Directory of Model A training run")
    parser.add_argument("--run-b-dir", default="runs/segment/peleng_yolo26x_seg_roboflow",
                        help="Directory of Model B training run")
    parser.add_argument("--output-dir", default="results/comparisons",
                        help="Directory to store comparison results")
    parser.add_argument("--device", default="0", help="CUDA device ID or 'cpu'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_a_p = Path(args.model_a)
    model_b_p = Path(args.model_b)

    if not model_a_p.exists():
        raise FileNotFoundError(f"Model A weights not found at: {model_a_p.resolve()}")
    if not model_b_p.exists():
        logger.warning(
            "Model B weights not yet found at %s.\n"
            "Once you finish training Model B, re-run this script to generate the full comparison!",
            model_b_p.resolve()
        )
    else:
        run_comparison(
            model_a_path=model_a_p,
            model_b_path=model_b_p,
            label_a=args.label_a,
            label_b=args.label_b,
            data_yaml=args.data,
            run_a_dir=Path(args.run_a_dir) if args.run_a_dir else None,
            run_b_dir=Path(args.run_b_dir) if args.run_b_dir else None,
            output_dir=Path(args.output_dir),
            device=args.device,
        )
