"""
Curate and Pre-Annotate Scene 2 Dataset with AI Polygon & Bounding Box Suggestions.

Pipeline:
1. Discovers all extracted frames from data/datasets/scen_2_raw/images.
2. Identifies all positive frames (~340 frames) where the truck is visible.
3. Generates AI-assisted AnyLabeling JSON polygons (class 'truck') and boxes (class 'cabin').
4. Samples ~110 snow background negative frames across all 9 cameras to suppress snow false positives.
5. Emits ready-to-verify AnyLabeling project in data/datasets/scen_2_curated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("CuratorPreannotator")


def approximate_polygon(pts: np.ndarray, epsilon: float = 1.8) -> List[List[float]]:
    """Simplify dense mask contours to ~25-45 clean vertices for smooth AnyLabeling editing."""
    if len(pts) < 4:
        return pts.tolist()
    approx = cv2.approxPolyDP(pts.astype(np.float32), epsilon=epsilon, closed=True)
    coords = approx[:, 0, :].tolist()
    # Guard: ensure at least 4 points remain
    if len(coords) < 3:
        return pts.tolist()
    return coords


def create_anylabeling_json(
    image_name: str,
    image_width: int,
    image_height: int,
    shapes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Construct a valid AnyLabeling / LabelMe v0.4.43 schema object."""
    return {
        "version": "0.4.43",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


def curate_and_preannotate(
    raw_images_dir: Path,
    output_dir: Path,
    seg_weights: Path,
    cabin_weights: Path,
    target_snow_negatives: int = 110,
    conf_threshold: float = 0.25,
    device: str = "0" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Execute AI pre-annotation on positive frames and sample diverse snow background negatives."""
    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Raw images directory not found: {raw_images_dir}")

    output_img_dir = output_dir / "images"
    output_img_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading segmentation model: %s", seg_weights)
    seg_model = YOLO(str(seg_weights))

    logger.info("Loading cabin model: %s", cabin_weights)
    cabin_model = YOLO(str(cabin_weights)) if cabin_weights.exists() else None

    all_jpgs = sorted(raw_images_dir.glob("*.jpg"))
    logger.info("Discovered %d raw frames in %s", len(all_jpgs), raw_images_dir)

    # 1. Run detection over all frames in batches to categorize positive vs snow background
    positive_records: List[Tuple[Path, Any, Any]] = []
    cam_backgrounds: Dict[int, List[Path]] = defaultdict(list)

    batch_size = 32
    logger.info("Scanning frames on %s for vehicle visibility...", device)

    for i in tqdm(range(0, len(all_jpgs), batch_size), desc="Detecting targets", unit="batch"):
        batch_paths = all_jpgs[i : i + batch_size]
        str_paths = [str(p) for p in batch_paths]
        seg_res = seg_model.predict(str_paths, conf=conf_threshold, device=device, verbose=False)

        for p, r_seg in zip(batch_paths, seg_res):
            cam_str = p.name.split("_")[1].replace("cam", "")
            cam_id = int(cam_str) if cam_str.isdigit() else 1

            if r_seg and len(r_seg.boxes) > 0 and r_seg.masks is not None:
                positive_records.append((p, r_seg, cam_id))
            else:
                cam_backgrounds[cam_id].append(p)

    logger.info(
        "Scan results: %d positive frames with vehicle | %d pure snow background frames.",
        len(positive_records),
        sum(len(v) for v in cam_backgrounds.values()),
    )

    # 2. Select diverse snow background negatives across all 9 cameras
    # Distribute target_snow_negatives evenly across available cameras
    selected_negatives: List[Path] = []
    num_cams = len(cam_backgrounds)
    neg_per_cam = max(1, target_snow_negatives // max(1, num_cams))

    for cid in sorted(cam_backgrounds.keys()):
        bg_list = cam_backgrounds[cid]
        if not bg_list:
            continue
        # Sample uniformly across the timeline (beginning, middle, end)
        indices = np.linspace(0, len(bg_list) - 1, min(len(bg_list), neg_per_cam), dtype=int)
        for idx in indices:
            selected_negatives.append(bg_list[idx])

    logger.info("Selected %d diverse snow background negative frames across all cameras.", len(selected_negatives))

    # 3. Process positive frames: generate AI pre-annotated polygons and cabin bboxes
    logger.info("Generating AI pre-annotated polygons for %d positive frames...", len(positive_records))
    positive_saved = 0

    for img_path, r_seg, cam_id in tqdm(positive_records, desc="Pre-annotating positives", unit="img"):
        dest_img = output_img_dir / img_path.name
        dest_json = output_img_dir / f"{img_path.stem}.json"

        # Copy frame
        shutil.copy2(img_path, dest_img)

        img_h, img_w = r_seg.orig_shape
        shapes: List[Dict[str, Any]] = []

        # A. Truck instance segmentation polygon
        # Pick the largest mask if multiple detections
        best_mask_idx = int(torch.argmax(r_seg.boxes.conf).item())
        mask_xy = r_seg.masks.xy[best_mask_idx]
        if len(mask_xy) >= 3:
            approx_pts = approximate_polygon(mask_xy, epsilon=1.8)
            shapes.append({
                "label": "truck",
                "text": "",
                "points": approx_pts,
                "group_id": None,
                "description": "AI-predicted truck polygon (YOLO26-seg)",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            })

        # B. Cabin bounding box (predict with cabin model)
        if cabin_model is not None:
            c_res = cabin_model.predict(str(img_path), conf=0.18, device=device, verbose=False)
            if c_res and len(c_res[0].boxes) > 0:
                best_c = int(torch.argmax(c_res[0].boxes.conf).item())
                cb = c_res[0].boxes.xyxy[best_c].cpu().numpy()
                cx1, cy1, cx2, cy2 = float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3])
                shapes.append({
                    "label": "cabin",
                    "text": "",
                    "points": [[cx1, cy1], [cx2, cy2]],
                    "group_id": None,
                    "description": "AI-predicted cabin bbox",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None,
                })

        json_data = create_anylabeling_json(img_path.name, img_w, img_h, shapes)
        with open(dest_json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        positive_saved += 1

    # 4. Process snow background frames: write empty AnyLabeling JSON
    logger.info("Writing empty annotations for %d snow background frames...", len(selected_negatives))
    for bg_path in selected_negatives:
        dest_img = output_img_dir / bg_path.name
        dest_json = output_img_dir / f"{bg_path.stem}.json"

        shutil.copy2(bg_path, dest_img)
        # Empty shapes list teaches YOLO that this image contains zero target objects
        json_data = create_anylabeling_json(bg_path.name, 1920, 1080, [])
        with open(dest_json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

    total_curated = positive_saved + len(selected_negatives)

    # 5. Write README.md for the annotator
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            f"# Scene 2 Curated Dataset for AnyLabeling\n\n"
            f"- **Total Frames:** {total_curated}\n"
            f"- **Positive Frames (Truck Visible):** {positive_saved} (Pre-annotated with AI polygons)\n"
            f"- **Snow Background Negatives (Pure Snow/Landscape):** {len(selected_negatives)} (Pre-filled with empty JSON)\n"
            f"- **Target Classes:**\n"
            f"  - `truck`: Full vehicle instance polygon (for YOLO26-seg)\n"
            f"  - `cabin`: Front vehicle cabin box (for Cabin Detector)\n\n"
            f"## How to Annotate / Verify in AnyLabeling\n"
            f"1. Open **AnyLabeling** (or X-AnyLabeling).\n"
            f"2. Select `Open Directory` -> `{output_img_dir.resolve()}`.\n"
            f"3. Scroll through frames using `D` (Next) and `A` (Prev):\n"
            f"   - On **positive frames**: Verify the green `truck` polygon and yellow `cabin` box. Adjust vertices if occluded by trees/posts.\n"
            f"   - On **snow background frames**: Verify they remain empty (no false positive boxes).\n"
            f"4. Save with `Ctrl+S`.\n"
            f"5. Once verified, run `python scripts/dataset/build_dual_dataset.py` to compile both final datasets!\n"
        )

    logger.info(
        "Successfully created curated annotation project at %s: %d total frames (%d positive, %d snow background).",
        output_dir,
        total_curated,
        positive_saved,
        len(selected_negatives),
    )

    return {
        "output_dir": str(output_dir),
        "total_curated": total_curated,
        "positive_count": positive_saved,
        "snow_background_count": len(selected_negatives),
    }


def main():
    parser = argparse.ArgumentParser(description="Curate Scene 2 frames with AI pre-annotations and snow background.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/datasets/scen_2_raw/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/datasets/scen_2_curated"))
    parser.add_argument("--seg-weights", type=Path, default=Path("weights/finetuned/peleng_yolo26x_seg_clean_polish_best.pt"))
    parser.add_argument("--cabin-weights", type=Path, default=Path("weights/cabin/peleng_cabin_best.pt"))
    parser.add_argument("--snow-negatives", type=int, default=110, help="Number of snow background negative frames")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    curate_and_preannotate(
        raw_images_dir=args.raw_dir,
        output_dir=args.output_dir,
        seg_weights=args.seg_weights,
        cabin_weights=args.cabin_weights,
        target_snow_negatives=args.snow_negatives,
        conf_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
