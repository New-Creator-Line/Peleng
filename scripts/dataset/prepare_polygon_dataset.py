"""
Prepare and convert custom AnyLabeling polygon annotations to YOLOv8/YOLO11/YOLO26-seg format.

Workflow:
1. Converts 144 JSON polygons to normalized YOLO segmentation format (0 x1 y1 x2 y2 ...).
2. Sets up 6 true background negative images (cam0_bg_*.jpg) with empty .txt labels.
3. Isolates 22 ambiguous/low-visibility/edge-clipped frames into skipped_low_visibility/.
4. Performs reproducible stratified train/val split (84% train / 16% val) across all 5 cameras.
5. Generates canonical data.yaml with absolute Windows paths.
"""

import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ultralytics.data.converter import merge_multi_segment

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetPreparer")

RAW_IMG_DIR = Path("data/datasets/Dataset_Raw/images/train")
TARGET_DIR = Path("data/datasets/peleng_yolo_seg")


def prepare_dataset(
    raw_dir: Path = RAW_IMG_DIR,
    target_dir: Path = TARGET_DIR,
    val_ratio: float = 0.16,
    seed: int = 42,
) -> Dict[str, int]:
    random.seed(seed)

    # Output directories
    train_img_dir = target_dir / "images" / "train"
    val_img_dir = target_dir / "images" / "val"
    train_lbl_dir = target_dir / "labels" / "train"
    val_lbl_dir = target_dir / "labels" / "val"
    skipped_dir = target_dir / "skipped_low_visibility"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir, skipped_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Discover all files
    all_jpgs = sorted(raw_dir.glob("*.jpg"))
    all_jsons = sorted(raw_dir.glob("*.json"))
    json_map = {j.stem: j for j in all_jsons}

    logger.info("Found %d total JPGs and %d JSON annotations in %s", len(all_jpgs), len(all_jsons), raw_dir)

    positive_samples: List[Path] = []
    pure_bg_samples: List[Path] = []
    skipped_samples: List[Path] = []

    for img_path in all_jpgs:
        stem = img_path.stem
        if stem in json_map:
            positive_samples.append(img_path)
        elif stem.startswith("cam0_bg"):
            pure_bg_samples.append(img_path)
        else:
            skipped_samples.append(img_path)

    logger.info("Categorized: %d Positive, %d Pure Background, %d Skipped (low visibility)",
                len(positive_samples), len(pure_bg_samples), len(skipped_samples))

    # 2. Isolate skipped low-visibility samples
    for img_path in skipped_samples:
        shutil.copy2(img_path, skipped_dir / img_path.name)

    with open(skipped_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("These 22 images were skipped due to extreme low visibility (<20%) or edge truncation.\n"
                "They are isolated to prevent background poisoning (labeling truck pixels as empty background).\n")

    # 3. Stratified group split by camera
    cams_positive = defaultdict(list)
    for p in positive_samples:
        cam_prefix = p.stem.split("_")[0]
        cams_positive[cam_prefix].append(p)

    train_set: List[Tuple[Path, bool]] = []
    val_set: List[Tuple[Path, bool]] = []

    # Split positive samples per camera
    for cam, imgs in sorted(cams_positive.items()):
        random.shuffle(imgs)
        n_val = max(1, int(round(len(imgs) * val_ratio)))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]
        for img in train_imgs:
            train_set.append((img, False))
        for img in val_imgs:
            val_set.append((img, False))
        logger.info("  %s: %d train, %d val (total %d)", cam, len(train_imgs), len(val_imgs), len(imgs))

    # Split background samples
    random.shuffle(pure_bg_samples)
    bg_val = pure_bg_samples[:1]
    bg_train = pure_bg_samples[1:]
    for img in bg_train:
        train_set.append((img, True))
    for img in bg_val:
        val_set.append((img, True))
    logger.info("  cam0_bg: %d train, %d val (total %d)", len(bg_train), len(bg_val), len(pure_bg_samples))

    # 4. Convert and copy files
    def process_split(split_list: List[Tuple[Path, bool]], dest_img_dir: Path, dest_lbl_dir: Path):
        for img_path, is_bg in split_list:
            stem = img_path.stem
            # Copy image
            shutil.copy2(img_path, dest_img_dir / img_path.name)

            lbl_file = dest_lbl_dir / f"{stem}.txt"
            if is_bg:
                # Write empty txt for background
                lbl_file.write_text("", encoding="utf-8")
            else:
                json_path = json_map[stem]
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)

                img_w = float(data.get("imageWidth", 1920))
                img_h = float(data.get("imageHeight", 1080))

                polygons = []
                for shape in data.get("shapes", []):
                    if shape.get("label") != "truck" or shape.get("shape_type") != "polygon":
                        continue
                    pts = shape.get("points", [])
                    if len(pts) >= 3:
                        polygons.append(pts)

                if not polygons:
                    lbl_file.write_text("", encoding="utf-8")
                elif len(polygons) == 1:
                    pts = polygons[0]
                    coords = []
                    for pt in pts:
                        norm_x = min(1.0, max(0.0, float(pt[0]) / img_w))
                        norm_y = min(1.0, max(0.0, float(pt[1]) / img_h))
                        coords.append(f"{norm_x:.6f} {norm_y:.6f}")
                    lbl_file.write_text(f"0 {' '.join(coords)}\n", encoding="utf-8")
                else:
                    # Multiple disjoint polygons belonging to the single truck (tree/branch occlusions)
                    # Connect into a single continuous polygon using keyhole seams
                    s = merge_multi_segment(polygons)
                    merged_pts = np.concatenate(s, axis=0)
                    coords = []
                    for pt in merged_pts:
                        norm_x = min(1.0, max(0.0, float(pt[0]) / img_w))
                        norm_y = min(1.0, max(0.0, float(pt[1]) / img_h))
                        coords.append(f"{norm_x:.6f} {norm_y:.6f}")
                    lbl_file.write_text(f"0 {' '.join(coords)}\n", encoding="utf-8")

    logger.info("Processing train split (%d images)...", len(train_set))
    process_split(train_set, train_img_dir, train_lbl_dir)

    logger.info("Processing val split (%d images)...", len(val_set))
    process_split(val_set, val_img_dir, val_lbl_dir)

    # 5. Generate data.yaml
    abs_path = target_dir.resolve().as_posix()
    data_yaml_content = f"""# Peleng Multi-Camera YOLO Segmentation Dataset
path: {abs_path}
train: images/train
val: images/val

# Class configuration (Single class: truck)
nc: 1
names:
  0: truck
"""
    yaml_path = target_dir / "data.yaml"
    yaml_path.write_text(data_yaml_content, encoding="utf-8")
    logger.info("Generated %s successfully", yaml_path)

    stats = {
        "train_count": len(train_set),
        "val_count": len(val_set),
        "skipped_count": len(skipped_samples),
        "total_active": len(train_set) + len(val_set),
    }
    return stats


if __name__ == "__main__":
    stats = prepare_dataset()
    print("Dataset Preparation Summary:", stats)
