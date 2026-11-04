"""
Stratified Group Split for Peleng YOLOv8 v3 Dataset.

Splits 150 unique source frames (450 augmented images) into train/val
without data leakage between augmented variants.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set

# Optimal validation set (23 unique frames, 15.3% of dataset)
RECOMMENDED_VAL_FRAMES: Set[str] = {
    # CAM0: Negative background control
    "cam0_bg_00003_jpg",
    # CAM1: Lateral drive (left and right coverage)
    "cam1_f00003_jpg",
    "cam1_f00009_jpg",
    # CAM2: Full longitudinal trajectory (near -> horizon)
    "cam2_f00004_jpg",
    "cam2_f00011_jpg",
    "cam2_f00018_jpg",
    "cam2_f00024_jpg",
    "cam2_f00030_jpg",
    # CAM3: Right entrance to far left
    "cam3_f00003_jpg",
    "cam3_f00010_jpg",
    "cam3_f00018_jpg",
    "cam3_f00028_jpg",
    # CAM4: Far approach to near pass
    "cam4_f00004_jpg",
    "cam4_f00012_jpg",
    "cam4_f00018_jpg",
    "cam4_f00029_jpg",
    # CAM5: Dynamic rotation camera (extreme far, active track, close-up, exit)
    "cam5_f00002_jpg",
    "cam5_f00010_jpg",
    "cam5_f00021_jpg",
    "cam5_f00030_jpg",
    "cam5_f00038_jpg",
    "cam5_f00044_jpg",
    "cam5_f00051_jpg",
}


def split_dataset(
    dataset_dir: str = "data/datasets/yolo_v3",
    val_frames: Set[str] = RECOMMENDED_VAL_FRAMES,
    dry_run: bool = False,
) -> Dict[str, int]:
    root = Path(dataset_dir)
    train_img_dir = root / "train" / "images"
    train_lbl_dir = root / "train" / "labels"
    val_img_dir = root / "valid" / "images"
    val_lbl_dir = root / "valid" / "labels"

    if not dry_run:
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)

    moved_images = 0
    moved_labels = 0

    all_images = list(train_img_dir.glob("*.jpg"))
    for img_path in all_images:
        base_name = img_path.name.split(".rf.")[0]
        if base_name in val_frames:
            lbl_name = img_path.name.replace(".jpg", ".txt")
            lbl_path = train_lbl_dir / lbl_name

            if lbl_path.exists():
                if not dry_run:
                    shutil.move(str(lbl_path), str(val_lbl_dir / lbl_name))
                moved_labels += 1
            if not dry_run:
                shutil.move(str(img_path), str(val_img_dir / img_path.name))
            moved_images += 1

    # Update data.yaml
    if not dry_run:
        yaml_path = root / "data.yaml"
        abs_root = root.resolve().as_posix()
        new_yaml = f"""path: {abs_root}
train: train/images
val: valid/images

nc: 1
names: ['truck']
"""
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(new_yaml)

    return {
        "val_images": moved_images,
        "val_labels": moved_labels,
        "remaining_train": len(all_images) - moved_images,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset/peleng.v3i.yolov8")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    res = split_dataset(args.dataset_dir, dry_run=args.dry_run)
    print("Split preview:" if args.dry_run else "Split completed:")
    print(f"  Validation images: {res['val_images']}")
    print(f"  Validation labels: {res['val_labels']}")
    print(f"  Training images:   {res['remaining_train']}")
