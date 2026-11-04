"""
Build Dual Dataset for YOLO26-seg (Polygons) and Cabin Detector (Bounding Boxes).

Takes verified/pre-annotated AnyLabeling JSON annotations from Scene 2
and combines them with Scene 1 (Dataset_Raw for segmentation and scen_1_cabin for GT cabins)
to produce two ready-to-train Ultralytics datasets:
1. data/datasets/peleng_yolo26_seg_combined (Instance Segmentation: class 'truck')
2. data/datasets/peleng_cabin_combined (Object Detection: class 'cabin')
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ultralytics.data.converter import merge_multi_segment

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("DualDatasetBuilder")


def build_dual_datasets(
    scen2_dir: Path = Path("data/datasets/scen_2_curated/images"),
    scen1_seg_dir: Optional[Path] = Path("data/datasets/Dataset_Raw/images/train"),
    scen1_cabin_dir: Optional[Path] = Path("data/datasets/scen_1_cabin"),
    output_seg_dir: Path = Path("data/datasets/peleng_yolo26_seg_combined"),
    output_cabin_dir: Path = Path("data/datasets/peleng_cabin_combined"),
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compile both YOLO26-seg (polygons) and Cabin (bboxes) datasets with stratified train/val split."""
    random.seed(seed)
    np.random.seed(seed)

    # 1. Clean and initialize output directories
    for base in [output_seg_dir, output_cabin_dir]:
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            p = base / sub
            p.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # PART A: Build YOLO26-seg (Truck Instance Segmentation)
    # Sources: Scene 1 (Dataset_Raw) + Scene 2 (scen_2_curated)
    # ---------------------------------------------------------
    logger.info("=== BUILDING YOLO26-SEG COMBINED DATASET ===")
    seg_samples: List[Tuple[Path, Optional[Path], bool, str]] = []  # (img, json, is_bg, cam_group)

    # Scene 2 frames
    if scen2_dir.exists():
        s2_jpgs = sorted(scen2_dir.glob("*.jpg"))
        s2_jsons = {j.stem: j for j in scen2_dir.glob("*.json")}
        for img in s2_jpgs:
            jf = s2_jsons.get(img.stem)
            is_bg = True
            if jf and jf.exists():
                with open(jf, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if any(s.get("label") == "truck" for s in d.get("shapes", [])):
                    is_bg = False
            cam_str = img.stem.split("_f")[0] if "_f" in img.stem else "s2_unknown"
            seg_samples.append((img, jf, is_bg, cam_str))

    # Scene 1 frames
    if scen1_seg_dir and scen1_seg_dir.exists():
        s1_jpgs = sorted(scen1_seg_dir.glob("*.jpg"))
        s1_jsons = {j.stem: j for j in scen1_seg_dir.glob("*.json")}
        for img in s1_jpgs:
            jf = s1_jsons.get(img.stem)
            is_bg = img.stem.startswith("cam0_bg") or (jf is None)
            cam_str = img.stem.split("_")[0] if "_" in img.stem else "s1_unknown"
            seg_samples.append((img, jf, is_bg, cam_str))

    logger.info("YOLO26-seg: collected %d total candidate images across Scene 1 and Scene 2", len(seg_samples))

    # Stratified split for YOLO26-seg
    seg_cams_pos = defaultdict(list)
    seg_bg = []
    for s in seg_samples:
        if s[2]:  # is_bg
            seg_bg.append(s)
        else:
            seg_cams_pos[s[3]].append(s)

    seg_train = []
    seg_val = []
    for cam, items in sorted(seg_cams_pos.items()):
        random.shuffle(items)
        nv = max(1, int(round(len(items) * val_ratio)))
        seg_val.extend(items[:nv])
        seg_train.extend(items[nv:])

    random.shuffle(seg_bg)
    nv_bg = max(1, int(round(len(seg_bg) * val_ratio)))
    seg_val.extend(seg_bg[:nv_bg])
    seg_train.extend(seg_bg[nv_bg:])

    def write_seg_split(items: List[Tuple[Path, Optional[Path], bool, str]], split_name: str) -> int:
        count = 0
        for img_path, json_path, is_bg, _ in items:
            stem = img_path.name
            shutil.copy2(img_path, output_seg_dir / "images" / split_name / stem)
            lbl_file = output_seg_dir / "labels" / split_name / f"{img_path.stem}.txt"

            if is_bg or not json_path or not json_path.exists():
                lbl_file.write_text("", encoding="utf-8")
                continue

            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            img_w = float(data.get("imageWidth", 1920))
            img_h = float(data.get("imageHeight", 1080))
            polygons = []
            for shape in data.get("shapes", []):
                if shape.get("label") == "truck" and shape.get("shape_type") == "polygon":
                    pts = shape.get("points", [])
                    if len(pts) >= 3:
                        polygons.append(pts)

            if not polygons:
                lbl_file.write_text("", encoding="utf-8")
            elif len(polygons) == 1:
                coords = [f"{min(1.0, max(0.0, float(pt[0])/img_w)):.6f} {min(1.0, max(0.0, float(pt[1])/img_h)):.6f}" for pt in polygons[0]]
                lbl_file.write_text(f"0 {' '.join(coords)}\n", encoding="utf-8")
                count += 1
            else:
                s = merge_multi_segment(polygons)
                merged_pts = np.concatenate(s, axis=0)
                coords = [f"{min(1.0, max(0.0, float(pt[0])/img_w)):.6f} {min(1.0, max(0.0, float(pt[1])/img_h)):.6f}" for pt in merged_pts]
                lbl_file.write_text(f"0 {' '.join(coords)}\n", encoding="utf-8")
                count += 1
        return count

    n_seg_tr = write_seg_split(seg_train, "train")
    n_seg_va = write_seg_split(seg_val, "val")
    seg_yaml = f"""# Peleng Combined YOLO26 Instance Segmentation Dataset (Scene 1 + Scene 2)
path: {output_seg_dir.resolve().as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: truck
"""
    (output_seg_dir / "data.yaml").write_text(seg_yaml, encoding="utf-8")
    logger.info("YOLO26-seg built: %d train (%d pos), %d val (%d pos)", len(seg_train), n_seg_tr, len(seg_val), n_seg_va)

    # ---------------------------------------------------------
    # PART B: Build Cabin Detector Dataset
    # Sources: Scene 1 GT (scen_1_cabin) + Scene 2 (scen_2_curated)
    # ---------------------------------------------------------
    logger.info("\n=== BUILDING CABIN DETECTOR COMBINED DATASET ===")
    cab_samples: List[Tuple[Path, Optional[Path], Optional[str], bool, str]] = [] # (img, json, raw_txt_label, is_bg, cam_group)

    # 1. Scene 1 GT cabin samples
    if scen1_cabin_dir and scen1_cabin_dir.exists():
        for sub in ["train", "val"]:
            sub_imgs = list((scen1_cabin_dir / "images" / sub).glob("*.jpg"))
            for img in sub_imgs:
                lbl = scen1_cabin_dir / "labels" / sub / f"{img.stem}.txt"
                txt_content = lbl.read_text(encoding="utf-8").strip() if lbl.exists() else None
                cam_str = img.stem.split("_")[0] if "_" in img.stem else "s1_unknown"
                cab_samples.append((img, None, txt_content, False, cam_str))

    # 2. Scene 2 curated cabin samples (both annotated and snow backgrounds)
    if scen2_dir.exists():
        s2_jpgs = sorted(scen2_dir.glob("*.jpg"))
        s2_jsons = {j.stem: j for j in scen2_dir.glob("*.json")}
        for img in s2_jpgs:
            jf = s2_jsons.get(img.stem)
            is_bg = True
            cab_bbox = None
            if jf and jf.exists():
                with open(jf, "r", encoding="utf-8") as f:
                    d = json.load(f)
                shapes = d.get("shapes", [])
                for sh in shapes:
                    if sh.get("label") == "cabin":
                        is_bg = False
                        break
            cam_str = img.stem.split("_f")[0] if "_f" in img.stem else "s2_unknown"
            cab_samples.append((img, jf, None, is_bg, cam_str))

    logger.info("Cabin: collected %d total candidate images across Scene 1 GT and Scene 2", len(cab_samples))

    # Stratified split for Cabin
    cab_cams_pos = defaultdict(list)
    cab_bg = []
    for s in cab_samples:
        if s[3]:  # is_bg
            cab_bg.append(s)
        else:
            cab_cams_pos[s[4]].append(s)

    cab_train = []
    cab_val = []
    for cam, items in sorted(cab_cams_pos.items()):
        random.shuffle(items)
        nv = max(1, int(round(len(items) * val_ratio)))
        cab_val.extend(items[:nv])
        cab_train.extend(items[nv:])

    random.shuffle(cab_bg)
    nv_bg = max(1, int(round(len(cab_bg) * val_ratio)))
    cab_val.extend(cab_bg[:nv_bg])
    cab_train.extend(cab_bg[nv_bg:])

    def write_cab_split(items: List[Tuple[Path, Optional[Path], Optional[str], bool, str]], split_name: str) -> int:
        count = 0
        for img_path, json_path, txt_content, is_bg, _ in items:
            stem = img_path.name
            shutil.copy2(img_path, output_cabin_dir / "images" / split_name / stem)
            lbl_file = output_cabin_dir / "labels" / split_name / f"{img_path.stem}.txt"

            if is_bg:
                lbl_file.write_text("", encoding="utf-8")
                continue

            if txt_content:
                lbl_file.write_text(txt_content + "\n", encoding="utf-8")
                count += 1
                continue

            if json_path and json_path.exists():
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                img_w = float(data.get("imageWidth", 1920))
                img_h = float(data.get("imageHeight", 1080))
                cabin_lines = []
                for shape in data.get("shapes", []):
                    if shape.get("label") != "cabin":
                        continue
                    pts = shape.get("points", [])
                    if shape.get("shape_type") == "rectangle" and len(pts) == 2:
                        x1, y1 = float(pts[0][0]), float(pts[0][1])
                        x2, y2 = float(pts[1][0]), float(pts[1][1])
                        xmin, xmax = min(x1, x2), max(x1, x2)
                        ymin, ymax = min(y1, y2), max(y1, y2)
                    elif shape.get("shape_type") == "polygon" and len(pts) >= 3:
                        xs = [float(p[0]) for p in pts]
                        ys = [float(p[1]) for p in pts]
                        xmin, xmax = min(xs), max(xs)
                        ymin, ymax = min(ys), max(ys)
                    else:
                        continue
                    w_box = xmax - xmin
                    h_box = ymax - ymin
                    if w_box >= 6.0 and h_box >= 6.0:
                        cx = ((xmin + xmax) / 2.0) / img_w
                        cy = ((ymin + ymax) / 2.0) / img_h
                        nw = w_box / img_w
                        nh = h_box / img_h
                        cabin_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                if cabin_lines:
                    lbl_file.write_text("\n".join(cabin_lines) + "\n", encoding="utf-8")
                    count += len(cabin_lines)
                else:
                    lbl_file.write_text("", encoding="utf-8")
            else:
                lbl_file.write_text("", encoding="utf-8")
        return count

    n_cab_tr = write_cab_split(cab_train, "train")
    n_cab_va = write_cab_split(cab_val, "val")
    cab_yaml = f"""# Peleng Combined Cabin Detector Dataset (Scene 1 + Scene 2)
path: {output_cabin_dir.resolve().as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: cabin
"""
    (output_cabin_dir / "data.yaml").write_text(cab_yaml, encoding="utf-8")
    logger.info("Cabin built: %d train (%d pos), %d val (%d pos)", len(cab_train), n_cab_tr, len(cab_val), n_cab_va)

    return {
        "yolo26_seg": {"train": len(seg_train), "val": len(seg_val), "positives": n_seg_tr + n_seg_va},
        "cabin": {"train": len(cab_train), "val": len(cab_val), "positives": n_cab_tr + n_cab_va},
    }


def main():
    parser = argparse.ArgumentParser(description="Compile dual datasets from verified AnyLabeling annotations.")
    parser.add_argument("--scen2-dir", type=Path, default=Path("data/datasets/scen_2_curated/images"))
    parser.add_argument("--scen1-seg", type=Path, default=Path("data/datasets/Dataset_Raw/images/train"))
    parser.add_argument("--scen1-cabin", type=Path, default=Path("data/datasets/scen_1_cabin"))
    parser.add_argument("--out-seg", type=Path, default=Path("data/datasets/peleng_yolo26_seg_combined"))
    parser.add_argument("--out-cabin", type=Path, default=Path("data/datasets/peleng_cabin_combined"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    build_dual_datasets(
        scen2_dir=args.scen2_dir,
        scen1_seg_dir=args.scen1_seg,
        scen1_cabin_dir=args.scen1_cabin,
        output_seg_dir=args.out_seg,
        output_cabin_dir=args.out_cabin,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
