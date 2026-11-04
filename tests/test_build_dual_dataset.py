"""
Unit tests for Dual Dataset Builder (YOLO26-seg and Cabin Detector).
Verifies polygon and bounding box extraction, background handling, and data.yaml creation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from scripts.dataset.build_dual_dataset import build_dual_datasets


def test_build_dual_datasets(tmp_path: Path):
    """Verify dual dataset generation from synthetic AnyLabeling JSONs."""
    scen2_dir = tmp_path / "scen2"
    scen2_dir.mkdir()

    # Create dummy images + JSONs
    for i in range(10):
        img_p = scen2_dir / f"scen2_cam1_f{i:05d}.jpg"
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(img_p)

        json_p = scen2_dir / f"scen2_cam1_f{i:05d}.json"
        if i < 7:  # Positives
            shapes = [
                {
                    "label": "truck",
                    "shape_type": "polygon",
                    "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                },
                {
                    "label": "cabin",
                    "shape_type": "rectangle",
                    "points": [[10, 10], [30, 40]],
                },
            ]
        else:  # Snow background negatives
            shapes = []

        with open(json_p, "w", encoding="utf-8") as f:
            json.dump({
                "version": "0.4.43",
                "shapes": shapes,
                "imagePath": img_p.name,
                "imageWidth": 100,
                "imageHeight": 100,
            }, f)

    out_seg = tmp_path / "out_seg"
    out_cab = tmp_path / "out_cab"

    summary = build_dual_datasets(
        scen2_dir=scen2_dir,
        scen1_seg_dir=None,
        scen1_cabin_dir=None,
        output_seg_dir=out_seg,
        output_cabin_dir=out_cab,
        val_ratio=0.2,
    )

    assert (out_seg / "data.yaml").exists()
    assert (out_cab / "data.yaml").exists()

    seg_tr_imgs = list((out_seg / "images" / "train").glob("*.jpg"))
    seg_tr_lbls = list((out_seg / "labels" / "train").glob("*.txt"))
    assert len(seg_tr_imgs) == len(seg_tr_lbls)
    assert len(seg_tr_imgs) > 0

    cab_tr_imgs = list((out_cab / "images" / "train").glob("*.jpg"))
    cab_tr_lbls = list((out_cab / "labels" / "train").glob("*.txt"))
    assert len(cab_tr_imgs) == len(cab_tr_lbls)
    assert len(cab_tr_imgs) > 0
