import os
import yaml
import pytest
from pathlib import Path


def test_data_yaml_structure_and_integrity():
    yaml_path = Path("dataset/peleng.v3i.yolov8/data.yaml")
    assert yaml_path.exists(), f"Dataset config {yaml_path} does not exist"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "train" in data
    assert "val" in data
    assert "nc" in data
    assert "names" in data
    assert data["nc"] == 1
    assert data["names"] == ["truck"]

    root_path = Path(data.get("path", "."))
    train_dir = root_path / data["train"]
    val_dir = root_path / data["val"]
    test_dir = root_path / data.get("test", "test/images")

    assert train_dir.exists(), f"Train dir {train_dir} not found"
    assert val_dir.exists(), f"Val dir {val_dir} not found"
    # test split is optional in v3

    train_imgs = list(train_dir.glob("*.jpg"))
    val_imgs = list(val_dir.glob("*.jpg"))
    test_imgs = list(test_dir.glob("*.jpg"))

    assert len(train_imgs) >= 300, f"Expected >= 300 train images, got {len(train_imgs)}"
    assert len(val_imgs) >= 20, f"Expected >= 20 val images, got {len(val_imgs)}"
    pass


def test_labels_validity():
    labels_dir = Path("dataset/peleng.v3i.yolov8/train/labels")
    assert labels_dir.exists()

    label_files = list(labels_dir.glob("*.txt"))
    assert len(label_files) > 0

    checked = 0
    for lf in label_files[:50]:  # sample check
        with open(lf, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            assert len(parts) == 5, f"Invalid label format in {lf.name}: {line}"
            cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            assert cls_id == 0, f"Unexpected class ID {cls_id}"
            assert 0.0 <= x <= 1.0, f"x out of bounds {x} in {lf.name}"
            assert 0.0 <= y <= 1.0, f"y out of bounds {y} in {lf.name}"
            assert 0.0 < w <= 1.0, f"w out of bounds {w} in {lf.name}"
            assert 0.0 < h <= 1.0, f"h out of bounds {h} in {lf.name}"
            checked += 1

    assert checked > 0
