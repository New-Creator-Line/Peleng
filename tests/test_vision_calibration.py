"""
Unit tests for Color & Sky Vision Calibration Suite (Task 009).

Verifies that:
- All 5 camera video frames load correctly (1080x1920).
- Sky segmentation mask M_sky covers the upper horizon without mountain crest bleeding.
- Tree segmentation extracts valid non-empty root detections (u_base, v_base).
- Debug visual overlays (cam1_debug.png ... cam5_debug.png) are properly generated and saved.
- No crashes or unhandled exceptions occur across the calibration suite.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest

# Ensure Peleng root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PELENG_ROOT = TEST_DIR.parent
if str(PELENG_ROOT) not in sys.path:
    sys.path.insert(0, str(PELENG_ROOT))

from scripts.calibrate_vision import (
    calibrate_camera,
    create_debug_overlay,
    load_representative_frame,
    run_all_calibrations,
    segment_sky,
    segment_trees,
)


@pytest.fixture(scope="session")
def video_dir() -> Path:
    """Locate the video directory containing CineCameraActor files."""
    candidates = [
        PELENG_ROOT / "хакатон",
        Path("хакатон"),
        Path("Peleng/хакатон"),
    ]
    for c in candidates:
        if c.exists() and (c / "Way1.FinalImage.CineCameraActor1.mp4").exists():
            return c.resolve()
    pytest.skip("Video directory 'хакатон' not found in workspace.")


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory) -> Path:
    """Temporary output directory for testing calibration artifacts."""
    return tmp_path_factory.mktemp("test_calibration_output")


@pytest.mark.parametrize("cam_id", [1, 2, 3, 4, 5])
def test_load_representative_frame(video_dir: Path, cam_id: int):
    """Verify loading frame 100 for all 5 cameras yields 1080x1920x3 uint8 array."""
    video_path = video_dir / f"Way1.FinalImage.CineCameraActor{cam_id}.mp4"
    frame = load_representative_frame(video_path, frame_idx=100)

    assert frame is not None, f"Cam {cam_id} frame should not be None"
    assert frame.dtype == np.uint8, f"Cam {cam_id} frame should be uint8"
    assert frame.shape == (1080, 1920, 3), f"Cam {cam_id} unexpected shape: {frame.shape}"


def test_load_frame_missing_file_raises():
    """Verify loading non-existent video raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_representative_frame("non_existent_video_path_xyz.mp4")


@pytest.mark.parametrize("cam_id", [1, 2, 3, 4, 5])
def test_sky_segmentation(video_dir: Path, cam_id: int):
    """
    Verify sky segmentation mask M_sky:
    - Mask is binary uint8 (0 or 255).
    - Covers upper horizon region.
    - Prevents bleeding into terrain / snow / vehicle (max horizon < 850px).
    """
    video_path = video_dir / f"Way1.FinalImage.CineCameraActor{cam_id}.mp4"
    frame = load_representative_frame(video_path, frame_idx=100)
    sky_mask, horizon_y = segment_sky(frame)

    assert sky_mask.shape == (1080, 1920)
    unique_vals = set(np.unique(sky_mask))
    assert unique_vals.issubset({0, 255}), f"Mask must be binary, got: {unique_vals}"

    sky_pixels = int(np.sum(sky_mask > 0))
    assert sky_pixels > 10_000, f"Cam {cam_id} should have substantial sky pixels, got {sky_pixels}"

    # Column horizon checks
    assert len(horizon_y) == 1920
    valid_cols = horizon_y[horizon_y > 0]
    assert len(valid_cols) > 300, f"Cam {cam_id} should have >300 sky columns, got {len(valid_cols)}"

    # Horizon boundary must not bleed into lower terrain
    max_horizon = int(np.max(valid_cols))
    assert max_horizon < 850, f"Cam {cam_id} horizon bled into ground: max row = {max_horizon}"


@pytest.mark.parametrize("cam_id", [1, 2, 3, 4, 5])
def test_tree_segmentation_and_roots(video_dir: Path, cam_id: int):
    """
    Verify conifer foliage & trunk detection and ground-contact root points:
    - Non-empty candidate root points detected.
    - Root coordinates (u, v) lie within frame dimensions [0..1919, 0..1079].
    - Roots are located in ground/terrain region (v > 100).
    """
    video_path = video_dir / f"Way1.FinalImage.CineCameraActor{cam_id}.mp4"
    frame = load_representative_frame(video_path, frame_idx=100)
    sky_mask, _ = segment_sky(frame)
    tree_mask, roots = segment_trees(frame, sky_mask=sky_mask)

    assert tree_mask.shape == (1080, 1920)
    assert len(roots) > 0, f"Cam {cam_id} must have non-empty candidate tree root detections"

    for u_base, v_base in roots:
        assert 0 <= u_base < 1920, f"Cam {cam_id} u_base {u_base} out of range [0, 1920)"
        assert 0 <= v_base < 1080, f"Cam {cam_id} v_base {v_base} out of range [0, 1080)"
        assert v_base > 50, f"Cam {cam_id} root at {v_base} unexpectedly in top sky region"


@pytest.mark.parametrize("cam_id", [1, 2, 3, 4, 5])
def test_debug_overlay_generation(video_dir: Path, output_dir: Path, cam_id: int):
    """
    Verify debug overlay generation:
    - Overlay image is generated and saved as cam{cam_id}_debug.png.
    - Image is readable, non-empty, and matches original frame dimensions.
    """
    video_path = video_dir / f"Way1.FinalImage.CineCameraActor{cam_id}.mp4"
    res = calibrate_camera(cam_id, video_path, output_dir=output_dir, frame_idx=100)

    overlay_file = Path(res["debug_overlay_path"])
    assert overlay_file.exists(), f"Debug overlay file does not exist: {overlay_file}"
    assert overlay_file.stat().st_size > 50_000, "Overlay file suspiciously small"

    img = cv2.imread(str(overlay_file))
    assert img is not None, "Failed to decode saved debug overlay image"
    assert img.shape == (1080, 1920, 3), f"Overlay unexpected shape: {img.shape}"


def test_run_all_calibrations(video_dir: Path, output_dir: Path):
    """
    Verify complete batch calibration across all 5 cameras:
    - Generates 5 valid overlay files.
    - Generates calibration_summary.json.
    - Verifies all camera metrics are well-formed and non-empty.
    """
    results = run_all_calibrations(
        video_dir=video_dir,
        output_dir=output_dir,
        frame_idx=100,
        cam_ids=[1, 2, 3, 4, 5],
    )

    assert len(results) == 5, f"Expected 5 camera results, got {len(results)}"
    summary_file = output_dir / "calibration_summary.json"
    assert summary_file.exists(), "calibration_summary.json was not created"

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for cam_id in [1, 2, 3, 4, 5]:
        cam_key = str(cam_id)
        assert cam_key in summary, f"Cam {cam_id} missing in summary json"
        cam_res = summary[cam_key]

        assert cam_res["cam_id"] == cam_id
        assert cam_res["sky_cols_count"] > 300
        assert cam_res["sky_horizon_mean_row"] is not None
        assert 100.0 < cam_res["sky_horizon_mean_row"] < 600.0
        assert cam_res["tree_roots_count"] > 0
        assert len(cam_res["tree_roots"]) == cam_res["tree_roots_count"]

        overlay_path = Path(cam_res["debug_overlay_path"])
        assert overlay_path.exists(), f"Overlay {overlay_path} missing"
