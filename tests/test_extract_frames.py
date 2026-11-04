"""
Unit and integration tests for the video frame extraction pipeline.
Verifies camera ID parsing, frame indexing, non-destructive read-only safety,
and manifest generation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.dataset.extract_frames import (
    parse_camera_id,
    discover_video_files,
    extract_frames_from_video,
    ExtractionConfig,
    extract_dataset,
    resolve_output_directories,
)


@pytest.mark.parametrize(
    "filename, expected_id",
    [
        ("Way2.FinalImage.CineCameraActor1.mp4", 1),
        ("Way2.FinalImage.CineCameraActor9.mp4", 9),
        ("CineCameraActor5.mp4", 5),
        ("camera_3.mp4", 3),
        ("cam_04.mp4", 4),
        ("cam7.mp4", 7),
        ("unrelated_file.mp4", None),
        ("12.mp4", 12),
    ],
)
def test_parse_camera_id(filename: str, expected_id: int | None):
    """Verify robust camera ID detection across various naming conventions."""
    assert parse_camera_id(filename) == expected_id


def test_resolve_output_directories(tmp_path: Path):
    """Verify output directory routing for dataset root vs images."""
    base_dir = tmp_path / "my_dataset"
    root, img, lbl = resolve_output_directories(base_dir)
    assert root == base_dir
    assert img == base_dir / "images"
    assert lbl == base_dir / "labels"

    # When pointing directly to images folder
    images_dir = tmp_path / "my_dataset" / "images"
    root2, img2, lbl2 = resolve_output_directories(images_dir)
    assert root2 == tmp_path / "my_dataset"
    assert img2 == images_dir


def test_discover_video_files(tmp_path: Path):
    """Verify discovery, sorting and filtering of camera videos."""
    scen2_dir = Path("data/raw/scen_2")
    if scen2_dir.exists():
        mp4_files = list(scen2_dir.glob("*.mp4"))
        if len(mp4_files) >= 9:
            videos = discover_video_files(scen2_dir)
            assert len(videos) == 9
            cam_ids = [vid.camera_id for vid in videos]
            assert sorted(cam_ids) == list(range(1, 10))

            # Test filtering by specific cameras
            filtered = discover_video_files(scen2_dir, camera_ids=[1, 5, 9])
            assert len(filtered) == 3
            assert [v.camera_id for v in filtered] == [1, 5, 9]
            return

    # Fallback synthetic test for CI environments without raw video assets
    dummy_dir = tmp_path / "dummy_cams"
    dummy_dir.mkdir()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for cid in [3, 1, 2]:
        vfile = dummy_dir / f"CineCameraActor{cid}.mp4"
        out = cv2.VideoWriter(str(vfile), fourcc, 30.0, (64, 48))
        out.write(np.zeros((48, 64, 3), dtype=np.uint8))
        out.release()

    discovered = discover_video_files(dummy_dir)
    assert len(discovered) == 3
    assert [v.camera_id for v in discovered] == [1, 2, 3]


def test_config_validation(tmp_path: Path):
    """Verify boundary conditions and defensive validation in ExtractionConfig."""
    # Invalid step
    with pytest.raises(ValueError, match="Step must be a positive integer"):
        ExtractionConfig(input_dir=tmp_path, output_dir=tmp_path / "out", step=0)

    # Invalid fps
    with pytest.raises(ValueError, match="Target FPS must be a positive float"):
        ExtractionConfig(input_dir=tmp_path, output_dir=tmp_path / "out", fps=-1.0)

    # Negative start frame
    with pytest.raises(ValueError, match="start_frame cannot be negative"):
        ExtractionConfig(input_dir=tmp_path, output_dir=tmp_path / "out", start_frame=-5)

    # end_frame < start_frame
    with pytest.raises(ValueError, match="end_frame .* cannot be smaller"):
        ExtractionConfig(input_dir=tmp_path, output_dir=tmp_path / "out", start_frame=10, end_frame=5)


def test_collision_guard(tmp_path: Path):
    """Verify that writing to the raw input directory raises ValueError."""
    cfg = ExtractionConfig(input_dir=tmp_path, output_dir=tmp_path)
    # Create a dummy video in tmp_path so discovery succeeds
    vfile = tmp_path / "CineCameraActor1.mp4"
    out = cv2.VideoWriter(str(vfile), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    out.write(np.zeros((48, 64, 3), dtype=np.uint8))
    out.release()

    with pytest.raises(ValueError, match="Target output directory .* cannot be identical"):
        extract_dataset(cfg)


def test_dry_run_creates_no_directories(tmp_path: Path):
    """Verify that dry_run does not create output directories on disk."""
    vdir = tmp_path / "videos"
    vdir.mkdir()
    out_dir = tmp_path / "will_not_be_created"

    vfile = vdir / "CineCameraActor1.mp4"
    out = cv2.VideoWriter(str(vfile), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    out.write(np.zeros((48, 64, 3), dtype=np.uint8))
    out.release()

    cfg = ExtractionConfig(
        input_dir=vdir,
        output_dir=out_dir,
        step=1,
        dry_run=True,
    )
    res = extract_dataset(cfg)
    assert res["total_extracted"] == 1
    assert not (out_dir / "images").exists()
    assert not (out_dir / "manifest.csv").exists()


def test_read_only_video_safety(tmp_path: Path):
    """
    Verify that video extraction strictly reads the input video and NEVER
    modifies its bytes, size, or hash.
    """
    # Create a synthetic dummy video
    test_video_path = tmp_path / "test_video_CineCameraActor1.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(test_video_path), fourcc, 30.0, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), i * 8, dtype=np.uint8)
        out.write(frame)
    out.release()

    with open(test_video_path, "rb") as f:
        original_hash = hashlib.sha256(f.read()).hexdigest()
    original_size = test_video_path.stat().st_size
    original_mtime = test_video_path.stat().st_mtime

    output_dir = tmp_path / "output_frames"
    cfg = ExtractionConfig(
        input_dir=tmp_path,
        output_dir=output_dir,
        step=5,
        prefix="test_cam{cam}_f{frame:05d}",
        image_format="jpg",
        quality=95,
    )

    records = extract_frames_from_video(test_video_path, 1, cfg)

    # 1. Verify original video was completely untouched
    with open(test_video_path, "rb") as f:
        after_hash = hashlib.sha256(f.read()).hexdigest()
    assert after_hash == original_hash
    assert test_video_path.stat().st_size == original_size
    assert test_video_path.stat().st_mtime == original_mtime

    # 2. Verify extracted frames: 30 frames with step 5 -> frames 0, 5, 10, 15, 20, 25 (6 frames)
    assert len(records) == 6
    extracted_files = list(output_dir.glob("*.jpg"))
    assert len(extracted_files) == 6
    assert (output_dir / "test_cam1_f00000.jpg").exists()
    assert (output_dir / "test_cam1_f00025.jpg").exists()


def test_extract_dataset_workflow(tmp_path: Path):
    """Verify high-level extract_dataset generates images and manifest.csv."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    out_dir = tmp_path / "dataset"

    # Create 2 small test videos
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for cam_id in [1, 2]:
        vpath = video_dir / f"test_CineCameraActor{cam_id}.mp4"
        out = cv2.VideoWriter(str(vpath), fourcc, 30.0, (160, 120))
        for _ in range(15):
            out.write(np.zeros((120, 160, 3), dtype=np.uint8))
        out.release()

    cfg = ExtractionConfig(
        input_dir=video_dir,
        output_dir=out_dir,
        step=5,
        prefix="cam{cam}_f{frame:05d}",
        workers=1,
    )

    summary = extract_dataset(cfg)
    assert summary["total_extracted"] == 6  # 3 frames * 2 videos
    assert summary["total_videos"] == 2
    assert (out_dir / "manifest.csv").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "README.md").exists()
    assert (out_dir / "images" / "cam1_f00000.jpg").exists()
