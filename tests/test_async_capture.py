from pathlib import Path
import numpy as np
import pytest
import torch

from src.async_capture import (
    AsyncMultiCameraCapture,
    NVDEC_STATUS,
    assemble_multiview_grid,
    compute_dashboard_grid_dims,
    configure_nvdec_environment,
    discover_camera_videos,
)
from src.camera_model import load_all_cameras


def test_nvdec_environment_configuration():
    """Verify NVIDIA Video Codec SDK 13.1.24 NVDEC configuration and hardware flags."""
    caps = configure_nvdec_environment(enable_cuvid=True)
    assert isinstance(caps, dict)
    assert caps["backend"] == "NVIDIA_NVDEC_H264_CUVID"
    assert NVDEC_STATUS["backend"] == "NVIDIA_NVDEC_H264_CUVID"


def test_discover_n_camera_videos_and_calibrations(tmp_path: Path):
    """Verify dynamic discovery of arbitrary N >= 1 camera videos and calibration CSVs."""
    for c_id in [1, 2, 4, 7, 9, 12]:
        (tmp_path / f"stream_cam_{c_id}.mp4").write_bytes(b"dummy")
    discovered = discover_camera_videos(str(tmp_path))
    assert sorted(discovered.keys()) == [1, 2, 4, 7, 9, 12]

    # Also test dynamic dashboard grid assembly for N = 2, 3, 5, 8 cameras
    for n_cams in [2, 3, 5, 8]:
        cols, rows, w, h = compute_dashboard_grid_dims(n_cams, include_telemetry=True, tile_w=160, tile_h=90)
        assert cols * rows >= n_cams + 1
        tiles = {i: np.zeros((90, 160, 3), dtype=np.uint8) for i in range(1, n_cams + 1)}
        telemetry = np.ones((90, 160, 3), dtype=np.uint8)
        grid = assemble_multiview_grid(tiles, telemetry, tile_w=160, tile_h=90)
        assert grid.shape == (h, w, 3)


def test_async_multi_camera_capture_lifecycle():
    """Verify clean startup, synchronized NVDEC frame retrieval, and shutdown."""
    video_dir = Path("data/raw/hackathon")
    if not video_dir.exists():
        pytest.skip("Hackathon videos directory not found on disk")

    cam_videos = discover_camera_videos(str(video_dir))
    if not cam_videos:
        pytest.skip("Hackathon videos not found on disk")

    with AsyncMultiCameraCapture(cam_videos, queue_size=3, use_nvdec=True) as multi_cap:
        assert multi_cap.fps > 0
        assert multi_cap.width == 1920
        assert multi_cap.height == 1080
        assert multi_cap.nvdec_info["backend"] == "NVIDIA_NVDEC_H264_CUVID"

        for _ in range(5):
            frames = multi_cap.read_sync(timeout=3.0)
            assert frames is not None
            assert len(frames) == len(cam_videos)
            for c_id in cam_videos:
                assert frames[c_id].shape == (1080, 1920, 3)
                assert frames[c_id].dtype == np.uint8

        if torch.cuda.is_available():
            gpu_frames = multi_cap.read_sync_gpu(device="cuda:0", timeout=3.0)
            assert gpu_frames is not None
            assert len(gpu_frames) == len(cam_videos)
            for c_id in cam_videos:
                assert gpu_frames[c_id].is_cuda
                assert tuple(gpu_frames[c_id].shape) == (1080, 1920, 3)
