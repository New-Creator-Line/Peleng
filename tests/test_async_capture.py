from pathlib import Path
import numpy as np
import pytest
import torch

from src.async_capture import (
    AsyncMultiCameraCapture,
    NVDEC_STATUS,
    configure_nvdec_environment,
)


def test_nvdec_environment_configuration():
    """Verify NVIDIA Video Codec SDK 13.1.24 NVDEC configuration and hardware flags."""
    caps = configure_nvdec_environment(enable_cuvid=True)
    assert isinstance(caps, dict)
    assert caps["backend"] == "NVIDIA_NVDEC_H264_CUVID"
    assert NVDEC_STATUS["backend"] == "NVIDIA_NVDEC_H264_CUVID"


def test_async_multi_camera_capture_lifecycle():
    """Verify clean startup, synchronized NVDEC frame retrieval, and shutdown."""
    video_dir = Path("data/raw/hackathon")
    cam_videos = {}
    for c_id in range(1, 6):
        matches = list(video_dir.glob(f"*Actor{c_id}.mp4"))
        if matches:
            cam_videos[c_id] = str(matches[0])

    if len(cam_videos) < 5:
        pytest.skip("Hackathon videos not found on disk")

    with AsyncMultiCameraCapture(cam_videos, queue_size=3, use_nvdec=True) as multi_cap:
        assert multi_cap.fps > 0
        assert multi_cap.width == 1920
        assert multi_cap.height == 1080
        assert multi_cap.nvdec_info["backend"] == "NVIDIA_NVDEC_H264_CUVID"

        for _ in range(5):
            frames = multi_cap.read_sync(timeout=3.0)
            assert frames is not None
            assert len(frames) == 5
            for c_id in range(1, 6):
                assert frames[c_id].shape == (1080, 1920, 3)
                assert frames[c_id].dtype == np.uint8

        if torch.cuda.is_available():
            gpu_frames = multi_cap.read_sync_gpu(device="cuda:0", timeout=3.0)
            assert gpu_frames is not None
            assert len(gpu_frames) == 5
            for c_id in range(1, 6):
                assert gpu_frames[c_id].is_cuda
                assert tuple(gpu_frames[c_id].shape) == (1080, 1920, 3)
