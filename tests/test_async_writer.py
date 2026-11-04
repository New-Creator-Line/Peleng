"""
Tests for AsyncMultiVideoWriter and parallel NVENC transcoding.
"""

from pathlib import Path
import numpy as np
import pytest
import cv2

from src.async_writer import AsyncMultiVideoWriter, transcode_to_nvenc_parallel

def test_async_multi_video_writer_lifecycle(tmp_path):
    w, h = 320, 240
    fps = 30.0
    out1 = tmp_path / "stream1.mp4"
    out2 = tmp_path / "stream2.mp4"
    
    writer = AsyncMultiVideoWriter(queue_size=8)
    writer.add_stream(1, out1, fps, (w, h))
    writer.add_stream("dash", out2, fps, (w, h))
    
    dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(15):
        dummy_frame[:] = (i * 10) % 255
        assert writer.write(1, dummy_frame)
        assert writer.write("dash", dummy_frame)
    
    writer.close()
    
    assert out1.exists()
    assert out2.exists()
    assert out1.stat().st_size > 0
    assert out2.stat().st_size > 0
    
    # Verify readable by OpenCV
    cap = cv2.VideoCapture(str(out1))
    assert cap.isOpened()
    n_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        n_frames += 1
    cap.release()
    assert n_frames == 15
