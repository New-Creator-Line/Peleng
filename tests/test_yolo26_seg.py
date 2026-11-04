import numpy as np
import pytest
from src.detector import YOLO26SegmentationDetector, compute_mask_moments, DetectionResult


def test_compute_mask_moments_rectangle():
    # Centered rectangle [100, 100] to [300, 200]
    # Expected centroid: x = 200, y = 150; width = 200, height = 100
    polygon = np.array([
        [100.0, 100.0],
        [300.0, 100.0],
        [300.0, 200.0],
        [100.0, 200.0],
    ], dtype=np.float32)

    u, v, w, h, is_trunc = compute_mask_moments(polygon, img_w=1920, img_h=1080)
    assert abs(u - 200.0) < 1.0
    assert abs(v - 150.0) < 1.0
    assert abs(w - 200.0) < 1.0
    assert abs(h - 100.0) < 1.0
    assert not is_trunc


def test_compute_mask_moments_edge_truncated():
    # Polygon touching left border (x = 1.0)
    polygon = np.array([
        [1.0, 500.0],
        [150.0, 500.0],
        [150.0, 600.0],
        [1.0, 600.0],
    ], dtype=np.float32)

    u, v, w, h, is_trunc = compute_mask_moments(polygon, img_w=1920, img_h=1080)
    assert is_trunc is True


def test_yolo26s_seg_model_loading_and_inference():
    # Load detector with downloaded yolo26s-seg.pt
    detector = YOLO26SegmentationDetector(model_name="yolo26s-seg.pt", conf_threshold=0.20)
    assert detector.model is not None

    # Target class 'truck' should be resolved
    assert 7 in detector.target_class_ids

    # Create synthetic blank test image
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Inference should execute cleanly without error
    results = detector.detect_frame(dummy_img, cam_id=1)
    assert isinstance(results, list)


def test_yolo26s_seg_batch_inference():
    detector = YOLO26SegmentationDetector(model_name="yolo26s-seg.pt", conf_threshold=0.20)
    frames = [
        np.zeros((540, 960, 3), dtype=np.uint8),
        np.zeros((540, 960, 3), dtype=np.uint8),
    ]
    batch_res = detector.detect_batch(frames, cam_ids=[1, 2])
    assert len(batch_res) == 2
    assert isinstance(batch_res[0], list)
    assert isinstance(batch_res[1], list)
