import pytest
from src.yolo_detector import estimate_occlusion_status, load_detection_model, PelengDetector


def test_estimate_occlusion_status_unoccluded():
    # Central detection with normal lateral aspect ratio (300 / 120 = 2.5)
    is_occ, score = estimate_occlusion_status(u=960.0, v=540.0, w=300.0, h=120.0, cam_id=1)
    assert not is_occ
    assert score == 0.0


def test_estimate_occlusion_status_edge_clipped():
    # Bounding box touching left edge of 1920x1080 frame
    is_occ, score = estimate_occlusion_status(u=40.0, v=540.0, w=80.0, h=120.0, cam_id=1)
    assert is_occ
    assert score >= 0.85


def test_estimate_occlusion_status_aspect_ratio_truncated():
    # Lateral camera view where truck width has shrunk to 100 while height is 120 (AR = 0.83)
    is_occ, score = estimate_occlusion_status(u=960.0, v=540.0, w=100.0, h=120.0, cam_id=2)
    assert is_occ
    assert score > 0.40


def test_load_detection_model_factory():
    from ultralytics import YOLO, RTDETR
    model_yolo = load_detection_model("yolov8x.pt")
    assert isinstance(model_yolo, YOLO)

    model_rtdetr = load_detection_model("rtdetr-l.pt")
    assert isinstance(model_rtdetr, RTDETR)
