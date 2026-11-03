import math
import numpy as np
import pytest
from src.camera_model import CameraModel, CameraParams, load_all_cameras


def test_camera_intrinsics():
    """Verify camera intrinsic calculation from physical sensor and focal length."""
    cam = CameraModel(
        cam_id=1,
        loc=np.array([0.0, 0.0, 0.0]),
        rot=np.array([0.0, 0.0, 0.0]),
        focal_length_mm=35.0,
        sensor_width_mm=23.76,
        sensor_height_mm=13.365,
        image_width_px=1920,
        image_height_px=1080,
    )
    fx, fy, cx, cy = cam.get_intrinsics()
    assert math.isclose(fx, 2828.2828, rel_tol=1e-4)
    assert math.isclose(fy, 2828.2828, rel_tol=1e-4)
    assert cx == 960.0
    assert cy == 540.0


def test_point_projection_optical_axis():
    """A point directly ahead in UE space (+X) should project to image center (cx, cy)."""
    # Camera at origin looking along +X (Roll=0, Pitch=0, Yaw=0)
    cam = CameraModel(
        cam_id=1,
        loc=np.array([0.0, 0.0, 0.0]),
        rot=np.array([0.0, 0.0, 0.0]),
        focal_length_mm=35.0,
        sensor_width_mm=23.76,
        sensor_height_mm=13.365,
        image_width_px=1920,
        image_height_px=1080,
    )
    # Point at X=1000, Y=0, Z=0 in UE space
    pt_3d = np.array([1000.0, 0.0, 0.0])
    u, v, z_cam = cam.project_point(pt_3d)

    assert math.isclose(u, 960.0, abs_tol=1e-4)
    assert math.isclose(v, 540.0, abs_tol=1e-4)
    assert math.isclose(z_cam, 1000.0, abs_tol=1e-4)


def test_point_projection_right_and_up():
    """In UE: +Y is Right -> +u in image. +Z is Up -> -v in image (towards top)."""
    cam = CameraModel(
        cam_id=1,
        loc=np.array([0.0, 0.0, 0.0]),
        rot=np.array([0.0, 0.0, 0.0]),
        focal_length_mm=35.0,
        sensor_width_mm=23.76,
        sensor_height_mm=13.365,
        image_width_px=1920,
        image_height_px=1080,
    )
    # Point slightly to the right (+Y) and up (+Z)
    pt_3d = np.array([1000.0, 50.0, 30.0])
    u, v, z_cam = cam.project_point(pt_3d)

    assert u > 960.0  # Right
    assert v < 540.0  # Up (towards 0 in image coordinates)


def test_load_all_cameras():
    """Verify loading real camera CSV files from hackathon folder with per-camera focal lengths."""
    cameras = load_all_cameras("хакатон")
    assert len(cameras) == 5
    expected_focal = {1: 35.0, 2: 60.0, 3: 35.0, 4: 50.0, 5: 20.0}
    for cam_id in range(1, 6):
        assert cam_id in cameras
        # Verify 1001 frames loaded
        assert len(cameras[cam_id].frames) >= 1000
        model0 = cameras[cam_id].get_model(0)
        assert math.isclose(model0.focal_length_mm, expected_focal[cam_id], rel_tol=1e-4)
        assert math.isclose(model0.sensor_width_mm, 23.76, rel_tol=1e-4)
        assert math.isclose(model0.sensor_height_mm, 13.365, rel_tol=1e-4)
