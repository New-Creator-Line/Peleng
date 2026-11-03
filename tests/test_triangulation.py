import math
import numpy as np
import pytest
from src.camera_model import CameraModel
from src.triangulator import MultiViewTriangulator


def create_two_camera_setup():
    """Create two synthetic cameras viewing a shared volume."""
    # Camera 1 at (-1000, 0, 0), looking toward origin (+X, Yaw=0)
    cam1 = CameraModel(
        cam_id=1,
        loc=np.array([-1000.0, 0.0, 0.0]),
        rot=np.array([0.0, 0.0, 0.0]),
        focal_length_mm=35.0,
        sensor_width_mm=23.76,
        sensor_height_mm=13.365,
        image_width_px=1920,
        image_height_px=1080,
    )
    # Camera 2 at (0, -1000, 0), looking toward origin (+Y, Yaw=90)
    cam2 = CameraModel(
        cam_id=2,
        loc=np.array([0.0, -1000.0, 0.0]),
        rot=np.array([0.0, 0.0, 90.0]),
        focal_length_mm=35.0,
        sensor_width_mm=23.76,
        sensor_height_mm=13.365,
        image_width_px=1920,
        image_height_px=1080,
    )
    return [cam1, cam2]


def test_perfect_triangulation():
    """Triangulating perfect projections should recover the exact 3D point."""
    cams = create_two_camera_setup()
    target_3d = np.array([50.0, -30.0, 20.0])

    observations = []
    projection_matrices = []

    for cam in cams:
        u, v, z_cam = cam.project_point(target_3d)
        observations.append((u, v))
        projection_matrices.append(cam.get_projection_matrix())

    triangulator = MultiViewTriangulator()
    recovered_3d, error = triangulator.triangulate(projection_matrices, observations)

    assert recovered_3d is not None
    assert np.allclose(recovered_3d, target_3d, atol=1e-3)
    assert error < 1e-4


def test_noisy_triangulation():
    """Triangulating with 1-pixel noise should still be highly accurate."""
    cams = create_two_camera_setup()
    target_3d = np.array([100.0, 50.0, -10.0])

    observations = []
    projection_matrices = []

    for cam in cams:
        u, v, z_cam = cam.project_point(target_3d)
        observations.append((u + 0.5, v - 0.5))
        projection_matrices.append(cam.get_projection_matrix())

    triangulator = MultiViewTriangulator()
    recovered_3d, error = triangulator.triangulate(projection_matrices, observations)

    assert recovered_3d is not None
    # With 1000 distance and 2828 focal length, 0.5 px corresponds to ~0.18 units
    assert np.allclose(recovered_3d, target_3d, atol=1.0)
