import math
import numpy as np
import pytest

from src.camera_model import CameraModel
from src.cuboid_model import (
    TruckCuboid3D,
    UncertaintyEllipsoid3D,
    compute_uncertainty_ellipsoid_3d,
    project_cuboid_to_camera_bbox,
    draw_cuboid_wireframe_hud,
    draw_uncertainty_ellipsoid_hud,
)


def test_cuboid_vertices_zero_rotation():
    """Verify 8 vertices generated with zero yaw rotation around cabin tracking pivot."""
    cuboid = TruckCuboid3D(
        x=1000.0, y=500.0, z=50.0,
        yaw=0.0,
        length=600.0, width=200.0, height=200.0
    )
    verts = cuboid.get_world_vertices()
    assert verts.shape == (8, 3)

    # In UE5: X is forward (length), Y is right (width), Z is up (height)
    # With cabin pivot (d_front=150.0), at yaw=0:
    # X ranges from 1000 + (150 - 600) to 1000 + 150 -> [550, 1150]
    # Y ranges from 500 - 100 to 500 + 100 -> [400, 600]
    # Z ranges from 50 to 50 + 200 -> [50, 250]
    assert np.isclose(np.min(verts[:, 0]), 550.0)
    assert np.isclose(np.max(verts[:, 0]), 1150.0)
    assert np.isclose(np.min(verts[:, 1]), 400.0)
    assert np.isclose(np.max(verts[:, 1]), 600.0)
    assert np.isclose(np.min(verts[:, 2]), 50.0)
    assert np.isclose(np.max(verts[:, 2]), 250.0)

    # Check centroid (shifted -150 cm along X from cabin pivot for L=600, d_front=150)
    centroid = cuboid.get_centroid()
    assert np.allclose(centroid, np.array([850.0, 500.0, 150.0]))


def test_cuboid_vertices_yaw_90_deg():
    """Verify 90-degree yaw rotation (+pi/2 around Z-axis) with cabin pivot."""
    # 90 deg rotation swaps X and Y in world
    cuboid = TruckCuboid3D(
        x=0.0, y=0.0, z=0.0,
        yaw=math.pi / 2.0,
        length=600.0, width=200.0, height=100.0
    )
    verts = cuboid.get_world_vertices()

    # Length (600, [-450, +150]) was along X, now rotated +90 deg -> along +Y (from -450 to +150 on Y)
    # Width (200) was along Y, now rotated +90 deg -> along -X (from -100 to +100 on X)
    assert np.isclose(np.min(verts[:, 0]), -100.0)
    assert np.isclose(np.max(verts[:, 0]), 100.0)
    assert np.isclose(np.min(verts[:, 1]), -450.0)
    assert np.isclose(np.max(verts[:, 1]), 150.0)
    assert np.isclose(np.min(verts[:, 2]), 0.0)
    assert np.isclose(np.max(verts[:, 2]), 100.0)


def test_cuboid_edges_and_front_face():
    """Verify 12 bounding box edges and front cabin indicators."""
    cuboid = TruckCuboid3D(0.0, 0.0, 0.0, 0.0, 950.0, 200.0, 200.0)
    edges = cuboid.get_edges()
    assert len(edges) == 12

    front_face_indices = cuboid.get_front_face_indices()
    assert len(front_face_indices) == 4

    world_verts = cuboid.get_world_vertices()
    front_verts = world_verts[front_face_indices]
    # At yaw=0 with d_front=150.0 and L=950.0, front face is at +150.0 and rear is at -800.0
    assert np.allclose(front_verts[:, 0], 150.0)
    assert np.isclose(np.min(world_verts[:, 0]), -800.0)


def test_uncertainty_ellipsoid_isotropic():
    """Verify 3D uncertainty ellipsoid for spherical/isotropic covariance."""
    cov = np.eye(3) * 4.0  # std = 2.0, var = 4.0
    ellipsoid = compute_uncertainty_ellipsoid_3d(cov, chi2_val=7.815)

    # Semi-axes radii: r = sqrt(7.815 * 4.0) = 2.0 * sqrt(7.815) approx 5.591
    expected_r = math.sqrt(7.815 * 4.0)
    assert np.allclose(ellipsoid.radii, expected_r, rtol=1e-3)
    assert math.isclose(ellipsoid.volume, (4.0 / 3.0) * math.pi * (expected_r ** 3), rel_tol=1e-3)


def test_uncertainty_ellipsoid_anisotropic():
    """Verify anisotropic ellipsoid semi-axes calculation and Mahalanobis distance."""
    cov = np.diag([9.0, 4.0, 1.0])
    center = np.array([10.0, 20.0, 30.0])
    ellipsoid = compute_uncertainty_ellipsoid_3d(cov, center=center, chi2_val=7.815)

    assert np.allclose(ellipsoid.radii, np.array([
        math.sqrt(7.815 * 1.0),
        math.sqrt(7.815 * 4.0),
        math.sqrt(7.815 * 9.0)
    ]))

    # Test Mahalanobis distance
    # Point at center -> d_M^2 = 0
    assert math.isclose(ellipsoid.mahalanobis_distance_sq(center), 0.0)

    # Point on 1-sigma boundary along X: (13, 20, 30) -> (3^2 / 9) = 1.0
    pt_1sig = np.array([13.0, 20.0, 30.0])
    assert math.isclose(ellipsoid.mahalanobis_distance_sq(pt_1sig), 1.0, rel_tol=1e-4)
    assert ellipsoid.is_inside_95(pt_1sig) is True

    # Point outside 95% boundary: (10 + 3*sqrt(8), 20, 30) -> d_M^2 = 8.0 > 7.815
    pt_out = np.array([10.0 + 3.0 * math.sqrt(8.0), 20.0, 30.0])
    assert ellipsoid.mahalanobis_distance_sq(pt_out) > 7.815
    assert ellipsoid.is_inside_95(pt_out) is False


def test_uncertainty_ellipsoid_regularization():
    """Verify non-symmetric or near-singular covariance is safely regularized."""
    cov_bad = np.array([
        [1.0, 0.5, 0.0],
        [0.4, 2.0, 0.1],  # Non-symmetric!
        [0.0, 0.1, -0.5], # Negative eigenvalue!
    ])
    ellipsoid = compute_uncertainty_ellipsoid_3d(cov_bad)
    assert np.all(ellipsoid.radii > 0)
    assert not np.isnan(ellipsoid.volume)


def test_project_cuboid_to_camera():
    """Verify 3D cuboid projection into 2D camera coordinates and bounding box calculation."""
    # Camera at origin looking along +X (UE5)
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

    # Cuboid directly ahead along +X at X=1000 cm, length=400, width=200, height=100
    cuboid = TruckCuboid3D(
        x=1000.0, y=0.0, z=-50.0,  # Center base at -50, top at +50 -> centered vertically around Z=0
        yaw=0.0,
        length=400.0, width=200.0, height=100.0
    )

    proj_res = project_cuboid_to_camera_bbox(cuboid, cam, img_w=1920, img_h=1080)
    assert proj_res is not None
    bbox_2d, pts_2d, edges_2d = proj_res

    # Center of bbox should be approximately image center (960, 540)
    u_c = (bbox_2d[0] + bbox_2d[2]) / 2.0
    v_c = (bbox_2d[1] + bbox_2d[3]) / 2.0
    assert math.isclose(u_c, 960.0, abs_tol=5.0)
    assert math.isclose(v_c, 540.0, abs_tol=5.0)

    # Check that all 8 points are strictly within bbox
    assert bbox_2d[0] <= np.min(pts_2d[:, 0]) + 1e-3
    assert bbox_2d[2] >= np.max(pts_2d[:, 0]) - 1e-3
    assert bbox_2d[1] <= np.min(pts_2d[:, 1]) + 1e-3
    assert bbox_2d[3] >= np.max(pts_2d[:, 1]) - 1e-3
    assert len(edges_2d) == 12


def test_project_cuboid_behind_camera_returns_none():
    """Verify cuboid entirely behind camera is safely rejected."""
    cam = CameraModel(
        cam_id=1,
        loc=np.array([0.0, 0.0, 0.0]),
        rot=np.array([0.0, 0.0, 0.0]),
    )
    # Cuboid at X=-500 (behind camera looking along +X)
    cuboid = TruckCuboid3D(-500.0, 0.0, 0.0, 0.0, 400.0, 200.0, 200.0)
    proj_res = project_cuboid_to_camera_bbox(cuboid, cam, img_w=1920, img_h=1080)
    assert proj_res is None


def test_draw_cuboid_wireframe_hud():
    """Verify 3D cuboid wireframe renders onto an image canvas without errors."""
    cam = CameraModel(
        cam_id=1,
        loc=np.array([0.0, 0.0, 0.0]),
        rot=np.array([0.0, 0.0, 0.0]),
    )
    cuboid = TruckCuboid3D(1000.0, 0.0, -50.0, 0.0, 400.0, 200.0, 100.0)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    rendered = draw_cuboid_wireframe_hud(frame, cuboid, cam)
    assert rendered is True
    # Canvas should have non-zero pixels from drawing
    assert np.count_nonzero(frame) > 0


def test_draw_uncertainty_ellipsoid_hud():
    """Verify 3D uncertainty ellipsoid renders onto an image canvas without errors."""
    cam = CameraModel(
        cam_id=1,
        loc=np.array([0.0, 0.0, 0.0]),
        rot=np.array([0.0, 0.0, 0.0]),
    )
    cov = np.diag([100.0, 50.0, 25.0])
    ellipsoid = compute_uncertainty_ellipsoid_3d(cov, center=np.array([1000.0, 0.0, 0.0]))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    rendered = draw_uncertainty_ellipsoid_hud(frame, ellipsoid, cam)
    assert rendered is True
    assert np.count_nonzero(frame) > 0

