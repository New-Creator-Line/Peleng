import math
import os
import numpy as np
import pytest

from src.camera_model import CameraModel
from src.truck_model import TruckModel
from src.fbx_reader import load_fbx_vertices, load_fbx_scene_vertices
from src.road_spline import RoadSplineConstraint
from src.cuboid_ukf import (
    triangulate_n_rays,
    ransac_triangulate,
    get_uncertainty_ellipsoid_3d,
    cuboid_world_corners,
    cuboid_to_camera_bbox,
)
from src.box_fitter import BoxObservation, MultiViewBoxFitter, project_box_extents, box_iou
from src.tracker_3d import ConstantVelocityKF


def test_truck_model_roundtrip_and_config():
    """Verify TruckModel pivot/center conversions and JSON loading."""
    tm = TruckModel.load("configs/truck_model.json")
    assert tm.length > 1900.0
    assert tm.width > 400.0
    assert tm.height > 500.0
    assert tm.center_forward < -1000.0

    pivot = np.array([1000.0, 2000.0, 350.0])
    yaw = 0.65
    center = tm.center_from_pivot(pivot, yaw)
    pivot_rec = tm.pivot_from_center(center, yaw)
    assert np.allclose(pivot, pivot_rec, atol=1e-6)

    corners = tm.box_corners(center, yaw)
    assert corners.shape == (8, 3)


def test_fbx_reader_on_cargo_truck():
    """Verify binary FBX parser reads CargoTruck.fbx without external dependencies."""
    fbx_path = "data/assets/CargoTruck.fbx"
    if not os.path.exists(fbx_path):
        pytest.skip("CargoTruck.fbx not present")
    verts, info = load_fbx_vertices(fbx_path)
    assert verts.shape[0] > 100
    assert verts.shape[1] == 3
    scene_v, per_model = load_fbx_scene_vertices(fbx_path)
    assert scene_v.shape[0] > 100
    assert len(per_model) >= 1


def test_road_spline_constraint():
    """Verify 3D cubic B-spline road centerline fitting and trajectory projection."""
    t = np.linspace(0.0, 10.0, 200)
    road_pts = np.column_stack([t * 1000.0, np.sin(t * 0.3) * 500.0, np.full_like(t, 350.0)])

    rsc = RoadSplineConstraint().fit_from_points(road_pts, smoothing=0.0, step=10)
    assert rsc.is_fitted is True

    noisy_pt = road_pts[50] + np.array([0.0, 120.0, 40.0])
    proj_pt, dist = rsc.project_point(noisy_pt)
    assert dist > 100.0
    assert np.linalg.norm(proj_pt - road_pts[50]) < 15.0

    noisy_traj = road_pts.copy()
    noisy_traj[80:100, 1] += 200.0
    occ_mask = np.zeros(len(road_pts), dtype=bool)
    occ_mask[80:100] = True
    constrained = rsc.project_trajectory(noisy_traj, occluded_mask=occ_mask, blend_clear=0.0, blend_occluded=1.0)
    assert np.allclose(constrained[80:100], road_pts[80:100], atol=15.0)


def test_cuboid_ukf_triangulate_and_ellipsoid():
    """Verify N-ray triangulation, RANSAC outlier rejection, and 95% Mahalanobis ellipsoid."""
    true_pt = np.array([1500.0, 2500.0, 400.0], dtype=np.float64)
    origins = [
        np.array([0.0, 0.0, 500.0]),
        np.array([3000.0, 0.0, 500.0]),
        np.array([1500.0, 5000.0, 600.0]),
    ]
    dirs = [(true_pt - c) / np.linalg.norm(true_pt - c) for c in origins]

    est_pt, skew = triangulate_n_rays(origins, dirs)
    assert np.allclose(est_pt, true_pt, atol=1e-4)
    assert skew < 1e-4

    # Add 4th outlier camera ray pointing elsewhere
    origins_out = origins + [np.array([-1000.0, 1000.0, 400.0])]
    dirs_out = dirs + [np.array([0.0, 1.0, 0.0])]
    ransac_pt, inliers, r_skew = ransac_triangulate(origins_out, dirs_out, inlier_thresh=50.0)
    assert ransac_pt is not None
    assert len(inliers) == 3
    assert np.allclose(ransac_pt, true_pt, atol=1e-3)

    cov = np.diag([100.0, 400.0, 25.0])
    radii, eigvecs = get_uncertainty_ellipsoid_3d(cov, chi2_95=7.815)
    assert radii.shape == (3,)
    assert eigvecs.shape == (3, 3)
    assert math.isclose(radii[2], math.sqrt(400.0 * 7.815), rel_tol=1e-3)


def test_box_fitter_and_constant_velocity_kf():
    """Verify MultiViewBoxFitter and ConstantVelocityKF with RTS smoothing."""
    tm = TruckModel()
    cam1 = CameraModel(cam_id=1, loc=np.array([0.0, -3500.0, 500.0]), rot=np.array([0.0, -5.0, 90.0]), focal_length_mm=24.0)
    cam2 = CameraModel(cam_id=2, loc=np.array([3500.0, 0.0, 500.0]), rot=np.array([0.0, -5.0, 180.0]), focal_length_mm=24.0)

    true_center = np.array([0.0, 0.0, 300.0])
    true_yaw = 0.2
    corners = tm.box_corners(true_center, true_yaw)
    ext1 = project_box_extents(cam1, corners)
    ext2 = project_box_extents(cam2, corners)
    assert ext1 is not None and ext2 is not None
    assert math.isclose(box_iou(ext1, ext1), 1.0)

    obs = [
        BoxObservation(cam_id=1, model=cam1, box=ext1, conf=0.95),
        BoxObservation(cam_id=2, model=cam2, box=ext2, conf=0.95),
    ]
    fitter = MultiViewBoxFitter(tm)
    res = fitter.fit(obs, init_center=np.array([80.0, -60.0, 280.0]), init_yaw=0.1)
    assert res is not None
    assert np.linalg.norm(res.center - true_center) < 30.0

    kf = ConstantVelocityKF(dim=3, dt=0.1, accel_std=50.0)
    kf.initialize(pos=[0.0, 0.0, 0.0], pos_cov=np.eye(3) * 10.0, vel=[100.0, 0.0, 0.0], vel_std=10.0)
    for step_i in range(1, 10):
        kf.step(z=[step_i * 10.0, 0.0, 0.0], R=np.eye(3) * 4.0)
    smoothed = kf.smooth()
    assert smoothed.shape == (10, 6)
    assert math.isclose(smoothed[-1, 0], 90.0, abs_tol=2.0)
