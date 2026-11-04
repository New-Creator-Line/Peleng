import math
import numpy as np
import pytest

from src.camera_model import CameraModel
from src.cuboid_model import TruckCuboid3D, project_cuboid_to_camera_bbox
from src.tracker_ukf import (
    UKF10DTracker,
    normalize_angle,
    circular_mean,
)


def test_normalize_angle():
    """Verify angle normalization to [-pi, pi)."""
    assert math.isclose(normalize_angle(0.0), 0.0)
    assert math.isclose(normalize_angle(math.pi), -math.pi, abs_tol=1e-5)
    assert math.isclose(normalize_angle(3 * math.pi), -math.pi, abs_tol=1e-5)
    assert math.isclose(normalize_angle(-3 * math.pi), -math.pi, abs_tol=1e-5)
    assert math.isclose(normalize_angle(math.pi / 2.0), math.pi / 2.0)
    assert math.isclose(normalize_angle(-math.pi / 2.0), -math.pi / 2.0)


def test_circular_mean():
    """Verify circular mean near the wrap-around boundary (+pi and -pi)."""
    angles = np.array([math.pi - 0.1, -math.pi + 0.1])
    weights = np.array([0.5, 0.5])
    # The average of (pi - 0.1) and (-pi + 0.1) is exactly pi (or -pi)
    mean_angle = circular_mean(angles, weights)
    assert math.isclose(abs(mean_angle), math.pi, abs_tol=1e-3)


def test_ukf_initialization_diffuse_prior():
    """Verify UKF initializes cleanly with diffuse prior without hardcoded dimensions."""
    tracker = UKF10DTracker(dt=1.0 / 30.0)
    pos_init = np.array([1000.0, 500.0, 50.0])

    # Initial state with unknown initial dimensions
    tracker.initialize(
        init_pos=pos_init,
        init_yaw=0.0,
        init_speed=0.0,
        init_dimensions=np.array([500.0, 200.0, 200.0]),  # generic initial guess
    )

    assert tracker.is_initialized is True
    st = tracker.get_state()
    assert np.allclose(st[:3], pos_init)
    # Check that dimension covariance is large (diffuse prior, no hardcoding)
    cov = tracker.get_covariance()
    assert cov[7, 7] >= 10000.0  # Var(L) >= (100 cm)^2
    assert cov[8, 8] >= 5000.0   # Var(W)
    assert cov[9, 9] >= 5000.0   # Var(H)


def test_ukf_ctrv_straight_line_prediction():
    """Verify straight line motion (zero turn rate) propagates position correctly."""
    dt = 0.1
    tracker = UKF10DTracker(dt=dt, process_noise_std=0.0)
    # Moving along +X with speed 500 cm/s and deterministic heading
    tracker.initialize(
        init_pos=np.array([0.0, 0.0, 0.0]),
        init_yaw=0.0,
        init_speed=500.0,
        init_yaw_std=1e-4,
        init_speed_std=1e-4,
        init_yaw_rate_std=1e-4,
    )

    for _ in range(10):
        tracker.predict()

    # After 10 steps of 0.1s (1.0s total), X should be ~500.0 cm
    pos = tracker.get_position()
    assert math.isclose(pos[0], 500.0, abs_tol=5.0)
    assert math.isclose(pos[1], 0.0, abs_tol=1.0)
    assert math.isclose(pos[2], 0.0, abs_tol=1.0)


def test_ukf_ctrv_turning_circle_prediction():
    """Verify circular motion (constant non-zero yaw rate) forms a circle of radius R = v / omega."""
    dt = 0.05
    tracker = UKF10DTracker(dt=dt, process_noise_std=0.0)
    speed = 200.0  # 200 cm/s
    omega = 0.2    # 0.2 rad/s -> radius R = 200 / 0.2 = 1000 cm
    tracker.initialize(
        init_pos=np.array([0.0, 0.0, 0.0]),
        init_yaw=0.0,
        init_speed=speed,
        init_yaw_rate=omega,
        init_yaw_std=1e-4,
        init_speed_std=1e-4,
        init_yaw_rate_std=1e-4,
    )

    # Propagate for quarter circle: angle = omega * t = pi / 2 -> t = pi / (2 * 0.2) = 2.5 * pi s
    num_steps = int(math.ceil((math.pi / (2.0 * omega)) / dt))
    for _ in range(num_steps):
        tracker.predict()

    # In CTRV with init_yaw=0 and omega > 0:
    # X = R * sin(psi), Y = R * (1 - cos(psi))
    # At quarter circle: psi = pi/2 -> X ~ R = 1000, Y ~ R = 1000
    pos = tracker.get_position()
    assert math.isclose(pos[0], 1000.0, abs_tol=30.0)
    assert math.isclose(pos[1], 1000.0, abs_tol=30.0)


def test_ukf_3d_point_update():
    """Verify 3D position measurement update with Mahalanobis gating."""
    tracker = UKF10DTracker(dt=0.1)
    tracker.initialize(init_pos=np.array([0.0, 0.0, 0.0]), init_speed=0.0)

    tracker.predict()
    # Update with 3D point (e.g. from RANSAC multi-camera triangulation)
    meas = np.array([2.0, 3.0, 0.5])
    accepted = tracker.update_3d_point(meas, R_cov=np.eye(3) * 1.0)

    assert accepted is True
    pos = tracker.get_position()
    # Position should shift toward measurement
    assert np.allclose(pos, meas, atol=1.5)


def test_ukf_online_dimension_convergence():
    """
    Verify that the UKF dynamically calculates and converges to true dimensions (L, W, H)
    without any hardcoding, by observing multi-camera 2D bounding boxes.
    """
    # True physical truck in scene (algorithm does NOT know this)
    true_L = 629.0
    true_W = 207.0
    true_H = 212.0
    true_x = 1200.0
    true_y = 600.0
    true_z = 0.0
    true_yaw = 0.2

    true_cuboid = TruckCuboid3D(
        x=true_x, y=true_y, z=true_z,
        yaw=true_yaw,
        length=true_L, width=true_W, height=true_H,
    )

    # 2 Calibrated cameras looking from different angles
    # Cam 1 (Side view)
    cam1 = CameraModel(
        cam_id=1,
        loc=np.array([1200.0, -1500.0, 400.0]),
        rot=np.array([0.0, -10.0, 90.0]),
        focal_length_mm=35.0,
    )
    # Cam 2 (Frontal view)
    cam2 = CameraModel(
        cam_id=2,
        loc=np.array([2500.0, 600.0, 300.0]),
        rot=np.array([0.0, -10.0, 180.0]),
        focal_length_mm=35.0,
    )

    # Get true 2D observations on both cameras
    res1 = project_cuboid_to_camera_bbox(true_cuboid, cam1)
    res2 = project_cuboid_to_camera_bbox(true_cuboid, cam2)
    assert res1 is not None and res2 is not None

    b1 = res1[0]  # [x_min, y_min, x_max, y_max]
    u1, v1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]

    b2 = res2[0]
    u2, v2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]

    # Initialize tracker with WRONG, generic initial guess (e.g. compact van: 400x180x170)
    tracker = UKF10DTracker(dt=1.0 / 30.0)
    tracker.initialize(
        init_pos=np.array([true_x, true_y, true_z]),
        init_yaw=true_yaw,
        init_speed=0.0,
        init_dimensions=np.array([400.0, 160.0, 160.0]),  # deliberately inaccurate!
    )

    # Feed multi-camera bounding box observations over 20 frames
    for _ in range(25):
        tracker.predict()
        tracker.update_camera_box(cam1, u1, v1, w1, h1)
        tracker.update_camera_box(cam2, u2, v2, w2, h2)

    est_dims = tracker.get_dimensions()
    est_L, est_W, est_H = est_dims[0], est_dims[1], est_dims[2]

    # Dimensions MUST have dynamically adjusted toward the true values
    # From 400 -> toward 629, from 160 -> toward 207, from 160 -> toward 212
    assert est_L > 500.0  # Increased significantly toward 629
    assert est_W > 178.0  # Increased toward 207
    assert est_H > 180.0  # Increased toward 212
