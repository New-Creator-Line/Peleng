import numpy as np
import pytest
from src.tracker_3d import KalmanFilter3D


def test_kalman_initialization():
    kf = KalmanFilter3D(dt=1.0 / 30.0)
    init_pos = np.array([100.0, 200.0, 300.0])
    kf.initialize(init_pos)

    assert np.allclose(kf.get_position(), init_pos)
    assert np.allclose(kf.get_velocity(), np.zeros(3))


def test_kalman_constant_velocity_tracking():
    dt = 0.1
    kf = KalmanFilter3D(dt=dt, process_noise_std=0.1, measurement_noise_std=0.5)

    # True trajectory: moving with velocity [10, 5, -2]
    v_true = np.array([10.0, 5.0, -2.0])
    p_true = np.array([0.0, 0.0, 0.0])
    kf.initialize(p_true)

    for step in range(1, 30):
        p_true = p_true + v_true * dt
        noise = np.random.normal(0, 0.2, size=3)
        z = p_true + noise

        kf.predict()
        kf.update(z)

    est_pos = kf.get_position()
    est_vel = kf.get_velocity()

    assert np.allclose(est_pos, p_true, atol=2.0)
    assert np.allclose(est_vel, v_true, atol=1.5)


def test_kalman_occlusion_prediction():
    """During occlusions (predict without update), the filter coasts along velocity vector."""
    dt = 0.1
    kf = KalmanFilter3D(dt=dt)
    kf.initialize(np.array([0.0, 0.0, 0.0]))

    # Push two measurements to establish velocity
    kf.predict()
    kf.update(np.array([1.0, 0.0, 0.0]))
    kf.predict()
    kf.update(np.array([2.0, 0.0, 0.0]))

    pos_before_occlusion = kf.get_position()
    vel = kf.get_velocity()

    # 3 missing frames (occlusion)
    for _ in range(3):
        kf.predict()

    expected_pos = pos_before_occlusion + vel * (3 * dt)
    assert np.allclose(kf.get_position(), expected_pos, atol=1e-2)
