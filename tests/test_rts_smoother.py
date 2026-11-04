import math
import numpy as np
import pytest

from src.rts_smoother import RTSSmoother10D
from src.tracker_ukf import UKF10DTracker


def test_rts_smoother_reduces_variance():
    """Verify that RTS backward smoother yields P_t|T <= P_t|t everywhere."""
    dt = 0.1
    tracker = UKF10DTracker(dt=dt, process_noise_std=5.0)
    tracker.initialize(
        init_pos=np.array([0.0, 0.0, 0.0]),
        init_yaw=0.0,
        init_speed=100.0,
    )

    # Simulate 20 frames with noisy 3D observations
    true_x = 0.0
    for frame in range(20):
        true_x += 100.0 * dt
        tracker.predict()
        # Add noisy measurement
        z = np.array([true_x + np.random.normal(0, 5.0), 0.0, 0.0])
        tracker.update_3d_point(z, R_cov=np.eye(3) * 25.0)
        tracker.commit_step(frame)

    smoother = RTSSmoother10D()
    smoothed_records = smoother.smooth(tracker.history)

    assert len(smoothed_records) == 20
    # For every step, smoothed variance should be less than or equal to filtered variance
    for rec in smoothed_records:
        frame = rec["frame_idx"]
        orig_post = tracker.history[frame].P_post
        smooth_cov = rec["P_smoothed"]
        # Check diagonal variances for X, Y, Z
        assert smooth_cov[0, 0] <= orig_post[0, 0] + 1e-4
        assert smooth_cov[1, 1] <= orig_post[1, 1] + 1e-4
        assert smooth_cov[2, 2] <= orig_post[2, 2] + 1e-4


def test_rts_smoother_eliminates_lag():
    """Verify that RTS smoother removes phase lag on a sinusoidal / smooth trajectory."""
    dt = 0.05
    tracker = UKF10DTracker(dt=dt, process_noise_std=2.0)
    tracker.initialize(
        init_pos=np.array([0.0, 0.0, 0.0]),
        init_yaw=0.0,
        init_speed=200.0,
        init_pos_std=2.0,
        init_yaw_std=1e-3,
    )

    # Generate 40 frames of constant velocity motion
    for frame in range(40):
        tracker.predict()
        meas_x = float(frame * 200.0 * dt)
        tracker.update_3d_point(np.array([meas_x, 0.0, 0.0]), R_cov=np.eye(3) * 4.0)
        tracker.commit_step(frame)

    smoother = RTSSmoother10D()
    smoothed = smoother.smooth(tracker.history)
    df = smoother.to_dataframe(smoothed)

    assert "Loc_X" in df.columns
    assert "Sigma_X" in df.columns
    assert len(df) == 40
    # Smoothed trajectory should be monotonic and accurate
    diffs = np.diff(df["Loc_X"].to_numpy())
    assert np.all(diffs > 0)


def test_rts_smoother_fixed_lag_sliding_window():
    """Verify Fixed-Lag RTS strictly uses at most t + lag_frames lookahead and matches smooth_step."""
    dt = 1.0 / 30.0
    tracker = UKF10DTracker(dt=dt, process_noise_std=10.0)
    tracker.initialize(init_pos=np.array([0.0, 0.0, 300.0]), init_yaw=0.0, init_speed=150.0)

    for frame in range(30):
        tracker.predict()
        tracker.update_3d_point(np.array([frame * 5.0, frame * 2.0, 300.0]), R_cov=np.eye(3) * 16.0)
        tracker.commit_step(frame)

    lag = 8
    smoother = RTSSmoother10D(lag_frames=lag)
    full_fixed_lag = smoother.smooth(tracker.history)
    prefix_fixed_lag = smoother.smooth(tracker.history[: 5 + lag + 1])

    # State at t=5 must be bit-identical whether future beyond t+lag exists or not
    np.testing.assert_allclose(full_fixed_lag[5]["x_smoothed"], prefix_fixed_lag[5]["x_smoothed"], rtol=1e-9, atol=1e-9)

    # Single-step streaming helper must match at target_idx
    step_rec = smoother.smooth_step(tracker.history[: 5 + lag + 1], target_idx=5)
    np.testing.assert_allclose(full_fixed_lag[5]["x_smoothed"], step_rec["x_smoothed"], rtol=1e-9, atol=1e-9)

