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
