from typing import Optional
import numpy as np

from src.tracker_ukf import UKF10DTracker
from src.rts_smoother import RTSSmoother10D

try:
    import peleng_core
    HAS_CPP_CORE = True
except ImportError:
    HAS_CPP_CORE = False


class KalmanFilter3D:
    """
    3D Constant Velocity Kalman Filter.
    State vector: [x, y, z, vx, vy, vz]^T
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 5.0,
        measurement_noise_std: float = 2.0,
    ) -> None:
        self.dt = float(dt)
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)
        self.is_initialized = False

        # State transition matrix F (6x6)
        self.F = np.eye(6, dtype=np.float64)
        for i in range(3):
            self.F[i, i + 3] = self.dt

        # Measurement matrix H (3x6)
        self.H = np.zeros((3, 6), dtype=np.float64)
        for i in range(3):
            self.H[i, i] = 1.0

        # State covariance P (6x6)
        self.P = np.eye(6, dtype=np.float64) * 100.0

        # Process noise covariance Q (6x6)
        # Continuous white noise acceleration model
        q = self.process_noise_std ** 2
        dt2 = self.dt ** 2
        dt3 = self.dt ** 3 / 2.0
        dt4 = self.dt ** 4 / 4.0
        self.Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            self.Q[i, i] = dt4 * q
            self.Q[i, i + 3] = dt3 * q
            self.Q[i + 3, i] = dt3 * q
            self.Q[i + 3, i + 3] = dt2 * q

        # Measurement noise covariance R (3x3)
        r = self.measurement_noise_std ** 2
        self.R = np.eye(3, dtype=np.float64) * r

        # State vector x (6,)
        self.x = np.zeros(6, dtype=np.float64)

    def initialize(self, init_pos_3d: np.ndarray, init_vel_3d: Optional[np.ndarray] = None) -> None:
        self.x[:3] = np.asarray(init_pos_3d, dtype=np.float64)
        if init_vel_3d is not None:
            self.x[3:] = np.asarray(init_vel_3d, dtype=np.float64)
        else:
            self.x[3:] = 0.0

        self.P = np.eye(6, dtype=np.float64) * 10.0
        self.is_initialized = True

    def predict(self) -> np.ndarray:
        if not self.is_initialized:
            return np.zeros(3)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, measurement_3d: np.ndarray, noise_scale: float = 1.0) -> np.ndarray:
        if not self.is_initialized:
            self.initialize(measurement_3d)
            return self.x[:3].copy()

        z = np.asarray(measurement_3d, dtype=np.float64)
        R = self.R * max(0.1, float(noise_scale))

        # Innovation: y = z - H * x
        y = z - self.H @ self.x
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + R
        # Kalman gain: K = P * H^T * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.x[:3].copy()

    def get_position(self) -> np.ndarray:
        return self.x[:3].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[3:].copy()
