from typing import Optional
import numpy as np

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



class ConstantVelocityKF:
    """
    N-dimensional constant-velocity Kalman filter with physical noise units and an
    offline Rauch-Tung-Striebel (RTS) smoother.

    State: [p_1..p_N, v_1..v_N]. Process noise: white acceleration with std
    ``accel_std`` (units/s^2). Measurement covariance is supplied per update, so the
    uncertainty of each measurement (e.g. from the multi-view box fit) is respected.

    ``predict``/``update`` give the causal (real-time) estimate. After the sequence is
    processed, ``smooth()`` returns the RTS-smoothed states which remove the lag of the
    forward filter (offline mode).
    """

    def __init__(self, dim: int, dt: float, accel_std) -> None:
        self.dim = int(dim)
        self.dt = float(dt)
        n = self.dim
        self.F = np.eye(2 * n)
        self.F[:n, n:] = np.eye(n) * self.dt
        self.H = np.hstack([np.eye(n), np.zeros((n, n))])
        a = np.broadcast_to(np.asarray(accel_std, dtype=np.float64), (n,))
        dt2, dt3, dt4 = self.dt ** 2, self.dt ** 3 / 2.0, self.dt ** 4 / 4.0
        self.Q = np.zeros((2 * n, 2 * n))
        for i in range(n):
            q = a[i] ** 2
            self.Q[i, i] = dt4 * q
            self.Q[i, i + n] = self.Q[i + n, i] = dt3 * q
            self.Q[i + n, i + n] = dt2 * q
        self.x = np.zeros(2 * n)
        self.P = np.eye(2 * n)
        self.is_initialized = False
        self._hist: list = []  # (x_pred, P_pred, x_filt, P_filt)

    def initialize(self, pos, pos_cov, vel=None, vel_std=None) -> None:
        n = self.dim
        self.x = np.zeros(2 * n)
        self.x[:n] = np.asarray(pos, dtype=np.float64)
        if vel is not None:
            self.x[n:] = np.asarray(vel, dtype=np.float64)
        self.P = np.zeros((2 * n, 2 * n))
        self.P[:n, :n] = np.asarray(pos_cov, dtype=np.float64)
        vs = np.broadcast_to(np.asarray(vel_std if vel_std is not None else 1.0, dtype=np.float64), (n,))
        self.P[n:, n:] = np.diag(vs ** 2)
        self.is_initialized = True
        self._hist.append((self.x.copy(), self.P.copy(), self.x.copy(), self.P.copy()))

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[: self.dim].copy()

    def update(self, z, R) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + np.asarray(R, dtype=np.float64)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(2 * self.dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return self.x[: self.dim].copy()

    def innovation_mahalanobis(self, z, R) -> float:
        y = np.asarray(z, dtype=np.float64) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + np.asarray(R, dtype=np.float64)
        return float(y @ np.linalg.solve(S, y))

    def step(self, z=None, R=None) -> np.ndarray:
        """predict + optional update, recording history for the RTS smoother."""
        self.begin_step()
        return self.end_step(z, R)

    def begin_step(self) -> np.ndarray:
        """Prediction half of a step (call ``end_step`` afterwards)."""
        pos = self.predict()
        self._pending = (self.x.copy(), self.P.copy())
        return pos

    def end_step(self, z=None, R=None) -> np.ndarray:
        """Optional measurement update; records the step for the RTS smoother."""
        x_pred, P_pred = self._pending
        if z is not None:
            self.update(z, R)
        self._hist.append((x_pred, P_pred, self.x.copy(), self.P.copy()))
        return self.x[: self.dim].copy()

    def smooth(self) -> np.ndarray:
        """RTS smoother over the recorded history. Returns smoothed states (T, 2N)."""
        T = len(self._hist)
        if T == 0:
            return np.zeros((0, 2 * self.dim))
        xs = np.array([h[2] for h in self._hist])
        Ps = np.array([h[3] for h in self._hist])
        for k in range(T - 2, -1, -1):
            x_pred_next, P_pred_next = self._hist[k + 1][0], self._hist[k + 1][1]
            C = Ps[k] @ self.F.T @ np.linalg.inv(P_pred_next)
            xs[k] = xs[k] + C @ (xs[k + 1] - x_pred_next)
            Ps[k] = Ps[k] + C @ (Ps[k + 1] - P_pred_next) @ C.T
        return xs

    def filtered_states(self) -> np.ndarray:
        return np.array([h[2] for h in self._hist])
