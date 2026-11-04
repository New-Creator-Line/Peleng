"""
Module 4: 10D Unscented Kalman Filter (CTRV Kinematics + Dynamic Dimension Estimation).
Implements the core 10D tracking filter for the Peleng 3D multi-camera tracking system:
- State vector: x = [X, Y, Z, v, psi, omega, vz, L, W, H]^T (units: cm, cm/s, rad, rad/s)
- Unscented Transform: 21 scaled sigma points with non-negative weights for numerical stability
- Circular mean and angle normalization for heading psi in [-pi, pi)
- Dynamic online dimension estimation (L, W, H) without hardcoding
- 4-Tier Sequential Update: RANSAC 3D points (Kt >= 2), Camera box projection (Kt = 1), Coasting (Kt = 0)
- Complete history caching for RTS Backward Smoothing
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.camera_model import CameraModel
from src.cuboid_model import TruckCuboid3D, project_cuboid_to_camera_bbox


def normalize_angle(angle: float) -> float:
    """Normalizes an angle in radians to [-pi, pi)."""
    a = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return a


def circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Computes circular weighted mean of angles to prevent wrap-around discontinuity."""
    s = float(np.sum(weights * np.sin(angles)))
    c = float(np.sum(weights * np.cos(angles)))
    return math.atan2(s, c)


@dataclass
class FilterStepRecord:
    """Cached filter step for Rauch-Tung-Striebel (RTS) smoothing."""
    frame_idx: int
    x_pred: np.ndarray        # (10,) Prior mean
    P_pred: np.ndarray        # (10, 10) Prior covariance
    x_post: np.ndarray        # (10,) Posterior mean
    P_post: np.ndarray        # (10, 10) Posterior covariance
    P_x_xpred: np.ndarray     # (10, 10) Cross-covariance between x_t and x_{t+1|t}
    delta_l_cabin: float = 0.0
    delta_z_cabin: float = 0.0


def compute_cabin_pivot_shift(
    yaw: float,
    delta_l_cabin: float = 800.0,
    delta_z_cabin: float = -65.0,
) -> np.ndarray:
    """
    Computes analytical 3D shift vector from vehicle body center to cabin pivot (SM_AA_track)
    using heading angle psi (yaw in radians):
      [dX, dY, dZ]^T = R_z(psi) * [delta_l_cabin, 0, delta_z_cabin]^T
    """
    c_psi = math.cos(float(yaw))
    s_psi = math.sin(float(yaw))
    rot_z = np.array([
        [c_psi, -s_psi, 0.0],
        [s_psi,  c_psi, 0.0],
        [0.0,    0.0,   1.0],
    ], dtype=np.float64)
    local_shift = np.array([float(delta_l_cabin), 0.0, float(delta_z_cabin)], dtype=np.float64)
    return rot_z @ local_shift


class UKF10DTracker:
    """
    10-Dimensional Unscented Kalman Filter with CTRV Motion Model
    and dynamic online estimation of vehicle dimensions (L, W, H).
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 160.0,
        alpha: float = 1.0,
        beta: float = 0.0,
        kappa: float = 0.0,
    ) -> None:
        self.dt = float(dt)
        self.process_noise_std = float(process_noise_std)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.n = 10  # State dimension: [X, Y, Z, v, psi, omega, vz, L, W, H]
        self.num_sigmas = 2 * self.n + 1

        self._compute_weights()
        self._init_process_noise()

        self.x = np.zeros(self.n, dtype=np.float64)
        self.P = np.eye(self.n, dtype=np.float64) * 100.0
        self.is_initialized = False
        self.coast_frames = 0
        self.max_coast_frames = 30
        self.history: List[FilterStepRecord] = []
        self._current_frame = 0
        self._cabin_pivot_active: bool = False
        self._cabin_yaw: float = 0.0
        self._last_cabin_offset: np.ndarray = np.zeros(3, dtype=np.float64)
        self._delta_l_cabin: float = 0.0
        self._delta_z_cabin: float = 0.0
        self._body_pos_window: List[np.ndarray] = []

    def _compute_weights(self) -> None:
        n = self.n
        # Using alpha=1.0, kappa=0.0 yields standard symmetric UT with non-negative weights
        self.lambda_param = (self.alpha ** 2) * (n + self.kappa) - n
        c = n + self.lambda_param

        # Mean weights W_m
        self.Wm = np.full(self.num_sigmas, 1.0 / (2.0 * c), dtype=np.float64)
        self.Wm[0] = self.lambda_param / c

        # Covariance weights W_c
        self.Wc = np.full(self.num_sigmas, 1.0 / (2.0 * c), dtype=np.float64)
        self.Wc[0] = self.lambda_param / c + (1.0 - self.alpha ** 2 + self.beta)

    def _init_process_noise(self) -> None:
        """Initializes continuous white noise CTRV process covariance matrix Q."""
        dt = self.dt
        dt2 = dt ** 2
        dt3 = dt ** 3 / 2.0
        dt4 = dt ** 4 / 4.0

        var_a = (self.process_noise_std) ** 2          # Longitudinal acceleration (cm/s^2)^2
        var_omega_dot = (0.5 ** 2) if self.process_noise_std > 0 else 0.0  # Yaw acceleration (rad/s^2)^2
        var_az = (3.0) ** 2                            # Vertical acceleration (cm/s^2)^2
        var_dim = 1e-4                                 # Quasistatic vehicle dimensions (cm^2)

        self.Q = np.zeros((self.n, self.n), dtype=np.float64)
        # Position / velocity in XY
        self.Q[0, 0] = dt4 * var_a
        self.Q[1, 1] = dt4 * var_a
        self.Q[3, 3] = dt2 * var_a
        self.Q[0, 3] = dt3 * var_a
        self.Q[3, 0] = dt3 * var_a
        self.Q[1, 3] = dt3 * var_a
        self.Q[3, 1] = dt3 * var_a

        # Vertical motion (rank-1 PSD block)
        self.Q[2, 2] = dt4 * var_az
        self.Q[6, 6] = dt2 * var_az
        self.Q[2, 6] = dt3 * var_az
        self.Q[6, 2] = dt3 * var_az

        # Yaw and turn rate (rank-1 PSD block)
        self.Q[4, 4] = dt4 * var_omega_dot
        self.Q[5, 5] = dt2 * var_omega_dot
        self.Q[4, 5] = dt3 * var_omega_dot
        self.Q[5, 4] = dt3 * var_omega_dot

        # Dimension quasi-constancy
        self.Q[7, 7] = var_dim
        self.Q[8, 8] = var_dim
        self.Q[9, 9] = var_dim

    def initialize(
        self,
        init_pos: np.ndarray,
        init_yaw: float = 0.0,
        init_speed: float = 0.0,
        init_yaw_rate: float = 0.0,
        init_vz: float = 0.0,
        init_dimensions: Optional[np.ndarray] = None,
        init_pos_std: float = 15.0,
        init_yaw_std: float = math.pi / 2.0,
        init_speed_std: float = 20.0,
        init_yaw_rate_std: float = 0.2,
        apply_cabin_shift: bool = False,
        delta_l_cabin: float = 800.0,
        delta_z_cabin: float = -65.0,
    ) -> None:
        """
        Initializes 10D filter state without hardcoding dimensions.
        Dimensions start from a generic prior with high initial uncertainty.
        """
        pos = np.asarray(init_pos, dtype=np.float64).flatten()
        self.x = np.zeros(self.n, dtype=np.float64)
        self.x[0:3] = pos[:3]
        self.x[3] = float(init_speed)
        self.x[4] = normalize_angle(float(init_yaw))
        self.x[5] = float(init_yaw_rate)
        self.x[6] = float(init_vz)

        # Vehicle dimensions: if not provided, start from generic diffuse vehicle prior
        if init_dimensions is not None:
            dims = np.asarray(init_dimensions, dtype=np.float64).flatten()
            self.x[7:10] = dims[:3]
        else:
            self.x[7] = 500.0  # Length (cm)
            self.x[8] = 200.0  # Width (cm)
            self.x[9] = 200.0  # Height (cm)

        # Initial covariance P
        self.P = np.eye(self.n, dtype=np.float64)
        # Position uncertainty
        self.P[0, 0] = self.P[1, 1] = float(init_pos_std) ** 2
        self.P[2, 2] = min(100.0, float(init_pos_std) ** 2)
        # Speed & turn rate uncertainty
        self.P[3, 3] = float(init_speed_std) ** 2
        self.P[4, 4] = float(init_yaw_std) ** 2
        self.P[5, 5] = float(init_yaw_rate_std) ** 2
        self.P[6, 6] = 25.0  # 5 cm/s vz std
        # Dimension uncertainty: wide prior (sigma_L = 250 cm, sigma_W = 100 cm, sigma_H = 100 cm)
        self.P[7, 7] = 62500.0       # Var(L) = (250 cm)^2
        self.P[8, 8] = 10000.0       # Var(W) = (100 cm)^2
        self.P[9, 9] = 10000.0       # Var(H) = (100 cm)^2

        self.is_initialized = True
        self.coast_frames = 0
        self.history.clear()
        self._current_frame = 0
        self._body_pos_window = [pos[:3].copy()]

        x_body = self.x.copy()
        self._current_step_prior = (x_body, self.P.copy(), np.eye(self.n, dtype=np.float64) * 1e-4)

        self._cabin_pivot_active = bool(apply_cabin_shift)
        self._cabin_yaw = normalize_angle(float(init_yaw))
        if self._cabin_pivot_active:
            self.x[4] = 0.0
            x_body[4] = 0.0
            self._delta_l_cabin = float(delta_l_cabin)
            self._delta_z_cabin = float(delta_z_cabin)
            self._last_cabin_offset = compute_cabin_pivot_shift(
                self._cabin_yaw, self._delta_l_cabin, self._delta_z_cabin
            )
            self.x[0:3] = pos[:3] + self._last_cabin_offset
        else:
            self._delta_l_cabin = 0.0
            self._delta_z_cabin = 0.0
            self._last_cabin_offset = np.zeros(3, dtype=np.float64)

    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generates 2n+1 scaled sigma points from mean x and covariance P."""
        n = self.n
        gamma = math.sqrt(n + self.lambda_param)

        # Symmetrize P
        P_sym = 0.5 * (P + P.T)
        P_reg = P_sym + np.eye(n) * 1e-6

        try:
            sqrt_P = np.linalg.cholesky(P_reg)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(P_reg)
            eigvals_clip = np.maximum(eigvals, 1e-6)
            sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals_clip))

        A = gamma * sqrt_P  # (10, 10)
        sigmas = np.zeros((self.num_sigmas, n), dtype=np.float64)
        sigmas[0] = x

        for i in range(n):
            sigmas[i + 1] = x + A[:, i]
            sigmas[n + i + 1] = x - A[:, i]

        # Normalize yaw angles and clamp physical dimensions >= 20 cm
        sigmas[:, 4] = [normalize_angle(val) for val in sigmas[:, 4]]
        sigmas[:, 7] = np.maximum(sigmas[:, 7], 20.0)
        sigmas[:, 8] = np.maximum(sigmas[:, 8], 20.0)
        sigmas[:, 9] = np.maximum(sigmas[:, 9], 20.0)

        return sigmas

    def _motion_ctrv(self, sigma_point: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagates a single 10D sigma point through CTRV kinematics
        with second-order Taylor expansion near omega = 0 to prevent numerical blowup.
        """
        X, Y, Z, v, psi, omega, vz, L, W, H = sigma_point
        out = np.zeros(10, dtype=np.float64)

        if abs(omega) > 1e-5:
            d_psi = omega * dt
            new_psi = psi + d_psi
            out[0] = X + (v / omega) * (math.sin(new_psi) - math.sin(psi))
            out[1] = Y + (v / omega) * (-math.cos(new_psi) + math.cos(psi))
            out[4] = normalize_angle(new_psi)
        else:
            # High-precision Taylor expansion around omega = 0
            d_psi = omega * dt
            out[0] = X + v * (dt * math.cos(psi) - 0.5 * omega * (dt ** 2) * math.sin(psi))
            out[1] = Y + v * (dt * math.sin(psi) + 0.5 * omega * (dt ** 2) * math.cos(psi))
            out[4] = normalize_angle(psi + d_psi)

        out[2] = Z + vz * dt
        out[3] = v
        out[5] = omega
        out[6] = vz
        # Quasistatic vehicle dimensions
        out[7] = L
        out[8] = W
        out[9] = H
        return out

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """Executes 10D UKF Prediction step using non-linear CTRV kinematics."""
        if not self.is_initialized:
            return np.zeros(3)

        dt_step = self.dt if dt is None else float(dt)

        if self._cabin_pivot_active:
            self.x[0:3] -= self._last_cabin_offset

        # 1. Generate sigma points from posterior
        sigmas = self._generate_sigma_points(self.x, self.P)

        # 2. Propagate through CTRV motion model
        sigmas_pred = np.zeros_like(sigmas)
        for i in range(self.num_sigmas):
            sigmas_pred[i] = self._motion_ctrv(sigmas[i], dt_step)

        # 3. Compute predicted mean
        x_pred = np.zeros(self.n, dtype=np.float64)
        for d in range(self.n):
            if d == 4:
                # Heading psi: circular mean
                x_pred[4] = circular_mean(sigmas_pred[:, 4], self.Wm)
            else:
                x_pred[d] = float(np.sum(self.Wm * sigmas_pred[:, d]))

        # 4. Compute predicted covariance P_pred and cross-covariance P_x_xpred (for RTS)
        P_pred = np.zeros((self.n, self.n), dtype=np.float64)
        P_x_xpred = np.zeros((self.n, self.n), dtype=np.float64)

        for i in range(self.num_sigmas):
            diff_pred = sigmas_pred[i] - x_pred
            diff_pred[4] = normalize_angle(diff_pred[4])

            diff_prev = sigmas[i] - self.x
            diff_prev[4] = normalize_angle(diff_prev[4])

            P_pred += self.Wc[i] * np.outer(diff_pred, diff_pred)
            P_x_xpred += self.Wc[i] * np.outer(diff_prev, diff_pred)

        P_pred += self.Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        self.x = x_pred
        self.P = P_pred
        self.coast_frames += 1

        # Cache step for RTS (unshifted kinematics)
        self._current_step_prior = (x_pred.copy(), P_pred.copy(), P_x_xpred.copy())

        if self._cabin_pivot_active:
            self._last_cabin_offset = compute_cabin_pivot_shift(
                self._cabin_yaw, self._delta_l_cabin, self._delta_z_cabin
            )
            self.x[0:3] += self._last_cabin_offset

        return self.x[:3].copy()

    def update_3d_measurement(
        self,
        meas_3d: np.ndarray,
        R_cov: Optional[np.ndarray] = None,
        noise_scale: float = 1.0,
        apply_cabin_shift: bool = False,
        delta_l_cabin: float = 800.0,
        delta_z_cabin: float = -65.0,
        is_centroid: bool = False,
        chi2_thresh: float = 99999.0,
    ) -> bool:
        """
        Sequential measurement update from 3D point (e.g. multi-view RANSAC/DLT triangulation)
        with analytical cabin pivot shift along heading psi and dynamic covariance noise scaling:
            [X_cabin, Y_cabin, Z_cabin]^T = [X_meas, Y_meas, Z_meas]^T + R_z(psi) [delta_l_cabin, 0, delta_z_cabin]^T
        """
        if not self.is_initialized:
            self.initialize(
                meas_3d,
                init_yaw=0.77 if apply_cabin_shift else 0.0,
                apply_cabin_shift=apply_cabin_shift,
                delta_l_cabin=delta_l_cabin,
                delta_z_cabin=delta_z_cabin,
            )
            return True

        z = np.asarray(meas_3d, dtype=np.float64).flatten()[:3].copy()
        base_R = np.eye(3, dtype=np.float64) * 50.0 if R_cov is None else np.asarray(R_cov, dtype=np.float64)
        R = base_R * max(float(noise_scale), 1e-6)

        if self._cabin_pivot_active:
            self.x[0:3] -= self._last_cabin_offset
        elif apply_cabin_shift:
            # Direct analytical cabin shift on standalone measurement update
            z = z + compute_cabin_pivot_shift(self.x[4], delta_l_cabin, delta_z_cabin)

        sigmas = self._generate_sigma_points(self.x, self.P)

        sigmas_z = np.zeros((self.num_sigmas, 3), dtype=np.float64)
        for i in range(self.num_sigmas):
            sigmas_z[i, 0] = sigmas[i, 0]
            sigmas_z[i, 1] = sigmas[i, 1]
            z_offset = (sigmas[i, 9] / 2.0) if is_centroid else 0.0
            sigmas_z[i, 2] = sigmas[i, 2] + z_offset

        # Mean predicted observation
        z_pred = np.sum(self.Wm[:, np.newaxis] * sigmas_z, axis=0)

        # Innovation covariance S and cross-covariance P_xz
        S = np.zeros((3, 3), dtype=np.float64)
        Pxz = np.zeros((self.n, 3), dtype=np.float64)

        for i in range(self.num_sigmas):
            dz = sigmas_z[i] - z_pred
            dx = sigmas[i] - self.x
            dx[4] = normalize_angle(dx[4])

            S += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        S += R
        S = 0.5 * (S + S.T)

        y = z - z_pred

        # Mahalanobis gating with Huber soft M-estimator
        try:
            inv_S = np.linalg.inv(S)
            d_m2 = float(y.T @ inv_S @ y)
        except np.linalg.LinAlgError:
            if self._cabin_pivot_active:
                self.x[0:3] += self._last_cabin_offset
            return False

        weight = 1.0
        if d_m2 > chi2_thresh:
            weight = math.sqrt(chi2_thresh / d_m2)
            y = y * weight

        # Kalman gain
        K = Pxz @ inv_S

        # State update
        dx_corr = K @ y
        self.x += dx_corr
        self.x[4] = normalize_angle(self.x[4])

        # Covariance update
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        if self._cabin_pivot_active:
            if apply_cabin_shift:
                self._delta_l_cabin = float(delta_l_cabin)
                self._delta_z_cabin = float(delta_z_cabin)
            self._body_pos_window.append(self.x[:3].copy())
            if len(self._body_pos_window) > 12:
                self._body_pos_window.pop(0)
            if len(self._body_pos_window) >= 8:
                disp = self._body_pos_window[-1][:2] - self._body_pos_window[0][:2]
                dist = float(np.linalg.norm(disp))
                if dist > 120.0:
                    self._cabin_yaw = math.atan2(float(disp[1]), float(disp[0]))
            self._last_cabin_offset = compute_cabin_pivot_shift(
                self._cabin_yaw, self._delta_l_cabin, self._delta_z_cabin
            )
            self.x[0:3] += self._last_cabin_offset

        self.coast_frames = 0
        return True

    def update_3d_point(
        self,
        meas_3d: np.ndarray,
        R_cov: Optional[np.ndarray] = None,
        is_centroid: bool = False,
        chi2_thresh: float = 99999.0,
        noise_scale: float = 1.0,
        apply_cabin_shift: bool = False,
        delta_l_cabin: float = 800.0,
        delta_z_cabin: float = -65.0,
    ) -> bool:
        """
        Sequential measurement update from 3D point (e.g. multi-view RANSAC triangulation).
        Delegates to update_3d_measurement while preserving backwards compatibility.
        """
        return self.update_3d_measurement(
            meas_3d=meas_3d,
            R_cov=R_cov,
            noise_scale=noise_scale,
            apply_cabin_shift=apply_cabin_shift,
            delta_l_cabin=delta_l_cabin,
            delta_z_cabin=delta_z_cabin,
            is_centroid=is_centroid,
            chi2_thresh=chi2_thresh,
        )

    def update_camera_box(
        self,
        cam_model: CameraModel,
        u: float,
        v: float,
        w: float,
        h: float,
        is_truncated: bool = False,
        chi2_thresh: float = 35.0,
    ) -> bool:
        """
        Sequential measurement update using camera 2D observation: z = [u_c, v_c, w_px, h_px]^T.
        Directly adjusts dimensions (L, W, H) and depth along the camera ray.
        """
        if not self.is_initialized or is_truncated:
            return False

        # Verify nominal cuboid visibility before evaluating sigma points
        nom_cuboid = TruckCuboid3D.from_state_10d(self.x)
        if project_cuboid_to_camera_bbox(nom_cuboid, cam_model) is None:
            return False

        z = np.array([u, v, w, h], dtype=np.float64)
        # Pixel variances: center (4 px std) -> 16 px^2, extent (15 px std / 12 px std)
        R = np.diag([16.0, 16.0, 225.0, 150.0])

        sigmas = self._generate_sigma_points(self.x, self.P)
        sigmas_z = np.zeros((self.num_sigmas, 4), dtype=np.float64)

        for i in range(self.num_sigmas):
            cuboid = TruckCuboid3D.from_state_10d(sigmas[i])
            proj = project_cuboid_to_camera_bbox(cuboid, cam_model)
            if proj is not None:
                b = proj[0]
                u_pred = (b[0] + b[2]) / 2.0
                v_pred = (b[1] + b[3]) / 2.0
                w_pred = max(1.0, b[2] - b[0])
                h_pred = max(1.0, b[3] - b[1])
                sigmas_z[i] = [u_pred, v_pred, w_pred, h_pred]
            else:
                # Fallback to current observation to avoid destabilizing filter
                sigmas_z[i] = z

        z_pred = np.sum(self.Wm[:, np.newaxis] * sigmas_z, axis=0)

        S = np.zeros((4, 4), dtype=np.float64)
        Pxz = np.zeros((self.n, 4), dtype=np.float64)

        for i in range(self.num_sigmas):
            dz = sigmas_z[i] - z_pred
            dx = sigmas[i] - self.x
            dx[4] = normalize_angle(dx[4])

            S += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        S += R
        S = 0.5 * (S + S.T)

        y = z - z_pred

        try:
            inv_S = np.linalg.inv(S)
            d_m2 = float(y.T @ inv_S @ y)
        except np.linalg.LinAlgError:
            return False

        weight = 1.0
        if d_m2 > chi2_thresh:
            weight = math.sqrt(chi2_thresh / d_m2)
            y = y * weight

        K = Pxz @ inv_S
        dx_corr = K @ y

        # Update physical dimensions L, W, H (indices 7, 8, 9) to decouple geometry from position
        self.x[7:10] += dx_corr[7:10]

        # Physical sanity bounds for dimensions (cm) - accommodates full semi-truck (2005x465x593 cm)
        self.x[7] = float(np.clip(self.x[7], 200.0, 2400.0))
        self.x[8] = float(np.clip(self.x[8], 150.0, 500.0))
        self.x[9] = float(np.clip(self.x[9], 150.0, 650.0))

        # Update covariance without distorting position/heading blocks
        K_dim = K[7:10, :]
        self.P[7:10, 7:10] -= weight * (K_dim @ S @ K_dim.T)
        self.P[7:10, 7:10] = 0.5 * (self.P[7:10, 7:10] + self.P[7:10, 7:10].T)

        # Enforce strict decoupling between dimensions and kinematics
        self.P[0:7, 7:10] = 0.0
        self.P[7:10, 0:7] = 0.0

        self.coast_frames = 0
        return True

    def commit_step(self, frame_idx: int) -> None:
        """Commits current step state and covariance to the RTS cache history."""
        if hasattr(self, "_current_step_prior"):
            x_pred, P_pred, P_x_xpred = self._current_step_prior
            x_post_unshifted = self.x.copy()
            if self._cabin_pivot_active:
                x_post_unshifted[0:3] -= self._last_cabin_offset
            self.history.append(FilterStepRecord(
                frame_idx=frame_idx,
                x_pred=x_pred,
                P_pred=P_pred,
                x_post=x_post_unshifted,
                P_post=self.P.copy(),
                P_x_xpred=P_x_xpred,
                delta_l_cabin=self._delta_l_cabin if self._cabin_pivot_active else 0.0,
                delta_z_cabin=self._delta_z_cabin if self._cabin_pivot_active else 0.0,
            ))
            self._current_frame = frame_idx

    def get_position(self) -> np.ndarray:
        """Returns current tracking pivot [X, Y, Z] in UE5 coordinates."""
        return self.x[0:3].copy()

    def get_body_position(self) -> np.ndarray:
        """Returns unshifted body center [X_body, Y_body, Z_body] in UE5 coordinates."""
        if self._cabin_pivot_active:
            return (self.x[0:3] - self._last_cabin_offset).copy()
        return self.x[0:3].copy()

    def get_cabin_yaw(self) -> float:
        """Returns current tangent heading angle psi (radians) used for cabin pivot alignment."""
        if self._cabin_pivot_active:
            return float(self._cabin_yaw)
        return float(self.x[4])

    def get_centroid(self) -> np.ndarray:
        """Returns current body centroid [X, Y, Z + H/2] in UE5 coordinates."""
        return np.array([self.x[0], self.x[1], self.x[2] + self.x[9] / 2.0], dtype=np.float64)

    def get_velocity(self) -> np.ndarray:
        """Returns 3D velocity vector in world coordinates [vx, vy, vz]."""
        v_long = self.x[3]
        psi = self.x[4]
        vx = v_long * math.cos(psi)
        vy = v_long * math.sin(psi)
        vz = self.x[6]
        return np.array([vx, vy, vz], dtype=np.float64)

    def get_dimensions(self) -> np.ndarray:
        """Returns estimated vehicle dimensions [L, W, H] in cm."""
        return self.x[7:10].copy()

    def get_cuboid(self) -> TruckCuboid3D:
        """Builds and returns current 3D rigid cuboid."""
        return TruckCuboid3D.from_state_10d(self.x)

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()
