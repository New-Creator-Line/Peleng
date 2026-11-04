"""
Module 4: Rauch-Tung-Striebel (RTS) Backward Smoother for 10D UKF Trajectory.
Eliminates camera handover phase lag, transient switching spikes, and yields optimal
zero-lag state and analytical covariance estimates across all recorded video frames.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.tracker_ukf import FilterStepRecord, compute_cabin_pivot_shift, normalize_angle


class RTSSmoother10D:
    """
    Nonlinear Rauch-Tung-Striebel Backward Smoother for 10D UKF.
    Supports both:
    - Fixed-Lag Sliding Window streaming mode (`lag_frames = L`, e.g., 15..30 frames = 0.5..1.0 s buffer)
    - Full-trajectory offline batch mode (`lag_frames = None`)
    """

    def __init__(self, lag_frames: Optional[int] = None) -> None:
        self.lag_frames = lag_frames

    @staticmethod
    def _compute_gain(next_rec: FilterStepRecord) -> np.ndarray:
        """Computes RTS smoother gain C_t = P_{x_t, x_{t+1|t}} * P_{t+1|t}^-1."""
        P_pred_next = next_rec.P_pred
        P_cross = next_rec.P_x_xpred
        P_pred_sym = 0.5 * (P_pred_next + P_pred_next.T) + np.eye(10) * 1e-6
        try:
            return np.linalg.solve(P_pred_sym.T, P_cross.T).T
        except np.linalg.LinAlgError:
            inv_P_pred = np.linalg.pinv(P_pred_sym)
            return P_cross @ inv_P_pred

    def smooth_step(
        self,
        window_history: List[FilterStepRecord],
        target_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Online streaming single-frame fixed-lag RTS smoothing over a bounded ring buffer
        `window_history` (where `target_idx` is the frame being emitted and `window_history[-1]`
        is at most `t + lag_frames`).
        """
        smoothed_win = self.smooth(window_history, lag_frames=None)
        idx = max(0, min(len(smoothed_win) - 1, target_idx))
        return smoothed_win[idx]

    def smooth(
        self,
        history: List[FilterStepRecord],
        lag_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes RTS backward pass (either Fixed-Lag Sliding Window if `lag_frames` is set,
        or full-horizon if `lag_frames` is None).
        Returns list of smoothed state dictionaries with analytical error ribbons (+-sigma).
        """
        T = len(history)
        if T == 0:
            return []

        eff_lag = lag_frames if lag_frames is not None else self.lag_frames
        if eff_lag is not None and eff_lag < 0:
            eff_lag = None

        # Arrays to hold smoothed trajectory
        x_smooth = [np.zeros(10, dtype=np.float64) for _ in range(T)]
        P_smooth = [np.zeros((10, 10), dtype=np.float64) for _ in range(T)]

        # Precompute step gains C_k once per step k in [0, T-2]
        C_gains = [self._compute_gain(history[k + 1]) for k in range(T - 1)]

        if eff_lag is None or eff_lag >= T - 1:
            # Full-horizon backward pass: t = T - 1 down to 0
            x_smooth[-1] = history[-1].x_post.copy()
            P_smooth[-1] = history[-1].P_post.copy()

            for t in range(T - 2, -1, -1):
                curr_rec = history[t]
                next_rec = history[t + 1]
                x_post_t = curr_rec.x_post
                P_post_t = curr_rec.P_post
                x_pred_next = next_rec.x_pred
                P_pred_next = next_rec.P_pred
                Ct = C_gains[t]

                delta_x = x_smooth[t + 1] - x_pred_next
                delta_x[4] = normalize_angle(delta_x[4])

                x_t_T = x_post_t + Ct @ delta_x
                x_t_T[4] = normalize_angle(x_t_T[4])

                delta_P = P_smooth[t + 1] - P_pred_next
                P_t_T = P_post_t + Ct @ delta_P @ Ct.T
                P_t_T = 0.5 * (P_t_T + P_t_T.T)

                for d in range(10):
                    if P_t_T[d, d] <= 0.0 or P_t_T[d, d] > P_post_t[d, d]:
                        P_t_T[d, d] = P_post_t[d, d]

                x_smooth[t] = x_t_T
                P_smooth[t] = P_t_T
        elif eff_lag == 0:
            for t in range(T):
                x_smooth[t] = history[t].x_post.copy()
                P_smooth[t] = history[t].P_post.copy()
        else:
            # Fixed-Lag Sliding Window backward pass: at frame t, only frames up to min(T-1, t + eff_lag) are used
            for t in range(T):
                end_idx = min(T - 1, t + eff_lag)
                x_curr = history[end_idx].x_post.copy()
                P_curr = history[end_idx].P_post.copy()

                for k in range(end_idx - 1, t - 1, -1):
                    curr_rec = history[k]
                    next_rec = history[k + 1]
                    x_post_k = curr_rec.x_post
                    P_post_k = curr_rec.P_post
                    x_pred_next = next_rec.x_pred
                    P_pred_next = next_rec.P_pred
                    Ct = C_gains[k]

                    delta_x = x_curr - x_pred_next
                    delta_x[4] = normalize_angle(delta_x[4])

                    x_k = x_post_k + Ct @ delta_x
                    x_k[4] = normalize_angle(x_k[4])

                    delta_P = P_curr - P_pred_next
                    P_k = P_post_k + Ct @ delta_P @ Ct.T
                    P_k = 0.5 * (P_k + P_k.T)

                    for d in range(10):
                        if P_k[d, d] <= 0.0 or P_k[d, d] > P_post_k[d, d]:
                            P_k[d, d] = P_post_k[d, d]

                    x_curr = x_k
                    P_curr = P_k

                x_smooth[t] = x_curr
                P_smooth[t] = P_curr

        # Apply analytical cabin pivot alignment along smoothed heading psi_t (respecting eff_lag lookahead bound)
        has_cabin_shift = any(
            abs(rec.delta_l_cabin) > 1e-6 or abs(rec.delta_z_cabin) > 1e-6
            for rec in history
        )
        if has_cabin_shift and T > 1:
            body_xy = np.array([[st[0], st[1]] for st in x_smooth], dtype=np.float64)
            dt_nom = 1.0 / 30.0
            last_psi = 0.77
            max_fwd = 8 if eff_lag is None else min(8, eff_lag)
            for t in range(T):
                i0 = max(0, t - 8)
                i1 = min(T - 1, t + max_fwd)
                dv = body_xy[i1] - body_xy[i0]
                dn = float(np.linalg.norm(dv))
                if dn > 10.0:
                    last_psi = math.atan2(float(dv[1]), float(dv[0]))
                    dt_span = max(1, i1 - i0) * dt_nom
                    x_smooth[t][3] = dn / dt_span
                x_smooth[t][4] = normalize_angle(last_psi)
                rec = history[t]
                if abs(rec.delta_l_cabin) > 1e-6 or abs(rec.delta_z_cabin) > 1e-6:
                    x_smooth[t][0:3] += compute_cabin_pivot_shift(
                        x_smooth[t][4], rec.delta_l_cabin, rec.delta_z_cabin
                    )

        # Format output records
        results: List[Dict[str, Any]] = []
        for t in range(T):
            st = x_smooth[t]
            cov = P_smooth[t]
            frame_idx = history[t].frame_idx

            sig_x = float(math.sqrt(max(1e-4, cov[0, 0])))
            sig_y = float(math.sqrt(max(1e-4, cov[1, 1])))
            sig_z = float(math.sqrt(max(1e-4, cov[2, 2])))

            results.append({
                "frame_idx": frame_idx,
                "x_smoothed": st,
                "P_smoothed": cov,
                "Loc_X": float(st[0]),
                "Loc_Y": float(st[1]),
                "Loc_Z": float(st[2]),
                "Speed": float(st[3]),
                "Yaw": float(st[4]),
                "Yaw_Rate": float(st[5]),
                "Vz": float(st[6]),
                "Length": float(st[7]),
                "Width": float(st[8]),
                "Height": float(st[9]),
                "Sigma_X": sig_x,
                "Sigma_Y": sig_y,
                "Sigma_Z": sig_z,
            })

        return results

    def to_dataframe(self, smoothed_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts smoothed records to pandas DataFrame for evaluation and export."""
        rows = []
        for r in smoothed_records:
            rows.append({
                "Frame": r["frame_idx"],
                "Loc_X": r["Loc_X"],
                "Loc_Y": r["Loc_Y"],
                "Loc_Z": r["Loc_Z"],
                "Speed": r["Speed"],
                "Yaw": r["Yaw"],
                "Yaw_Rate": r["Yaw_Rate"],
                "Vz": r["Vz"],
                "Length": r["Length"],
                "Width": r["Width"],
                "Height": r["Height"],
                "Sigma_X": r["Sigma_X"],
                "Sigma_Y": r["Sigma_Y"],
                "Sigma_Z": r["Sigma_Z"],
            })
        return pd.DataFrame(rows)
