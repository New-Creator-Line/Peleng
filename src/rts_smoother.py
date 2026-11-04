"""
Module 4: Rauch-Tung-Striebel (RTS) Backward Smoother for 10D UKF Trajectory.
Eliminates camera handover phase lag, transient switching spikes, and yields optimal
zero-lag state and analytical covariance estimates across all recorded video frames.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.tracker_ukf import FilterStepRecord, normalize_angle


class RTSSmoother10D:
    """
    Nonlinear Rauch-Tung-Striebel Backward Smoother for 10D UKF.
    Runs backward from t = T-1 to 0 over cached prior and posterior filter records.
    """

    def __init__(self) -> None:
        pass

    def smooth(self, history: List[FilterStepRecord]) -> List[Dict[str, Any]]:
        """
        Executes RTS backward pass.
        Returns list of smoothed state dictionaries with analytical error ribbons (+-sigma).
        """
        T = len(history)
        if T == 0:
            return []

        # Arrays to hold smoothed trajectory
        x_smooth = [np.zeros(10, dtype=np.float64) for _ in range(T)]
        P_smooth = [np.zeros((10, 10), dtype=np.float64) for _ in range(T)]

        # Terminal boundary condition at t = T - 1: smoothed equals filtered
        x_smooth[-1] = history[-1].x_post.copy()
        P_smooth[-1] = history[-1].P_post.copy()

        # Backward iteration: t = T - 2 down to 0
        for t in range(T - 2, -1, -1):
            curr_rec = history[t]
            next_rec = history[t + 1]

            x_post_t = curr_rec.x_post
            P_post_t = curr_rec.P_post

            x_pred_next = next_rec.x_pred
            P_pred_next = next_rec.P_pred
            P_cross = next_rec.P_x_xpred  # Cross-covariance between x_t and x_{t+1|t}

            # Smoother gain C_t = P_{x_t, x_{t+1}} * P_{t+1|t}^-1
            P_pred_sym = 0.5 * (P_pred_next + P_pred_next.T) + np.eye(10) * 1e-6
            try:
                # Solve C_t * P_pred_sym = P_cross -> P_pred_sym.T * C_t.T = P_cross.T
                Ct = np.linalg.solve(P_pred_sym.T, P_cross.T).T
            except np.linalg.LinAlgError:
                inv_P_pred = np.linalg.pinv(P_pred_sym)
                Ct = P_cross @ inv_P_pred

            # State difference delta = x_{t+1|T} - x_{t+1|t}
            delta_x = x_smooth[t + 1] - x_pred_next
            # Unwind yaw angle difference
            delta_x[4] = normalize_angle(delta_x[4])

            # Smoothed state: x_{t|T} = x_{t|t} + C_t * delta
            x_t_T = x_post_t + Ct @ delta_x
            x_t_T[4] = normalize_angle(x_t_T[4])

            # Smoothed covariance: P_{t|T} = P_{t|t} + C_t * (P_{t+1|T} - P_{t+1|t}) * C_t^T
            delta_P = P_smooth[t + 1] - P_pred_next
            P_t_T = P_post_t + Ct @ delta_P @ Ct.T
            P_t_T = 0.5 * (P_t_T + P_t_T.T)

            # Enforce positive variance guarantee: P_{t|T} <= P_{t|t}
            for d in range(10):
                if P_t_T[d, d] <= 0.0 or P_t_T[d, d] > P_post_t[d, d]:
                    P_t_T[d, d] = P_post_t[d, d]

            x_smooth[t] = x_t_T
            P_smooth[t] = P_t_T

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
