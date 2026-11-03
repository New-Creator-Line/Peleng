import json
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def load_ground_truth(gt_csv_path: str) -> pd.DataFrame:
    """
    Loads ground truth trajectory from object_transform_SM_AA_track.csv (cp1251).
    Robustly repairs Excel date glitch at row 402 ('авг.13') via numeric coercion and interpolation.
    """
    df = pd.read_csv(gt_csv_path, sep=';', encoding='cp1251')

    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    # Convert coordinates to numeric, coercing any corrupted string values to NaN
    for col in ['Loc_X', 'Loc_Y', 'Loc_Z']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        # Linearly interpolate missing/coerced values
        df[col] = df[col].interpolate(method='linear').bfill().ffill()

    df['Frame'] = df['Frame'].astype(int)
    return df


class TrajectoryEvaluator:
    """Official Peleng 3D Trajectory Evaluator."""

    def compute_metrics(
        self,
        estimated_pts: np.ndarray,
        ground_truth_pts: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculates:
        - mae_3d: Official Peleng 3D Euclidean MAE = (1/N) * sum(||p_hat - p_gt||_2)
        - mae_x, mae_y, mae_z
        - rmse_x, rmse_y, rmse_z, rmse_3d
        """
        est = np.asarray(estimated_pts, dtype=np.float64)
        gt = np.asarray(ground_truth_pts, dtype=np.float64)

        if len(est) != len(gt):
            min_len = min(len(est), len(gt))
            est = est[:min_len]
            gt = gt[:min_len]

        diff = est - gt  # (N, 3)
        euclidean_dist = np.linalg.norm(diff, axis=1)  # (N,)

        mae_3d = float(np.mean(euclidean_dist))
        mae_x = float(np.mean(np.abs(diff[:, 0])))
        mae_y = float(np.mean(np.abs(diff[:, 1])))
        mae_z = float(np.mean(np.abs(diff[:, 2])))

        rmse_x = float(np.sqrt(np.mean(diff[:, 0] ** 2)))
        rmse_y = float(np.sqrt(np.mean(diff[:, 1] ** 2)))
        rmse_z = float(np.sqrt(np.mean(diff[:, 2] ** 2)))
        rmse_3d = float(np.sqrt(np.mean(euclidean_dist ** 2)))

        return {
            "mae_3d": mae_3d,
            "mae_x": mae_x,
            "mae_y": mae_y,
            "mae_z": mae_z,
            "rmse_3d": rmse_3d,
            "rmse_x": rmse_x,
            "rmse_y": rmse_y,
            "rmse_z": rmse_z,
            "sample_count": len(est),
        }

    def save_metrics(self, metrics: Dict[str, Any], output_path: str) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
