import os
import numpy as np
import pytest
from src.evaluator import TrajectoryEvaluator, load_ground_truth


def test_peleng_mae_calculation():
    """Verify official 3D Euclidean MAE formula."""
    evaluator = TrajectoryEvaluator()

    # True trajectory
    gt = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],  # norm = 5.0
        [1.0, 2.0, 2.0],  # norm = 3.0
    ])

    # Estimated trajectory (error: [0, 0, 0], [3, 4, 0] offset, [1, 2, 2] offset)
    est = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 8.0, 0.0],  # dist = 5.0
        [2.0, 4.0, 4.0],  # dist = 3.0
    ])

    metrics = evaluator.compute_metrics(est, gt)

    # Expected MAE_3D = (0 + 5 + 3) / 3 = 8 / 3 = 2.6667
    assert np.isclose(metrics["mae_3d"], 8.0 / 3.0)
    assert "mae_x" in metrics
    assert "mae_y" in metrics
    assert "mae_z" in metrics
    assert "rmse_3d" in metrics


def test_load_ground_truth_excel_repair():
    """Verify loading object_transform_SM_AA_track.csv repairs row 402 ('авг.13')."""
    gt_path = (
        "data/ground_truth/object_transform_SM_AA_track.csv"
        if os.path.exists("data/ground_truth/object_transform_SM_AA_track.csv")
        else (
            "data/raw/hackathon/object_transform_SM_AA_track.csv"
            if os.path.exists("data/raw/hackathon/object_transform_SM_AA_track.csv")
            else "хакатон/object_transform_SM_AA_track.csv"
        )
    )
    gt_df = load_ground_truth(gt_path)
    assert len(gt_df) >= 1000
    # Frame 402 should be numeric, not NaN or string
    val_402 = gt_df.loc[gt_df["Frame"] == 402, "Loc_X"].values[0]
    assert isinstance(val_402, (float, np.floating))
    assert not np.isnan(val_402)
    # Surrounding values are -34.37 and 51.98, so value should be around +8.8
    assert -30.0 < val_402 < 50.0
