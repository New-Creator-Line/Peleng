import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_trajectory_comparison(
    estimated_df: pd.DataFrame,
    ground_truth_df: Optional[pd.DataFrame],
    output_png_path: str = "results/trajectory_comparison.png",
) -> None:
    """
    Plots the final 3D trajectory comparison matching the exact styling
    of Slide 8 in the Peleng hackathon presentation:
    - X, true vs X, det
    - Y, true vs Y, det
    - Z, true vs Z, det
    """
    os.makedirs(os.path.dirname(output_png_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    frames_est = estimated_df['Frame'].to_numpy()
    x_est = estimated_df['Loc_X'].to_numpy()
    y_est = estimated_df['Loc_Y'].to_numpy()
    z_est = estimated_df['Loc_Z'].to_numpy()

    # If ground truth is provided, plot true vs detected
    if ground_truth_df is not None and not ground_truth_df.empty:
        # Merge on Frame to align perfectly
        merged = pd.merge(ground_truth_df, estimated_df, on='Frame', suffixes=('_gt', '_det'))
        f_arr = merged['Frame'].to_numpy()

        # True lines (Solid)
        ax.plot(f_arr, merged['Loc_X_gt'], color='#2b6cb0', linewidth=2.2, label='X, true')
        ax.plot(f_arr, merged['Loc_Y_gt'], color='#dd6b20', linewidth=2.2, label='Y, true')
        ax.plot(f_arr, merged['Loc_Z_gt'], color='#718096', linewidth=2.2, label='Z, true')

        # Detected lines (Dotted with markers)
        ax.plot(f_arr, merged['Loc_X_det'], color='#2b6cb0', linestyle=':', linewidth=2.5,
                marker='o', markersize=3.5, markevery=max(1, len(f_arr)//50), label='X, det')
        ax.plot(f_arr, merged['Loc_Y_det'], color='#dd6b20', linestyle=':', linewidth=2.5,
                marker='o', markersize=3.5, markevery=max(1, len(f_arr)//50), label='Y, det')
        ax.plot(f_arr, merged['Loc_Z_det'], color='#718096', linestyle=':', linewidth=2.5,
                marker='o', markersize=3.5, markevery=max(1, len(f_arr)//50), label='Z, det')
    else:
        # Blind test mode: plot detected trajectory
        ax.plot(frames_est, x_est, color='#2b6cb0', linewidth=2.0, label='X, det')
        ax.plot(frames_est, y_est, color='#dd6b20', linewidth=2.0, label='Y, det')
        ax.plot(frames_est, z_est, color='#718096', linewidth=2.0, label='Z, det')

    ax.set_title('Итоговый результат: Восстановление 3D-траектории во времени (Пеленг)', fontsize=15, pad=15, fontweight='bold')
    ax.set_xlabel('Кадр (Frame index)', fontsize=12, labelpad=10)
    ax.set_ylabel('Координаты в пространстве UE (Loc_X, Loc_Y, Loc_Z)', fontsize=12, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e0')

    # Legend at the bottom center (like Slide 8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              fancybox=True, shadow=True, ncol=6, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150, bbox_inches='tight')
    plt.close()
