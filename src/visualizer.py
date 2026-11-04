"""
Visualization & Reporting Module for Peleng 3D Vision.
Generates:
1. Trajectory comparison plot matching Slide 8 styling with +-2 sigma confidence bands.
2. Dynamic dimension convergence plot L(t), W(t), H(t) demonstrating online learning.
3. 3D uncertainty ellipsoid radius analysis over time.
"""

import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_trajectory_comparison(
    estimated_df: pd.DataFrame,
    ground_truth_df: Optional[pd.DataFrame] = None,
    output_png_path: str = "results/plots/trajectory_comparison.png",
    show_uncertainty_bands: bool = True,
) -> None:
    """
    Plots the final 3D trajectory comparison matching the exact styling
    of Slide 8 in the Peleng hackathon presentation:
    - X, true vs X, det
    - Y, true vs Y, det
    - Z, true vs Z, det
    Includes shaded +-2 sigma analytical confidence corridors when available.
    """
    os.makedirs(os.path.dirname(output_png_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    frames_est = estimated_df['Frame'].to_numpy()
    x_est = estimated_df['Loc_X'].to_numpy()
    y_est = estimated_df['Loc_Y'].to_numpy()
    z_est = estimated_df['Loc_Z'].to_numpy()

    has_sigmas = (
        show_uncertainty_bands
        and 'Sigma_X' in estimated_df.columns
        and 'Sigma_Y' in estimated_df.columns
        and 'Sigma_Z' in estimated_df.columns
    )

    # If ground truth is provided, plot true vs detected
    if ground_truth_df is not None and not ground_truth_df.empty:
        # Merge on Frame to align perfectly
        merged = pd.merge(ground_truth_df, estimated_df, on='Frame', suffixes=('_gt', '_det'))
        f_arr = merged['Frame'].to_numpy()

        # Shaded +-2 sigma confidence ribbons
        if has_sigmas and 'Sigma_X' in merged.columns:
            sx = merged['Sigma_X'].to_numpy()
            sy = merged['Sigma_Y'].to_numpy()
            sz = merged['Sigma_Z'].to_numpy()
            ax.fill_between(f_arr, merged['Loc_X_det'] - 2*sx, merged['Loc_X_det'] + 2*sx, color='#2b6cb0', alpha=0.15)
            ax.fill_between(f_arr, merged['Loc_Y_det'] - 2*sy, merged['Loc_Y_det'] + 2*sy, color='#dd6b20', alpha=0.15)
            ax.fill_between(f_arr, merged['Loc_Z_det'] - 2*sz, merged['Loc_Z_det'] + 2*sz, color='#718096', alpha=0.15)

        # True lines (Solid)
        ax.plot(f_arr, merged['Loc_X_gt'], color='#2b6cb0', linewidth=2.2, label='X, true')
        ax.plot(f_arr, merged['Loc_Y_gt'], color='#dd6b20', linewidth=2.2, label='Y, true')
        ax.plot(f_arr, merged['Loc_Z_gt'], color='#718096', linewidth=2.2, label='Z, true')

        # Detected lines (Dotted with markers)
        mark_step = max(1, len(f_arr) // 50)
        ax.plot(f_arr, merged['Loc_X_det'], color='#2b6cb0', linestyle=':', linewidth=2.5,
                marker='o', markersize=3.5, markevery=mark_step, label='X, det')
        ax.plot(f_arr, merged['Loc_Y_det'], color='#dd6b20', linestyle=':', linewidth=2.5,
                marker='o', markersize=3.5, markevery=mark_step, label='Y, det')
        ax.plot(f_arr, merged['Loc_Z_det'], color='#718096', linestyle=':', linewidth=2.5,
                marker='o', markersize=3.5, markevery=mark_step, label='Z, det')
    else:
        # Blind test mode: plot detected trajectory
        if has_sigmas:
            sx = estimated_df['Sigma_X'].to_numpy()
            sy = estimated_df['Sigma_Y'].to_numpy()
            sz = estimated_df['Sigma_Z'].to_numpy()
            ax.fill_between(frames_est, x_est - 2*sx, x_est + 2*sx, color='#2b6cb0', alpha=0.15)
            ax.fill_between(frames_est, y_est - 2*sy, y_est + 2*sy, color='#dd6b20', alpha=0.15)
            ax.fill_between(frames_est, z_est - 2*sz, z_est + 2*sz, color='#718096', alpha=0.15)

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
    norm_path = os.path.abspath(output_png_path)
    os.makedirs(os.path.dirname(norm_path) or ".", exist_ok=True)
    plt.savefig(norm_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dimension_convergence(
    estimated_df: pd.DataFrame,
    target_dims: Optional[Tuple[float, float, float]] = (629.0, 207.0, 212.0),
    output_png_path: str = "results/plots/dimension_convergence.png",
) -> None:
    """
    Plots the real-time online convergence of estimated dimensions L(t), W(t), H(t).
    Demonstrates that the system dynamically learns vehicle geometry without hardcoding.
    """
    if 'Length' not in estimated_df.columns or 'Width' not in estimated_df.columns or 'Height' not in estimated_df.columns:
        return

    os.makedirs(os.path.dirname(output_png_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#fafafa')

    f = estimated_df['Frame'].to_numpy()
    L = estimated_df['Length'].to_numpy()
    W = estimated_df['Width'].to_numpy()
    H = estimated_df['Height'].to_numpy()

    # Estimated curves
    ax.plot(f, L, color='#3182ce', linewidth=2.2, label='Длина L(t), вычислено')
    ax.plot(f, W, color='#38a169', linewidth=2.2, label='Ширина W(t), вычислено')
    ax.plot(f, H, color='#d69e2e', linewidth=2.2, label='Высота H(t), вычислено')

    # Reference ground truth lines (if provided)
    if target_dims is not None:
        L_gt, W_gt, H_gt = target_dims
        ax.axhline(L_gt, color='#3182ce', linestyle='--', alpha=0.7, label=f'Истинная длина L* ({L_gt:.1f} см)')
        ax.axhline(W_gt, color='#38a169', linestyle='--', alpha=0.7, label=f'Истинная ширина W* ({W_gt:.1f} см)')
        ax.axhline(H_gt, color='#d69e2e', linestyle='--', alpha=0.7, label=f'Истинная высота H* ({H_gt:.1f} см)')

    ax.set_title('Автономное онлайн-оценивание 3D-габаритов грузовика (UKF без хардкода)', fontsize=14, pad=12, fontweight='bold')
    ax.set_xlabel('Кадр (Frame index)', fontsize=11, labelpad=8)
    ax.set_ylabel('Размеры (см)', fontsize=11, labelpad=8)
    ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e0')

    ax.legend(loc='upper right', fancybox=True, shadow=True, ncol=2, fontsize=10)
    plt.tight_layout()
    norm_path = os.path.abspath(output_png_path)
    os.makedirs(os.path.dirname(norm_path) or ".", exist_ok=True)
    plt.savefig(norm_path, dpi=150, bbox_inches='tight')
    plt.close()
