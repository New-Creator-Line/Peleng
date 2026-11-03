import argparse
import os
import sys
import time

# Ensure UTF-8 output on Windows consoles to support Cyrillic characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.camera_model import CameraTimeline, load_all_cameras
from src.evaluator import TrajectoryEvaluator, load_ground_truth
from src.mock_detections import generate_mock_detections, load_detections
from src.tracker_3d import KalmanFilter3D
from src.triangulator import MultiViewTriangulator
from src.visualizer import plot_trajectory_comparison


def run_pipeline(
    calib_dir: str,
    detections_csv: Optional[str] = None,
    gt_csv: Optional[str] = None,
    output_csv: str = "results/output_trajectory.csv",
    metrics_json: str = "results/metrics_summary.json",
    plot: bool = True,
    conf_threshold: float = 0.3,
    dt: float = 1.0 / 30.0,
) -> pd.DataFrame:
    """
    Executes the full 3D reconstruction and tracking pipeline.
    Works in both Validation Mode (when gt_csv is provided) and
    Blind Test Mode (when gt_csv is None).
    """
    start_time = time.perf_counter()
    print("=" * 60)
    print("  PELENG 3D VISION & TRACKING ENGINE (C++ / Python)")
    print("=" * 60)

    # 1. Load Camera Calibrations
    print(f"[1/5] Loading camera calibrations from: '{calib_dir}'...")
    cameras = load_all_cameras(calib_dir)
    print(f"      Loaded {len(cameras)} cameras: {list(cameras.keys())}")

    # 2. Handle 2D Detections
    if not detections_csv or not os.path.exists(detections_csv):
        if gt_csv and os.path.exists(gt_csv):
            print(f"[2/5] Detections file not found. Generating mock detections from GT: '{gt_csv}'...")
            mock_path = "results/detections_2d.csv"
            generate_mock_detections(calib_dir, gt_csv, mock_path, noise_std_px=1.5)
            detections_csv = mock_path
        else:
            raise FileNotFoundError(f"Detections CSV '{detections_csv}' not found and no GT available to mock.")

    print(f"[2/5] Loading 2D detections from: '{detections_csv}'...")
    grouped_detections = load_detections(detections_csv)
    all_frames = sorted(grouped_detections.keys())
    if not all_frames:
        raise ValueError("No detections found in CSV!")
    max_frame = max(all_frames)
    print(f"      Total detection frames: {len(all_frames)} (frames 0 to {max_frame})")

    # 3. Setup 3D Triangulation & Kalman Filter
    triangulator = MultiViewTriangulator(reprojection_error_threshold_px=40.0)
    tracker = KalmanFilter3D(dt=dt, process_noise_std=4.0, measurement_noise_std=1.5)

    print("[3/5] Running Multi-view 3D Triangulation & Kalman Filtering...")
    estimated_records = []
    triangulated_count = 0
    coasted_count = 0

    t_recon_start = time.perf_counter()

    for frame_idx in range(max_frame + 1):
        obs_list = grouped_detections.get(frame_idx, [])
        valid_obs = [o for o in obs_list if o['conf'] >= conf_threshold and o['cam_id'] in cameras]

        proj_matrices = []
        points_2d = []

        for obs in valid_obs:
            cam_model = cameras[obs['cam_id']].get_model(frame_idx)
            proj_matrices.append(cam_model.get_projection_matrix())
            points_2d.append((obs['u'], obs['v']))

        pt_3d = None
        err = float("inf")
        if len(proj_matrices) >= 2:
            pt_3d, err = triangulator.triangulate(proj_matrices, points_2d)

        # Kalman Filter Step
        if pt_3d is not None and err < 60.0:
            if not tracker.is_initialized:
                tracker.initialize(pt_3d)
                curr_pos = pt_3d
            else:
                tracker.predict()
                noise_scale = max(0.5, err / 5.0)
                curr_pos = tracker.update(pt_3d, noise_scale=noise_scale)
            triangulated_count += 1
        else:
            # Coast on Kalman prediction (occlusion / single-cam view)
            if tracker.is_initialized:
                curr_pos = tracker.predict()
                coasted_count += 1
            else:
                curr_pos = np.zeros(3)

        estimated_records.append({
            "Frame": frame_idx,
            "Loc_X": round(float(curr_pos[0]), 4),
            "Loc_Y": round(float(curr_pos[1]), 4),
            "Loc_Z": round(float(curr_pos[2]), 4),
        })

    recon_elapsed = time.perf_counter() - t_recon_start
    fps = len(estimated_records) / max(1e-6, recon_elapsed)

    print(f"      Completed {len(estimated_records)} frames in {recon_elapsed*1000:.1f} ms ({fps:.1f} FPS)")
    print(f"      Triangulated: {triangulated_count} frames, Kalman coasted: {coasted_count} frames")

    # 4. Save Output Trajectory
    out_df = pd.DataFrame(estimated_records)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, sep=';', index=False)
    print(f"[4/5] Saved output trajectory to: '{output_csv}'")

    # 5. Evaluation & Plotting
    gt_df = None
    if gt_csv and os.path.exists(gt_csv):
        print(f"[5/5] Evaluating metrics against Ground Truth: '{gt_csv}'...")
        gt_df = load_ground_truth(gt_csv)
        evaluator = TrajectoryEvaluator()

        merged = pd.merge(gt_df, out_df, on='Frame', suffixes=('_gt', '_det'))
        gt_pts = merged[['Loc_X_gt', 'Loc_Y_gt', 'Loc_Z_gt']].to_numpy()
        est_pts = merged[['Loc_X_det', 'Loc_Y_det', 'Loc_Z_det']].to_numpy()

        metrics = evaluator.compute_metrics(est_pts, gt_pts)
        metrics["runtime_ms"] = round(recon_elapsed * 1000.0, 2)
        metrics["fps"] = round(fps, 1)

        evaluator.save_metrics(metrics, metrics_json)
        print(f"      Saved metrics to: '{metrics_json}'")

        print("\n" + "-" * 40)
        print("       ИТОГОВЫЕ МЕТРИКИ ПЕЛЕНГ")
        print("-" * 40)
        print(f"  Официальный MAE 3D:  {metrics['mae_3d']:.4f} units")
        print(f"  RMSE 3D:             {metrics['rmse_3d']:.4f} units")
        print(f"  MAE X / Y / Z:       {metrics['mae_x']:.2f} / {metrics['mae_y']:.2f} / {metrics['mae_z']:.2f}")
        print(f"  RMSE X / Y / Z:      {metrics['rmse_x']:.2f} / {metrics['rmse_y']:.2f} / {metrics['rmse_z']:.2f}")
        print(f"  Скорость обработки:  {metrics['fps']:.1f} FPS")
        print("-" * 40)
    else:
        print("[5/5] Running in Blind Test mode (no Ground Truth provided).")

    if plot:
        plot_path = "results/trajectory_comparison.png"
        print(f"      Generating Slide 8 comparison plot -> '{plot_path}'...")
        plot_trajectory_comparison(out_df, gt_df, plot_path)
        print(f"      Saved comparison plot: '{plot_path}'")

    total_time = time.perf_counter() - start_time
    print(f"\nPipeline finished successfully in {total_time:.2f}s.\n")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Peleng 3D Real-Time Tracking Core")
    parser.add_argument("--calib-dir", default="хакатон", help="Directory with camera calibration CSV files")
    parser.add_argument("--detections", default="results/detections_2d.csv", help="Path to 2D detections CSV")
    parser.add_argument("--gt", default="хакатон/object_transform_SM_AA_track.csv", help="Path to Ground Truth CSV (optional)")
    parser.add_argument("--output", default="results/output_trajectory.csv", help="Output trajectory CSV file")
    parser.add_argument("--metrics-out", default="results/metrics_summary.json", help="Output metrics JSON file")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate Slide 8 comparison plot")
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="Disable plotting")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Confidence threshold for 2D detections")

    args = parser.parse_args()

    # If --gt is explicitly passed as 'none' or empty, treat as Blind Test mode
    gt_file = args.gt if (args.gt and args.gt.lower() != 'none') else None

    run_pipeline(
        calib_dir=args.calib_dir,
        detections_csv=args.detections,
        gt_csv=gt_file,
        output_csv=args.output,
        metrics_json=args.metrics_out,
        plot=args.plot,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
