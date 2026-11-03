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
    viewer_3d: bool = True,
    viewer_out: str = "results/peleng_3d_viewer.html",
    use_depth_model: bool = True,
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
    print(f"[1/6] Loading camera calibrations from: '{calib_dir}'...")
    cameras = load_all_cameras(calib_dir)
    print(f"      Loaded {len(cameras)} cameras: {list(cameras.keys())}")

    # 2. Handle 2D Detections
    if not detections_csv or not os.path.exists(detections_csv):
        if gt_csv and os.path.exists(gt_csv):
            print(f"[2/6] Detections file not found. Generating mock detections from GT: '{gt_csv}'...")
            mock_path = "results/detections_2d.csv"
            generate_mock_detections(calib_dir, gt_csv, mock_path, noise_std_px=1.5)
            detections_csv = mock_path
        else:
            raise FileNotFoundError(f"Detections CSV '{detections_csv}' not found and no GT available to mock.")

    print(f"[2/6] Loading 2D detections from: '{detections_csv}'...")
    grouped_detections = load_detections(detections_csv)
    all_frames = sorted(grouped_detections.keys())
    if not all_frames:
        raise ValueError("No detections found in CSV!")
    max_frame = max(all_frames)
    print(f"      Total detection frames: {len(all_frames)} (frames 0 to {max_frame})")

    # 3. Setup 3D Triangulation & Kalman Filter
    triangulator = MultiViewTriangulator(reprojection_error_threshold_px=40.0)
    tracker = KalmanFilter3D(dt=dt, process_noise_std=4.0, measurement_noise_std=1.5)

    print("[3/6] Running Multi-view 3D Triangulation & Kalman Filtering...")
    estimated_records = []
    triangulated_count = 0
    coasted_count = 0

    t_recon_start = time.perf_counter()

    for frame_idx in range(max_frame + 1):
        obs_list = grouped_detections.get(frame_idx, [])
        valid_obs = [o for o in obs_list if o['conf'] >= conf_threshold and o['cam_id'] in cameras]

        # Group observations by camera ID (keep highest confidence per camera first,
        # and support pairwise consensus to reject single-camera false positives)
        by_cam: Dict[int, List[Dict[str, float]]] = {}
        for obs in sorted(valid_obs, key=lambda x: x['conf'], reverse=True):
            by_cam.setdefault(obs['cam_id'], []).append(obs)

        pt_3d = None
        err = float("inf")
        cam_ids = list(by_cam.keys())

        if len(cam_ids) >= 2:
            pred_pos = tracker.x[:3] if tracker.is_initialized else None
            best_score = float("inf")
            for i in range(len(cam_ids)):
                for j in range(i + 1, len(cam_ids)):
                    c1, c2 = cam_ids[i], cam_ids[j]
                    P1 = cameras[c1].get_model(frame_idx).get_projection_matrix()
                    P2 = cameras[c2].get_model(frame_idx).get_projection_matrix()
                    for o1 in by_cam[c1][:2]:
                        for o2 in by_cam[c2][:2]:
                            cand_pt, cand_err = triangulator.triangulate(
                                [P1, P2], [(o1['u'], o1['v']), (o2['u'], o2['v'])]
                            )
                            if cand_pt is None or cand_err > 28.0:
                                continue
                            # Reject unphysical underground or sky points (e.g. false-positive drone)
                            if not (-500.0 <= cand_pt[2] <= 2500.0):
                                continue
                            inlier_Ps = [P1, P2]
                            inlier_uvs = [(o1['u'], o1['v']), (o2['u'], o2['v'])]
                            for k, c3 in enumerate(cam_ids):
                                if k in (i, j):
                                    continue
                                m3 = cameras[c3].get_model(frame_idx)
                                u_p, v_p, z_c = m3.project_point(cand_pt)
                                if z_c <= 0:
                                    continue
                                for o3 in by_cam[c3][:2]:
                                    if np.hypot(u_p - o3['u'], v_p - o3['v']) < 28.0:
                                        inlier_Ps.append(m3.get_projection_matrix())
                                        inlier_uvs.append((o3['u'], o3['v']))
                                        break
                            if len(inlier_Ps) > 2:
                                cand_pt, cand_err = triangulator.triangulate(inlier_Ps, inlier_uvs)
                            if cand_pt is None or cand_err > 35.0:
                                continue
                            score = cand_err - 12.0 * len(inlier_Ps)
                            if pred_pos is not None:
                                dist_pred = float(np.linalg.norm(cand_pt - pred_pos))
                                if dist_pred > 3500.0:
                                    score += dist_pred * 0.05
                            if score < best_score:
                                best_score = score
                                pt_3d, err = cand_pt, cand_err

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
            elif valid_obs:
                best_single = max(valid_obs, key=lambda x: x['conf'])
                cam_model = cameras[best_single['cam_id']].get_model(frame_idx)
                est_single = cam_model.project_ray_to_z(best_single['u'], best_single['v'], target_z=450.0)
                if est_single is not None:
                    tracker.initialize(est_single)
                    curr_pos = est_single
                else:
                    curr_pos = np.zeros(3)
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
    print(f"[4/6] Saved output trajectory to: '{output_csv}'")

    # 5. Evaluation & Plotting
    gt_df = None
    metrics: Optional[Dict[str, float]] = None
    if gt_csv and os.path.exists(gt_csv):
        print(f"[5/6] Evaluating metrics against Ground Truth: '{gt_csv}'...")
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
        print("[5/6] Running in Blind Test mode (no Ground Truth provided).")

    if plot:
        plot_path = "results/trajectory_comparison.png"
        print(f"      Generating Slide 8 comparison plot -> '{plot_path}'...")
        plot_trajectory_comparison(out_df, gt_df, plot_path)
        print(f"      Saved comparison plot: '{plot_path}'")

    # 6. Interactive 3D Terrain & Trajectory Viewer
    if viewer_3d:
        print(f"[6/6] Building 3D Terrain, Depth Point Cloud & Interactive Three.js Viewer -> '{viewer_out}'...")
        try:
            from src.terrain_builder import build_scene_3d_payload
            from src.viewer_3d import export_3d_viewer_html

            scene_payload = build_scene_3d_payload(
                calib_dir=calib_dir,
                estimated_df=out_df,
                ground_truth_df=gt_df,
                grouped_detections=grouped_detections,
                metrics=metrics,
                use_depth_model=use_depth_model,
            )
            export_3d_viewer_html(scene_payload, output_html_path=viewer_out)
            print(f"      Saved interactive 3D WebGL viewer: '{viewer_out}'")
        except ImportError as exc:
            print(f"      [WARN] 3D terrain builder module not yet available ({exc}). Skipping 3D viewer.")
        except Exception as exc:
            print(f"      [WARN] Failed to generate 3D viewer ({exc}).")
    else:
        print("[6/6] 3D Viewer generation skipped (--no-viewer-3d).")

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
    parser.add_argument("--viewer-3d", action="store_true", default=True, help="Generate interactive 3D Three.js HTML viewer")
    parser.add_argument("--no-viewer-3d", action="store_false", dest="viewer_3d", help="Disable 3D viewer generation")
    parser.add_argument("--viewer-out", default="results/peleng_3d_viewer.html", help="Output path for 3D HTML viewer")
    parser.add_argument(
        "--no-depth-model",
        action="store_false",
        dest="use_depth_model",
        default=True,
        help="Disable neural depth model for fast geometric terrain reconstruction",
    )

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
        viewer_3d=args.viewer_3d,
        viewer_out=args.viewer_out,
        use_depth_model=args.use_depth_model,
    )


if __name__ == "__main__":
    main()
