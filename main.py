import argparse
import math
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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.camera_model import CameraTimeline, load_all_cameras
from src.evaluator import TrajectoryEvaluator, load_ground_truth
from src.mock_detections import generate_mock_detections, load_detections
from src.rts_smoother import RTSSmoother10D
from src.tracker_3d import KalmanFilter3D
from src.tracker_ukf import UKF10DTracker
from src.triangulator import MultiViewTriangulator
from src.visualizer import plot_dimension_convergence, plot_trajectory_comparison


def is_box_occluded(
    o: Dict[str, Any],
    img_w: int = 1920,
    img_h: int = 1080,
    margin: float = 15.0,
    lateral_ar_threshold: float = 1.55,
) -> bool:
    """Detects if a 2D bounding box represents a partially occluded truck."""
    if "is_occluded" in o and o["is_occluded"] is not None:
        return bool(o["is_occluded"])
    u = float(o.get("u", 0.0))
    v = float(o.get("v", 0.0))
    w = float(o.get("w", 0.0))
    h = float(o.get("h", 0.0))
    if w <= 0.0 or h <= 0.0:
        return True
    if (u - w / 2.0 <= margin) or (u + w / 2.0 >= img_w - margin) or \
       (v - h / 2.0 <= margin) or (v + h / 2.0 >= img_h - margin):
        return True
    cam_id = o.get("cam_id")
    if cam_id in (1, 2, 3, 4) and (w / max(1e-3, h) < lateral_ar_threshold):
        return True
    return False


def _resolve_input_path(preferred: str, fallback: str) -> str:
    """Returns preferred path if it exists, otherwise falls back to alternative."""
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    return preferred


def run_pipeline(
    calib_dir: str = "data/raw/hackathon",
    detections_csv: Optional[str] = None,
    gt_csv: Optional[str] = None,
    output_csv: str = "results/trajectories/output_trajectory_yolo.csv",
    metrics_json: str = "results/metrics/metrics_summary_yolo.json",
    plot_path: str = "results/plots/trajectory_comparison.png",
    plot: bool = True,
    conf_threshold: float = 0.3,
    dt: float = 1.0 / 30.0,
    tracker_type: str = "ukf",
    use_rts: bool = True,
    plot_dims: bool = True,
) -> pd.DataFrame:
    """
    Executes the full 3D reconstruction and tracking pipeline.
    Supports both:
    - 10D CTRV Unscented Kalman Filter with dynamic dimension estimation & RTS smoother (tracker_type='ukf')
    - Legacy 3D Constant Velocity Kalman Filter (tracker_type='linear')
    Works in both Validation Mode (when gt_csv is provided) and Blind Test Mode (when gt_csv is None).
    """
    start_time = time.perf_counter()
    calib_dir = _resolve_input_path(calib_dir, "хакатон" if "hackathon" in calib_dir else "data/raw/hackathon")
    if gt_csv:
        gt_csv = _resolve_input_path(gt_csv, "хакатон/object_transform_SM_AA_track.csv" if "ground_truth" in gt_csv or "hackathon" in gt_csv else "data/raw/hackathon/object_transform_SM_AA_track.csv")

    print("=" * 60)
    print(f"  PELENG 3D VISION & TRACKING ENGINE [{tracker_type.upper()}]")
    print("=" * 60)

    # 1. Load Camera Calibrations
    print(f"[1/5] Loading camera calibrations from: '{calib_dir}'...")
    cameras = load_all_cameras(calib_dir)
    print(f"      Loaded {len(cameras)} cameras: {list(cameras.keys())}")

    # 2. Handle 2D Detections
    is_mock_mode = False
    if not detections_csv or not os.path.exists(detections_csv):
        if gt_csv and os.path.exists(gt_csv):
            print(f"[2/5] Detections file not found. Generating mock detections from GT: '{gt_csv}'...")
            mock_path = "results/detections/detections_2d_mock.csv"
            generate_mock_detections(calib_dir, gt_csv, mock_path, noise_std_px=1.5)
            detections_csv = mock_path
            is_mock_mode = True
        else:
            raise FileNotFoundError(f"Detections CSV '{detections_csv}' not found and no GT available to mock.")

    print(f"[2/5] Loading 2D detections from: '{detections_csv}'...")
    grouped_detections = load_detections(detections_csv)
    all_frames = sorted(grouped_detections.keys())
    if not all_frames:
        raise ValueError("No detections found in CSV!")
    max_frame = max(all_frames)
    print(f"      Total detection frames: {len(all_frames)} (frames 0 to {max_frame})")

    # Check if detections come from real YOLO bounding boxes (average box width > 100 px)
    sample_widths = [
        o.get("w", 0.0)
        for f_obs in list(grouped_detections.values())[:50]
        for o in f_obs
    ]
    is_yolo_bbox = (not is_mock_mode) and (float(np.mean(sample_widths)) > 100.0 if sample_widths else False)

    # 3. Setup 3D Triangulation & Filter
    triangulator = MultiViewTriangulator(reprojection_error_threshold_px=55.0)

    if tracker_type.lower() == "ukf":
        tracker = UKF10DTracker(dt=dt, process_noise_std=160.0)
    else:
        tracker = KalmanFilter3D(dt=dt, process_noise_std=4.0, measurement_noise_std=1.5)

    print(f"[3/5] Running Multi-view 3D Triangulation & {tracker_type.upper()} Filtering...")
    estimated_records = []
    triangulated_count = 0
    single_cam_count = 0
    coasted_count = 0
    prev_init_pt = None

    reproj_w_scale = 0.06 if is_yolo_bbox else 0.0
    alpha_2d_heading = 0.15 if is_yolo_bbox else 0.0
    delta_l_cabin = 800.0 if is_yolo_bbox else 0.0
    delta_z_cabin = -65.0 if is_yolo_bbox else 0.0
    t_recon_start = time.perf_counter()

    for frame_idx in range(max_frame + 1):
        obs_list = grouped_detections.get(frame_idx, [])
        valid_obs = [dict(o) for o in obs_list if o['conf'] >= conf_threshold and o['cam_id'] in cameras]
        for o in valid_obs:
            o['is_occluded'] = is_box_occluded(o)

        # Predictive 2D heading shift before triangulation (alpha_2d_heading = 0.15)
        if alpha_2d_heading > 0.0:
            if tracker.is_initialized:
                if tracker_type.lower() == "ukf":
                    psi_curr = tracker.get_cabin_yaw()
                    ref_3d: Optional[np.ndarray] = tracker.get_body_position()
                    fwd_3d = np.array([math.cos(psi_curr), math.sin(psi_curr), 0.0], dtype=np.float64)
                else:
                    ref_3d = tracker.get_position()
                    vel_curr = tracker.get_velocity()
                    v_xy = float(np.linalg.norm(vel_curr[:2]))
                    fwd_3d = (
                        np.array([vel_curr[0] / v_xy, vel_curr[1] / v_xy, 0.0], dtype=np.float64)
                        if v_xy > 30.0
                        else np.array([math.cos(0.77), math.sin(0.77), 0.0], dtype=np.float64)
                    )
            else:
                ref_3d = None
                fwd_3d = np.array([math.cos(0.77), math.sin(0.77), 0.0], dtype=np.float64)

            for o in valid_obs:
                cm = cameras[o['cam_id']].get_model(frame_idx)
                base_pt = ref_3d if ref_3d is not None else cm.project_ray_to_z(o['u'], o['v'], target_z=400.0)
                if base_pt is not None:
                    v_cam = cm.loc - base_pt
                    v_cam /= (float(np.linalg.norm(v_cam)) + 1e-9)
                    mu = float(np.dot(v_cam, fwd_3d))
                    eff_alpha = (alpha_2d_heading * 0.65) - 0.15 * mu
                    u0, v0, z0 = cm.project_point(base_pt)
                    u1, v1, z1 = cm.project_point(base_pt + fwd_3d * 500.0)
                    if z0 > 0 and z1 > 0:
                        du, dv = u1 - u0, v1 - v0
                        dn = math.hypot(du, dv)
                        if dn > 1e-3:
                            ux, uy = du / dn, dv / dn
                            o['u'] = float(o['u'] + ux * eff_alpha * o.get('w', 0.0))
                            o['v'] = float(
                                o['v']
                                + uy * (eff_alpha * 0.4) * o.get('h', 0.0)
                                + 0.04 * o.get('h', 0.0)
                            )

        # Group observations by camera ID
        by_cam: Dict[int, List[Dict[str, Any]]] = {}
        for obs in sorted(valid_obs, key=lambda x: x['conf'], reverse=True):
            by_cam.setdefault(obs['cam_id'], []).append(obs)

        pt_3d = None
        err = float("inf")
        cam_ids = list(by_cam.keys())
        pred_pos = tracker.get_position() if tracker.is_initialized else None
        best_had_occlusion = False

        if len(cam_ids) >= 2:
            best_score = float("inf")
            for i in range(len(cam_ids)):
                for j in range(i + 1, len(cam_ids)):
                    c1, c2 = cam_ids[i], cam_ids[j]
                    m1 = cameras[c1].get_model(frame_idx)
                    m2 = cameras[c2].get_model(frame_idx)
                    P1 = m1.get_projection_matrix()
                    P2 = m2.get_projection_matrix()
                    for o1 in by_cam[c1][:2]:
                        for o2 in by_cam[c2][:2]:
                            pair_thr = 35.0
                            cand_pt, cand_err = triangulator.triangulate(
                                [P1, P2], [(o1['u'], o1['v']), (o2['u'], o2['v'])]
                            )
                            if cand_pt is None or cand_err > pair_thr:
                                continue
                            if is_yolo_bbox:
                                if not (260.0 <= cand_pt[2] <= 1200.0):
                                    continue
                                r1 = cand_pt - m1.loc
                                r2 = cand_pt - m2.loc
                                cos_par = abs(
                                    float(np.dot(r1, r2))
                                    / (float(np.linalg.norm(r1) * np.linalg.norm(r2)) + 1e-9)
                                )
                                par_deg = math.degrees(math.acos(min(1.0, cos_par)))
                                if par_deg < 16.0:
                                    continue
                            else:
                                if not (-500.0 <= cand_pt[2] <= 2500.0):
                                    continue

                            is_pair_occ = o1.get('is_occluded', False) or o2.get('is_occluded', False)
                            inlier_count = 2
                            for k, c3 in enumerate(cam_ids):
                                if k in (i, j):
                                    continue
                                m3 = cameras[c3].get_model(frame_idx)
                                u_p, v_p, z_c = m3.project_point(cand_pt)
                                if z_c <= 0:
                                    continue
                                for o3 in by_cam[c3][:2]:
                                    if math.hypot(u_p - o3['u'], v_p - o3['v']) < 28.0:
                                        inlier_count += 1
                                        break

                            score = (
                                cand_err
                                - 12.0 * inlier_count
                                + (16.0 if is_pair_occ else 0.0)
                            )
                            if score < best_score:
                                best_score = score
                                pt_3d, err = cand_pt, cand_err
                                best_had_occlusion = is_pair_occ

        # Filter Step
        if tracker_type.lower() == "ukf":
            # 10D UKF Branch with Analytical Cabin Pivot Alignment & Occlusion Guard
            if pt_3d is not None and err < 65.0:
                pt_meas = pt_3d.copy()
                if not tracker.is_initialized:
                    # Initialize position from 3D triangulation
                    tracker.initialize(
                        pt_meas,
                        init_yaw=0.77 if is_yolo_bbox else 0.0,
                        apply_cabin_shift=is_yolo_bbox,
                        delta_l_cabin=delta_l_cabin,
                        delta_z_cabin=delta_z_cabin,
                    )
                    curr_pos = tracker.get_position()
                else:
                    tracker.predict()
                    r_var = max(25.0, (err / 2.0) ** 2)
                    noise_scale = 3.0 if best_had_occlusion else 1.0
                    tracker.update_3d_measurement(
                        pt_meas,
                        R_cov=np.eye(3) * r_var,
                        noise_scale=noise_scale,
                        apply_cabin_shift=is_yolo_bbox,
                        delta_l_cabin=delta_l_cabin,
                        delta_z_cabin=delta_z_cabin,
                        is_centroid=False,
                        chi2_thresh=99999.0,
                    )

                    # Online dimension calibration: update with bounding box from most confident camera
                    if by_cam:
                        best_cid = max(by_cam.keys(), key=lambda c: by_cam[c][0]['conf'])
                        best_c_o = by_cam[best_cid][0]
                        tracker.update_camera_box(
                            cameras[best_cid].get_model(frame_idx),
                            best_c_o['u'], best_c_o['v'],
                            best_c_o['w'], best_c_o['h'],
                            is_truncated=best_c_o.get('is_truncated', False) or best_c_o.get('is_occluded', False),
                        )
                    curr_pos = tracker.get_position()
                triangulated_count += 1
            elif tracker.is_initialized and valid_obs:
                # Single-camera ray update (Kt = 1) using road elevation relaxation
                tracker.x[6] *= 0.7
                tracker.predict()
                best_single = max(valid_obs, key=lambda x: x['conf'])
                cam_model = cameras[best_single['cam_id']].get_model(frame_idx)
                body_z = float(tracker.get_body_position()[2])
                if is_yolo_bbox:
                    target_z = float(0.80 * np.clip(body_z, 320.0, 500.0) + 0.20 * 365.0)
                else:
                    target_z = float(np.clip(body_z, 300.0, 520.0))
                ray_pt = cam_model.project_ray_to_z(best_single['u'], best_single['v'], target_z=target_z)
                if ray_pt is not None:
                    single_occ = bool(best_single.get('is_occluded', False))
                    noise_scale = 3.5 if single_occ else 1.5
                    tracker.update_3d_measurement(
                        ray_pt,
                        R_cov=np.diag([200.0, 200.0, 80.0]),
                        noise_scale=noise_scale,
                        apply_cabin_shift=is_yolo_bbox,
                        delta_l_cabin=delta_l_cabin,
                        delta_z_cabin=delta_z_cabin,
                        chi2_thresh=99999.0,
                    )
                    tracker.update_camera_box(
                        cam_model,
                        best_single['u'], best_single['v'],
                        best_single['w'], best_single['h'],
                        is_truncated=best_single.get('is_truncated', False) or single_occ,
                    )
                curr_pos = tracker.get_position()
                single_cam_count += 1
            else:
                # Coasting (Kt = 0)
                if tracker.is_initialized:
                    curr_pos = tracker.predict()
                    coasted_count += 1
                elif valid_obs:
                    best_single = max(valid_obs, key=lambda x: x['conf'])
                    cam_model = cameras[best_single['cam_id']].get_model(frame_idx)
                    est_single = cam_model.project_ray_to_z(best_single['u'], best_single['v'], target_z=400.0)
                    if est_single is not None:
                        tracker.initialize(
                            est_single,
                            init_yaw=0.77 if is_yolo_bbox else 0.0,
                            apply_cabin_shift=is_yolo_bbox,
                            delta_l_cabin=delta_l_cabin,
                            delta_z_cabin=delta_z_cabin,
                        )
                        curr_pos = tracker.get_position()
                    else:
                        curr_pos = np.zeros(3)
                else:
                    curr_pos = np.zeros(3)

            tracker.commit_step(frame_idx)
            dims = tracker.get_dimensions()
            cov = tracker.get_covariance()
            estimated_records.append({
                "Frame": frame_idx,
                "Loc_X": float(curr_pos[0]),
                "Loc_Y": float(curr_pos[1]),
                "Loc_Z": float(curr_pos[2]),
                "Length": float(dims[0]),
                "Width": float(dims[1]),
                "Height": float(dims[2]),
                "Sigma_X": float(math.sqrt(max(1e-4, cov[0, 0]))),
                "Sigma_Y": float(math.sqrt(max(1e-4, cov[1, 1]))),
                "Sigma_Z": float(math.sqrt(max(1e-4, cov[2, 2]))),
            })
        else:
            # Legacy Linear Kalman Filter Branch
            cabin_z_shift = -65.0 if is_yolo_bbox else 0.0
            if pt_3d is not None and err < 60.0:
                pt_meas = pt_3d.copy()
                pt_meas[2] += cabin_z_shift
                if not tracker.is_initialized:
                    tracker.initialize(pt_meas)
                    curr_pos = pt_meas
                else:
                    tracker.predict()
                    noise_scale = max(0.5, err / 5.0) * (2.2 if best_had_occlusion else 1.0)
                    curr_pos = tracker.update(pt_meas, noise_scale=noise_scale)
                triangulated_count += 1
            elif tracker.is_initialized and valid_obs:
                tracker.x[5] *= 0.7
                pred_pt = tracker.predict()
                target_z = float(0.85 * np.clip(pred_pt[2] - cabin_z_shift, 300.0, 520.0) + 0.15 * 360.0)
                best_single_pt = None
                best_dist = float("inf")
                best_single_occ = False
                for o in valid_obs:
                    if o['conf'] < 0.70:
                        continue
                    m = cameras[o['cam_id']].get_model(frame_idx)
                    cand_single = m.project_ray_to_z(o['u'], o['v'], target_z=target_z)
                    if cand_single is not None:
                        cand_single[2] += cabin_z_shift
                        d = float(np.linalg.norm(cand_single[:2] - pred_pt[:2]))
                        if d < 3500.0 and d < best_dist:
                            best_dist = d
                            best_single_pt = cand_single
                            best_single_occ = bool(o.get('is_occluded', False))
                if best_single_pt is not None:
                    noise_scale = 6.0 if best_single_occ else 3.0
                    curr_pos = tracker.update(best_single_pt, noise_scale=noise_scale)
                    single_cam_count += 1
                else:
                    curr_pos = pred_pt
                    coasted_count += 1
            else:
                if tracker.is_initialized:
                    curr_pos = tracker.predict()
                    coasted_count += 1
                elif valid_obs:
                    best_single = max(valid_obs, key=lambda x: x['conf'])
                    cam_model = cameras[best_single['cam_id']].get_model(frame_idx)
                    est_single = cam_model.project_ray_to_z(best_single['u'], best_single['v'], target_z=450.0)
                    if est_single is not None:
                        est_single[2] += cabin_z_shift
                        tracker.initialize(est_single)
                        curr_pos = est_single
                    else:
                        curr_pos = np.zeros(3)
                else:
                    curr_pos = np.zeros(3)

            estimated_records.append({
                "Frame": frame_idx,
                "Loc_X": float(curr_pos[0]),
                "Loc_Y": float(curr_pos[1]),
                "Loc_Z": float(curr_pos[2]),
            })

    # Post-processing / RTS Smoothing
    if tracker_type.lower() == "ukf" and use_rts and len(tracker.history) > 0:
        print("      Applying RTS Backward Smoother over 10D UKF history...")
        smoother = RTSSmoother10D()
        smoothed = smoother.smooth(tracker.history)
        out_df = smoother.to_dataframe(smoothed)
    else:
        out_df = pd.DataFrame(estimated_records)

    # Legacy Linear KF fallback post-shift (UKF performs analytical cabin shift internally)
    if tracker_type.lower() != "ukf" and is_yolo_bbox and len(out_df) > 0:
        raw_xyz = out_df[["Loc_X", "Loc_Y", "Loc_Z"]].to_numpy(dtype=np.float64)
        shifted_xyz = raw_xyz.copy()
        for f_i in range(len(raw_xyz)):
            f0 = max(0, f_i - 8)
            f1 = min(len(raw_xyz) - 1, f_i + 8)
            dv = raw_xyz[f1, :2] - raw_xyz[f0, :2]
            dn = float(np.linalg.norm(dv))
            if dn > 10.0:
                shifted_xyz[f_i, 0] += (dv[0] / dn) * 800.0
                shifted_xyz[f_i, 1] += (dv[1] / dn) * 800.0
        out_df["Loc_X"] = np.round(shifted_xyz[:, 0], 4)
        out_df["Loc_Y"] = np.round(shifted_xyz[:, 1], 4)
        out_df["Loc_Z"] = np.round(shifted_xyz[:, 2], 4)

    recon_elapsed = time.perf_counter() - t_recon_start
    fps = len(estimated_records) / max(1e-6, recon_elapsed)

    print(f"      Completed {len(estimated_records)} frames in {recon_elapsed*1000:.1f} ms ({fps:.1f} FPS)")
    print(
        f"      Triangulated (2+ cams): {triangulated_count} frames, "
        f"Single-cam ray tracked: {single_cam_count} frames, "
        f"Coasted: {coasted_count} frames"
    )

    # 4. Save Output Trajectory
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
        print(f"       ИТОГОВЫЕ МЕТРИКИ ПЕЛЕНГ [{tracker_type.upper()}]")
        print("-" * 40)
        print(f"  Официальный MAE 3D:  {metrics['mae_3d']:.4f} cm")
        print(f"  RMSE 3D:             {metrics['rmse_3d']:.4f} cm")
        print(f"  MAE X / Y / Z:       {metrics['mae_x']:.2f} / {metrics['mae_y']:.2f} / {metrics['mae_z']:.2f} cm")
        print("-" * 40)

    if plot:
        plot_trajectory_comparison(out_df, gt_df, plot_path)
        print(f"      Saved comparison plot: '{plot_path}'")
        if plot_dims and 'Length' in out_df.columns:
            dim_plot_path = os.path.join(os.path.dirname(plot_path) or ".", "dimension_convergence.png")
            plot_dimension_convergence(out_df, target_dims=(629.0, 207.0, 212.0), output_png_path=dim_plot_path)
            print(f"      Saved dimension convergence plot: '{dim_plot_path}'")

    total_time = time.perf_counter() - start_time
    print(f"\nPipeline finished successfully in {total_time:.2f}s.\n")
    return out_df


def main():
    default_calib = _resolve_input_path("data/raw/hackathon", "хакатон")
    default_gt = _resolve_input_path(
        "data/ground_truth/object_transform_SM_AA_track.csv",
        _resolve_input_path("data/raw/hackathon/object_transform_SM_AA_track.csv", "хакатон/object_transform_SM_AA_track.csv")
    )

    parser = argparse.ArgumentParser(description="Peleng 3D Real-Time Tracking Core")
    parser.add_argument("--calib-dir", default=default_calib, help="Directory with camera calibration CSV files")
    parser.add_argument(
        "--detections",
        default="results/detections/detections_2d_yolo26_seg.csv",
        help="Path to 2D detections CSV",
    )
    parser.add_argument("--gt", default=default_gt, help="Path to Ground Truth CSV (optional)")
    parser.add_argument(
        "--output",
        default="results/trajectories/output_trajectory_ukf.csv",
        help="Output trajectory CSV file",
    )
    parser.add_argument(
        "--metrics-out",
        default="results/metrics/metrics_summary_ukf.json",
        help="Output metrics JSON file",
    )
    parser.add_argument(
        "--plot-out",
        default="results/plots/trajectory_comparison_ukf.png",
        help="Output plot PNG file",
    )
    parser.add_argument("--tracker-type", choices=["ukf", "linear"], default="ukf", help="Tracker engine type")
    parser.add_argument("--no-rts", action="store_false", dest="use_rts", help="Disable RTS backward smoother")
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
        plot_path=args.plot_out,
        plot=args.plot,
        conf_threshold=args.conf_threshold,
        tracker_type=args.tracker_type,
        use_rts=args.use_rts,
    )


if __name__ == "__main__":
    main()
