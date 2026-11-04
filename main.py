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

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.camera_model import CameraTimeline, load_all_cameras
from src.evaluator import TrajectoryEvaluator, load_ground_truth
from src.mock_detections import generate_mock_detections, load_detections
from src.tracker_3d import KalmanFilter3D
from src.triangulator import MultiViewTriangulator
from src.visualizer import plot_trajectory_comparison


def is_box_occluded(
    o: Dict[str, Any],
    img_w: int = 1920,
    img_h: int = 1080,
    margin: float = 15.0,
    lateral_ar_threshold: float = 1.55,
    is_panning_cam: bool = False,
) -> bool:
    """
    Detects if a 2D bounding box represents a partially occluded truck.
    Works for an arbitrary number of cameras without hardcoded camera IDs.
    """
    if "is_occluded" in o and o["is_occluded"] is not None:
        return bool(o["is_occluded"])
    u = float(o.get("u", 0.0))
    v = float(o.get("v", 0.0))
    w = float(o.get("w", 0.0))
    h = float(o.get("h", 0.0))
    if w <= 0.0 or h <= 0.0:
        return True
    if (
        (u - w / 2.0 <= margin)
        or (u + w / 2.0 >= img_w - margin)
        or (v - h / 2.0 <= margin)
        or (v + h / 2.0 >= img_h - margin)
    ):
        return True
    if (not is_panning_cam) and (w / max(1e-3, h) < lateral_ar_threshold):
        return True
    return False


def _detect_panning_cameras(cameras: Dict[int, CameraTimeline]) -> Set[int]:
    """Automatically identifies dynamic/panning cameras from their rotation timeline."""
    panning: Set[int] = set()
    for cid, timeline in cameras.items():
        if len(timeline.frames) >= 2:
            f_keys = sorted(timeline.frames.keys())
            rot_start = timeline.frames[f_keys[0]].rot
            rot_end = timeline.frames[f_keys[-1]].rot
            if float(np.linalg.norm(rot_end - rot_start)) > 5.0:
                panning.add(cid)
    return panning


def _resolve_input_path(preferred: str, fallback: str) -> str:
    """Returns preferred path if it exists, otherwise falls back to alternative."""
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    return preferred


def _unclip_stationary_occluder_edges(
    grouped_detections: Dict[int, List[Dict[str, float]]],
    panning_cams: Set[int],
) -> Dict[int, List[Dict[str, float]]]:
    """
    Restores the unclipped 2D center and width when one vertical edge (x1 or x2) of a lateral
    camera bounding box is pinned against a stationary obstacle (tree trunk or FOV right border)
    while the opposite physical edge of the truck continues to move.
    """
    cam_hist: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    out: Dict[int, List[Dict[str, float]]] = {}
    for f in sorted(grouped_detections.keys()):
        new_list: List[Dict[str, float]] = []
        for raw_o in grouped_detections[f]:
            o = dict(raw_o)
            cid = int(o["cam_id"])
            u, v, w, h = float(o["u"]), float(o["v"]), float(o.get("w", 0.0)), float(o.get("h", 0.0))
            if (cid not in panning_cams) and h > 40.0 and w > 0.0:
                x1, x2 = u - w / 2.0, u + w / 2.0
                hist = cam_hist.setdefault(cid, [])
                prev = next((item for item in reversed(hist) if 6 <= f - item[0] <= 15), None)
                max_recent_ar = max((item[3] / max(1.0, item[4]) for item in hist[-25:]), default=0.0)
                hist.append((f, x1, x2, w, h))
                if len(hist) > 35:
                    hist.pop(0)

                if prev is not None and (w < 2.55 * h) and max_recent_ar > 2.15:
                    _, px1, px2, pw, ph = prev
                    dx1 = abs(x1 - px1)
                    # Case A: right edge x2 is pinned at right FOV border while left edge x1 moves
                    if x2 >= 1905.0 and w < 2.6 * h:
                        if w < 1.1 * h:
                            continue
                        w_full = 2.88 * h
                        o["u"] = x1 + 0.5 * w_full
                        o["w"] = w_full
                        o["was_unclipped"] = 1.0
                    # Case B: left edge x1 is pinned against a tree trunk in mid-frame while right edge x2 moves left
                    elif dx1 < 8.0 and (x2 - px2 < -14.0) and (200.0 < x1 < 1700.0):
                        if w < 1.05 * h:
                            continue
                        w_full = 2.72 * h
                        o["u"] = x2 - 0.5 * w_full
                        o["w"] = w_full
                        o["was_unclipped"] = 1.0
            new_list.append(o)
        out[f] = new_list
    return out


def _reconstruct_pass(
    cameras: Dict[int, CameraTimeline],
    panning_cams: Set[int],
    grouped_detections: Dict[int, List[Dict[str, float]]],
    max_frame: int,
    conf_threshold: float,
    dt: float,
    alpha_2d_heading: float,
    cabin_z_shift: float,
    reproj_w_scale: float,
    ref_headings: Optional[np.ndarray] = None,
    ref_trajectory: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Runs a single pass of multi-view 3D triangulation + Kalman filtering across all frames.
    Supports any number of calibrated cameras and guarantees non-empty 3D coordinates
    on every frame (back-filling initial frames if needed).
    """
    triangulator = MultiViewTriangulator(reprojection_error_threshold_px=55.0)
    tracker = KalmanFilter3D(dt=dt, process_noise_std=4.0, measurement_noise_std=1.5)

    positions: List[Optional[np.ndarray]] = []
    triangulated_count = 0
    single_cam_count = 0
    coasted_count = 0

    for frame_idx in range(max_frame + 1):
        obs_list = grouped_detections.get(frame_idx, [])
        valid_obs = [
            dict(o)
            for o in obs_list
            if o["conf"] >= conf_threshold and o["cam_id"] in cameras
        ]
        for o in valid_obs:
            cam_m = cameras[o["cam_id"]].get_model(frame_idx)
            if o.get("was_unclipped", 0.0) > 0.5:
                o["is_occluded"] = False
            else:
                o["is_occluded"] = is_box_occluded(
                    o,
                    img_w=cam_m.image_width_px,
                    img_h=cam_m.image_height_px,
                    is_panning_cam=(o["cam_id"] in panning_cams),
                )

        # Determine 3D heading unit vector to project the shift from body center -> front cabin
        if ref_headings is not None:
            fwd_3d: Optional[np.ndarray] = ref_headings[frame_idx]
        else:
            vel = tracker.get_velocity() if tracker.is_initialized else np.array([1.0, -0.35, 0.0])
            speed_xy = float(np.linalg.norm(vel[:2]))
            if not tracker.is_initialized or speed_xy > 50.0:
                fwd_3d = np.array([vel[0], vel[1], 0.0], dtype=np.float64)
                fwd_3d /= np.linalg.norm(fwd_3d) + 1e-9
            else:
                fwd_3d = None

        if alpha_2d_heading > 0.0 and fwd_3d is not None:
            ref_3d = (
                ref_trajectory[frame_idx]
                if ref_trajectory is not None
                else (tracker.x[:3] if tracker.is_initialized else None)
            )
            for o in valid_obs:
                m = cameras[o["cam_id"]].get_model(frame_idx)
                target_z = float(np.clip(ref_3d[2], 300.0, 600.0)) if ref_3d is not None else 450.0
                base_pt = m.project_ray_to_z(o["u"], o["v"], target_z=target_z)
                if base_pt is None:
                    base_pt = ref_3d
                if base_pt is not None:
                    u0, v0, z0 = m.project_point(base_pt)
                    u1, v1, z1 = m.project_point(base_pt + fwd_3d * 1117.0)
                    if z0 > 0 and z1 > 0:
                        du, dv = u1 - u0, v1 - v0
                        dn = np.hypot(du, dv)
                        if dn > 1e-3:
                            ux, uy = du / dn, dv / dn
                            max_shift_u = alpha_2d_heading * o.get("w", 0.0)
                            max_shift_v = (alpha_2d_heading * 0.45) * o.get("h", 0.0)
                            shift_u = float(np.sign(du)) * min(abs(float(du)), abs(float(ux * max_shift_u)))
                            shift_v = float(np.sign(dv)) * min(abs(float(dv)), abs(float(uy * max_shift_v)))
                            o["u"] = o["u"] + shift_u
                            o["v"] = o["v"] + shift_v + 0.04 * o.get("h", 0.0)

        # Group observations by camera ID (highest confidence first)
        by_cam: Dict[int, List[Dict[str, float]]] = {}
        for obs in sorted(valid_obs, key=lambda x: x["conf"], reverse=True):
            by_cam.setdefault(obs["cam_id"], []).append(obs)

        pt_3d = None
        err = float("inf")
        cam_ids = list(by_cam.keys())
        pred_pos = tracker.x[:3] if tracker.is_initialized else None
        best_had_occlusion = False

        if len(cam_ids) >= 2:
            best_score = float("inf")
            cos_max_parallax = float(np.cos(np.radians(12.0)))
            for i in range(len(cam_ids)):
                for j in range(i + 1, len(cam_ids)):
                    c1, c2 = cam_ids[i], cam_ids[j]
                    m1 = cameras[c1].get_model(frame_idx)
                    m2 = cameras[c2].get_model(frame_idx)
                    P1 = m1.get_projection_matrix()
                    P2 = m2.get_projection_matrix()
                    for o1 in by_cam[c1][:2]:
                        for o2 in by_cam[c2][:2]:
                            pair_thr = 28.0 + reproj_w_scale * max(o1.get("w", 0.0), o2.get("w", 0.0))
                            cand_pt, cand_err = triangulator.triangulate(
                                [P1, P2], [(o1["u"], o1["v"]), (o2["u"], o2["v"])]
                            )
                            if cand_pt is None or cand_err > pair_thr:
                                continue
                            if not (-500.0 <= cand_pt[2] <= 2500.0):
                                continue
                            # Reject near-collinear camera rays (parallax angle < 12 deg causes depth spikes)
                            v1 = m1.loc - cand_pt
                            v2 = m2.loc - cand_pt
                            v1 /= np.linalg.norm(v1) + 1e-9
                            v2 /= np.linalg.norm(v2) + 1e-9
                            if abs(float(np.dot(v1, v2))) > cos_max_parallax:
                                continue

                            is_pair_occ = o1.get("is_occluded", False) or o2.get("is_occluded", False)
                            inlier_Ps = [P1, P2]
                            inlier_uvs = [(o1["u"], o1["v"]), (o2["u"], o2["v"])]
                            for k, c3 in enumerate(cam_ids):
                                if k in (i, j):
                                    continue
                                m3 = cameras[c3].get_model(frame_idx)
                                u_p, v_p, z_c = m3.project_point(cand_pt)
                                if z_c <= 0:
                                    continue
                                for o3 in by_cam[c3][:2]:
                                    # Do not let a partially occluded 3rd camera corrupt a clean 2-camera pair
                                    if o3.get("is_occluded", False) and not is_pair_occ:
                                        continue
                                    thr3 = 28.0 + reproj_w_scale * o3.get("w", 0.0)
                                    if np.hypot(u_p - o3["u"], v_p - o3["v"]) < thr3:
                                        inlier_Ps.append(m3.get_projection_matrix())
                                        inlier_uvs.append((o3["u"], o3["v"]))
                                        break
                            if len(inlier_Ps) > 2:
                                cand_pt, cand_err = triangulator.triangulate(inlier_Ps, inlier_uvs)
                            if cand_pt is None or cand_err > pair_thr + 7.0:
                                continue
                            score = cand_err - 12.0 * len(inlier_Ps) + (22.0 if is_pair_occ else 0.0)
                            if pred_pos is not None:
                                dist_pred = float(np.linalg.norm(cand_pt - pred_pos))
                                if dist_pred > 2500.0:
                                    score += dist_pred * 0.08
                            if score < best_score:
                                best_score = score
                                pt_3d, err = cand_pt, cand_err
                                best_had_occlusion = is_pair_occ

        # Kalman Filter Step
        if pt_3d is not None and err < 60.0:
            pt_meas = pt_3d.copy()
            pt_meas[2] += cabin_z_shift
            if not tracker.is_initialized:
                tracker.initialize(pt_meas)
                curr_pos: Optional[np.ndarray] = pt_meas
            else:
                tracker.predict()
                noise_scale = max(0.5, err / 5.0) * (2.2 if best_had_occlusion else 1.0)
                curr_pos = tracker.update(pt_meas, noise_scale=noise_scale)
            triangulated_count += 1
        elif tracker.is_initialized and valid_obs:
            # Single-camera ray tracking when occluded on other cameras
            tracker.x[5] *= 0.7
            pred_pt = tracker.predict()
            target_z = float(0.75 * np.clip(pred_pt[2] - cabin_z_shift, 300.0, 520.0) + 0.25 * 365.0)
            best_single_pt = None
            best_dist = float("inf")
            best_single_occ = False
            for o in valid_obs:
                if o["conf"] < 0.70:
                    continue
                m = cameras[o["cam_id"]].get_model(frame_idx)
                cand_single = m.project_ray_to_z(o["u"], o["v"], target_z=target_z)
                if cand_single is not None:
                    cand_single[2] += cabin_z_shift
                    d = float(np.linalg.norm(cand_single[:2] - pred_pt[:2]))
                    if d < 3500.0 and d < best_dist:
                        best_dist = d
                        best_single_pt = cand_single
                        best_single_occ = bool(o.get("is_occluded", False))
            if best_single_pt is not None:
                noise_scale = 3.5 if best_single_occ else 2.0
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
                best_single = max(valid_obs, key=lambda x: x["conf"])
                cam_model = cameras[best_single["cam_id"]].get_model(frame_idx)
                est_single = cam_model.project_ray_to_z(best_single["u"], best_single["v"], target_z=450.0)
                if est_single is not None:
                    est_single[2] += cabin_z_shift
                    tracker.initialize(est_single)
                    curr_pos = est_single
                    single_cam_count += 1
                else:
                    curr_pos = None
                    coasted_count += 1
            else:
                curr_pos = None
                coasted_count += 1

        positions.append(curr_pos.copy() if curr_pos is not None else None)

    # Guarantee valid coordinates on every frame (back-fill leading None frames if any)
    first_valid = next((p for p in positions if p is not None), np.zeros(3))
    filled: List[np.ndarray] = []
    last_valid = first_valid
    for p in positions:
        if p is not None:
            last_valid = p
        filled.append(last_valid.copy())

    return np.array(filled, dtype=np.float64), triangulated_count, single_cam_count, coasted_count


def run_pipeline(
    calib_dir: str = "хакатон",
    detections_csv: Optional[str] = None,
    gt_csv: Optional[str] = None,
    output_csv: str = "results/trajectories/output_trajectory_yolo.csv",
    metrics_json: str = "results/metrics/metrics_summary_yolo.json",
    plot_path: str = "results/plots/trajectory_comparison.png",
    plot: bool = True,
    conf_threshold: float = 0.3,
    dt: float = 1.0 / 30.0,
    realtime_only: bool = False,
) -> pd.DataFrame:
    """
    Executes the full 3D reconstruction and tracking pipeline.
    Works in both Validation Mode (when gt_csv is provided) and
    Blind Test Mode (when gt_csv is None), and supports an arbitrary number of cameras.
    """
    start_time = time.perf_counter()
    calib_dir = _resolve_input_path(calib_dir, "хакатон" if "hackathon" in calib_dir else "data/raw/hackathon")
    if gt_csv:
        gt_csv = _resolve_input_path(
            gt_csv,
            "хакатон/object_transform_SM_AA_track.csv"
            if ("ground_truth" in gt_csv or "hackathon" in gt_csv)
            else "data/raw/hackathon/object_transform_SM_AA_track.csv",
        )

    print("=" * 60)
    print("  PELENG 3D VISION & TRACKING ENGINE (C++ / Python)")
    print("=" * 60)

    # 1. Load Camera Calibrations
    print(f"[1/5] Loading camera calibrations from: '{calib_dir}'...")
    cameras = load_all_cameras(calib_dir)
    panning_cams = _detect_panning_cameras(cameras)
    print(f"      Loaded {len(cameras)} cameras: {sorted(cameras.keys())} (panning: {sorted(panning_cams)})")

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

    sample_widths = [
        o.get("w", 0.0)
        for f_obs in list(grouped_detections.values())[:50]
        for o in f_obs
    ]
    is_yolo_bbox = (not is_mock_mode) and (float(np.mean(sample_widths)) > 100.0 if sample_widths else False)
    if is_yolo_bbox:
        grouped_detections = _unclip_stationary_occluder_edges(grouped_detections, panning_cams)

    # 3. Run Multi-View 3D Triangulation & Cabin Pivot Alignment
    print("[3/5] Running Multi-view 3D Triangulation & Kalman Filtering...")
    t_recon_start = time.perf_counter()

    if is_yolo_bbox and not realtime_only:
        # Pass 1: Reconstruct smooth 3D trajectory of bounding-box centers to extract exact 3D heading
        pass1_xyz, _, _, _ = _reconstruct_pass(
            cameras=cameras,
            panning_cams=panning_cams,
            grouped_detections=grouped_detections,
            max_frame=max_frame,
            conf_threshold=conf_threshold,
            dt=dt,
            alpha_2d_heading=0.0,
            cabin_z_shift=0.0,
            reproj_w_scale=0.06,
        )
        smooth_p1 = pd.DataFrame(pass1_xyz).rolling(window=31, center=True, min_periods=1).mean().to_numpy()
        headings = np.zeros_like(smooth_p1)
        for f_i in range(len(smooth_p1)):
            f0, f1 = max(0, f_i - 35), min(len(smooth_p1) - 1, f_i + 35)
            dv = smooth_p1[f1] - smooth_p1[f0]
            dv[2] *= 0.3
            dn = float(np.linalg.norm(dv))
            headings[f_i] = dv / (dn + 1e-9) if dn > 10.0 else np.array([1.0, 0.0, 0.0])

        # Pass 2: Shift each 2D detection along the projected 3D heading vector from body center to cabin
        raw_xyz, triangulated_count, single_cam_count, coasted_count = _reconstruct_pass(
            cameras=cameras,
            panning_cams=panning_cams,
            grouped_detections=grouped_detections,
            max_frame=max_frame,
            conf_threshold=conf_threshold,
            dt=dt,
            alpha_2d_heading=0.38,
            cabin_z_shift=-60.0,
            reproj_w_scale=0.06,
            ref_headings=headings,
            ref_trajectory=smooth_p1,
        )
        # Residual longitudinal alignment for Kalman filter phase lag (180 units) + 7-frame zero-phase smoothing
        raw_xyz[:, 0] += headings[:, 0] * 180.0
        raw_xyz[:, 1] += headings[:, 1] * 180.0
        raw_xyz = pd.DataFrame(raw_xyz).rolling(window=7, center=True, min_periods=1).mean().to_numpy()
    else:
        # Causal real-time single-pass mode (or mock point-detections mode)
        raw_xyz, triangulated_count, single_cam_count, coasted_count = _reconstruct_pass(
            cameras=cameras,
            panning_cams=panning_cams,
            grouped_detections=grouped_detections,
            max_frame=max_frame,
            conf_threshold=conf_threshold,
            dt=dt,
            alpha_2d_heading=0.20 if is_yolo_bbox else 0.0,
            cabin_z_shift=-60.0 if is_yolo_bbox else 0.0,
            reproj_w_scale=0.06 if is_yolo_bbox else 0.0,
        )

    # Optional Road Spline Constraint (curbs lateral tree-occlusion drift in blind zones)
    road_spline_path = os.path.join("configs", "road_spline_anchors.json")
    if os.path.exists(road_spline_path):
        from src.road_spline import RoadSplineConstraint
        rsc = RoadSplineConstraint().load(road_spline_path)
        if rsc.is_fitted:
            print("      [RoadSpline] Enforcing 3D Road Centerline Constraint (curbing lateral drift)...")
            raw_xyz = rsc.project_trajectory(raw_xyz, blend_clear=0.15, blend_occluded=0.85)

    out_df = pd.DataFrame({
        "Frame": np.arange(max_frame + 1, dtype=int),
        "Loc_X": np.round(raw_xyz[:, 0], 4),
        "Loc_Y": np.round(raw_xyz[:, 1], 4),
        "Loc_Z": np.round(raw_xyz[:, 2], 4),
    })

    recon_elapsed = time.perf_counter() - t_recon_start
    fps = len(out_df) / max(1e-6, recon_elapsed)

    print(f"      Completed {len(out_df)} frames in {recon_elapsed*1000:.1f} ms ({fps:.1f} FPS)")
    print(
        f"      Triangulated (2+ cams): {triangulated_count} frames, "
        f"Single-cam ray tracked: {single_cam_count} frames, "
        f"Kalman coasted: {coasted_count} frames"
    )

    # 4. Save Output Trajectory
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, sep=";", index=False)
    print(f"[4/5] Saved output trajectory to: '{output_csv}'")

    # 5. Evaluation & Plotting
    gt_df = None
    if gt_csv and os.path.exists(gt_csv):
        print(f"[5/5] Evaluating metrics against Ground Truth: '{gt_csv}'...")
        gt_df = load_ground_truth(gt_csv)
        evaluator = TrajectoryEvaluator()

        merged = pd.merge(gt_df, out_df, on="Frame", suffixes=("_gt", "_det"))
        gt_pts = merged[["Loc_X_gt", "Loc_Y_gt", "Loc_Z_gt"]].to_numpy()
        est_pts = merged[["Loc_X_det", "Loc_Y_det", "Loc_Z_det"]].to_numpy()

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
        print(f"      Generating Slide 8 comparison plot -> '{plot_path}'...")
        plot_trajectory_comparison(out_df, gt_df, plot_path)
        print(f"      Saved comparison plot: '{plot_path}'")

    total_time = time.perf_counter() - start_time
    print(f"\nPipeline finished successfully in {total_time:.2f}s.\n")
    return out_df


def main():
    default_calib = _resolve_input_path("хакатон", "data/raw/hackathon")
    default_gt = _resolve_input_path(
        "хакатон/object_transform_SM_AA_track.csv",
        _resolve_input_path("data/ground_truth/object_transform_SM_AA_track.csv", "data/raw/hackathon/object_transform_SM_AA_track.csv"),
    )

    parser = argparse.ArgumentParser(description="Peleng 3D Real-Time Tracking Core")
    parser.add_argument("--calib-dir", default=default_calib, help="Directory with camera calibration CSV files")
    parser.add_argument(
        "--detections",
        default="results/detections/detections_2d_yolo_real.csv",
        help="Path to 2D detections CSV",
    )
    parser.add_argument("--gt", default=default_gt, help="Path to Ground Truth CSV (optional)")
    parser.add_argument(
        "--output",
        default="results/trajectories/output_trajectory_yolo.csv",
        help="Output trajectory CSV file",
    )
    parser.add_argument(
        "--metrics-out",
        default="results/metrics/metrics_summary_yolo.json",
        help="Output metrics JSON file",
    )
    parser.add_argument(
        "--plot-out",
        default="results/plots/trajectory_comparison.png",
        help="Output plot PNG file",
    )
    parser.add_argument("--plot", action="store_true", default=True, help="Generate Slide 8 comparison plot")
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="Disable plotting")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Confidence threshold for 2D detections")
    parser.add_argument("--realtime", action="store_true", default=False, help="Run strictly causal single-pass mode")

    args = parser.parse_args()

    gt_file = args.gt if (args.gt and args.gt.lower() != "none") else None

    run_pipeline(
        calib_dir=args.calib_dir,
        detections_csv=args.detections,
        gt_csv=gt_file,
        output_csv=args.output,
        metrics_json=args.metrics_out,
        plot_path=args.plot_out,
        plot=args.plot,
        conf_threshold=args.conf_threshold,
        realtime_only=args.realtime,
    )


if __name__ == "__main__":
    main()
