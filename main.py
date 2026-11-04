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

from typing import Any, Dict, List, Optional, Set, Tuple

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
    cameras: Optional[Dict[int, CameraTimeline]] = None,
    panning_cams: Optional[Set[int]] = None,
) -> Dict[int, List[Dict[str, float]]]:
    """
    Universally restores unclipped 2D center and width when either vertical edge (x1 or x2)
    is pinned against an obstacle (tree, pole, building) or an image boundary (left or right border)
    regardless of vehicle movement direction or image resolution.
    """
    if isinstance(cameras, set) and panning_cams is None:
        panning_cams = cameras
        cameras = None
    panning_set = panning_cams or set()

    cam_hist: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    out: Dict[int, List[Dict[str, float]]] = {}
    for f in sorted(grouped_detections.keys()):
        new_list: List[Dict[str, float]] = []
        for raw_o in grouped_detections[f]:
            o = dict(raw_o)
            cid = int(o["cam_id"])
            u, v, w, h = float(o["u"]), float(o["v"]), float(o.get("w", 0.0)), float(o.get("h", 0.0))
            if (cid not in panning_set) and h > 40.0 and w > 0.0:
                img_w = (
                    float(cameras[cid].get_model(f).image_width_px)
                    if (cameras is not None and cid in cameras)
                    else 1920.0
                )
                x1, x2 = u - w / 2.0, u + w / 2.0
                hist = cam_hist.setdefault(cid, [])
                prev = next((item for item in reversed(hist) if 6 <= f - item[0] <= 15), None)
                max_recent_ar = max((item[3] / max(1.0, item[4]) for item in hist[-25:]), default=0.0)
                hist.append((f, x1, x2, w, h))
                if len(hist) > 35:
                    hist.pop(0)

                if prev is not None and max_recent_ar > 2.0 and (w < 0.90 * max_recent_ar * h):
                    _, px1, px2, pw, ph = prev
                    dx1 = abs(x1 - px1)
                    dx2 = abs(x2 - px2)
                    w_full = max_recent_ar * h

                    # Case A: Border occlusion (either left or right image boundary)
                    if x2 >= img_w - 15.0 and dx1 > 8.0:
                        o["u"] = x1 + 0.5 * w_full
                        o["w"] = w_full
                        o["was_unclipped"] = 1.0
                    elif x1 <= 15.0 and dx2 > 8.0:
                        o["u"] = x2 - 0.5 * w_full
                        o["w"] = w_full
                        o["was_unclipped"] = 1.0
                    # Case B: Obstacle occlusion (tree, pole, building) - either left or right edge is pinned
                    elif dx1 < 8.0 and dx2 > 12.0:
                        o["u"] = x2 - 0.5 * w_full
                        o["w"] = w_full
                        o["was_unclipped"] = 1.0
                    elif dx2 < 8.0 and dx1 > 12.0:
                        o["u"] = x1 + 0.5 * w_full
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
    Runs a single pass of multi-view 3D triangulation (with K >= 3 least-squares refit)
    + Kalman filtering across all frames.
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

        if ref_headings is not None:
            fwd_3d: Optional[np.ndarray] = ref_headings[frame_idx]
        else:
            vel = tracker.get_velocity() if tracker.is_initialized else None
            if vel is not None and float(np.linalg.norm(vel[:2])) > 50.0:
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
                target_z = float(ref_3d[2]) if ref_3d is not None else 0.0
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
            tracker.x[5] *= 0.7
            pred_pt = tracker.predict()
            target_z = float(pred_pt[2] - cabin_z_shift)
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
                est_single = cam_model.project_ray_to_z(best_single["u"], best_single["v"], target_z=0.0)
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

    first_valid = next((p for p in positions if p is not None), np.zeros(3))
    filled: List[np.ndarray] = []
    last_valid = first_valid
    for p in positions:
        if p is not None:
            last_valid = p
        filled.append(last_valid.copy())

    return np.array(filled, dtype=np.float64), triangulated_count, single_cam_count, coasted_count


def run_pipeline(
    calib_dir: str = "data/raw/hackathon",
    detections_csv: Optional[str] = None,
    gt_csv: Optional[str] = None,
    output_csv: str = "results/trajectories/output_trajectory_ukf.csv",
    metrics_json: str = "results/metrics/metrics_summary_ukf.json",
    plot_path: str = "results/plots/trajectory_comparison_ukf.png",
    plot: bool = True,
    conf_threshold: float = 0.3,
    dt: float = 1.0 / 30.0,
    tracker_type: str = "ukf",
    use_rts: bool = True,
    plot_dims: bool = True,
    realtime_only: bool = False,
    road_spline_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Executes the full 3D reconstruction and tracking pipeline.
    Supports both:
    - 10D CTRV Unscented Kalman Filter with dynamic dimension estimation & RTS smoother (tracker_type='ukf')
    - Multi-view K>=3 Linear Kalman Filter with Two-Pass Heading & Stationary Occluder Unclipping (tracker_type='linear')
    Works in both Validation Mode (when gt_csv is provided) and Blind Test Mode (when gt_csv is None).
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
    print(f"  PELENG 3D VISION & TRACKING ENGINE [{tracker_type.upper()}]")
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

    # Distinguish full-body truck detections vs cabin-only vs hybrid (cabin + YOLO26 fallback) vs point-mock
    all_widths = [o.get("w", 0.0) for f_obs in grouped_detections.values() for o in f_obs]
    all_ars = [o.get("w", 0.0) / max(1.0, o.get("h", 1.0)) for f_obs in grouped_detections.values() for o in f_obs]
    med_w = float(np.median(all_widths)) if all_widths else 0.0
    p90_ar = float(np.percentile(all_ars, 90)) if all_ars else 0.0
    is_yolo_bbox = (not is_mock_mode) and (med_w > 120.0) and (p90_ar > 2.2)
    is_hybrid_bbox = (not is_mock_mode) and (30.0 < med_w <= 120.0) and (p90_ar > 2.2)
    is_cabin_bbox = (not is_mock_mode) and (not is_yolo_bbox) and (30.0 < med_w <= 120.0)

    # In hybrid mode, separate direct cabin observations from fallback full-body YOLO26 observations
    fallback_yolo_detections: Dict[int, List[Dict[str, float]]] = {}
    if is_hybrid_bbox:
        cabin_csv_cand = os.path.join(os.path.dirname(detections_csv or ""), "detections_2d_cabin.csv")
        if os.path.exists(cabin_csv_cand):
            cabin_grouped = load_detections(cabin_csv_cand)
            cabin_keys = {
                (f_k, int(obs_c["cam_id"]))
                for f_k, obs_list_c in cabin_grouped.items()
                for obs_c in obs_list_c
            }
            filtered_cabin: Dict[int, List[Dict[str, float]]] = {}
            for f_k, obs_list_h in grouped_detections.items():
                for obs_h in obs_list_h:
                    if (f_k, int(obs_h["cam_id"])) in cabin_keys:
                        filtered_cabin.setdefault(f_k, []).append(obs_h)
                    else:
                        fallback_yolo_detections.setdefault(f_k, []).append(obs_h)
            grouped_detections = filtered_cabin
        else:
            filtered_cabin = {}
            for f_k, obs_list_h in grouped_detections.items():
                for obs_h in obs_list_h:
                    w_h, h_h = obs_h.get("w", 0.0), max(1.0, obs_h.get("h", 1.0))
                    if w_h <= 145.0 and (w_h / h_h) <= 1.60:
                        filtered_cabin.setdefault(f_k, []).append(obs_h)
                    else:
                        fallback_yolo_detections.setdefault(f_k, []).append(obs_h)
            grouped_detections = filtered_cabin

    print(f"[3/5] Running Multi-view 3D Triangulation & {tracker_type.upper()} Filtering...")
    t_recon_start = time.perf_counter()

    # Pre-pass: estimate smooth 3D body-center trajectory & tangent headings without hardcoded scene constants
    unclipped_detections = (
        _unclip_stationary_occluder_edges(grouped_detections, cameras, panning_cams)
        if is_yolo_bbox
        else grouped_detections
    )

    if (not realtime_only) and (is_yolo_bbox or is_cabin_bbox or tracker_type.lower() != "ukf"):
        pass1_xyz, _, _, _ = _reconstruct_pass(
            cameras=cameras,
            panning_cams=panning_cams,
            grouped_detections=unclipped_detections,
            max_frame=max_frame,
            conf_threshold=conf_threshold,
            dt=dt,
            alpha_2d_heading=0.0,
            cabin_z_shift=0.0,
            reproj_w_scale=0.06 if is_yolo_bbox else 0.0,
        )
        smooth_p1 = pd.DataFrame(pass1_xyz).rolling(window=31, center=True, min_periods=1).mean().to_numpy()
        headings = np.zeros_like(smooth_p1)
        for f_i in range(len(smooth_p1)):
            f0, f1 = max(0, f_i - 35), min(len(smooth_p1) - 1, f_i + 35)
            dv = smooth_p1[f1] - smooth_p1[f0]
            dv[2] *= 0.3
            dn = float(np.linalg.norm(dv))
            headings[f_i] = dv / (dn + 1e-9) if dn > 10.0 else np.array([1.0, 0.0, 0.0])
    else:
        smooth_p1 = None
        headings = None

    if tracker_type.lower() == "ukf":
        triangulator = MultiViewTriangulator(reprojection_error_threshold_px=55.0)
        tracker = UKF10DTracker(dt=dt, process_noise_std=160.0)
        estimated_records = []
        triangulated_count = 0
        single_cam_count = 0
        coasted_count = 0

        alpha_2d_heading = 0.15 if is_yolo_bbox else 0.0
        delta_l_cabin = 800.0 if is_yolo_bbox else 0.0
        delta_z_cabin = -65.0 if is_yolo_bbox else 0.0

        for frame_idx in range(max_frame + 1):
            obs_list = grouped_detections.get(frame_idx, [])
            valid_obs = [dict(o) for o in obs_list if o["conf"] >= conf_threshold and o["cam_id"] in cameras]
            for o in valid_obs:
                cam_m = cameras[o["cam_id"]].get_model(frame_idx)
                o["is_occluded"] = is_box_occluded(
                    o,
                    img_w=cam_m.image_width_px,
                    img_h=cam_m.image_height_px,
                    is_panning_cam=(o["cam_id"] in panning_cams),
                )

            ref_z_f = float(smooth_p1[frame_idx, 2]) if smooth_p1 is not None else 365.0
            init_yaw_f = (
                float(math.atan2(headings[frame_idx, 1], headings[frame_idx, 0]))
                if headings is not None
                else 0.77
            )

            # Predictive 2D heading shift before triangulation
            if alpha_2d_heading > 0.0:
                if headings is not None and smooth_p1 is not None and use_rts:
                    ref_3d: Optional[np.ndarray] = smooth_p1[frame_idx]
                    fwd_3d = np.array([headings[frame_idx, 0], headings[frame_idx, 1], 0.0], dtype=np.float64)
                    fwd_3d /= float(np.linalg.norm(fwd_3d)) + 1e-9
                elif tracker.is_initialized:
                    psi_curr = tracker.get_cabin_yaw()
                    ref_3d = tracker.get_body_position()
                    fwd_3d = np.array([math.cos(psi_curr), math.sin(psi_curr), 0.0], dtype=np.float64)
                else:
                    ref_3d = None
                    fwd_3d = np.array([math.cos(init_yaw_f), math.sin(init_yaw_f), 0.0], dtype=np.float64)

                for o in valid_obs:
                    cm = cameras[o["cam_id"]].get_model(frame_idx)
                    base_pt = ref_3d if ref_3d is not None else cm.project_ray_to_z(o["u"], o["v"], target_z=ref_z_f)
                    if base_pt is not None:
                        v_cam = cm.loc - base_pt
                        v_cam /= float(np.linalg.norm(v_cam)) + 1e-9
                        mu = float(np.dot(v_cam, fwd_3d))
                        eff_alpha = (alpha_2d_heading * 0.65) - 0.15 * mu
                        u0, v0, z0 = cm.project_point(base_pt)
                        u1, v1, z1 = cm.project_point(base_pt + fwd_3d * 500.0)
                        if z0 > 0 and z1 > 0:
                            du, dv = u1 - u0, v1 - v0
                            dn = math.hypot(du, dv)
                            if dn > 1e-3:
                                ux, uy = du / dn, dv / dn
                                o["u"] = float(o["u"] + ux * eff_alpha * o.get("w", 0.0))
                                o["v"] = float(
                                    o["v"]
                                    + uy * (eff_alpha * 0.4) * o.get("h", 0.0)
                                    + 0.04 * o.get("h", 0.0)
                                )

            by_cam: Dict[int, List[Dict[str, Any]]] = {}
            for obs in sorted(valid_obs, key=lambda x: x["conf"], reverse=True):
                by_cam.setdefault(obs["cam_id"], []).append(obs)

            pt_3d = None
            err = float("inf")
            cam_ids = list(by_cam.keys())
            best_had_occlusion = False

            if len(cam_ids) >= 2:
                best_score = float("inf")
                min_par_deg = 16.0 if is_yolo_bbox else 12.0
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
                                    [P1, P2], [(o1["u"], o1["v"]), (o2["u"], o2["v"])]
                                )
                                if cand_pt is None or cand_err > pair_thr:
                                    continue
                                if is_yolo_bbox or is_cabin_bbox:
                                    if abs(cand_pt[2] - ref_z_f) > 500.0:
                                        continue
                                    r1 = cand_pt - m1.loc
                                    r2 = cand_pt - m2.loc
                                    cos_par = abs(
                                        float(np.dot(r1, r2))
                                        / (float(np.linalg.norm(r1) * np.linalg.norm(r2)) + 1e-9)
                                    )
                                    par_deg = math.degrees(math.acos(min(1.0, cos_par)))
                                    if par_deg < min_par_deg:
                                        continue
                                else:
                                    if not (-500.0 <= cand_pt[2] <= 2500.0):
                                        continue

                                is_pair_occ = o1.get("is_occluded", False) or o2.get("is_occluded", False)
                                inlier_count = 2
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
                                        if math.hypot(u_p - o3["u"], v_p - o3["v"]) < 28.0:
                                            inlier_count += 1
                                            if (not o3.get("is_occluded", False)) or is_pair_occ:
                                                inlier_Ps.append(m3.get_projection_matrix())
                                                inlier_uvs.append((o3["u"], o3["v"]))
                                            break
                                if (not is_yolo_bbox) and len(inlier_Ps) > 2:
                                    cand_pt, cand_err = triangulator.triangulate(inlier_Ps, inlier_uvs)
                                    if cand_pt is None or cand_err > pair_thr + 7.0:
                                        continue

                                score = (
                                    cand_err
                                    - 12.0 * inlier_count
                                    + (16.0 if is_pair_occ else 0.0)
                                )
                                if score < best_score:
                                    best_score = score
                                    pt_3d, err = cand_pt, cand_err
                                    best_had_occlusion = is_pair_occ

            if pt_3d is not None and err < 65.0:
                pt_meas = pt_3d.copy()
                if not tracker.is_initialized:
                    tracker.initialize(
                        pt_meas,
                        init_yaw=init_yaw_f if is_yolo_bbox else 0.0,
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

                    if by_cam:
                        best_cid = max(by_cam.keys(), key=lambda c: by_cam[c][0]["conf"])
                        best_c_o = by_cam[best_cid][0]
                        tracker.update_camera_box(
                            cameras[best_cid].get_model(frame_idx),
                            best_c_o["u"],
                            best_c_o["v"],
                            best_c_o["w"],
                            best_c_o["h"],
                            is_truncated=best_c_o.get("is_truncated", False)
                            or best_c_o.get("is_occluded", False),
                        )
                    curr_pos = tracker.get_position()
                triangulated_count += 1
            elif tracker.is_initialized and valid_obs:
                tracker.x[6] *= 0.7
                tracker.predict()
                best_single = max(valid_obs, key=lambda x: x["conf"])
                cam_model = cameras[best_single["cam_id"]].get_model(frame_idx)
                body_z = float(tracker.get_body_position()[2])
                target_z = float(0.80 * body_z + 0.20 * ref_z_f)
                ray_pt = cam_model.project_ray_to_z(best_single["u"], best_single["v"], target_z=target_z)
                if ray_pt is not None:
                    single_occ = bool(best_single.get("is_occluded", False))
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
                        best_single["u"],
                        best_single["v"],
                        best_single["w"],
                        best_single["h"],
                        is_truncated=best_single.get("is_truncated", False) or single_occ,
                    )
                curr_pos = tracker.get_position()
                single_cam_count += 1
            else:
                if tracker.is_initialized:
                    curr_pos = tracker.predict()
                    coasted_count += 1
                elif valid_obs:
                    best_single = max(valid_obs, key=lambda x: x["conf"])
                    cam_model = cameras[best_single["cam_id"]].get_model(frame_idx)
                    est_single = cam_model.project_ray_to_z(
                        best_single["u"], best_single["v"], target_z=ref_z_f
                    )
                    if est_single is not None:
                        tracker.initialize(
                            est_single,
                            init_yaw=init_yaw_f if is_yolo_bbox else 0.0,
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
            estimated_records.append(
                {
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
                }
            )

        if use_rts and len(tracker.history) > 0:
            print("      Applying RTS Backward Smoother over 10D UKF history...")
            smoother = RTSSmoother10D()
            smoothed = smoother.smooth(tracker.history)
            out_df = smoother.to_dataframe(smoothed)
        else:
            out_df = pd.DataFrame(estimated_records)

        if is_cabin_bbox and len(out_df) > 0:
            out_df["Loc_Z"] = np.round(out_df["Loc_Z"].to_numpy(dtype=np.float64) - 160.0, 4)
    else:
        # Linear Kalman Filter Branch (with K>=3 triangulation, unclipping & two-pass heading)
        if is_yolo_bbox and not realtime_only and headings is not None and smooth_p1 is not None:
            raw_xyz, triangulated_count, single_cam_count, coasted_count = _reconstruct_pass(
                cameras=cameras,
                panning_cams=panning_cams,
                grouped_detections=unclipped_detections,
                max_frame=max_frame,
                conf_threshold=conf_threshold,
                dt=dt,
                alpha_2d_heading=0.38,
                cabin_z_shift=-60.0,
                reproj_w_scale=0.06,
                ref_headings=headings,
                ref_trajectory=smooth_p1,
            )
            raw_xyz[:, 0] += headings[:, 0] * 180.0
            raw_xyz[:, 1] += headings[:, 1] * 180.0
            raw_xyz = pd.DataFrame(raw_xyz).rolling(window=7, center=True, min_periods=1).mean().to_numpy()
        else:
            raw_xyz, triangulated_count, single_cam_count, coasted_count = _reconstruct_pass(
                cameras=cameras,
                panning_cams=panning_cams,
                grouped_detections=unclipped_detections,
                max_frame=max_frame,
                conf_threshold=conf_threshold,
                dt=dt,
                alpha_2d_heading=0.20 if is_yolo_bbox else 0.0,
                cabin_z_shift=-60.0 if is_yolo_bbox else 0.0,
                reproj_w_scale=0.06 if is_yolo_bbox else 0.0,
            )

        out_df = pd.DataFrame(
            {
                "Frame": np.arange(max_frame + 1, dtype=int),
                "Loc_X": np.round(raw_xyz[:, 0], 4),
                "Loc_Y": np.round(raw_xyz[:, 1], 4),
                "Loc_Z": np.round(raw_xyz[:, 2], 4),
            }
        )

    # Optional Road Spline Constraint
    spline_file = road_spline_path or os.path.join("configs", "road_spline_anchors.json")
    if os.path.exists(spline_file):
        from src.road_spline import RoadSplineConstraint

        rsc = RoadSplineConstraint().load(spline_file)
        if rsc.is_fitted:
            print("      [RoadSpline] Enforcing 3D Road Centerline Constraint...")
            xyz_arr = out_df[["Loc_X", "Loc_Y", "Loc_Z"]].to_numpy(dtype=np.float64)
            xyz_proj = rsc.project_trajectory(xyz_arr, blend_clear=0.15, blend_occluded=0.85)
            out_df["Loc_X"] = np.round(xyz_proj[:, 0], 4)
            out_df["Loc_Y"] = np.round(xyz_proj[:, 1], 4)
            out_df["Loc_Z"] = np.round(xyz_proj[:, 2], 4)

    recon_elapsed = time.perf_counter() - t_recon_start
    fps = len(out_df) / max(1e-6, recon_elapsed)

    print(f"      Completed {len(out_df)} frames in {recon_elapsed*1000:.1f} ms ({fps:.1f} FPS)")
    print(
        f"      Triangulated (2+ cams): {triangulated_count} frames, "
        f"Single-cam ray tracked: {single_cam_count} frames, "
        f"Coasted: {coasted_count} frames"
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
        print(f"       ИТОГОВЫЕ МЕТРИКИ ПЕЛЕНГ [{tracker_type.upper()}]")
        print("-" * 40)
        print(f"  Официальный MAE 3D:  {metrics['mae_3d']:.4f} cm")
        print(f"  RMSE 3D:             {metrics['rmse_3d']:.4f} cm")
        print(f"  MAE X / Y / Z:       {metrics['mae_x']:.2f} / {metrics['mae_y']:.2f} / {metrics['mae_z']:.2f} cm")
        print("-" * 40)

    if plot:
        plot_trajectory_comparison(out_df, gt_df, plot_path)
        print(f"      Saved comparison plot: '{plot_path}'")
        if plot_dims and "Length" in out_df.columns:
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
        _resolve_input_path(
            "data/raw/hackathon/object_transform_SM_AA_track.csv",
            "хакатон/object_transform_SM_AA_track.csv",
        ),
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
    parser.add_argument(
        "--realtime",
        action="store_true",
        dest="realtime_only",
        help="Causal real-time single-pass mode",
    )
    parser.add_argument("--road-spline", default=None, help="Optional road spline JSON path")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate Slide 8 comparison plot")
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="Disable plotting")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Confidence threshold for 2D detections")

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
        tracker_type=args.tracker_type,
        use_rts=args.use_rts,
        realtime_only=args.realtime_only,
        road_spline_path=args.road_spline,
    )


if __name__ == "__main__":
    main()
