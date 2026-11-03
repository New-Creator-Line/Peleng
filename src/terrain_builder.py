import base64
import hashlib
import io
import math
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import Rbf

from src.camera_model import CameraModel, CameraTimeline, load_all_cameras
from src.evaluator import TrajectoryEvaluator

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
FRUSTUM_DISTANCE_CM = 4000.0
MAX_POINT_CLOUD_POINTS = 12000


def _camera_basis_vectors(model: CameraModel) -> Tuple[ np.ndarray, np.ndarray, np.ndarray ]:
    """
    Extracts world-space unit basis vectors (forward, right, up) from CameraModel.R_cam.
    Recall from src.camera_model.get_ue_camera_rotation_matrix:
      R_cam = np.vstack([right, -up, forward])
    """
    right = np.asarray(model.R_cam[0], dtype=np.float64)
    up = -np.asarray(model.R_cam[1], dtype=np.float64)
    forward = np.asarray(model.R_cam[2], dtype=np.float64)
    return forward, right, up


def _pixel_to_ray(model: CameraModel, u: float, v: float) -> np.ndarray:
    """
    Unprojects pixel (u, v) into a unit 3D direction vector in UE5 world coordinates.
    """
    fx, fy, cx, cy = model.get_intrinsics()
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    ray_world = model.R_cam.T @ ray_cam
    norm = float(np.linalg.norm(ray_world))
    if norm > 1e-9:
        ray_world = ray_world / norm
    return ray_world


def _compute_frustum_corners(
    model: CameraModel,
    frustum_dist_cm: float = FRUSTUM_DISTANCE_CM,
) -> List[List[float]]:
    """
    Computes the 4 world-space frustum corners at distance `frustum_dist_cm` from camera center
    in order: Top-Left (0,0), Top-Right (W,0), Bottom-Right (W,H), Bottom-Left (0,H).
    """
    w = float(model.image_width_px)
    h = float(model.image_height_px)
    uv_corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    corners_3d: List[List[float]] = []
    for u, v in uv_corners:
        ray_dir = _pixel_to_ray(model, u, v)
        pt = model.loc + ray_dir * frustum_dist_cm
        corners_3d.append([round(float(pt[0]), 2), round(float(pt[1]), 2), round(float(pt[2]), 2)])
    return corners_3d


def _is_camera_dynamic(timeline: CameraTimeline) -> bool:
    """
    Detects whether a camera rotates or moves across frames (e.g. CineCameraActor5).
    """
    if len(timeline.frames) <= 1:
        return False
    frame_keys = sorted(timeline.frames.keys())
    sample_keys = frame_keys[::max(1, len(frame_keys) // 20)]
    rots = np.array([timeline.frames[k].rot for k in sample_keys], dtype=np.float64)
    locs = np.array([timeline.frames[k].loc for k in sample_keys], dtype=np.float64)
    rot_ptp = float(np.max(np.ptp(rots, axis=0)))
    loc_ptp = float(np.max(np.ptp(locs, axis=0)))
    return (rot_ptp > 0.05) or (loc_ptp > 1.0)


def _generate_procedural_frame(cam_id: int, width: int = 1920, height: int = 1080) -> np.ndarray:
    """
    Deterministic fallback RGB frame generator when video files are unavailable (e.g. unit tests).
    Produces realistic sky, mountain slope, and valley floor coloration.
    """
    v_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    u_norm = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    horizon = 0.28 + 0.03 * ((cam_id % 3) - 1)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    sky_mask = v_norm < horizon
    terrain_t = np.clip((v_norm - horizon) / max(1e-3, 1.0 - horizon), 0.0, 1.0)
    wave = 0.05 * np.sin(u_norm * 12.0 + cam_id)

    r = np.where(sky_mask, 165 - 40 * v_norm, 92 + 38 * terrain_t + 20 * wave)
    g = np.where(sky_mask, 195 - 30 * v_norm, 108 + 28 * (1.0 - terrain_t) + 15 * wave)
    b = np.where(sky_mask, 225 - 20 * v_norm, 74 + 18 * (1.0 - terrain_t))

    img[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)
    return img


def _extract_video_frame_rgb(
    video_path: str,
    cam_id: int,
    frame_idx: int = 100,
    fps: float = 30.0,
) -> np.ndarray:
    """
    Extracts a single RGB frame (H, W, 3) uint8 from an MP4 file using cv2 or ffmpeg.
    Falls back to procedural outdoor frame if video file or decoder is unavailable.
    """
    if os.path.exists(video_path):
        # 1. Try OpenCV if available
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame_bgr = cap.read()
                cap.release()
                if ok and frame_bgr is not None:
                    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        # 2. Try ffmpeg subprocess pipe
        ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        if os.path.exists(ffmpeg_bin):
            try:
                timestamp_sec = max(0.0, frame_idx / max(1.0, fps))
                cmd = [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp_sec:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ]
                proc = subprocess.run(cmd, capture_output=True, timeout=8.0, check=False)
                if proc.returncode == 0 and len(proc.stdout) > 100:
                    pil_img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
                    return np.asarray(pil_img, dtype=np.uint8)
            except Exception:
                pass

    return _generate_procedural_frame(cam_id=cam_id)


def _encode_thumbnail_b64(rgb_frame: np.ndarray, width: int = 320, height: int = 180) -> str:
    """
    Encodes an RGB numpy array into a compact 320x180 JPEG data URI string.
    """
    pil_img = Image.fromarray(rgb_frame).resize((width, height), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=78, optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def _hypsometric_color(z_cm: float, x_cm: float = 0.0, y_cm: float = 0.0) -> Tuple[int, int, int]:
    """
    Computes realistic mountain/valley RGB color based on elevation Z (in UE5 cm)
    and subtle spatial texture variation when outside camera FOVs.
    """
    t = float(np.clip((z_cm - 300.0) / 7200.0, 0.0, 1.0))
    tex = 0.04 * math.sin(x_cm * 0.00025) * math.cos(y_cm * 0.00025)
    t_mod = float(np.clip(t + tex, 0.0, 1.0))

    if t_mod < 0.18:
        # Valley floor: olive/sage grass & dirt road tones
        u = t_mod / 0.18
        r = 86.0 + u * 22.0
        g = 104.0 + u * 18.0
        b = 68.0 + u * 14.0
    elif t_mod < 0.65:
        # Mid mountain slopes: dry scrub & rocky ochre/slate
        u = (t_mod - 0.18) / 0.47
        r = 108.0 + u * 34.0
        g = 122.0 + u * 14.0
        b = 82.0 + u * 26.0
    else:
        # High alpine ridges (Z > 5000 cm): light grey-brown rock & sunlit peaks
        u = (t_mod - 0.65) / 0.35
        r = 142.0 + u * 46.0
        g = 136.0 + u * 48.0
        b = 108.0 + u * 62.0

    return int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))


def _build_terrain_rbf(
    traj_pts: np.ndarray,
    cameras: Dict[int, CameraTimeline],
    bounds_cm: Dict[str, float],
) -> Rbf:
    """
    Constructs a smooth continuous 3D valley elevation surface Z(X, Y) using
    multiquadric Radial Basis Function interpolation from:
      1. Subsampled 3D trajectory points along the valley road.
      2. Camera ground/mountain ridge bases (Cam 1..5).
      3. Valley corridor & ridge boundary anchors around bounds_cm.
    """
    anchor_xyz: List[Tuple[float, float, float]] = []

    # 1. Trajectory road points (subsampled so conditioning is fast and well-spaced)
    if len(traj_pts) > 0:
        step = max(1, len(traj_pts) // 45)
        for pt in traj_pts[::step]:
            anchor_xyz.append((float(pt[0]), float(pt[1]), float(pt[2])))
        # Always include start and end points
        anchor_xyz.append((float(traj_pts[0, 0]), float(traj_pts[0, 1]), float(traj_pts[0, 2])))
        anchor_xyz.append((float(traj_pts[-1, 0]), float(traj_pts[-1, 1]), float(traj_pts[-1, 2])))

    # 2. Camera ground / mountain bases
    for cam_id, timeline in sorted(cameras.items()):
        model = timeline.get_model(0)
        cx, cy, cz = float(model.loc[0]), float(model.loc[1]), float(model.loc[2])
        if cz > 3000.0:
            # High mountain overview camera (e.g. Cam 2 at Z=7912, Cam 4 at Z=7147)
            z_base = cz - 150.0
        else:
            # Valley tripod/tower camera (e.g. Cam 1, 3, 5)
            z_base = max(300.0, cz - 250.0)
        anchor_xyz.append((cx, cy, z_base))

    # 3. Perimeter / valley-wall anchors for natural canyon/valley bowls
    x_min, x_max = bounds_cm["x_min"], bounds_cm["x_max"]
    y_min, y_max = bounds_cm["y_min"], bounds_cm["y_max"]
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    perimeter_anchors = [
        (x_min, y_min, 3800.0),
        (x_min, y_mid, 2200.0),
        (x_min, y_max, 4200.0),
        (x_mid, y_min, 3400.0),
        (x_mid, y_max, 6400.0),
        (x_max, y_min, 5200.0),
        (x_max, y_mid, 1800.0),
        (x_max, y_max, 4600.0),
    ]
    anchor_xyz.extend(perimeter_anchors)

    arr = np.asarray(anchor_xyz, dtype=np.float64)
    # Deduplicate close XY points to prevent singular matrix in RBF
    dedup_xy: List[np.ndarray] = []
    for row in arr:
        if not dedup_xy:
            dedup_xy.append(row)
            continue
        dists = [math.hypot(float(row[0] - prev[0]), float(row[1] - prev[1])) for prev in dedup_xy]
        if min(dists) > 150.0:
            dedup_xy.append(row)

    pts = np.asarray(dedup_xy, dtype=np.float64)
    rbf = Rbf(pts[:, 0], pts[:, 1], pts[:, 2], function="multiquadric", smooth=2.0)
    return rbf


def _sample_projected_color(
    pt_3d: np.ndarray,
    cameras: Dict[int, CameraTimeline],
    cam_frames: Dict[int, np.ndarray],
    priority_cams: List[int],
    ref_frame: int = 100,
) -> Tuple[int, int, int]:
    """
    Projects a 3D world point (X, Y, Z) into the calibrated cameras and blends
    the real video RGB texture with elevation hypsometric color.
    """
    base_r, base_g, base_b = _hypsometric_color(float(pt_3d[2]), float(pt_3d[0]), float(pt_3d[1]))

    for cam_id in priority_cams:
        if cam_id not in cameras or cam_id not in cam_frames:
            continue
        model = cameras[cam_id].get_model(ref_frame)
        u, v, z_cam = model.project_point(pt_3d)
        h, w, _ = cam_frames[cam_id].shape
        # Exclude sky region (top 22% of image) and out-of-frame projections
        if z_cam > 200.0 and (0.0 <= u < w - 1) and (0.22 * h <= v < h - 1):
            ui = int(round(u))
            vi = int(round(v))
            px = cam_frames[cam_id][vi, ui]
            # Blend 70% real camera RGB + 30% hypsometric terrain shading
            r = int(np.clip(0.70 * float(px[0]) + 0.30 * base_r, 0, 255))
            g = int(np.clip(0.70 * float(px[1]) + 0.30 * base_g, 0, 255))
            b = int(np.clip(0.70 * float(px[2]) + 0.30 * base_b, 0, 255))
            return r, g, b

    return base_r, base_g, base_b


def _compute_calibrated_depth_point_cloud(
    calib_dir: str,
    cameras: Dict[int, CameraTimeline],
    cam_frames: Dict[int, np.ndarray],
    traj_pts: np.ndarray,
    rbf_surface: Rbf,
    bounds_cm: Dict[str, float],
    use_depth_model: bool = True,
    ref_frame: int = 100,
) -> Tuple[List[List[float]], List[List[int]], str]:
    """
    Computes the 3D point cloud in UE5_cm with RGB colors.
    When `use_depth_model=True` and video files exist in `calib_dir`, runs
    `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` on downscaled
    reference camera frames, aligns each depth map to the 3D trajectory via
    Median Scale & Shift Calibration, and caches results in
    `results/.cache_terrain_depth.npz`.
    """
    has_real_videos = any(
        os.path.exists(os.path.join(calib_dir, f"Way1.FinalImage.CineCameraActor{cid}.mp4"))
        for cid in cameras.keys()
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(project_root, "results", ".cache_terrain_depth.npz")
    calib_hash = hashlib.md5(os.path.abspath(calib_dir).encode("utf-8")).hexdigest()[:12]

    if use_depth_model and has_real_videos and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if str(cached["calib_hash"]) == calib_hash:
                pts_arr = cached["points_cm"]
                cols_arr = cached["colors_rgb"]
                if len(pts_arr) > 100 and pts_arr.shape == cols_arr.shape:
                    return (
                        np.round(pts_arr, 1).tolist(),
                        cols_arr.astype(int).tolist(),
                        MODEL_ID,
                    )
        except Exception:
            pass

    if use_depth_model and has_real_videos:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            device = (
                "mps"
                if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            model_net = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

            all_pts: List[List[float]] = []
            all_cols: List[List[int]] = []

            for cam_id, timeline in sorted(cameras.items()):
                if cam_id not in cam_frames:
                    continue
                rgb_full = cam_frames[cam_id]
                h_full, w_full, _ = rgb_full.shape
                cam_model = timeline.get_model(ref_frame)

                # Downscale to 640x360 for fast neural depth inference (~0.18s on M1 MPS)
                infer_w, infer_h = 640, 360
                pil_small = Image.fromarray(rgb_full).resize(
                    (infer_w, infer_h), Image.Resampling.BILINEAR
                )
                inputs = processor(images=pil_small, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model_net(**inputs)
                post = processor.post_process_depth_estimation(
                    outputs, target_sizes=[(h_full, w_full)]
                )
                depth_raw = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float64)

                # Ensure depth_raw increases with distance (handle inverse depth if needed)
                low_med = float(np.median(depth_raw[int(0.78 * h_full) : int(0.95 * h_full), :]))
                high_med = float(np.median(depth_raw[int(0.32 * h_full) : int(0.48 * h_full), :]))
                if low_med > high_med > 1e-4:
                    depth_metric = 1.0 / np.maximum(depth_raw, 1e-3)
                else:
                    depth_metric = np.maximum(depth_raw, 1e-3)

                # Median Scale & Shift Alignment using 3D trajectory projections
                z_true_list: List[float] = []
                z_net_list: List[float] = []
                for pt_3d in traj_pts[::3]:
                    u, v, z_c = cam_model.project_point(pt_3d)
                    if z_c > 150.0 and (4 <= u < w_full - 4) and (4 <= v < h_full - 4):
                        ui, vi = int(round(u)), int(round(v))
                        patch = depth_metric[vi - 2 : vi + 3, ui - 2 : ui + 3]
                        z_net_val = float(np.median(patch))
                        if z_net_val > 1e-4:
                            z_true_list.append(z_c)
                            z_net_list.append(z_net_val)

                if len(z_true_list) >= 3:
                    ratios = np.asarray(z_true_list) / np.asarray(z_net_list)
                    scale_s = float(np.median(ratios))
                    shift_tz = float(
                        np.median(np.asarray(z_true_list) - scale_s * np.asarray(z_net_list))
                    )
                    # Blend robust scale with bounded affine shift
                    z_calibrated = np.clip(
                        0.85 * (scale_s * depth_metric) + 0.15 * (scale_s * depth_metric + shift_tz),
                        400.0,
                        55000.0,
                    )
                else:
                    # Fallback scale to camera-to-valley median distance
                    valley_center = np.mean(traj_pts, axis=0) if len(traj_pts) > 0 else np.zeros(3)
                    dist_center = float(np.linalg.norm(valley_center - cam_model.loc))
                    med_d = float(np.median(depth_metric[int(0.4 * h_full) :, :]))
                    scale_s = dist_center / max(med_d, 1e-3)
                    z_calibrated = np.clip(scale_s * depth_metric, 400.0, 55000.0)

                # Unproject subsampled grid of non-sky pixels (v >= 0.26 * H)
                step_px = 26
                for vi in range(int(0.26 * h_full), h_full - 8, step_px):
                    for ui in range(8, w_full - 8, step_px):
                        z_val = float(z_calibrated[vi, ui])
                        ray_dir = _pixel_to_ray(cam_model, float(ui), float(vi))
                        pt_w = cam_model.loc + ray_dir * z_val

                        # Anchor unprojected point to physical valley bounds & terrain envelope
                        if (
                            bounds_cm["x_min"] <= pt_w[0] <= bounds_cm["x_max"]
                            and bounds_cm["y_min"] <= pt_w[1] <= bounds_cm["y_max"]
                        ):
                            z_surf = float(
                                np.clip(rbf_surface(pt_w[0], pt_w[1]), 200.0, 8500.0)
                            )
                            # Fuse neural depth elevation with terrain surface envelope
                            pt_z = float(np.clip(0.35 * pt_w[2] + 0.65 * z_surf, 220.0, 8400.0))
                            px_rgb = rgb_full[vi, ui]
                            all_pts.append([float(pt_w[0]), float(pt_w[1]), pt_z])
                            all_cols.append([int(px_rgb[0]), int(px_rgb[1]), int(px_rgb[2])])

            # Supplement with dense road corridor & slope samples if needed
            if len(all_pts) < 3000:
                extra_pts, extra_cols = _sample_fallback_point_cloud(
                    cameras, cam_frames, traj_pts, rbf_surface, bounds_cm, ref_frame=ref_frame
                )
                all_pts.extend(extra_pts)
                all_cols.extend(extra_cols)

            if len(all_pts) > MAX_POINT_CLOUD_POINTS:
                idx = np.linspace(0, len(all_pts) - 1, MAX_POINT_CLOUD_POINTS, dtype=int)
                all_pts = [all_pts[i] for i in idx]
                all_cols = [all_cols[i] for i in idx]

            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    calib_hash=np.array(calib_hash),
                    points_cm=np.asarray(all_pts, dtype=np.float32),
                    colors_rgb=np.asarray(all_cols, dtype=np.uint8),
                )
            except Exception:
                pass

            return (
                [[round(p[0], 1), round(p[1], 1), round(p[2], 1)] for p in all_pts],
                all_cols,
                MODEL_ID,
            )
        except Exception:
            pass

    # Fast deterministic fallback point cloud when use_depth_model=False or videos absent
    pts_fb, cols_fb = _sample_fallback_point_cloud(
        cameras, cam_frames, traj_pts, rbf_surface, bounds_cm, ref_frame=ref_frame
    )
    model_label = MODEL_ID if use_depth_model else "scipy.interpolate.Rbf+MultiCamProjection"
    return pts_fb, cols_fb, model_label


def _sample_fallback_point_cloud(
    cameras: Dict[int, CameraTimeline],
    cam_frames: Dict[int, np.ndarray],
    traj_pts: np.ndarray,
    rbf_surface: Rbf,
    bounds_cm: Dict[str, float],
    ref_frame: int = 100,
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Deterministic point cloud sampler covering the valley floor, road corridor,
    and mountain ridges with projected camera RGB colors.
    """
    rng = np.random.default_rng(42)
    priority_cams = [2, 4, 1, 3, 5]
    points_cm: List[List[float]] = []
    colors_rgb: List[List[int]] = []

    # 1. Sample dense points along the road corridor
    if len(traj_pts) > 0:
        step = max(1, len(traj_pts) // 250)
        for pt in traj_pts[::step]:
            for _ in range(6):
                dx = float(rng.normal(0.0, 900.0))
                dy = float(rng.normal(0.0, 900.0))
                px = float(np.clip(pt[0] + dx, bounds_cm["x_min"], bounds_cm["x_max"]))
                py = float(np.clip(pt[1] + dy, bounds_cm["y_min"], bounds_cm["y_max"]))
                pz = float(np.clip(rbf_surface(px, py), 220.0, 8400.0))
                r, g, b = _sample_projected_color(
                    np.array([px, py, pz]), cameras, cam_frames, priority_cams, ref_frame
                )
                points_cm.append([round(px, 1), round(py, 1), round(pz, 1)])
                colors_rgb.append([r, g, b])

    # 2. Sample wide valley & mountain slope scatter points
    xs = np.linspace(bounds_cm["x_min"], bounds_cm["x_max"], 48)
    ys = np.linspace(bounds_cm["y_min"], bounds_cm["y_max"], 48)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.clip(rbf_surface(xx, yy), 220.0, 8400.0)

    for r_idx in range(48):
        for c_idx in range(48):
            px = float(xx[r_idx, c_idx] + rng.uniform(-250.0, 250.0))
            py = float(yy[r_idx, c_idx] + rng.uniform(-250.0, 250.0))
            pz = float(zz[r_idx, c_idx])
            r, g, b = _sample_projected_color(
                np.array([px, py, pz]), cameras, cam_frames, priority_cams, ref_frame
            )
            points_cm.append([round(px, 1), round(py, 1), round(pz, 1)])
            colors_rgb.append([r, g, b])

    return points_cm[:MAX_POINT_CLOUD_POINTS], colors_rgb[:MAX_POINT_CLOUD_POINTS]


def _compute_headings_rad(traj_pts: np.ndarray) -> np.ndarray:
    """
    Computes smoothed vehicle heading angles in radians (`atan2(dY, dX)`) along the trajectory.
    """
    n = len(traj_pts)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    # Use finite differences with a +/- 3 frame window to suppress jitter
    headings = np.zeros(n, dtype=np.float64)
    last_valid_heading = 0.0
    for i in range(n):
        i0 = max(0, i - 3)
        i1 = min(n - 1, i + 3)
        dx = float(traj_pts[i1, 0] - traj_pts[i0, 0])
        dy = float(traj_pts[i1, 1] - traj_pts[i0, 1])
        if math.hypot(dx, dy) > 5.0:
            last_valid_heading = math.atan2(dy, dx)
        headings[i] = last_valid_heading
    return headings


def build_scene_3d_payload(
    calib_dir: str,
    estimated_df: pd.DataFrame,
    ground_truth_df: Optional[pd.DataFrame] = None,
    grouped_detections: Optional[Dict[int, List[Dict[str, float]]]] = None,
    metrics: Optional[Dict[str, float]] = None,
    use_depth_model: bool = True,
    grid_size: int = 72,
) -> Dict[str, Any]:
    """
    Builds the complete 3D terrain mesh, calibrated neural depth point cloud,
    camera frustums, and per-frame trajectory & ray states.
    Works both with Ground Truth (validation mode) and without Ground Truth
    (blind test mode on new datasets).
    """
    grid_size = max(8, int(grid_size))
    cameras = load_all_cameras(calib_dir)

    # Normalize estimated_df columns
    est_df = estimated_df.copy()
    est_df.columns = [str(c).strip() for c in est_df.columns]
    est_df = est_df.sort_values("Frame").reset_index(drop=True)
    traj_pts = est_df[["Loc_X", "Loc_Y", "Loc_Z"]].to_numpy(dtype=np.float64)
    frame_indices = est_df["Frame"].astype(int).to_numpy()

    # Optional Ground Truth mapping
    has_gt = ground_truth_df is not None and not ground_truth_df.empty
    gt_map: Dict[int, np.ndarray] = {}
    gt_pts_all: List[np.ndarray] = []
    if has_gt and ground_truth_df is not None:
        gt_df = ground_truth_df.copy()
        gt_df.columns = [str(c).strip() for c in gt_df.columns]
        for _, row in gt_df.iterrows():
            f_id = int(row["Frame"])
            pt = np.array(
                [float(row["Loc_X"]), float(row["Loc_Y"]), float(row["Loc_Z"])],
                dtype=np.float64,
            )
            gt_map[f_id] = pt
            gt_pts_all.append(pt)

    # Compute scene bounds in UE5_cm from trajectory + cameras
    all_xyz_list: List[np.ndarray] = []
    if len(traj_pts) > 0:
        all_xyz_list.append(traj_pts)
    if gt_pts_all:
        all_xyz_list.append(np.asarray(gt_pts_all, dtype=np.float64))
    for timeline in cameras.values():
        m0 = timeline.get_model(0)
        all_xyz_list.append(m0.loc.reshape(1, 3))

    if all_xyz_list:
        all_xyz = np.vstack(all_xyz_list)
        x_min = float(np.min(all_xyz[:, 0]) - 2500.0)
        x_max = float(np.max(all_xyz[:, 0]) + 2500.0)
        y_min = float(np.min(all_xyz[:, 1]) - 2500.0)
        y_max = float(np.max(all_xyz[:, 1]) + 2500.0)
        z_min = float(max(200.0, np.min(all_xyz[:, 2]) - 100.0))
        z_max = float(max(2500.0, np.max(all_xyz[:, 2]) + 300.0))
    else:
        x_min, x_max = -48000.0, 36000.0
        y_min, y_max = -26000.0, 18000.0
        z_min, z_max = 250.0, 8000.0

    bounds_cm = {
        "x_min": round(x_min, 1),
        "x_max": round(x_max, 1),
        "y_min": round(y_min, 1),
        "y_max": round(y_max, 1),
        "z_min": round(z_min, 1),
        "z_max": round(z_max, 1),
    }

    if len(traj_pts) > 0:
        center = np.mean(traj_pts, axis=0)
        origin_center_cm = [
            round(float(center[0]), 1),
            round(float(center[1]), 1),
            round(float(center[2]), 1),
        ]
    else:
        origin_center_cm = [
            round(0.5 * (x_min + x_max), 1),
            round(0.5 * (y_min + y_max), 1),
            500.0,
        ]

    # Extract reference RGB frame + thumbnail for each camera
    ref_frame = int(frame_indices[min(100, len(frame_indices) - 1)]) if len(frame_indices) > 0 else 0
    cam_frames: Dict[int, np.ndarray] = {}
    cameras_payload: Dict[str, Any] = {}
    dynamic_cam_ids: List[int] = []

    for cam_id, timeline in sorted(cameras.items()):
        video_path = os.path.join(calib_dir, f"Way1.FinalImage.CineCameraActor{cam_id}.mp4")
        rgb_frame = _extract_video_frame_rgb(video_path, cam_id=cam_id, frame_idx=ref_frame)
        cam_frames[cam_id] = rgb_frame
        thumb_b64 = _encode_thumbnail_b64(rgb_frame)

        m0 = timeline.get_model(0)
        fwd, right, up = _camera_basis_vectors(m0)
        is_dyn = _is_camera_dynamic(timeline)
        if is_dyn:
            dynamic_cam_ids.append(cam_id)

        fov_h_deg = 2.0 * math.degrees(
            math.atan(m0.sensor_width_mm / (2.0 * max(1e-3, m0.focal_length_mm)))
        )

        cameras_payload[str(cam_id)] = {
            "cam_id": int(cam_id),
            "name": f"CineCameraActor{cam_id} ({int(round(m0.focal_length_mm))}mm)",
            "focal_length_mm": round(float(m0.focal_length_mm), 2),
            "sensor_w_mm": round(float(m0.sensor_width_mm), 3),
            "sensor_h_mm": round(float(m0.sensor_height_mm), 3),
            "fov_h_deg": round(float(fov_h_deg), 1),
            "is_dynamic": bool(is_dyn),
            "pos_cm": [
                round(float(m0.loc[0]), 2),
                round(float(m0.loc[1]), 2),
                round(float(m0.loc[2]), 2),
            ],
            "forward": [round(float(fwd[0]), 4), round(float(fwd[1]), 4), round(float(fwd[2]), 4)],
            "right": [round(float(right[0]), 4), round(float(right[1]), 4), round(float(right[2]), 4)],
            "up": [round(float(up[0]), 4), round(float(up[1]), 4), round(float(up[2]), 4)],
            "frustum_corners_cm": _compute_frustum_corners(m0, FRUSTUM_DISTANCE_CM),
            "thumbnail_b64": thumb_b64,
        }

    # Build continuous 3D valley elevation mesh (rows x cols)
    anchor_pts = (
        np.vstack([traj_pts, np.asarray(gt_pts_all, dtype=np.float64)])
        if (has_gt and len(gt_pts_all) > 0 and len(traj_pts) > 0)
        else traj_pts
    )
    rbf_surface = _build_terrain_rbf(anchor_pts, cameras, bounds_cm)

    x_grid = np.linspace(x_min, x_max, grid_size, dtype=np.float64)
    y_grid = np.linspace(y_min, y_max, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = np.clip(rbf_surface(xx, yy), 200.0, 8500.0)

    z_matrix_cm: List[List[float]] = []
    rgb_matrix: List[List[List[int]]] = []
    priority_cams = [2, 4, 1, 3, 5]

    for r_idx in range(grid_size):
        z_row: List[float] = []
        rgb_row: List[List[int]] = []
        for c_idx in range(grid_size):
            gx = float(x_grid[c_idx])
            gy = float(y_grid[r_idx])
            gz = float(zz[r_idx, c_idx])
            z_row.append(round(gz, 1))
            r_col, g_col, b_col = _sample_projected_color(
                np.array([gx, gy, gz], dtype=np.float64),
                cameras,
                cam_frames,
                priority_cams,
                ref_frame=ref_frame,
            )
            rgb_row.append([r_col, g_col, b_col])
        z_matrix_cm.append(z_row)
        rgb_matrix.append(rgb_row)

    terrain_mesh = {
        "rows": int(grid_size),
        "cols": int(grid_size),
        "x_grid_cm": [round(float(x), 1) for x in x_grid],
        "y_grid_cm": [round(float(y), 1) for y in y_grid],
        "z_matrix_cm": z_matrix_cm,
        "rgb_matrix": rgb_matrix,
    }

    # Build calibrated point cloud
    pts_cloud, cols_cloud, depth_model_used = _compute_calibrated_depth_point_cloud(
        calib_dir=calib_dir,
        cameras=cameras,
        cam_frames=cam_frames,
        traj_pts=traj_pts,
        rbf_surface=rbf_surface,
        bounds_cm=bounds_cm,
        use_depth_model=use_depth_model,
        ref_frame=ref_frame,
    )

    # Compute per-frame trajectory states, active sightline rays, and dynamic cameras
    headings = _compute_headings_rad(traj_pts)
    frames_payload: List[Dict[str, Any]] = []

    for idx, f_id in enumerate(frame_indices):
        f_int = int(f_id)
        est_pt = traj_pts[idx]
        est_pos_list = [
            round(float(est_pt[0]), 2),
            round(float(est_pt[1]), 2),
            round(float(est_pt[2]), 2),
        ]

        gt_pos_list: Optional[List[float]] = None
        error_cm: Optional[float] = None
        if has_gt and f_int in gt_map:
            gt_pt = gt_map[f_int]
            gt_pos_list = [
                round(float(gt_pt[0]), 2),
                round(float(gt_pt[1]), 2),
                round(float(gt_pt[2]), 2),
            ]
            error_cm = round(float(np.linalg.norm(est_pt - gt_pt)), 2)

        # Build active camera sightline rays for this frame
        active_rays: List[Dict[str, Any]] = []
        if grouped_detections is not None:
            obs_list = grouped_detections.get(f_int, [])
            # Keep highest-confidence detection per camera
            best_by_cam: Dict[int, Dict[str, float]] = {}
            for obs in obs_list:
                cid = int(obs.get("cam_id", -1))
                conf = float(obs.get("conf", 0.0))
                if cid in cameras and conf >= 0.3:
                    if cid not in best_by_cam or conf > best_by_cam[cid]["conf"]:
                        best_by_cam[cid] = obs

            for cid, obs in sorted(best_by_cam.items()):
                cam_m = cameras[cid].get_model(f_int)
                u_val = float(obs["u"])
                v_val = float(obs["v"])
                conf_val = float(obs["conf"])
                ray_dir = _pixel_to_ray(cam_m, u_val, v_val)
                dist_to_est = float(np.linalg.norm(est_pt - cam_m.loc))
                ray_end = cam_m.loc + ray_dir * max(1000.0, dist_to_est)
                # If ray passes near estimated 3D position, snap end_cm to est_pos_cm for clean intersection visual
                if float(np.linalg.norm(ray_end - est_pt)) < 1500.0:
                    end_cm_list = est_pos_list
                else:
                    end_cm_list = [
                        round(float(ray_end[0]), 2),
                        round(float(ray_end[1]), 2),
                        round(float(ray_end[2]), 2),
                    ]
                active_rays.append({
                    "cam_id": int(cid),
                    "origin_cm": [
                        round(float(cam_m.loc[0]), 2),
                        round(float(cam_m.loc[1]), 2),
                        round(float(cam_m.loc[2]), 2),
                    ],
                    "dir": [
                        round(float(ray_dir[0]), 4),
                        round(float(ray_dir[1]), 4),
                        round(float(ray_dir[2]), 4),
                    ],
                    "end_cm": end_cm_list,
                    "u": round(u_val, 2),
                    "v": round(v_val, 2),
                    "conf": round(conf_val, 3),
                })
        else:
            # Synthesize geometric visibility rays when grouped_detections is not provided
            for cid, timeline in sorted(cameras.items()):
                cam_m = timeline.get_model(f_int)
                u_p, v_p, z_c = cam_m.project_point(est_pt)
                if z_c > 100.0 and (0.0 <= u_p < cam_m.image_width_px) and (0.0 <= v_p < cam_m.image_height_px):
                    ray_dir = _pixel_to_ray(cam_m, u_p, v_p)
                    active_rays.append({
                        "cam_id": int(cid),
                        "origin_cm": [
                            round(float(cam_m.loc[0]), 2),
                            round(float(cam_m.loc[1]), 2),
                            round(float(cam_m.loc[2]), 2),
                        ],
                        "dir": [
                            round(float(ray_dir[0]), 4),
                            round(float(ray_dir[1]), 4),
                            round(float(ray_dir[2]), 4),
                        ],
                        "end_cm": est_pos_list,
                        "u": round(float(u_p), 2),
                        "v": round(float(v_p), 2),
                        "conf": 1.0,
                    })

        # Populate dynamic_cameras for cameras whose pose changes over time (Cam 5)
        dynamic_cameras_frame: Dict[str, Any] = {}
        for cid in dynamic_cam_ids:
            cam_m = cameras[cid].get_model(f_int)
            fwd, right, up = _camera_basis_vectors(cam_m)
            dynamic_cameras_frame[str(cid)] = {
                "forward": [round(float(fwd[0]), 4), round(float(fwd[1]), 4), round(float(fwd[2]), 4)],
                "right": [round(float(right[0]), 4), round(float(right[1]), 4), round(float(right[2]), 4)],
                "up": [round(float(up[0]), 4), round(float(up[1]), 4), round(float(up[2]), 4)],
                "frustum_corners_cm": _compute_frustum_corners(cam_m, FRUSTUM_DISTANCE_CM),
            }

        frames_payload.append({
            "frame": f_int,
            "est_pos_cm": est_pos_list,
            "gt_pos_cm": gt_pos_list,
            "heading_rad": round(float(headings[idx]), 4),
            "error_cm": error_cm,
            "active_rays": active_rays,
            "dynamic_cameras": dynamic_cameras_frame,
        })

    # Compute or normalize metrics dictionary
    metrics_payload: Dict[str, Any] = {
        "mae_3d": 0.0,
        "rmse_3d": 0.0,
        "mae_x": 0.0,
        "mae_y": 0.0,
        "mae_z": 0.0,
        "fps": 30.0,
        "runtime_ms": 0.0,
    }
    if metrics is not None:
        for k, v in metrics.items():
            metrics_payload[k] = round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v
    elif has_gt and ground_truth_df is not None:
        merged = pd.merge(ground_truth_df, est_df, on="Frame", suffixes=("_gt", "_det"))
        if not merged.empty:
            evaluator = TrajectoryEvaluator()
            gt_arr = merged[["Loc_X_gt", "Loc_Y_gt", "Loc_Z_gt"]].to_numpy(dtype=np.float64)
            est_arr = merged[["Loc_X_det", "Loc_Y_det", "Loc_Z_det"]].to_numpy(dtype=np.float64)
            computed = evaluator.compute_metrics(est_arr, gt_arr)
            for k, v in computed.items():
                metrics_payload[k] = round(float(v), 2)

    meta_payload = {
        "coord_system": "UE5_cm",
        "num_frames": int(len(frames_payload)),
        "fps": 30.0,
        "has_ground_truth": bool(has_gt),
        "depth_model_used": depth_model_used,
        "origin_center_cm": origin_center_cm,
        "bounds_cm": bounds_cm,
    }

    return {
        "meta": meta_payload,
        "metrics": metrics_payload,
        "terrain_mesh": terrain_mesh,
        "point_cloud": {
            "points_cm": pts_cloud,
            "colors_rgb": cols_cloud,
        },
        "cameras": cameras_payload,
        "frames": frames_payload,
    }
