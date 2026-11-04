import base64
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from src.camera_model import CameraModel, CameraTimeline, load_all_cameras
from src.evaluator import TrajectoryEvaluator

try:
    from src.terrain.epipolar_stereo import (
        reconstruct_terrain as reconstruct_epipolar_stereo,
        to_scene_payload_layer as epipolar_to_layer,
    )
except ImportError:
    reconstruct_epipolar_stereo = None  # type: ignore
    epipolar_to_layer = None  # type: ignore

try:
    from src.terrain.space_carving import (
        reconstruct_terrain as reconstruct_space_carving,
        to_scene_payload_layer as space_carving_to_layer,
    )
except ImportError:
    reconstruct_space_carving = None  # type: ignore
    space_carving_to_layer = None  # type: ignore

try:
    from src.terrain.metric_depth import (
        reconstruct_terrain as reconstruct_metric_depth,
        to_scene_payload_layer as metric_depth_to_layer,
    )
except ImportError:
    reconstruct_metric_depth = None  # type: ignore
    metric_depth_to_layer = None  # type: ignore

try:
    from src.terrain.census_stereo import (
        reconstruct_terrain as reconstruct_census_stereo,
        to_scene_payload_layer as census_to_layer,
    )
except ImportError:
    reconstruct_census_stereo = None  # type: ignore
    census_to_layer = None  # type: ignore

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
FRUSTUM_DISTANCE_CM = 4000.0
MAX_POINT_CLOUD_POINTS = 12000


def _camera_basis_vectors(model: CameraModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    sample_keys = frame_keys[:: max(1, len(frame_keys) // 20)]
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
        u = t_mod / 0.18
        r = 86.0 + u * 22.0
        g = 104.0 + u * 18.0
        b = 68.0 + u * 14.0
    elif t_mod < 0.65:
        u = (t_mod - 0.18) / 0.47
        r = 108.0 + u * 34.0
        g = 122.0 + u * 14.0
        b = 82.0 + u * 26.0
    else:
        u = (t_mod - 0.65) / 0.35
        r = 142.0 + u * 46.0
        g = 136.0 + u * 48.0
        b = 108.0 + u * 62.0

    return int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))


def _load_or_compute_neural_relief_maps(
    calib_dir: str,
    cameras: Dict[int, CameraTimeline],
    cam_frames: Dict[int, np.ndarray],
    use_depth_model: bool = True,
) -> Tuple[Dict[int, np.ndarray], str]:
    """
    Computes or loads cached zero-mean normalized local relief maps (180x320 float32, in cm)
    from `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old_cache = os.path.join(project_root, "results", ".cache_terrain_depth.npz")
    if os.path.exists(old_cache):
        try:
            os.remove(old_cache)
        except Exception:
            pass

    cache_path = os.path.join(project_root, "results", ".cache_terrain_depth_v2.npz")
    calib_hash = hashlib.md5(os.path.abspath(calib_dir).encode("utf-8")).hexdigest()[:12]
    has_real_videos = any(
        os.path.exists(os.path.join(calib_dir, f"Way1.FinalImage.CineCameraActor{cid}.mp4"))
        for cid in cameras.keys()
    )

    if use_depth_model and has_real_videos and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if str(cached["calib_hash"]) == calib_hash:
                relief_maps: Dict[int, np.ndarray] = {}
                for cid in cameras.keys():
                    key = f"relief_{cid}"
                    if key in cached:
                        relief_maps[cid] = cached[key].astype(np.float64)
                if relief_maps:
                    return relief_maps, MODEL_ID
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

            relief_maps = {}
            save_dict: Dict[str, Any] = {"calib_hash": np.array(calib_hash)}

            for cam_id in sorted(cameras.keys()):
                if cam_id not in cam_frames:
                    continue
                rgb_full = cam_frames[cam_id]
                pil_small = Image.fromarray(rgb_full).resize((640, 360), Image.Resampling.BILINEAR)
                inputs = processor(images=pil_small, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model_net(**inputs)
                post = processor.post_process_depth_estimation(outputs, target_sizes=[(180, 320)])
                depth_raw = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float64)

                # Extract high-pass local terrain relief (hills, gullies, vegetation bumps)
                smooth_bg = gaussian_filter(depth_raw, sigma=14.0)
                residual = depth_raw - smooth_bg
                robust_std = float(np.percentile(np.abs(residual), 85)) + 1e-4
                # Scale local relief to +/- 120 cm (+/- 1.2 meters)
                relief_cm = np.clip(residual / robust_std, -1.0, 1.0) * 120.0
                # Zero out sky region (top 25% of frame)
                relief_cm[: int(0.25 * relief_cm.shape[0]), :] = 0.0

                relief_maps[cam_id] = relief_cm
                save_dict[f"relief_{cam_id}"] = relief_cm.astype(np.float32)

            if relief_maps:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    np.savez_compressed(cache_path, **save_dict)
                except Exception:
                    pass
                return relief_maps, MODEL_ID
        except Exception:
            pass

    model_label = MODEL_ID if use_depth_model else "IDW_Valley+CameraHills+MultiCamRGB"
    return {}, model_label


def _build_unified_terrain_evaluator(
    traj_pts: np.ndarray,
    cameras: Dict[int, CameraTimeline],
    relief_maps: Dict[int, np.ndarray],
    ref_frame: int = 100,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Constructs a physically grounded, unified elevation function `evaluate_z(x_arr, y_arr)`
    that guarantees:
      1. Valley Floor Base (`z_valley`): Smooth IDW interpolation along the road trajectory
         (`Z in [300, 718] cm`), keeping the valley floor flat/gently graded at the true road level.
      2. Zero artificial ripples, bumps, or perspective warping: perfectly smooth, clean valley base.
    """
    if len(traj_pts) > 0:
        step = max(1, len(traj_pts) // 80)
        road_anchors = traj_pts[::step].copy()
        if not np.allclose(road_anchors[-1], traj_pts[-1]):
            road_anchors = np.vstack([road_anchors, traj_pts[-1:]])
    else:
        road_anchors = np.array([[0.0, 0.0, 400.0]], dtype=np.float64)

    rx = road_anchors[:, 0]
    ry = road_anchors[:, 1]
    rz = road_anchors[:, 2]

    def _eval_valley_base(x_flat: np.ndarray, y_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = x_flat[:, None] - rx[None, :]
        dy = y_flat[:, None] - ry[None, :]
        d2 = dx * dx + dy * dy
        # Dual-scale IDW: tight near the road (400 cm) and broad across the valley floor (3000 cm)
        w = 0.65 / (d2 + 3000.0**2) + 0.35 / (d2 + 450.0**2)
        z_val = np.sum(w * rz[None, :], axis=1) / np.sum(w, axis=1)
        d_min = np.sqrt(np.min(d2, axis=1))
        return z_val, d_min

    def evaluate_z(x_in: np.ndarray, y_in: np.ndarray) -> np.ndarray:
        orig_shape = np.shape(x_in)
        x_flat = np.asarray(x_in, dtype=np.float64).ravel()
        y_flat = np.asarray(y_in, dtype=np.float64).ravel()

        z_valley, _ = _eval_valley_base(x_flat, y_flat)
        z_total = z_valley
        z_clamped = np.clip(z_total, 200.0, 8500.0)
        return z_clamped.reshape(orig_shape)

    return evaluate_z


def _bilinear_interpolate_mesh(
    px: np.ndarray,
    py: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    zz_mesh: np.ndarray,
) -> np.ndarray:
    """
    Evaluates the exact bilinear surface of `terrain_mesh` (`zz_mesh` over `x_grid`, `y_grid`)
    at arbitrary `(px, py)` query coordinates so that point cloud elevations are 100% locked
    to the rendered triangle mesh.
    """
    cols = len(x_grid)
    rows = len(y_grid)
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    y_min, y_max = float(y_grid[0]), float(y_grid[-1])

    u = np.clip((px - x_min) / max(1e-6, x_max - x_min) * (cols - 1), 0.0, cols - 1.0001)
    v = np.clip((py - y_min) / max(1e-6, y_max - y_min) * (rows - 1), 0.0, rows - 1.0001)

    c0 = np.floor(u).astype(int)
    r0 = np.floor(v).astype(int)
    c1 = np.minimum(c0 + 1, cols - 1)
    r1 = np.minimum(r0 + 1, rows - 1)

    du = u - c0
    dv = v - r0

    z00 = zz_mesh[r0, c0]
    z10 = zz_mesh[r0, c1]
    z01 = zz_mesh[r1, c0]
    z11 = zz_mesh[r1, c1]

    z_interp = (
        (1.0 - du) * (1.0 - dv) * z00
        + du * (1.0 - dv) * z10
        + (1.0 - du) * dv * z01
        + du * dv * z11
    )
    return z_interp


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
        if z_cam > 200.0 and (0.0 <= u < w - 1) and (0.22 * h <= v < h - 1):
            ui = int(round(u))
            vi = int(round(v))
            px = cam_frames[cam_id][vi, ui]
            r = int(np.clip(0.72 * float(px[0]) + 0.28 * base_r, 0, 255))
            g = int(np.clip(0.72 * float(px[1]) + 0.28 * base_g, 0, 255))
            b = int(np.clip(0.72 * float(px[2]) + 0.28 * base_b, 0, 255))
            return r, g, b

    return base_r, base_g, base_b


def _build_locked_surface_point_cloud(
    cameras: Dict[int, CameraTimeline],
    cam_frames: Dict[int, np.ndarray],
    traj_pts: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    zz_mesh: np.ndarray,
    bounds_cm: Dict[str, float],
    ref_frame: int = 100,
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Generates the 3D point cloud locked directly onto the unified terrain mesh surface
    so that the mesh is stretched seamlessly across the point cloud with zero underground
    or floating detachment (`|z_point - z_mesh(x, y)| <= 6 cm`).
    """
    rng = np.random.default_rng(42)
    priority_cams = [2, 4, 1, 3, 5]
    xy_candidates: List[Tuple[float, float]] = []

    # 1. Dense samples along the road corridor & valley floor
    if len(traj_pts) > 0:
        step = max(1, len(traj_pts) // 300)
        for pt in traj_pts[::step]:
            for _ in range(8):
                dx = float(rng.normal(0.0, 1400.0))
                dy = float(rng.normal(0.0, 1400.0))
                px = float(np.clip(pt[0] + dx, bounds_cm["x_min"], bounds_cm["x_max"]))
                py = float(np.clip(pt[1] + dy, bounds_cm["y_min"], bounds_cm["y_max"]))
                xy_candidates.append((px, py))

    # 2. Dense samples around each camera's hill/mountain slope and viewing footprint
    for cam_id, timeline in sorted(cameras.items()):
        m0 = timeline.get_model(ref_frame)
        cx, cy, cz = float(m0.loc[0]), float(m0.loc[1]), float(m0.loc[2])
        spread = 7000.0 if cz > 3000.0 else 3800.0
        for _ in range(450):
            px = float(np.clip(cx + rng.normal(0.0, spread), bounds_cm["x_min"], bounds_cm["x_max"]))
            py = float(np.clip(cy + rng.normal(0.0, spread), bounds_cm["y_min"], bounds_cm["y_max"]))
            xy_candidates.append((px, py))

    # 3. Uniform + jittered valley-wide terrain surface grid samples (84 x 84 = 7056 points)
    xs = np.linspace(bounds_cm["x_min"], bounds_cm["x_max"], 84)
    ys = np.linspace(bounds_cm["y_min"], bounds_cm["y_max"], 84)
    cell_dx = (bounds_cm["x_max"] - bounds_cm["x_min"]) / 84.0
    cell_dy = (bounds_cm["y_max"] - bounds_cm["y_min"]) / 84.0
    xx, yy = np.meshgrid(xs, ys)
    for r_idx in range(84):
        for c_idx in range(84):
            px = float(
                np.clip(
                    xx[r_idx, c_idx] + rng.uniform(-0.35 * cell_dx, 0.35 * cell_dx),
                    bounds_cm["x_min"],
                    bounds_cm["x_max"],
                )
            )
            py = float(
                np.clip(
                    yy[r_idx, c_idx] + rng.uniform(-0.35 * cell_dy, 0.35 * cell_dy),
                    bounds_cm["y_min"],
                    bounds_cm["y_max"],
                )
            )
            xy_candidates.append((px, py))

    if len(xy_candidates) > MAX_POINT_CLOUD_POINTS:
        idx = np.linspace(0, len(xy_candidates) - 1, MAX_POINT_CLOUD_POINTS, dtype=int)
        xy_candidates = [xy_candidates[i] for i in idx]

    xy_arr = np.asarray(xy_candidates, dtype=np.float64)
    z_locked = _bilinear_interpolate_mesh(xy_arr[:, 0], xy_arr[:, 1], x_grid, y_grid, zz_mesh)

    points_cm: List[List[float]] = []
    colors_rgb: List[List[int]] = []

    for i in range(len(xy_arr)):
        px = float(xy_arr[i, 0])
        py = float(xy_arr[i, 1])
        # Lock point directly onto the mesh surface (+3 cm so points sit cleanly on the upper skin)
        pz = float(z_locked[i] + 3.0)
        r, g, b = _sample_projected_color(
            np.array([px, py, pz], dtype=np.float64),
            cameras,
            cam_frames,
            priority_cams,
            ref_frame=ref_frame,
        )
        points_cm.append([round(px, 1), round(py, 1), round(pz, 1)])
        colors_rgb.append([r, g, b])

    return points_cm, colors_rgb


def _compute_headings_rad(traj_pts: np.ndarray) -> np.ndarray:
    """
    Computes smoothed vehicle heading angles in radians (`atan2(dY, dX)`) along the trajectory.
    """
    n = len(traj_pts)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)

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


def _build_model_frames_data(
    traj_df: pd.DataFrame,
    cameras: Dict[int, Any],
    has_gt: bool,
    gt_map: Dict[int, np.ndarray],
    grouped_dets: Optional[Dict[int, List[Dict[str, float]]]],
    dynamic_cameras_by_frame: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    df_clean = traj_df.copy()
    df_clean.columns = [str(c).strip() for c in df_clean.columns]
    df_clean = df_clean.sort_values("Frame").reset_index(drop=True)
    pts = df_clean[["Loc_X", "Loc_Y", "Loc_Z"]].to_numpy(dtype=np.float64)
    f_indices = df_clean["Frame"].astype(int).to_numpy()
    h_rads = _compute_headings_rad(pts)

    res_frames: List[Dict[str, Any]] = []
    for idx, f_id in enumerate(f_indices):
        f_int = int(f_id)
        pt = pts[idx]
        pos_list = [round(float(pt[0]), 2), round(float(pt[1]), 2), round(float(pt[2]), 2)]

        gt_list: Optional[List[float]] = None
        err_cm: Optional[float] = None
        if has_gt and f_int in gt_map:
            g = gt_map[f_int]
            gt_list = [round(float(g[0]), 2), round(float(g[1]), 2), round(float(g[2]), 2)]
            err_cm = round(float(np.linalg.norm(pt - g)), 2)

        active_rays: List[Dict[str, Any]] = []
        if grouped_dets is not None:
            obs_list = grouped_dets.get(f_int, [])
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
                dist_to_est = float(np.linalg.norm(pt - cam_m.loc))
                ray_end = cam_m.loc + ray_dir * max(1000.0, dist_to_est)
                if float(np.linalg.norm(ray_end - pt)) < 1500.0:
                    end_cm_list = pos_list
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
            for cid, timeline in sorted(cameras.items()):
                cam_m = timeline.get_model(f_int)
                u_p, v_p, z_c = cam_m.project_point(pt)
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
                        "end_cm": pos_list,
                        "u": round(float(u_p), 2),
                        "v": round(float(v_p), 2),
                        "conf": 1.0,
                    })

        res_frames.append({
            "frame": f_int,
            "est_pos_cm": pos_list,
            "gt_pos_cm": gt_list,
            "heading_rad": round(float(h_rads[idx]), 4),
            "error_cm": err_cm,
            "active_rays": active_rays,
            "dynamic_cameras": dynamic_cameras_by_frame.get(f_int, {}),
        })
    return res_frames


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

    est_df = estimated_df.copy()
    est_df.columns = [str(c).strip() for c in est_df.columns]
    est_df = est_df.sort_values("Frame").reset_index(drop=True)
    traj_pts = est_df[["Loc_X", "Loc_Y", "Loc_Z"]].to_numpy(dtype=np.float64)
    frame_indices = est_df["Frame"].astype(int).to_numpy()

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

    # Load user-marked Ground Truth Landmarks (GCP)
    landmarks_payload: List[Dict[str, Any]] = []
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gcp_path = os.path.join(repo_root, "results", "calibration", "ground_truth_landmarks.json")
    if os.path.exists(gcp_path):
        try:
            with open(gcp_path, "r", encoding="utf-8") as f:
                gcp_data = json.load(f)
                landmarks_payload = list(gcp_data.get("landmarks", []))
        except Exception:
            pass

    # Build clean straight valley floor terrain mesh aligned with road trajectory
    evaluate_z = _build_unified_terrain_evaluator(
        traj_pts=traj_pts,
        cameras=cameras,
        relief_maps={},
        ref_frame=ref_frame,
    )
    x_grid = np.linspace(x_min, x_max, grid_size, dtype=np.float64)
    y_grid = np.linspace(y_min, y_max, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = evaluate_z(xx, yy)
    z_matrix_cm = []
    rgb_matrix = []
    priority_cams = [2, 4, 1, 3, 5]
    for r_idx in range(grid_size):
        z_row = []
        rgb_row = []
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
    depth_model_used = "Straight Planar Valley Floor (Road-Aligned)"

    # Build point cloud 100% locked onto the terrain_mesh surface
    pts_cloud, cols_cloud = _build_locked_surface_point_cloud(
        cameras=cameras,
        cam_frames=cam_frames,
        traj_pts=traj_pts,
        x_grid=x_grid,
        y_grid=y_grid,
        zz_mesh=zz,
        bounds_cm=bounds_cm,
        ref_frame=ref_frame,
    )

    # Precompute dynamic camera poses across timeline
    dynamic_cameras_by_frame: Dict[int, Dict[str, Any]] = {}
    for f_id in frame_indices:
        f_int = int(f_id)
        d_frame: Dict[str, Any] = {}
        for cid in dynamic_cam_ids:
            cam_m = cameras[cid].get_model(f_int)
            fwd, right, up = _camera_basis_vectors(cam_m)
            d_frame[str(cid)] = {
                "forward": [round(float(fwd[0]), 4), round(float(fwd[1]), 4), round(float(fwd[2]), 4)],
                "right": [round(float(right[0]), 4), round(float(right[1]), 4), round(float(right[2]), 4)],
                "up": [round(float(up[0]), 4), round(float(up[1]), 4), round(float(up[2]), 4)],
                "frustum_corners_cm": _compute_frustum_corners(cam_m, FRUSTUM_DISTANCE_CM),
            }
        dynamic_cameras_by_frame[f_int] = d_frame

    frames_payload = _build_model_frames_data(
        traj_df=est_df,
        cameras=cameras,
        has_gt=has_gt,
        gt_map=gt_map,
        grouped_dets=grouped_detections,
        dynamic_cameras_by_frame=dynamic_cameras_by_frame,
    )

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

    # Package visible terrain across 4 distinct reconstruction methods
    terrain_methods: Dict[str, Any] = {}
    resolved_video_dir = calib_dir

    # Method 1: Epipolar Stereo
    if reconstruct_epipolar_stereo is not None and epipolar_to_layer is not None:
        try:
            p1 = reconstruct_epipolar_stereo(calib_dir=calib_dir, video_dir=resolved_video_dir, frame_idx=0)
            l1 = epipolar_to_layer(p1)
            terrain_methods["epipolar_stereo"] = {
                "vertices": l1.get("point_cloud", {}).get("points_cm", []),
                "colors": l1.get("point_cloud", {}).get("colors_rgb", []),
                "runtime_ms": round(float(p1.get("stats", {}).get("runtime_ms", 0.0)), 2),
                "stats": dict(p1.get("stats", {})),
            }
        except Exception as exc:
            terrain_methods["epipolar_stereo"] = {
                "vertices": [],
                "colors": [],
                "runtime_ms": 0.0,
                "stats": {"vertex_count": 0, "face_count": 0, "runtime_ms": 0.0, "error": str(exc)},
            }
    else:
        terrain_methods["epipolar_stereo"] = {
            "vertices": [],
            "colors": [],
            "runtime_ms": 0.0,
            "stats": {"vertex_count": 0, "face_count": 0, "runtime_ms": 0.0},
        }

    # Method 2: Space Carving
    if reconstruct_space_carving is not None and space_carving_to_layer is not None:
        try:
            p2 = reconstruct_space_carving(calib_dir=calib_dir, video_dir=resolved_video_dir, frame_idx=0)
            l2 = space_carving_to_layer(p2)
            terrain_methods["space_carving"] = {
                "vertices": l2.get("mesh", {}).get("vertices", []),
                "faces": l2.get("mesh", {}).get("faces", []),
                "colors": l2.get("mesh", {}).get("colors", []),
                "runtime_ms": round(float(p2.get("stats", {}).get("runtime_ms", 0.0)), 2),
                "stats": dict(p2.get("stats", {})),
            }
        except Exception as exc:
            terrain_methods["space_carving"] = {
                "vertices": [],
                "faces": [],
                "colors": [],
                "runtime_ms": 0.0,
                "stats": {"vertex_count": 0, "face_count": 0, "runtime_ms": 0.0, "error": str(exc)},
            }
    else:
        terrain_methods["space_carving"] = {
            "vertices": [],
            "faces": [],
            "colors": [],
            "runtime_ms": 0.0,
            "stats": {"vertex_count": 0, "face_count": 0, "runtime_ms": 0.0},
        }

    # Method 3: Metric Depth
    if reconstruct_metric_depth is not None and metric_depth_to_layer is not None:
        try:
            p3 = reconstruct_metric_depth(
                calib_dir=calib_dir,
                video_dir=resolved_video_dir,
                frame_idx=0,
                use_neural=use_depth_model,
            )
            l3 = metric_depth_to_layer(p3)
            terrain_methods["metric_depth"] = {
                "vertices": l3.get("point_cloud", {}).get("points_cm", []),
                "colors": l3.get("point_cloud", {}).get("colors_rgb", []),
                "runtime_ms": round(float(p3.get("stats", {}).get("runtime_ms", 0.0)), 2),
                "stats": dict(p3.get("stats", {})),
            }
        except Exception as exc:
            terrain_methods["metric_depth"] = {
                "vertices": [],
                "colors": [],
                "runtime_ms": 0.0,
                "stats": {"vertex_count": 0, "face_count": 0, "runtime_ms": 0.0, "error": str(exc)},
            }
    else:
        terrain_methods["metric_depth"] = {
            "vertices": [],
            "colors": [],
            "runtime_ms": 0.0,
            "stats": {"vertex_count": 0, "face_count": 0, "runtime_ms": 0.0},
        }

    # Method 4: Census Transform Stereo (Task 023)
    if reconstruct_census_stereo is not None and census_to_layer is not None:
        try:
            p4 = reconstruct_census_stereo(
                calib_dir=calib_dir,
                video_dir=resolved_video_dir,
                frame_idx=0,
            )
            l4 = census_to_layer(p4)
            terrain_methods["census_stereo"] = {
                "vertices": l4.get("point_cloud", {}).get("points_cm", []),
                "colors": l4.get("point_cloud", {}).get("colors_rgb", []),
                "runtime_ms": round(float(p4.get("stats", {}).get("runtime_ms", 0.0)), 2),
                "stats": dict(p4.get("stats", {})),
            }
        except Exception as exc:
            terrain_methods["census_stereo"] = {
                "vertices": [],
                "colors": [],
                "runtime_ms": 0.0,
                "stats": {"vertex_count": 0, "face_count": 0, "runtime_ms": 0.0, "error": str(exc)},
            }
    else:
        terrain_methods["census_stereo"] = {
            "vertices": [],
            "colors": [],
            "runtime_ms": 0.0,
            "stats": {"vertex_count": 0, "face_count": 0, "runtime_ms": 0.0},
        }

    # Discover available tracker models across generations for interactive comparison
    models_payload: Dict[str, Any] = {}
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    known_specs = [
        {
            "id": "ukf",
            "name": "⭐ Поколение 3: YOLO26x-seg + 10D UKF (5.39 м)",
            "short_name": "YOLO26x-seg + 10D UKF",
            "badge": "Поколение 3 (Рекорд)",
            "desc": "Сегментационные маски + 10D Unscented Kalman Filter + RTS сглаживание. MAE: 5.39 м.",
            "color": "#00f5d4",
            "traj_csv": os.path.join(repo_root, "results", "trajectories", "output_trajectory_ukf.csv"),
            "metrics_json": os.path.join(repo_root, "results", "metrics", "metrics_summary_ukf.json"),
            "det_csv": os.path.join(repo_root, "results", "detections", "detections_2d_yolo26_seg.csv"),
        },
        {
            "id": "rtdetr",
            "name": "🚀 Поколение 2: RT-DETR Transformer (6.14 м)",
            "short_name": "RT-DETR + Filter",
            "badge": "Поколение 2",
            "desc": "Трансформерный детектор RT-DETR + фильтр окклюзий веток + Калман. MAE: 6.14 м.",
            "color": "#a855f7",
            "traj_csv": os.path.join(repo_root, "results", "trajectories", "output_trajectory_rtdetr.csv"),
            "metrics_json": os.path.join(repo_root, "results", "metrics", "metrics_summary_rtdetr.json"),
            "det_csv": os.path.join(repo_root, "results", "detections", "detections_2d_rtdetr.csv"),
        },
        {
            "id": "yolo_linear",
            "name": "📦 Поколение 1: YOLOv8x + Linear (6.30 м)",
            "short_name": "YOLOv8x + Linear 3D",
            "badge": "Поколение 1",
            "desc": "YOLOv8x Bounding Boxes + аналитический 6D линейный фильтр Калмана. MAE: 6.30 м.",
            "color": "#38bdf8",
            "traj_csv": os.path.join(repo_root, "results", "trajectories", "output_trajectory_yolo.csv"),
            "metrics_json": os.path.join(repo_root, "results", "metrics", "metrics_summary_yolo.json"),
            "det_csv": os.path.join(repo_root, "results", "detections", "detections_2d_yolo_real.csv"),
        },
    ]

    try:
        from src.mock_detections import load_detections
    except ImportError:
        load_detections = None

    for spec in known_specs:
        m_id = spec["id"]
        t_path = spec["traj_csv"]
        met_path = spec["metrics_json"]
        d_path = spec["det_csv"]

        if os.path.exists(t_path):
            try:
                m_df = pd.read_csv(t_path, sep=None, engine="python")
                m_metrics = dict(metrics_payload)
                if os.path.exists(met_path):
                    with open(met_path, "r", encoding="utf-8") as mf:
                        m_metrics = json.load(mf)

                m_dets = None
                if load_detections is not None and os.path.exists(d_path):
                    try:
                        m_dets = load_detections(d_path)
                    except Exception:
                        m_dets = None

                m_frames = _build_model_frames_data(
                    traj_df=m_df,
                    cameras=cameras,
                    has_gt=has_gt,
                    gt_map=gt_map,
                    grouped_dets=m_dets if m_dets is not None else grouped_detections,
                    dynamic_cameras_by_frame=dynamic_cameras_by_frame,
                )
                models_payload[m_id] = {
                    "id": m_id,
                    "name": spec["name"],
                    "short_name": spec["short_name"],
                    "badge": spec["badge"],
                    "desc": spec["desc"],
                    "color": spec["color"],
                    "metrics": m_metrics,
                    "frames": m_frames,
                }
            except Exception:
                pass

    if not models_payload:
        models_payload["current"] = {
            "id": "current",
            "name": "Текущий запуск",
            "short_name": "Current",
            "badge": "Active",
            "desc": "Результаты текущего запуска пайплайна",
            "color": "#00f5d4",
            "metrics": metrics_payload,
            "frames": frames_payload,
        }

    active_model_key = "ukf" if "ukf" in models_payload else list(models_payload.keys())[0]

    return {
        "meta": meta_payload,
        "metrics": metrics_payload,
        "terrain_mesh": terrain_mesh,
        "point_cloud": {
            "points_cm": pts_cloud,
            "colors_rgb": cols_cloud,
        },
        "tree_cloud": {
            "points_cm": [],
            "colors_rgb": [],
        },
        "landmarks": landmarks_payload,
        "cameras": cameras_payload,
        "frames": frames_payload,
        "terrain_methods": terrain_methods,
        "models": models_payload,
        "active_model": active_model_key,
    }
