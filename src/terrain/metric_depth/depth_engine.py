"""
Depth Engine for Metric Depth Estimation & Loading.

Provides multi-source depth map loading and estimation for multi-view cameras:
1. Precomputed cache loading (e.g. results/.cache_terrain_depth_v2.npz).
2. Neural metric depth inference (Depth-Anything-V2-Metric-Outdoor-Small-hf)
   using local HuggingFace cache when available.
3. Geometric optical disparity / camera-ray ground plane fallback.
4. Sky segmentation / horizon filtering ensuring non-negative, clean depth maps.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

from src.camera_model import CameraModel, CameraTimeline


MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEFAULT_DEPTH_SHAPE = (180, 320)  # (H, W)


@dataclass
class CameraFrameData:
    cam_id: int
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float64/float32
    valid_mask: np.ndarray  # (H, W) bool, True for non-sky valid terrain points
    source: str  # 'cache' | 'neural' | 'disparity' | 'geometric'


def generate_procedural_frame(
    cam_id: int,
    width: int = 320,
    height: int = 180,
) -> np.ndarray:
    """
    Deterministic procedural RGB frame fallback for headless/test environments.
    Produces realistic sky, mountain slope, and valley floor coloration.
    """
    v_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    u_norm = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    horizon = 0.28 + 0.03 * ((cam_id % 3) - 1)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    sky_mask = v_norm < horizon
    terrain_t = np.clip((v_norm - horizon) / max(1e-3, 1.0 - horizon), 0.0, 1.0)
    wave = 0.05 * np.sin(u_norm * 12.0 + float(cam_id))

    r = np.where(sky_mask, 165 - 40 * v_norm, 92 + 38 * terrain_t + 20 * wave)
    g = np.where(sky_mask, 195 - 30 * v_norm, 108 + 28 * (1.0 - terrain_t) + 15 * wave)
    b = np.where(sky_mask, 225 - 20 * v_norm, 74 + 18 * (1.0 - terrain_t))

    img[:, :, 0] = np.clip(r, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(b, 0, 255).astype(np.uint8)
    return img


def extract_video_frame(
    video_path: str,
    cam_id: int,
    frame_idx: int = 0,
    target_shape: Optional[Tuple[int, int]] = DEFAULT_DEPTH_SHAPE,
) -> np.ndarray:
    """
    Extracts an RGB frame from an MP4 video file, resized to target_shape (H, W).
    Falls back gracefully to procedural generation if video cannot be decoded.
    """
    h_target, w_target = target_shape if target_shape is not None else (180, 320)

    if os.path.exists(video_path):
        if cv2 is not None:
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame_bgr = cap.read()
                    cap.release()
                    if ok and frame_bgr is not None:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        if target_shape is not None:
                            frame_rgb = cv2.resize(
                                frame_rgb, (w_target, h_target), interpolation=cv2.INTER_LINEAR
                            )
                        return frame_rgb
            except Exception:
                pass

        ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        if os.path.exists(ffmpeg_bin):
            try:
                timestamp_sec = max(0.0, frame_idx / 30.0)
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
                proc = subprocess.run(cmd, capture_output=True, timeout=5.0, check=False)
                if proc.returncode == 0 and len(proc.stdout) > 100:
                    pil_img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
                    if target_shape is not None:
                        pil_img = pil_img.resize((w_target, h_target), Image.Resampling.BILINEAR)
                    return np.asarray(pil_img, dtype=np.uint8)
            except Exception:
                pass

    return generate_procedural_frame(cam_id, width=w_target, height=h_target)


def estimate_sky_mask(rgb_frame: np.ndarray, threshold_blue_ratio: float = 1.05) -> np.ndarray:
    """
    Detects sky pixels in an RGB frame based on color and vertical position.
    Returns boolean array of shape (H, W) where True = sky (to be excluded).
    """
    h, w, _ = rgb_frame.shape
    r = rgb_frame[:, :, 0].astype(np.float32)
    g = rgb_frame[:, :, 1].astype(np.float32)
    b = rgb_frame[:, :, 2].astype(np.float32)

    # Blue-dominant or bright haze regions
    is_blue_sky = (b > r * threshold_blue_ratio) & (b > 110)
    is_bright_haze = (r > 190) & (g > 200) & (b > 210)

    # Sky can only appear in the upper 45% of the camera image
    upper_mask = np.zeros((h, w), dtype=bool)
    upper_mask[: int(0.45 * h), :] = True

    sky_mask = (is_blue_sky | is_bright_haze) & upper_mask
    return sky_mask


def compute_geometric_ground_depth(
    camera_model: CameraModel,
    shape: Tuple[int, int] = DEFAULT_DEPTH_SHAPE,
    nominal_ground_z: float = 400.0,
    relief_offset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes optical depth D(u, v) by intersecting rays with the valley floor
    (nominal_ground_z + relief_offset).
    Returns:
        depth: (H, W) float64 in centimeters
        valid_mask: (H, W) bool, True where ray hits the ground in front of camera
    """
    h, w = shape
    fx_orig, fy_orig, cx_orig, cy_orig = camera_model.get_intrinsics()
    scale_x = w / float(camera_model.image_width_px)
    scale_y = h / float(camera_model.image_height_px)

    fx = fx_orig * scale_x
    fy = fy_orig * scale_y
    cx = cx_orig * scale_x
    cy = cy_orig * scale_y

    u_coords = np.linspace(0, w - 1, w, dtype=np.float64)
    v_coords = np.linspace(0, h - 1, h, dtype=np.float64)
    uu, vv = np.meshgrid(u_coords, v_coords)

    rx = (uu - cx) / fx
    ry = (vv - cy) / fy
    rz = np.ones_like(rx)

    ray_cam = np.stack([rx, ry, rz], axis=-1)  # (H, W, 3)
    # CameraModel.R_cam maps world -> cam, so R_cam.T maps cam -> world:
    ray_world = ray_cam @ camera_model.R_cam  # (H, W, 3)

    rw_z = ray_world[..., 2]
    # Rays pointing downwards towards ground:
    down_mask = rw_z < -0.005

    target_z = np.full((h, w), nominal_ground_z, dtype=np.float64)
    if relief_offset is not None:
        target_z = target_z + np.asarray(relief_offset, dtype=np.float64)

    depth = np.zeros((h, w), dtype=np.float64)
    depth[down_mask] = (target_z[down_mask] - camera_model.loc[2]) / rw_z[down_mask]

    # Non-downward pointing rays (sky or distant horizon)
    valid_mask = down_mask & (depth > 10.0) & (depth < 250000.0)

    # Ensure strictly non-negative
    depth = np.clip(depth, 0.0, 300000.0)
    return depth, valid_mask


def compute_geometric_disparity_depth(
    rgb_frame: np.ndarray,
    camera_model: Optional[CameraModel] = None,
    shape: Tuple[int, int] = DEFAULT_DEPTH_SHAPE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast geometric optical disparity fallback from frame perspective.
    Near bottom = close (small depth, large disparity).
    Near horizon = far (large depth, small disparity).
    """
    h, w = shape
    v_coords = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    horizon = 0.28
    if camera_model is not None:
        # Cam pitch shifts the horizon
        pitch = camera_model.rot[1]
        horizon = float(np.clip(0.30 - pitch * 0.015, 0.15, 0.60))

    t = np.clip((v_coords - horizon) / max(1e-3, 1.0 - horizon), 0.0, 1.0)
    # Perspective optical depth scales inversely with normalized distance from horizon:
    # d ~ d_min / (t + eps)
    d_min = 2500.0  # 25 meters closest
    d_max = 85000.0  # 850 meters horizon
    depth = d_min + (1.0 - t) ** 1.8 * (d_max - d_min)
    depth = np.broadcast_to(depth, (h, w)).copy()

    sky_mask = v_coords < horizon
    sky_mask_2d = np.broadcast_to(sky_mask, (h, w))
    valid_mask = ~sky_mask_2d

    return depth, valid_mask


class DepthEngine:
    """
    Orchestrates depth estimation for the 5-camera Peleng rig.
    Supports cache lookup, Depth-Anything-V2 neural inference, and geometric disparity fallback.
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        use_neural: bool = False,
        target_shape: Tuple[int, int] = DEFAULT_DEPTH_SHAPE,
    ) -> None:
        self.cache_path = cache_path
        self.use_neural = use_neural
        self.target_shape = target_shape
        self._cached_relief: Optional[Dict[int, np.ndarray]] = None
        self._neural_model = None
        self._neural_processor = None
        self._init_cache()

    def _init_cache(self) -> None:
        """Attempts to load precomputed terrain relief cache if present."""
        candidate_paths = []
        if self.cache_path:
            candidate_paths.append(self.cache_path)
        # Standard repository location
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        candidate_paths.append(os.path.join(repo_root, "results", ".cache_terrain_depth_v2.npz"))

        for path in candidate_paths:
            if os.path.isfile(path):
                try:
                    data = np.load(path, allow_pickle=False)
                    relief: Dict[int, np.ndarray] = {}
                    for cid in range(1, 6):
                        k = f"relief_{cid}"
                        if k in data:
                            relief[cid] = data[k].astype(np.float64)
                    if relief:
                        self._cached_relief = relief
                        self.cache_path = path
                        break
                except Exception:
                    continue

    def _load_neural_model(self) -> bool:
        """Loads Depth-Anything-V2 offline from HuggingFace local cache if available."""
        if self._neural_model is not None:
            return True
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                MODEL_ID, local_files_only=True
            ).to(device).eval()

            self._neural_processor = processor
            self._neural_model = model
            self._neural_device = device
            return True
        except Exception:
            return False

    def get_camera_frame_data(
        self,
        cam_id: int,
        camera_model: CameraModel,
        video_path: Optional[str] = None,
        frame_idx: int = 0,
    ) -> CameraFrameData:
        """
        Retrieves RGB frame and estimated metric depth map for a specific camera.
        Guarantees clean, non-negative depth values.
        """
        # 1. Obtain RGB frame
        if video_path and os.path.exists(video_path):
            rgb = extract_video_frame(video_path, cam_id, frame_idx, self.target_shape)
        else:
            h, w = self.target_shape
            rgb = generate_procedural_frame(cam_id, width=w, height=h)

        # 2. Try cache-assisted geometric depth first (fastest and cleanest)
        if self._cached_relief is not None and cam_id in self._cached_relief:
            relief = self._cached_relief[cam_id]
            # Ensure shape matches
            if relief.shape != self.target_shape:
                pil_relief = Image.fromarray(relief)
                pil_relief = pil_relief.resize(
                    (self.target_shape[1], self.target_shape[0]), Image.Resampling.BILINEAR
                )
                relief = np.asarray(pil_relief, dtype=np.float64)

            depth, valid_mask = compute_geometric_ground_depth(
                camera_model, shape=self.target_shape, relief_offset=relief
            )
            sky_mask = estimate_sky_mask(rgb)
            valid_mask = valid_mask & (~sky_mask)
            return CameraFrameData(
                cam_id=cam_id,
                rgb=rgb,
                depth=depth,
                valid_mask=valid_mask,
                source="cache",
            )

        # 3. Try neural inference if enabled
        if self.use_neural and self._load_neural_model():
            try:
                import torch

                h_t, w_t = self.target_shape
                inputs = self._neural_processor(images=rgb, return_tensors="pt").to(
                    self._neural_device
                )
                with torch.no_grad():
                    out = self._neural_model(**inputs)
                post = self._neural_processor.post_process_depth_estimation(
                    out, target_sizes=[(h_t, w_t)]
                )
                pred_depth = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float64)
                # Raw output is in meters, convert to centimeters:
                depth = np.clip(pred_depth * 100.0, 0.0, 300000.0)
                sky_mask = estimate_sky_mask(rgb)
                valid_mask = (depth > 100.0) & (~sky_mask)
                return CameraFrameData(
                    cam_id=cam_id,
                    rgb=rgb,
                    depth=depth,
                    valid_mask=valid_mask,
                    source="neural",
                )
            except Exception:
                pass

        # 4. Fallback: Geometric ground depth or disparity
        depth, valid_mask = compute_geometric_ground_depth(camera_model, shape=self.target_shape)
        if not np.any(valid_mask):
            depth, valid_mask = compute_geometric_disparity_depth(
                rgb, camera_model, shape=self.target_shape
            )

        sky_mask = estimate_sky_mask(rgb)
        valid_mask = valid_mask & (~sky_mask)
        return CameraFrameData(
            cam_id=cam_id,
            rgb=rgb,
            depth=depth,
            valid_mask=valid_mask,
            source="geometric",
        )

    def load_all_depth_maps(
        self,
        cameras: Dict[int, CameraModel],
        video_dir: Optional[str] = None,
        frame_idx: int = 0,
    ) -> Dict[int, CameraFrameData]:
        """
        Processes and returns depth map data for all supplied cameras.
        """
        results: Dict[int, CameraFrameData] = {}
        for cam_id, model in sorted(cameras.items()):
            video_path = None
            if video_dir:
                candidate = os.path.join(
                    video_dir, f"Way1.FinalImage.CineCameraActor{cam_id}.mp4"
                )
                if os.path.exists(candidate):
                    video_path = candidate

            data = self.get_camera_frame_data(cam_id, model, video_path, frame_idx)
            results[cam_id] = data

        return results
