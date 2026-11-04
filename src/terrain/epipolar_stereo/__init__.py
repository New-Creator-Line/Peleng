"""
Epipolar Feature Stereo & Multi-View Triangulation Package (Task 014: Method 1).

Conforms to:
- .memory_bank/contracts/terrain_method_contract.md (TerrainPayload)
- .memory_bank/tasks/014-terrain-epipolar-stereo.md

Exposes:
- reconstruct_terrain(calib_dir='хакатон', video_dir='хакатон', frame_idx=0) -> Dict[str, Any]
- to_scene_payload_layer(terrain_payload: Dict[str, Any]) -> Dict[str, Any]
- compute_fundamental_matrix
- EpipolarMatcher
- StereoMatch
- EpipolarReconstructor
- TriangulatedPoint
- dlt_triangulate_point
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.camera_model import CameraModel, load_all_cameras
from src.terrain.epipolar_stereo.matcher import (
    EpipolarMatcher,
    StereoMatch,
    compute_fundamental_matrix,
    extract_features_and_patches,
)
from src.terrain.epipolar_stereo.reconstructor import (
    BOUNDS_X_MAX,
    BOUNDS_X_MIN,
    BOUNDS_Y_MAX,
    BOUNDS_Y_MIN,
    BOUNDS_Z_MAX,
    BOUNDS_Z_MIN,
    EpipolarReconstructor,
    TriangulatedPoint,
    dlt_triangulate_point,
    is_point_within_bounds,
)

__all__ = [
    "reconstruct_terrain",
    "to_scene_payload_layer",
    "compute_fundamental_matrix",
    "EpipolarMatcher",
    "StereoMatch",
    "EpipolarReconstructor",
    "TriangulatedPoint",
    "dlt_triangulate_point",
    "BOUNDS_X_MIN",
    "BOUNDS_X_MAX",
    "BOUNDS_Y_MIN",
    "BOUNDS_Y_MAX",
    "BOUNDS_Z_MIN",
    "BOUNDS_Z_MAX",
]


def _resolve_dir(path: str) -> str:
    """Resolves directory relative to current working directory or Peleng root."""
    if os.path.isdir(path):
        return os.path.abspath(path)

    # Check relative to parent directory (Space Hackathon workspace)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate1 = os.path.join(repo_root, path)
    if os.path.isdir(candidate1):
        return candidate1

    # Check inside Peleng subdirectory
    candidate2 = os.path.join(repo_root, "Peleng", path)
    if os.path.isdir(candidate2):
        return candidate2

    return os.path.abspath(path)


def _generate_fallback_frame(cam_id: int, width: int = 1920, height: int = 1080) -> np.ndarray:
    """
    Generates a deterministic synthetic outdoor RGB image when video files are missing.
    Used for headless and synthetic isolated unit testing.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Sky gradient
    horizon = int(height * 0.35)
    for y in range(horizon):
        t = y / max(1, horizon)
        img[y, :, 0] = int(220 - 40 * t)  # B
        img[y, :, 1] = int(180 - 30 * t)  # G
        img[y, :, 2] = int(140 - 20 * t)  # R

    # Terrain gradient with high-frequency texture
    rng = np.random.RandomState(cam_id * 1000)
    texture = rng.randint(0, 40, (height - horizon, width), dtype=np.uint8)
    for y in range(horizon, height):
        t = (y - horizon) / max(1, height - horizon)
        base_b = int(70 + 30 * t)
        base_g = int(110 + 20 * t)
        base_r = int(90 + 35 * t)
        img[y, :, 0] = np.clip(base_b + texture[y - horizon], 0, 255)
        img[y, :, 1] = np.clip(base_g + texture[y - horizon], 0, 255)
        img[y, :, 2] = np.clip(base_r + texture[y - horizon], 0, 255)

    return img


def _load_or_fallback_frame(
    video_dir: str,
    cam_id: int,
    frame_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attempts to read frame_idx from CineCameraActor video; falls back to procedural frame.
    Returns (gray_frame, rgb_frame).
    """
    resolved_dir = _resolve_dir(video_dir)
    video_candidates = [
        os.path.join(resolved_dir, f"Way1.FinalImage.CineCameraActor{cam_id}.mp4"),
        os.path.join(resolved_dir, f"CineCameraActor{cam_id}.mp4"),
    ]

    for vpath in video_candidates:
        if os.path.exists(vpath):
            try:
                cap = cv2.VideoCapture(vpath)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
                    ok, frame_bgr = cap.read()
                    cap.release()
                    if ok and frame_bgr is not None:
                        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        return frame_gray, frame_rgb
            except Exception:
                pass

    # Fallback to procedural synthetic frame
    synth_bgr = _generate_fallback_frame(cam_id)
    synth_gray = cv2.cvtColor(synth_bgr, cv2.COLOR_BGR2GRAY)
    synth_rgb = cv2.cvtColor(synth_bgr, cv2.COLOR_BGR2RGB)
    return synth_gray, synth_rgb


def reconstruct_terrain(
    calib_dir: str = "хакатон",
    video_dir: str = "хакатон",
    frame_idx: int = 0,
) -> Dict[str, Any]:
    """
    Executes epipolar feature stereo matching & multi-view triangulation.
    
    Conforms to the frozen contract in .memory_bank/contracts/terrain_method_contract.md.
    
    Args:
        calib_dir: Directory containing camera CSV calibration files.
        video_dir: Directory containing camera MP4 video files.
        frame_idx: Static frame index to reconstruct (default: 0).
        
    Returns:
        TerrainPayload dictionary:
        {
            "method_name": "epipolar_stereo",
            "vertices": np.ndarray (N, 3), float32 in UE5_cm
            "faces": None,
            "colors": np.ndarray (N, 3), uint8 [0, 255] RGB
            "point_cloud": np.ndarray (N, 3), float32 in UE5_cm
            "density": "sparse_landmarks",
            "stats": {
                "vertex_count": int,
                "face_count": 0,
                "runtime_ms": float,
                "z_min_cm": float,
                "z_max_cm": float,
            }
        }
    """
    start_time = time.perf_counter()

    # 1. Load camera timelines and models
    resolved_calib = _resolve_dir(calib_dir)
    raw_cameras = load_all_cameras(resolved_calib)

    cameras: Dict[int, CameraModel] = {}
    if raw_cameras:
        for cid, timeline in raw_cameras.items():
            cameras[cid] = timeline.get_model(frame_idx)
    else:
        # Graceful synthetic fallback if calibration directory is empty
        default_locs = {
            1: np.array([-24701.96, 6623.57, 1200.0]),
            2: np.array([-45784.00, -6500.0, 7911.88]),
            3: np.array([12600.0, -22700.0, 1400.0]),
            4: np.array([33281.73, 13996.66, 7146.74]),
            5: np.array([9799.81, -3900.0, 1487.53]),
        }
        default_rots = {
            1: np.array([0.0, 0.0, -70.0]),
            2: np.array([0.0, -10.0, 0.0]),
            3: np.array([-0.68, -3.11, 115.11]),
            4: np.array([0.0, -10.0, -160.0]),
            5: np.array([-4.41, -6.97, -164.0]),
        }
        for cid in (1, 2, 3, 4, 5):
            cameras[cid] = CameraModel(
                cam_id=cid,
                loc=default_locs[cid],
                rot=default_rots[cid],
                focal_length_mm=35.0 if cid in (1, 3) else (60.0 if cid == 2 else (50.0 if cid == 4 else 20.0)),
            )

    # 2. Extract representative frames for available cameras
    frames_gray: Dict[int, np.ndarray] = {}
    frames_rgb: Dict[int, np.ndarray] = {}
    for cid in sorted(cameras.keys()):
        gray, rgb = _load_or_fallback_frame(video_dir, cid, frame_idx=frame_idx)
        frames_gray[cid] = gray
        frames_rgb[cid] = rgb

    # 3. Instantiate matcher and reconstructor
    matcher = EpipolarMatcher(
        patch_size=9,
        ncc_threshold=0.80,
        epipolar_threshold_px=2.5,
        max_keypoints_per_frame=2500,
        detector_type="fast",
        fast_threshold=15,
        min_contrast_std=5.0,
    )
    reconstructor = EpipolarReconstructor(
        max_reprojection_error_px=3.5,
        x_bounds=(BOUNDS_X_MIN, BOUNDS_X_MAX),
        y_bounds=(BOUNDS_Y_MIN, BOUNDS_Y_MAX),
        z_bounds=(BOUNDS_Z_MIN, BOUNDS_Z_MAX),
    )

    # 4. Precompute features and patches on each camera once
    cached_features: Dict[int, Any] = {}
    for cid in cameras.keys():
        cached_features[cid] = extract_features_and_patches(
            frames_gray[cid],
            rgb_img=frames_rgb[cid],
            max_keypoints=2500,
            patch_size=9,
            detector_type="fast",
            fast_threshold=15,
            min_contrast_std=5.0,
        )

    # 5. Define pairs to match (conforming to (1-2, 2-3, 3-4, 4-5) plus high-overlap pairs)
    available_cams = set(cameras.keys())
    candidate_pairs = [
        (1, 2), (2, 1),
        (2, 3), (3, 2),
        (3, 4), (4, 3),
        (4, 5), (5, 4),
        (3, 5), (5, 3),
        (1, 5), (5, 1),
        (2, 5), (5, 2),
        (1, 4), (4, 1),
        (2, 4), (4, 2),
    ]
    pairs = [(c1, c2) for c1, c2 in candidate_pairs if c1 in available_cams and c2 in available_cams]

    # 6. Execute epipolar feature matching across pairs
    all_matches: List[StereoMatch] = []
    for c1, c2 in pairs:
        cam_1 = cameras[c1]
        cam_2 = cameras[c2]
        matches = matcher.match_camera_pair(
            cam_i=cam_1,
            cam_j=cam_2,
            frame_i_gray=frames_gray[c1],
            frame_j_gray=frames_gray[c2],
            frame_i_rgb=frames_rgb[c1],
            cached_i=cached_features[c1],
            cached_j=cached_features[c2],
        )
        all_matches.extend(matches)

    # 7. Triangulate verified matches via SVD DLT and filter by reprojection error & spatial bounds
    triangulated = reconstructor.reconstruct_matches(all_matches, cameras)

    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    # 8. Assemble standard TerrainPayload dictionary
    payload = reconstructor.create_terrain_payload(triangulated, runtime_ms=runtime_ms)
    return payload


def to_scene_payload_layer(terrain_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts TerrainPayload into a layer dict ingestible by Peleng/src/viewer_3d.py.
    
    Args:
        terrain_payload: Output of reconstruct_terrain conforming to TerrainPayload.
        
    Returns:
        Dictionary with 'point_cloud' containing 'points_cm' and 'colors_rgb'.
    """
    pts = terrain_payload.get("vertices")
    if pts is None or len(pts) == 0:
        pts = terrain_payload.get("point_cloud")

    if pts is None or len(pts) == 0:
        points_list = []
        colors_list = []
    else:
        pts_arr = np.asarray(pts, dtype=np.float64)
        points_list = pts_arr.tolist()

        raw_colors = terrain_payload.get("colors")
        if raw_colors is not None and len(raw_colors) == len(pts_arr):
            col_arr = np.asarray(raw_colors)
            if col_arr.dtype in (np.float32, np.float64) and col_arr.max() <= 1.0:
                col_arr = (col_arr * 255).astype(np.uint8)
            colors_list = col_arr.astype(int).tolist()
        else:
            colors_list = [[180, 180, 180] for _ in range(len(pts_arr))]

    return {
        "point_cloud": {
            "points_cm": points_list,
            "colors_rgb": colors_list,
        }
    }
