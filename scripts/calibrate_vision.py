#!/usr/bin/env python3
"""
Color & Sky Vision Calibration Suite for Tree & Mountain Extraction (Task 009).

Analyzes video frames from all 5 cameras (Peleng/хакатон/Way1.FinalImage.CineCameraActor1..5.mp4),
extracts exact threshold ranges and morphological boundaries for:
1. Sky segmentation mask M_sky (based on high blue tint B > 130 and B > R + 15, morphology
   snapping to the mountain ridge without mountain crest bleeding).
2. Conifer foliage & trunk detection (ExG = 2G - R - B > 10, G > R, G > B, luma in [18, 145])
   and bottom-most ground-contact root points (u_base, v_base).
3. Generates debug visual overlays with cyan line for sky/mountain boundary and red circles
   for detected tree root contact points, saved to Peleng/results/calibration/cam{id}_debug.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.ndimage
from scipy.signal import find_peaks

# Ensure Peleng root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_representative_frame(video_path: str | Path, frame_idx: int = 100) -> np.ndarray:
    """
    Extract a representative frame from a video file using OpenCV.
    
    Args:
        video_path: Path to video file.
        frame_idx: Index of frame to read (default: 100).
        
    Returns:
        frame_bgr: BGR uint8 image array of shape (H, W, 3).
        
    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the frame cannot be decoded or read.
    """
    path_str = str(video_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Video file not found: {path_str}")

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {path_str}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_idx = max(0, min(frame_idx, total_frames - 1)) if total_frames > 0 else frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Unable to read any frame from video: {path_str}")
        return frame
    finally:
        cap.release()


def segment_sky(
    frame_bgr: np.ndarray,
    close_kernel_size: int = 7,
    b_min: float = 130.0,
    b_r_margin: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment sky mask M_sky and compute column-wise mountain ridge horizon line.
    
    Sky criterion:
        High blue tint: (B > b_min) & (B > R + b_r_margin) & (B >= G).
        Top-down column ray marching establishes true atmospheric sky from the upper
        horizon boundary down to the first mountain crest or foliage encounter,
        strictly preventing mountain crests and ground snow shadows from bleeding into the sky.
        Morphological closing snaps cleanly to the natural mountain ridge silhouette.
        
    Args:
        frame_bgr: Input BGR image (H, W, 3).
        close_kernel_size: Structuring element size for morphological closing.
        b_min: Minimum blue channel threshold (default: 130.0).
        b_r_margin: Margin by which blue must exceed red (default: 15.0).
        
    Returns:
        sky_mask: Binary uint8 mask (H, W), values 0 or 255.
        horizon_y: 1D int array of length W containing the bottom-most sky pixel row
                   for each column (0 if column has no sky).
    """
    H, W = frame_bgr.shape[:2]
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)

    # High blue tint condition
    is_sky_color = (B > b_min) & (B > (R + b_r_margin)) & (B >= G)

    # Trace contiguous sky run starting from the top of the frame downwards.
    # Mountain crests and trees terminate the sky column.
    sky_mask = np.zeros((H, W), dtype=np.uint8)
    for x in range(W):
        y = 0
        if not is_sky_color[0, x]:
            top_sky = np.where(is_sky_color[:10, x])[0]
            if len(top_sky) > 0:
                y = int(top_sky[0])
            else:
                continue
        while y < H:
            if is_sky_color[y, x]:
                sky_mask[y, x] = 255
                y += 1
            else:
                # Allow a tiny lookahead (up to 4 pixels) across micro-haze or fine foliage needles
                lookahead = is_sky_color[y : min(y + 4, H), x]
                if np.any(lookahead):
                    next_y = y + int(np.argmax(lookahead))
                    sky_mask[y : next_y + 1, x] = 255
                    y = next_y + 1
                else:
                    break

    # Morphological closing to smooth fine contour and snap cleanly to mountain ridge silhouette
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)

    # Column-wise horizon extraction (bottom-most sky pixel per column)
    horizon_y = np.zeros(W, dtype=int)
    for u in range(W):
        ys = np.where(sky_mask[:, u] > 0)[0]
        if len(ys) > 0:
            horizon_y[u] = int(ys.max())

    return sky_mask, horizon_y


def segment_trees(
    frame_bgr: np.ndarray,
    sky_mask: Optional[np.ndarray] = None,
    exg_thresh: float = 10.0,
    luma_min: float = 18.0,
    luma_max: float = 145.0,
    min_tree_area: int = 100,
    min_tree_height: int = 20,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Segment conifer foliage & trunks, and extract bottom-most ground-contact root points.
    
    Tree criterion:
        ExG = 2G - R - B > exg_thresh
        G > R and G > B
        Luma in [luma_min, luma_max]
        
    Vertical morphological filtering preserves vertical trunk and foliage structures while
    suppressing horizontal noise on the terrain or vehicle.
    
    Args:
        frame_bgr: Input BGR image (H, W, 3).
        sky_mask: Optional binary sky mask.
        exg_thresh: Excess Green Index threshold (default: 10.0).
        luma_min: Minimum luminance (default: 18.0).
        luma_max: Maximum luminance (default: 145.0).
        min_tree_area: Minimum component area to consider as tree.
        min_tree_height: Minimum component height.
        
    Returns:
        tree_mask: Binary uint8 mask (H, W) after morphological filtering.
        roots: List of (u_base, v_base) ground-contact root coordinates.
    """
    H, W = frame_bgr.shape[:2]
    B = frame_bgr[:, :, 0].astype(np.float32)
    G = frame_bgr[:, :, 1].astype(np.float32)
    R = frame_bgr[:, :, 2].astype(np.float32)

    # Standard Rec.601 Luminance
    luma = 0.299 * R + 0.587 * G + 0.114 * B
    # Excess Green Index
    exg = 2.0 * G - R - B

    raw_tree = (
        (exg > exg_thresh)
        & (G > R)
        & (G > B)
        & (luma >= luma_min)
        & (luma <= luma_max)
    ).astype(np.uint8) * 255

    # Vertical morphological filtering
    # 1. Opening with thin vertical kernel eliminates horizontal speckles
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    tree_opened = cv2.morphologyEx(raw_tree, cv2.MORPH_OPEN, k_v)

    # 2. Closing with vertical kernel consolidates canopy and trunk into contiguous tree entities
    k_c = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 25))
    tree_mask = cv2.morphologyEx(tree_opened, cv2.MORPH_CLOSE, k_c)

    # Extract connected tree components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tree_mask)

    candidate_roots: List[Tuple[int, int]] = []
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        h = stats[lab, cv2.CC_STAT_HEIGHT]
        w = stats[lab, cv2.CC_STAT_WIDTH]
        if area < min_tree_area or h < min_tree_height:
            continue

        ys, xs = np.where(labels == lab)
        v_max = int(np.max(ys))

        # Filter bottom profile near ground contact (within bottom 50px of component)
        u_min, u_max = int(np.min(xs)), int(np.max(xs))
        col_bottoms: Dict[int, int] = {}
        for u in range(u_min, u_max + 1):
            mask_u = (xs == u)
            if np.any(mask_u):
                col_b = int(np.max(ys[mask_u]))
                if col_b >= v_max - 50:
                    col_bottoms[u] = col_b

        if len(col_bottoms) == 0:
            bot_xs = xs[ys >= v_max - 3]
            candidate_roots.append((int(np.median(bot_xs)), v_max))
        elif len(col_bottoms) < 30:
            u_arr = np.array(list(col_bottoms.keys()))
            candidate_roots.append((int(np.median(u_arr)), v_max))
        else:
            # Multi-tree clump: detect distinct trunk ground contact peaks
            u_arr = np.array(sorted(col_bottoms.keys()))
            v_arr = np.array([col_bottoms[u] for u in u_arr])
            v_smooth = scipy.ndimage.gaussian_filter1d(v_arr.astype(float), sigma=4.0)
            peaks, _ = find_peaks(v_smooth, distance=25, prominence=4)
            if len(peaks) > 0:
                for p in peaks:
                    candidate_roots.append((int(u_arr[p]), int(v_arr[p])))
            else:
                candidate_roots.append((int(np.median(u_arr)), v_max))

    # Spatial Non-Maximum Suppression (merge detections within 15px, prioritizing ground contact)
    unique_roots: List[Tuple[int, int]] = []
    # Sort descending by vertical row (deepest ground contact first)
    for u, v in sorted(candidate_roots, key=lambda pt: pt[1], reverse=True):
        if not any(np.hypot(u - fu, v - fv) < 15 for fu, fv in unique_roots):
            unique_roots.append((u, v))

    return tree_mask, unique_roots


def create_debug_overlay(
    frame_bgr: np.ndarray,
    horizon_y: np.ndarray,
    roots: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Generate visual verification overlay with cyan sky horizon boundary and red root circles.
    
    Args:
        frame_bgr: Original BGR image (H, W, 3).
        horizon_y: 1D array of horizon pixel row per column.
        roots: List of (u_base, v_base) root coordinates.
        
    Returns:
        overlay: BGR image with drawn visual annotations.
    """
    overlay = frame_bgr.copy()
    W = frame_bgr.shape[1]

    # 1. Draw cyan line for sky/mountain boundary (BGR: 255, 255, 0)
    horizon_pts: List[Tuple[int, int]] = []
    for u in range(W):
        if horizon_y[u] > 0:
            horizon_pts.append((u, int(horizon_y[u])))

    for i in range(len(horizon_pts) - 1):
        pt1 = horizon_pts[i]
        pt2 = horizon_pts[i + 1]
        # Only draw segment if columns are adjacent and row delta is reasonable
        if pt2[0] - pt1[0] <= 3 and abs(pt2[1] - pt1[1]) < 60:
            cv2.line(overlay, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

    # 2. Draw red circles for detected tree root contact points (BGR: 0, 0, 255)
    for u_base, v_base in roots:
        # Solid red circle
        cv2.circle(overlay, (u_base, v_base), 5, (0, 0, 255), -1, cv2.LINE_AA)
        # Black border for contrast
        cv2.circle(overlay, (u_base, v_base), 6, (0, 0, 0), 1, cv2.LINE_AA)

    return overlay


def calibrate_camera(
    cam_id: int,
    video_path: str | Path,
    output_dir: str | Path,
    frame_idx: int = 100,
) -> Dict[str, Any]:
    """
    Execute full calibration pipeline for a single camera.
    
    Args:
        cam_id: Camera identifier (1..5).
        video_path: Path to video file.
        output_dir: Directory to save debug overlay image.
        frame_idx: Index of representative frame.
        
    Returns:
        Dict containing calibration metrics and file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame_bgr = load_representative_frame(video_path, frame_idx=frame_idx)
    sky_mask, horizon_y = segment_sky(frame_bgr)
    tree_mask, roots = segment_trees(frame_bgr, sky_mask=sky_mask)

    overlay = create_debug_overlay(frame_bgr, horizon_y, roots)
    overlay_file = output_path / f"cam{cam_id}_debug.png"
    cv2.imwrite(str(overlay_file), overlay)

    valid_h = horizon_y[horizon_y > 0]
    result: Dict[str, Any] = {
        "cam_id": cam_id,
        "video_path": str(video_path),
        "frame_idx": frame_idx,
        "frame_shape": list(frame_bgr.shape),
        "sky_pixel_count": int(np.sum(sky_mask > 0)),
        "sky_cols_count": int(len(valid_h)),
        "sky_horizon_min_row": int(np.min(valid_h)) if len(valid_h) > 0 else None,
        "sky_horizon_max_row": int(np.max(valid_h)) if len(valid_h) > 0 else None,
        "sky_horizon_mean_row": float(np.mean(valid_h)) if len(valid_h) > 0 else None,
        "tree_pixel_count": int(np.sum(tree_mask > 0)),
        "tree_roots_count": len(roots),
        "tree_roots": roots,
        "debug_overlay_path": str(overlay_file),
    }
    return result


def run_all_calibrations(
    video_dir: str | Path = "Peleng/хакатон",
    output_dir: str | Path = "Peleng/results/calibration",
    frame_idx: int = 100,
    cam_ids: Optional[List[int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Run calibration pipeline across all specified cameras.
    
    Args:
        video_dir: Directory containing CineCameraActor videos.
        output_dir: Output directory for debug overlays and results.
        frame_idx: Index of representative frame.
        cam_ids: Optional list of camera IDs (default: [1, 2, 3, 4, 5]).
        
    Returns:
        Dictionary mapping cam_id to calibration results.
    """
    if cam_ids is None:
        cam_ids = [1, 2, 3, 4, 5]

    video_base = Path(video_dir)
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    results: Dict[int, Dict[str, Any]] = {}
    print("=" * 80)
    print(f"PELENG VISION CALIBRATION SUITE (Task 009) - Frame idx: {frame_idx}")
    print("=" * 80)

    for cam_id in cam_ids:
        video_filename = f"Way1.FinalImage.CineCameraActor{cam_id}.mp4"
        video_path = video_base / video_filename
        if not video_path.exists():
            alt_path = Path("хакатон") / video_filename
            if alt_path.exists():
                video_path = alt_path
            else:
                raise FileNotFoundError(f"Cannot locate video for Cam {cam_id}: {video_path}")

        res = calibrate_camera(cam_id, video_path, output_base, frame_idx=frame_idx)
        results[cam_id] = res

        print(
            f"Cam {cam_id}: Sky cols={res['sky_cols_count']}/1920, "
            f"Horizon=[{res['sky_horizon_min_row']}..{res['sky_horizon_max_row']}] "
            f"(mean {res['sky_horizon_mean_row']:.1f}px) | "
            f"Tree roots={res['tree_roots_count']} | Overlay: {res['debug_overlay_path']}"
        )

    summary_path = output_base / "calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("=" * 80)
    print(f"Calibration completed successfully. Summary saved to: {summary_path}")
    print("=" * 80)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Vision: Sky & Tree Extraction")
    parser.add_argument(
        "--video-dir",
        type=str,
        default="Peleng/хакатон",
        help="Path to folder containing CineCameraActor videos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Peleng/results/calibration",
        help="Directory to save debug overlays",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=100,
        help="Representative frame index (default: 100)",
    )
    parser.add_argument(
        "--cam-ids",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Camera IDs to process (default: 1 2 3 4 5)",
    )

    args = parser.parse_args()

    v_dir = Path(args.video_dir)
    if not v_dir.exists() and Path("хакатон").exists():
        v_dir = Path("хакатон")

    o_dir = Path(args.output_dir)
    if not o_dir.is_absolute() and not Path("Peleng").exists() and Path("results").exists():
        o_dir = Path("results/calibration")

    run_all_calibrations(
        video_dir=v_dir,
        output_dir=o_dir,
        frame_idx=args.frame_idx,
        cam_ids=args.cam_ids,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
