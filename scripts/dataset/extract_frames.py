"""
Video Frame Extraction Utility for Dataset Preparation (Scene 2 and Multi-Camera Streams).

Extracts video frames at specified intervals or target FPS while guaranteeing
that source video files are accessed strictly read-only and never modified.
Produces structured image datasets with manifests (CSV & JSON) and optional
YOLO pseudo-labeling.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("FrameExtractor")


@dataclass(frozen=True)
class VideoInfo:
    """Metadata describing a discovered video stream."""
    path: Path
    camera_id: int
    filename: str
    total_frames: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class FrameRecord:
    """Metadata describing an extracted dataset frame."""
    filename: str
    camera_id: int
    frame_idx: int
    timestamp_sec: float
    video_file: str
    width: int
    height: int
    file_size_bytes: int


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration parameters for frame extraction."""
    input_dir: Path
    output_dir: Path
    step: int = 15
    fps: Optional[float] = None
    start_frame: int = 0
    end_frame: Optional[int] = None
    camera_filter: Optional[List[int]] = None
    prefix: str = "scen2_cam{cam}_f{frame:05d}"
    image_format: str = "jpg"
    quality: int = 95
    workers: int = 4
    save_manifest: bool = True
    dry_run: bool = False
    auto_label: bool = False
    weights_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError(f"Step must be a positive integer >= 1, got {self.step}")
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"Target FPS must be a positive float > 0.0, got {self.fps}")
        if self.start_frame < 0:
            raise ValueError(f"start_frame cannot be negative, got {self.start_frame}")
        if self.end_frame is not None and self.end_frame < self.start_frame:
            raise ValueError(f"end_frame ({self.end_frame}) cannot be smaller than start_frame ({self.start_frame})")


def parse_camera_id(filename: str) -> Optional[int]:
    """
    Extract camera integer ID from video filename using multiple known naming conventions.

    Supported patterns:
      - CineCameraActor1 -> 1
      - CineCameraActor09 -> 9
      - camera_3 -> 3
      - cam_04 -> 4
      - cam1 -> 1
    """
    patterns = [
        r"CineCameraActor(\d+)",
        r"(?:camera|cam)[_\s-]?(\d+)",
        r"^(\d+)\.mp4$",
    ]
    for pat in patterns:
        m = re.search(pat, filename, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def discover_video_files(
    input_dir: Path,
    camera_ids: Optional[List[int]] = None,
) -> List[VideoInfo]:
    """
    Discover all video files in the input directory and parse their camera parameters.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Search common video formats
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_paths: List[Path] = []
    for ext in extensions:
        video_paths.extend(input_dir.glob(ext))
        video_paths.extend(input_dir.glob(ext.upper()))

    video_paths = sorted(set(video_paths))
    discovered: List[VideoInfo] = []

    for vpath in video_paths:
        cid = parse_camera_id(vpath.name)
        if cid is None:
            logger.debug("Skipping non-camera video: %s", vpath.name)
            continue

        if camera_ids is not None and cid not in camera_ids:
            logger.debug("Skipping camera %d (not in filter: %s)", cid, camera_ids)
            continue

        cap = cv2.VideoCapture(str(vpath), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.warning("Failed to open video for inspection: %s", vpath)
            continue

        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        discovered.append(
            VideoInfo(
                path=vpath,
                camera_id=cid,
                filename=vpath.name,
                total_frames=total_f,
                fps=fps,
                width=w,
                height=h,
            )
        )

    discovered.sort(key=lambda x: x.camera_id)
    return discovered


def resolve_output_directories(base_output_dir: Path) -> tuple[Path, Path, Path]:
    """
    Resolve base dataset directory, images directory, and optional labels directory.
    If base_output_dir ends with 'images', parent is root; otherwise base_output_dir is root.
    """
    if base_output_dir.name.lower() == "images":
        root_dir = base_output_dir.parent
        img_dir = base_output_dir
    else:
        root_dir = base_output_dir
        img_dir = base_output_dir / "images"

    lbl_dir = root_dir / "labels"
    return root_dir, img_dir, lbl_dir


def extract_frames_from_video(
    video_path: Path,
    camera_id: int,
    cfg: ExtractionConfig,
    target_img_dir: Optional[Path] = None,
) -> List[FrameRecord]:
    """
    Extract frames from a single video stream in a strictly non-destructive, read-only manner.
    Uses cv2.VideoCapture with grab() skipping and retrieve() decoding for optimal speed.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    img_dir = target_img_dir if target_img_dir is not None else cfg.output_dir
    if not cfg.dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine step: either explicitly provided or derived from target fps
    effective_step = cfg.step
    if cfg.fps is not None and cfg.fps > 0:
        effective_step = max(1, int(round(native_fps / cfg.fps)))

    # Determine imwrite encode parameters
    encode_params: List[int] = []
    fmt_lower = cfg.image_format.lower().lstrip(".")
    if fmt_lower in ("jpg", "jpeg"):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, cfg.quality))]
    elif fmt_lower == "png":
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    records: List[FrameRecord] = []
    frame_idx = 0

    pbar_total = max(0, total_frames - cfg.start_frame) if total_frames > 0 else None
    pbar = tqdm(
        total=pbar_total,
        desc=f"Cam {camera_id} ({video_path.name})",
        unit="frame",
        leave=False,
        disable=(cfg.workers > 1),
    )

    try:
        while cap.isOpened():
            # Check inclusive upper frame boundary
            if cfg.end_frame is not None and frame_idx > cfg.end_frame:
                break

            ret = cap.grab()
            if not ret:
                break

            if frame_idx >= cfg.start_frame:
                should_extract = ((frame_idx - cfg.start_frame) % effective_step == 0)

                if should_extract:
                    ret_img, frame = cap.retrieve()
                    if ret_img and frame is not None:
                        out_filename = cfg.prefix.format(cam=camera_id, frame=frame_idx) + f".{fmt_lower}"
                        out_path = img_dir / out_filename

                        if not cfg.dry_run:
                            success = cv2.imwrite(str(out_path), frame, encode_params)
                            if not success:
                                logger.error("Failed to write frame to %s", out_path)
                            f_size = out_path.stat().st_size if out_path.exists() else 0
                        else:
                            f_size = 0

                        h_f, w_f = frame.shape[:2]
                        records.append(
                            FrameRecord(
                                filename=out_filename,
                                camera_id=camera_id,
                                frame_idx=frame_idx,
                                timestamp_sec=round(frame_idx / native_fps, 4),
                                video_file=video_path.name,
                                width=w_f,
                                height=h_f,
                                file_size_bytes=f_size,
                            )
                        )
                pbar.update(1)

            frame_idx += 1
    finally:
        cap.release()
        pbar.close()

    return records


def run_auto_labeling(
    img_dir: Path,
    lbl_dir: Path,
    records: List[FrameRecord],
    weights_path: Path,
    conf_threshold: float = 0.25,
    batch_size: int = 16,
) -> int:
    """
    Run YOLO inference in batches on sliced frames to generate initial pseudo-labels in YOLO format.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics package is required for auto-labeling. Skipping pseudo-labels.")
        return 0

    if not weights_path.exists():
        logger.error("Model weights not found at %s. Skipping pseudo-labeling.", weights_path)
        return 0

    lbl_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing YOLO model from %s for pseudo-labeling (batch_size=%d)...", weights_path, batch_size)
    model = YOLO(str(weights_path))

    labeled_count = 0
    logger.info("Generating pseudo-labels for %d extracted frames...", len(records))

    for i in tqdm(range(0, len(records), batch_size), desc="Auto-labeling", unit="batch"):
        batch_recs = records[i : i + batch_size]
        batch_paths = [str(img_dir / r.filename) for r in batch_recs if (img_dir / r.filename).exists()]
        if not batch_paths:
            continue

        results = model.predict(batch_paths, conf=conf_threshold, verbose=False)

        for rec, res in zip(batch_recs, results):
            lbl_path = lbl_dir / (Path(rec.filename).stem + ".txt")
            lines: List[str] = []
            if res and len(res.boxes) > 0:
                for box in res.boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    xywhn = box.xywhn[0].cpu().numpy()
                    lines.append(f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")

            with open(lbl_path, "w", encoding="utf-8") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")
                    labeled_count += 1

    logger.info("Auto-labeling complete: %d / %d frames have detections.", labeled_count, len(records))
    return labeled_count


def extract_dataset(cfg: ExtractionConfig) -> Dict[str, Any]:
    """
    Main orchestration routine: discovers videos, slices frames, and produces manifest.
    """
    t0 = time.time()
    logger.info("Scanning for video streams in: %s", cfg.input_dir)
    videos = discover_video_files(cfg.input_dir, camera_ids=cfg.camera_filter)

    if not videos:
        raise ValueError(f"No valid video streams found in {cfg.input_dir}")

    root_dir, img_dir, lbl_dir = resolve_output_directories(cfg.output_dir)

    if cfg.input_dir.resolve() == root_dir.resolve():
        raise ValueError(
            f"Target output directory ({root_dir}) cannot be identical to raw input directory ({cfg.input_dir})."
        )

    logger.info(
        "Discovered %d video streams (Camera IDs: %s)",
        len(videos),
        [v.camera_id for v in videos],
    )
    logger.info(
        "Dataset root: %s | Images: %s | Step: %d | Format: %s (Quality: %d) | Workers: %d | Dry-run: %s",
        root_dir,
        img_dir,
        cfg.step,
        cfg.image_format,
        cfg.quality,
        cfg.workers,
        cfg.dry_run,
    )

    all_records: List[FrameRecord] = []
    camera_stats: Dict[int, int] = {}

    if cfg.workers > 1 and len(videos) > 1 and not cfg.dry_run:
        logger.info("Running parallel multi-camera extraction with %d worker threads...", cfg.workers)
        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            future_to_vid = {
                executor.submit(extract_frames_from_video, v.path, v.camera_id, cfg, img_dir): v
                for v in videos
            }
            for future in as_completed(future_to_vid):
                vinfo = future_to_vid[future]
                cam_records = future.result()
                all_records.extend(cam_records)
                camera_stats[vinfo.camera_id] = len(cam_records)
                logger.info("  Camera %d finished: %d frames extracted.", vinfo.camera_id, len(cam_records))
    else:
        for vinfo in videos:
            logger.info("Extracting from Camera %d (%s, %d frames)...", vinfo.camera_id, vinfo.filename, vinfo.total_frames)
            cam_records = extract_frames_from_video(vinfo.path, vinfo.camera_id, cfg, img_dir)
            all_records.extend(cam_records)
            camera_stats[vinfo.camera_id] = len(cam_records)
            logger.info("  Camera %d: saved %d frames.", vinfo.camera_id, len(cam_records))

    # Sort all records by camera_id, then frame_idx
    all_records.sort(key=lambda r: (r.camera_id, r.frame_idx))

    elapsed = time.time() - t0
    total_extracted = len(all_records)
    total_bytes = sum(r.file_size_bytes for r in all_records)

    manifest_csv = root_dir / "manifest.csv"
    manifest_json = root_dir / "manifest.json"

    if cfg.save_manifest and not cfg.dry_run and all_records:
        root_dir.mkdir(parents=True, exist_ok=True)
        # Write CSV manifest
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "filename",
                    "camera_id",
                    "frame_idx",
                    "timestamp_sec",
                    "video_file",
                    "width",
                    "height",
                    "file_size_bytes",
                ],
            )
            writer.writeheader()
            for r in all_records:
                writer.writerow(asdict(r))

        # Write JSON manifest
        summary_payload = {
            "source_dir": str(cfg.input_dir.resolve()),
            "dataset_root": str(root_dir.resolve()),
            "images_dir": str(img_dir.resolve()),
            "step": cfg.step,
            "fps_target": cfg.fps,
            "total_extracted": total_extracted,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "elapsed_seconds": round(elapsed, 2),
            "cameras": camera_stats,
            "records": [asdict(r) for r in all_records],
        }
        with open(manifest_json, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        # Write dynamic README.md in dataset directory
        sample_w = all_records[0].width if all_records else 0
        sample_h = all_records[0].height if all_records else 0
        res_desc = f"{sample_w}x{sample_h}" if sample_w > 0 else "Unknown"
        effective_fps_desc = f"{cfg.fps:.1f}" if cfg.fps else f"{30.0 / cfg.step:.1f}"

        readme_path = root_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                f"# {cfg.input_dir.name.capitalize()} Frame Dataset\n\n"
                f"- **Source Directory:** `{cfg.input_dir}`\n"
                f"- **Total Frames:** {total_extracted}\n"
                f"- **Number of Cameras:** {len(videos)}\n"
                f"- **Step Interval:** {cfg.step} frames (~{effective_fps_desc} FPS)\n"
                f"- **Image Resolution:** {res_desc} ({cfg.image_format.upper()})\n"
                f"- **Total Disk Usage:** {total_bytes / (1024 * 1024):.1f} MB\n\n"
                f"## Camera Distribution\n"
            )
            for cid in sorted(camera_stats.keys()):
                f.write(f"- Camera {cid}: {camera_stats[cid]} frames\n")

    # Optional auto-labeling
    pseudo_labeled_count = 0
    if cfg.auto_label and not cfg.dry_run and all_records:
        weights = cfg.weights_path or Path("weights/finetuned/peleng_yolov8x_best.pt")
        pseudo_labeled_count = run_auto_labeling(img_dir, lbl_dir, all_records, weights)

    logger.info(
        "Extraction completed: %d frames sliced across %d cameras in %.2f seconds (%.1f MB).",
        total_extracted,
        len(videos),
        elapsed,
        total_bytes / (1024 * 1024),
    )

    return {
        "total_extracted": total_extracted,
        "total_videos": len(videos),
        "total_size_bytes": total_bytes,
        "elapsed_seconds": elapsed,
        "cameras": camera_stats,
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "pseudo_labeled_count": pseudo_labeled_count,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(
        description="Extract frames from multi-camera videos for dataset creation without modifying source videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/scen_2"),
        help="Path to folder containing source raw videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets/scen_2_raw"),
        help="Target folder where sliced frames and dataset files will be saved",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=15,
        help="Frame step interval (must be >= 1, e.g. 15 = every 15th frame, ~2 FPS for 30 FPS video)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target frames-per-second to extract (must be > 0.0, overrides --step if provided)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Initial frame index to start extraction (>= 0)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Maximum frame index to extract up to (inclusive)",
    )
    parser.add_argument(
        "--cams",
        type=str,
        default=None,
        help="Comma-separated camera IDs to extract (e.g. '1,2,5' or omit for all)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="scen2_cam{cam}_f{frame:05d}",
        help="Naming pattern for output files ({cam} and {frame} format variables)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Output image format",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Image encoding quality (1-100 for JPEG)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads for parallel camera extraction",
    )
    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="Run YOLO inference on sliced frames to generate initial pseudo-labels in YOLO format",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/finetuned/peleng_yolov8x_best.pt"),
        help="Weights path for auto-labeling",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not generate manifest.csv and manifest.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect videos and plan extraction without writing images to disk",
    )

    args = parser.parse_args()

    if args.step <= 0:
        parser.error("--step must be a positive integer (>= 1).")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be a positive float (> 0.0).")
    if args.start_frame < 0:
        parser.error("--start-frame cannot be negative.")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        parser.error("--end-frame cannot be less than --start-frame.")

    return args


def main() -> None:
    args = parse_args()
    cam_filter: Optional[List[int]] = None
    if args.cams:
        cam_filter = [int(c.strip()) for c in args.cams.split(",") if c.strip()]

    config = ExtractionConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step=args.step,
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        camera_filter=cam_filter,
        prefix=args.prefix,
        image_format=args.format,
        quality=args.quality,
        workers=args.workers,
        save_manifest=not args.no_manifest,
        dry_run=args.dry_run,
        auto_label=args.auto_label,
        weights_path=args.weights,
    )

    try:
        extract_dataset(config)
    except Exception as exc:
        logger.exception("Extraction failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
