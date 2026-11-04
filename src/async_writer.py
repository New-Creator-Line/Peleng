"""
Hardware-Accelerated Asynchronous Multi-Stream Video Writing Pipeline.

Features:
1. Threaded Video Writer Pool (AsyncMultiVideoWriter):
   - Offloads OpenCV VideoWriter encoding from the main thread to dedicated background workers.
   - Completely eliminates main thread pipeline stalls caused by synchronous video compression.
   - Each stream (Cameras 1-5 + Dashboard) encodes concurrently across multiple CPU cores.
2. Parallel Hardware NVENC Transcoding:
   - Concurrently re-encodes raw intermediate videos to clean H.264 using NVIDIA NVENC sessions in parallel.
"""

import logging
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = None

logger = logging.getLogger("AsyncMultiVideoWriter")


class _StreamWriterThread:
    """Worker thread dedicated to encoding a single video stream on a separate CPU core."""

    def __init__(
        self,
        key: Union[int, str],
        output_path: str,
        fourcc: int,
        fps: float,
        frame_size: Tuple[int, int],
        queue_size: int = 16,
    ) -> None:
        self.key = key
        self.output_path = output_path
        self.queue: Queue = Queue(maxsize=queue_size)
        self.stopped = threading.Event()
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            logger.error("Failed to initialize cv2.VideoWriter for %s", output_path)

        self.thread = threading.Thread(
            target=self._worker_loop,
            name=f"AsyncWriter-{key}",
            daemon=True,
        )
        self.thread.start()

    def _worker_loop(self) -> None:
        while True:
            try:
                frame = self.queue.get(timeout=0.1)
            except Empty:
                if self.stopped.is_set():
                    break
                continue

            if frame is None:
                self.queue.task_done()
                break

            try:
                self.writer.write(frame)
            except Exception as e:
                logger.error("Error writing frame to %s: %s", self.output_path, e)
            finally:
                self.queue.task_done()

        self.writer.release()

    def write(self, frame: np.ndarray, timeout: float = 1.0) -> bool:
        """Enqueues frame for background encoding without blocking main inference thread."""
        if self.stopped.is_set():
            return False
        # Push a copy to avoid in-place corruption by subsequent pipeline steps
        frame_copy = frame.copy()
        try:
            self.queue.put(frame_copy, timeout=timeout)
            return True
        except Full:
            logger.warning("Writer queue full for stream %s, dropping frame", self.key)
            return False

    def close(self) -> None:
        """Signals stop and waits for all queued frames to be written."""
        self.stopped.set()
        try:
            self.queue.put(None, timeout=0.5)
        except Exception:
            pass
        self.thread.join(timeout=30.0)


class AsyncMultiVideoWriter:
    """
    High-throughput non-blocking multi-stream video writing manager.
    Encodes up to N parallel video streams concurrently in background threads.
    """

    def __init__(self, fourcc: Optional[int] = None, queue_size: int = 16) -> None:
        self.fourcc = fourcc or cv2.VideoWriter_fourcc(*"mp4v")
        self.queue_size = queue_size
        self.streams: Dict[Union[int, str], _StreamWriterThread] = {}
        self.paths: Dict[Union[int, str], Path] = {}

    def add_stream(
        self,
        key: Union[int, str],
        output_path: Union[str, Path],
        fps: float,
        frame_size: Tuple[int, int],
    ) -> None:
        """Registers a new video stream with a dedicated worker thread."""
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        self.paths[key] = path_obj
        self.streams[key] = _StreamWriterThread(
            key=key,
            output_path=str(path_obj),
            fourcc=self.fourcc,
            fps=fps,
            frame_size=frame_size,
            queue_size=self.queue_size,
        )

    def write(self, key: Union[int, str], frame: np.ndarray) -> bool:
        """Dispatches frame to the background stream worker. Returns instantly."""
        stream = self.streams.get(key)
        if stream is None:
            return False
        return stream.write(frame)

    def close(self) -> None:
        """Cleanly drains all stream queues and finalizes video files."""
        for stream in self.streams.values():
            stream.close()
        self.streams.clear()

    def __enter__(self) -> "AsyncMultiVideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def transcode_to_nvenc_parallel(
    video_paths: list[Union[str, Path]],
    preset: str = "p2",
    bitrate: str = "5M",
    max_workers: int = 6,
) -> Dict[str, bool]:
    """
    Transcodes multiple video files in parallel using hardware NVENC H.264 encoder.
    Saves massive amounts of time compared to sequential single-file transcoding.
    """
    if FFMPEG_EXE is None:
        logger.warning("FFmpeg executable not available. Skipping NVENC post-transcode.")
        return {str(p): False for p in video_paths}

    results: Dict[str, bool] = {}

    def _transcode_single(v_path: Path) -> Tuple[str, bool]:
        temp_path = v_path.with_name(f"{v_path.stem}_raw_tmp.mp4")
        try:
            if not v_path.exists():
                return str(v_path), False
            os.replace(str(v_path), str(temp_path))
            cmd = [
                FFMPEG_EXE,
                "-y",
                "-i", str(temp_path),
                "-c:v", "h264_nvenc",
                "-preset", preset,
                "-b:v", bitrate,
                "-pix_fmt", "yuv420p",
                str(v_path),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if temp_path.exists():
                os.remove(str(temp_path))
            logger.info("NVIDIA NVENC hardware encoding complete: %s", v_path.name)
            return str(v_path), True
        except Exception as e:
            logger.warning("NVENC encoding fallback for %s: %s", v_path.name, e)
            if temp_path.exists() and not v_path.exists():
                os.replace(str(temp_path), str(v_path))
            return str(v_path), False

    paths_to_run = [Path(p) for p in video_paths if Path(p).exists()]
    with ThreadPoolExecutor(max_workers=min(len(paths_to_run), max_workers)) as executor:
        futures = {executor.submit(_transcode_single, p): p for p in paths_to_run}
        for future in as_completed(futures):
            v_name, success = future.result()
            results[v_name] = success

    return results
