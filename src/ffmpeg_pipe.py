"""
GPU-first video I/O through ffmpeg pipes.

- FFmpegNvdecReader: decodes one video with the NVDEC hardware decoder (h264_cuvid)
  in a separate ffmpeg process and prefetches BGR frames in a background thread.
- FFmpegNvencWriter: feeds raw BGR frames straight into h264_nvenc (NVENC hardware
  encoder) via stdin, so no CPU mp4v encode and no second transcode pass are needed.
"""

import logging
import subprocess
import threading
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Optional, Union

import cv2
import numpy as np

try:
    import imageio_ffmpeg

    FFMPEG_EXE: Optional[str] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    FFMPEG_EXE = None

logger = logging.getLogger("FFmpegPipe")


class FFmpegNvdecReader:
    """Hardware (NVDEC) decoding of a single video into BGR frames with prefetch."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        max_frames: Optional[int] = None,
        queue_size: int = 8,
        use_nvdec: bool = True,
    ) -> None:
        if FFMPEG_EXE is None:
            raise RuntimeError("ffmpeg executable not available")
        self.width, self.height = width, height
        self.frame_bytes = width * height * 3 // 2  # NV12 from NVDEC; converted to BGR in-thread
        cmd = [FFMPEG_EXE, "-hide_banner", "-loglevel", "error"]
        if use_nvdec:
            cmd += ["-c:v", "h264_cuvid"]
        cmd += ["-i", str(path)]
        if max_frames is not None:
            cmd += ["-frames:v", str(max_frames)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "nv12", "-"]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.frame_bytes * 2
        )
        self.queue: Queue = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        assert self.proc.stdout is not None
        while not self._stop.is_set():
            buf = self.proc.stdout.read(self.frame_bytes)
            if len(buf) < self.frame_bytes:
                break
            yuv = np.frombuffer(buf, dtype=np.uint8).reshape(self.height * 3 // 2, self.width)
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            while not self._stop.is_set():
                try:
                    self.queue.put(frame, timeout=0.2)
                    break
                except Full:
                    continue
        while not self._stop.is_set():
            try:
                self.queue.put(None, timeout=0.2)
                break
            except Full:
                continue

    def read(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        try:
            self.proc.kill()
        except Exception:
            pass
        self.thread.join(timeout=2.0)


class FFmpegNvencWriter:
    """Streams raw BGR frames into h264_nvenc (hardware encoder) through a pipe."""

    def __init__(
        self,
        path: Union[str, Path],
        width: int,
        height: int,
        fps: float,
        bitrate: str = "8M",
        preset: str = "p1",
    ) -> None:
        if FFMPEG_EXE is None:
            raise RuntimeError("ffmpeg executable not available")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "h264_nvenc", "-preset", preset, "-b:v", bitrate,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.path = Path(path)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        self.proc.wait(timeout=120)
