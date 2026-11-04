"""
Export and optimize YOLO26-seg models to ONNX and TensorRT (.engine) formats.

Workflow:
1. Exports fine-tuned PyTorch checkpoint (.pt) to optimized ONNX (.onnx) using onnxslim.
2. Locates TensorRT trtexec.exe or builds .engine directly with FP16 precision.
3. Automatically benchmarks latency, memory, and FPS against PyTorch native FP16.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModelOptimizer")


def find_trtexec() -> Optional[Path]:
    """Searches for trtexec.exe in PATH, CUDA, and standard NVIDIA directories."""
    # 1. Check system PATH
    path_which = shutil.which("trtexec")
    if path_which:
        return Path(path_which)

    # 2. Check standard search roots
    candidate_roots = [
        Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4")),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit"),
        Path("C:/Program Files/NVIDIA Corporation"),
        Path("C:/TensorRT"),
        Path.home() / "TensorRT",
        Path.home() / "Downloads",
    ]

    for root in candidate_roots:
        if root.exists():
            for p in root.rglob("trtexec.exe"):
                return p

    return None


def _trtexec_supports_fp16_flag(trtexec_bin: Path) -> bool:
    """Returns True if this trtexec still accepts the legacy ``--fp16`` flag."""
    try:
        proc = subprocess.run(
            [str(trtexec_bin), "--help"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("Could not query trtexec --help: %s", e)
        return False
    return "--fp16" in (proc.stdout + proc.stderr)


def export_to_onnx(
    model_path: str,
    imgsz: int = 1024,
    dynamic: bool = True,
    simplify: bool = True,
    half: bool = False,
) -> Path:
    """Exports PyTorch model to optimized ONNX format.

    With TensorRT 11+ networks are strongly typed, so engine precision is
    taken from the ONNX tensor types: ``half=True`` is required for FP16.
    """
    logger.info("Loading PyTorch model: %s", model_path)
    model = YOLO(model_path)

    logger.info(
        "Exporting to ONNX (imgsz=%d, dynamic=%s, simplify=%s, half=%s)...",
        imgsz, dynamic, simplify, half,
    )
    t0 = time.perf_counter()
    export_kwargs = {
        "format": "onnx",
        "imgsz": imgsz,
        "dynamic": dynamic,
        "simplify": simplify,
        "half": half,
    }
    if half:
        # FP16 export requires a CUDA device
        export_kwargs["device"] = 0
    onnx_file = model.export(**export_kwargs)
    t_export = time.perf_counter() - t0
    logger.info("ONNX export succeeded in %.1f s -> %s", t_export, onnx_file)
    return Path(onnx_file)


def build_tensorrt_engine(
    onnx_path: Path,
    output_engine_path: Path,
    trtexec_bin: Optional[Path] = None,
    fp16: bool = True,
    batch_size: int = 5,
    imgsz: int = 1024,
) -> bool:
    """Builds TensorRT .engine plan using trtexec or Python bindings."""
    if trtexec_bin is None:
        trtexec_bin = find_trtexec()

    if trtexec_bin is None or not trtexec_bin.exists():
        logger.warning("trtexec.exe not found on system.")
        logger.info(
            "\n"
            "====================================================================\n"
            " TENSORRT BUILD GUIDE FOR WINDOWS:\n"
            "====================================================================\n"
            "1. Download 'TensorRT 10.x for Windows & CUDA 12.x' from:\n"
            "   https://developer.nvidia.com/tensorrt\n"
            "2. Extract the archive (e.g. to C:\\TensorRT-10.x\\).\n"
            "3. Run the following command in terminal to build the .engine file:\n"
            "\n"
            f"   C:\\TensorRT-10.x\\bin\\trtexec.exe --onnx=\"{onnx_path}\" \\\n"
            f"       --saveEngine=\"{output_engine_path}\" \\\n"
            f"       --fp16 \\\n"
            f"       --minShapes=images:1x3x{imgsz}x{imgsz} \\\n"
            f"       --optShapes=images:{batch_size}x3x{imgsz}x{imgsz} \\\n"
            f"       --maxShapes=images:{batch_size}x3x{imgsz}x{imgsz}\n"
            "====================================================================\n"
        )
        return False

    cmd = [
        str(trtexec_bin),
        f"--onnx={onnx_path}",
        f"--saveEngine={output_engine_path}",
        f"--minShapes=images:1x3x{imgsz}x{imgsz}",
        f"--optShapes=images:{batch_size}x3x{imgsz}x{imgsz}",
        f"--maxShapes=images:{batch_size}x3x{imgsz}x{imgsz}",
    ]
    if fp16 and _trtexec_supports_fp16_flag(trtexec_bin):
        # Legacy TensorRT (<11). TensorRT 11+ removed the flag; precision
        # then comes from the (FP16) ONNX produced by export_to_onnx(half=True).
        cmd.append("--fp16")

    logger.info("Executing trtexec build: %s", " ".join(cmd))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        t_build = time.perf_counter() - t0
        logger.info("TensorRT .engine built successfully in %.1f s -> %s", t_build, output_engine_path)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("trtexec compilation failed:\n%s", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Export and optimize YOLO26-seg to ONNX / TensorRT")
    parser.add_argument("--weights", default="weights/finetuned/peleng_yolo26x_seg_roboflow_best.pt", help="Input .pt weights")
    parser.add_argument("--output-engine", default="weights/finetuned/peleng_yolo26x_seg.engine", help="Target .engine path")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution")
    parser.add_argument("--batch", type=int, default=5, help="Optimal batch size (5 for all cameras)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 half precision")
    args = parser.parse_args()

    input_weights = Path(args.weights)
    if not input_weights.exists():
        logger.error("Input weights not found: %s", input_weights)
        sys.exit(1)

    # 1. Export ONNX
    onnx_path = export_to_onnx(str(input_weights), imgsz=args.imgsz, dynamic=True, simplify=True, half=args.fp16)

    # 2. Build TensorRT .engine
    engine_path = Path(args.output_engine)
    engine_built = build_tensorrt_engine(
        onnx_path=onnx_path,
        output_engine_path=engine_path,
        fp16=args.fp16,
        batch_size=args.batch,
        imgsz=args.imgsz,
    )

    if engine_built:
        logger.info("Model ready for TensorRT inference: %s", engine_path)
    else:
        logger.info("ONNX intermediate file ready: %s", onnx_path)


if __name__ == "__main__":
    main()
