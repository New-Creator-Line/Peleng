"""
Benchmark: Pure Python/NumPy vs C++20 Core (peleng_core).
Measures:
1. Multi-view 3D Triangulation (NumPy SVD vs C++20 Jacobi SVD)
2. 3D Kalman Filter (Python Joseph update vs C++20 native matrix math)
3. End-to-End 3D Reconstruction Pipeline (Python fallback vs C++ accelerated)
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import peleng_core
from src.tracker_3d import KalmanFilter3D


def benchmark_triangulation(num_iterations: int = 10000):
    P1 = np.array([
        [0.0, 2828.28, 0.0, 0.0],
        [0.0, 0.0, -2828.28, 0.0],
        [1.0, 0.0, 0.0, 1000.0]
    ], dtype=np.float64)

    P2 = np.array([
        [-2828.28, 0.0, 0.0, 0.0],
        [0.0, 0.0, -2828.28, 0.0],
        [0.0, 1.0, 0.0, 1000.0]
    ], dtype=np.float64)

    pts = [(10.5, 20.2), (12.1, 19.8)]

    # 1. C++20 Core
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = peleng_core.triangulate([P1, P2], pts)
    t_cpp = time.perf_counter() - t0

    # 2. Pure NumPy SVD
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        A = np.zeros((4, 4), dtype=np.float64)
        A[0] = pts[0][0] * P1[2, :] - P1[0, :]
        A[1] = pts[0][1] * P1[2, :] - P1[1, :]
        A[2] = pts[1][0] * P2[2, :] - P2[0, :]
        A[3] = pts[1][1] * P2[2, :] - P2[1, :]
        _, _, vh = np.linalg.svd(A)
        X = vh[-1]
        _ = X[:3] / (X[3] + 1e-9)
    t_py = time.perf_counter() - t0

    return {
        "iterations": num_iterations,
        "time_py_ms": t_py * 1000.0,
        "time_cpp_ms": t_cpp * 1000.0,
        "fps_py": num_iterations / t_py,
        "fps_cpp": num_iterations / t_cpp,
        "speedup": t_py / t_cpp,
    }


def benchmark_kalman(num_iterations: int = 20000):
    meas = np.array([120.0, 250.0, 35.0], dtype=np.float64)

    # 1. C++20 Core
    kf_cpp = peleng_core.KalmanTracker3D(1.0 / 30.0, 4.0, 1.5)
    kf_cpp.initialize(meas)
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        kf_cpp.predict()
        kf_cpp.update(meas, 1.2)
    t_cpp = time.perf_counter() - t0

    # 2. Pure Python / NumPy Kalman Filter (disabling C++ delegate)
    kf_py = KalmanFilter3D(1.0 / 30.0, 4.0, 1.5)
    kf_py._cpp_tracker = None
    kf_py.initialize(meas)
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        kf_py.predict()
        kf_py.update(meas, 1.2)
    t_py = time.perf_counter() - t0

    return {
        "iterations": num_iterations,
        "time_py_ms": t_py * 1000.0,
        "time_cpp_ms": t_cpp * 1000.0,
        "fps_py": num_iterations / t_py,
        "fps_cpp": num_iterations / t_cpp,
        "speedup": t_py / t_cpp,
    }


def benchmark_end_to_end():
    import main

    # A. With C++ Core
    t0 = time.perf_counter()
    main.run_pipeline(
        calib_dir="хакатон",
        detections_csv="results/detections/detections_2d_yolo26_seg.csv",
        gt_csv="хакатон/object_transform_SM_AA_track.csv",
        plot=False,
    )
    t_cpp = time.perf_counter() - t0

    # B. Force Pure Python
    from src import triangulator, tracker_3d
    triangulator.HAS_CPP_CORE = False
    tracker_3d.HAS_CPP_CORE = False

    t0 = time.perf_counter()
    main.run_pipeline(
        calib_dir="хакатон",
        detections_csv="results/detections/detections_2d_yolo26_seg.csv",
        gt_csv="хакатон/object_transform_SM_AA_track.csv",
        plot=False,
    )
    t_py = time.perf_counter() - t0

    # Restore
    triangulator.HAS_CPP_CORE = True
    tracker_3d.HAS_CPP_CORE = True

    return {
        "time_py_ms": t_py * 1000.0,
        "time_cpp_ms": t_cpp * 1000.0,
        "speedup": t_py / t_cpp,
    }


if __name__ == "__main__":
    print("=" * 65)
    print("      PELENG BENCHMARK: PURE PYTHON vs NATIVE C++20 CORE")
    print("=" * 65)

    res_tri = benchmark_triangulation(10000)
    print(f"\n1. МУЛЬТИВИДОВАЯ 3D-ТРИАНГУЛЯЦИЯ ({res_tri['iterations']} итераций):")
    print(f"   - Pure NumPy SVD:        {res_tri['time_py_ms']:.2f} ms ({res_tri['fps_py']:.0f} ops/sec)")
    print(f"   - C++20 Jacobi SVD Core: {res_tri['time_cpp_ms']:.2f} ms ({res_tri['fps_cpp']:.0f} ops/sec)")
    print(f"   -> Ускорение C++:        {res_tri['speedup']:.2f}x быстрее!")

    res_kf = benchmark_kalman(20000)
    print(f"\n2. 3D ФИЛЬТР КАЛМАНА ({res_kf['iterations']} циклов Predict + Update):")
    print(f"   - Pure Python / NumPy:   {res_kf['time_py_ms']:.2f} ms ({res_kf['fps_py']:.0f} ops/sec)")
    print(f"   - C++20 Native Matrix:   {res_kf['time_cpp_ms']:.2f} ms ({res_kf['fps_cpp']:.0f} ops/sec)")
    print(f"   -> Ускорение C++:        {res_kf['speedup']:.2f}x быстрее!")

    print("\n3. ПОЛНЫЙ 3D ПАЙПЛАЙН (1001 кадр, 2 прохода + калибровка + оценка):")
    res_e2e = benchmark_end_to_end()
    print(f"   - Pure Python / NumPy:   {res_e2e['time_py_ms']:.2f} ms")
    print(f"   - C++20 Accelerated:     {res_e2e['time_cpp_ms']:.2f} ms")
    print(f"   -> Ускорение пайплайна:  {res_e2e['speedup']:.2f}x быстрее!")
    print("=" * 65)

