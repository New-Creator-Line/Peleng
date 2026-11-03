---
name: peleng-cpp-core
description: >-
  Use this skill when building, testing, profiling, or optimizing the high-performance
  C++20 native extension (peleng_core) via CMake and pybind11 for 3D multi-camera triangulation
  and Kalman filtering in the Peleng project.
---

# Peleng C++20 Core Development & Optimization

Guide for working with the high-performance C++20 computation engine (`peleng_core`).

## 1. Architecture Overview
- **Location:** `src/peleng_core/`
  - `bindings.cpp`: Pybind11 Python bindings
  - `camera_model.hpp`: UE5 to OpenCV projection and ray generation
  - `triangulator.hpp`: Eigen-based SVD DLT multi-view triangulation
  - `kalman_filter.hpp`: 6-state constant velocity 3D Kalman filter
- **Compilation target:** C++20 (`/std:c++20` on MSVC, `-std=c++20` on GCC/Clang)
- **Dependencies:** Header-only Eigen3 and Pybind11.

## 2. Building the C++ Extension

### In-place build with Setuptools (Recommended):
```bash
python setup.py build_ext --inplace
```
This produces `peleng_core.cp<version>-win_amd64.pyd` in the project root.

### Build via CMake:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## 3. Performance Requirements & Benchmark
- Target Hackathon Throughput: $\ge 50\text{ FPS}$ (30 FPS input stream).
- Observed Throughput: **$> 70,000\text{ FPS}$** (1001 frames processed in under 15 ms).
- Eigen Jacobi SVD provides numerical stability without requiring external linear solver libraries.

## 4. Verification & Testing
Always run the dedicated C++ core unit tests after modifying C++ headers or bindings:
```bash
python -m pytest tests/test_cpp_core.py -v
```

Verification includes:
- Synthetic 3D point triangulation matching ground truth within $< 10^{-4}$ units.
- Kalman state covariance convergence across 100 timesteps.
- Zero-copy NumPy buffer passing.
