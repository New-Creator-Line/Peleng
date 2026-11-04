---
name: peleng-3d-tracking
description: >-
  Use this skill when implementing, debugging, or running multi-view 3D camera calibration,
  ray triangulation, 3D Kalman filtering, trajectory evaluation (3D Euclidean MAE),
  or running inference in blind test mode for the Peleng 3D Vision project.
---

# Peleng Multi-Camera 3D Tracking & Reconstruction

Architecture, mathematical conventions, and procedures for 3D multi-camera triangulation and tracking.

## 1. Camera Model & Unreal Engine 5 Coordinate Transforms
- **Coordinate System:** Unreal Engine 5 (UE5) world coordinates:
  - $X$: Forward / North
  - $Y$: Right / East
  - $Z$: Up
- **Sensor Parameters:**
  - Focal length: $35.0\text{ mm}$
  - Sensor dimensions: $23.76\text{ mm} \times 13.365\text{ mm}$
  - Image dimensions: $1920 \times 1080\text{ px}$
- **Rotation Matrix Conversion:**
  UE5 uses intrinsic Z-Y-X (Yaw-Pitch-Roll) rotation. Converting from UE5 camera basis to standard OpenCV camera basis:
  $$M_{\text{ue\_to\_cv}} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & -1 \\ 1 & 0 & 0 \end{bmatrix}$$
  Full camera extrinsics:
  $$R_{\text{cv}} = M_{\text{ue\_to\_cv}} R_{\text{ue}}^T, \quad T_{\text{cv}} = -R_{\text{cv}} C_{\text{ue}}$$

## 2. Multi-View Triangulation
- Direct Linear Transform (DLT) solved via Singular Value Decomposition (SVD):
  $$A X = 0$$
- **Reprojection Error Filtering:**
  Detections with reprojection error exceeding threshold (default: $40.0\text{ px}$) are rejected to protect against false positives and occluded camera rays.
- Minimum cameras required for triangulation: $\ge 2$.

## 3. 3D Kalman Filter (Constant Velocity)
- State vector: $\mathbf{x} = [x, y, z, \dot{x}, \dot{y}, \dot{z}]^T \in \mathbb{R}^6$
- Measurement vector: $\mathbf{z} = [x, y, z]^T \in \mathbb{R}^3$
- Standard tuning:
  - Time step: $dt = 1/30\text{ s}$
  - Process noise standard deviation: $\sigma_a = 4.0\text{ cm/s}^2$
  - Measurement noise standard deviation: $\sigma_m = 1.5\text{ cm}$
- **Occlusion Handling (Coasting):** When $< 2$ cameras detect the target, the filter executes prediction-only steps up to max coast limit (30 frames / 1.0 second).

## 4. Trajectory Evaluation & Metric
The official validation metric is 3D Euclidean Mean Absolute Error:
$$\text{MAE}_{\text{3D}} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{p}_i - \mathbf{p}_i^{\text{gt}}\|_2$$

### Critical Gotchas:
- **Excel Date Glitch:** At row 402 of `object_transform_SM_AA_track.csv`, Excel auto-formatted float values into Russian date strings (e.g., `'авг.13'`). `src/evaluator.py` automatically parses and repairs these anomalies via cubic spline / linear interpolation.

## 5. Execution Commands
Run full pipeline on real YOLO detections:
```bash
python main.py \
  --calib-dir data/raw/hackathon \
  --detections results/detections/detections_2d_yolo_real.csv \
  --gt data/ground_truth/object_transform_SM_AA_track.csv \
  --output results/trajectories/output_trajectory_yolo.csv \
  --metrics-out results/metrics/metrics_summary_yolo.json
```
*(Note: `--calib-dir хакатон` and `--gt хакатон/object_transform_SM_AA_track.csv` are also fully supported for backwards compatibility).*

Blind Test Mode (when ground truth is withheld):
```bash
python main.py \
  --calib-dir test_dataset \
  --detections results/detections_test.csv \
  --output results/output_trajectory_blind.csv \
  --no-plot
```
