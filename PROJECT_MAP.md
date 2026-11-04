# Project Map: Peleng 3D Multi-Camera Tracking Core

## 1. Quick Navigation & Architecture Map
- **Project Root**: `C:\projects\Peleng`
- **Domain**: Real-Time 3D Trajectory Reconstruction & Multi-Camera Vehicle Tracking (Unreal Engine 5 synthetic environment & real-world camera geometry).
- **Core Stacks**: Python 3.14 (NumPy, OpenCV, PyTorch, Ultralytics YOLOv8 / RT-DETR), C++20 (Eigen3, pybind11).

```
Peleng/
├── data/                               # Central data hub
│   ├── raw/
│   │   └── hackathon/                  # 5-camera video streams (.mp4) and camera transform CSVs
│   ├── ground_truth/                   # Clean ground truth trajectory (object_transform_SM_AA_track.csv)
│   ├── datasets/
│   │   ├── yolo_v3/                    # Primary 450-image YOLOv8/RT-DETR dataset (train/valid/data.yaml)
│   │   └── legacy_v1/                  # Historical v1 dataset
│   └── assets/                         # 3D assets (CargoTruck.fbx)
├── docs/                               # Documentation & specifications
│   ├── specifications/                 # Hackathon requirements (TZ_hackathon.pdf)
│   └── architecture/                   # Math & coordinate system diagrams
├── weights/                            # Neural network weights
│   ├── pretrained/                     # Base models (yolov8x.pt, rtdetr-l.pt)
│   └── finetuned/                      # Best checkpoints (peleng_yolov8x_best.pt, peleng_rtdetr_best.pt)
├── scripts/                            # Auxiliary operational CLI scripts
│   ├── training/                       # train_yolo.py, train_rtdetr.py
│   ├── inference/                      # process_videos_yolo.py (multi-view video & 2D detection generator)
│   └── dataset/                        # split_v3_dataset.py (stratified train/val group splitter)
├── src/                                # Core library & pipeline algorithms
│   ├── peleng_core/                    # C++20 acceleration engine (Eigen3 + pybind11)
│   │   ├── camera_model.hpp            # UE5-to-OpenCV coordinate transform & ray generation
│   │   ├── triangulator.hpp            # Multi-view DLT ray intersection via Eigen Jacobi SVD
│   │   ├── kalman_filter.hpp           # 6-state constant velocity 3D Kalman Filter
│   │   └── bindings.cpp                # Python bindings for peleng_core
│   ├── camera_model.py                 # Python camera timeline & ray projection
│   ├── triangulator.py                 # Multi-view SVD triangulation with C++ fallback
│   ├── tracker_3d.py                   # 3D Kalman tracking with coasting & C++ fallback
│   ├── yolo_detector.py                # YOLOv8x & RT-DETR detector wrapper with occlusion reasoning
│   ├── evaluator.py                    # Official 3D Euclidean MAE/RMSE evaluator & Excel bug repair
│   ├── visualizer.py                   # 3D trajectory comparison visualizer
│   └── mock_detections.py              # Synthetic detection generator for offline testing
├── results/                            # Generated run artifacts
│   ├── detections/                     # detections_2d_yolo_real.csv
│   ├── trajectories/                   # output_trajectory_yolo.csv
│   ├── metrics/                        # metrics_summary_yolo.json
│   ├── plots/                          # trajectory_comparison.png
│   └── videos/                         # Annotated per-camera videos & 6-panel dashboard
├── tests/                              # Automated test suite (pytest)
│   ├── test_camera_model.py            # Intrinsics & projection tests
│   ├── test_triangulation.py           # Multi-view SVD triangulation tests
│   ├── test_kalman.py                  # 3D Kalman filter & occlusion coasting tests
│   ├── test_cpp_core.py                # C++20 peleng_core numerical verification
│   ├── test_evaluator.py               # MAE calculation & row-402 Excel glitch repair test
│   ├── test_occlusion_detector.py      # Bounding box edge & aspect ratio analysis tests
│   ├── test_yolo_dataset.py            # Dataset split & YOLO annotation format tests
│   ├── test_yolo_pipeline.py           # YOLO training configuration & reproducibility tests
│   └── test_rtdetr_pipeline.py         # RT-DETR training configuration & reproducibility tests
├── .agents/skills/                     # Antigravity / Gemini skills
│   ├── peleng-3d-tracking/             # Full pipeline execution & tuning runbook
│   ├── peleng-cpp-core/                # C++20 extension build & benchmark runbook
│   └── peleng-yolo-vision/             # YOLO fine-tuning & video dashboard runbook
├── main.py                             # Primary pipeline orchestrator CLI
├── CMakeLists.txt                      # C++20 CMake configuration
├── setup.py                            # Python C-extension build setup
├── README.md                           # Quickstart & user documentation
└── хакатон/                            # Backward-compatibility alias folder for legacy runs
```

## 2. Key Entry Points & CLI Commands
- **Main 3D Tracking Pipeline**:
  `python main.py`
  *(Defaults: `--calib-dir data/raw/hackathon --gt data/ground_truth/object_transform_SM_AA_track.csv`)*
- **Run YOLO Inference over 5 Cameras**:
  `python scripts/inference/process_videos_yolo.py`
- **Train YOLOv8x Model**:
  `python scripts/training/train_yolo.py`
- **Train RT-DETR Transformer**:
  `python scripts/training/train_rtdetr.py`
- **Run Automated Test Suite**:
  `python -m pytest tests/ -v`
