---
name: peleng-yolo-vision
description: >-
  Use this skill when training, fine-tuning, evaluating, or running inference with
  YOLO object detection models (YOLOv8x / YOLOv11) on Peleng multi-camera video streams,
  or generating synchronized multi-view annotated videos and 2D detection CSVs.
---

# Peleng YOLO Vision Pipeline

Workflow and operational runbook for fine-tuning and running YOLO object detection on the 5-camera Peleng tracking dataset.

## Target Object & Specifications
- **Class:** Single class `truck` (ID: `0`)
- **Camera Streams:** 5 synchronized 1920x1080 @ 30 FPS cameras (Cameras 1-4 static, Camera 5 rotating).
- **Default Architecture:** `yolov8x.pt` (68.2M params, 258 GFLOPs) for maximum precision in long-distance small-object detection.

## 1. Dataset Configuration (`data.yaml`)
Dataset structure follows standard Ultralytics format:
```yaml
path: C:/projects/Peleng/data/datasets/yolo_v3
train: train/images
val: valid/images
nc: 1
names: ['truck']
```
*Note:* Always keep `path` absolute or relative to project root to avoid working-directory resolution issues on Windows.

## 2. Model Training & Fine-Tuning
Execute fine-tuning with early stopping, AMP, and weight preservation:
```bash
python scripts/training/train_yolo.py --model weights/pretrained/yolov8x.pt --epochs 30 --batch 8 --imgsz 512 --device 0
```

### Key Hyperparameters:
- `imgsz=512`: Matches Roboflow preprocessing resolution.
- `batch=8`: Optimal for 8GB VRAM (prevents OOM on RTX 4070 Laptop).
- `patience=15`: Early stopping patience.
- `seed=42`: Explicit seed for NumPy, PyTorch, and Python random.

Outputs:
- Best weights: `weights/finetuned/peleng_yolov8x_best.pt`
- Training metrics JSON: `results/metrics/metrics_summary_yolo.json`

## 3. Video Processing & Multi-View Dashboard Generation
Generate individual annotated camera videos, the synchronized 6-panel multi-camera grid dashboard, and the 2D detections CSV in one pass:
```bash
python scripts/inference/process_videos_yolo.py --weights weights/finetuned/peleng_yolov8x_best.pt --videos_dir data/raw/hackathon --output_dir results/videos --conf 0.25 --device 0
```

Outputs:
- `results/videos/camera_{1..5}_detected.mp4`: Individual Full-HD streams with bounding boxes and HUD.
- `results/videos/multiview_dashboard_yolo.mp4`: Synchronized 3x2 grid with real-time telemetry card.
- `results/detections/detections_2d_yolo_real.csv`: Format `Frame;Cam_ID;u;v;w;h;conf`.

## 4. Programmatic Inference via `src.yolo_detector`
```python
from src.yolo_detector import YOLOv8Detector

detector = YOLOv8Detector("weights/peleng_yolov8x_best.pt", device="0", conf_threshold=0.25)
detections = detector.detect_frame("debug_cam1_f0_projected.jpg")
# Returns: [{'u': 282.68, 'v': 657.08, 'w': 298.3, 'h': 121.15, 'conf': 0.818}]
```
