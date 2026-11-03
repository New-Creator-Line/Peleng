# Comprehensive Plan: Preparing a Multi-Camera Video Dataset for Object Detection

This document outlines the end-to-end pipeline for converting raw MP4 video streams from multiple cameras into a robust, noise-augmented training dataset for object detection models (e.g., YOLO / RT-DETR).

---

## Phase 1: Frame Extraction & Filtering
The goal is to extract relevant frames while discarding redundant data and ensuring proper tracking of source cameras.

1. **Downsampling Frames:**
   - Extract frames from raw MP4 files at **1 to 3 FPS** using `ffmpeg` or `OpenCV`. Avoid 30 FPS extraction to prevent massive redundancy and overfitting.
   - Name files systematically using camera ID and timestamp/index (e.g., `cam1_frame_00001.jpg`).
2. **Deduplication & Background Sampling:**
   - Use perceptual hashing (`imagehash` library) or Structural Similarity Index (SSIM) to remove static, unchanging frames.
   - Retain **5–10% empty background frames** (scenes without target objects) to reduce False Positives during inference.

---

## Phase 2: Annotation Workflow
Accurate bounding boxes are critical before applying any noise or augmentations.

1. **Semi-Automated Pre-Labeling:**
   - Pass extracted frames through a foundational model (e.g., Grounding DINO, YOLOv8/v11 pre-trained model, or SAM 2) to generate initial draft annotations.
2. **Manual Verification & Correction:**
   - Import frames and draft boxes into an annotation tool (**CVAT** or **Label Studio**).
   - Carefully verify, adjust, or draw missing bounding boxes.
3. **Export Annotations:**
   - Export annotations into the target training format (e.g., YOLO `.txt` format or COCO JSON).

---

## Phase 3: Data Splitting (Preventing Data Leakage)
A strict splitting strategy ensures your validation metrics reflect real-world performance.

1. **Split by Video, Not by Frame:**
   - Split your dataset into `train` / `val` / `test` (e.g., **75% / 15% / 10%**) based on **entire video segments or distinct camera hours**, *never* by random frame shuffling.
   - *Why?* Adjacent frames share nearly identical contexts; random shuffling causes data leakage, leading to artificially inflated validation scores.

---

## Phase 4: Noise Injection & Augmentation Strategy
Simulating real-world camera imperfections (low light, motion blur, compression artifacts).

1. **Keep Validation & Test Sets Clean:**
   - Never apply noise or severe augmentations to validation and test datasets. They must reflect true production conditions.
2. **Apply On-The-Fly Augmentations (Recommended):**
   - Integrate **Albumentations** directly into your training data loader so augmentations are dynamically randomized on every epoch, saving disk space.
3. **Core Camera Augmentation Stack:**
   - `GaussNoise` or `ISONoise`: Simulates sensor grain in low-light conditions.
   - `MotionBlur` / `GaussianBlur`: Accounts for fast-moving objects or out-of-focus camera lenses.
   - `ImageCompression` (JPEG/WebP/Bitrate reduction): Replicates compression artifacts common in RTSP streams or saved MP4 footage.
   - `RandomBrightnessContrast` & `HueSaturationValue`: Handles fluctuating ambient lighting conditions throughout the day.

---

## Phase 5: Final Assembly & Validation
1. **Directory Structure Organization:**
   Set up your dataset directory following standard formats:
   ```text
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```
2. **Configuration File (`dataset.yaml`):**
   - Define absolute or relative paths to train/val images, number of classes, and class names.
3. **Visual Quality Check:**
   - Run a quick debugging script to plot 20–30 random training images with their corresponding bounding boxes and applied augmentations to verify that bounding box coordinates remain correctly aligned.