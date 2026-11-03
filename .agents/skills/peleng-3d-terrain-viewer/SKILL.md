---
name: peleng-3d-terrain-viewer
description: >-
  Use this skill when reconstructing 3D outdoor terrain and point clouds from calibrated
  multi-camera videos using Depth-Anything-V2-Metric-Outdoor-Small-hf, aligning monocular
  depth maps with 3D triangulated trajectory anchors via median scale-shift calibration,
  projecting camera RGB textures onto elevation meshes, and rendering large-scale (800m+)
  interactive 3D trajectories and camera frustums in Three.js.
---

# Peleng 3D Outdoor Terrain Reconstruction & Three.js WebGL Visualization

Distilled engineering patterns from NotebookLM (`ForAgents` & `Hackathon Пеленг 2.0` outdoor metric depth benchmarks) and Context7 (`huggingface/transformers`, `mrdoob/three.js`).

## 1. Outdoor Metric Depth Model Selection (`Depth-Anything-V2-Metric-Outdoor`)
Based on the outdoor natural terrain benchmark (*arXiv:2510.04723*, *CVPR 2025 MoGe*, *Metric3D v2*):
- **Primary Model**: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` (HuggingFace `transformers`).
  - **Parameters / Size**: `24.8M` parameters (~`99 MB` in `fp32`, ~`50 MB` in `fp16`).
  - **Outdoor Benchmark Accuracy**: Best overall outdoor MAE (`0.454 m`) and fastest inference (`~0.22 s` per frame on GPU/Apple Silicon MPS), outperforming `ML Depth Pro` (`MAE 1.127 m`, `0.65 s`) and `ZoeDepth` (`MAE 3.087 m`).
  - **Higher-Capacity Option**: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf` (`97.5M` params, `~390 MB`) or `Large-hf` (`335M` params, `~1.34 GB`).

### Standard HuggingFace Inference Snippet
```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

def predict_metric_depth(pil_image: Image.Image) -> "np.ndarray":
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    post = processor.post_process_depth_estimation(
        outputs, target_sizes=[(pil_image.height, pil_image.width)]
    )
    return post[0]["predicted_depth"].detach().cpu().numpy()
```

## 2. Scale & Shift Alignment with Calibrated Cameras ($P = K[R|t]$)
Even metric monocular depth models experience focal-length scale ambiguity across heterogeneous lenses (`Cam 1: 35mm`, `Cam 2: 60mm`, `Cam 3: 35mm`, `Cam 4: 50mm`, `Cam 5: 20mm`).
To lock the neural depth map $D_{\text{net}}(u, v)$ onto the exact Unreal Engine 5 coordinate system (`1 unit = 1 cm`):
1. **Anchor Projection**: Project known 3D triangulated trajectory points $\{P_{\text{tri}}^{(i)}\}$ into camera $c$ using `CameraModel.project_point(P_tri)` to obtain ground-truth camera depths $z_{\text{true}}^{(i)}$ at pixels $(u_i, v_i)$.
2. **Median Patch Extraction**: Extract $z_{\text{net}}^{(i)}$ as the **median** within a $5 \times 5$ pixel window around $(u_i, v_i)$ (proven in outdoor benchmarks to suppress vegetation/shadow outliers).
3. **Robust Affine Calibration**:
   $$z_{\text{calibrated}}(u, v) = s \cdot D_{\text{net}}(u, v) + t_z$$
   Solve $(s, t_z)$ via RANSAC / Huber linear regression (or median ratio $s = \text{median}(z_{\text{true}}^{(i)} / z_{\text{net}}^{(i)})$ when anchor depth span is narrow).
4. **World Unprojection**:
   For each subsampled pixel $(u, v)$ with RGB color $(R, G, B)$:
   $$\mathbf{X}_w(u, v) = \mathbf{O}_{\text{cam}} + z_{\text{calibrated}}(u, v) \cdot \mathbf{d}_{\text{ray}}(u, v)$$
5. **Continuous Valley Surface Fusion (`scipy.interpolate.Rbf`)**:
   Fuse sparse/dense 3D terrain points from the trajectory road (`Z in [300, 718] cm`), depth-unprojected valley points, and mountain camera bases (`Cam 2: Z=7912 cm`, `Cam 4: Z=7147 cm`) into a continuous elevation grid $Z(X, Y)$ with projected RGB textures from overview cameras (`Cam 2` and `Cam 4`).

## 3. Three.js Large-Scale (800m+) Rendering Best Practices
1. **Coordinate System Handedness Conversion**:
   - Unreal Engine 5 is **Left-Handed Z-Up**: $(X_{\text{ue}}\text{ Forward}, Y_{\text{ue}}\text{ Right}, Z_{\text{ue}}\text{ Up})$.
   - Three.js is **Right-Handed Y-Up**: map coordinates as:
     $$X_{\text{three}} = X_{\text{ue}}, \quad Y_{\text{three}} = Z_{\text{ue}}, \quad Z_{\text{three}} = -Y_{\text{ue}}$$
     (or set `scene.up.set(0, 0, 1)` with scaled meters `1 unit = 1 meter = 100 UE cm`).
2. **Scale Normalization & Floating Origin (Anti-Z-Fighting)**:
   - Convert UE5 centimeters (`-45,784 .. +33,282`) to **meters** (`divide by 100.0`) so coordinates lie within `[-460, +340] m`.
   - Set perspective camera `near = 0.5`, `far = 5000.0`, and `WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true })` to completely eliminate Z-fighting between the road trajectory and terrain mesh.
3. **Single Draw-Call Point Cloud & Terrain Mesh**:
   - Use `THREE.BufferGeometry` with `Float32BufferAttribute` for `position` and `color` (`vertexColors: true`).
   - Keep total draw calls `< 50` for 60 FPS performance on Apple M1 integrated GPUs.
