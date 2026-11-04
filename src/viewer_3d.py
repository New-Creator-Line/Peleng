"""
Interactive Three.js 3D Terrain, Camera Frustum & Trajectory Viewer Generator.

Consumes a `Scene3DPayload` dictionary (defined in `.memory_bank/contracts/scene_3d_contract.md`)
and exports a self-contained, single-file HTML5 + Three.js (`three@0.160.0`) WebGL application.
"""

import base64
import json
import math
import os
from typing import Any, Dict

import numpy as np


def _load_cargo_truck_fbx_b64() -> str:
    """Loads Peleng/assets/CargoTruck.fbx (or /Users/test/Desktop/CargoTruck.fbx) as a Base64 string."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_paths = [
        os.path.join(repo_root, "assets", "CargoTruck.fbx"),
        "/Users/test/Desktop/CargoTruck.fbx",
    ]
    for path in candidate_paths:
        try:
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("ascii")
        except Exception:
            continue
    return ""


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NumPy types and non-finite floats into JSON-safe Python primitives."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _normalize_payload(scene_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures all required top-level keys exist with safe defaults before rendering."""
    if not isinstance(scene_payload, dict):
        raise TypeError("scene_payload must be a dictionary conforming to Scene3DPayload.")

    meta = dict(scene_payload.get("meta") or {})
    frames = scene_payload.get("frames") if scene_payload.get("frames") is not None else []
    meta.setdefault("coord_system", "UE5_cm")
    meta.setdefault("num_frames", len(frames))
    meta.setdefault("fps", 30.0)
    meta.setdefault("has_ground_truth", False)
    meta.setdefault("depth_model_used", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    meta.setdefault("origin_center_cm", [0.0, 0.0, 0.0])
    meta.setdefault(
        "bounds_cm",
        {
            "x_min": -48000.0,
            "x_max": 36000.0,
            "y_min": -26000.0,
            "y_max": 18000.0,
            "z_min": 250.0,
            "z_max": 8000.0,
        },
    )

    metrics = dict(scene_payload.get("metrics") or {})
    terrain_mesh = dict(scene_payload.get("terrain_mesh") or {})
    terrain_mesh.setdefault("rows", 0)
    terrain_mesh.setdefault("cols", 0)
    terrain_mesh.setdefault("x_grid_cm", [])
    terrain_mesh.setdefault("y_grid_cm", [])
    terrain_mesh.setdefault("z_matrix_cm", [])
    terrain_mesh.setdefault("rgb_matrix", [])

    point_cloud = dict(scene_payload.get("point_cloud") or {})
    point_cloud.setdefault("points_cm", [])
    point_cloud.setdefault("colors_rgb", [])

    cameras = dict(scene_payload.get("cameras") or {})

    normalized = {
        "meta": meta,
        "metrics": metrics,
        "terrain_mesh": terrain_mesh,
        "point_cloud": point_cloud,
        "cameras": cameras,
        "frames": frames,
    }

    if "tree_cloud" in scene_payload and scene_payload["tree_cloud"] is not None:
        tree_cloud = dict(scene_payload["tree_cloud"])
        tree_cloud.setdefault("points_cm", [])
        tree_cloud.setdefault("colors_rgb", [])
        normalized["tree_cloud"] = tree_cloud

    if "landmarks" in scene_payload and scene_payload["landmarks"] is not None:
        normalized["landmarks"] = list(scene_payload["landmarks"])

    if "terrain_methods" in scene_payload and scene_payload["terrain_methods"] is not None:
        normalized["terrain_methods"] = dict(scene_payload["terrain_methods"])

    if "models" in scene_payload and scene_payload["models"] is not None:
        normalized["models"] = dict(scene_payload["models"])

    if "active_model" in scene_payload and scene_payload["active_model"] is not None:
        normalized["active_model"] = str(scene_payload["active_model"])

    return _sanitize_for_json(normalized)


def export_3d_viewer_html(
    scene_payload: Dict[str, Any],
    output_html_path: str = "results/peleng_3d_viewer.html",
) -> str:
    """
    Renders a self-contained, interactive Three.js WebGL HTML application
    from a Scene3DPayload dictionary and writes it to output_html_path.
    Returns output_html_path.
    """
    normalized_payload = _normalize_payload(scene_payload)
    payload_json = json.dumps(normalized_payload, ensure_ascii=False, separators=(",", ":"))
    # Prevent accidental script tag termination inside embedded JSON
    payload_json = payload_json.replace("</script>", "<\\/script>")

    fbx_b64 = _load_cargo_truck_fbx_b64()
    html_content = _HTML_TEMPLATE.replace("__SCENE_PAYLOAD_JSON__", payload_json)
    html_content = html_content.replace("__CARGO_TRUCK_FBX_B64__", fbx_b64)

    out_dir = os.path.dirname(output_html_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ПЕЛЕНГ 2.0 — 3D Tactical Terrain & Multi-Camera Trajectory Viewer</title>
  <script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
    }
  }
  </script>
  <style>
    :root {
      --bg-deep: #0b0f19;
      --panel-bg: rgba(13, 19, 33, 0.82);
      --panel-border: rgba(56, 189, 248, 0.24);
      --accent-cyan: #00f5d4;
      --accent-blue: #38bdf8;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --text-main: #e2e8f0;
      --text-muted: #94a3b8;
      --font-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }

    html, body {
      width: 100%;
      height: 100%;
      overflow: hidden;
      background-color: var(--bg-deep);
      color: var(--text-main);
      font-family: var(--font-sans);
    }

    #canvas-container {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
    }

    /* 2D Camera Viewport Overlay (HUD) - Exact 16:9 Frame */
    #cam-hud-overlay {
      position: absolute;
      pointer-events: none;
      z-index: 2;
      display: none;
      overflow: hidden;
      border: 1px solid rgba(0, 245, 212, 0.4);
      box-shadow: 0 0 35px rgba(0, 0, 0, 0.85);
    }
    #cam-hud-img {
      width: 100%;
      height: 100%;
      object-fit: fill;
      display: block;
      opacity: 0.6;
      transition: opacity 0.1s;
      transform-origin: center center;
    }

    #labels-container {
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 2;
      overflow: hidden;
    }

    .cam-badge-3d {
      position: absolute;
      transform: translate(-50%, -120%);
      padding: 4px 8px;
      border-radius: 6px;
      background: rgba(11, 15, 25, 0.86);
      border: 1px solid rgba(56, 189, 248, 0.5);
      color: #f8fafc;
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
      pointer-events: auto;
      cursor: pointer;
      transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }

    .cam-badge-3d:hover {
      border-color: var(--accent-cyan);
      background: rgba(15, 23, 42, 0.95);
    }

    .cam-badge-3d.active-ray {
      border-color: var(--accent-cyan);
      box-shadow: 0 0 12px rgba(0, 245, 212, 0.45);
      color: var(--accent-cyan);
    }

    .cam-badge-3d.ptz-cam {
      border-color: #a855f7;
    }

    /* Glassmorphism HUD */
    .hud-panel {
      background: var(--panel-bg);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.55);
      z-index: 10;
    }

    .brand-subtitle {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
    }

    .pill-badge {
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(30, 41, 59, 0.78);
      border: 1px solid rgba(148, 163, 184, 0.25);
      color: var(--text-main);
      display: inline-block;
    }

    .pill-badge.mode-gt {
      border-color: rgba(0, 245, 212, 0.45);
      color: var(--accent-cyan);
    }

    .pill-badge.mode-blind {
      border-color: rgba(245, 158, 11, 0.55);
      color: var(--accent-amber);
    }

    /* Collapsible Panel Header & Floating Reopen Pills */
    .panel-header-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(56, 189, 248, 0.2);
    }

    .panel-header-title {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #f8fafc;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .panel-collapse-btn {
      background: rgba(30, 41, 59, 0.75);
      border: 1px solid rgba(148, 163, 184, 0.25);
      color: var(--text-muted);
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 3px 8px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .panel-collapse-btn:hover {
      color: var(--accent-cyan);
      border-color: var(--accent-cyan);
      background: rgba(0, 245, 212, 0.12);
    }

    .hud-reopen-pill {
      position: absolute;
      top: 16px;
      z-index: 12;
      background: var(--panel-bg);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid var(--panel-border);
      color: #f8fafc;
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 600;
      padding: 8px 14px;
      border-radius: 999px;
      cursor: pointer;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
      transition: all 0.2s ease;
      display: none;
      align-items: center;
      gap: 6px;
    }

    .hud-reopen-pill:hover {
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
      background: rgba(15, 23, 42, 0.94);
    }

    .hud-reopen-pill.left-pill {
      left: 16px;
    }

    .hud-reopen-pill.right-pill {
      right: 16px;
    }

    .hud-reopen-pill.visible {
      display: inline-flex;
    }

    /* Left Controls Panel */
    #hud-left {
      position: absolute;
      top: 16px;
      left: 16px;
      width: 280px;
      max-height: calc(100vh - 104px);
      overflow-y: auto;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      transition: transform 0.24s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.2s ease;
    }

    #hud-left.collapsed {
      transform: translateX(calc(-100% - 24px));
      opacity: 0;
      pointer-events: none;
    }

    .section-title {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent-blue);
      margin-bottom: 8px;
      border-bottom: 1px solid rgba(56, 189, 248, 0.16);
      padding-bottom: 4px;
    }

    .preset-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }

    .hud-btn {
      background: rgba(30, 41, 59, 0.75);
      border: 1px solid rgba(148, 163, 184, 0.22);
      color: var(--text-main);
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 7px 8px;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.15s ease;
      text-align: center;
    }

    .hud-btn:hover {
      border-color: var(--accent-blue);
      background: rgba(56, 189, 248, 0.14);
    }

    .hud-btn.active {
      background: rgba(0, 245, 212, 0.18);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
      font-weight: 600;
    }

    .layer-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .layer-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-main);
      cursor: pointer;
    }

    .layer-item input[type="checkbox"] {
      accent-color: var(--accent-cyan);
      width: 15px;
      height: 15px;
      cursor: pointer;
    }

    .slider-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 11px;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }

    .slider-row input[type="range"] {
      width: 100%;
      accent-color: var(--accent-blue);
      cursor: pointer;
    }

    /* Picture-in-Picture Camera Reference Preview */
    #pip-card {
      display: none;
      flex-direction: column;
      gap: 6px;
      background: rgba(11, 15, 25, 0.9);
      border: 1px solid rgba(56, 189, 248, 0.35);
      border-radius: 8px;
      padding: 8px;
    }

    #pip-card.visible {
      display: flex;
    }

    #pip-canvas-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #050811;
      border-radius: 5px;
      overflow: hidden;
    }

    #pip-img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    #pip-crosshair {
      position: absolute;
      width: 14px;
      height: 14px;
      border: 2px solid var(--accent-cyan);
      border-radius: 50%;
      transform: translate(-50%, -50%);
      box-shadow: 0 0 8px var(--accent-cyan);
      pointer-events: none;
      display: none;
    }

    /* Right Telemetry Panel */
    #hud-right {
      position: absolute;
      top: 16px;
      right: 16px;
      width: 330px;
      min-width: 280px;
      max-width: 700px;
      max-height: calc(100vh - 104px);
      overflow-y: auto;
      overflow-x: hidden;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      transition: transform 0.24s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.2s ease;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    }

    .panel-resizer-left {
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 8px;
      cursor: ew-resize;
      background: transparent;
      transition: background 0.15s ease;
      z-index: 100;
    }
    .panel-resizer-left:hover, .panel-resizer-left.resizing {
      background: rgba(0, 245, 212, 0.6);
      box-shadow: 0 0 8px rgba(0, 245, 212, 0.8);
    }

    #hud-right.collapsed {
      transform: translateX(calc(100% + 24px));
      opacity: 0;
      pointer-events: none;
    }

    .telemetry-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .metric-box {
      background: rgba(15, 23, 42, 0.72);
      border: 1px solid rgba(148, 163, 184, 0.16);
      border-radius: 8px;
      padding: 8px 10px;
    }

    .metric-box.full-width {
      grid-column: 1 / -1;
    }

    .metric-label {
      font-size: 10px;
      font-family: var(--font-mono);
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 3px;
    }

    .metric-value {
      font-size: 13px;
      font-family: var(--font-mono);
      font-weight: 700;
      color: #f8fafc;
    }

    .metric-value.cyan {
      color: var(--accent-cyan);
    }

    .metric-value.amber {
      color: var(--accent-amber);
    }

    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }

    .legend-swatch {
      width: 12px;
      height: 4px;
      border-radius: 2px;
    }

    /* Bottom Timeline Scrubber Dock */
    #hud-bottom {
      position: absolute;
      bottom: 16px;
      left: 16px;
      right: 16px;
      height: 68px;
      padding: 0 18px;
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .transport-controls {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .scrubber-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 5px;
    }

    .scrubber-header {
      display: flex;
      justify-content: space-between;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
    }

    #timeline-slider {
      width: 100%;
      accent-color: var(--accent-cyan);
      cursor: pointer;
      height: 6px;
    }

    .speed-select {
      background: rgba(30, 41, 59, 0.85);
      border: 1px solid rgba(148, 163, 184, 0.28);
      color: var(--text-main);
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 6px 8px;
      border-radius: 7px;
      cursor: pointer;
    }

    #fallback-banner {
      display: none;
      position: absolute;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      padding: 10px 18px;
      background: rgba(239, 68, 68, 0.9);
      color: #fff;
      font-family: var(--font-mono);
      font-size: 12px;
      border-radius: 8px;
      z-index: 100;
    }
  </style>
</head>
<body>
  <div id="canvas-container"></div>
  <!-- 2D Synchronized Camera Viewport HUD (16:9 overlay) -->
  <div id="cam-hud-overlay">
    <img id="cam-hud-img" src="" alt="Camera POV Frame" />
  </div>
  <div id="labels-container"></div>
  <div id="fallback-banner"></div>

  <!-- Compact Floating Reopen Pill Buttons (shown when side panels are collapsed) -->
  <button id="btn-open-left" data-toggle="btn-toggle-left" class="hud-reopen-pill left-pill" title="Показать панель камер и слоев">
    ☰ Камеры и Слои
  </button>
  <button id="btn-open-right" data-toggle="btn-toggle-right" class="hud-reopen-pill right-pill visible" title="Показать панель телеметрии">
    📊 Телеметрия
  </button>

  <!-- Left Panel: Camera View Presets & Layer Visibility -->
  <aside id="hud-left" class="hud-panel">
    <div class="panel-header-bar">
      <span class="panel-header-title">☰ Камеры и Слои</span>
      <button id="btn-collapse-left" class="panel-collapse-btn" title="Скрыть панель">[ — Скрыть ]</button>
    </div>

    <div style="margin-bottom: 12px; padding: 10px; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(0, 245, 212, 0.35); border-radius: 8px;">
      <div class="section-title" style="margin-bottom: 6px; color: #00f5d4; font-size: 11px;">🤖 Выбор Модели и Алгоритма</div>
      <select id="select-active-model" class="speed-select" style="width: 100%; font-size: 11px; padding: 6px 8px; font-weight: 600;">
        <option value="cabin" selected>🏆 Поколение 4: Cabin SOTA + Z-Fix (1.06 м)</option>
        <option value="ukf">⭐ Поколение 3: YOLO26x-seg + 10D UKF (3.51 м)</option>
        <option value="rtdetr">🚀 Поколение 2: RT-DETR + Filter (6.14 м)</option>
        <option value="yolo_linear">📦 Поколение 1: YOLOv8x + Linear (6.30 м)</option>
      </select>
      <div id="model-desc-badge" style="font-size: 10px; color: #94a3b8; margin-top: 6px; line-height: 1.35;">
        <span style="color:#22c55e;font-weight:700;">Cabin SOTA + Z-Fix</span>: Fine-tuned Cabin Detector, NVIDIA Anisotropic Covariance, Dynamic Altitude UKF. MAE: <b>1.06 м (106.2 см)</b>!
      </div>
    </div>

    <div>
      <div class="section-title">Camera View Presets</div>
      <div class="preset-grid" id="camera-presets">
        <button class="hud-btn active" data-view="orbit">Free Orbit</button>
        <button class="hud-btn" data-view="top">Top-Down Valley</button>
        <button class="hud-btn" data-view="chase" style="grid-column: 1 / -1;">Chase Truck (SM_AA_track)</button>
        <button class="hud-btn" data-view="cam_1">Cam 1 (35mm)</button>
        <button class="hud-btn" data-view="cam_2">Cam 2 (60mm)</button>
        <button class="hud-btn" data-view="cam_3">Cam 3 (35mm)</button>
        <button class="hud-btn" data-view="cam_4">Cam 4 (50mm)</button>
        <button class="hud-btn" data-view="cam_5" style="grid-column: 1 / -1;">Cam 5 PTZ (20mm Dynamic)</button>
      </div>
      <!-- Photo Overlay Opacity Slider -->
      <div style="margin-top: 8px; padding: 8px 10px; background: rgba(15, 23, 42, 0.6); border-radius: 6px; border: 1px solid rgba(0, 245, 212, 0.25);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
          <span style="font-weight:600; color:#fff; font-size:11px;">Наложение фото камеры:</span>
          <span id="photo-opacity-val" style="color:#00f5d4; font-weight:700; font-size:11px;">60%</span>
        </div>
        <input type="range" class="slider" id="slider-photo-opacity" min="0" max="100" step="5" value="60" style="width: 100%;" />
      </div>
    </div>

    <div id="pip-card">
      <div class="section-title" id="pip-title" style="margin-bottom: 2px;">Camera Reference Frame</div>
      <div id="pip-canvas-wrap">
        <img id="pip-img" alt="Camera Reference Thumbnail" />
        <div id="pip-crosshair"></div>
      </div>
      <div class="brand-subtitle" id="pip-meta">FOV: --</div>
    </div>

    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div class="section-title" style="margin-bottom:0;">3D Scene Layers</div>
        <button id="btn-reset-settings" style="background:none;border:none;color:#94a3b8;font-size:11px;cursor:pointer;text-decoration:underline;padding:0;" title="Сбросить все слои и настройки к исходным">Сбросить</button>
      </div>
      <div class="layer-list">
        <label class="layer-item">
          <span>Photometric Terrain Mesh</span>
          <input type="checkbox" id="toggle-terrain" checked />
        </label>
        <label class="layer-item">
          <span>Tactical Wireframe Grid</span>
          <input type="checkbox" id="toggle-wireframe" checked />
        </label>
        <label class="layer-item">
          <span>Neural Depth Point Cloud</span>
          <input type="checkbox" id="toggle-pointcloud" checked />
        </label>
        <div class="slider-row">
          <div style="display:flex;justify-content:space-between;">
            <span>Point Cloud Size</span>
            <span id="pt-size-label">2.6 px</span>
          </div>
          <input type="range" id="slider-pt-size" min="0.5" max="8.0" step="0.1" value="2.6" />
        </div>
        <label class="layer-item">
          <span>3D Roadway</span>
          <input type="checkbox" id="toggle-road" checked />
        </label>
        <label class="layer-item" id="row-toggle-landmarks">
          <span style="color:#10b981;font-weight:600;">🎯 Эталонные деревья (GCP)</span>
          <input type="checkbox" id="toggle-landmarks" checked />
        </label>
        <label class="layer-item">
          <span>Camera Frustums &amp; Badges</span>
          <input type="checkbox" id="toggle-frustums" checked />
        </label>
        <label class="layer-item">
          <span>Live Triangulation Rays</span>
          <input type="checkbox" id="toggle-rays" checked />
        </label>
        <label class="layer-item">
          <span>Predicted Trajectory</span>
          <input type="checkbox" id="toggle-est-traj" checked />
        </label>
        <label class="layer-item" id="row-toggle-gt">
          <span>Ground Truth Trajectory</span>
          <input type="checkbox" id="toggle-gt-traj" checked />
        </label>
        <div class="slider-row">
          <div style="display:flex;justify-content:space-between;">
            <span>Толщина линий пути</span>
            <span id="traj-width-label">0.60 m</span>
          </div>
          <input type="range" id="slider-traj-width" min="0.15" max="2.50" step="0.05" value="0.60" />
        </div>
        <div class="slider-row">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span>Цвета линий пути</span>
          </div>
          <select class="speed-select" id="select-traj-style" style="width:100%;">
            <option value="contrast" selected>Контраст: Чёрно-красный / Ярко-оранжевый (GT)</option>
            <option value="neon">Неон: Бирюзовый / Ярко-оранжевый (GT)</option>
            <option value="vivid">Яркий: Красный / Изумрудный (GT)</option>
          </select>
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Legend</div>
      <div class="layer-list" style="gap: 5px;">
        <div class="legend-row">
          <span class="legend-swatch" id="swatch-est" style="background:#111827;border:1.5px solid #d90429;"></span>
          <span>Predicted 3D Path &amp; SM_AA_track</span>
        </div>
        <div class="legend-row" id="legend-gt-row">
          <span class="legend-swatch" id="swatch-gt" style="background:#ff9f00;"></span>
          <span>Ground Truth 3D Path &amp; Truck (UE5)</span>
        </div>
        <div class="legend-row">
          <span class="legend-swatch" style="background:#38bdf8;"></span>
          <span>Active Camera Line-of-Sight Ray</span>
        </div>
      </div>
    </div>
  </aside>

  <!-- Right Panel: Live Telemetry Readout -->
  <aside id="hud-right" class="hud-panel collapsed">
    <div id="hud-right-resizer" class="panel-resizer-left" title="Потяните мышью для изменения ширины панели"></div>
    <div class="panel-header-bar">
      <span class="panel-header-title">📊 Телеметрия</span>
      <button id="btn-collapse-right" class="panel-collapse-btn" title="Скрыть панель">[ Скрыть — ]</button>
    </div>

    <div>
      <div class="section-title">Live Target Telemetry (UE5_cm)</div>
      <div class="telemetry-grid">
        <div class="metric-box">
          <div class="metric-label">Frame / Time</div>
          <div class="metric-value" id="tel-frame">0 / 0 (0.00s)</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Heading / Speed</div>
          <div class="metric-value" id="tel-heading">0.0°</div>
        </div>
        <div class="metric-box full-width">
          <div class="metric-label">Est Position (Loc_X, Loc_Y, Loc_Z) [cm]</div>
          <div class="metric-value cyan" id="tel-est-pos">--</div>
        </div>
        <div class="metric-box full-width" id="box-gt-pos">
          <div class="metric-label">Ground Truth (GT_X, GT_Y, GT_Z) [cm]</div>
          <div class="metric-value amber" id="tel-gt-pos">--</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Instant 3D Error</div>
          <div class="metric-value amber" id="tel-err">--</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Active Cameras</div>
          <div class="metric-value cyan" id="tel-active-cams">0</div>
        </div>
        <div class="metric-box full-width">
          <div class="metric-label">Active Ray Detections (u, v, conf)</div>
          <div class="metric-value" id="tel-ray-details" style="font-size:11px;font-weight:500;line-height:1.4;">None</div>
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Точность модели и текущая разница с эталоном</div>
      <div class="telemetry-grid">
        <!-- 1. Итоговый константный MAE выбранной модели -->
        <div class="metric-box full-width" style="border: 1px solid rgba(0, 245, 212, 0.4); background: rgba(0, 245, 212, 0.08); padding: 8px 10px;">
          <div class="metric-label" style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#00f5d4; font-weight:700;">★ ИТОГОВЫЙ MAE_3D (КОНСТАНТА МОДЕЛИ)</span>
            <span id="active-model-summary-badge" style="font-size:10px; color:#38bdf8; font-weight:600;">Cabin Detector</span>
          </div>
          <div class="metric-value cyan" id="sum-mae3d" style="font-size:16px; margin: 3px 0;">--</div>
          <div style="font-size:10px; color:#94a3b8;">Погрешность текущей модели по всем 1001 кадрам (фиксированная константа)</div>
        </div>

        <!-- 2. Динамическая ошибка в текущем кадре -->
        <div class="metric-box">
          <div class="metric-label">ТЕКУЩАЯ РАЗНИЦА (КАДР #<span id="dyn-frame-idx">0</span>)</div>
          <div class="metric-value amber" id="dyn-instant-err" style="font-size:14px; font-weight:700;">--</div>
        </div>

        <!-- 3. Динамический накопительный MAE до текущего кадра -->
        <div class="metric-box">
          <div class="metric-label">НАКОПЛЕННЫЙ MAE (ДО КАДРА)</div>
          <div class="metric-value cyan" id="dyn-running-mae" style="font-size:14px; font-weight:700;">--</div>
        </div>

        <!-- 4. Динамическая разница по осям в кадре -->
        <div class="metric-box full-width">
          <div class="metric-label">ТЕКУЩАЯ РАЗНИЦА ПО ОСЯМ (ΔX / ΔY / ΔZ) [см]</div>
          <div class="metric-value" id="dyn-axis-diff" style="font-size:12px; font-family:monospace;">ΔX: -- | ΔY: -- | ΔZ: --</div>
        </div>

        <!-- 5. Статические константы RMSE и Axis MAE -->
        <div class="metric-box">
          <div class="metric-label">ИТОГОВЫЙ RMSE_3D (КОНСТАНТА)</div>
          <div class="metric-value" id="sum-rmse3d" style="font-size:13px; color:#a5f3fc;">--</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">ИТОГОВЫЙ AXIS MAE (X / Y / Z)</div>
          <div class="metric-value" id="sum-axes" style="font-size:11px; color:#e2e8f0;">--</div>
        </div>
      </div>
    </div>
  </aside>

  <!-- Bottom Timeline Scrubber -->
  <footer id="hud-bottom" class="hud-panel">
    <div class="transport-controls">
      <button class="hud-btn" id="btn-prev" title="Previous Frame (Left Arrow)">&lt;</button>
      <button class="hud-btn active" id="btn-play" style="min-width: 74px;" title="Play / Pause (Space)">PAUSE</button>
      <button class="hud-btn" id="btn-next" title="Next Frame (Right Arrow)">&gt;</button>
    </div>

    <div class="scrubber-container">
      <div class="scrubber-header">
        <span id="scrubber-frame-label">FRAME 0</span>
        <span id="scrubber-coords-label">X: 0.0m | Y: 0.0m | Z: 0.0m</span>
      </div>
      <input type="range" id="timeline-slider" min="0" max="0" step="1" value="0" />
    </div>

    <select class="speed-select" id="select-speed" title="Playback Speed">
      <option value="0.25">0.25x</option>
      <option value="0.5">0.5x</option>
      <option value="1" selected>1.0x</option>
      <option value="2">2.0x</option>
    </select>

    <button class="hud-btn" id="btn-toggle-ui" title="Скрыть или показать боковые панели (клавиша H)">
      Скрыть панели (H)
    </button>
  </footer>

  <!-- Embedded Scene3DPayload Data & CargoTruck.fbx Asset -->
  <script>
    window.SCENE_DATA = __SCENE_PAYLOAD_JSON__;
  </script>
  <script>
    window.CARGO_TRUCK_FBX_B64 = "__CARGO_TRUCK_FBX_B64__";
  </script>

  <!-- Three.js Application Logic -->
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

    const SCENE_DATA = window.SCENE_DATA || {};
    const meta = SCENE_DATA.meta || {};
    const modelsData = SCENE_DATA.models || {};
    const defaultModelKey = SCENE_DATA.active_model || 'cabin';

    const initialChosen = modelsData[defaultModelKey] || (Object.keys(modelsData).length > 0 ? Object.values(modelsData)[0] : null);
    let metrics = (initialChosen && initialChosen.metrics) ? initialChosen.metrics : (SCENE_DATA.metrics || {});
    let frames = (initialChosen && initialChosen.frames && initialChosen.frames.length > 0)
      ? initialChosen.frames
      : (SCENE_DATA.frames || []);

    // Running MAE cache for currently active model frames
    let runningMaeArray = [];
    function precomputeRunningMaeForActiveFrames() {
      if (!frames || frames.length === 0) {
        runningMaeArray = [];
        return;
      }
      runningMaeArray = new Float32Array(frames.length);
      let sumErr = 0.0;
      let count = 0;
      for (let i = 0; i < frames.length; i++) {
        const f = frames[i];
        let err = 0.0;
        if (typeof f.error_cm === 'number' && !isNaN(f.error_cm)) {
          err = f.error_cm;
        } else if (f.est_pos_cm && f.gt_pos_cm) {
          const dx = f.est_pos_cm[0] - f.gt_pos_cm[0];
          const dy = f.est_pos_cm[1] - f.gt_pos_cm[1];
          const dz = f.est_pos_cm[2] - f.gt_pos_cm[2];
          err = Math.hypot(dx, dy, dz);
        }
        sumErr += err;
        count++;
        runningMaeArray[i] = sumErr / count;
      }
    }
    precomputeRunningMaeForActiveFrames();
    const camerasData = SCENE_DATA.cameras || {};
    const hasGT = Boolean(meta.has_ground_truth);
    const originCenter = meta.origin_center_cm || [0.0, 0.0, 0.0];
    const cx = Number(originCenter[0] || 0.0);
    const cy = Number(originCenter[1] || 0.0);
    const cz = Number(originCenter[2] || 0.0);

    /**
     * Exact coordinate conversion from Unreal Engine 5 cm (Left-Handed: +X Forward, +Y Right, +Z Up)
     * centered at meta.origin_center_cm [cx, cy, cz] into Three.js Right-Handed meters (1 unit = 1m):
     *   x3 = (x_ue - cx) / 100.0
     *   y3 = (z_ue - cz) / 100.0
     *   z3 = (y_ue - cy) / 100.0
     */
    function ueToThree(posCm) {
      const x_ue = Number(posCm[0] || 0.0);
      const y_ue = Number(posCm[1] || 0.0);
      const z_ue = Number(posCm[2] || 0.0);
      const x3 = (x_ue - cx) / 100.0;
      const y3 = (z_ue - cz) / 100.0;
      const z3 = (y_ue - cy) / 100.0;
      return new THREE.Vector3(x3, y3, z3);
    }

    function ueDirToThree(dirVec) {
      const dx = Number(dirVec[0] || 0.0);
      const dy = Number(dirVec[1] || 0.0);
      const dz = Number(dirVec[2] || 0.0);
      return new THREE.Vector3(dx, dz, dy).normalize();
    }

    // Mode initialization & defensive element guards
    if (!hasGT) {
      const toggleGtEl = document.getElementById('toggle-gt-traj');
      if (toggleGtEl) {
        toggleGtEl.checked = false;
        toggleGtEl.disabled = true;
      }
      const rowToggleGt = document.getElementById('row-toggle-gt');
      if (rowToggleGt) rowToggleGt.style.opacity = '0.45';
      const legendGtRow = document.getElementById('legend-gt-row');
      if (legendGtRow) legendGtRow.style.display = 'none';
      const telGtPos = document.getElementById('tel-gt-pos');
      if (telGtPos) telGtPos.textContent = 'N/A (Blind Test Mode)';
      const telErr = document.getElementById('tel-err');
      if (telErr) telErr.textContent = 'N/A';
    }

    const sumMaeEl = document.getElementById('sum-mae3d');
    if (sumMaeEl) {
      if (typeof metrics.mae_3d === 'number') {
        const maeCm = metrics.mae_3d.toFixed(1);
        const maeM = (metrics.mae_3d / 100.0).toFixed(2);
        sumMaeEl.textContent = `${maeCm} cm (${maeM} m)`;
      } else {
        sumMaeEl.textContent = 'N/A';
      }
    }

    const sumRmseEl = document.getElementById('sum-rmse3d');
    if (sumRmseEl) {
      if (typeof metrics.rmse_3d === 'number') {
        const rmseCm = metrics.rmse_3d.toFixed(1);
        const rmseM = (metrics.rmse_3d / 100.0).toFixed(2);
        sumRmseEl.textContent = `${rmseCm} cm (${rmseM} m)`;
      } else {
        sumRmseEl.textContent = 'N/A';
      }
    }

    const sumAxesEl = document.getElementById('sum-axes');
    if (sumAxesEl) {
      if (typeof metrics.mae_x === 'number') {
        sumAxesEl.textContent =
          `${metrics.mae_x.toFixed(1)} / ${(metrics.mae_y || 0).toFixed(1)} / ${(metrics.mae_z || 0).toFixed(1)} cm`;
      } else {
        sumAxesEl.textContent = 'N/A (Blind Test)';
      }
    }

    const activeKeyInit = SCENE_DATA.active_model || 'cabin';
    const activeModelInit = (SCENE_DATA.models && SCENE_DATA.models[activeKeyInit]) || null;
    const badgeInit = document.getElementById('active-model-summary-badge');
    if (badgeInit && activeModelInit) {
      badgeInit.textContent = activeModelInit.short_name || activeModelInit.name || activeKeyInit;
    }

    // Collapsible Side Panels Logic
    const hudLeft = document.getElementById('hud-left');
    const hudRight = document.getElementById('hud-right');
    const btnCollapseLeft = document.getElementById('btn-collapse-left');
    const btnOpenLeft = document.getElementById('btn-open-left');
    const btnCollapseRight = document.getElementById('btn-collapse-right');
    const btnOpenRight = document.getElementById('btn-open-right');
    const btnToggleUi = document.getElementById('btn-toggle-ui');

    function syncPanelToggleButtons() {
      const leftCollapsed = hudLeft.classList.contains('collapsed');
      const rightCollapsed = hudRight.classList.contains('collapsed');
      btnOpenLeft.classList.toggle('visible', leftCollapsed);
      btnOpenRight.classList.toggle('visible', rightCollapsed);
      if (btnToggleUi) {
        const bothHidden = leftCollapsed && rightCollapsed;
        btnToggleUi.textContent = bothHidden ? 'Показать панели (H)' : 'Скрыть панели (H)';
        btnToggleUi.classList.toggle('active', bothHidden);
      }
    }

    function setLeftPanelCollapsed(collapsed) {
      hudLeft.classList.toggle('collapsed', Boolean(collapsed));
      syncPanelToggleButtons();
      if (typeof saveViewerSettings === 'function') saveViewerSettings();
    }

    function setRightPanelCollapsed(collapsed) {
      hudRight.classList.toggle('collapsed', Boolean(collapsed));
      syncPanelToggleButtons();
      if (typeof saveViewerSettings === 'function') saveViewerSettings();
    }

    function toggleBothSidePanels() {
      const anyVisible =
        !hudLeft.classList.contains('collapsed') || !hudRight.classList.contains('collapsed');
      setLeftPanelCollapsed(anyVisible);
      setRightPanelCollapsed(anyVisible);
    }

    if (btnCollapseLeft) btnCollapseLeft.addEventListener('click', () => setLeftPanelCollapsed(true));
    if (btnOpenLeft) btnOpenLeft.addEventListener('click', () => setLeftPanelCollapsed(false));
    if (btnCollapseRight) btnCollapseRight.addEventListener('click', () => setRightPanelCollapsed(true));
    if (btnOpenRight) btnOpenRight.addEventListener('click', () => setRightPanelCollapsed(false));
    if (btnToggleUi) btnToggleUi.addEventListener('click', toggleBothSidePanels);

    // Setup Scene, Camera, and Renderer with logarithmicDepthBuffer
    const container = document.getElementById('canvas-container');
    const labelsContainer = document.getElementById('labels-container');

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f19);
    scene.fog = new THREE.FogExp2(0x0b0f19, 0.00032);

    const camera = new THREE.PerspectiveCamera(
      48,
      window.innerWidth / window.innerHeight,
      0.5,
      6000.0
    );
    camera.position.set(-180, 140, 260);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      logarithmicDepthBuffer: true,
      powerPreference: 'high-performance'
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.18;
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxDistance = 2800;
    controls.minDistance = 4;
    controls.target.set(0, 5, 0);

    // Enhanced Lighting for Crisp Terrain & Point Cloud Cohesion
    const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x1e293b, 1.35);
    hemiLight.position.set(0, 500, 0);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xfffbeb, 1.55);
    dirLight.position.set(220, 420, 180);
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0x38bdf8, 0.55);
    fillLight.position.set(-250, 180, -200);
    scene.add(fillLight);

    // 1. Build Indexed Photometric Terrain Mesh + Tactical Wireframe Overlay
    const terrainGroup = new THREE.Group();
    scene.add(terrainGroup);
    let terrainMeshObj = null;
    let wireframeMeshObj = null;

    function buildTerrain() {
      const tm = SCENE_DATA.terrain_mesh || {};
      const rows = Number(tm.rows || 0);
      const cols = Number(tm.cols || 0);
      const xGrid = tm.x_grid_cm || [];
      const yGrid = tm.y_grid_cm || [];
      const zMatrix = tm.z_matrix_cm || [];
      const rgbMatrix = tm.rgb_matrix || [];

      if (rows < 2 || cols < 2 || xGrid.length < cols || yGrid.length < rows) {
        return;
      }

      const numVerts = rows * cols;
      const positions = new Float32Array(numVerts * 3);
      const colors = new Float32Array(numVerts * 3);

      let ptr = 0;
      for (let r = 0; r < rows; r++) {
        const y_ue = yGrid[r];
        const zRow = zMatrix[r] || [];
        const rgbRow = rgbMatrix[r] || [];
        for (let c = 0; c < cols; c++) {
          const x_ue = xGrid[c];
          const z_ue = zRow[c] !== undefined ? zRow[c] : 0.0;
          const v3 = ueToThree([x_ue, y_ue, z_ue]);

          positions[ptr] = v3.x;
          positions[ptr + 1] = v3.y;
          positions[ptr + 2] = v3.z;

          const rgb = rgbRow[c] || [90, 110, 82];
          colors[ptr] = (rgb[0] || 90) / 255.0;
          colors[ptr + 1] = (rgb[1] || 110) / 255.0;
          colors[ptr + 2] = (rgb[2] || 82) / 255.0;
          ptr += 3;
        }
      }

      const indices = [];
      for (let r = 0; r < rows - 1; r++) {
        for (let c = 0; c < cols - 1; c++) {
          const i00 = r * cols + c;
          const i10 = r * cols + (c + 1);
          const i01 = (r + 1) * cols + c;
          const i11 = (r + 1) * cols + (c + 1);
          indices.push(i00, i10, i01);
          indices.push(i10, i11, i01);
        }
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setIndex(indices);
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.computeVertexNormals();

      const material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.78,
        metalness: 0.05,
        side: THREE.DoubleSide
      });

      terrainMeshObj = new THREE.Mesh(geometry, material);
      terrainGroup.add(terrainMeshObj);

      const wireMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        wireframe: true,
        transparent: true,
        opacity: 0.12,
        depthWrite: false
      });
      wireframeMeshObj = new THREE.Mesh(geometry, wireMat);
      wireframeMeshObj.position.y += 0.06;
      terrainGroup.add(wireframeMeshObj);
    }

    buildTerrain();

    // 2. Build Calibrated Neural Depth Point Cloud
    let pointCloudObj = null;
    let pointCloudMaterial = null;

    function buildPointCloud() {
      const pc = SCENE_DATA.point_cloud || {};
      const pts = pc.points_cm || [];
      const cols = pc.colors_rgb || [];
      if (!pts.length) return;

      const n = pts.length;
      const positions = new Float32Array(n * 3);
      const colors = new Float32Array(n * 3);

      for (let i = 0; i < n; i++) {
        const v3 = ueToThree(pts[i]);
        positions[i * 3] = v3.x;
        positions[i * 3 + 1] = v3.y + 0.08;
        positions[i * 3 + 2] = v3.z;

        const c = cols[i] || [120, 145, 110];
        colors[i * 3] = (c[0] || 120) / 255.0;
        colors[i * 3 + 1] = (c[1] || 145) / 255.0;
        colors[i * 3 + 2] = (c[2] || 110) / 255.0;
      }

      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      pointCloudMaterial = new THREE.PointsMaterial({
        size: 2.6,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.94
      });

      pointCloudObj = new THREE.Points(geom, pointCloudMaterial);
      scene.add(pointCloudObj);
    }

    buildPointCloud();



    // 3. Build 5 Calibrated Camera Rigs & Frustums (with Dynamic PTZ Support for Cam 5)
    const camerasGroup = new THREE.Group();
    scene.add(camerasGroup);
    const cameraRigMap = {};

    function computeFrustumLinePositions(originThree, cornersCm) {
      const c = (cornersCm || []).map(pt => ueToThree(pt));
      while (c.length < 4) {
        c.push(originThree.clone());
      }
      // 8 segments: 4 origin->corner edges + 4 far-plane perimeter edges
      const segs = [
        originThree, c[0],
        originThree, c[1],
        originThree, c[2],
        originThree, c[3],
        c[0], c[1],
        c[1], c[2],
        c[2], c[3],
        c[3], c[0]
      ];
      const arr = new Float32Array(segs.length * 3);
      for (let i = 0; i < segs.length; i++) {
        arr[i * 3] = segs[i].x;
        arr[i * 3 + 1] = segs[i].y;
        arr[i * 3 + 2] = segs[i].z;
      }
      return arr;
    }

    function sampleTerrainYThree(x3, z3) {
      const tm = SCENE_DATA.terrain_mesh || {};
      const rows = Number(tm.rows || 0);
      const cols = Number(tm.cols || 0);
      const xGrid = tm.x_grid_cm || [];
      const yGrid = tm.y_grid_cm || [];
      const zMatrix = tm.z_matrix_cm || [];
      if (rows < 2 || cols < 2 || !xGrid.length || !yGrid.length) return 0.0;

      const x_ue = x3 * 100.0 + cx;
      const y_ue = z3 * 100.0 + cy;
      const xMin = xGrid[0], xMax = xGrid[cols - 1];
      const yMin = yGrid[0], yMax = yGrid[rows - 1];
      const u = Math.max(0, Math.min(1, (x_ue - xMin) / Math.max(1.0, xMax - xMin))) * (cols - 1);
      const v = Math.max(0, Math.min(1, (y_ue - yMin) / Math.max(1.0, yMax - yMin))) * (rows - 1);
      const c0 = Math.floor(u), c1 = Math.min(cols - 1, c0 + 1);
      const r0 = Math.floor(v), r1 = Math.min(rows - 1, r0 + 1);
      const tx = u - c0, ty = v - r0;
      const z00 = (zMatrix[r0] && zMatrix[r0][c0] !== undefined) ? zMatrix[r0][c0] : cz;
      const z10 = (zMatrix[r0] && zMatrix[r0][c1] !== undefined) ? zMatrix[r0][c1] : cz;
      const z01 = (zMatrix[r1] && zMatrix[r1][c0] !== undefined) ? zMatrix[r1][c0] : cz;
      const z11 = (zMatrix[r1] && zMatrix[r1][c1] !== undefined) ? zMatrix[r1][c1] : cz;
      const z_ue = (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11;
      return (z_ue - cz) / 100.0;
    }

    function buildCameras() {
      const pylonMat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        roughness: 0.6,
        metalness: 0.65
      });
      Object.keys(camerasData).forEach(camKey => {
        const camInfo = camerasData[camKey];
        const posThree = ueToThree(camInfo.pos_cm || [0, 0, 0]);
        const isDynamic = Boolean(camInfo.is_dynamic || camKey === '5');
        const accentColor = isDynamic ? 0xa855f7 : 0x38bdf8;

        const rigGroup = new THREE.Group();
        camerasGroup.add(rigGroup);

        // 3D Vertical Pylon / Support Tower grounding camera to terrain
        const groundY = sampleTerrainYThree(posThree.x, posThree.z);
        const towerHeight = Math.max(8.0, posThree.y - groundY + 4.0);
        const towerGeom = new THREE.CylinderGeometry(0.35, 0.65, towerHeight, 8);
        const towerMesh = new THREE.Mesh(towerGeom, pylonMat);
        towerMesh.position.set(posThree.x, posThree.y - towerHeight * 0.5, posThree.z);
        rigGroup.add(towerMesh);

        // Directional 3D Camera Housing Box + Glowing Lens Marker
        const housingGeom = new THREE.BoxGeometry(2.6, 1.5, 1.6);
        const housingMat = new THREE.MeshStandardMaterial({
          color: 0x0f172a,
          emissive: accentColor,
          emissiveIntensity: 0.45,
          roughness: 0.3,
          metalness: 0.7
        });
        const markerMesh = new THREE.Mesh(housingGeom, housingMat);
        markerMesh.position.copy(posThree);
        const fwdDir = ueDirToThree(camInfo.forward || [1, 0, 0]);
        markerMesh.lookAt(posThree.clone().add(fwdDir));
        rigGroup.add(markerMesh);

        // Frustum wireframe lines
        const linePositions = computeFrustumLinePositions(posThree, camInfo.frustum_corners_cm || []);
        const frustumGeom = new THREE.BufferGeometry();
        frustumGeom.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
        const frustumMat = new THREE.LineBasicMaterial({
          color: accentColor,
          transparent: true,
          opacity: 0.65
        });
        const frustumLines = new THREE.LineSegments(frustumGeom, frustumMat);
        rigGroup.add(frustumLines);

        // Floating HTML Badge
        const badge = document.createElement('div');
        badge.className = `cam-badge-3d ${isDynamic ? 'ptz-cam' : ''}`;
        const focal = camInfo.focal_length_mm ? `${Math.round(camInfo.focal_length_mm)}mm` : '';
        badge.textContent = isDynamic
          ? `Cam ${camKey} PTZ (${focal})`
          : `Cam ${camKey} (${focal})`;
        badge.addEventListener('click', () => setCameraPreset(`cam_${camKey}`));
        labelsContainer.appendChild(badge);

        cameraRigMap[String(camKey)] = {
          camKey: String(camKey),
          info: camInfo,
          posThree,
          markerMesh,
          frustumGeom,
          frustumMat,
          badge,
          currentForward: camInfo.forward || [1, 0, 0],
          currentUp: camInfo.up || [0, 0, 1]
        };
      });
    }

    buildCameras();

    // 4. Build Detailed 3D CargoTruck (CargoTruck.fbx via FBXLoader.parse) + GT Truck + Trajectory Ribbons
    function base64ToArrayBuffer(base64) {
      const bin = atob(base64);
      const len = bin.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
      return bytes.buffer;
    }

    function parseCargoTruckFbx() {
      const b64 = window.CARGO_TRUCK_FBX_B64 || '';
      if (!b64 || b64.length < 64) return null;
      const buf = base64ToArrayBuffer(b64);
      const loader = new FBXLoader();
      const fbxGroup = loader.parse(buf, '');
      // CargoTruck.fbx is in cm (+Z forward, +Y up); scale 0.014 (~9.4m long) and rotate +90deg around Y to align +Z with +X forward
      fbxGroup.scale.set(0.014, 0.014, 0.014);
      fbxGroup.rotation.set(0, Math.PI / 2, 0);
      return fbxGroup;
    }

    function fixFbxTruckMaterial(mat) {
      if (!mat) return mat;
      const cloned = mat.clone();
      const mName = String(cloned.name || '').toLowerCase();
      // Reset Blender 5.0 default white EmissiveColor [1,1,1] so diffuse colors render accurately
      if (cloned.emissive) {
        if (mName.includes('headlight')) {
          cloned.emissive.set(0xf59e0b);
          if ('emissiveIntensity' in cloned) cloned.emissiveIntensity = 0.8;
        } else if (mName.includes('backlight')) {
          cloned.emissive.set(0xef4444);
          if ('emissiveIntensity' in cloned) cloned.emissiveIntensity = 0.8;
        } else {
          cloned.emissive.set(0x000000);
          if ('emissiveIntensity' in cloned) cloned.emissiveIntensity = 0.0;
        }
      }
      cloned.side = THREE.DoubleSide;
      return cloned;
    }

    function createAATruckMesh() {
      const group = new THREE.Group();
      let loadedFbx = false;
      try {
        const fbxGroup = parseCargoTruckFbx();
        if (fbxGroup) {
          fbxGroup.traverse(child => {
            if (child.isMesh) {
              if (Array.isArray(child.material)) {
                child.material = child.material.map(fixFbxTruckMaterial);
              } else {
                child.material = fixFbxTruckMaterial(child.material);
              }
            }
          });
          group.add(fbxGroup);
          loadedFbx = true;
        }
      } catch (err) {
        console.warn('FBXLoader fallback to procedural truck mesh:', err);
      }

      if (!loadedFbx) {
        const hullMat = new THREE.MeshStandardMaterial({
          color: 0x1e293b,
          roughness: 0.45,
          metalness: 0.55
        });
        const accentMat = new THREE.MeshStandardMaterial({
          color: 0x00f5d4,
          emissive: 0x00f5d4,
          emissiveIntensity: 0.35,
          roughness: 0.3
        });
        const chassis = new THREE.Mesh(new THREE.BoxGeometry(9.4, 0.9, 2.8), hullMat);
        chassis.position.set(0, 0.95, 0);
        group.add(chassis);
        const cab = new THREE.Mesh(new THREE.BoxGeometry(2.8, 1.85, 2.7), hullMat);
        cab.position.set(3.2, 2.1, 0);
        group.add(cab);
        const visor = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.55, 2.4), accentMat);
        visor.position.set(4.6, 2.45, 0);
        group.add(visor);
      }

      // Tactical neon outline box (SM_AA_track / CargoTruck)
      const boxEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(9.6, 3.8, 3.2));
      const boxLines = new THREE.LineSegments(
        boxEdges,
        new THREE.LineBasicMaterial({ color: 0x00f5d4, transparent: true, opacity: 0.45 })
      );
      boxLines.position.set(0, 1.9, 0);
      group.add(boxLines);

      return group;
    }

    const truckMesh = createAATruckMesh();
    scene.add(truckMesh);

    // Bold 3D Ground Truth Truck Model (Authentic CargoTruck.fbx in Vivid Golden-Orange + Bright Outline)
    function createGTTruckMesh() {
      const group = new THREE.Group();
      const gtArmorMat = new THREE.MeshStandardMaterial({
        color: 0xff9f00,
        emissive: 0xff6d00,
        emissiveIntensity: 0.65,
        roughness: 0.25,
        metalness: 0.4,
        transparent: true,
        opacity: 0.85,
        side: THREE.DoubleSide
      });

      let loadedFbx = false;
      try {
        const fbxGroup = parseCargoTruckFbx();
        if (fbxGroup) {
          fbxGroup.traverse(child => {
            if (child.isMesh) {
              child.material = gtArmorMat;
            }
          });
          group.add(fbxGroup);
          loadedFbx = true;
        }
      } catch (err) {
        console.warn('FBXLoader GT fallback to procedural truck mesh:', err);
      }

      if (!loadedFbx) {
        const gtChassis = new THREE.Mesh(new THREE.BoxGeometry(9.4, 0.9, 2.8), gtArmorMat);
        gtChassis.position.set(0, 0.95, 0);
        group.add(gtChassis);
        const gtCab = new THREE.Mesh(new THREE.BoxGeometry(2.8, 1.85, 2.7), gtArmorMat);
        gtCab.position.set(3.2, 2.1, 0);
        group.add(gtCab);
      }

      const gtEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(9.8, 3.9, 3.3));
      const gtOutline = new THREE.LineSegments(
        gtEdges,
        new THREE.LineBasicMaterial({ color: 0xffe600, transparent: true, opacity: 0.95 })
      );
      gtOutline.position.set(0, 1.95, 0);
      group.add(gtOutline);
      return group;
    }

    const gtGhostGroup = new THREE.Group();
    gtGhostGroup.add(createGTTruckMesh());
    gtGhostGroup.visible = hasGT;
    scene.add(gtGhostGroup);

    // 3D Volumetric Tube Trajectories with Adjustable Thickness & Color Palette
    let estTrajectoryLine = null;
    let gtTrajectoryLine = null;
    let estCurve = null;
    let gtCurve = null;
    let roadBaseCurve = null;
    let currentTrajRadius = 0.60;
    let currentTrajStyle = 'contrast';

    const TRAJ_STYLES = {
      contrast: {
        estColor: 0x111827,
        estEmissive: 0xd90429,
        estEmissiveIntensity: 0.45,
        gtColor: 0xff9f00,
        gtEmissive: 0xff6d00,
        gtEmissiveIntensity: 0.75,
        swatchEstBg: '#111827',
        swatchEstBorder: '1.5px solid #d90429',
        swatchGtBg: '#ff9f00'
      },
      neon: {
        estColor: 0x00f5d4,
        estEmissive: 0x00b4d8,
        estEmissiveIntensity: 0.55,
        gtColor: 0xff9f00,
        gtEmissive: 0xff6d00,
        gtEmissiveIntensity: 0.75,
        swatchEstBg: '#00f5d4',
        swatchEstBorder: 'none',
        swatchGtBg: '#ff9f00'
      },
      vivid: {
        estColor: 0xef4444,
        estEmissive: 0xb91c1c,
        estEmissiveIntensity: 0.55,
        gtColor: 0x10b981,
        gtEmissive: 0x059669,
        gtEmissiveIntensity: 0.65,
        swatchEstBg: '#ef4444',
        swatchEstBorder: 'none',
        swatchGtBg: '#10b981'
      }
    };

    let activeModelColor = null;

    function deduplicatePoints(pts, minDist = 0.05) {
      const out = [];
      for (let i = 0; i < pts.length; i++) {
        if (out.length === 0 || pts[i].distanceTo(out[out.length - 1]) >= minDist) {
          out.push(pts[i]);
        }
      }
      if (out.length === 1 && pts.length >= 2) {
        out.push(pts[pts.length - 1].clone().add(new THREE.Vector3(0.1, 0, 0.1)));
      }
      return out;
    }

    function rebuildTrajectories(radiusM, styleKey) {
      const style = TRAJ_STYLES[styleKey] || TRAJ_STYLES.contrast;
      const estVis = document.getElementById('toggle-est-traj')
        ? document.getElementById('toggle-est-traj').checked
        : true;
      const gtVis = document.getElementById('toggle-gt-traj')
        ? document.getElementById('toggle-gt-traj').checked
        : true;

      if (estTrajectoryLine) {
        scene.remove(estTrajectoryLine);
        if (estTrajectoryLine.geometry) estTrajectoryLine.geometry.dispose();
        if (estTrajectoryLine.material) estTrajectoryLine.material.dispose();
        estTrajectoryLine = null;
      }
      if (gtTrajectoryLine) {
        scene.remove(gtTrajectoryLine);
        if (gtTrajectoryLine.geometry) gtTrajectoryLine.geometry.dispose();
        if (gtTrajectoryLine.material) gtTrajectoryLine.material.dispose();
        gtTrajectoryLine = null;
      }

      if (estCurve) {
        const tubularSegs = Math.min(360, Math.max(32, estCurve.points.length * 2));
        const geom = new THREE.TubeGeometry(estCurve, tubularSegs, radiusM, 8, false);
        const col = (activeModelColor !== null) ? activeModelColor : style.estColor;
        const mat = new THREE.MeshStandardMaterial({
          color: col,
          emissive: col,
          emissiveIntensity: 0.45,
          roughness: 0.25,
          metalness: 0.35
        });
        estTrajectoryLine = new THREE.Mesh(geom, mat);
        estTrajectoryLine.visible = estVis;
        scene.add(estTrajectoryLine);
      }

      if (hasGT && gtCurve) {
        const tubularSegs = Math.min(360, Math.max(32, gtCurve.points.length * 2));
        const geom = new THREE.TubeGeometry(gtCurve, tubularSegs, radiusM * 0.92, 8, false);
        const mat = new THREE.MeshStandardMaterial({
          color: style.gtColor,
          emissive: style.gtEmissive,
          emissiveIntensity: style.gtEmissiveIntensity,
          roughness: 0.2,
          metalness: 0.3,
          transparent: true,
          opacity: 0.92
        });
        gtTrajectoryLine = new THREE.Mesh(geom, mat);
        gtTrajectoryLine.visible = gtVis;
        scene.add(gtTrajectoryLine);
      }

      const swEst = document.getElementById('swatch-est');
      if (swEst) {
        swEst.style.background = style.swatchEstBg;
        swEst.style.border = style.swatchEstBorder;
      }
      const swGt = document.getElementById('swatch-gt');
      if (swGt) {
        swGt.style.background = style.swatchGtBg;
      }
    }

    function buildTrajectories() {
      if (!frames.length) return;

      const rawEst = [];
      const rawGt = [];
      const rawRoad = [];

      frames.forEach(f => {
        if (f.est_pos_cm) {
          const p = ueToThree(f.est_pos_cm);
          rawEst.push(new THREE.Vector3(p.x, p.y - 2.80 + 0.35, p.z));
          if (!hasGT || !f.gt_pos_cm) {
            rawRoad.push(new THREE.Vector3(p.x, p.y - 2.80, p.z));
          }
        }
        if (hasGT && f.gt_pos_cm) {
          const g = ueToThree(f.gt_pos_cm);
          rawGt.push(new THREE.Vector3(g.x, g.y - 2.80 + 0.35, g.z));
          rawRoad.push(new THREE.Vector3(g.x, g.y - 2.80, g.z));
        }
      });

      const cleanEst = deduplicatePoints(rawEst);
      const cleanGt = deduplicatePoints(rawGt);
      const cleanRoad = deduplicatePoints(rawRoad, 0.25);

      if (cleanEst.length >= 2) {
        estCurve = new THREE.CatmullRomCurve3(cleanEst);
      }
      if (hasGT && cleanGt.length >= 2) {
        gtCurve = new THREE.CatmullRomCurve3(cleanGt);
      }
      if (cleanRoad.length >= 2) {
        roadBaseCurve = new THREE.CatmullRomCurve3(cleanRoad);
      }

      rebuildTrajectories(currentTrajRadius, currentTrajStyle);
    }

    buildTrajectories();

    // 4b. Build 3D Roadway Ribbon & Centerline Markings
    const roadGroup = new THREE.Group();
    scene.add(roadGroup);

    function createRibbonGeometry(curve, samples, widthM, yLiftM) {
      const positions = new Float32Array((samples + 1) * 2 * 3);
      const indices = [];
      for (let i = 0; i <= samples; i++) {
        const t = i / samples;
        const pt = curve.getPoint(t);
        const tan = curve.getTangent(t);
        const right = new THREE.Vector3(-tan.z, 0, tan.x);
        if (right.lengthSq() < 1e-6) right.set(0, 0, 1);
        else right.normalize();

        const halfW = widthM * 0.5;
        const lx = pt.x - right.x * halfW;
        const ly = pt.y + yLiftM;
        const lz = pt.z - right.z * halfW;

        const rx = pt.x + right.x * halfW;
        const ry = pt.y + yLiftM;
        const rz = pt.z + right.z * halfW;

        const base = i * 6;
        positions[base] = lx;
        positions[base + 1] = ly;
        positions[base + 2] = lz;
        positions[base + 3] = rx;
        positions[base + 4] = ry;
        positions[base + 5] = rz;

        if (i < samples) {
          const v0 = i * 2;
          const v1 = v0 + 1;
          const v2 = v0 + 2;
          const v3 = v0 + 3;
          indices.push(v0, v2, v1);
          indices.push(v1, v2, v3);
        }
      }
      const geom = new THREE.BufferGeometry();
      geom.setIndex(indices);
      geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geom.computeVertexNormals();
      return geom;
    }

    function rebuildRoadwayRibbons() {
      while (roadGroup.children.length > 0) {
        const obj = roadGroup.children[0];
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
          else obj.material.dispose();
        }
        roadGroup.remove(obj);
      }
      buildRoadAndScenery();
    }

    function buildRoadAndScenery() {
      const activeCurve = roadBaseCurve || gtCurve || estCurve;
      if (!activeCurve) return;

      const numSamples = 220;

      // Gravel shoulder bed (11.5m wide, lifted +0.10m)
      const shoulderGeom = createRibbonGeometry(activeCurve, numSamples, 11.5, 0.10);
      const shoulderMat = new THREE.MeshStandardMaterial({
        color: 0x5c5346,
        roughness: 0.92,
        metalness: 0.05,
        side: THREE.DoubleSide
      });
      roadGroup.add(new THREE.Mesh(shoulderGeom, shoulderMat));

      // Dark asphalt roadbed (8.0m wide, lifted +0.18m)
      const asphaltGeom = createRibbonGeometry(activeCurve, numSamples, 8.0, 0.18);
      const asphaltMat = new THREE.MeshStandardMaterial({
        color: 0x232730,
        roughness: 0.78,
        metalness: 0.12,
        side: THREE.DoubleSide
      });
      roadGroup.add(new THREE.Mesh(asphaltGeom, asphaltMat));

      // Dashed bright yellow-white centerline & edge lines
      const dashPositions = [];
      const edgePositions = [];
      for (let i = 0; i < numSamples; i++) {
        const p0 = activeCurve.getPoint(i / numSamples);
        const p1 = activeCurve.getPoint((i + 1) / numSamples);
        const tan = activeCurve.getTangent(i / numSamples);
        const right = new THREE.Vector3(-tan.z, 0, tan.x);
        if (right.lengthSq() < 1e-6) right.set(0, 0, 1);
        else right.normalize();

        if (i % 2 === 0) {
          dashPositions.push(p0.x, p0.y + 0.25, p0.z, p1.x, p1.y + 0.25, p1.z);
        }
        const edgeOffset = 3.65;
        edgePositions.push(
          p0.x - right.x * edgeOffset, p0.y + 0.22, p0.z - right.z * edgeOffset,
          p1.x - right.x * edgeOffset, p1.y + 0.22, p1.z - right.z * edgeOffset,
          p0.x + right.x * edgeOffset, p0.y + 0.22, p0.z + right.z * edgeOffset,
          p1.x + right.x * edgeOffset, p1.y + 0.22, p1.z + right.z * edgeOffset
        );
      }

      const dashGeom = new THREE.BufferGeometry();
      dashGeom.setAttribute('position', new THREE.Float32BufferAttribute(dashPositions, 3));
      roadGroup.add(
        new THREE.LineSegments(
          dashGeom,
          new THREE.LineBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.85 })
        )
      );

      const edgeGeom = new THREE.BufferGeometry();
      edgeGeom.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
      roadGroup.add(
        new THREE.LineSegments(
          edgeGeom,
          new THREE.LineBasicMaterial({ color: 0xe2e8f0, transparent: true, opacity: 0.55 })
        )
      );
    }

    buildRoadAndScenery();

    // 4b. Ground Truth Landmarks (GCP Vertical Beacons)
    const landmarksGroup = new THREE.Group();
    scene.add(landmarksGroup);

    function buildLandmarks() {
      const landmarksList = SCENE_DATA.landmarks || [];
      if (!landmarksList.length) return;

      const beaconHeight = 6.0;
      const cylinderGeom = new THREE.CylinderGeometry(0.15, 0.2, beaconHeight, 16);
      const cylinderMat = new THREE.MeshStandardMaterial({
        color: 0x10b981,
        emissive: 0x059669,
        emissiveIntensity: 0.5,
        roughness: 0.3,
        metalness: 0.4
      });

      const sphereGeom = new THREE.SphereGeometry(0.55, 20, 20);
      const sphereMat = new THREE.MeshStandardMaterial({
        color: 0xfbbf24,
        emissive: 0xd97706,
        emissiveIntensity: 0.85,
        roughness: 0.25,
        metalness: 0.5
      });

      const ringGeom = new THREE.RingGeometry(0.8, 1.4, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.75
      });

      landmarksList.forEach(lm => {
        const posThree = ueToThree(lm.position_3d_cm || [0, 0, 0]);
        const lmGroup = new THREE.Group();

        // 1. Concentric ground target ring flat on the terrain floor
        const ringMesh = new THREE.Mesh(ringGeom, ringMat);
        ringMesh.rotation.x = -Math.PI / 2;
        ringMesh.position.set(posThree.x, posThree.y + 0.05, posThree.z);
        lmGroup.add(ringMesh);

        // 2. Vertical beacon needle/cylinder: base sitting exactly on the ground posThree
        const needleMesh = new THREE.Mesh(cylinderGeom, cylinderMat);
        needleMesh.position.set(posThree.x, posThree.y + beaconHeight / 2, posThree.z);
        lmGroup.add(needleMesh);

        // 3. Glowing amber/yellow sphere at the top tip (posThree.y + 6.0m)
        const sphereMesh = new THREE.Mesh(sphereGeom, sphereMat);
        sphereMesh.position.set(posThree.x, posThree.y + beaconHeight, posThree.z);
        lmGroup.add(sphereMesh);

        landmarksGroup.add(lmGroup);
      });
    }
    buildLandmarks();

    // 5. Dynamic Active Triangulation Rays (Pre-allocated LineSegments Pool)
    const MAX_RAYS = 16;
    const rayPositions = new Float32Array(MAX_RAYS * 2 * 3);
    const rayGeometry = new THREE.BufferGeometry();
    rayGeometry.setAttribute('position', new THREE.BufferAttribute(rayPositions, 3));
    rayGeometry.setDrawRange(0, 0);

    const rayMaterial = new THREE.LineBasicMaterial({
      color: 0x00f5d4,
      transparent: true,
      opacity: 0.9
    });
    const activeRaysLines = new THREE.LineSegments(rayGeometry, rayMaterial);
    scene.add(activeRaysLines);

    // State & Playback Controls
    let currentFrameIdx = 0;
    let isPlaying = true;
    let playbackSpeed = 1.0;
    let activeCameraPreset = 'orbit';
    const baseFps = Number(meta.fps || 30.0);
    let lastFrameTime = performance.now();
    let frameAccumulator = 0.0;

    const slider = document.getElementById('timeline-slider');
    slider.max = String(Math.max(0, frames.length - 1));

    function updateFrameState(frameIdx) {
      if (!frames.length) return;
      currentFrameIdx = Math.max(0, Math.min(frames.length - 1, frameIdx));
      slider.value = String(currentFrameIdx);

      const frameData = frames[currentFrameIdx];
      const estCm = frameData.est_pos_cm || [0, 0, 0];
      const estThree = ueToThree(estCm);
      estThree.y -= 2.80;
      const headingRad = Number(frameData.heading_rad || 0.0);

      // Position & Rotate 3D Truck
      truckMesh.position.copy(estThree);
      truckMesh.rotation.set(0, -headingRad, 0);
      const estCheck = document.getElementById('toggle-est-traj');
      if (estCheck) {
        truckMesh.visible = Boolean(estCheck.checked);
      }

      // Update GT Ghost Marker
      const gtCheck = document.getElementById('toggle-gt-traj');
      const shouldShowGt = hasGT && Boolean(frameData.gt_pos_cm) && (!gtCheck || gtCheck.checked);
      if (shouldShowGt) {
        const gtThree = ueToThree(frameData.gt_pos_cm);
        gtThree.y -= 2.80;
        gtGhostGroup.position.copy(gtThree);
        gtGhostGroup.rotation.set(0, -headingRad, 0);
        gtGhostGroup.visible = true;
      } else {
        gtGhostGroup.visible = false;
      }

      // Update Dynamic Camera Frustums (Cam 5 PTZ)
      const dynCams = frameData.dynamic_cameras || {};
      Object.keys(dynCams).forEach(camKey => {
        const rig = cameraRigMap[String(camKey)];
        const dynState = dynCams[camKey];
        if (rig && dynState) {
          if (dynState.frustum_corners_cm) {
            const newVerts = computeFrustumLinePositions(rig.posThree, dynState.frustum_corners_cm);
            rig.frustumGeom.attributes.position.array.set(newVerts);
            rig.frustumGeom.attributes.position.needsUpdate = true;
          }
          if (dynState.forward) {
            rig.currentForward = dynState.forward;
            rig.markerMesh.lookAt(rig.posThree.clone().add(ueDirToThree(dynState.forward)));
          }
          if (dynState.up) rig.currentUp = dynState.up;
        }
      });

      // Update Active Triangulation Rays & Camera Badge Highlights
      const activeRays = frameData.active_rays || [];
      const activeCamSet = new Set();
      let rayCount = Math.min(activeRays.length, MAX_RAYS);

      for (let i = 0; i < rayCount; i++) {
        const r = activeRays[i];
        activeCamSet.add(String(r.cam_id));
        const pOrig = ueToThree(r.origin_cm || [0, 0, 0]);
        const pEnd = estThree.clone();
        pEnd.y += 1.8; // Aim at truck center mass (1.8m above road)

        const base = i * 6;
        rayPositions[base] = pOrig.x;
        rayPositions[base + 1] = pOrig.y;
        rayPositions[base + 2] = pOrig.z;
        rayPositions[base + 3] = pEnd.x;
        rayPositions[base + 4] = pEnd.y;
        rayPositions[base + 5] = pEnd.z;
      }
      rayGeometry.setDrawRange(0, rayCount * 2);
      rayGeometry.attributes.position.needsUpdate = true;

      Object.values(cameraRigMap).forEach(rig => {
        if (activeCamSet.has(rig.camKey)) {
          rig.badge.classList.add('active-ray');
          rig.frustumMat.opacity = 0.95;
        } else {
          rig.badge.classList.remove('active-ray');
          rig.frustumMat.opacity = 0.45;
        }
      });

      // Update Camera View if in Chase or POV mode
      if (activeCameraPreset === 'chase') {
        const backOffset = new THREE.Vector3(-32, 16, 0).applyAxisAngle(
          new THREE.Vector3(0, 1, 0),
          -headingRad
        );
        camera.position.lerp(estThree.clone().add(backOffset), 0.25);
        controls.target.lerp(estThree.clone().add(new THREE.Vector3(0, 2.5, 0)), 0.35);
      } else if (activeCameraPreset.startsWith('cam_')) {
        const camKey = activeCameraPreset.replace('cam_', '');
        const rig = cameraRigMap[camKey];
        if (rig) {
          let lookDir, upDir;
          if (rig.info.rot_deg) {
            const r = THREE.MathUtils.degToRad(rig.info.rot_deg[0]);
            const p = THREE.MathUtils.degToRad(rig.info.rot_deg[1]);
            const y = THREE.MathUtils.degToRad(rig.info.rot_deg[2]);
            const forwardUE = [
              Math.cos(p) * Math.cos(y),
              Math.cos(p) * Math.sin(y),
              Math.sin(p)
            ];
            lookDir = ueDirToThree(forwardUE);
            const upUE = [
              -(Math.cos(r)*Math.sin(p)*Math.cos(y) + Math.sin(r)*Math.sin(y)),
              -(Math.cos(r)*Math.sin(p)*Math.sin(y) - Math.sin(r)*Math.cos(y)),
              Math.cos(r)*Math.cos(p)
            ];
            upDir = ueDirToThree(upUE);
          } else {
            lookDir = ueDirToThree(rig.currentForward);
            upDir = ueDirToThree(rig.currentUp);
          }
          camera.position.copy(rig.posThree);
          camera.up.copy(upDir);
          camera.lookAt(rig.posThree.clone().add(lookDir));
          controls.target.copy(rig.posThree.clone().add(lookDir.clone().multiplyScalar(50.0)));

          // Update PiP Crosshair if active ray exists for this camera
          const rayForCam = activeRays.find(r => String(r.cam_id) === camKey);
          const crosshair = document.getElementById('pip-crosshair');
          if (rayForCam && typeof rayForCam.u === 'number' && typeof rayForCam.v === 'number') {
            crosshair.style.display = 'block';
            crosshair.style.left = `${(rayForCam.u / 1920.0) * 100.0}%`;
            crosshair.style.top = `${(rayForCam.v / 1080.0) * 100.0}%`;
          } else {
            crosshair.style.display = 'none';
          }
        }
      }

      // Update Telemetry Readouts
      const timeSec = ((frameData.frame || currentFrameIdx) / baseFps).toFixed(2);
      document.getElementById('tel-frame').textContent =
        `${frameData.frame ?? currentFrameIdx} / ${frames.length - 1} (${timeSec}s)`;
      document.getElementById('scrubber-frame-label').textContent =
        `FRAME ${frameData.frame ?? currentFrameIdx} (${timeSec}s)`;

      const deg = ((headingRad * 180.0) / Math.PI).toFixed(1);
      document.getElementById('tel-heading').textContent = `${deg}° (${headingRad.toFixed(2)} rad)`;

      document.getElementById('tel-est-pos').textContent =
        `X: ${estCm[0].toFixed(1)} | Y: ${estCm[1].toFixed(1)} | Z: ${estCm[2].toFixed(1)}`;
      document.getElementById('scrubber-coords-label').textContent =
        `UE5 [m] X: ${(estCm[0] / 100).toFixed(1)}m | Y: ${(estCm[1] / 100).toFixed(1)}m | Z: ${(estCm[2] / 100).toFixed(1)}m`;

      let curErrCm = null;
      let diffX = 0, diffY = 0, diffZ = 0;
      let hasDiff = false;

      if (hasGT && frameData.gt_pos_cm && frameData.est_pos_cm) {
        const gtCm = frameData.gt_pos_cm;
        diffX = frameData.est_pos_cm[0] - gtCm[0];
        diffY = frameData.est_pos_cm[1] - gtCm[1];
        diffZ = frameData.est_pos_cm[2] - gtCm[2];
        curErrCm = Math.hypot(diffX, diffY, diffZ);
        hasDiff = true;

        document.getElementById('tel-gt-pos').textContent =
          `X: ${gtCm[0].toFixed(1)} | Y: ${gtCm[1].toFixed(1)} | Z: ${gtCm[2].toFixed(1)}`;
      } else if (typeof frameData.error_cm === 'number') {
        curErrCm = frameData.error_cm;
      }

      const telErrEl = document.getElementById('tel-err');
      const dynInstErrEl = document.getElementById('dyn-instant-err');
      const dynFrameIdxEl = document.getElementById('dyn-frame-idx');
      const dynRunningMaeEl = document.getElementById('dyn-running-mae');
      const dynAxisDiffEl = document.getElementById('dyn-axis-diff');

      if (dynFrameIdxEl) dynFrameIdxEl.textContent = String(frameData.frame ?? currentFrameIdx);

      if (typeof curErrCm === 'number' && !isNaN(curErrCm)) {
        const errText = `${curErrCm.toFixed(1)} cm (${(curErrCm / 100).toFixed(2)} m)`;
        if (telErrEl) telErrEl.textContent = errText;
        if (dynInstErrEl) {
          dynInstErrEl.textContent = errText;
          if (curErrCm < 150) {
            dynInstErrEl.style.color = '#22c55e';
          } else if (curErrCm < 300) {
            dynInstErrEl.style.color = '#fbbf24';
          } else {
            dynInstErrEl.style.color = '#f87171';
          }
        }
      } else {
        if (telErrEl) telErrEl.textContent = 'N/A';
        if (dynInstErrEl) dynInstErrEl.textContent = 'N/A';
      }

      if (dynRunningMaeEl) {
        const rMae = (runningMaeArray && runningMaeArray.length > currentFrameIdx) ? runningMaeArray[currentFrameIdx] : null;
        if (typeof rMae === 'number' && !isNaN(rMae)) {
          dynRunningMaeEl.textContent = `${rMae.toFixed(1)} cm (${(rMae / 100).toFixed(2)} m)`;
        } else {
          dynRunningMaeEl.textContent = 'N/A';
        }
      }

      if (dynAxisDiffEl) {
        if (hasDiff) {
          const sX = (diffX >= 0 ? '+' : '') + diffX.toFixed(1);
          const sY = (diffY >= 0 ? '+' : '') + diffY.toFixed(1);
          const sZ = (diffZ >= 0 ? '+' : '') + diffZ.toFixed(1);
          dynAxisDiffEl.textContent = `ΔX: ${sX} | ΔY: ${sY} | ΔZ: ${sZ} cm`;
        } else {
          dynAxisDiffEl.textContent = 'N/A (Blind Test)';
        }
      }

      document.getElementById('tel-active-cams').textContent =
        activeRays.length > 0
          ? `${activeRays.length} (${activeRays.map(r => 'C' + r.cam_id).join(', ')})`
          : '0 (Kalman Coasting)';

      if (activeRays.length > 0) {
        document.getElementById('tel-ray-details').innerHTML = activeRays
          .map(r => `Cam ${r.cam_id}: u=${Number(r.u || 0).toFixed(0)}, v=${Number(r.v || 0).toFixed(0)} (conf ${Number(r.conf || 0).toFixed(2)})`)
          .join('<br/>');
      } else {
        document.getElementById('tel-ray-details').textContent = 'Kalman 3D Constant-Velocity Prediction';
      }
    }

    function updateCameraHudLayout() {
      const hudOverlay = document.getElementById('cam-hud-overlay');
      if (!hudOverlay) return;

      const winW = window.innerWidth;
      const winH = window.innerHeight;

      if (!activeCameraPreset || !activeCameraPreset.startsWith('cam_')) {
        renderer.setViewport(0, 0, winW, winH);
        renderer.setScissor(0, 0, winW, winH);
        renderer.setScissorTest(false);
        camera.aspect = winW / winH;
        camera.updateProjectionMatrix();
        hudOverlay.style.display = 'none';
        return;
      }

      const targetAspect = 16.0 / 9.0;
      let viewW, viewH, left, top;

      if (winW / winH > targetAspect) {
        viewH = winH;
        viewW = winH * targetAspect;
        left = (winW - viewW) / 2;
        top = 0;
      } else {
        viewW = winW;
        viewH = winW / targetAspect;
        left = 0;
        top = (winH - viewH) / 2;
      }

      renderer.setViewport(left, winH - top - viewH, viewW, viewH);
      renderer.setScissor(left, winH - top - viewH, viewW, viewH);
      renderer.setScissorTest(true);

      camera.aspect = targetAspect;

      const camKey = activeCameraPreset.replace('cam_', '');
      const rig = cameraRigMap[camKey];
      if (rig) {
        const focal = Number(rig.info.focal_length_mm || 35.0);
        const sensorH = 13.365;
        camera.fov = 2.0 * Math.atan(sensorH / (2.0 * focal)) * (180.0 / Math.PI);
      }
      camera.updateProjectionMatrix();

      hudOverlay.style.display = 'block';
      hudOverlay.style.left = `${left}px`;
      hudOverlay.style.top = `${top}px`;
      hudOverlay.style.width = `${viewW}px`;
      hudOverlay.style.height = `${viewH}px`;
    }

    window.addEventListener('resize', updateCameraHudLayout);

    // Camera Preset Switching
    function setCameraPreset(preset) {
      activeCameraPreset = preset;
      document.querySelectorAll('#camera-presets .hud-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === preset);
      });

      const pipCard = document.getElementById('pip-card');
      const hudOverlay = document.getElementById('cam-hud-overlay');
      const hudImg = document.getElementById('cam-hud-img');
      camera.up.set(0, 1, 0);

      if (preset === 'orbit') {
        pipCard.classList.remove('visible');
        updateCameraHudLayout();
        controls.enabled = true;
        camera.fov = 48;
        camera.updateProjectionMatrix();
        camera.position.set(-180, 140, 260);
        controls.target.set(0, 5, 0);
      } else if (preset === 'top') {
        pipCard.classList.remove('visible');
        updateCameraHudLayout();
        controls.enabled = true;
        camera.fov = 45;
        camera.updateProjectionMatrix();
        camera.position.set(0, 620, 0.1);
        controls.target.set(0, 0, 0);
      } else if (preset === 'chase') {
        pipCard.classList.remove('visible');
        updateCameraHudLayout();
        controls.enabled = true;
        camera.fov = 52;
        camera.updateProjectionMatrix();
      } else if (preset.startsWith('cam_')) {
        const camKey = preset.replace('cam_', '');
        const rig = cameraRigMap[camKey];
        if (rig) {
          controls.enabled = false;

          let lookDir, upDir;
          if (rig.info.rot_deg) {
            const r = THREE.MathUtils.degToRad(rig.info.rot_deg[0]);
            const p = THREE.MathUtils.degToRad(rig.info.rot_deg[1]);
            const y = THREE.MathUtils.degToRad(rig.info.rot_deg[2]);
            const forwardUE = [
              Math.cos(p) * Math.cos(y),
              Math.cos(p) * Math.sin(y),
              Math.sin(p)
            ];
            lookDir = ueDirToThree(forwardUE);
            const upUE = [
              -(Math.cos(r)*Math.sin(p)*Math.cos(y) + Math.sin(r)*Math.sin(y)),
              -(Math.cos(r)*Math.sin(p)*Math.sin(y) - Math.sin(r)*Math.cos(y)),
              Math.cos(r)*Math.cos(p)
            ];
            upDir = ueDirToThree(upUE);
          } else {
            lookDir = ueDirToThree(rig.info.forward || [1, 0, 0]);
            upDir = ueDirToThree(rig.info.up || [0, 0, 1]);
          }

          camera.position.copy(rig.posThree);
          camera.up.copy(upDir);
          camera.lookAt(rig.posThree.clone().add(lookDir));
          controls.target.copy(rig.posThree.clone().add(lookDir.clone().multiplyScalar(50.0)));

          if (hudOverlay && hudImg && rig.info.thumbnail_b64) {
            hudImg.src = rig.info.thumbnail_b64;
          }

          updateCameraHudLayout();

          if (rig.info.thumbnail_b64) {
            pipCard.classList.add('visible');
            document.getElementById('pip-title').textContent = rig.info.name || `Camera ${camKey}`;
            document.getElementById('pip-img').src = rig.info.thumbnail_b64;
            document.getElementById('pip-meta').textContent =
              `Focal: ${rig.info.focal_length_mm || '--'}mm | FOV_H: ${(rig.info.fov_h_deg || 0).toFixed(1)}°`;
          } else {
            pipCard.classList.remove('visible');
          }
        }
      }
      updateFrameState(currentFrameIdx);
      if (typeof saveViewerSettings === 'function') {
        saveViewerSettings();
      }
    }

    document.querySelectorAll('#camera-presets .hud-btn').forEach(btn => {
      btn.addEventListener('click', () => setCameraPreset(btn.dataset.view));
    });

    const sliderPhoto = document.getElementById('slider-photo-opacity');
    if (sliderPhoto) {
      sliderPhoto.addEventListener('input', e => {
        const val = parseFloat(e.target.value) / 100.0;
        const lbl = document.getElementById('photo-opacity-val');
        if (lbl) lbl.textContent = `${e.target.value}%`;
        const hudImg = document.getElementById('cam-hud-img');
        if (hudImg) hudImg.style.opacity = String(val);
      });
    }

    window.addEventListener('keydown', e => {
      if (e.key === 'Escape' && activeCameraPreset && activeCameraPreset.startsWith('cam_')) {
        setCameraPreset('orbit');
      }
    });

    // -------------------------------------------------------------
    // Persistent User Preferences & Startup Synchronization Module
    // -------------------------------------------------------------
    const STORAGE_KEY = 'peleng_viewer_settings_v5';

    const DEFAULT_SETTINGS = {
      layers: {
        'toggle-terrain': true,
        'toggle-wireframe': true,
        'toggle-pointcloud': true,
        'toggle-road': true,
        'toggle-landmarks': true,
        'toggle-frustums': true,
        'toggle-rays': true,
        'toggle-est-traj': true,
        'toggle-gt-traj': true
      },
      sliders: {
        'slider-pt-size': 2.6,
        'slider-traj-width': 0.60
      },
      selects: {
        'select-traj-style': 'contrast',
        'select-speed': '1',
        'select-active-model': defaultModelKey
      },
      cameraPreset: 'orbit',
      panels: {
        leftCollapsed: false,
        rightCollapsed: true
      }
    };

    function loadViewerSettings() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return JSON.parse(JSON.stringify(DEFAULT_SETTINGS));
        const parsed = JSON.parse(raw);
        return {
          layers: Object.assign({}, DEFAULT_SETTINGS.layers, parsed.layers || {}),
          sliders: Object.assign({}, DEFAULT_SETTINGS.sliders, parsed.sliders || {}),
          selects: Object.assign({}, DEFAULT_SETTINGS.selects, parsed.selects || {}),
          cameraPreset: typeof parsed.cameraPreset === 'string' ? parsed.cameraPreset : DEFAULT_SETTINGS.cameraPreset,
          panels: Object.assign({}, DEFAULT_SETTINGS.panels, parsed.panels || {})
        };
      } catch (err) {
        console.warn('[Viewer3D] Failed to read settings from localStorage:', err);
        return JSON.parse(JSON.stringify(DEFAULT_SETTINGS));
      }
    }

    function saveViewerSettings() {
      try {
        const settings = {
          layers: {},
          sliders: {},
          selects: {},
          cameraPreset: activeCameraPreset || 'orbit',
          panels: {
            leftCollapsed: hudLeft ? hudLeft.classList.contains('collapsed') : false,
            rightCollapsed: hudRight ? hudRight.classList.contains('collapsed') : false
          }
        };

        const layerIds = [
          'toggle-terrain',
          'toggle-wireframe',
          'toggle-pointcloud',
          'toggle-road',
          'toggle-landmarks',
          'toggle-frustums',
          'toggle-rays',
          'toggle-est-traj',
          'toggle-gt-traj'
        ];
        layerIds.forEach(id => {
          const el = document.getElementById(id);
          if (el) settings.layers[id] = Boolean(el.checked);
        });

        const ptSlider = document.getElementById('slider-pt-size');
        if (ptSlider) settings.sliders['slider-pt-size'] = parseFloat(ptSlider.value);

        const trajSlider = document.getElementById('slider-traj-width');
        if (trajSlider) settings.sliders['slider-traj-width'] = parseFloat(trajSlider.value);

        const styleSelect = document.getElementById('select-traj-style');
        if (styleSelect) settings.selects['select-traj-style'] = styleSelect.value;

        const speedSelect = document.getElementById('select-speed');
        if (speedSelect) settings.selects['select-speed'] = speedSelect.value;

        const modelSelect = document.getElementById('select-active-model');
        if (modelSelect) settings.selects['select-active-model'] = modelSelect.value;

        localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
      } catch (err) {
        console.warn('[Viewer3D] Failed to save settings to localStorage:', err);
      }
    }

    function syncAllLayersAndControls() {
      // 1. Terrain Mesh
      const toggleTerrain = document.getElementById('toggle-terrain');
      if (terrainMeshObj && toggleTerrain) {
        terrainMeshObj.visible = Boolean(toggleTerrain.checked);
      }

      // 2. Wireframe Overlay
      const toggleWireframe = document.getElementById('toggle-wireframe');
      if (wireframeMeshObj && toggleWireframe) {
        wireframeMeshObj.visible = Boolean(toggleWireframe.checked);
      }

      // 3. Point Cloud
      const togglePointCloud = document.getElementById('toggle-pointcloud');
      if (pointCloudObj && togglePointCloud) {
        pointCloudObj.visible = Boolean(togglePointCloud.checked);
      }
      const sliderPt = document.getElementById('slider-pt-size');
      if (sliderPt) {
        const ptVal = parseFloat(sliderPt.value);
        const ptLabel = document.getElementById('pt-size-label');
        if (ptLabel) ptLabel.textContent = `${ptVal.toFixed(1)} px`;
        if (pointCloudMaterial) pointCloudMaterial.size = ptVal;
      }

      // 4. Roadway
      const toggleRoad = document.getElementById('toggle-road');
      if (roadGroup && toggleRoad) {
        roadGroup.visible = Boolean(toggleRoad.checked);
      }

      // 5. Landmarks (GCP Trees)
      const toggleLm = document.getElementById('toggle-landmarks');
      if (landmarksGroup && toggleLm) {
        landmarksGroup.visible = Boolean(toggleLm.checked);
      }

      // 6. Camera Frustums & Badges
      const toggleFrustums = document.getElementById('toggle-frustums');
      if (toggleFrustums) {
        if (camerasGroup) camerasGroup.visible = Boolean(toggleFrustums.checked);
      }
      if (labelsContainer) {
        labelsContainer.style.display = (toggleFrustums?.checked || toggleLm?.checked) ? 'block' : 'none';
      }

      // 7. Live Triangulation Rays
      const toggleRays = document.getElementById('toggle-rays');
      if (activeRaysLines && toggleRays) {
        activeRaysLines.visible = Boolean(toggleRays.checked);
      }

      // 8. Trajectory Width & Style
      const sliderTraj = document.getElementById('slider-traj-width');
      if (sliderTraj) {
        currentTrajRadius = parseFloat(sliderTraj.value);
        const trajWidthLabel = document.getElementById('traj-width-label');
        if (trajWidthLabel) trajWidthLabel.textContent = `${currentTrajRadius.toFixed(2)} m`;
      }
      const selectStyle = document.getElementById('select-traj-style');
      if (selectStyle) {
        currentTrajStyle = selectStyle.value;
      }
      rebuildTrajectories(currentTrajRadius, currentTrajStyle);

      // 9. Predicted Trajectory & AA Truck
      const toggleEst = document.getElementById('toggle-est-traj');
      if (toggleEst) {
        if (estTrajectoryLine) estTrajectoryLine.visible = Boolean(toggleEst.checked);
        if (truckMesh) truckMesh.visible = Boolean(toggleEst.checked);
      }

      // 10. Ground Truth Trajectory & Truck Ghost
      const toggleGt = document.getElementById('toggle-gt-traj');
      if (toggleGt) {
        if (gtTrajectoryLine) gtTrajectoryLine.visible = Boolean(toggleGt.checked);
        if (gtGhostGroup) gtGhostGroup.visible = Boolean(hasGT && toggleGt.checked);
      }

      // 11. Playback Speed
      const selectSpeed = document.getElementById('select-speed');
      if (selectSpeed) {
        playbackSpeed = parseFloat(selectSpeed.value);
      }
    }

    function applyModel(mKey, skipSave = false) {
      const models = SCENE_DATA.models || {};
      const chosen = models[mKey];
      if (!chosen || !chosen.frames || chosen.frames.length === 0) return;

      frames = chosen.frames;
      slider.max = String(Math.max(0, frames.length - 1));

      precomputeRunningMaeForActiveFrames();

      const descEl = document.getElementById('model-desc-badge');
      if (descEl) {
        descEl.innerHTML = `<span style="color:${chosen.color || '#00f5d4'};font-weight:700;">${chosen.short_name || chosen.name}</span>: ${chosen.desc || ''}`;
      }
      const activeBadge = document.getElementById('active-model-summary-badge');
      if (activeBadge) {
        activeBadge.textContent = chosen.short_name || chosen.name;
      }

      if (chosen.metrics) {
        const m = chosen.metrics;
        const mae = Number(m.mae_3d || 0);
        const rmse = Number(m.rmse_3d || 0);
        const elMae = document.getElementById('sum-mae3d');
        if (elMae) elMae.textContent = `${mae.toFixed(1)} cm (${(mae / 100).toFixed(2)} m)`;
        const elRmse = document.getElementById('sum-rmse3d');
        if (elRmse) elRmse.textContent = `${rmse.toFixed(1)} cm (${(rmse / 100).toFixed(2)} m)`;
        const elAxes = document.getElementById('sum-axes');
        if (elAxes && m.mae_x !== undefined) {
          elAxes.textContent = `${Number(m.mae_x || 0).toFixed(1)} / ${Number(m.mae_y || 0).toFixed(1)} / ${Number(m.mae_z || 0).toFixed(1)}`;
        }
      }

      if (chosen.color) {
        activeModelColor = parseInt(chosen.color.replace('#', '0x'), 16);
      } else {
        activeModelColor = null;
      }
      const swEst = document.getElementById('swatch-est');
      if (swEst && chosen.color) {
        swEst.style.background = chosen.color;
      }

      const rawEst = [];
      const rawRoad = [];
      frames.forEach(f => {
        if (f.est_pos_cm) {
          const p = ueToThree(f.est_pos_cm);
          rawEst.push(new THREE.Vector3(p.x, p.y - 2.80 + 0.35, p.z));
          if (!hasGT || !f.gt_pos_cm) {
            rawRoad.push(new THREE.Vector3(p.x, p.y - 2.80, p.z));
          }
        }
        if (hasGT && f.gt_pos_cm) {
          const g = ueToThree(f.gt_pos_cm);
          rawRoad.push(new THREE.Vector3(g.x, g.y - 2.80, g.z));
        }
      });
      const cleanEst = deduplicatePoints(rawEst);
      if (cleanEst.length >= 2) {
        estCurve = new THREE.CatmullRomCurve3(cleanEst);
      }
      const cleanRoad = deduplicatePoints(rawRoad, 0.25);
      if (cleanRoad.length >= 2) {
        roadBaseCurve = new THREE.CatmullRomCurve3(cleanRoad);
        rebuildRoadwayRibbons();
      }

      rebuildTrajectories(currentTrajRadius, currentTrajStyle);

      const selectModel = document.getElementById('select-active-model');
      if (selectModel && selectModel.value !== mKey) {
        selectModel.value = mKey;
      }

      updateFrameState(currentFrameIdx);

      if (!skipSave) {
        saveViewerSettings();
      }
    }

    function initViewerSettings() {
      const settings = loadViewerSettings();

      // Apply saved layers to DOM checkboxes
      if (settings.layers) {
        Object.keys(settings.layers).forEach(id => {
          const el = document.getElementById(id);
          if (el) el.checked = Boolean(settings.layers[id]);
        });
      }

      // Apply saved sliders to DOM
      if (settings.sliders) {
        if (settings.sliders['slider-pt-size'] !== undefined) {
          const el = document.getElementById('slider-pt-size');
          if (el) el.value = String(settings.sliders['slider-pt-size']);
        }
        if (settings.sliders['slider-traj-width'] !== undefined) {
          const el = document.getElementById('slider-traj-width');
          if (el) el.value = String(settings.sliders['slider-traj-width']);
        }
      }

      // Apply saved selects to DOM
      if (settings.selects) {
        if (settings.selects['select-traj-style'] !== undefined) {
          const el = document.getElementById('select-traj-style');
          if (el) el.value = String(settings.selects['select-traj-style']);
        }
        if (settings.selects['select-speed'] !== undefined) {
          const el = document.getElementById('select-speed');
          if (el) el.value = String(settings.selects['select-speed']);
        }
      }

      // Apply panel collapsed states
      if (settings.panels) {
        if (hudLeft && settings.panels.leftCollapsed !== undefined) {
          hudLeft.classList.toggle('collapsed', Boolean(settings.panels.leftCollapsed));
        }
        if (hudRight && settings.panels.rightCollapsed !== undefined) {
          hudRight.classList.toggle('collapsed', Boolean(settings.panels.rightCollapsed));
        }
        syncPanelToggleButtons();
      }

      // Explicitly synchronize Three.js objects with current DOM state
      syncAllLayersAndControls();

      // Apply camera preset
      if (settings.cameraPreset && settings.cameraPreset !== 'orbit') {
        setCameraPreset(settings.cameraPreset);
      }

      // Bind change listeners to checkboxes
      const layerIds = [
        'toggle-terrain',
        'toggle-wireframe',
        'toggle-pointcloud',
        'toggle-road',
        'toggle-landmarks',
        'toggle-frustums',
        'toggle-rays',
        'toggle-est-traj',
        'toggle-gt-traj'
      ];
      layerIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          el.addEventListener('change', () => {
            syncAllLayersAndControls();
            updateFrameState(currentFrameIdx);
            saveViewerSettings();
          });
        }
      });

      // Bind slider listeners
      const sliderPt = document.getElementById('slider-pt-size');
      if (sliderPt) {
        sliderPt.addEventListener('input', e => {
          const val = parseFloat(e.target.value);
          const ptLabel = document.getElementById('pt-size-label');
          if (ptLabel) ptLabel.textContent = `${val.toFixed(1)} px`;
          if (pointCloudMaterial) pointCloudMaterial.size = val;
          saveViewerSettings();
        });
      }

      const sliderTraj = document.getElementById('slider-traj-width');
      if (sliderTraj) {
        sliderTraj.addEventListener('input', e => {
          currentTrajRadius = parseFloat(e.target.value);
          const trajWidthLabel = document.getElementById('traj-width-label');
          if (trajWidthLabel) trajWidthLabel.textContent = `${currentTrajRadius.toFixed(2)} m`;
          rebuildTrajectories(currentTrajRadius, currentTrajStyle);
          saveViewerSettings();
        });
      }

      // Bind select listeners
      const selectStyle = document.getElementById('select-traj-style');
      if (selectStyle) {
        selectStyle.addEventListener('change', e => {
          currentTrajStyle = e.target.value;
          rebuildTrajectories(currentTrajRadius, currentTrajStyle);
          saveViewerSettings();
        });
      }

      const selectSpeed = document.getElementById('select-speed');
      if (selectSpeed) {
        selectSpeed.addEventListener('change', e => {
          playbackSpeed = parseFloat(e.target.value);
          saveViewerSettings();
        });
      }

      // Bind model selector listener
      const selectModel = document.getElementById('select-active-model');
      if (selectModel) {
        const modelsData = SCENE_DATA.models || null;
        let activeKey = (settings?.selects && settings.selects['select-active-model']) || SCENE_DATA.active_model || 'cabin';
        if (modelsData && Object.keys(modelsData).length > 0) {
          if (!modelsData[activeKey]) {
            activeKey = Object.keys(modelsData)[0];
          }
          selectModel.innerHTML = '';
          Object.keys(modelsData).forEach(mId => {
            const m = modelsData[mId];
            const opt = document.createElement('option');
            opt.value = mId;
            opt.textContent = m.name || mId;
            if (mId === activeKey) opt.selected = true;
            selectModel.appendChild(opt);
          });
        }
        selectModel.value = activeKey;
        selectModel.addEventListener('change', e => {
          applyModel(e.target.value);
        });
        applyModel(activeKey, true);
      }

      // Resizable Telemetry Panel (Left Edge Drag)
      const resizer = document.getElementById('hud-right-resizer');
      const panelRight = document.getElementById('hud-right');
      if (resizer && panelRight) {
        let isResizing = false;
        let startX = 0;
        let startW = 0;
        resizer.addEventListener('mousedown', e => {
          isResizing = true;
          startX = e.clientX;
          startW = panelRight.getBoundingClientRect().width;
          resizer.classList.add('resizing');
          document.body.style.cursor = 'ew-resize';
          document.body.style.userSelect = 'none';
          e.preventDefault();
        });
        window.addEventListener('mousemove', e => {
          if (!isResizing) return;
          const dx = startX - e.clientX; // dragging left expands panel width
          const newW = Math.max(280, Math.min(700, startW + dx));
          panelRight.style.width = `${newW}px`;
        });
        window.addEventListener('mouseup', () => {
          if (isResizing) {
            isResizing = false;
            resizer.classList.remove('resizing');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
          }
        });
      }

      // Reset button listener
      const btnReset = document.getElementById('btn-reset-settings');
      if (btnReset) {
        btnReset.addEventListener('click', () => {
          try {
            localStorage.removeItem(STORAGE_KEY);
          } catch (e) {}
          Object.keys(DEFAULT_SETTINGS.layers).forEach(id => {
            const el = document.getElementById(id);
            if (el) el.checked = Boolean(DEFAULT_SETTINGS.layers[id]);
          });
          if (sliderPt) sliderPt.value = String(DEFAULT_SETTINGS.sliders['slider-pt-size']);
          if (sliderTraj) sliderTraj.value = String(DEFAULT_SETTINGS.sliders['slider-traj-width']);
          if (selectStyle) selectStyle.value = DEFAULT_SETTINGS.selects['select-traj-style'];
          if (selectSpeed) selectSpeed.value = DEFAULT_SETTINGS.selects['select-speed'];

          if (hudLeft) hudLeft.classList.toggle('collapsed', Boolean(DEFAULT_SETTINGS.panels.leftCollapsed));
          if (hudRight) hudRight.classList.toggle('collapsed', Boolean(DEFAULT_SETTINGS.panels.rightCollapsed));
          syncPanelToggleButtons();

          syncAllLayersAndControls();
          setCameraPreset('orbit');
          updateFrameState(currentFrameIdx);
          saveViewerSettings();
        });
      }
    }

    // Timeline Transport Event Listeners
    const btnPlay = document.getElementById('btn-play');
    function togglePlay() {
      isPlaying = !isPlaying;
      btnPlay.textContent = isPlaying ? 'PAUSE' : 'PLAY';
      btnPlay.classList.toggle('active', isPlaying);
    }

    btnPlay.addEventListener('click', togglePlay);
    document.getElementById('btn-prev').addEventListener('click', () => {
      isPlaying = false;
      btnPlay.textContent = 'PLAY';
      btnPlay.classList.remove('active');
      updateFrameState(currentFrameIdx - 1);
    });
    document.getElementById('btn-next').addEventListener('click', () => {
      isPlaying = false;
      btnPlay.textContent = 'PLAY';
      btnPlay.classList.remove('active');
      updateFrameState(currentFrameIdx + 1);
    });
    slider.addEventListener('input', e => {
      isPlaying = false;
      btnPlay.textContent = 'PLAY';
      btnPlay.classList.remove('active');
      updateFrameState(parseInt(e.target.value, 10));
    });
    document.getElementById('select-speed').addEventListener('change', e => {
      playbackSpeed = parseFloat(e.target.value);
    });

    window.addEventListener('keydown', e => {
      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'KeyH') {
        e.preventDefault();
        toggleBothSidePanels();
      } else if (e.code === 'ArrowLeft') {
        isPlaying = false;
        btnPlay.textContent = 'PLAY';
        btnPlay.classList.remove('active');
        updateFrameState(currentFrameIdx - 1);
      } else if (e.code === 'ArrowRight') {
        isPlaying = false;
        btnPlay.textContent = 'PLAY';
        btnPlay.classList.remove('active');
        updateFrameState(currentFrameIdx + 1);
      }
    });

    // Project 3D Camera Badges to 2D Screen Coordinates
    const tempProjVec = new THREE.Vector3();
    function updateCameraBadgesScreenPos() {
      if (labelsContainer && labelsContainer.style.display === 'none') {
        return;
      }
      const w = window.innerWidth;
      const h = window.innerHeight;

      const frustumsVisible = document.getElementById('toggle-frustums')?.checked ?? true;
      Object.values(cameraRigMap).forEach(rig => {
        if (!frustumsVisible) {
          rig.badge.style.display = 'none';
          return;
        }
        tempProjVec.copy(rig.posThree);
        tempProjVec.y += 4.5;
        tempProjVec.project(camera);
        if (tempProjVec.z > 1.0 || tempProjVec.z < -1.0) {
          rig.badge.style.display = 'none';
        } else {
          rig.badge.style.display = 'block';
          const sx = (tempProjVec.x * 0.5 + 0.5) * w;
          const sy = (-(tempProjVec.y * 0.5) + 0.5) * h;
          rig.badge.style.left = `${sx}px`;
          rig.badge.style.top = `${sy}px`;
        }
      });
    }

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / Math.max(1, window.innerHeight);
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // Initial Preferences Restoration & Frame Setup
    initViewerSettings();
    updateFrameState(0);

    function animate(now) {
      requestAnimationFrame(animate);
      const dtSec = Math.min(0.1, (now - lastFrameTime) / 1000.0);
      lastFrameTime = now;

      if (isPlaying && frames.length > 1) {
        frameAccumulator += dtSec * baseFps * playbackSpeed;
        if (frameAccumulator >= 1.0) {
          const steps = Math.floor(frameAccumulator);
          frameAccumulator -= steps;
          const nextFrame = (currentFrameIdx + steps) % frames.length;
          updateFrameState(nextFrame);
        }
      }

      if (!activeCameraPreset || !activeCameraPreset.startsWith('cam_')) {
        controls.update();
      }
      updateCameraBadgesScreenPos();
      renderer.render(scene, camera);
    }

    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""
