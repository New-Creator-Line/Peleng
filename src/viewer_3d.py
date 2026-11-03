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
      width: 310px;
      max-height: calc(100vh - 104px);
      overflow-y: auto;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      transition: transform 0.24s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.2s ease;
    }

    #hud-right.collapsed {
      transform: translateX(calc(100% + 24px));
      opacity: 0;
      pointer-events: none;
    }

    .telemetry-badges {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      flex-wrap: wrap;
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
  <div id="labels-container"></div>
  <div id="fallback-banner"></div>

  <!-- Compact Floating Reopen Pill Buttons (shown when side panels are collapsed) -->
  <button id="btn-open-left" data-toggle="btn-toggle-left" class="hud-reopen-pill left-pill" title="Показать панель камер и слоев">
    ☰ Камеры и Слои
  </button>
  <button id="btn-open-right" data-toggle="btn-toggle-right" class="hud-reopen-pill right-pill" title="Показать панель телеметрии">
    📊 Телеметрия
  </button>

  <!-- Left Panel: Camera View Presets & Layer Visibility -->
  <aside id="hud-left" class="hud-panel">
    <div class="panel-header-bar">
      <span class="panel-header-title">☰ Камеры и Слои</span>
      <button id="btn-collapse-left" class="panel-collapse-btn" title="Скрыть панель">[ — Скрыть ]</button>
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
      <div class="section-title">3D Scene Layers</div>
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
  <aside id="hud-right" class="hud-panel">
    <div class="panel-header-bar">
      <span class="panel-header-title">📊 Телеметрия</span>
      <button id="btn-collapse-right" class="panel-collapse-btn" title="Скрыть панель">[ Скрыть — ]</button>
    </div>

    <div class="telemetry-badges">
      <span class="pill-badge" id="badge-mode">VALIDATION MODE</span>
      <span class="pill-badge" id="badge-depth-model">Depth: Depth-Anything-V2</span>
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
      <div class="section-title">Pipeline Accuracy Summary</div>
      <div class="telemetry-grid">
        <div class="metric-box">
          <div class="metric-label">MAE_3D (cm / m)</div>
          <div class="metric-value cyan" id="sum-mae3d">--</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">RMSE_3D (cm / m)</div>
          <div class="metric-value cyan" id="sum-rmse3d">--</div>
        </div>
        <div class="metric-box full-width">
          <div class="metric-label">Axis MAE (X / Y / Z) [cm]</div>
          <div class="metric-value" id="sum-axes">--</div>
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
    const metrics = SCENE_DATA.metrics || {};
    const frames = SCENE_DATA.frames || [];
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

    // Initialize Mode & Static Accuracy Metrics (with defensive element guards)
    const badgeDepthEl = document.getElementById('badge-depth-model');
    if (badgeDepthEl) {
      badgeDepthEl.textContent = `Depth: ${(meta.depth_model_used || 'Depth-Anything-V2').split('/').pop()}`;
    }

    const badgeMode = document.getElementById('badge-mode');
    if (hasGT) {
      if (badgeMode) {
        badgeMode.textContent = 'VALIDATION (WITH GT)';
        badgeMode.className = 'pill-badge mode-gt';
      }
    } else {
      if (badgeMode) {
        badgeMode.textContent = 'BLIND TEST MODE (NO GT)';
        badgeMode.className = 'pill-badge mode-blind';
      }
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
    }

    function setRightPanelCollapsed(collapsed) {
      hudRight.classList.toggle('collapsed', Boolean(collapsed));
      syncPanelToggleButtons();
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
        const mat = new THREE.MeshStandardMaterial({
          color: style.estColor,
          emissive: style.estEmissive,
          emissiveIntensity: style.estEmissiveIntensity,
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
          rawEst.push(new THREE.Vector3(p.x, p.y + 0.55, p.z));
          if (!hasGT || !f.gt_pos_cm) {
            rawRoad.push(new THREE.Vector3(p.x, p.y, p.z));
          }
        }
        if (hasGT && f.gt_pos_cm) {
          const g = ueToThree(f.gt_pos_cm);
          rawGt.push(new THREE.Vector3(g.x, g.y + 0.48, g.z));
          rawRoad.push(new THREE.Vector3(g.x, g.y, g.z));
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
      const headingRad = Number(frameData.heading_rad || 0.0);

      // Position & Rotate 3D Truck
      truckMesh.position.copy(estThree);
      truckMesh.rotation.set(0, -headingRad, 0);

      // Update GT Ghost Marker
      if (hasGT && frameData.gt_pos_cm && document.getElementById('toggle-gt-traj').checked) {
        const gtThree = ueToThree(frameData.gt_pos_cm);
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
        const pEnd = ueToThree(r.end_cm || estCm);
        pEnd.y += 1.4; // Aim at truck center mass

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
          const fwdThree = ueDirToThree(rig.currentForward);
          const upThree = ueDirToThree(rig.currentUp);
          camera.position.copy(rig.posThree);
          camera.up.copy(upThree);
          controls.target.copy(rig.posThree.clone().add(fwdThree.multiplyScalar(80.0)));

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

      if (hasGT && frameData.gt_pos_cm) {
        const gtCm = frameData.gt_pos_cm;
        document.getElementById('tel-gt-pos').textContent =
          `X: ${gtCm[0].toFixed(1)} | Y: ${gtCm[1].toFixed(1)} | Z: ${gtCm[2].toFixed(1)}`;
        if (typeof frameData.error_cm === 'number') {
          document.getElementById('tel-err').textContent =
            `${frameData.error_cm.toFixed(1)} cm (${(frameData.error_cm / 100).toFixed(2)} m)`;
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

    // Camera Preset Switching
    function setCameraPreset(preset) {
      activeCameraPreset = preset;
      document.querySelectorAll('#camera-presets .hud-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === preset);
      });

      const pipCard = document.getElementById('pip-card');
      camera.up.set(0, 1, 0);

      if (preset === 'orbit') {
        pipCard.classList.remove('visible');
        camera.fov = 48;
        camera.updateProjectionMatrix();
        camera.position.set(-180, 140, 260);
        controls.target.set(0, 5, 0);
      } else if (preset === 'top') {
        pipCard.classList.remove('visible');
        camera.fov = 45;
        camera.updateProjectionMatrix();
        camera.position.set(0, 620, 0.1);
        controls.target.set(0, 0, 0);
      } else if (preset === 'chase') {
        pipCard.classList.remove('visible');
        camera.fov = 52;
        camera.updateProjectionMatrix();
      } else if (preset.startsWith('cam_')) {
        const camKey = preset.replace('cam_', '');
        const rig = cameraRigMap[camKey];
        if (rig) {
          const fovH = Number(rig.info.fov_h_deg || 40.0);
          const aspect = window.innerWidth / Math.max(1, window.innerHeight);
          const fovV = (2.0 * Math.atan(Math.tan((fovH * Math.PI) / 360.0) / aspect) * 180.0) / Math.PI;
          camera.fov = Math.max(15, Math.min(85, fovV));
          camera.updateProjectionMatrix();

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
    }

    document.querySelectorAll('#camera-presets .hud-btn').forEach(btn => {
      btn.addEventListener('click', () => setCameraPreset(btn.dataset.view));
    });

    // Layer Visibility Event Listeners
    document.getElementById('toggle-terrain').addEventListener('change', e => {
      if (terrainMeshObj) terrainMeshObj.visible = e.target.checked;
    });
    document.getElementById('toggle-wireframe').addEventListener('change', e => {
      if (wireframeMeshObj) wireframeMeshObj.visible = e.target.checked;
    });
    document.getElementById('toggle-pointcloud').addEventListener('change', e => {
      if (pointCloudObj) pointCloudObj.visible = e.target.checked;
    });
    document.getElementById('slider-pt-size').addEventListener('input', e => {
      const val = parseFloat(e.target.value);
      document.getElementById('pt-size-label').textContent = `${val.toFixed(1)} px`;
      if (pointCloudMaterial) pointCloudMaterial.size = val;
    });
    document.getElementById('toggle-road').addEventListener('change', e => {
      roadGroup.visible = e.target.checked;
    });
    document.getElementById('toggle-frustums').addEventListener('change', e => {
      camerasGroup.visible = e.target.checked;
      labelsContainer.style.display = e.target.checked ? 'block' : 'none';
    });
    document.getElementById('toggle-rays').addEventListener('change', e => {
      activeRaysLines.visible = e.target.checked;
    });
    document.getElementById('toggle-est-traj').addEventListener('change', e => {
      if (estTrajectoryLine) estTrajectoryLine.visible = e.target.checked;
    });
    document.getElementById('toggle-gt-traj').addEventListener('change', e => {
      if (gtTrajectoryLine) gtTrajectoryLine.visible = e.target.checked;
      updateFrameState(currentFrameIdx);
    });
    document.getElementById('slider-traj-width').addEventListener('input', e => {
      currentTrajRadius = parseFloat(e.target.value);
      document.getElementById('traj-width-label').textContent = `${currentTrajRadius.toFixed(2)} m`;
      rebuildTrajectories(currentTrajRadius, currentTrajStyle);
    });
    document.getElementById('select-traj-style').addEventListener('change', e => {
      currentTrajStyle = e.target.value;
      rebuildTrajectories(currentTrajRadius, currentTrajStyle);
    });

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
      const w = window.innerWidth;
      const h = window.innerHeight;
      Object.values(cameraRigMap).forEach(rig => {
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

    // Initial Frame Setup & Render Loop
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

      controls.update();
      updateCameraBadgesScreenPos();
      renderer.render(scene, camera);
    }

    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""
