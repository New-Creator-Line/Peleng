import json
import os
import re
from typing import Any, Dict

import numpy as np
import pytest

from src.viewer_3d import export_3d_viewer_html


def _make_mock_scene_payload(has_ground_truth: bool = True) -> Dict[str, Any]:
    """Builds a realistic Scene3DPayload dictionary matching scene_3d_contract.md."""
    rows, cols = 4, 4
    x_grid = np.linspace(-25000.0, 15000.0, cols).tolist()
    y_grid = np.linspace(-15000.0, 10000.0, rows).tolist()
    z_matrix = [[300.0 + 50.0 * r + 25.0 * c for c in range(cols)] for r in range(rows)]
    rgb_matrix = [[ [94, 112, 78] for _ in range(cols)] for _ in range(rows)]

    cameras = {}
    focal_map = {1: 35.0, 2: 60.0, 3: 35.0, 4: 50.0, 5: 20.0}
    for cid in range(1, 6):
        cameras[str(cid)] = {
            "cam_id": cid,
            "name": f"CineCameraActor{cid} ({int(focal_map[cid])}mm)",
            "focal_length_mm": focal_map[cid],
            "sensor_w_mm": 23.76,
            "sensor_h_mm": 13.365,
            "fov_h_deg": 37.5,
            "is_dynamic": cid == 5,
            "pos_cm": [-24702.0 + cid * 2000.0, 6624.0 - cid * 1000.0, 1200.0],
            "forward": [0.342, -0.939, 0.0],
            "right": [0.939, 0.342, 0.0],
            "up": [0.0, 0.0, 1.0],
            "frustum_corners_cm": [
                [-22000.0, 2800.0, 1950.0],
                [-24500.0, 1900.0, 1950.0],
                [-24500.0, 1900.0, 450.0],
                [-22000.0, 2800.0, 450.0],
            ],
            "thumbnail_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        }

    frames = []
    for f_idx in range(5):
        est_pos = [-21500.0 + f_idx * 120.0, -12800.0 + f_idx * 80.0, 302.0 + f_idx * 2.0]
        gt_pos = [-21502.0 + f_idx * 120.0, -12801.0 + f_idx * 80.0, 300.0 + f_idx * 2.0] if has_ground_truth else None
        err_cm = 3.1 if has_ground_truth else None
        frames.append(
            {
                "frame": f_idx,
                "est_pos_cm": est_pos,
                "gt_pos_cm": gt_pos,
                "heading_rad": 0.62 + f_idx * 0.01,
                "error_cm": err_cm,
                "active_rays": [
                    {
                        "cam_id": 1,
                        "origin_cm": [-24702.0, 6624.0, 1200.0],
                        "dir": [0.162, -0.985, -0.045],
                        "end_cm": est_pos,
                        "u": 1024.5,
                        "v": 580.2,
                        "conf": 0.88,
                    },
                    {
                        "cam_id": 5,
                        "origin_cm": [-14702.0, 1624.0, 1200.0],
                        "dir": [-0.312, -0.945, -0.055],
                        "end_cm": est_pos,
                        "u": 960.0,
                        "v": 540.0,
                        "conf": 0.92,
                    },
                ],
                "dynamic_cameras": {
                    "5": {
                        "forward": [-0.992, -0.003 - f_idx * 0.01, -0.122],
                        "right": [0.003, -1.0, 0.0],
                        "up": [-0.122, 0.0, 0.992],
                        "frustum_corners_cm": [
                            [5800.0, -6200.0, 2200.0],
                            [5800.0, -1600.0, 2200.0],
                            [5800.0, -1600.0, 180.0],
                            [5800.0, -6200.0, 180.0],
                        ],
                    }
                },
            }
        )

    payload: Dict[str, Any] = {
        "meta": {
            "coord_system": "UE5_cm",
            "num_frames": len(frames),
            "fps": 30.0,
            "has_ground_truth": has_ground_truth,
            "depth_model_used": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
            "origin_center_cm": [-5000.0, -1000.0, 500.0],
            "bounds_cm": {
                "x_min": -48000.0,
                "x_max": 36000.0,
                "y_min": -26000.0,
                "y_max": 18000.0,
                "z_min": 250.0,
                "z_max": 8000.0,
            },
        },
        "metrics": (
            {
                "mae_3d": 1730.5,
                "rmse_3d": 2004.9,
                "mae_x": 980.1,
                "mae_y": 1120.4,
                "mae_z": 145.2,
                "fps": 420.0,
                "runtime_ms": 2380.0,
            }
            if has_ground_truth
            else {}
        ),
        "terrain_mesh": {
            "rows": rows,
            "cols": cols,
            "x_grid_cm": x_grid,
            "y_grid_cm": y_grid,
            "z_matrix_cm": z_matrix,
            "rgb_matrix": rgb_matrix,
        },
        "point_cloud": {
            "points_cm": [
                [-12400.5, -4300.2, 412.8],
                [-10200.0, -3100.0, 520.0],
            ],
            "colors_rgb": [
                [118, 132, 96],
                [102, 124, 88],
            ],
        },
        "cameras": cameras,
        "frames": frames,
    }
    return payload


def _extract_embedded_json(html_text: str) -> Dict[str, Any]:
    match = re.search(r"window\.SCENE_DATA\s*=\s*(\{.*?\});\s*</script>", html_text, re.DOTALL)
    assert match is not None, "Could not locate embedded window.SCENE_DATA JSON in HTML"
    return json.loads(match.group(1))


def test_export_3d_viewer_html_validation_mode(tmp_path):
    """Verifies export_3d_viewer_html with Ground Truth present (Validation Mode)."""
    payload = _make_mock_scene_payload(has_ground_truth=True)
    out_file = tmp_path / "subdir" / "peleng_3d_viewer.html"

    returned_path = export_3d_viewer_html(payload, str(out_file))
    assert returned_path == str(out_file)
    assert out_file.exists()

    html = out_file.read_text(encoding="utf-8")
    assert len(html) > 5000

    # Check Three.js 0.160.0 CDN importmap & WebGL anti-Z-fighting configuration
    assert "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js" in html
    assert "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/" in html
    assert "logarithmicDepthBuffer: true" in html
    assert "computeVertexNormals()" in html
    assert "SM_AA_track" in html

    # Verify top header bar (#hud-header) is removed and collapsible HUD controls exist
    assert "hud-header" not in html
    assert "btn-collapse-left" in html
    assert "btn-open-left" in html
    assert "btn-toggle-left" in html
    assert "btn-collapse-right" in html
    assert "btn-open-right" in html
    assert "btn-toggle-right" in html
    assert "btn-toggle-ui" in html
    assert "Камеры и Слои" in html
    assert "Телеметрия" in html
    assert "Скрыть панели (H)" in html
    assert "KeyH" in html

    # Verify embedded JSON round-trip integrity
    embedded = _extract_embedded_json(html)
    assert embedded["meta"]["has_ground_truth"] is True
    assert embedded["meta"]["num_frames"] == 5
    assert embedded["metrics"]["mae_3d"] == pytest.approx(1730.5)
    assert len(embedded["cameras"]) == 5
    assert embedded["cameras"]["5"]["is_dynamic"] is True
    assert len(embedded["frames"]) == 5
    assert embedded["frames"][0]["gt_pos_cm"] is not None


def test_export_3d_viewer_html_blind_test_mode(tmp_path):
    """Verifies export_3d_viewer_html when Ground Truth is absent (Blind Test Mode)."""
    payload = _make_mock_scene_payload(has_ground_truth=False)
    out_file = tmp_path / "blind_viewer.html"

    returned_path = export_3d_viewer_html(payload, str(out_file))
    assert os.path.exists(returned_path)

    html = out_file.read_text(encoding="utf-8")
    embedded = _extract_embedded_json(html)

    assert embedded["meta"]["has_ground_truth"] is False
    assert embedded["metrics"] == {}
    for frame in embedded["frames"]:
        assert frame["gt_pos_cm"] is None
        assert frame["error_cm"] is None


def test_export_3d_viewer_html_numpy_and_nan_sanitization(tmp_path):
    """Verifies that NumPy arrays, np.float32, and NaN/Inf values serialize cleanly."""
    payload = {
        "meta": {
            "coord_system": "UE5_cm",
            "origin_center_cm": np.array([-5000.0, -1000.0, 500.0], dtype=np.float32),
            "has_ground_truth": np.bool_(False),
        },
        "metrics": {
            "mae_3d": np.float64("nan"),
            "rmse_3d": np.float64("inf"),
        },
        "point_cloud": {
            "points_cm": np.array([[-1000.0, 2000.0, 350.0]], dtype=np.float32),
            "colors_rgb": np.array([[100, 150, 200]], dtype=np.uint8),
        },
        "frames": [
            {
                "frame": np.int64(0),
                "est_pos_cm": np.array([10.0, 20.0, 30.0]),
                "gt_pos_cm": None,
                "heading_rad": np.float32(0.5),
                "error_cm": None,
                "active_rays": [],
                "dynamic_cameras": {},
            }
        ],
    }
    out_file = tmp_path / "sanitized_viewer.html"
    export_3d_viewer_html(payload, str(out_file))
    embedded = _extract_embedded_json(out_file.read_text(encoding="utf-8"))

    assert embedded["meta"]["origin_center_cm"] == [-5000.0, -1000.0, 500.0]
    assert embedded["metrics"]["mae_3d"] == 0.0
    assert embedded["metrics"]["rmse_3d"] == 0.0
    assert embedded["point_cloud"]["points_cm"] == [[-1000.0, 2000.0, 350.0]]
    assert embedded["frames"][0]["frame"] == 0


def test_export_3d_viewer_html_rejects_non_dict(tmp_path):
    with pytest.raises(TypeError):
        export_3d_viewer_html(["invalid_payload"], str(tmp_path / "fail.html"))  # type: ignore[arg-type]


def test_main_cli_flags_and_pipeline_signature(monkeypatch):
    """Verifies that main.py exposes the required CLI flags and run_pipeline parameters."""
    import inspect
    import main as peleng_main

    sig = inspect.signature(peleng_main.run_pipeline)
    assert "viewer_3d" in sig.parameters
    assert sig.parameters["viewer_3d"].default is True
    assert "viewer_out" in sig.parameters
    assert sig.parameters["viewer_out"].default == "results/peleng_3d_viewer.html"
    assert "use_depth_model" in sig.parameters
    assert sig.parameters["use_depth_model"].default is True

    captured_kwargs = {}

    def _fake_run_pipeline(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(peleng_main, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--no-viewer-3d",
            "--viewer-out",
            "results/custom_3d.html",
            "--no-depth-model",
        ],
    )
    peleng_main.main()
    assert captured_kwargs["viewer_3d"] is False
    assert captured_kwargs["viewer_out"] == "results/custom_3d.html"
    assert captured_kwargs["use_depth_model"] is False

