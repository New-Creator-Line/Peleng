"""
Unit Tests for Task 016: Metric Depth Unprojection & Multi-View Fusion.
Conforms to `.memory_bank/contracts/terrain_method_contract.md`.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pytest

from src.camera_model import CameraModel, load_all_cameras
from src.terrain.metric_depth import (
    DEFAULT_BOUNDS,
    DEFAULT_DEPTH_SHAPE,
    DepthEngine,
    MetricDepthUnprojector,
    ScaleAligner,
    reconstruct_terrain,
    to_scene_payload_layer,
)
from src.terrain.metric_depth.depth_engine import (
    compute_geometric_disparity_depth,
    compute_geometric_ground_depth,
    estimate_sky_mask,
    generate_procedural_frame,
)
from src.terrain.metric_depth.scale_aligner import solve_affine_scale_shift
from src.terrain.metric_depth.unprojector import (
    filter_points_by_bounds,
    fuse_voxel_grid,
    unproject_depth_map,
)


@pytest.fixture
def synthetic_camera() -> CameraModel:
    """Fixture providing a standard synthetic camera."""
    return CameraModel(
        cam_id=1,
        loc=np.array([-20000.0, 5000.0, 1000.0]),
        rot=np.array([0.0, -5.0, -45.0]),
        focal_length_mm=35.0,
        sensor_width_mm=23.76,
        sensor_height_mm=13.365,
        image_width_px=1920,
        image_height_px=1080,
    )


def test_generate_procedural_frame():
    """Verifies procedural frame generator outputs valid RGB uint8 images."""
    frame = generate_procedural_frame(cam_id=1, width=320, height=180)
    assert frame.shape == (180, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.min() >= 0
    assert frame.max() <= 255
    # Upper part (sky) has higher blue than lower part (terrain)
    assert frame[:30, :, 2].mean() > frame[-30:, :, 2].mean()


def test_estimate_sky_mask():
    """Verifies sky detection identifies upper sky and preserves ground."""
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    # Bright blue sky in top 30 rows
    img[:30, :, :] = [180, 200, 250]
    # Brown earth in bottom 150 rows
    img[30:, :, :] = [100, 80, 60]

    sky_mask = estimate_sky_mask(img)
    assert sky_mask.shape == (180, 320)
    assert sky_mask.dtype == bool
    assert np.all(sky_mask[:30, :])
    assert not np.any(sky_mask[30:, :])


def test_geometric_ground_depth(synthetic_camera):
    """Verifies geometric ray-ground intersection produces non-negative depths."""
    depth, valid_mask = compute_geometric_ground_depth(
        synthetic_camera, shape=(180, 320), nominal_ground_z=400.0
    )
    assert depth.shape == (180, 320)
    assert valid_mask.shape == (180, 320)
    assert np.all(depth >= 0.0)

    # Valid points must have positive depth
    valid_depths = depth[valid_mask]
    assert len(valid_depths) > 0
    assert np.all(valid_depths > 10.0)


def test_solve_affine_scale_shift():
    """Verifies affine solver accurately recovers known scale and shift."""
    # Synthetic ground truth
    x = np.linspace(10.0, 50.0, 20)
    true_s = 3.5
    true_t = 120.0
    y = true_s * x + true_t

    s_est, t_est, rmse = solve_affine_scale_shift(x, y)
    assert pytest.approx(s_est, rel=1e-4) == true_s
    assert pytest.approx(t_est, rel=1e-4) == true_t
    assert rmse < 1e-4

    # Single point fallback
    s_single, t_single, _ = solve_affine_scale_shift(np.array([10.0]), np.array([50.0]))
    assert pytest.approx(s_single, rel=1e-4) == 5.0
    assert t_single == 0.0


def test_scale_aligner_inter_camera_baseline():
    """Verifies mutual visibility between Cam 3 and Cam 5 produces baseline anchor."""
    if not os.path.isdir("хакатон"):
        pytest.skip("хакатон dataset not available")

    cams = load_all_cameras("хакатон")
    cam_models = {cid: cams[cid].get_model(0) for cid in cams}

    aligner = ScaleAligner(calib_dir="хакатон")
    anchors_c3 = aligner.get_inter_camera_anchors(
        cam_id=3,
        camera_model=cam_models[3],
        all_cameras=cam_models,
        depth_shape=(180, 320),
    )

    # Cam 3 must observe Cam 5
    cam5_anchors = [a for a in anchors_c3 if "5" in a.source]
    assert len(cam5_anchors) >= 1
    # Known ground truth optical depth is ~18180.0 cm
    assert pytest.approx(cam5_anchors[0].target_metric_depth, abs=100.0) == 18180.0


def test_unprojector_pinhole_inversion_roundtrip(synthetic_camera):
    """
    Verifies that unprojecting pixel (u, v) with depth D and projecting back
    recovers the exact original (u, v, D) to machine precision.
    """
    h_d, w_d = 180, 320
    test_u, test_v, test_d = 160.0, 120.0, 15000.0

    depth_map = np.full((h_d, w_d), test_d, dtype=np.float64)
    rgb_frame = np.full((h_d, w_d, 3), 100, dtype=np.uint8)

    pts, cols = unproject_depth_map(depth_map, synthetic_camera, rgb_frame)
    # Locate point corresponding to (test_v, test_u)
    idx = int(test_v) * w_d + int(test_u)
    pt_world = pts[idx]

    # Reproject using camera model
    u_proj, v_proj, z_proj = synthetic_camera.project_point(pt_world)
    u_proj_scaled = u_proj * (w_d / float(synthetic_camera.image_width_px))
    v_proj_scaled = v_proj * (h_d / float(synthetic_camera.image_height_px))

    assert pytest.approx(u_proj_scaled, abs=1e-3) == test_u
    assert pytest.approx(v_proj_scaled, abs=1e-3) == test_v
    assert pytest.approx(z_proj, abs=1e-3) == test_d


def test_filter_points_by_bounds():
    """Verifies that spatial filtering strictly removes points outside the envelope."""
    pts = np.array([
        [-10000.0, 0.0, 500.0],     # Inside
        [-50000.0, 0.0, 500.0],     # Outside X min
        [30000.0, 0.0, 500.0],      # Outside X max
        [0.0, -35000.0, 500.0],     # Outside Y min
        [0.0, 35000.0, 500.0],      # Outside Y max
        [0.0, 0.0, -100.0],         # Outside Z min
        [0.0, 0.0, 6000.0],         # Outside Z max
    ])
    cols = np.full((len(pts), 3), 200, dtype=np.uint8)

    filtered_pts, filtered_cols = filter_points_by_bounds(pts, cols, DEFAULT_BOUNDS)
    assert len(filtered_pts) == 1
    assert np.allclose(filtered_pts[0], [-10000.0, 0.0, 500.0])
    assert len(filtered_cols) == 1


def test_fuse_voxel_grid():
    """Verifies voxel fusion merges points into centroids within cell size."""
    # Two points in the same 100cm voxel: (120, 210, 410) and (180, 290, 490) -> (150, 250, 450)
    pts = np.array([
        [120.0, 210.0, 410.0],
        [180.0, 290.0, 490.0],
        [500.0, 500.0, 500.0],  # Different voxel
    ])
    cols = np.array([
        [100, 100, 100],
        [200, 200, 200],
        [50, 50, 50],
    ], dtype=np.uint8)

    fused_pts, fused_cols = fuse_voxel_grid(pts, cols, voxel_size_cm=100.0)
    assert len(fused_pts) == 2

    # First voxel centroid
    assert np.allclose(fused_pts[0], [150.0, 250.0, 450.0])
    # Blended color (150, 150, 150)
    assert np.allclose(fused_cols[0], [150, 150, 150])


def test_reconstruct_terrain_contract_compliance():
    """Verifies reconstruct_terrain satisfies Frozen Contract schema exactly."""
    payload = reconstruct_terrain(calib_dir="хакатон", video_dir="хакатон", frame_idx=0)

    # Required keys
    assert payload["method_name"] == "metric_depth"
    assert isinstance(payload["vertices"], np.ndarray)
    assert payload["vertices"].ndim == 2 and payload["vertices"].shape[1] == 3
    assert payload["vertices"].dtype in (np.float32, np.float64)

    assert payload["faces"] is None
    assert isinstance(payload["colors"], np.ndarray)
    assert payload["colors"].shape == payload["vertices"].shape
    assert payload["colors"].dtype == np.uint8

    assert isinstance(payload["point_cloud"], np.ndarray)
    assert payload["point_cloud"].shape == payload["vertices"].shape
    assert payload["density"] == "dense_surface"

    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert stats["vertex_count"] == len(payload["vertices"])
    assert stats["face_count"] == 0
    assert stats["runtime_ms"] > 0.0
    assert "z_min_cm" in stats
    assert "z_max_cm" in stats


def test_reconstruct_terrain_qa_gates():
    """QA Acceptance Gate: Point count > 5,000, bounds within envelope, runtime < 2.5s."""
    payload = reconstruct_terrain(calib_dir="хакатон", video_dir="хакатон", frame_idx=0)

    # 1. Point count gate (> 5,000)
    v_count = payload["stats"]["vertex_count"]
    assert v_count > 5000, f"Expected > 5,000 points, got {v_count}"

    # 2. Bounding envelope gate
    vertices = payload["vertices"]
    assert vertices[:, 0].min() >= DEFAULT_BOUNDS["x_min"]
    assert vertices[:, 0].max() <= DEFAULT_BOUNDS["x_max"]
    assert vertices[:, 1].min() >= DEFAULT_BOUNDS["y_min"]
    assert vertices[:, 1].max() <= DEFAULT_BOUNDS["y_max"]
    assert vertices[:, 2].min() >= DEFAULT_BOUNDS["z_min"]
    assert vertices[:, 2].max() <= DEFAULT_BOUNDS["z_max"]

    # 3. Performance gate (< 2.5s CPU target)
    runtime = payload["stats"]["runtime_ms"]
    assert runtime < 2500.0, f"Expected runtime < 2500 ms, got {runtime} ms"


def test_to_scene_payload_layer():
    """Verifies integration adapter formats data correctly for viewer_3d.py."""
    payload = reconstruct_terrain(calib_dir="хакатон", video_dir="хакатон", frame_idx=0)
    layer = to_scene_payload_layer(payload)

    assert "point_cloud" in layer
    assert "points_cm" in layer["point_cloud"]
    assert "colors_rgb" in layer["point_cloud"]
    assert len(layer["point_cloud"]["points_cm"]) == payload["stats"]["vertex_count"]
    assert layer["method_name"] == "metric_depth"


def test_reconstruct_terrain_missing_inputs_fallback():
    """Verifies graceful handling of nonexistent directories using synthetic fallbacks."""
    payload = reconstruct_terrain(
        calib_dir="nonexistent_calib_dir",
        video_dir="nonexistent_video_dir",
        frame_idx=0,
    )
    assert payload["method_name"] == "metric_depth"
    assert payload["stats"]["vertex_count"] > 1000
    assert payload["vertices"].shape[1] == 3
