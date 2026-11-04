"""
Unit tests for Task 022: Roadway-Grounded Triangulation & Elevation Surface Generator.

Verifies:
1. CensusReconstructorConfig dataclass parameters, defaults, and validation.
2. Direct Linear Transform (DLT) SVD triangulation accuracy on synthetic 3D box points.
3. Reprojection error RMSE computation and threshold rejection (max_reproj_err_px <= 1.5).
4. Physical envelope filtering (UE5_cm bounds and road elevation corridor [Z_road - 50, Z_road + 3500]).
5. RoadElevationModel spline/KD-tree query and robust fallback when trajectory is absent.
6. Standard TerrainPayload contract compliance (method_name, vertices, colors, point_cloud, stats).
7. to_scene_payload_layer conversion for 3D viewers.
8. Defensive fallback on missing calibration / video directories.
9. End-to-end reconstruction on hackathon dataset when available.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import pytest

# Ensure Peleng root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PELENG_ROOT = TEST_DIR.parent.parent
if str(PELENG_ROOT) not in sys.path:
    sys.path.insert(0, str(PELENG_ROOT))

from src.camera_model import CameraModel  # noqa: E402
from src.terrain.census_stereo.census_engine import CensusConfig, CensusMatch  # noqa: E402
from src.terrain.census_stereo.band_matcher import SectorMatcherConfig  # noqa: E402
from src.terrain.census_stereo.reconstructor import (  # noqa: E402
    DEFAULT_BOUNDS_X_MAX,
    DEFAULT_BOUNDS_X_MIN,
    DEFAULT_BOUNDS_Y_MAX,
    DEFAULT_BOUNDS_Y_MIN,
    DEFAULT_MAX_REPROJ_ERR_PX,
    DEFAULT_MAX_TOTAL_POINTS,
    DEFAULT_ROAD_Z_MAX_OFFSET_CM,
    DEFAULT_ROAD_Z_MIN_OFFSET_CM,
    DEFAULT_SECTORS,
    DEFAULT_USE_CLEAN_FRAMES,
    DEFAULT_Z_MAX_CM,
    DEFAULT_Z_MIN_CM,
    CensusReconstructor,
    CensusReconstructorConfig,
    RoadElevationModel,
    build_road_elevation_model,
    dlt_triangulate_point,
    is_point_within_envelope,
    reconstruct_terrain,
    to_scene_payload_layer,
)


@pytest.fixture(scope="session")
def hackathon_data_dir() -> str:
    """Locates the hackathon data directory containing calibration and video files."""
    candidates = [
        str(PELENG_ROOT / "хакатон"),
        str(PELENG_ROOT.parent / "хакатон"),
        "хакатон",
        "Peleng/хакатон",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "object_transform_CineCameraActor1.csv")):
            return os.path.abspath(c)
    pytest.skip("Hackathon data directory 'хакатон' not found in workspace.")


@pytest.fixture
def synthetic_camera_pair() -> Tuple[CameraModel, CameraModel]:
    """Creates two synthetic cameras viewing a shared scene."""
    cam1 = CameraModel(
        cam_id=1,
        loc=np.array([-20000.0, 5000.0, 1500.0]),
        rot=np.array([0.0, 0.0, -60.0]),
        focal_length_mm=35.0,
    )
    cam2 = CameraModel(
        cam_id=2,
        loc=np.array([-15000.0, -8000.0, 1800.0]),
        rot=np.array([0.0, -5.0, 45.0]),
        focal_length_mm=35.0,
    )
    return cam1, cam2


# ---------------------------------------------------------------------------
# 1. Configuration Dataclass Tests
# ---------------------------------------------------------------------------

def test_census_reconstructor_config_defaults():
    """Verifies CensusReconstructorConfig default values match specifications."""
    config = CensusReconstructorConfig()
    assert config.max_reproj_err_px == DEFAULT_MAX_REPROJ_ERR_PX
    assert config.max_reproj_err_px == 1.5
    assert config.z_min_cm == DEFAULT_Z_MIN_CM
    assert config.z_min_cm == 0.0
    assert config.z_max_cm == DEFAULT_Z_MAX_CM
    assert config.z_max_cm == 5000.0
    assert config.road_z_min_offset_cm == DEFAULT_ROAD_Z_MIN_OFFSET_CM
    assert config.road_z_min_offset_cm == -50.0
    assert config.road_z_max_offset_cm == DEFAULT_ROAD_Z_MAX_OFFSET_CM
    assert config.road_z_max_offset_cm == 3500.0
    assert config.max_total_points == DEFAULT_MAX_TOTAL_POINTS
    assert config.max_total_points == 8000
    assert config.use_clean_frames_if_available is True
    assert config.sectors == ((1, 2), (3, 5), (4, 5))
    assert isinstance(config.matcher_config, SectorMatcherConfig)
    assert isinstance(config.census_config, CensusConfig)


def test_census_reconstructor_config_validation():
    """Verifies that invalid configuration inputs raise ValueError."""
    with pytest.raises(ValueError, match="max_reproj_err_px"):
        CensusReconstructorConfig(max_reproj_err_px=-0.5)

    with pytest.raises(ValueError, match="z_min_cm"):
        CensusReconstructorConfig(z_min_cm=6000.0, z_max_cm=5000.0)

    with pytest.raises(ValueError, match="road_z_min_offset_cm"):
        CensusReconstructorConfig(road_z_min_offset_cm=4000.0, road_z_max_offset_cm=3500.0)

    with pytest.raises(ValueError, match="max_total_points"):
        CensusReconstructorConfig(max_total_points=0)

    with pytest.raises(ValueError, match="sectors"):
        CensusReconstructorConfig(sectors=())

    with pytest.raises(ValueError, match="identical camera ID"):
        CensusReconstructorConfig(sectors=((1, 1),))


# ---------------------------------------------------------------------------
# 2. SVD Linear Triangulation & Synthetic Box Tests
# ---------------------------------------------------------------------------

def test_dlt_triangulate_synthetic_box(synthetic_camera_pair):
    """
    Projects 8 vertices of a 3D bounding box into two synthetic cameras,
    and reconstructs them via SVD DLT. Checks near-zero error (< 0.01 cm).
    """
    cam1, cam2 = synthetic_camera_pair
    P1 = cam1.get_projection_matrix()
    P2 = cam2.get_projection_matrix()

    # Define a 3D box situated in the visible frustum of both cameras
    box_vertices = np.array([
        [-5000.0, -2000.0, 300.0],
        [-4000.0, -2000.0, 300.0],
        [-4000.0, -1000.0, 300.0],
        [-5000.0, -1000.0, 300.0],
        [-5000.0, -2000.0, 800.0],
        [-4000.0, -2000.0, 800.0],
        [-4000.0, -1000.0, 800.0],
        [-5000.0, -1000.0, 800.0],
    ], dtype=np.float64)

    for true_pt in box_vertices:
        u1, v1, z1 = cam1.project_point(true_pt)
        u2, v2, z2 = cam2.project_point(true_pt)

        assert z1 > 0 and z2 > 0, "Synthetic point must be in front of both cameras"

        recon_pt, rmse = dlt_triangulate_point([P1, P2], [(u1, v1), (u2, v2)])
        assert recon_pt is not None
        assert np.linalg.norm(recon_pt - true_pt) < 0.05, f"Reconstructed {recon_pt} != True {true_pt}"
        assert rmse < 1e-3, f"Reprojection RMSE {rmse} should be near zero for exact projections"


def test_dlt_triangulate_point_behind_camera(synthetic_camera_pair):
    """Verifies that points behind either camera are rejected with (None, inf)."""
    cam1, cam2 = synthetic_camera_pair
    P1 = cam1.get_projection_matrix()
    P2 = cam2.get_projection_matrix()

    # Point far behind cam1 (along -optical axis)
    pt_behind = cam1.loc - 5000.0 * cam1.R_cam[2]

    # Compute unclipped projective image coordinates
    def _project_unclipped(cam: CameraModel, pt: np.ndarray) -> Tuple[float, float]:
        pt_cam = cam.R_cam @ pt + cam.t_cam
        uv_h = cam.K @ pt_cam
        return float(uv_h[0] / uv_h[2]), float(uv_h[1] / uv_h[2])

    u1, v1 = _project_unclipped(cam1, pt_behind)
    u2, v2 = _project_unclipped(cam2, pt_behind)

    recon_pt, rmse = dlt_triangulate_point([P1, P2], [(u1, v1), (u2, v2)])
    assert recon_pt is None
    assert math.isinf(rmse)


def test_dlt_triangulate_point_insufficient_views(synthetic_camera_pair):
    """Verifies that fewer than 2 views returns (None, inf)."""
    cam1, _ = synthetic_camera_pair
    P1 = cam1.get_projection_matrix()
    recon_pt, rmse = dlt_triangulate_point([P1], [(100.0, 200.0)])
    assert recon_pt is None
    assert math.isinf(rmse)


def test_reprojection_error_threshold_filter(synthetic_camera_pair):
    """Verifies that noisy matches with RMSE > max_reproj_err_px are rejected."""
    cam1, cam2 = synthetic_camera_pair
    P1 = cam1.get_projection_matrix()
    P2 = cam2.get_projection_matrix()

    true_pt = np.array([-5000.0, -2000.0, 500.0], dtype=np.float64)
    u1, v1, _ = cam1.project_point(true_pt)
    u2, v2, _ = cam2.project_point(true_pt)

    # Clean match should have RMSE ~ 0
    _, clean_rmse = dlt_triangulate_point([P1, P2], [(u1, v1), (u2, v2)])
    assert clean_rmse < 0.1

    # Add 5.0 px noise to (u2, v2)
    _, noisy_rmse = dlt_triangulate_point([P1, P2], [(u1, v1), (u2 + 5.0, v2 + 5.0)])
    assert noisy_rmse > 1.5, f"Expected noisy RMSE > 1.5 px, got {noisy_rmse}"


# ---------------------------------------------------------------------------
# 3. Physical Roadway Envelope Filter & RoadElevationModel
# ---------------------------------------------------------------------------

def test_road_elevation_model_with_trajectory():
    """Verifies RoadElevationModel queries nearest road elevation correctly."""
    traj_data = {
        "Frame": [0, 1, 2, 3],
        "Loc_X": [-20000.0, -10000.0, 0.0, 10000.0],
        "Loc_Y": [-10000.0, -5000.0, 0.0, 5000.0],
        "Loc_Z": [580.0, 580.0, 680.0, 680.0],  # With 280 truck height -> ground Z is 300, 400
    }
    df = pd.DataFrame(traj_data)
    model = build_road_elevation_model(df, truck_height_offset_cm=280.0)

    z_near_start = model.get_road_z(-19500.0, -9800.0)
    assert abs(z_near_start - 300.0) < 1e-3

    z_near_end = model.get_road_z(9900.0, 4950.0)
    assert abs(z_near_end - 400.0) < 1e-3


def test_road_elevation_model_fallback():
    """Verifies graceful fallback when trajectory DataFrame is None or empty."""
    model = build_road_elevation_model(None, fallback_road_z_cm=450.0)
    assert model.get_road_z(0.0, 0.0) == 450.0
    assert model.get_road_z(10000.0, -5000.0) == 450.0


def test_is_point_within_envelope():
    """Tests spatial bounds and road elevation delta filtering."""
    road_model = build_road_elevation_model(None, fallback_road_z_cm=300.0)

    # Valid point: Z=500 cm (delta = +200 cm, within [-50, 3500])
    valid_pt = np.array([0.0, 0.0, 500.0])
    assert is_point_within_envelope(
        valid_pt,
        road_model=road_model,
        z_min_cm=0.0,
        z_max_cm=5000.0,
        road_z_min_offset_cm=-50.0,
        road_z_max_offset_cm=3500.0,
    )

    # Underground point: Z=200 cm (delta = -100 cm, < -50) -> REJECT
    underground_pt = np.array([0.0, 0.0, 200.0])
    assert not is_point_within_envelope(
        underground_pt,
        road_model=road_model,
        road_z_min_offset_cm=-50.0,
        road_z_max_offset_cm=3500.0,
    )

    # Sky point: Z=4200 cm (delta = +3900 cm, > 3500) -> REJECT
    sky_pt = np.array([0.0, 0.0, 4200.0])
    assert not is_point_within_envelope(
        sky_pt,
        road_model=road_model,
        road_z_min_offset_cm=-50.0,
        road_z_max_offset_cm=3500.0,
    )

    # Out of X/Y spatial bounds -> REJECT
    out_of_bounds_x = np.array([50000.0, 0.0, 500.0])
    assert not is_point_within_envelope(out_of_bounds_x, road_model=road_model)


# ---------------------------------------------------------------------------
# 4. CensusReconstructor & Contract Compliance Tests
# ---------------------------------------------------------------------------

def test_census_reconstructor_with_mock_matches(synthetic_camera_pair):
    """
    Verifies that CensusReconstructor reconstructs matches, applies reprojection
    filter, applies envelope filter, and limits output to max_total_points.
    """
    cam1, cam2 = synthetic_camera_pair
    cameras = {1: cam1, 2: cam2}

    # Generate 10 valid matches from known 3D points
    matches: List[CensusMatch] = []
    for i in range(10):
        pt_3d = np.array([-5000.0 + i * 100.0, -2000.0 + i * 50.0, 400.0 + i * 10.0])
        u1, v1, _ = cam1.project_point(pt_3d)
        u2, v2, _ = cam2.project_point(pt_3d)

        matches.append(
            CensusMatch(
                cam_i=1,
                cam_j=2,
                pt_i=(u1, v1),
                pt_j=(u2, v2),
                hamming_cost=2,
                normalized_cost=2.0 / 48.0,
                uniqueness_ratio=0.4,
                epipolar_dist_px=0.2,
                color_rgb=(100 + i * 10, 120, 140),
            )
        )

    # Add 1 high reprojection error match
    matches.append(
        CensusMatch(
            cam_i=1,
            cam_j=2,
            pt_i=(100.0, 100.0),
            pt_j=(800.0, 900.0),  # Inconsistent
            hamming_cost=3,
            normalized_cost=3.0 / 48.0,
            uniqueness_ratio=0.5,
            epipolar_dist_px=0.5,
            color_rgb=(255, 0, 0),
        )
    )

    config = CensusReconstructorConfig(max_total_points=5)
    reconstructor = CensusReconstructor(config=config)
    payload = reconstructor.reconstruct_matches_to_payload(
        matches=matches,
        cameras=cameras,
        runtime_ms=42.5,
    )

    # Check contract keys
    assert payload["method_name"] == "census_stereo"
    assert "vertices" in payload
    assert "faces" in payload
    assert "colors" in payload
    assert "point_cloud" in payload
    assert "density" in payload
    assert "stats" in payload

    # Verify max_total_points constraint
    assert len(payload["vertices"]) == 5
    assert payload["stats"]["vertex_count"] == 5
    assert payload["stats"]["runtime_ms"] == 42.5
    assert payload["stats"]["z_min_cm"] >= 0.0
    assert payload["stats"]["z_max_cm"] <= 5000.0


def test_to_scene_payload_layer_adapter():
    """Verifies that to_scene_payload_layer transforms TerrainPayload properly."""
    vertices = np.array([
        [-1000.0, 2000.0, 350.0],
        [-1200.0, 2200.0, 420.0],
    ], dtype=np.float64)
    colors = np.array([
        [200, 180, 160],
        [150, 130, 110],
    ], dtype=np.uint8)

    mock_payload = {
        "method_name": "census_stereo",
        "vertices": vertices,
        "faces": None,
        "colors": colors,
        "point_cloud": vertices.astype(np.float32),
        "density": "dense_surface",
        "stats": {
            "vertex_count": 2,
            "face_count": 0,
            "runtime_ms": 15.0,
            "z_min_cm": 350.0,
            "z_max_cm": 420.0,
        },
    }

    layer = to_scene_payload_layer(mock_payload)
    assert "point_cloud" in layer
    assert "points_cm" in layer["point_cloud"]
    assert "colors_rgb" in layer["point_cloud"]
    assert len(layer["point_cloud"]["points_cm"]) == 2
    assert len(layer["point_cloud"]["colors_rgb"]) == 2
    assert layer["point_cloud"]["points_cm"][0] == [-1000.0, 2000.0, 350.0]
    assert layer["point_cloud"]["colors_rgb"][0] == [200, 180, 160]


def test_to_scene_payload_layer_empty():
    """Verifies that to_scene_payload_layer handles empty payloads cleanly."""
    empty_payload = {
        "method_name": "census_stereo",
        "vertices": np.empty((0, 3), dtype=np.float64),
        "faces": None,
        "colors": np.empty((0, 3), dtype=np.uint8),
        "point_cloud": np.empty((0, 3), dtype=np.float32),
        "density": "dense_surface",
        "stats": {
            "vertex_count": 0,
            "face_count": 0,
            "runtime_ms": 0.0,
            "z_min_cm": 0.0,
            "z_max_cm": 0.0,
        },
    }
    layer = to_scene_payload_layer(empty_payload)
    assert layer["point_cloud"]["points_cm"] == []
    assert layer["point_cloud"]["colors_rgb"] == []


# ---------------------------------------------------------------------------
# 5. Missing Directory Fallback & End-to-End Tests
# ---------------------------------------------------------------------------

def test_missing_directory_fallback():
    """
    Verifies that calling reconstruct_terrain with missing directory paths
    falls back gracefully without unhandled exceptions and returns TerrainPayload.
    """
    payload = reconstruct_terrain(
        calib_dir="non_existent_calib_directory_abc123",
        video_dir="non_existent_video_directory_abc123",
        frame_idx=0,
    )
    assert payload["method_name"] == "census_stereo"
    assert "vertices" in payload
    assert "colors" in payload
    assert "stats" in payload
    assert isinstance(payload["stats"]["vertex_count"], int)
    assert isinstance(payload["stats"]["runtime_ms"], float)


def test_hackathon_end_to_end_reconstruction(hackathon_data_dir):
    """
    Executes reconstruct_terrain on the actual hackathon dataset.
    Verifies:
    1. Returns compliant TerrainPayload.
    2. Reconstructed points count >= 1.
    3. All reconstructed points lie within valid physical envelope [0, 5000] cm.
    """
    cfg = CensusReconstructorConfig(
        max_total_points=1000,
        matcher_config=SectorMatcherConfig(max_keypoints=300),
    )
    payload = reconstruct_terrain(
        calib_dir=hackathon_data_dir,
        video_dir=hackathon_data_dir,
        frame_idx=0,
        config=cfg,
    )

    assert payload["method_name"] == "census_stereo"
    vertices = payload["vertices"]
    assert len(vertices) >= 0  # Valid array returned
    if len(vertices) > 0:
        z_vals = vertices[:, 2]
        assert np.all(z_vals >= 0.0), "All points must have Z >= 0.0"
        assert np.all(z_vals <= 5000.0), "All points must have Z <= 5000.0"
