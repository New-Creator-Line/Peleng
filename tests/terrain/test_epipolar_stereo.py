"""
Unit tests for Task 014: Method 1 — Epipolar Feature Stereo & Multi-View Triangulation.

Verifies:
1. Pairwise Fundamental Matrix calculation F_ij = K_j^{-T} [t_ij]_\\times R_ij K_i^{-1}
   and epipolar constraint x_j^T F_ij x_i = 0 for true projected points.
2. Keypoint extraction and normalized patch precomputation (9x9).
3. Epipolar line verification and NCC matching (NCC >= 0.80).
4. Direct Linear Transform (DLT) SVD triangulation accuracy and reprojection filtering (<= 3.5 px).
5. Peleng spatial bounding envelope filtering:
   X in [-40000, 20000], Y in [-30000, 30000], Z in [0, 5000] cm.
6. Full pipeline compliance with TerrainPayload schema in terrain_method_contract.md.
7. Acceptance Criteria (QA Gate):
   - Reconstructed points count >= 200.
   - All points inside Peleng bounding box.
   - Total runtime < 2.0s.
8. Integration adapter to_scene_payload_layer.
9. Defensive fallback on missing files / synthetic environments.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pytest

# Ensure Peleng root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PELENG_ROOT = TEST_DIR.parent.parent
if str(PELENG_ROOT) not in sys.path:
    sys.path.insert(0, str(PELENG_ROOT))

from src.camera_model import CameraModel, load_all_cameras
from src.terrain.epipolar_stereo import (
    BOUNDS_X_MAX,
    BOUNDS_X_MIN,
    BOUNDS_Y_MAX,
    BOUNDS_Y_MIN,
    BOUNDS_Z_MAX,
    BOUNDS_Z_MIN,
    EpipolarMatcher,
    EpipolarReconstructor,
    compute_fundamental_matrix,
    dlt_triangulate_point,
    is_point_within_bounds,
    reconstruct_terrain,
    to_scene_payload_layer,
)
from src.terrain.epipolar_stereo.matcher import (
    compute_epipolar_line_distances,
    extract_features_and_patches,
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
def synthetic_camera_pair() -> tuple[CameraModel, CameraModel]:
    """Creates two synthetic cameras with known ground truth locations and orientations."""
    cam1 = CameraModel(
        cam_id=1,
        loc=np.array([-20000.0, 5000.0, 1200.0]),
        rot=np.array([0.0, 0.0, -60.0]),
        focal_length_mm=35.0,
    )
    cam2 = CameraModel(
        cam_id=2,
        loc=np.array([-15000.0, -8000.0, 1500.0]),
        rot=np.array([0.0, -5.0, 45.0]),
        focal_length_mm=35.0,
    )
    return cam1, cam2


# ---------------------------------------------------------------------------
# 1. Fundamental Matrix & Epipolar Geometry Tests
# ---------------------------------------------------------------------------

def test_fundamental_matrix_geometry(synthetic_camera_pair):
    """
    Verifies that Fundamental Matrix F_ij computed from camera extrinsics and intrinsics
    satisfies the exact epipolar constraint x_j^T @ F_ij @ x_i = 0 for 3D world points.
    """
    cam1, cam2 = synthetic_camera_pair
    F12 = compute_fundamental_matrix(cam1, cam2)

    assert F12.shape == (3, 3)
    assert np.isfinite(F12).all()
    # Fundamental matrix must have rank 2 (determinant zero within numerical precision)
    assert abs(float(np.linalg.det(F12))) < 1e-6

    # Test projection of several 3D points inside bounding box
    test_points_3d = [
        np.array([-10000.0, 0.0, 400.0]),
        np.array([-5000.0, -2000.0, 650.0]),
        np.array([0.0, 1000.0, 300.0]),
        np.array([5000.0, -5000.0, 1200.0]),
    ]

    for pt in test_points_3d:
        u1, v1, z1 = cam1.project_point(pt)
        u2, v2, z2 = cam2.project_point(pt)

        # Both projections must be in front of optical centers
        assert z1 > 0 and z2 > 0

        x1 = np.array([u1, v1, 1.0], dtype=np.float64)
        x2 = np.array([u2, v2, 1.0], dtype=np.float64)

        # Algebraic epipolar error
        alg_err = float(x2.T @ F12 @ x1)
        assert abs(alg_err) < 1e-10

        # Geometric perpendicular distance to epipolar line in image 2
        l2 = F12 @ x1
        line_norm = float(np.hypot(l2[0], l2[1]))
        assert line_norm > 1e-9
        dist_px = abs(l2[0] * u2 + l2[1] * v2 + l2[2]) / line_norm
        assert dist_px < 1e-8, f"Point projection distance to epipolar line {dist_px} px exceeds tolerance"


def test_epipolar_line_distances_vectorization(synthetic_camera_pair):
    """Verifies vectorized epipolar line distance matrix computation."""
    cam1, cam2 = synthetic_camera_pair
    F12 = compute_fundamental_matrix(cam1, cam2)

    # 3 true pairs + 1 perturbed pair
    pt1_3d = np.array([-10000.0, 0.0, 400.0])
    pt2_3d = np.array([-5000.0, -2000.0, 650.0])

    u1_a, v1_a, _ = cam1.project_point(pt1_3d)
    u2_a, v2_a, _ = cam2.project_point(pt1_3d)

    u1_b, v1_b, _ = cam1.project_point(pt2_3d)
    u2_b, v2_b, _ = cam2.project_point(pt2_3d)

    pts1_hom = np.array([[u1_a, v1_a, 1.0], [u1_b, v1_b, 1.0]], dtype=np.float64)
    pts2_hom = np.array([[u2_a, v2_a, 1.0], [u2_b, v2_b, 1.0]], dtype=np.float64)

    dist_mat = compute_epipolar_line_distances(F12, pts1_hom, pts2_hom)
    assert dist_mat.shape == (2, 2)

    # Diagonal elements must be virtually 0 (true correspondences)
    assert dist_mat[0, 0] < 1e-7
    assert dist_mat[1, 1] < 1e-7

    # Off-diagonal elements (mismatched pairs) must have positive distance
    assert dist_mat[0, 1] > 1.0
    assert dist_mat[1, 0] > 1.0


# ---------------------------------------------------------------------------
# 2. Keypoint & Patch Extraction Tests
# ---------------------------------------------------------------------------

def test_feature_and_patch_extraction():
    """Verifies FAST feature extraction and normalized 9x9 patch precomputation."""
    # Synthetic textured test pattern
    rng = np.random.RandomState(42)
    gray = np.zeros((300, 400), dtype=np.uint8)
    # Add high-contrast corners and edges
    gray[50:150, 50:150] = 220
    gray[180:260, 200:320] = 180
    gray[70:90, 70:90] = 40
    noise = rng.randint(0, 30, (300, 400), dtype=np.uint8)
    gray = np.clip(gray + noise, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    kps, pts_hom, patches_flat, stds, colors = extract_features_and_patches(
        gray,
        rgb_img=rgb,
        max_keypoints=50,
        patch_size=9,
        detector_type="fast",
        fast_threshold=10,
        min_contrast_std=3.0,
    )

    assert len(kps) > 0
    N = len(kps)
    assert pts_hom.shape == (N, 3)
    assert patches_flat.shape == (N, 81)
    assert stds.shape == (N,)
    assert colors.shape == (N, 3)

    # Patches must be approximately zero mean and unit variance
    for i in range(min(5, N)):
        if stds[i] > 1e-3:
            p = patches_flat[i]
            assert abs(float(np.mean(p))) < 1e-4
            assert abs(float(np.std(p)) - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# 3. DLT Triangulation & Filtering Tests
# ---------------------------------------------------------------------------

def test_dlt_triangulation_accuracy(synthetic_camera_pair):
    """Verifies DLT SVD triangulation recovers ground truth 3D points within 0.05 cm."""
    cam1, cam2 = synthetic_camera_pair
    P1 = cam1.get_projection_matrix()
    P2 = cam2.get_projection_matrix()

    gt_point_3d = np.array([-12450.0, -3200.0, 520.0], dtype=np.float64)
    u1, v1, _ = cam1.project_point(gt_point_3d)
    u2, v2, _ = cam2.project_point(gt_point_3d)

    reconstructed_3d, err = dlt_triangulate_point([P1, P2], [(u1, v1), (u2, v2)])
    assert reconstructed_3d is not None
    assert err < 0.01

    diff = np.linalg.norm(reconstructed_3d - gt_point_3d)
    assert diff < 0.05, f"Triangulation error {diff} cm exceeds 0.05 cm"


def test_dlt_triangulation_with_noise(synthetic_camera_pair):
    """Verifies DLT triangulation is robust to sub-pixel reprojection noise."""
    cam1, cam2 = synthetic_camera_pair
    P1 = cam1.get_projection_matrix()
    P2 = cam2.get_projection_matrix()

    gt_point_3d = np.array([-10000.0, 1500.0, 450.0])
    u1, v1, _ = cam1.project_point(gt_point_3d)
    u2, v2, _ = cam2.project_point(gt_point_3d)

    # Add 0.5 px noise
    reconstructed_3d, err = dlt_triangulate_point(
        [P1, P2], [(u1 + 0.3, v1 - 0.2), (u2 - 0.3, v2 + 0.2)]
    )
    assert reconstructed_3d is not None
    assert err < 2.0  # Reprojection error within noise magnitude
    diff = np.linalg.norm(reconstructed_3d - gt_point_3d)
    assert diff < 150.0  # Within ~1.5 meters across 300+ meter baseline


def test_dlt_triangulation_degenerate_cases():
    """Verifies DLT triangulation handles invalid / degenerate cases gracefully."""
    # Less than 2 cameras
    P = np.eye(3, 4)
    pt, err = dlt_triangulate_point([P], [(100.0, 100.0)])
    assert pt is None
    assert err == float("inf")

    # Mismatched lengths
    pt, err = dlt_triangulate_point([P, P], [(100.0, 100.0)])
    assert pt is None
    assert err == float("inf")


def test_bounding_box_filtering():
    """Verifies spatial bounds checking according to terrain_method_contract.md."""
    # Inside
    assert is_point_within_bounds(np.array([0.0, 0.0, 500.0]))
    assert is_point_within_bounds(np.array([-40000.0, -30000.0, 0.0]))
    assert is_point_within_bounds(np.array([20000.0, 30000.0, 5000.0]))

    # Outside X
    assert not is_point_within_bounds(np.array([-40001.0, 0.0, 500.0]))
    assert not is_point_within_bounds(np.array([20001.0, 0.0, 500.0]))

    # Outside Y
    assert not is_point_within_bounds(np.array([0.0, -30001.0, 500.0]))
    assert not is_point_within_bounds(np.array([0.0, 30001.0, 500.0]))

    # Outside Z
    assert not is_point_within_bounds(np.array([0.0, 0.0, -1.0]))
    assert not is_point_within_bounds(np.array([0.0, 0.0, 5001.0]))


# ---------------------------------------------------------------------------
# 4. Full Pipeline QA Gate on Hackathon Dataset
# ---------------------------------------------------------------------------

def test_reconstruct_terrain_hackathon_dataset(hackathon_data_dir):
    """
    Primary Acceptance Gate (Task 014 QA Gate):
    - Reconstructed points count >= 200.
    - All coordinates strictly inside Peleng bounding box:
      [-40000, 20000] x [-30000, 30000] x [0, 5000] cm.
    - Total runtime < 2.0 seconds on CPU.
    - Strict schema compliance with TerrainPayload.
    """
    t0 = time.perf_counter()
    payload = reconstruct_terrain(
        calib_dir=hackathon_data_dir,
        video_dir=hackathon_data_dir,
        frame_idx=0,
    )
    elapsed_sec = time.perf_counter() - t0

    # 1. Performance Target: < 2.0s
    assert elapsed_sec < 2.0, f"Reconstruction runtime {elapsed_sec:.2f}s exceeded 2.0s target"

    # 2. Schema compliance
    assert payload["method_name"] == "epipolar_stereo"
    assert payload["density"] == "sparse_landmarks"
    assert payload["faces"] is None
    assert isinstance(payload["vertices"], np.ndarray)
    assert isinstance(payload["colors"], np.ndarray)
    assert isinstance(payload["point_cloud"], np.ndarray)

    vertices = payload["vertices"]
    colors = payload["colors"]
    n_points = len(vertices)

    # 3. Acceptance Criteria: >= 200 points
    assert n_points >= 200, f"Expected >= 200 triangulated points, got {n_points}"
    assert vertices.shape == (n_points, 3)
    assert colors.shape == (n_points, 3)
    assert colors.dtype == np.uint8

    # 4. Spatial envelope verification: All points inside bounding box
    assert (vertices[:, 0] >= BOUNDS_X_MIN).all(), "Vertices contain X < -40000 cm"
    assert (vertices[:, 0] <= BOUNDS_X_MAX).all(), "Vertices contain X > 20000 cm"
    assert (vertices[:, 1] >= BOUNDS_Y_MIN).all(), "Vertices contain Y < -30000 cm"
    assert (vertices[:, 1] <= BOUNDS_Y_MAX).all(), "Vertices contain Y > 30000 cm"
    assert (vertices[:, 2] >= BOUNDS_Z_MIN).all(), "Vertices contain Z < 0 cm"
    assert (vertices[:, 2] <= BOUNDS_Z_MAX).all(), "Vertices contain Z > 5000 cm"

    # 5. Stats consistency
    stats = payload["stats"]
    assert stats["vertex_count"] == n_points
    assert stats["face_count"] == 0
    assert stats["runtime_ms"] > 0
    assert stats["z_min_cm"] == round(float(np.min(vertices[:, 2])), 2)
    assert stats["z_max_cm"] == round(float(np.max(vertices[:, 2])), 2)


def test_to_scene_payload_layer_integration(hackathon_data_dir):
    """Verifies that to_scene_payload_layer returns a dictionary ingestible by viewer_3d."""
    payload = reconstruct_terrain(
        calib_dir=hackathon_data_dir,
        video_dir=hackathon_data_dir,
        frame_idx=0,
    )
    layer = to_scene_payload_layer(payload)

    assert "point_cloud" in layer
    pc = layer["point_cloud"]
    assert "points_cm" in pc
    assert "colors_rgb" in pc

    pts = pc["points_cm"]
    cols = pc["colors_rgb"]
    assert len(pts) >= 200
    assert len(cols) == len(pts)

    # Check that points and colors are list of 3-element lists
    assert len(pts[0]) == 3
    assert len(cols[0]) == 3


def test_defensive_fallback_missing_files(tmp_path):
    """
    Verifies that reconstruct_terrain gracefully handles non-existent or empty directories
    via procedural fallback generation without crashing.
    """
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)

    payload = reconstruct_terrain(calib_dir=empty_dir, video_dir=empty_dir, frame_idx=0)
    assert isinstance(payload, dict)
    assert payload["method_name"] == "epipolar_stereo"
    assert "vertices" in payload
    assert "stats" in payload
    assert payload["stats"]["vertex_count"] == len(payload["vertices"])

    # Test empty payload layer conversion
    layer = to_scene_payload_layer(payload)
    assert "point_cloud" in layer
    assert isinstance(layer["point_cloud"]["points_cm"], list)
