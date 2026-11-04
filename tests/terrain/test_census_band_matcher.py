"""
Unit tests for Task 021: Sector-Based Epipolar Band Matcher & Left-Right Consistency.

Verifies:
1. SectorMatcherConfig configuration parameters, defaults, and validation.
2. Camera viewing angle computation and counter-facing camera pair rejection (e.g. Cams 2 & 4).
3. Co-directional sector verification and standard sectors (1, 2), (3, 5), (4, 5), (4, 6).
4. Pairwise Fundamental Matrix computation and epipolar constraint validation.
5. Epipolar ray-corridor projection and search band sampling.
6. Tri-Filter Pipeline:
   - Filter 1: Absolute Hamming cost <= max_hamming_cost (<= 9 bits).
   - Filter 2: Uniqueness ratio <= uniqueness_ratio_thresh (<= 0.75).
   - Filter 3: Left-Right consistency <= left_right_dist_thresh_px (<= 1.0 px).
7. Parabolic sub-pixel peak refinement accuracy.
8. Synthetic pattern correspondence and repetitive texture rejection.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pytest

# Ensure Peleng root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PELENG_ROOT = TEST_DIR.parent.parent
if str(PELENG_ROOT) not in sys.path:
    sys.path.insert(0, str(PELENG_ROOT))

from src.camera_model import CameraModel
from src.terrain.census_stereo.census_engine import (
    CensusConfig,
    CensusMatch,
    compute_census_descriptor_map,
)
from src.terrain.census_stereo.band_matcher import (
    DEFAULT_BAND_HALF_WIDTH_PX,
    DEFAULT_BAND_STEP_PX,
    DEFAULT_LEFT_RIGHT_DIST_THRESH_PX,
    DEFAULT_LINE_SAMPLE_STEP_PX,
    DEFAULT_MAX_COUNTER_FACING_ANGLE_DEG,
    DEFAULT_MAX_HAMMING_COST,
    DEFAULT_MIN_SECOND_PEAK_DIST_PX,
    DEFAULT_ROAD_Z_CM,
    DEFAULT_ROAD_Z_MAX_OFFSET_CM,
    DEFAULT_ROAD_Z_MIN_OFFSET_CM,
    DEFAULT_UNIQUENESS_RATIO_THRESH,
    STANDARD_SECTORS,
    EpipolarBandMatcher,
    SectorMatcherConfig,
    compute_camera_viewing_angle_deg,
    compute_fundamental_matrix,
    refine_subpixel_parabola,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_camera_pair() -> Tuple[CameraModel, CameraModel]:
    """
    Creates a co-directional stereo camera pair viewing a ground plane at Z ~ 450 cm.
    In Unreal Engine coordinates:
    - X is Forward.
    - Y is Right (lateral baseline between cameras).
    - Z is Up (elevation).
    Both cameras look forward along +X with a -15 deg downward pitch toward the road.
    """
    cam1 = CameraModel(
        cam_id=1,
        loc=np.array([0.0, -300.0, 1000.0]),
        rot=np.array([0.0, -15.0, 0.0]),
        focal_length_mm=35.0,
        sensor_width_mm=23.76,
        sensor_height_mm=13.365,
        image_width_px=1920,
        image_height_px=1080,
    )
    cam2 = CameraModel(
        cam_id=2,
        loc=np.array([0.0, 300.0, 1000.0]),
        rot=np.array([0.0, -15.0, 0.0]),
        focal_length_mm=35.0,
        sensor_width_mm=23.76,
        sensor_height_mm=13.365,
        image_width_px=1920,
        image_height_px=1080,
    )
    return cam1, cam2


@pytest.fixture
def counter_facing_camera_pair() -> Tuple[CameraModel, CameraModel]:
    """
    Creates two cameras facing opposite directions (angle ~ 160 deg > 100 deg),
    simulating Camera 2 vs Camera 4 in the Peleng dataset.
    """
    cam2 = CameraModel(
        cam_id=2,
        loc=np.array([-45000.0, -6500.0, 7900.0]),
        rot=np.array([0.0, -10.0, 0.0]),
        focal_length_mm=35.0,
    )
    cam4 = CameraModel(
        cam_id=4,
        loc=np.array([33000.0, 14000.0, 7100.0]),
        rot=np.array([0.0, -10.0, -160.0]),
        focal_length_mm=35.0,
    )
    return cam2, cam4


# ---------------------------------------------------------------------------
# 1. SectorMatcherConfig Tests
# ---------------------------------------------------------------------------

class TestSectorMatcherConfig:
    """Verifies dataclass configuration, defaults, invariants, and validation."""

    def test_default_values(self):
        cfg = SectorMatcherConfig()
        assert cfg.band_half_width_px == DEFAULT_BAND_HALF_WIDTH_PX
        assert cfg.band_half_width_px == 1.5
        assert cfg.band_step_px == DEFAULT_BAND_STEP_PX
        assert cfg.band_step_px == 0.75
        assert cfg.line_sample_step_px == DEFAULT_LINE_SAMPLE_STEP_PX
        assert cfg.line_sample_step_px == 0.5
        assert cfg.max_hamming_cost == DEFAULT_MAX_HAMMING_COST
        assert cfg.max_hamming_cost == 9
        assert cfg.uniqueness_ratio_thresh == DEFAULT_UNIQUENESS_RATIO_THRESH
        assert cfg.uniqueness_ratio_thresh == 0.75
        assert cfg.min_second_peak_dist_px == DEFAULT_MIN_SECOND_PEAK_DIST_PX
        assert cfg.min_second_peak_dist_px == 3.0
        assert cfg.left_right_dist_thresh_px == DEFAULT_LEFT_RIGHT_DIST_THRESH_PX
        assert cfg.left_right_dist_thresh_px == 1.0
        assert cfg.max_counter_facing_angle_deg == DEFAULT_MAX_COUNTER_FACING_ANGLE_DEG
        assert cfg.max_counter_facing_angle_deg == 100.0
        assert cfg.subpixel_refinement is True
        assert cfg.road_z_cm == DEFAULT_ROAD_Z_CM
        assert cfg.road_z_cm == 450.0
        assert cfg.road_z_min_offset_cm == DEFAULT_ROAD_Z_MIN_OFFSET_CM
        assert cfg.road_z_min_offset_cm == -50.0
        assert cfg.road_z_max_offset_cm == DEFAULT_ROAD_Z_MAX_OFFSET_CM
        assert cfg.road_z_max_offset_cm == 3500.0

    def test_invalid_parameters_raise_value_error(self):
        with pytest.raises(ValueError, match="band_half_width_px must be non-negative"):
            SectorMatcherConfig(band_half_width_px=-1.0)

        with pytest.raises(ValueError, match="band_step_px must be positive"):
            SectorMatcherConfig(band_step_px=0.0)

        with pytest.raises(ValueError, match="line_sample_step_px must be positive"):
            SectorMatcherConfig(line_sample_step_px=-0.5)

        with pytest.raises(ValueError, match="max_hamming_cost must be non-negative"):
            SectorMatcherConfig(max_hamming_cost=-1)

        with pytest.raises(ValueError, match="uniqueness_ratio_thresh must be in"):
            SectorMatcherConfig(uniqueness_ratio_thresh=0.0)

        with pytest.raises(ValueError, match="uniqueness_ratio_thresh must be in"):
            SectorMatcherConfig(uniqueness_ratio_thresh=1.5)

        with pytest.raises(ValueError, match="left_right_dist_thresh_px must be non-negative"):
            SectorMatcherConfig(left_right_dist_thresh_px=-0.1)

        with pytest.raises(ValueError, match="max_counter_facing_angle_deg must be in"):
            SectorMatcherConfig(max_counter_facing_angle_deg=190.0)

        with pytest.raises(ValueError, match="road_z_min_offset_cm must be < road_z_max_offset_cm"):
            SectorMatcherConfig(road_z_min_offset_cm=100.0, road_z_max_offset_cm=50.0)


# ---------------------------------------------------------------------------
# 2. Camera Viewing Angle & Counter-Facing Rejection Tests
# ---------------------------------------------------------------------------

class TestCameraViewingAngleAndSectorVerification:
    """Tests camera viewing angles and rejection of counter-facing camera pairs."""

    def test_parallel_cameras_viewing_angle(self, synthetic_camera_pair):
        cam1, cam2 = synthetic_camera_pair
        angle = compute_camera_viewing_angle_deg(cam1, cam2)
        assert abs(angle) < 1e-4

    def test_counter_facing_angle_detected(self, counter_facing_camera_pair):
        cam2, cam4 = counter_facing_camera_pair
        angle = compute_camera_viewing_angle_deg(cam2, cam4)
        assert angle > 100.0
        assert 150.0 < angle < 165.0

    def test_counter_facing_pair_rejected_by_exception(self, counter_facing_camera_pair):
        cam2, cam4 = counter_facing_camera_pair
        matcher = EpipolarBandMatcher(SectorMatcherConfig(strict_pair_validation=True))

        dummy_img = np.zeros((1080, 1920), dtype=np.uint8)
        with pytest.raises(ValueError, match="Counter-facing camera pair rejected for stereo matching"):
            matcher.match_camera_pair(cam2, cam4, dummy_img, dummy_img)

    def test_counter_facing_pair_returns_empty_when_non_strict(self, counter_facing_camera_pair):
        cam2, cam4 = counter_facing_camera_pair
        matcher = EpipolarBandMatcher(SectorMatcherConfig(strict_pair_validation=False))

        dummy_img = np.zeros((1080, 1920), dtype=np.uint8)
        matches = matcher.match_camera_pair(cam2, cam4, dummy_img, dummy_img)
        assert matches == []

    def test_standard_sectors_definition(self):
        assert "sector_1" in STANDARD_SECTORS
        assert STANDARD_SECTORS["sector_1"] == (1, 2)
        assert STANDARD_SECTORS["sector_2"] == (3, 5)
        assert STANDARD_SECTORS["sector_3"] == (4, 5)
        assert STANDARD_SECTORS["sector_4"] == (4, 6)


# ---------------------------------------------------------------------------
# 3. Fundamental Matrix & Epipolar Geometry
# ---------------------------------------------------------------------------

class TestFundamentalMatrixGeometry:
    """Verifies Fundamental Matrix computation and epipolar constraint x_j^T F x_i = 0."""

    def test_epipolar_constraint_satisfied(self, synthetic_camera_pair):
        cam1, cam2 = synthetic_camera_pair
        F_12 = compute_fundamental_matrix(cam1, cam2)

        # 3D test point on the road plane in front of both cameras
        pt_3d = np.array([3000.0, 0.0, 450.0])
        u1, v1, z1 = cam1.project_point(pt_3d)
        u2, v2, z2 = cam2.project_point(pt_3d)

        assert z1 > 0 and z2 > 0

        p1_hom = np.array([u1, v1, 1.0])
        p2_hom = np.array([u2, v2, 1.0])

        # Epipolar algebraic error: p2^T F p1 = 0
        alg_error = abs(float(p2_hom @ F_12 @ p1_hom))
        assert alg_error < 1e-4

        # Geometric perpendicular distance from p2 to epipolar line l2 = F_12 p1
        l2 = F_12 @ p1_hom
        line_norm = math.hypot(l2[0], l2[1])
        geom_dist = abs(float(l2 @ p2_hom)) / line_norm
        assert geom_dist < 1e-4


# ---------------------------------------------------------------------------
# 4. Sub-pixel Refinement Tests
# ---------------------------------------------------------------------------

class TestSubPixelRefinement:
    """Tests 1D parabolic sub-pixel peak refinement."""

    def test_symmetric_minimum_has_zero_shift(self):
        # Quadratic C(x) = 2*x^2 + 4, minimum at x=0
        shift = refine_subpixel_parabola(c_neg=6.0, c_zero=4.0, c_pos=6.0)
        assert abs(shift) < 1e-6

    def test_asymmetric_left_shift(self):
        # Parabola with minimum shifted to the left (e.g. x = -0.25)
        # f(x) = x^2 + 0.5*x + 1 -> f(-1) = 1.5, f(0) = 1.0, f(1) = 2.5
        # delta = (1.5 - 2.5) / (2 * (1.5 - 2.0 + 2.5)) = -1.0 / (2 * 2.0) = -0.25
        shift = refine_subpixel_parabola(c_neg=1.5, c_zero=1.0, c_pos=2.5)
        assert pytest.approx(shift, abs=1e-5) == -0.25

    def test_flat_or_concave_returns_zero(self):
        # Flat: denominator is zero
        shift = refine_subpixel_parabola(c_neg=5.0, c_zero=5.0, c_pos=5.0)
        assert shift == 0.0

        # Concave: peak instead of trough (c_zero > c_neg and c_pos)
        shift = refine_subpixel_parabola(c_neg=2.0, c_zero=8.0, c_pos=2.0)
        assert shift == 0.0


# ---------------------------------------------------------------------------
# 5. Tri-Filter Pipeline & Correspondence Matching
# ---------------------------------------------------------------------------

class TestEpipolarBandMatcherTriFilter:
    """Tests the three gating filters: Absolute Cost, Uniqueness Ratio, and Left-Right Consistency."""

    def test_perfect_synthetic_pattern_match(self, synthetic_camera_pair):
        """
        Creates synthetic textured images where a distinct random noise patch
        at a known 3D road location projects into both cameras.
        The matcher must find the true correspondence with high accuracy.
        """
        cam1, cam2 = synthetic_camera_pair
        H, W = 1080, 1920

        # Create textured ground image
        np.random.seed(42)
        img1 = np.full((H, W), 128, dtype=np.uint8)
        img2 = np.full((H, W), 128, dtype=np.uint8)

        # 3D test point on the road
        pt_3d = np.array([3000.0, 0.0, 450.0])
        u1, v1, z1 = cam1.project_point(pt_3d)
        u2, v2, z2 = cam2.project_point(pt_3d)

        assert z1 > 0 and z2 > 0

        iu1, iv1 = int(round(u1)), int(round(v1))
        iu2, iv2 = int(round(u2)), int(round(v2))

        # Paint an identical high-contrast distinct textured patch in both images
        patch_size = 25
        half = patch_size // 2
        texture = np.random.randint(20, 240, (patch_size, patch_size), dtype=np.uint8)

        img1[iv1 - half:iv1 + half + 1, iu1 - half:iu1 + half + 1] = texture
        img2[iv2 - half:iv2 + half + 1, iu2 - half:iu2 + half + 1] = texture

        matcher = EpipolarBandMatcher()
        matches = matcher.match_camera_pair(
            cam1,
            cam2,
            img1,
            img2,
            keypoints_i=[(float(iu1), float(iv1))],
        )

        assert len(matches) == 1
        m = matches[0]
        assert isinstance(m, CensusMatch)
        assert m.cam_i == 1
        assert m.cam_j == 2
        assert m.hamming_cost <= 9
        assert m.uniqueness_ratio <= 0.75
        assert m.epipolar_dist_px <= 1.5

        # Sub-pixel match coordinate should be very close to ground truth iu2, iv2
        assert abs(m.pt_j[0] - iu2) <= 1.5
        assert abs(m.pt_j[1] - iv2) <= 1.5

    def test_filter_1_rejects_high_cost(self, synthetic_camera_pair):
        """Verifies that completely dissimilar patches (hamming cost > 9) are rejected."""
        cam1, cam2 = synthetic_camera_pair
        H, W = 1080, 1920

        # Uniform gray vs random white noise -> completely dissimilar Census codes
        img1 = np.full((H, W), 128, dtype=np.uint8)
        img2 = np.random.randint(0, 255, (H, W), dtype=np.uint8)

        matcher = EpipolarBandMatcher(SectorMatcherConfig(max_hamming_cost=9))
        matches = matcher.match_camera_pair(
            cam1,
            cam2,
            img1,
            img2,
            keypoints_i=[(960.0, 540.0)],
        )
        assert len(matches) == 0

    def test_filter_2_rejects_repetitive_pattern(self, synthetic_camera_pair):
        """
        Creates a periodic striped pattern along the epipolar direction.
        The Uniqueness Ratio filter must reject the correspondence because
        multiple identical local minima occur along the epipolar line.
        """
        cam1, cam2 = synthetic_camera_pair
        H, W = 1080, 1920

        # Generate repetitive sinusoidal or square stripes along horizontal axis
        x = np.arange(W, dtype=np.float32)
        # Period of 8 pixels creates recurring peaks every 8 pixels
        stripes = (128.0 + 100.0 * np.sin(2.0 * np.pi * x / 8.0)).astype(np.uint8)
        img_striped = np.tile(stripes, (H, 1))

        matcher = EpipolarBandMatcher(SectorMatcherConfig(
            uniqueness_ratio_thresh=0.75,
            min_second_peak_dist_px=3.0,
        ))

        # Query a point in the periodic region
        matches = matcher.match_camera_pair(
            cam1,
            cam2,
            img_striped,
            img_striped,
            keypoints_i=[(960.0, 540.0)],
        )

        # Repetitive pattern MUST be rejected by Filter 2 or Filter 3
        assert len(matches) == 0

    def test_filter_3_left_right_consistency_rejection(self, synthetic_camera_pair):
        """
        Simulates an occlusion or false match where camera 1's keypoint finds
        a feature in camera 2, but camera 2's reverse search lands far away (> 1.0 px).
        The match MUST be rejected.
        """
        cam1, cam2 = synthetic_camera_pair
        H, W = 1080, 1920

        np.random.seed(99)
        img1 = np.random.randint(100, 150, (H, W), dtype=np.uint8)
        img2 = np.random.randint(100, 150, (H, W), dtype=np.uint8)

        # Place a unique patch in img1 at (960, 540)
        patch = np.random.randint(10, 240, (15, 15), dtype=np.uint8)
        img1[533:548, 953:968] = patch

        # In img2, place the same patch at a location that does NOT project back to (960, 540)
        # E.g., shifted by 50 pixels away from the true epipolar back-projection
        img2[533:548, 1053:1068] = patch

        # Force tight left-right threshold
        matcher = EpipolarBandMatcher(SectorMatcherConfig(
            left_right_dist_thresh_px=1.0,
        ))

        matches = matcher.match_camera_pair(
            cam1,
            cam2,
            img1,
            img2,
            keypoints_i=[(960.0, 540.0)],
        )
        assert len(matches) == 0

    def test_automatic_keypoint_detection_and_matching(self, synthetic_camera_pair):
        """Verifies that matcher can automatically detect keypoints when none are supplied."""
        cam1, cam2 = synthetic_camera_pair
        H, W = 1080, 1920

        np.random.seed(123)
        img1 = np.full((H, W), 128, dtype=np.uint8)
        img2 = np.full((H, W), 128, dtype=np.uint8)

        # Plant 3 distinct textured features on the road
        for offset_y in [-150.0, 0.0, 150.0]:
            pt_3d = np.array([2800.0, offset_y, 450.0])
            u1, v1, z1 = cam1.project_point(pt_3d)
            u2, v2, z2 = cam2.project_point(pt_3d)
            assert z1 > 0 and z2 > 0

            iu1, iv1 = int(round(u1)), int(round(v1))
            iu2, iv2 = int(round(u2)), int(round(v2))

            tex = np.random.randint(10, 245, (25, 25), dtype=np.uint8)
            img1[iv1 - 12:iv1 + 13, iu1 - 12:iu1 + 13] = tex
            img2[iv2 - 12:iv2 + 13, iu2 - 12:iu2 + 13] = tex

        matcher = EpipolarBandMatcher(SectorMatcherConfig(
            max_hamming_cost=9,
            uniqueness_ratio_thresh=0.75,
            fast_threshold=15,
            max_keypoints=30,
        ))

        # Do not supply keypoints; matcher will extract them
        matches = matcher.match_camera_pair(
            cam1,
            cam2,
            img1,
            img2,
            keypoints_i=None,
        )

        # Should find matches for the planted features
        assert len(matches) >= 1
        for m in matches:
            assert m.is_valid()
