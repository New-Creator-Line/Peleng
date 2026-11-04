"""
Unit tests for Task 020: Census Transform Engine & Vectorized Hamming Distance.

Verifies:
1. CensusConfig parameters, validation, and offset properties.
2. Frozen contract data structures (CensusMatch, SectorConfig).
3. Popcount64 bit counting accuracy and legacy fallback consistency.
4. Pairwise and matrix Hamming distance correctness.
5. Invariance properties:
   - Identical patches yield 0 Hamming distance.
   - Strictly inverted patches yield 48 Hamming distance.
   - Illumination invariance: 2.0x gain and +40 offset yield 0 Hamming distance.
6. MCT (Modified Census Transform) box filter mean invariance.
7. Input validation, color auto-conversion, and arbitrary spatial dimensions.
8. Performance benchmarks:
   - 1080p (1920x1080) Census encoding < 45 ms.
   - 10,000 pairs Hamming distance calculation < 5 ms.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure Peleng root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PELENG_ROOT = TEST_DIR.parent.parent
if str(PELENG_ROOT) not in sys.path:
    sys.path.insert(0, str(PELENG_ROOT))

from src.terrain.census_stereo.census_engine import (
    DEFAULT_BORDER_MODE,
    DEFAULT_NUM_THREADS,
    DEFAULT_RADIUS_U,
    DEFAULT_RADIUS_V,
    MAX_CENSUS_BITS,
    MAX_EPIPOLAR_DIST_PX_THRESHOLD,
    MAX_NORMALIZED_COST_THRESHOLD,
    MAX_UNIQUENESS_RATIO_THRESHOLD,
    CensusConfig,
    CensusMatch,
    SectorConfig,
    _popcount64_fallback,
    compute_census_7x7,
    compute_census_descriptor_map,
    compute_hamming_distance,
    compute_hamming_distance_matrix,
    compute_modified_census_7x7,
    popcount64,
)


class TestCensusConfig:
    """Tests configuration initialization, defaults, invariants, and validation."""

    def test_default_config(self) -> None:
        cfg = CensusConfig()
        assert cfg.radius_u == 3
        assert cfg.radius_v == 3
        assert cfg.window_width == 7
        assert cfg.window_height == 7
        assert cfg.num_bits == 48
        assert not cfg.use_mct
        assert cfg.border_mode == cv2.BORDER_REFLECT
        assert cfg.strict_comparison is True
        assert len(cfg.offsets) == 48
        assert (0, 0) not in cfg.offsets

    def test_custom_valid_config(self) -> None:
        cfg = CensusConfig(
            radius_u=2,
            radius_v=2,
            use_mct=True,
            mct_kernel_size=(5, 5),
            num_threads=2,
        )
        assert cfg.window_width == 5
        assert cfg.window_height == 5
        assert cfg.num_bits == 24
        assert cfg.use_mct
        assert len(cfg.offsets) == 24

    def test_invalid_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius_u must be positive"):
            CensusConfig(radius_u=0)

        with pytest.raises(ValueError, match="radius_v must be positive"):
            CensusConfig(radius_v=-1)

    def test_excessive_window_size_raises(self) -> None:
        # 11x11 window -> 121 - 1 = 120 bits > 64
        with pytest.raises(ValueError, match="exceeds maximum uint64 capacity"):
            CensusConfig(radius_u=5, radius_v=5)

    def test_invalid_num_threads_raises(self) -> None:
        with pytest.raises(ValueError, match="num_threads must be >= 1"):
            CensusConfig(num_threads=0)

    def test_invalid_mct_kernel_size_raises(self) -> None:
        with pytest.raises(ValueError, match="mct_kernel_size must have positive dimensions"):
            CensusConfig(mct_kernel_size=(0, 5))
        with pytest.raises(ValueError, match="mct_kernel_size must have positive dimensions"):
            CensusConfig(mct_kernel_size=(5, 0))


class TestContractDataStructures:
    """Verifies frozen contract types CensusMatch and SectorConfig."""

    def test_census_match_creation_and_validation(self) -> None:
        match = CensusMatch(
            cam_i=1,
            cam_j=2,
            pt_i=(100.5, 200.25),
            pt_j=(105.0, 202.0),
            hamming_cost=4,
            normalized_cost=4.0 / 48.0,
            uniqueness_ratio=0.5,
            epipolar_dist_px=0.8,
            color_rgb=(255, 128, 64),
        )
        assert match.cam_i == 1
        assert match.cam_j == 2
        assert match.is_valid()

        # Test exceeding uniqueness ratio
        invalid_match_ratio = CensusMatch(
            cam_i=1,
            cam_j=2,
            pt_i=(100.0, 200.0),
            pt_j=(105.0, 200.0),
            hamming_cost=5,
            normalized_cost=5.0 / 48.0,
            uniqueness_ratio=0.85,  # > 0.75
            epipolar_dist_px=0.5,
            color_rgb=(0, 0, 0),
        )
        assert not invalid_match_ratio.is_valid()

        # Test exceeding normalized cost (> 0.20)
        invalid_match_cost = CensusMatch(
            cam_i=1,
            cam_j=2,
            pt_i=(100.0, 200.0),
            pt_j=(105.0, 200.0),
            hamming_cost=15,
            normalized_cost=0.30,  # > 0.20
            uniqueness_ratio=0.5,
            epipolar_dist_px=0.5,
            color_rgb=(0, 0, 0),
        )
        assert not invalid_match_cost.is_valid()

        # Test exceeding epipolar distance (> 1.5 px)
        invalid_match_epipolar = CensusMatch(
            cam_i=1,
            cam_j=2,
            pt_i=(100.0, 200.0),
            pt_j=(105.0, 200.0),
            hamming_cost=4,
            normalized_cost=4.0 / 48.0,
            uniqueness_ratio=0.5,
            epipolar_dist_px=2.5,  # > 1.5
            color_rgb=(0, 0, 0),
        )
        assert not invalid_match_epipolar.is_valid()

    def test_sector_config_immutability(self) -> None:
        sec = SectorConfig(
            sector_id="sector_1_entry",
            cam_primary=2,
            cam_secondary=1,
            baseline_angle_deg=35.0,
            road_s_range=(0.0, 150.0),
        )
        assert sec.sector_id == "sector_1_entry"
        with pytest.raises(Exception):
            sec.cam_primary = 3  # type: ignore


class TestPopcountAndHammingDistance:
    """Verifies popcount64 SIMD, fallback, and Hamming distance calculations."""

    def test_popcount64_scalars_and_arrays(self) -> None:
        assert popcount64(0) == 0
        assert popcount64(1) == 1
        assert popcount64(0xFF) == 8
        assert popcount64(0xFFFFFFFFFFFFFFFF) == 64

        arr = np.array([0, 1, 3, 7, 15, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        expected = np.array([0, 1, 2, 3, 4, 64], dtype=np.int32)
        res = popcount64(arr)
        np.testing.assert_array_equal(res, expected)

    def test_popcount64_fallback_matches_simd(self) -> None:
        np.random.seed(42)
        test_vals = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=2000, dtype=np.uint64)
        simd_res = popcount64(test_vals)
        fallback_res = _popcount64_fallback(test_vals)
        np.testing.assert_array_equal(simd_res, fallback_res)

    def test_compute_hamming_distance_elementwise(self) -> None:
        c1 = np.uint64(0b101100)
        c2 = np.uint64(0b001101)
        # XOR is 0b100001 -> 2 bits set
        assert compute_hamming_distance(c1, c2) == 2

        arr1 = np.array([0b11, 0b00, 0b101], dtype=np.uint64)
        arr2 = np.array([0b10, 0b00, 0b010], dtype=np.uint64)
        dist = compute_hamming_distance(arr1, arr2)
        np.testing.assert_array_equal(dist, np.array([1, 0, 3], dtype=np.int32))

    def test_compute_hamming_distance_matrix(self) -> None:
        desc_i = np.array([0b00, 0b01, 0b11], dtype=np.uint64)
        desc_j = np.array([0b00, 0b10], dtype=np.uint64)
        mat = compute_hamming_distance_matrix(desc_i, desc_j)

        assert mat.shape == (3, 2)
        # i=0 (00) vs j=0 (00)->0, j=1 (10)->1
        # i=1 (01) vs j=0 (00)->1, j=1 (10)->2
        # i=2 (11) vs j=0 (00)->2, j=1 (10)->1
        expected = np.array([[0, 1], [1, 2], [2, 1]], dtype=np.int32)
        np.testing.assert_array_equal(mat, expected)

    def test_hamming_matrix_symmetry_and_identity(self) -> None:
        desc = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=50, dtype=np.uint64)
        mat = compute_hamming_distance_matrix(desc, desc)
        assert mat.shape == (50, 50)
        # Diagonal must be 0
        np.testing.assert_array_equal(np.diag(mat), np.zeros(50, dtype=np.int32))
        # Matrix must be symmetric
        np.testing.assert_array_equal(mat, mat.T)


class TestCensusTransformProperties:
    """Verifies Census bit encoding, bit indexing, and invariant properties."""

    def test_synthetic_patch_known_bits(self) -> None:
        cfg = CensusConfig(radius_u=3, radius_v=3, strict_comparison=True)
        # Create a 7x7 image with center at (3, 3)
        img = np.full((7, 7), 100, dtype=np.uint8)

        # 1. All neighbors > center -> descriptor should be 0
        img_all_greater = np.full((7, 7), 150, dtype=np.uint8)
        img_all_greater[3, 3] = 100
        desc_0 = compute_census_descriptor_map(img_all_greater, config=cfg)
        assert desc_0[3, 3] == np.uint64(0)

        # 2. All neighbors < center -> all 48 bits set
        img_all_smaller = np.full((7, 7), 50, dtype=np.uint8)
        img_all_smaller[3, 3] = 100
        desc_all = compute_census_descriptor_map(img_all_smaller, config=cfg)
        expected_all = (np.uint64(1) << np.uint64(48)) - np.uint64(1)
        assert desc_all[3, 3] == expected_all

        # 3. Exactly one specific neighbor set: offset k
        for k in [0, 10, 25, 47]:
            dv, du = cfg.offsets[k]
            img_single = np.full((7, 7), 150, dtype=np.uint8)
            img_single[3, 3] = 100
            # Set target neighbor strictly smaller
            img_single[3 + dv, 3 + du] = 50
            d = compute_census_descriptor_map(img_single, config=cfg)
            expected_mask = np.uint64(1) << np.uint64(k)
            assert d[3, 3] == expected_mask, f"Bit {k} at offset ({dv}, {du}) did not match!"

    def test_inverted_patch_hamming_distance(self) -> None:
        """Verifies that an inverted patch with distinct values produces 48 bit flips."""
        # 7x7 patch where all values are distinct: 0 to 48
        patch = np.arange(49, dtype=np.uint8).reshape(7, 7)
        inverted = (255 - patch).astype(np.uint8)

        cfg = CensusConfig(radius_u=3, radius_v=3)
        d_orig = compute_census_descriptor_map(patch, config=cfg)
        d_inv = compute_census_descriptor_map(inverted, config=cfg)

        h_dist = compute_hamming_distance(d_orig[3, 3], d_inv[3, 3])
        assert h_dist == 48

    def test_illumination_invariance_standard_census(self) -> None:
        """
        Verifies that scaling brightness by 2.0x or adding uniform offset +40
        produces identical Census bitstrings for non-saturated pixels.
        """
        np.random.seed(123)
        # Generate base image in range [25, 100] so 2.0x and +40 do not saturate uint8
        img = np.random.randint(25, 101, (120, 160), dtype=np.uint8)

        img_gain = (img.astype(np.float32) * 2.0).astype(np.uint8)
        img_offset = (img.astype(np.int32) + 40).astype(np.uint8)

        desc_base = compute_census_7x7(img)
        desc_gain = compute_census_7x7(img_gain)
        desc_offset = compute_census_7x7(img_offset)

        # Non-saturated pixels must have exact zero Hamming distance
        dist_gain = compute_hamming_distance(desc_base, desc_gain)
        dist_offset = compute_hamming_distance(desc_base, desc_offset)

        assert np.max(dist_gain) == 0, "Gain invariance violated in standard Census!"
        assert np.max(dist_offset) == 0, "Offset invariance violated in standard Census!"

    def test_illumination_invariance_mct(self) -> None:
        """Verifies illumination invariance for Modified Census Transform (MCT)."""
        np.random.seed(456)
        img = np.random.randint(25, 101, (100, 120), dtype=np.uint8)

        img_gain = (img.astype(np.float32) * 2.0).astype(np.uint8)
        img_offset = (img.astype(np.int32) + 40).astype(np.uint8)

        desc_base = compute_modified_census_7x7(img)
        desc_gain = compute_modified_census_7x7(img_gain)
        desc_offset = compute_modified_census_7x7(img_offset)

        dist_gain = compute_hamming_distance(desc_base, desc_gain)
        dist_offset = compute_hamming_distance(desc_base, desc_offset)

        assert np.max(dist_gain) == 0, "Gain invariance violated in MCT!"
        assert np.max(dist_offset) == 0, "Offset invariance violated in MCT!"

    def test_strict_comparison_flag(self) -> None:
        """Verifies strict (<) vs non-strict (<=) neighbor comparison."""
        # Flat image where all pixels are 100
        flat = np.full((7, 7), 100, dtype=np.uint8)

        # Under strict comparison (neighbor < 100), all bits are 0
        cfg_strict = CensusConfig(strict_comparison=True)
        desc_strict = compute_census_descriptor_map(flat, config=cfg_strict)
        assert desc_strict[3, 3] == 0

        # Under non-strict comparison (neighbor <= 100), all 48 bits are 1
        cfg_non_strict = CensusConfig(strict_comparison=False)
        desc_non_strict = compute_census_descriptor_map(flat, config=cfg_non_strict)
        expected_all = (np.uint64(1) << np.uint64(48)) - np.uint64(1)
        assert desc_non_strict[3, 3] == expected_all

    def test_mct_integer_precision_mode(self) -> None:
        """Verifies MCT execution when mct_float_precision=False."""
        img = np.random.randint(30, 90, (50, 50), dtype=np.uint8)
        cfg_int = CensusConfig(use_mct=True, mct_float_precision=False)
        desc_int = compute_census_descriptor_map(img, config=cfg_int)
        assert desc_int.shape == (50, 50)
        assert desc_int.dtype == np.uint64

    def test_multithread_matches_single_thread(self) -> None:
        """Verifies multi-threaded strip computation matches single-threaded bit-for-bit."""
        np.random.seed(789)
        img = np.random.randint(0, 256, (300, 400), dtype=np.uint8)

        cfg_st = CensusConfig(num_threads=1)
        cfg_mt = CensusConfig(num_threads=4)

        desc_st = compute_census_descriptor_map(img, config=cfg_st)
        desc_mt = compute_census_descriptor_map(img, config=cfg_mt)

        np.testing.assert_array_equal(desc_st, desc_mt)


class TestInputHandlingAndEdgeCases:
    """Verifies edge cases, color auto-conversion, and error handling."""

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="Input image cannot be None"):
            compute_census_descriptor_map(None)  # type: ignore

        with pytest.raises(ValueError, match="Expected numpy.ndarray"):
            compute_census_descriptor_map([[1, 2], [3, 4]])  # type: ignore

        with pytest.raises(ValueError, match="must not be empty"):
            compute_census_descriptor_map(np.array([], dtype=np.uint8))

        with pytest.raises(ValueError, match="Expected 2D or 3D image"):
            compute_census_descriptor_map(np.zeros((10, 10, 2, 2), dtype=np.uint8))

    def test_color_and_channel_conversions(self) -> None:
        # BGR 3-channel
        bgr = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        desc_from_bgr = compute_census_7x7(bgr)
        desc_from_gray = compute_census_7x7(gray)
        np.testing.assert_array_equal(desc_from_bgr, desc_from_gray)

        # 1-channel 3D array (H, W, 1)
        gray_3d = gray[:, :, None]
        desc_from_3d = compute_census_7x7(gray_3d)
        np.testing.assert_array_equal(desc_from_3d, desc_from_gray)

        # 4-channel BGRA image
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        desc_from_bgra = compute_census_7x7(bgra)
        np.testing.assert_array_equal(desc_from_bgra, desc_from_gray)

        # Unsupported channel count (5 channels) raises ValueError
        img_5ch = np.zeros((80, 80, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported channel count: 5"):
            compute_census_descriptor_map(img_5ch)

    def test_oversubscribed_threads_strip_worker(self) -> None:
        """Verifies strip worker handles images with fewer rows than threads."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        # 16 threads with H=100
        cfg = CensusConfig(num_threads=16)
        desc = compute_census_descriptor_map(img, config=cfg)
        desc_st = compute_census_descriptor_map(img, config=CensusConfig(num_threads=1))
        np.testing.assert_array_equal(desc, desc_st)

    def test_small_and_odd_dimensions(self) -> None:
        small = np.random.randint(0, 256, (15, 23), dtype=np.uint8)
        desc = compute_census_7x7(small)
        assert desc.shape == (15, 23)
        assert desc.dtype == np.uint64


class TestPerformanceGates:
    """Verifies acceptance criteria performance thresholds."""

    def test_hamming_distance_batch_performance(self) -> None:
        """Hamming distance batch calculation < 5 ms for 10,000 candidate pairs."""
        np.random.seed(999)
        d_i = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=100, dtype=np.uint64)
        d_j = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=100, dtype=np.uint64)

        # Warm-up
        _ = compute_hamming_distance_matrix(d_i, d_j)

        # Timed execution: 100 x 100 = 10,000 pairs
        t0 = time.perf_counter()
        mat = compute_hamming_distance_matrix(d_i, d_j)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert mat.shape == (100, 100)
        assert (
            t_elapsed_ms < 5.0
        ), f"Hamming 10k pairs took {t_elapsed_ms:.2f} ms (gate is < 5 ms)"

    def test_census_1080p_performance(self) -> None:
        """Census transform computation on 1920x1080 frame < 45 ms."""
        np.random.seed(101)
        frame_1080p = np.random.randint(0, 256, (1080, 1920), dtype=np.uint8)

        # Warm-up
        _ = compute_census_7x7(frame_1080p)

        # Timed run
        t0 = time.perf_counter()
        desc = compute_census_7x7(frame_1080p)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert desc.shape == (1080, 1920)
        assert desc.dtype == np.uint64
        assert (
            t_elapsed_ms < 45.0
        ), f"1080p Census took {t_elapsed_ms:.2f} ms (gate is < 45 ms)"
