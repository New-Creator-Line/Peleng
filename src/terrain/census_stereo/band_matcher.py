"""
Sector-Based Epipolar Band Matcher & Left-Right Consistency Engine.

Implements:
1. SectorMatcherConfig dataclass with verified default thresholds and validation.
2. Camera baseline viewing angle calculation and counter-facing pair rejection (> 100°).
3. Sector configuration and support for standard co-directional sectors:
   (1, 2), (3, 5), (4, 5), (4, 6).
4. Pairwise Fundamental Matrix computation F_ij = K_j^{-T} [t_ij]_x R_ij K_i^{-1}.
5. Epipolar ray corridor bounding [Z_road - 0.5m, Z_road + 35.0m] and 2D band sampling.
6. Tri-Filter Pipeline:
   - Filter 1 (Absolute Match): Hamming cost <= max_hamming_cost (9 bits / 20%).
   - Filter 2 (Uniqueness Ratio): Cost_min / Cost_2nd <= uniqueness_ratio_thresh (0.75).
   - Filter 3 (Left-Right Consistency): ||p_i - p_{i, back}|| <= left_right_dist_thresh_px (1.0 px).
7. Parabolic sub-pixel peak refinement along epipolar trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.camera_model import CameraModel
from .census_engine import (
    CensusConfig,
    CensusMatch,
    SectorConfig,
    compute_census_descriptor_map,
    popcount64,
)

# --- Verified System-Wide Constants (Zero Magic Numbers) ---

DEFAULT_BAND_HALF_WIDTH_PX: float = 1.5
DEFAULT_BAND_STEP_PX: float = 0.75
DEFAULT_LINE_SAMPLE_STEP_PX: float = 0.5
DEFAULT_MAX_HAMMING_COST: int = 9  # 20% of 48-bit 7x7 Census window
DEFAULT_UNIQUENESS_RATIO_THRESH: float = 0.75
DEFAULT_MIN_SECOND_PEAK_DIST_PX: float = 3.0
DEFAULT_LEFT_RIGHT_DIST_THRESH_PX: float = 1.0
DEFAULT_MAX_COUNTER_FACING_ANGLE_DEG: float = 100.0
DEFAULT_SUBPIXEL_REFINEMENT: bool = True

DEFAULT_ROAD_Z_CM: float = 450.0
DEFAULT_ROAD_Z_MIN_OFFSET_CM: float = -50.0    # -0.5m in Unreal Engine cm
DEFAULT_ROAD_Z_MAX_OFFSET_CM: float = 3500.0   # +35.0m in Unreal Engine cm

DEFAULT_MIN_CENSUS_ENTROPY_BITS: int = 4
DEFAULT_MIN_CORRIDOR_DEPTH_CM: float = 10.0
DEFAULT_MAX_CORRIDOR_DEPTH_CM: float = 100000.0

DEFAULT_STRICT_PAIR_VALIDATION: bool = True
DEFAULT_BORDER_MARGIN_PX: int = 4
DEFAULT_MAX_KEYPOINTS: int = 2000
DEFAULT_FAST_THRESHOLD: int = 20

# Standard co-directional sectors
STANDARD_SECTORS: Dict[str, Tuple[int, int]] = {
    "sector_1": (1, 2),
    "sector_2": (3, 5),
    "sector_3": (4, 5),
    "sector_4": (4, 6),
}


@dataclass(frozen=True)
class SectorMatcherConfig:
    """
    Configuration for Sector-Based Epipolar Band Matching and Correspondence Filtering.

    Attributes:
        band_half_width_px: Half-width of normal search band orthogonal to epipolar line.
        band_step_px: Sampling step across normal search band.
        line_sample_step_px: Sampling step along epipolar line trajectory.
        max_hamming_cost: Maximum allowed bit mismatches for Filter 1 (absolute match).
        uniqueness_ratio_thresh: Maximum allowed cost_min / cost_second_min ratio for Filter 2.
        min_second_peak_dist_px: Minimum spatial distance required for the second minimum peak.
        left_right_dist_thresh_px: Maximum allowed reverse match discrepancy in pixels for Filter 3.
        max_counter_facing_angle_deg: Maximum allowed camera viewing angle before pair rejection.
        subpixel_refinement: Whether to apply 1D parabolic sub-pixel peak refinement.
        road_z_cm: Base road elevation in Unreal Engine world coordinates (cm).
        road_z_min_offset_cm: Lower elevation bound relative to road_z_cm (cm).
        road_z_max_offset_cm: Upper elevation bound relative to road_z_cm (cm).
        min_census_entropy_bits: Minimum required popcount (and maximum num_bits - min_entropy) for code entropy.
        min_corridor_depth_cm: Minimum optical depth along camera ray for corridor intersection (cm).
        max_corridor_depth_cm: Maximum optical depth along camera ray for corridor intersection (cm).
        strict_pair_validation: If True, raises ValueError on counter-facing camera pairs.
        border_margin_px: Minimum margin from image borders for candidate sampling.
        max_keypoints: Maximum number of keypoints to extract if not provided by caller.
        fast_threshold: FAST detector corner response threshold.
        census_config: CensusConfig instance for Census descriptor generation.
    """

    band_half_width_px: float = DEFAULT_BAND_HALF_WIDTH_PX
    band_step_px: float = DEFAULT_BAND_STEP_PX
    line_sample_step_px: float = DEFAULT_LINE_SAMPLE_STEP_PX
    max_hamming_cost: int = DEFAULT_MAX_HAMMING_COST
    uniqueness_ratio_thresh: float = DEFAULT_UNIQUENESS_RATIO_THRESH
    min_second_peak_dist_px: float = DEFAULT_MIN_SECOND_PEAK_DIST_PX
    left_right_dist_thresh_px: float = DEFAULT_LEFT_RIGHT_DIST_THRESH_PX
    max_counter_facing_angle_deg: float = DEFAULT_MAX_COUNTER_FACING_ANGLE_DEG
    subpixel_refinement: bool = DEFAULT_SUBPIXEL_REFINEMENT
    road_z_cm: float = DEFAULT_ROAD_Z_CM
    road_z_min_offset_cm: float = DEFAULT_ROAD_Z_MIN_OFFSET_CM
    road_z_max_offset_cm: float = DEFAULT_ROAD_Z_MAX_OFFSET_CM
    min_census_entropy_bits: int = DEFAULT_MIN_CENSUS_ENTROPY_BITS
    min_corridor_depth_cm: float = DEFAULT_MIN_CORRIDOR_DEPTH_CM
    max_corridor_depth_cm: float = DEFAULT_MAX_CORRIDOR_DEPTH_CM
    strict_pair_validation: bool = DEFAULT_STRICT_PAIR_VALIDATION
    border_margin_px: int = DEFAULT_BORDER_MARGIN_PX
    max_keypoints: int = DEFAULT_MAX_KEYPOINTS
    fast_threshold: int = DEFAULT_FAST_THRESHOLD
    census_config: CensusConfig = field(default_factory=CensusConfig)

    def __post_init__(self) -> None:
        if self.band_half_width_px < 0.0:
            raise ValueError(
                f"band_half_width_px must be non-negative, got {self.band_half_width_px}"
            )
        if self.band_step_px <= 0.0:
            raise ValueError(f"band_step_px must be positive, got {self.band_step_px}")
        if self.line_sample_step_px <= 0.0:
            raise ValueError(
                f"line_sample_step_px must be positive, got {self.line_sample_step_px}"
            )
        if self.max_hamming_cost < 0:
            raise ValueError(
                f"max_hamming_cost must be non-negative, got {self.max_hamming_cost}"
            )
        if not (0.0 < self.uniqueness_ratio_thresh <= 1.0):
            raise ValueError(
                f"uniqueness_ratio_thresh must be in (0.0, 1.0], got {self.uniqueness_ratio_thresh}"
            )
        if self.min_second_peak_dist_px < 0.0:
            raise ValueError(
                f"min_second_peak_dist_px must be non-negative, got {self.min_second_peak_dist_px}"
            )
        if self.left_right_dist_thresh_px < 0.0:
            raise ValueError(
                f"left_right_dist_thresh_px must be non-negative, got {self.left_right_dist_thresh_px}"
            )
        if not (0.0 < self.max_counter_facing_angle_deg <= 180.0):
            raise ValueError(
                f"max_counter_facing_angle_deg must be in (0.0, 180.0], got {self.max_counter_facing_angle_deg}"
            )
        if self.road_z_min_offset_cm >= self.road_z_max_offset_cm:
            raise ValueError(
                f"road_z_min_offset_cm must be < road_z_max_offset_cm, "
                f"got min={self.road_z_min_offset_cm}, max={self.road_z_max_offset_cm}"
            )
        if (
            self.min_census_entropy_bits < 0
            or self.min_census_entropy_bits > (self.census_config.num_bits // 2)
        ):
            raise ValueError(
                f"min_census_entropy_bits must be in [0, {self.census_config.num_bits // 2}], "
                f"got {self.min_census_entropy_bits}"
            )
        if (
            self.min_corridor_depth_cm <= 0.0
            or self.min_corridor_depth_cm >= self.max_corridor_depth_cm
        ):
            raise ValueError(
                f"min_corridor_depth_cm must be positive and < max_corridor_depth_cm, "
                f"got min={self.min_corridor_depth_cm}, max={self.max_corridor_depth_cm}"
            )
        if self.border_margin_px < 0:
            raise ValueError(
                f"border_margin_px must be non-negative, got {self.border_margin_px}"
            )


def compute_camera_viewing_angle_deg(cam_i: CameraModel, cam_j: CameraModel) -> float:
    """
    Computes angle in degrees between optical axes (forward viewing vectors) of two cameras.

    In Unreal Engine / OpenCV coordinate transform used by CameraModel:
    The third row of R_cam (cam.R_cam[2, :]) represents the camera forward optical axis in world space.

    Args:
        cam_i: First CameraModel instance.
        cam_j: Second CameraModel instance.

    Returns:
        Viewing angle in degrees in [0.0, 180.0].
    """
    fwd_i = np.asarray(cam_i.R_cam[2, :], dtype=np.float64)
    fwd_j = np.asarray(cam_j.R_cam[2, :], dtype=np.float64)

    norm_i = np.linalg.norm(fwd_i)
    norm_j = np.linalg.norm(fwd_j)

    if norm_i < 1e-12 or norm_j < 1e-12:
        return 0.0

    dot = np.dot(fwd_i, fwd_j) / (norm_i * norm_j)
    clamped_dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.degrees(np.arccos(clamped_dot)))


def compute_fundamental_matrix(cam_i: CameraModel, cam_j: CameraModel) -> np.ndarray:
    """
    Computes pairwise Fundamental Matrix F_ij = K_j^{-T} [t_ij]_x R_ij K_i^{-1}.

    Transforms a point x_i in image i to an epipolar line l_j = F_ij @ x_i in image j,
    satisfying the epipolar constraint x_j^T @ F_ij @ x_i = 0.

    Args:
        cam_i: Source camera model.
        cam_j: Destination camera model.

    Returns:
        3x3 Fundamental Matrix as float64 array, scale-normalized by Frobenius norm.
    """
    K_i = np.asarray(cam_i.K, dtype=np.float64)
    K_j = np.asarray(cam_j.K, dtype=np.float64)
    R_i = np.asarray(cam_i.R_cam, dtype=np.float64)
    R_j = np.asarray(cam_j.R_cam, dtype=np.float64)
    t_i = np.asarray(cam_i.t_cam, dtype=np.float64)
    t_j = np.asarray(cam_j.t_cam, dtype=np.float64)

    R_ij = R_j @ R_i.T
    t_ij = t_j - R_ij @ t_i

    tx, ty, tz = t_ij
    t_x = np.array([
        [0.0, -tz, ty],
        [tz, 0.0, -tx],
        [-ty, tx, 0.0],
    ], dtype=np.float64)

    E_ij = t_x @ R_ij

    inv_K_j = np.linalg.inv(K_j)
    inv_K_i = np.linalg.inv(K_i)
    F_ij = inv_K_j.T @ E_ij @ inv_K_i

    norm = np.linalg.norm(F_ij)
    if norm > 1e-12:
        F_ij = F_ij / norm

    return F_ij


def clip_line_segment_to_rect(
    p1: np.ndarray,
    p2: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Clips a 2D line segment [p1, p2] to a rectangle using Liang-Barsky parametric clipping.

    Returns:
        (clipped_p1, clipped_p2) if segment intersects rectangle, else None.
    """
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    p = [-dx, dx, -dy, dy]
    q = [float(p1[0] - x_min), float(x_max - p1[0]), float(p1[1] - y_min), float(y_max - p1[1])]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t

    if u1 > u2:
        return None

    delta = np.array([dx, dy], dtype=np.float64)
    clipped_p1 = p1 + u1 * delta
    clipped_p2 = p1 + u2 * delta
    return clipped_p1, clipped_p2


def refine_subpixel_parabola(c_neg: float, c_zero: float, c_pos: float) -> float:
    """
    Performs 1D parabolic sub-pixel peak refinement over three cost samples [-1, 0, +1].

    Formula:
        delta = (C_{-1} - C_{+1}) / (2 * (C_{-1} - 2 * C_0 + C_{+1}))

    Args:
        c_neg: Cost at step -1.
        c_zero: Cost at center step 0 (minimum candidate).
        c_pos: Cost at step +1.

    Returns:
        Sub-pixel shift delta in [-1.0, 1.0]. Returns 0.0 if flat or concave.
    """
    denom = 2.0 * (c_neg - 2.0 * c_zero + c_pos)
    if denom <= 1e-12:
        return 0.0

    delta = (c_neg - c_pos) / denom
    if abs(delta) > 1.0:
        return float(np.clip(delta, -1.0, 1.0))
    return float(delta)


class EpipolarBandMatcher:
    """
    Sector-based epipolar band matcher utilizing Census Transform and Tri-Filter verification.

    Executes:
    1. Angle-gated camera sector verification.
    2. Fundamental Matrix epipolar line and 3D road corridor ray projection.
    3. 2D candidate band sampling [-band_half_width, +band_half_width].
    4. Hardware-accelerated bitwise Census Hamming distance matching.
    5. Tri-Filter gating (Absolute Match, Uniqueness Ratio, Left-Right Consistency).
    6. Parabolic sub-pixel peak refinement.
    """

    def __init__(self, config: Optional[SectorMatcherConfig] = None) -> None:
        self.config = config if config is not None else SectorMatcherConfig()

    def is_pair_allowed(self, cam_i: CameraModel, cam_j: CameraModel) -> bool:
        """Checks whether camera pair viewing angle is within configured threshold."""
        angle = compute_camera_viewing_angle_deg(cam_i, cam_j)
        return angle <= self.config.max_counter_facing_angle_deg

    def _cast_ray_to_corridor_segment(
        self,
        cam_src: CameraModel,
        cam_dst: CameraModel,
        pt_src: Tuple[float, float],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Casts a ray from cam_src through pixel pt_src, clips to the road elevation corridor,
        and projects the segment endpoints into cam_dst.
        """
        u, v = pt_src
        fx, fy, cx, cy = cam_src.get_intrinsics()
        ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        d_world = cam_src.R_cam.T @ ray_cam

        z_road = self.config.road_z_cm
        z_min = z_road + self.config.road_z_min_offset_cm
        z_max = z_road + self.config.road_z_max_offset_cm

        min_depth = self.config.min_corridor_depth_cm
        max_depth = self.config.max_corridor_depth_cm

        if abs(d_world[2]) > 1e-9:
            t_min_plane = (z_min - cam_src.loc[2]) / d_world[2]
            t_max_plane = (z_max - cam_src.loc[2]) / d_world[2]
            t_corridor_1 = min(t_min_plane, t_max_plane)
            t_corridor_2 = max(t_min_plane, t_max_plane)

            t1 = max(t_corridor_1, min_depth)
            t2 = min(t_corridor_2, max_depth)
        else:
            if z_min <= cam_src.loc[2] <= z_max:
                t1 = min_depth
                t2 = max_depth
            else:
                return None

        if t1 >= t2 or t2 <= min_depth:
            return None

        P1 = cam_src.loc + t1 * d_world
        P2 = cam_src.loc + t2 * d_world

        u1, v1, z1 = cam_dst.project_point(P1)
        u2, v2, z2 = cam_dst.project_point(P2)

        if z1 <= 0.0 and z2 <= 0.0:
            return None

        # If one endpoint is behind destination camera, interpolate along ray
        if z1 <= 0.0 or z2 <= 0.0:
            # Solve for t where point in cam_dst has z_cam = min_depth
            diff_C = cam_src.loc - cam_dst.loc
            r2 = cam_dst.R_cam[2]
            denom = float(np.dot(r2, d_world))
            if abs(denom) > 1e-9:
                t_clip = (min_depth - float(np.dot(r2, diff_C))) / denom
                if z1 <= 0.0:
                    t1 = max(t1, t_clip)
                    P1 = cam_src.loc + t1 * d_world
                    u1, v1, z1 = cam_dst.project_point(P1)
                else:
                    t2 = min(t2, t_clip)
                    P2 = cam_src.loc + t2 * d_world
                    u2, v2, z2 = cam_dst.project_point(P2)
            else:
                return None

        if t1 >= t2:
            return None

        return np.array([u1, v1], dtype=np.float64), np.array([u2, v2], dtype=np.float64)

    def _sample_band_and_find_match(
        self,
        cam_src: CameraModel,
        cam_dst: CameraModel,
        pt_src: Tuple[float, float],
        desc_map_src: np.ndarray,
        desc_map_dst: np.ndarray,
        F_src_dst: np.ndarray,
        check_left_right: bool = True,
    ) -> Optional[CensusMatch]:
        """
        Samples candidates along the epipolar band in cam_dst for pt_src in cam_src,
        evaluates Census Hamming costs, applies Tri-Filter pipeline, and sub-pixel refines.
        """
        u_src, v_src = pt_src
        H_src, W_src = desc_map_src.shape
        H_dst, W_dst = desc_map_dst.shape
        margin = self.config.border_margin_px

        iu_src = int(round(u_src))
        iv_src = int(round(v_src))
        if not (margin <= iu_src < W_src - margin and margin <= iv_src < H_src - margin):
            return None

        code_src = desc_map_src[iv_src, iu_src]

        # Low-entropy descriptor rejection (filter out flat / aperture patches)
        popcount = int(code_src).bit_count()
        num_bits = self.config.census_config.num_bits
        if (
            popcount < self.config.min_census_entropy_bits
            or popcount > (num_bits - self.config.min_census_entropy_bits)
        ):
            return None

        # Epipolar line in destination camera: l_dst = F_src_dst @ [u, v, 1]^T
        p_hom_src = np.array([u_src, v_src, 1.0], dtype=np.float64)
        l_dst = F_src_dst @ p_hom_src
        norm_l = math.hypot(l_dst[0], l_dst[1])
        if norm_l < 1e-12:
            return None
        a, b, c = l_dst[0] / norm_l, l_dst[1] / norm_l, l_dst[2] / norm_l

        # Project ray corridor segment
        endpoints = self._cast_ray_to_corridor_segment(cam_src, cam_dst, pt_src)
        if endpoints is None:
            return None
        q1_raw, q2_raw = endpoints

        # Project endpoints onto epipolar line to guarantee collinearity
        line_n = np.array([a, b], dtype=np.float64)
        q1 = q1_raw - (a * q1_raw[0] + b * q1_raw[1] + c) * line_n
        q2 = q2_raw - (a * q2_raw[0] + b * q2_raw[1] + c) * line_n

        # Clip segment to valid destination image bounds (with margin)
        clipped = clip_line_segment_to_rect(
            q1,
            q2,
            float(margin),
            float(W_dst - margin - 1),
            float(margin),
            float(H_dst - margin - 1),
        )
        if clipped is None:
            return None
        q1_clip, q2_clip = clipped

        seg_vec = q2_clip - q1_clip
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < 1e-4:
            tangent = np.array([-b, a], dtype=np.float64)
            seg_len = 1.0
        else:
            tangent = seg_vec / seg_len

        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

        # Sampling grids
        step_s = self.config.line_sample_step_px
        S = np.arange(0.0, seg_len + 0.5 * step_s, step_s, dtype=np.float64)
        if len(S) == 0:
            return None

        half_w = self.config.band_half_width_px
        step_d = self.config.band_step_px
        D = np.arange(-half_w, half_w + 0.5 * step_d, step_d, dtype=np.float64)
        if len(D) == 0:
            D = np.array([0.0], dtype=np.float64)

        # Broadcast coordinates to (N_s, N_d)
        U_grid = q1_clip[0] + S[:, None] * tangent[0] + D[None, :] * normal[0]
        V_grid = q1_clip[1] + S[:, None] * tangent[1] + D[None, :] * normal[1]

        IU_grid = np.rint(U_grid).astype(np.int32)
        IV_grid = np.rint(V_grid).astype(np.int32)

        valid_mask = (
            (IU_grid >= margin)
            & (IU_grid < W_dst - margin)
            & (IV_grid >= margin)
            & (IV_grid < H_dst - margin)
        )

        if not np.any(valid_mask):
            return None

        costs = np.full((len(S), len(D)), 999, dtype=np.int32)
        valid_codes = desc_map_dst[IV_grid[valid_mask], IU_grid[valid_mask]]
        diffs = np.bitwise_xor(code_src, valid_codes)
        costs[valid_mask] = popcount64(diffs)

        # Minimum cost across band offsets for each step along line trajectory
        cost_profile = np.min(costs, axis=1)
        best_d_idx_per_s = np.argmin(costs, axis=1)

        k_min = int(np.argmin(cost_profile))
        cost_min = int(cost_profile[k_min])
        d_min_idx = int(best_d_idx_per_s[k_min])

        if cost_min >= 999:
            return None

        # Filter 1: Absolute Hamming cost
        if cost_min > self.config.max_hamming_cost:
            return None

        # Filter 2: Uniqueness Ratio
        # Find second minimum at least min_second_peak_dist_px away
        u_best = U_grid[k_min, d_min_idx]
        v_best = V_grid[k_min, d_min_idx]
        dists = np.hypot(U_grid - u_best, V_grid - v_best)

        second_peak_mask = valid_mask & (dists >= self.config.min_second_peak_dist_px)
        if np.any(second_peak_mask):
            cost_second = int(np.min(costs[second_peak_mask]))
        else:
            cost_second = self.config.census_config.num_bits

        if cost_second <= 0:
            uniqueness_ratio = 1.0  # Identical duplicate match; ambiguous
        else:
            uniqueness_ratio = float(cost_min) / float(cost_second)

        if uniqueness_ratio > self.config.uniqueness_ratio_thresh:
            return None

        # Sub-pixel peak refinement along trajectory
        if self.config.subpixel_refinement and (0 < k_min < len(S) - 1):
            c_neg = float(cost_profile[k_min - 1])
            c_zero = float(cost_profile[k_min])
            c_pos = float(cost_profile[k_min + 1])
            delta = refine_subpixel_parabola(c_neg, c_zero, c_pos)
            s_refined = S[k_min] + delta * step_s
            u_j_sub = q1_clip[0] + s_refined * tangent[0] + D[d_min_idx] * normal[0]
            v_j_sub = q1_clip[1] + s_refined * tangent[1] + D[d_min_idx] * normal[1]
        else:
            u_j_sub = float(u_best)
            v_j_sub = float(v_best)

        epipolar_dist = float(abs(D[d_min_idx]))

        # Filter 3: Left-Right Consistency Check
        if check_left_right:
            F_dst_src = F_src_dst.T
            match_back = self._sample_band_and_find_match(
                cam_src=cam_dst,
                cam_dst=cam_src,
                pt_src=(u_j_sub, v_j_sub),
                desc_map_src=desc_map_dst,
                desc_map_dst=desc_map_src,
                F_src_dst=F_dst_src,
                check_left_right=False,
            )
            if match_back is None:
                return None

            dist_lr = math.hypot(match_back.pt_j[0] - u_src, match_back.pt_j[1] - v_src)
            if dist_lr > self.config.left_right_dist_thresh_px:
                return None

        return CensusMatch(
            cam_i=cam_src.cam_id,
            cam_j=cam_dst.cam_id,
            pt_i=(float(u_src), float(v_src)),
            pt_j=(float(u_j_sub), float(v_j_sub)),
            hamming_cost=cost_min,
            normalized_cost=float(cost_min) / float(self.config.census_config.num_bits),
            uniqueness_ratio=float(uniqueness_ratio),
            epipolar_dist_px=epipolar_dist,
            color_rgb=(180, 180, 180),
        )

    def extract_keypoints(self, gray_img: np.ndarray) -> List[Tuple[float, float]]:
        """
        Extracts prominent FAST keypoints within valid margin from image bounds.
        """
        fast = cv2.FastFeatureDetector_create(threshold=self.config.fast_threshold)
        kps = fast.detect(gray_img, None)
        H, W = gray_img.shape[:2]
        margin = self.config.border_margin_px + self.config.census_config.radius_u

        valid_kps = [
            k for k in kps
            if margin <= k.pt[0] < W - margin and margin <= k.pt[1] < H - margin
        ]
        valid_kps.sort(key=lambda k: k.response, reverse=True)
        return [(float(k.pt[0]), float(k.pt[1])) for k in valid_kps[:self.config.max_keypoints]]

    def match_camera_pair(
        self,
        cam_i: CameraModel,
        cam_j: CameraModel,
        frame_i: np.ndarray,
        frame_j: np.ndarray,
        keypoints_i: Optional[Sequence[Union[Tuple[float, float], cv2.KeyPoint]]] = None,
        frame_i_rgb: Optional[np.ndarray] = None,
        cached_desc_i: Optional[np.ndarray] = None,
        cached_desc_j: Optional[np.ndarray] = None,
    ) -> List[CensusMatch]:
        """
        Executes sector-based epipolar band Census matching for a pair of cameras.

        Args:
            cam_i: Source camera model.
            cam_j: Destination camera model.
            frame_i: Image from camera i (grayscale or color).
            frame_j: Image from camera j (grayscale or color).
            keypoints_i: Optional list of (u, v) or cv2.KeyPoint points in camera i.
            frame_i_rgb: Optional RGB frame for sampling vertex color.
            cached_desc_i: Optional precomputed uint64 descriptor map for camera i.
            cached_desc_j: Optional precomputed uint64 descriptor map for camera j.

        Returns:
            List of verified CensusMatch instances.
        """
        # Step 1: Sector & viewing angle verification
        viewing_angle_deg = compute_camera_viewing_angle_deg(cam_i, cam_j)
        if viewing_angle_deg > self.config.max_counter_facing_angle_deg:
            msg = (
                f"Counter-facing camera pair rejected for stereo matching: "
                f"cam {cam_i.cam_id} and cam {cam_j.cam_id} have viewing angle "
                f"{viewing_angle_deg:.2f}° > {self.config.max_counter_facing_angle_deg:.2f}°"
            )
            if self.config.strict_pair_validation:
                raise ValueError(msg)
            return []

        # Step 2: Compute Fundamental Matrix F_ij
        F_ij = compute_fundamental_matrix(cam_i, cam_j)

        # Step 3: Compute or retrieve Census descriptor maps
        desc_i = (
            cached_desc_i
            if cached_desc_i is not None
            else compute_census_descriptor_map(frame_i, self.config.census_config)
        )
        desc_j = (
            cached_desc_j
            if cached_desc_j is not None
            else compute_census_descriptor_map(frame_j, self.config.census_config)
        )

        # Step 4: Extract keypoints in camera i if not supplied
        if keypoints_i is not None:
            pts: List[Tuple[float, float]] = []
            for kp in keypoints_i:
                if isinstance(kp, cv2.KeyPoint):
                    pts.append((float(kp.pt[0]), float(kp.pt[1])))
                else:
                    pts.append((float(kp[0]), float(kp[1])))
        else:
            gray_i = frame_i if frame_i.ndim == 2 else cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
            pts = self.extract_keypoints(gray_i)

        if not pts:
            return []

        # Step 5: Match candidate points along epipolar band
        matches: List[CensusMatch] = []
        for pt_i in pts:
            match = self._sample_band_and_find_match(
                cam_src=cam_i,
                cam_dst=cam_j,
                pt_src=pt_i,
                desc_map_src=desc_i,
                desc_map_dst=desc_j,
                F_src_dst=F_ij,
                check_left_right=True,
            )
            if match is not None:
                # Assign sampled color if available
                u_int = int(round(match.pt_i[0]))
                v_int = int(round(match.pt_i[1]))
                if frame_i_rgb is not None:
                    color = tuple(int(c) for c in frame_i_rgb[v_int, u_int])
                elif frame_i.ndim == 3:
                    # Treat frame_i as RGB
                    color = tuple(int(c) for c in frame_i[v_int, u_int])
                else:
                    g = int(frame_i[v_int, u_int])
                    color = (g, g, g)

                # CensusMatch is frozen; reconstruct with sampled color
                color_match = CensusMatch(
                    cam_i=match.cam_i,
                    cam_j=match.cam_j,
                    pt_i=match.pt_i,
                    pt_j=match.pt_j,
                    hamming_cost=match.hamming_cost,
                    normalized_cost=match.normalized_cost,
                    uniqueness_ratio=match.uniqueness_ratio,
                    epipolar_dist_px=match.epipolar_dist_px,
                    color_rgb=color,
                )
                matches.append(color_match)

        return matches
