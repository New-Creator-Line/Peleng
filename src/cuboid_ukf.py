"""
Модуль 3–5 по руководству разработки:
1. Линейная N-лучевая триангуляция ортогональных расстояний (triangulate_n_rays) и RANSAC (ransac_triangulate).
2. 10D состояние 3D-параллелепипеда [X, Y, Z, v, yaw, omega, vz, L, W, H] с кинематикой CTRV,
   оценкой ковариационной матрицы P_XYZ, Mahalanobis gating и RTS-сглаживанием.
3. Доверительный 95% 3D-эллипсоид нахождения объекта (get_uncertainty_ellipsoid_3d, chi2_3,0.95 = 7.815).
4. Репроекция 8 вершин 3D-параллелепипеда в стабильный 2D Bounding Box (cuboid_to_camera_bbox).
"""

from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.camera_model import CameraModel
from src.truck_model import TruckModel


def triangulate_n_rays(
    origins: Sequence[np.ndarray],
    dirs: Sequence[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """
    Линейная триангуляция для произвольного числа лучей K >= 2.
    Минимизирует сумму квадратов ортогональных расстояний до всех активных лучей (C_i, d_i).
    Возвращает 3D-точку и среднеквадратичный зазор скрещивания лучей (skew_rmse).
    """
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    I = np.eye(3, dtype=np.float64)

    for c, d in zip(origins, dirs):
        d_unit = np.asarray(d, dtype=np.float64).reshape(3)
        d_unit = d_unit / (np.linalg.norm(d_unit) + 1e-12)
        d_col = d_unit.reshape(3, 1)
        proj = I - (d_col @ d_col.T)
        A += proj
        b += proj @ np.asarray(c, dtype=np.float64).reshape(3)

    point_3d, *_ = np.linalg.lstsq(A, b, rcond=None)

    dists_sq = []
    for c, d in zip(origins, dirs):
        c_vec = np.asarray(c, dtype=np.float64).reshape(3)
        d_unit = np.asarray(d, dtype=np.float64).reshape(3)
        d_unit = d_unit / (np.linalg.norm(d_unit) + 1e-12)
        diff = point_3d - c_vec
        perp = diff - np.dot(diff, d_unit) * d_unit
        dists_sq.append(float(np.dot(perp, perp)))

    skew_rmse = float(np.sqrt(np.mean(dists_sq))) if dists_sq else 0.0
    return point_3d, skew_rmse


def ransac_triangulate(
    origins: Sequence[np.ndarray],
    dirs: Sequence[np.ndarray],
    inlier_thresh: float = 350.0,
    min_parallax_deg: float = 12.0,
) -> Tuple[Optional[np.ndarray], List[int], float]:
    """
    Устойчивая триангуляция с отсевом камер-выбросов (RANSAC) и проверкой угла параллакса для произвольного K >= 2.
    Возвращает (point_3d, inlier_indices, skew_rmse).
    """
    k = len(origins)
    if k < 2:
        return None, [], float("inf")

    cos_max = float(np.cos(np.radians(min_parallax_deg)))
    unit_dirs = [np.asarray(d, dtype=np.float64) / (np.linalg.norm(d) + 1e-12) for d in dirs]
    origs = [np.asarray(c, dtype=np.float64) for c in origins]

    if k == 2:
        if abs(float(np.dot(unit_dirs[0], unit_dirs[1]))) > cos_max:
            return None, [], float("inf")
        pt, err = triangulate_n_rays(origs, unit_dirs)
        if err > inlier_thresh:
            return None, [], err
        if any(np.dot(pt - c, d) <= 50.0 for c, d in zip(origs, unit_dirs)):
            return None, [], err
        return pt, [0, 1], err

    best_inliers: List[int] = []
    best_err = float("inf")

    for i in range(k):
        for j in range(i + 1, k):
            if abs(float(np.dot(unit_dirs[i], unit_dirs[j]))) > cos_max:
                continue
            pt_pair, err_pair = triangulate_n_rays([origs[i], origs[j]], [unit_dirs[i], unit_dirs[j]])
            if err_pair > inlier_thresh:
                continue
            if np.dot(pt_pair - origs[i], unit_dirs[i]) <= 50.0 or np.dot(pt_pair - origs[j], unit_dirs[j]) <= 50.0:
                continue

            inliers: List[int] = []
            for idx, (c, d) in enumerate(zip(origs, unit_dirs)):
                diff = pt_pair - c
                if np.dot(diff, d) <= 50.0:
                    continue
                dist_to_ray = float(np.linalg.norm(diff - np.dot(diff, d) * d))
                if dist_to_ray < inlier_thresh:
                    inliers.append(idx)

            if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers) and err_pair < best_err):
                best_inliers = inliers
                best_err = err_pair

    if len(best_inliers) < 2:
        return None, [], float("inf")

    final_pt, final_skew = triangulate_n_rays(
        [origs[idx] for idx in best_inliers],
        [unit_dirs[idx] for idx in best_inliers],
    )
    return final_pt, best_inliers, final_skew


def get_uncertainty_ellipsoid_3d(
    cov_xyz: np.ndarray,
    chi2_95: float = 7.815,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает полуоси (a, b, c) и матрицу собственных векторов 95% 3D-эллипсоида неопределённости
    по ковариационной подматрице P_XYZ (3x3).
    """
    sym_cov = 0.5 * (cov_xyz + cov_xyz.T)
    eigvals, eigvecs = np.linalg.eigh(sym_cov)
    eigvals = np.maximum(eigvals, 1e-6)
    radii = np.sqrt(eigvals * chi2_95)
    return radii, eigvecs


def cuboid_world_corners(
    cabin_xyz: np.ndarray,
    yaw: float,
    truck: Optional[TruckModel] = None,
) -> np.ndarray:
    """
    Вычисляет 8 мировых вершин 3D-параллелепипеда грузовика по точке кабины (SM_AA_track pivot) и курсу yaw.
    Порядок вершин: 4 нижних [0..3], 4 верхних [4..7].
    """
    tm = truck or TruckModel()
    body_center = tm.center_from_pivot(cabin_xyz, yaw)
    hl, hw, hh = tm.length / 2.0, tm.width / 2.0, tm.height / 2.0
    local_pts = np.array([
        [-hl, -hw, -hh],
        [ hl, -hw, -hh],
        [ hl,  hw, -hh],
        [-hl,  hw, -hh],
        [-hl, -hw,  hh],
        [ hl, -hw,  hh],
        [ hl,  hw,  hh],
        [-hl,  hw,  hh],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (local_pts @ rot_z.T) + body_center.reshape(1, 3)


def cuboid_to_camera_bbox(
    state_10d: np.ndarray,
    cam: CameraModel,
    img_w: int = 1920,
    img_h: int = 1080,
    truck: Optional[TruckModel] = None,
) -> Optional[np.ndarray]:
    """
    Проецирует 3D-параллелепипед [X, Y, Z, v, yaw, omega, vz, L, W, H] в стабильный 2D BBox камеры.
    Возвращает [x_min, y_min, x_max, y_max] для наложения на контрольное видео.
    """
    X, Y, Z, _, yaw, _, _, L, W, H = state_10d[:10]
    tm = truck or TruckModel(length=float(L), width=float(W), height=float(H))
    world_pts = cuboid_world_corners(np.array([X, Y, Z], dtype=np.float64), float(yaw), tm)

    cam_pts = (world_pts @ cam.R_cam.T) + cam.t_cam.reshape(1, 3)
    if np.any(cam_pts[:, 2] <= 20.0):
        return None

    pts_2d = (cam_pts @ cam.K.T)[:, :2] / cam_pts[:, 2:3]
    x_min = float(np.clip(np.min(pts_2d[:, 0]), 0, img_w - 1))
    y_min = float(np.clip(np.min(pts_2d[:, 1]), 0, img_h - 1))
    x_max = float(np.clip(np.max(pts_2d[:, 0]), 0, img_w - 1))
    y_max = float(np.clip(np.max(pts_2d[:, 1]), 0, img_h - 1))
    if x_max - x_min < 2.0 or y_max - y_min < 2.0:
        return None
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float64)


def project_cuboid_wireframe(
    cabin_xyz: np.ndarray,
    yaw: float,
    cam: CameraModel,
    truck: Optional[TruckModel] = None,
) -> Optional[np.ndarray]:
    """Проецирует 8 вершин 3D-кубоида на плоскость камеры (8, 2), если кубоид перед камерой."""
    world_pts = cuboid_world_corners(cabin_xyz, yaw, truck)
    cam_pts = (world_pts @ cam.R_cam.T) + cam.t_cam.reshape(1, 3)
    if np.any(cam_pts[:, 2] <= 20.0):
        return None
    pts_2d = (cam_pts @ cam.K.T)[:, :2] / cam_pts[:, 2:3]
    return pts_2d

