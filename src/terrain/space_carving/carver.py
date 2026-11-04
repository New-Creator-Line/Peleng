"""
3D Voxel Grid Carving & Multi-Camera Visual Hull Reconstruction.

Constructs a 3D voxel volume over UE5 scene bounds [-40000, 20000] x [-30000, 30000] x [0, 5000] cm,
projects voxels into all calibrated cameras, and carves out voxels that project to sky
on ANY camera in-FOV (Visual Hull Intersection).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.camera_model import CameraModel, CameraTimeline, load_all_cameras


# Default Scene Bounds in Unreal Engine 5 centimeters (UE5_cm)
DEFAULT_SCENE_BOUNDS: Dict[str, float] = {
    "x_min": -40000.0,
    "x_max": 20000.0,
    "y_min": -30000.0,
    "y_max": 30000.0,
    "z_min": 0.0,
    "z_max": 5000.0,
}


@dataclass
class VoxelVolume:
    volume: np.ndarray             # 3D float32 array (Nx, Ny, Nz), 1.0 = solid, 0.0 = carved
    xs: np.ndarray                 # 1D coordinates of X grid lines
    ys: np.ndarray                 # 1D coordinates of Y grid lines
    zs: np.ndarray                 # 1D coordinates of Z grid lines
    spacing: Tuple[float, float, float]  # (dx, dy, dz)
    bounds: Dict[str, float]


def get_default_camera_models() -> Dict[int, CameraModel]:
    """
    Provides deterministic baseline camera models matching the Hackathon setup
    if CSV calibrations are missing or unreadable.
    """
    default_cams_spec = {
        1: {"loc": [-24701.96, 6623.57, 1200.0], "rot": [0.0, 0.0, -70.0], "focal": 35.0},
        2: {"loc": [-45784.00, -6500.0, 7911.88], "rot": [0.0, -10.0, 0.0], "focal": 60.0},
        3: {"loc": [12600.0, -22700.0, 1400.0], "rot": [-0.68, -3.11, 115.11], "focal": 35.0},
        4: {"loc": [33281.73, 13996.66, 7146.74], "rot": [0.0, -10.0, -160.0], "focal": 50.0},
        5: {"loc": [9799.81, -3900.0, 1487.53], "rot": [-4.41, -6.97, -164.0], "focal": 20.0},
    }
    models: Dict[int, CameraModel] = {}
    for cid, spec in default_cams_spec.items():
        models[cid] = CameraModel(
            cam_id=cid,
            loc=np.array(spec["loc"], dtype=np.float64),
            rot=np.array(spec["rot"], dtype=np.float64),
            focal_length_mm=spec["focal"],
        )
    return models


def load_camera_models_or_defaults(
    calib_dir: str = "хакатон",
    frame_idx: int = 0,
) -> Dict[int, CameraModel]:
    """
    Loads camera models for a specified frame_idx from calib_dir,
    falling back to standard baseline parameters if not found.
    """
    models: Dict[int, CameraModel] = {}
    try:
        timelines = load_all_cameras(calib_dir)
        for cid, timeline in timelines.items():
            models[cid] = timeline.get_model(frame_idx)
    except Exception:
        pass

    if not models:
        models = get_default_camera_models()
    return models


def _project_and_carve_points(
    points_xyz: np.ndarray,
    cameras: Dict[int, CameraModel],
    silhouettes: Dict[int, Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Vectorized evaluation of Visual Hull condition on an (N, 3) array of 3D points.
    
    Returns:
        carved_mask: boolean ndarray of shape (N,), where True means the point
                     projects into the sky on at least one camera in FOV (carved out).
    """
    n_pts = len(points_xyz)
    if n_pts == 0:
        return np.zeros(0, dtype=bool)

    # Homogeneous coordinates (N, 4)
    pts_h = np.column_stack([points_xyz, np.ones(n_pts, dtype=np.float32)])
    carved_mask = np.zeros(n_pts, dtype=bool)

    for cid, cam_model in cameras.items():
        if cid not in silhouettes:
            continue

        crest = silhouettes[cid]["crest"]  # Shape (W,)
        w_img = float(len(crest))
        h_img = float(silhouettes[cid]["mask"].shape[0])

        P = cam_model.get_projection_matrix().astype(np.float32)  # (3, 4)
        cam_pts = pts_h @ P.T  # (N, 3)

        z_cam = cam_pts[:, 2]
        # Only evaluate points strictly in front of the camera optical center
        valid = z_cam > 1.0

        u = np.empty_like(z_cam)
        v = np.empty_like(z_cam)
        u[valid] = cam_pts[valid, 0] / z_cam[valid]
        v[valid] = cam_pts[valid, 1] / z_cam[valid]

        # Points within camera field of view
        in_fov = valid & (u >= 0.0) & (u < w_img) & (v >= 0.0) & (v < h_img)

        if not np.any(in_fov):
            continue

        u_fov = u[in_fov]
        v_fov = v[in_fov]

        # Vectorized crest lookup for columns u
        u_indices = np.clip(np.round(u_fov).astype(np.int32), 0, int(w_img) - 1)
        crest_thresh = crest[u_indices]

        # Sky test: v < crest
        is_sky = v_fov < crest_thresh

        # Mark carved points
        in_fov_indices = np.where(in_fov)[0]
        carved_indices = in_fov_indices[is_sky]
        carved_mask[carved_indices] = True

    return carved_mask


def carve_space_visual_hull(
    cameras: Dict[int, CameraModel],
    silhouettes: Dict[int, Dict[str, np.ndarray]],
    bounds: Optional[Dict[str, float]] = None,
    step_xy: float = 250.0,
    step_z: float = 125.0,
    coarse_to_fine: bool = True,
    coarse_factor: int = 2,
) -> VoxelVolume:
    """
    Executes coarse-to-fine space carving over the 3D voxel grid.
    
    Args:
        cameras: Dict mapping cam_id -> CameraModel.
        silhouettes: Dict mapping cam_id -> {"crest": ndarray, "mask": ndarray, ...}.
        bounds: Bounding box dictionary (x_min, x_max, y_min, y_max, z_min, z_max).
        step_xy: Grid resolution in X and Y directions (UE5_cm).
        step_z: Grid resolution in Z direction (UE5_cm).
        coarse_to_fine: Whether to accelerate evaluation via multi-resolution carving.
        coarse_factor: Downsampling factor for the initial coarse pass.
        
    Returns:
        VoxelVolume containing the 3D carved occupancy grid and spatial coordinate axes.
    """
    if bounds is None:
        bounds = DEFAULT_SCENE_BOUNDS

    x_min, x_max = float(bounds["x_min"]), float(bounds["x_max"])
    y_min, y_max = float(bounds["y_min"]), float(bounds["y_max"])
    z_min, z_max = float(bounds["z_min"]), float(bounds["z_max"])

    # Fine grid coordinate axes
    xs = np.arange(x_min, x_max + 0.5 * step_xy, step_xy, dtype=np.float32)
    ys = np.arange(y_min, y_max + 0.5 * step_xy, step_xy, dtype=np.float32)
    zs = np.arange(z_min, z_max + 0.5 * step_z, step_z, dtype=np.float32)

    nx, ny, nz = len(xs), len(ys), len(zs)
    vol = np.ones((nx, ny, nz), dtype=np.float32)

    if coarse_to_fine and coarse_factor > 1:
        # 1. Coarse Pass
        c_step_xy = step_xy * coarse_factor
        c_step_z = step_z * coarse_factor
        c_xs = np.arange(x_min, x_max + 0.5 * c_step_xy, c_step_xy, dtype=np.float32)
        c_ys = np.arange(y_min, y_max + 0.5 * c_step_xy, c_step_xy, dtype=np.float32)
        c_zs = np.arange(z_min, z_max + 0.5 * c_step_z, c_step_z, dtype=np.float32)

        c_X, c_Y, c_Z = np.meshgrid(c_xs, c_ys, c_zs, indexing="ij")
        c_pts = np.column_stack([c_X.ravel(), c_Y.ravel(), c_Z.ravel()])

        c_carved = _project_and_carve_points(c_pts, cameras, silhouettes)
        c_vol = np.ones(c_X.shape, dtype=bool)
        c_vol.ravel()[c_carved] = False

        # Identify coarse voxels that are guaranteed empty (carved) and whose neighbors are also carved
        # Dilate the occupied coarse voxels to form the candidate region for fine evaluation
        from scipy.ndimage import binary_dilation
        candidate_coarse = binary_dilation(c_vol, iterations=1)

        # 2. Fine Pass: only evaluate fine voxels inside candidate coarse cells
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        fine_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Map fine coordinates to coarse cell indices
        c_ix = np.clip(np.floor((fine_pts[:, 0] - x_min) / c_step_xy).astype(np.int32), 0, c_vol.shape[0] - 1)
        c_iy = np.clip(np.floor((fine_pts[:, 1] - y_min) / c_step_xy).astype(np.int32), 0, c_vol.shape[1] - 1)
        c_iz = np.clip(np.floor((fine_pts[:, 2] - z_min) / c_step_z).astype(np.int32), 0, c_vol.shape[2] - 1)

        candidate_fine_mask = candidate_coarse[c_ix, c_iy, c_iz]

        # Points outside candidate_coarse are guaranteed sky
        carved_all = ~candidate_fine_mask

        # Evaluate only candidate fine points
        candidate_indices = np.where(candidate_fine_mask)[0]
        if len(candidate_indices) > 0:
            fine_candidate_pts = fine_pts[candidate_indices]
            fine_carved = _project_and_carve_points(fine_candidate_pts, cameras, silhouettes)
            carved_all[candidate_indices[fine_carved]] = True

        vol.ravel()[carved_all] = 0.0

    else:
        # Direct unified fine pass
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        carved = _project_and_carve_points(pts, cameras, silhouettes)
        vol.ravel()[carved] = 0.0

    # Ensure baseline ground connectivity: bottom-most layer (Z = z_min) is solid ground
    vol[:, :, 0] = 1.0

    return VoxelVolume(
        volume=vol,
        xs=xs,
        ys=ys,
        zs=zs,
        spacing=(step_xy, step_xy, step_z),
        bounds=bounds,
    )
