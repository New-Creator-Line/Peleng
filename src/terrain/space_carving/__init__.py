"""
Space Carving & Visual Hull from Multi-View Silhouettes.

Implements Method 2 of the Peleng Visible-Scene Terrain Reconstruction Suite.
Conforms strictly to .memory_bank/contracts/terrain_method_contract.md.
"""

from typing import Any, Dict, Optional
import time
import numpy as np

from src.terrain.space_carving.silhouette import (
    detect_skyline_crest,
    create_sky_mask,
    extract_all_camera_silhouettes,
    extract_video_frame,
    generate_procedural_frame,
)
from src.terrain.space_carving.carver import (
    DEFAULT_SCENE_BOUNDS,
    VoxelVolume,
    carve_space_visual_hull,
    load_camera_models_or_defaults,
    get_default_camera_models,
)
from src.terrain.space_carving.marching_cubes import (
    compute_hypsometric_colors,
    compute_mesh_normals,
    extract_mesh_marching_cubes,
    sample_point_cloud,
)

__all__ = [
    "reconstruct_terrain",
    "to_scene_payload_layer",
    "detect_skyline_crest",
    "create_sky_mask",
    "extract_all_camera_silhouettes",
    "carve_space_visual_hull",
    "extract_mesh_marching_cubes",
    "compute_mesh_normals",
    "compute_hypsometric_colors",
    "sample_point_cloud",
    "VoxelVolume",
    "DEFAULT_SCENE_BOUNDS",
]


def reconstruct_terrain(
    calib_dir: str = "хакатон",
    video_dir: str = "хакатон",
    frame_idx: int = 0,
    step_xy: float = 250.0,
    step_z: float = 125.0,
    coarse_to_fine: bool = True,
) -> Dict[str, Any]:
    """
    Executes Method 2: Space Carving & Visual Hull from Multi-View Silhouettes.
    
    1. Extracts frame 0 silhouettes across 5 cameras using vertical luminance gradient
       and Canny edge filtering (zero color rules).
    2. Builds coarse-to-fine 3D voxel grid over scene bounds [-40000, 20000] x [-30000, 30000] x [0, 5000] cm.
    3. Projects voxels into all cameras; carves out voxels that project to sky on ANY camera in-FOV.
    4. Extracts watertight 3D triangle surface mesh and smooth normal vectors using marching cubes.
    5. Returns TerrainPayload strictly conforming to terrain_method_contract.md.
    
    Args:
        calib_dir: Directory containing camera calibration CSVs.
        video_dir: Directory containing camera video MP4s.
        frame_idx: Frame index for static silhouette extraction (default: 0).
        step_xy: Voxel resolution in X and Y (default: 250 cm = 2.5 m).
        step_z: Voxel resolution in Z (default: 125 cm = 1.25 m).
        coarse_to_fine: Multi-resolution coarse-to-fine acceleration flag.
        
    Returns:
        TerrainPayload conforming to .memory_bank/contracts/terrain_method_contract.md.
    """
    t_start = time.perf_counter()

    # 1. Load Camera Models (with defensive procedural fallbacks)
    cameras = load_camera_models_or_defaults(calib_dir=calib_dir, frame_idx=frame_idx)
    cam_ids = sorted(cameras.keys()) if cameras else [1, 2, 3, 4, 5]

    # 2. Extract multi-camera skyline silhouettes (zero color rules)
    silhouettes = extract_all_camera_silhouettes(
        video_dir=video_dir,
        cam_ids=cam_ids,
        frame_idx=frame_idx,
    )

    # 3. 3D Voxel Carving (coarse-to-fine visual hull)
    voxel_volume = carve_space_visual_hull(
        cameras=cameras,
        silhouettes=silhouettes,
        bounds=DEFAULT_SCENE_BOUNDS,
        step_xy=step_xy,
        step_z=step_z,
        coarse_to_fine=coarse_to_fine,
        coarse_factor=2,
    )

    # 4. Marching Cubes Surface Extraction
    vertices, faces, normals = extract_mesh_marching_cubes(
        voxel_volume=voxel_volume,
        level=0.5,
        bounds=DEFAULT_SCENE_BOUNDS,
    )

    # 5. Hypsometric Elevation Vertex Coloring
    colors = compute_hypsometric_colors(vertices, bounds=DEFAULT_SCENE_BOUNDS)

    # 6. Surface Landmark Point Cloud
    point_cloud = sample_point_cloud(vertices, max_points=12000)

    t_end = time.perf_counter()
    runtime_ms = (t_end - t_start) * 1000.0

    z_min = float(np.min(vertices[:, 2])) if len(vertices) > 0 else 0.0
    z_max = float(np.max(vertices[:, 2])) if len(vertices) > 0 else 0.0

    payload: Dict[str, Any] = {
        "method_name": "space_carving",
        "vertices": vertices,
        "faces": faces,
        "colors": colors,
        "normals": normals,
        "point_cloud": point_cloud,
        "density": "watertight_hull",
        "stats": {
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "runtime_ms": round(float(runtime_ms), 2),
            "z_min_cm": round(z_min, 2),
            "z_max_cm": round(z_max, 2),
        },
    }

    return payload


def to_scene_payload_layer(terrain_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts TerrainPayload into a layer dict ingestible by Peleng/src/viewer_3d.py.
    """
    verts = terrain_payload.get("vertices")
    faces = terrain_payload.get("faces")
    colors = terrain_payload.get("colors")
    normals = terrain_payload.get("normals")
    pts = terrain_payload.get("point_cloud")

    verts_list = verts.tolist() if isinstance(verts, np.ndarray) else (verts or [])
    faces_list = faces.tolist() if isinstance(faces, np.ndarray) else (faces or [])
    colors_list = colors.tolist() if isinstance(colors, np.ndarray) else (colors or [])
    normals_list = normals.tolist() if isinstance(normals, np.ndarray) else (normals or [])
    pts_list = pts.tolist() if isinstance(pts, np.ndarray) else (pts or verts_list)

    return {
        "layer_name": terrain_payload.get("method_name", "space_carving"),
        "density": terrain_payload.get("density", "watertight_hull"),
        "stats": dict(terrain_payload.get("stats", {})),
        "mesh": {
            "vertices": verts_list,
            "faces": faces_list,
            "colors": colors_list,
            "normals": normals_list,
        },
        "point_cloud": {
            "points_cm": pts_list,
            "colors_rgb": colors_list[:len(pts_list)] if colors_list else [],
        },
    }
