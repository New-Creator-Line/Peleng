"""
Unit Tests for Task 015: Space Carving & Visual Hull from Multi-View Silhouettes.

Verifies:
1. Silhouette / Skyline crest detection with zero color rules.
2. Binary sky/terrain mask generation.
3. 3D Voxel carving and camera projection.
4. Marching Cubes surface mesh and normal extraction.
5. Strict compliance with TerrainPayload schema (terrain_method_contract.md).
6. Acceptance criteria:
   - Vertices > 1,000, Faces > 2,000.
   - All vertices satisfy 0 <= Z <= 5000 cm.
   - Total runtime < 3.5 seconds on CPU.
   - 100% defensive fallbacks when directories/videos are missing.
"""

import time
import numpy as np
import pytest

from src.camera_model import CameraModel
from src.terrain.space_carving import (
    DEFAULT_SCENE_BOUNDS,
    VoxelVolume,
    carve_space_visual_hull,
    compute_hypsometric_colors,
    compute_mesh_normals,
    create_sky_mask,
    detect_skyline_crest,
    extract_all_camera_silhouettes,
    extract_mesh_marching_cubes,
    generate_procedural_frame,
    get_default_camera_models,
    reconstruct_terrain,
    sample_point_cloud,
    to_scene_payload_layer,
)


class TestSilhouetteExtraction:
    """Verifies silhouette and skyline crest detection."""

    def test_detect_skyline_crest_synthetic(self) -> None:
        """Synthetic image with known crest at y=400."""
        height, width = 1080, 1920
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky: rows 0..399 bright
        img[:400, :] = [230, 210, 180]
        # Terrain: rows 400..1079 darker with texture
        img[400:, :] = [60, 95, 70]
        # Add slight texture to terrain
        noise = np.random.randint(-15, 15, (height - 400, width, 3), dtype=np.int16)
        img[400:, :] = np.clip(img[400:, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        crest = detect_skyline_crest(img, canny_thresh_low=20.0, canny_thresh_high=80.0)

        assert isinstance(crest, np.ndarray)
        assert crest.shape == (width,)
        assert crest.dtype == np.float32
        # Crest should be detected near row 400 (+/- 5 pixels)
        assert np.abs(np.median(crest) - 400.0) < 5.0
        assert (crest >= 0.0).all() and (crest < float(height)).all()

    def test_detect_skyline_crest_empty_or_flat_fallback(self) -> None:
        """Completely flat image (no edges) should fall back gracefully without error."""
        flat_img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        crest = detect_skyline_crest(flat_img)

        assert crest.shape == (1920,)
        assert np.all(crest > 0.0)
        assert np.all(crest < 1080.0)

    def test_create_sky_mask(self) -> None:
        """Binary mask generation from crest curve."""
        crest = np.full(1920, 300.0, dtype=np.float32)
        mask = create_sky_mask(crest, height=1080, width=1920)

        assert mask.shape == (1080, 1920)
        assert mask.dtype == np.uint8
        # Sky (v < 300) should be 0
        assert (mask[:290, :] == 0).all()
        # Terrain (v >= 300) should be 1
        assert (mask[310:, :] == 1).all()

    def test_generate_procedural_frame(self) -> None:
        """Procedural frame generator produces deterministic valid RGB images."""
        frame = generate_procedural_frame(cam_id=1, width=1920, height=1080)
        assert frame.shape == (1080, 1920, 3)
        assert frame.dtype == np.uint8
        assert frame.min() >= 0 and frame.max() <= 255


class TestVoxelCarver:
    """Verifies 3D voxel grid definition, camera projection, and visual hull carving."""

    def test_carve_space_visual_hull_geometry(self) -> None:
        """Carving with synthetic camera correctly removes sky voxels."""
        cams = get_default_camera_models()
        assert len(cams) >= 5

        # Create synthetic silhouettes with horizontal crest at row 540 (half sky, half terrain)
        silhouettes = {}
        for cid in cams.keys():
            crest = np.full(1920, 540.0, dtype=np.float32)
            mask = create_sky_mask(crest, 1080, 1920)
            silhouettes[cid] = {"crest": crest, "mask": mask}

        # Carve with moderate resolution
        voxel_vol = carve_space_visual_hull(
            cameras=cams,
            silhouettes=silhouettes,
            step_xy=500.0,
            step_z=250.0,
            coarse_to_fine=True,
            coarse_factor=2,
        )

        assert isinstance(voxel_vol, VoxelVolume)
        assert voxel_vol.volume.ndim == 3
        # Volume should have carved out sky space while keeping terrain solid
        occupied_ratio = float(np.mean(voxel_vol.volume))
        assert 0.10 < occupied_ratio < 0.95
        # Bottom ground layer must be solid ground
        assert (voxel_vol.volume[:, :, 0] == 1.0).all()

    def test_voxel_bounds_consistency(self) -> None:
        """Verify grid coordinate axes match default scene bounds."""
        cams = get_default_camera_models()
        silhouettes = {
            1: {"crest": np.full(1920, 400.0, dtype=np.float32), "mask": np.zeros((1080, 1920), dtype=np.uint8)}
        }
        bounds = {"x_min": -20000.0, "x_max": 10000.0, "y_min": -10000.0, "y_max": 10000.0, "z_min": 0.0, "z_max": 2000.0}
        vol = carve_space_visual_hull(cams, silhouettes, bounds=bounds, step_xy=1000.0, step_z=500.0, coarse_to_fine=False)

        assert vol.xs[0] == bounds["x_min"]
        assert vol.xs[-1] == bounds["x_max"]
        assert vol.ys[0] == bounds["y_min"]
        assert vol.ys[-1] == bounds["y_max"]
        assert vol.zs[0] == bounds["z_min"]
        assert vol.zs[-1] == bounds["z_max"]


class TestMarchingCubesAndNormals:
    """Verifies surface mesh extraction, normals, and hypsometric coloring."""

    def test_marching_cubes_watertight_mesh(self) -> None:
        """Marching cubes extracts valid watertight mesh from synthetic solid block."""
        nx, ny, nz = 30, 30, 20
        vol_arr = np.zeros((nx, ny, nz), dtype=np.float32)
        # Solid ground mound in the center
        vol_arr[5:25, 5:25, :12] = 1.0

        xs = np.linspace(-40000.0, 20000.0, nx, dtype=np.float32)
        ys = np.linspace(-30000.0, 30000.0, ny, dtype=np.float32)
        zs = np.linspace(0.0, 5000.0, nz, dtype=np.float32)
        dx = float(xs[1] - xs[0])
        dy = float(ys[1] - ys[0])
        dz = float(zs[1] - zs[0])

        vol = VoxelVolume(
            volume=vol_arr,
            xs=xs,
            ys=ys,
            zs=zs,
            spacing=(dx, dy, dz),
            bounds=DEFAULT_SCENE_BOUNDS,
        )

        verts, faces, normals = extract_mesh_marching_cubes(vol, level=0.5)

        assert len(verts) > 1000, f"Expected > 1,000 vertices, got {len(verts)}"
        assert len(faces) > 2000, f"Expected > 2,000 faces, got {len(faces)}"
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        assert normals.shape == verts.shape

        # All vertices must strictly lie within scene bounds
        assert (verts[:, 0] >= DEFAULT_SCENE_BOUNDS["x_min"]).all()
        assert (verts[:, 0] <= DEFAULT_SCENE_BOUNDS["x_max"]).all()
        assert (verts[:, 1] >= DEFAULT_SCENE_BOUNDS["y_min"]).all()
        assert (verts[:, 1] <= DEFAULT_SCENE_BOUNDS["y_max"]).all()
        assert (verts[:, 2] >= 0.0).all(), f"Z min violation: {verts[:, 2].min()}"
        assert (verts[:, 2] <= 5000.0).all(), f"Z max violation: {verts[:, 2].max()}"

        # Normals must be normalized unit vectors
        lengths = np.linalg.norm(normals, axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-3)

    def test_compute_mesh_normals_pure_numpy(self) -> None:
        """Pure NumPy normal computation on a simple tetrahedron."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ], dtype=np.int32)

        normals = compute_mesh_normals(verts, faces)
        assert normals.shape == (4, 3)
        lengths = np.linalg.norm(normals, axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-3)

    def test_hypsometric_colors(self) -> None:
        """Elevation coloring correctly assigns RGB colors."""
        verts = np.array([
            [0.0, 0.0, 50.0],     # Valley low
            [0.0, 0.0, 2500.0],   # Mid elevation
            [0.0, 0.0, 4800.0],   # High crest
        ], dtype=np.float32)
        colors = compute_hypsometric_colors(verts, bounds=DEFAULT_SCENE_BOUNDS)

        assert colors.shape == (3, 3)
        assert colors.dtype == np.uint8
        # Valley is more green (G >= R, G >= B)
        assert colors[0, 1] >= colors[0, 0]
        # High crest is rock slate (balanced gray tones)
        assert colors[2, 0] > 130 and colors[2, 1] > 130


class TestTerrainMethodContractCompliance:
    """Verifies complete end-to-end reconstruct_terrain compliance with terrain_method_contract.md."""

    def test_reconstruct_terrain_contract_keys_and_schema(self) -> None:
        """Full pipeline execution against real dataset or procedural fallback."""
        t0 = time.perf_counter()
        payload = reconstruct_terrain(calib_dir="хакатон", video_dir="хакатон", frame_idx=0)
        dt = time.perf_counter() - t0

        # 1. Required Top-Level Keys
        required_keys = ["method_name", "vertices", "faces", "colors", "point_cloud", "density", "stats"]
        for k in required_keys:
            assert k in payload, f"Missing required contract key: {k}"

        # 2. Key Values and Shapes
        assert payload["method_name"] == "space_carving"
        assert payload["density"] == "watertight_hull"

        vertices = payload["vertices"]
        faces = payload["faces"]
        colors = payload["colors"]
        point_cloud = payload["point_cloud"]
        stats = payload["stats"]

        assert isinstance(vertices, np.ndarray)
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert vertices.dtype in (np.float32, np.float64)

        assert isinstance(faces, np.ndarray)
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert faces.dtype in (np.int32, np.int64)

        assert isinstance(colors, np.ndarray)
        assert colors.shape == vertices.shape
        assert colors.dtype == np.uint8

        assert isinstance(point_cloud, np.ndarray)
        assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3

        # 3. Acceptance Criteria Counts
        assert len(vertices) > 1000, f"Vertices count {len(vertices)} <= 1000"
        assert len(faces) > 2000, f"Faces count {len(faces)} <= 2000"

        # 4. Acceptance Criteria Z-bounds: 0 <= Z <= 5000 cm
        z_min = float(np.min(vertices[:, 2]))
        z_max = float(np.max(vertices[:, 2]))
        assert z_min >= 0.0, f"Vertex Z below 0 cm: {z_min}"
        assert z_max <= 5000.0, f"Vertex Z above 5000 cm: {z_max}"

        # 5. Stats Dictionary Compliance
        assert stats["vertex_count"] == len(vertices)
        assert stats["face_count"] == len(faces)
        assert stats["z_min_cm"] == round(z_min, 2)
        assert stats["z_max_cm"] == round(z_max, 2)
        assert isinstance(stats["runtime_ms"], (float, int))

        # 6. Performance Target: Runtime < 3.5 seconds on CPU
        assert dt < 3.5, f"Execution took {dt:.2f}s, exceeding 3.5s limit"

    def test_reconstruct_terrain_missing_directories_graceful_fallback(self) -> None:
        """Pipeline must never crash when directories or files are missing."""
        payload = reconstruct_terrain(calib_dir="non_existent_dir_12345", video_dir="non_existent_dir_12345")

        assert payload["method_name"] == "space_carving"
        assert payload["density"] == "watertight_hull"
        assert len(payload["vertices"]) > 1000
        assert len(payload["faces"]) > 2000
        assert (payload["vertices"][:, 2] >= 0.0).all()
        assert (payload["vertices"][:, 2] <= 5000.0).all()

    def test_to_scene_payload_layer_compatibility(self) -> None:
        """Viewer 3D integration helper produces valid JSON-serializable layer dict."""
        payload = reconstruct_terrain(calib_dir="хакатон", video_dir="хакатон", frame_idx=0)
        layer = to_scene_payload_layer(payload)

        assert layer["layer_name"] == "space_carving"
        assert layer["density"] == "watertight_hull"
        assert "mesh" in layer
        assert "point_cloud" in layer
        assert "stats" in layer
        assert isinstance(layer["mesh"]["vertices"], list)
        assert isinstance(layer["mesh"]["faces"], list)
        assert isinstance(layer["mesh"]["colors"], list)
        assert isinstance(layer["point_cloud"]["points_cm"], list)
        assert isinstance(layer["point_cloud"]["colors_rgb"], list)
        assert len(layer["mesh"]["vertices"]) == len(payload["vertices"])
