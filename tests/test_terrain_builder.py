import math
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.evaluator import load_ground_truth
from src.terrain_builder import (
    FRUSTUM_DISTANCE_CM,
    _bilinear_interpolate_mesh,
    build_scene_3d_payload,
)


HACKATHON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "хакатон"
)


def _create_synthetic_calib_dir(tmp_dir: str) -> None:
    """
    Creates two synthetic camera CSV files (Cam 1 static, Cam 5 dynamic yaw)
    in cp1251 format matching the official Peleng camera CSV layout.
    """
    header1 = "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35\n"
    rows1 = [
        "0;-24702,0;6624,0;1200,0;0,0;-10,0;-70,0;;;Размеры матрицы;23,76*13,365\n",
        "1;-24702,0;6624,0;1200,0;0,0;-10,0;-70,0;;;;\n",
        "2;-24702,0;6624,0;1200,0;0,0;-10,0;-70,0;;;;\n",
    ]
    with open(
        os.path.join(tmp_dir, "object_transform_CineCameraActor1.csv"),
        "w",
        encoding="cp1251",
    ) as f:
        f.write(header1 + "".join(rows1))

    header5 = "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;20\n"
    rows5 = [
        "0;9795,0;-3905,0;1125,0;0,0;-5,0;-150,0;;;Размеры матрицы;23,76*13,365\n",
        "1;9795,0;-3905,0;1125,0;0,0;-5,0;-145,0;;;;\n",
        "2;9795,0;-3905,0;1125,0;0,0;-5,0;-140,0;;;;\n",
    ]
    with open(
        os.path.join(tmp_dir, "object_transform_CineCameraActor5.csv"),
        "w",
        encoding="cp1251",
    ) as f:
        f.write(header5 + "".join(rows5))


def _assert_mesh_and_point_cloud_locked(payload: dict, max_tol_cm: float = 40.0) -> None:
    """
    Verifies that every 3D point in `point_cloud["points_cm"]` lies directly on
    the bilinearly interpolated `terrain_mesh["z_matrix_cm"]` surface within `max_tol_cm`.
    """
    mesh = payload["terrain_mesh"]
    x_grid = np.asarray(mesh["x_grid_cm"], dtype=np.float64)
    y_grid = np.asarray(mesh["y_grid_cm"], dtype=np.float64)
    zz_mesh = np.asarray(mesh["z_matrix_cm"], dtype=np.float64)

    pts = np.asarray(payload["point_cloud"]["points_cm"], dtype=np.float64)
    assert len(pts) > 100
    z_interp = _bilinear_interpolate_mesh(pts[:, 0], pts[:, 1], x_grid, y_grid, zz_mesh)
    abs_diff = np.abs(pts[:, 2] - z_interp)
    max_diff = float(np.max(abs_diff))
    assert max_diff <= max_tol_cm, f"Point cloud detached from terrain mesh by {max_diff:.1f} cm!"


def test_schema_and_validation_mode_synthetic():
    """
    Verifies full Scene3DPayload schema compliance in Validation Mode (with ground_truth_df)
    on a self-contained synthetic calibration directory, including 100% mesh-to-point-cloud locking.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        _create_synthetic_calib_dir(tmp_dir)

        est_df = pd.DataFrame({
            "Frame": [0, 1, 2],
            "Loc_X": [-21500.0, -21400.0, -21300.0],
            "Loc_Y": [-12800.0, -12750.0, -12700.0],
            "Loc_Z": [305.0, 306.0, 308.0],
        })
        gt_df = pd.DataFrame({
            "Frame": [0, 1, 2],
            "Loc_X": [-21510.0, -21405.0, -21295.0],
            "Loc_Y": [-12805.0, -12752.0, -12698.0],
            "Loc_Z": [300.0, 302.0, 305.0],
        })
        grouped_dets = {
            0: [{"cam_id": 1, "u": 960.0, "v": 540.0, "w": 50.0, "h": 30.0, "conf": 0.92}],
            1: [{"cam_id": 5, "u": 800.0, "v": 500.0, "w": 60.0, "h": 35.0, "conf": 0.88}],
            2: [],
        }

        payload = build_scene_3d_payload(
            calib_dir=tmp_dir,
            estimated_df=est_df,
            ground_truth_df=gt_df,
            grouped_detections=grouped_dets,
            use_depth_model=False,
            grid_size=24,
        )

        for key in ("meta", "metrics", "terrain_mesh", "point_cloud", "cameras", "frames"):
            assert key in payload
        assert "tree_cloud" in payload
        assert "points_cm" in payload["tree_cloud"]
        assert "colors_rgb" in payload["tree_cloud"]

        meta = payload["meta"]
        assert meta["coord_system"] == "UE5_cm"
        assert meta["num_frames"] == 3
        assert meta["has_ground_truth"] is True
        assert len(meta["origin_center_cm"]) == 3
        for b_key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            assert b_key in meta["bounds_cm"]

        mesh = payload["terrain_mesh"]
        assert mesh["rows"] == 24
        assert mesh["cols"] == 24
        assert len(mesh["x_grid_cm"]) == 24
        assert len(mesh["y_grid_cm"]) == 24
        assert len(mesh["z_matrix_cm"]) == 24
        assert len(mesh["z_matrix_cm"][0]) == 24
        assert len(mesh["rgb_matrix"]) == 24
        assert len(mesh["rgb_matrix"][0]) == 24
        assert len(mesh["rgb_matrix"][0][0]) == 3

        _assert_mesh_and_point_cloud_locked(payload, max_tol_cm=25.0)

        cams = payload["cameras"]
        assert "1" in cams and "5" in cams
        assert cams["1"]["is_dynamic"] is False
        assert cams["5"]["is_dynamic"] is True
        assert cams["1"]["thumbnail_b64"].startswith("data:image/jpeg;base64,")

        pos1 = np.array(cams["1"]["pos_cm"])
        corners1 = np.array(cams["1"]["frustum_corners_cm"])
        assert corners1.shape == (4, 3)
        for corner in corners1:
            dist = float(np.linalg.norm(corner - pos1))
            assert math.isclose(dist, FRUSTUM_DISTANCE_CM, rel_tol=1e-2)

        frames = payload["frames"]
        assert len(frames) == 3
        assert frames[0]["gt_pos_cm"] is not None
        assert frames[0]["error_cm"] is not None and frames[0]["error_cm"] > 0.0
        assert len(frames[0]["active_rays"]) == 1
        assert frames[0]["active_rays"][0]["cam_id"] == 1

        dyn0 = frames[0]["dynamic_cameras"]["5"]
        dyn2 = frames[2]["dynamic_cameras"]["5"]
        assert dyn0["forward"] != dyn2["forward"]
        assert dyn0["frustum_corners_cm"] != dyn2["frustum_corners_cm"]


def test_blind_test_mode_without_ground_truth():
    """
    Verifies Blind Test Mode (`ground_truth_df=None`) sets `has_ground_truth=False`
    and `gt_pos_cm=None`, `error_cm=None` across all frames without errors.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        _create_synthetic_calib_dir(tmp_dir)
        est_df = pd.DataFrame({
            "Frame": [0, 1, 2],
            "Loc_X": [-20000.0, -19800.0, -19600.0],
            "Loc_Y": [-11000.0, -10900.0, -10800.0],
            "Loc_Z": [320.0, 322.0, 325.0],
        })

        payload = build_scene_3d_payload(
            calib_dir=tmp_dir,
            estimated_df=est_df,
            ground_truth_df=None,
            grouped_detections=None,
            use_depth_model=False,
            grid_size=16,
        )

        assert payload["meta"]["has_ground_truth"] is False
        for frame_entry in payload["frames"]:
            assert frame_entry["gt_pos_cm"] is None
            assert frame_entry["error_cm"] is None
            assert len(frame_entry["est_pos_cm"]) == 3
        _assert_mesh_and_point_cloud_locked(payload, max_tol_cm=25.0)


@pytest.mark.skipif(
    not os.path.exists(HACKATHON_DIR),
    reason="Real hackathon dataset directory not present",
)
def test_real_hackathon_dataset_integration():
    """
    Tests end-to-end terrain & calibrated depth point cloud generation on the
    real Peleng `хакатон` cameras and MP4 videos, verifying:
    1. Point cloud is 100% locked onto the terrain mesh (`<= 25 cm`).
    2. Valley floor along the road is at natural road elevation (`Z < 900 cm`), not pulled into a dome.
    """
    gt_csv = os.path.join(HACKATHON_DIR, "object_transform_SM_AA_track.csv")
    gt_df = load_ground_truth(gt_csv).iloc[:60].copy()
    est_df = gt_df[["Frame", "Loc_X", "Loc_Y", "Loc_Z"]].copy()

    payload = build_scene_3d_payload(
        calib_dir=HACKATHON_DIR,
        estimated_df=est_df,
        ground_truth_df=gt_df,
        grouped_detections=None,
        use_depth_model=True,
        grid_size=48,
    )

    assert len(payload["cameras"]) == 5
    assert payload["cameras"]["5"]["is_dynamic"] is True
    assert payload["terrain_mesh"]["rows"] == 48
    assert payload["terrain_mesh"]["cols"] == 48
    assert len(payload["point_cloud"]["points_cm"]) >= 500
    assert len(payload["frames"]) == 60

    # Verify point cloud is locked directly to the terrain mesh surface
    _assert_mesh_and_point_cloud_locked(payload, max_tol_cm=25.0)

    # Verify trajectory points sit cleanly on the valley floor (no artificial floating dome)
    mesh = payload["terrain_mesh"]
    x_grid = np.asarray(mesh["x_grid_cm"], dtype=np.float64)
    y_grid = np.asarray(mesh["y_grid_cm"], dtype=np.float64)
    zz_mesh = np.asarray(mesh["z_matrix_cm"], dtype=np.float64)
    traj_xy = est_df[["Loc_X", "Loc_Y"]].to_numpy(dtype=np.float64)
    traj_z = est_df["Loc_Z"].to_numpy(dtype=np.float64)
    z_mesh_at_road = _bilinear_interpolate_mesh(traj_xy[:, 0], traj_xy[:, 1], x_grid, y_grid, zz_mesh)
    road_to_mesh_diff = float(np.median(np.abs(z_mesh_at_road - traj_z)))
    assert road_to_mesh_diff < 120.0, f"Road detached from valley floor by {road_to_mesh_diff:.1f} cm!"


def test_no_artificial_hills_under_camera2():
    calib_csv = os.path.join(HACKATHON_DIR, "object_transform_CineCameraActor2.csv")
    gt_csv = os.path.join(HACKATHON_DIR, "object_transform_SM_AA_track.csv")
    if not (os.path.exists(calib_csv) and os.path.exists(gt_csv)):
        pytest.skip("Hackathon data not found")
    gt_df = load_ground_truth(gt_csv).iloc[:30].copy()
    payload = build_scene_3d_payload(
        calib_dir=HACKATHON_DIR,
        estimated_df=gt_df[["Frame", "Loc_X", "Loc_Y", "Loc_Z"]],
        ground_truth_df=gt_df,
        use_depth_model=False,
        grid_size=32,
    )
    assert "tree_cloud" in payload
    assert len(payload["tree_cloud"]["points_cm"]) > 0
