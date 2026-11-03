import math
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.evaluator import load_ground_truth
from src.mock_detections import load_detections
from src.terrain_builder import FRUSTUM_DISTANCE_CM, build_scene_3d_payload


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


def test_schema_and_validation_mode_synthetic():
    """
    Verifies full Scene3DPayload schema compliance in Validation Mode (with ground_truth_df)
    on a self-contained synthetic calibration directory.
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

        # 1. Top-level keys
        for key in ("meta", "metrics", "terrain_mesh", "point_cloud", "cameras", "frames"):
            assert key in payload

        # 2. Meta verification
        meta = payload["meta"]
        assert meta["coord_system"] == "UE5_cm"
        assert meta["num_frames"] == 3
        assert meta["has_ground_truth"] is True
        assert len(meta["origin_center_cm"]) == 3
        for b_key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            assert b_key in meta["bounds_cm"]

        # 3. Terrain mesh verification
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

        # 4. Point cloud verification
        pc = payload["point_cloud"]
        assert len(pc["points_cm"]) > 100
        assert len(pc["points_cm"]) == len(pc["colors_rgb"])
        assert len(pc["points_cm"][0]) == 3
        assert len(pc["colors_rgb"][0]) == 3

        # 5. Cameras & frustum geometry verification
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

        # 6. Frames & dynamic camera 5 rotation verification
        frames = payload["frames"]
        assert len(frames) == 3
        assert frames[0]["gt_pos_cm"] is not None
        assert frames[0]["error_cm"] is not None and frames[0]["error_cm"] > 0.0
        assert len(frames[0]["active_rays"]) == 1
        assert frames[0]["active_rays"][0]["cam_id"] == 1

        # Verify dynamic camera 5 changes forward direction across frames 0 and 2
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


@pytest.mark.skipif(
    not os.path.exists(HACKATHON_DIR),
    reason="Real hackathon dataset directory not present",
)
def test_real_hackathon_dataset_integration():
    """
    Tests end-to-end terrain & calibrated depth point cloud generation on the
    real Peleng `хакатон` cameras and MP4 videos.
    """
    gt_csv = os.path.join(HACKATHON_DIR, "object_transform_SM_AA_track.csv")
    gt_df = load_ground_truth(gt_csv).iloc[:30].copy()
    est_df = gt_df[["Frame", "Loc_X", "Loc_Y", "Loc_Z"]].copy()

    payload = build_scene_3d_payload(
        calib_dir=HACKATHON_DIR,
        estimated_df=est_df,
        ground_truth_df=gt_df,
        grouped_detections=None,
        use_depth_model=True,
        grid_size=36,
    )

    assert len(payload["cameras"]) == 5
    assert payload["cameras"]["5"]["is_dynamic"] is True
    assert payload["terrain_mesh"]["rows"] == 36
    assert payload["terrain_mesh"]["cols"] == 36
    assert len(payload["point_cloud"]["points_cm"]) >= 500
    assert len(payload["frames"]) == 30
