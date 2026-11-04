"""
Calibrates the rigid truck model (box size + pivot offset) used by the box tracker.

The truck is a fixed object, so its geometry is measured once (like a camera intrinsic
calibration): given a reference sequence with known actor poses (ground truth pivot +
heading from motion), find the box (length, width, height) and the position of the box
center relative to the pivot that best explain the detected 2D boxes in all cameras.

To show the model is not over-fitted, it is calibrated on one half of the sequence and
checked on the other half (2D box error), then re-fitted on all frames and saved.

Usage:
    python scripts/calibration/calibrate_truck_model.py \
        --calib-dir хакатон --gt хакатон/object_transform_SM_AA_track.csv \
        --detections results/detections/detections_2d_yolo_real.csv
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.box_fitter import project_box_extents  # noqa: E402
from src.camera_model import load_all_cameras  # noqa: E402
from src.evaluator import load_ground_truth  # noqa: E402
from src.mock_detections import load_detections  # noqa: E402
from src.truck_model import DEFAULT_TRUCK_MODEL_PATH, TruckModel  # noqa: E402


def collect_samples(cams, gt_xyz: np.ndarray, dets, step: int, min_conf: float,
                    img_w: int, img_h: int) -> List[Tuple]:
    samples = []
    n = len(gt_xyz)
    for f in range(0, n, step):
        f0, f1 = max(0, f - 3), min(n - 1, f + 3)
        d = gt_xyz[f1] - gt_xyz[f0]
        if np.linalg.norm(d[:2]) < 30.0:
            continue
        yaw = float(np.arctan2(d[1], d[0]))
        best = {}
        for o in dets.get(f, []):
            if o["conf"] < min_conf or o["cam_id"] not in cams:
                continue
            if o["cam_id"] not in best or o["conf"] > best[o["cam_id"]]["conf"]:
                best[o["cam_id"]] = o
        for cid, o in best.items():
            box = np.array([o["u"] - o["w"] / 2, o["v"] - o["h"] / 2, o["u"] + o["w"] / 2, o["v"] + o["h"] / 2])
            if box[0] < 5 or box[1] < 5 or box[2] > img_w - 5 or box[3] > img_h - 5:
                continue  # truncated by the image border
            samples.append((f, cid, gt_xyz[f], yaw, box))
    return samples


def residuals(theta: np.ndarray, samples, cams) -> np.ndarray:
    L, W, H, cf, cu = theta
    tm = TruckModel(length=L, width=W, height=H, center_forward=cf, center_up=cu)
    out = []
    for f, cid, pivot, yaw, box in samples:
        ext = project_box_extents(cams[cid].get_model(f), tm.box_corners(tm.center_from_pivot(pivot, yaw), yaw))
        if ext is None:
            out.extend([1.0] * 4)
            continue
        out.extend(((ext - box) / max(20.0, box[2] - box[0])).tolist())
    return np.asarray(out)


def fit(samples, cams, x0) -> np.ndarray:
    sol = least_squares(residuals, x0, args=(samples, cams), loss="soft_l1", f_scale=0.05)
    return sol.x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-dir", default="хакатон")
    ap.add_argument("--gt", default="хакатон/object_transform_SM_AA_track.csv")
    ap.add_argument("--detections", default="results/detections/detections_2d_yolo_real.csv")
    ap.add_argument("--out", default=DEFAULT_TRUCK_MODEL_PATH)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--min-conf", type=float, default=0.6)
    args = ap.parse_args()

    cams = load_all_cameras(args.calib_dir)
    gt = load_ground_truth(args.gt).sort_values("Frame")
    gt_xyz = gt[["Loc_X", "Loc_Y", "Loc_Z"]].to_numpy(dtype=np.float64)
    dets = load_detections(args.detections)
    samples = collect_samples(cams, gt_xyz, dets, args.step, args.min_conf, 1920, 1080)
    print(f"Samples (frame, camera) used: {len(samples)}")

    x0 = np.array([1500.0, 400.0, 450.0, -600.0, 100.0])
    mid = len(gt_xyz) // 2
    first = [s for s in samples if s[0] < mid]
    second = [s for s in samples if s[0] >= mid]
    for name, tr, te in (("A: fit 1st half -> test 2nd half", first, second),
                         ("B: fit 2nd half -> test 1st half", second, first)):
        th = fit(tr, cams, x0)
        err_tr = np.median(np.abs(residuals(th, tr, cams)))
        err_te = np.median(np.abs(residuals(th, te, cams)))
        print(f"{name}: L/W/H={th[0]:.0f}/{th[1]:.0f}/{th[2]:.0f}, center_fwd={th[3]:.0f}, center_up={th[4]:.0f} "
              f"| median rel. edge error train={err_tr:.3f} test={err_te:.3f}")

    th = fit(samples, cams, x0)
    tm = TruckModel(length=float(th[0]), width=float(th[1]), height=float(th[2]),
                    center_forward=float(th[3]), center_up=float(th[4]),
                    source=f"calibrated on {os.path.basename(args.gt)} with {len(samples)} box samples")
    tm.save(args.out)
    print(f"Saved truck model -> {args.out}")
    print(f"  length={tm.length:.0f}, width={tm.width:.0f}, height={tm.height:.0f}")
    print(f"  box center is {abs(tm.center_forward):.0f} units BEHIND the pivot "
          f"(pivot lies {abs(tm.center_forward) - tm.length / 2:.0f} units from the front face)")


if __name__ == "__main__":
    main()

