"""
Rigid 3D model of the tracked truck (oriented box + actor pivot).

The ground-truth / output coordinate of the truck is the Unreal Engine actor pivot
of ``SM_AA_track`` which sits at the front of the vehicle (cabin), NOT at the
geometric center of the long body. Detectors, however, see the whole vehicle, so
the bounding-box center corresponds to the body center.

This module describes the truck as an oriented box of size (length, width, height)
and stores where the box center is located relative to the pivot:

    box_center = pivot + center_forward * heading + center_up * z_axis

``center_forward`` is negative: the body center is behind the cabin.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

DEFAULT_TRUCK_MODEL_PATH = os.path.join("configs", "truck_model.json")


@dataclass
class TruckModel:
    length: float = 2005.0         # along heading (UE units, cm)
    width: float = 465.0
    height: float = 593.0
    center_forward: float = -1117.0  # box center relative to pivot along heading
    center_up: float = 61.0          # box center relative to pivot along world Z
    source: str = "default"

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def heading_vector(yaw: float) -> np.ndarray:
        return np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)

    def box_corners(self, center: np.ndarray, yaw: float) -> np.ndarray:
        """8 corners (8, 3) of the oriented box with given center and yaw (rad)."""
        c, s = np.cos(yaw), np.sin(yaw)
        hl, hw, hh = self.length / 2.0, self.width / 2.0, self.height / 2.0
        local = np.array([[a, b, h] for a in (-hl, hl) for b in (-hw, hw) for h in (-hh, hh)],
                         dtype=np.float64)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return np.asarray(center, dtype=np.float64) + local @ R.T

    def pivot_from_center(self, center: np.ndarray, heading_yaw: float) -> np.ndarray:
        """Converts box center -> actor pivot (cabin) for a truck facing ``heading_yaw``."""
        return (np.asarray(center, dtype=np.float64)
                - self.center_forward * self.heading_vector(heading_yaw)
                - np.array([0.0, 0.0, self.center_up]))

    def center_from_pivot(self, pivot: np.ndarray, heading_yaw: float) -> np.ndarray:
        return (np.asarray(pivot, dtype=np.float64)
                + self.center_forward * self.heading_vector(heading_yaw)
                + np.array([0.0, 0.0, self.center_up]))

    # ------------------------------------------------------------------ I/O
    def save(self, path: str = DEFAULT_TRUCK_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Optional[str] = DEFAULT_TRUCK_MODEL_PATH) -> "TruckModel":
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
            return cls(**known)
        return cls()

