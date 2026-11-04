"""
Road Centerline Spline Constraint for 3D Vehicle Tracking.

Enforces physical highway road constraints:
A vehicle travels along a smooth 3D road lane without drifting laterally into curbs/forest.
During blind zones (occlusions by trees/hills), projecting onto the road centerline
eliminates lateral drift and stabilizes Z elevation.
"""

import json
import os
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


class RoadSplineConstraint:
    def __init__(self, num_dense_pts: int = 20000):
        self.num_dense_pts = num_dense_pts
        self.tck = None
        self.u_dense = np.linspace(0, 1.0, self.num_dense_pts)
        self.road_dense: Optional[np.ndarray] = None
        self.tree: Optional[cKDTree] = None
        self.is_fitted = False

    def fit_from_points(self, points: np.ndarray, smoothing: float = 0.0, step: int = 15) -> "RoadSplineConstraint":
        """
        Fits a 3D parametric cubic B-spline through spatial anchor points.
        """
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 10:
            return self

        # Sample anchors along trajectory to remove local noise
        anchors = pts[::step]
        if not np.allclose(anchors[-1], pts[-1]):
            anchors = np.vstack([anchors, pts[-1:]])

        # Ensure no duplicate consecutive points
        diffs = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
        valid_mask = np.concatenate(([True], diffs > 50.0))
        anchors = anchors[valid_mask]

        if len(anchors) < 4:
            return self

        self.tck, _ = splprep([anchors[:, 0], anchors[:, 1], anchors[:, 2]], s=smoothing, k=3)
        self.road_dense = np.column_stack(splev(self.u_dense, self.tck))
        self.tree = cKDTree(self.road_dense)
        self.is_fitted = True
        return self

    def project_point(self, pt_3d: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Projects a single 3D point orthogonally to the nearest point on the road spline.
        Returns: (projected_point_3d, distance_to_centerline)
        """
        if not self.is_fitted or self.tree is None:
            return pt_3d, 0.0

        pt = np.asarray(pt_3d, dtype=np.float64).reshape(1, 3)
        dist, idx = self.tree.query(pt)
        return self.road_dense[idx[0]], float(dist[0])

    def project_trajectory(
        self,
        trajectory: np.ndarray,
        occluded_mask: Optional[np.ndarray] = None,
        blend_clear: float = 0.25,
        blend_occluded: float = 1.0,
    ) -> np.ndarray:
        """
        Applies road constraint across full trajectory:
        - In occluded frames: strong projection to road centerline (kills lateral tree drift).
        - In clear frames: soft regularization towards road centerline.
        """
        if not self.is_fitted or self.tree is None:
            return trajectory

        pts = np.asarray(trajectory, dtype=np.float64).copy()
        dists, indices = self.tree.query(pts)
        road_nearest = self.road_dense[indices]

        if occluded_mask is None:
            weights = np.full((len(pts), 1), blend_clear, dtype=np.float64)
        else:
            weights = np.where(occluded_mask[:, None], blend_occluded, blend_clear)

        constrained_pts = (1.0 - weights) * pts + weights * road_nearest
        return constrained_pts

    def save(self, filepath: str) -> None:
        """Saves fitted road dense points to JSON or CSV."""
        if not self.is_fitted or self.road_dense is None:
            return
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        # Downsample to 200 anchors for compact config
        anchors = self.road_dense[::100].tolist()
        data = {"road_anchors": anchors}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> "RoadSplineConstraint":
        """Loads road anchors from JSON config."""
        if not os.path.exists(filepath):
            return self
        with open(filepath, "r") as f:
            data = json.load(f)
        anchors = np.array(data.get("road_anchors", []), dtype=np.float64)
        if len(anchors) >= 4:
            self.fit_from_points(anchors, smoothing=0.0, step=1)
        return self

