from typing import List, Optional, Tuple
import numpy as np

# Try importing high-performance C++ module peleng_core
try:
    import peleng_core
    HAS_CPP_CORE = True
except ImportError:
    HAS_CPP_CORE = False


class MultiViewTriangulator:
    """Multi-view 3D Triangulation using Direct Linear Transform (DLT) via SVD."""

    def __init__(self, reprojection_error_threshold_px: float = 50.0) -> None:
        self.reprojection_threshold = float(reprojection_error_threshold_px)

    def triangulate(
        self,
        projection_matrices: List[np.ndarray],
        points_2d: List[Tuple[float, float]],
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Triangulates a 3D point given at least two 3x4 projection matrices and 2D observations.
        Returns: (pt_3d, mean_reprojection_error) or (None, inf) if triangulation fails.
        """
        if len(projection_matrices) < 2 or len(points_2d) < 2:
            return None, float("inf")

        if HAS_CPP_CORE:
            try:
                pt_3d, err = peleng_core.triangulate(projection_matrices, points_2d)
                return pt_3d, err
            except Exception:
                pass

        # Pure Python / NumPy DLT Triangulation
        num_cams = len(projection_matrices)
        A = np.zeros((2 * num_cams, 4), dtype=np.float64)

        for i, (P, (u, v)) in enumerate(zip(projection_matrices, points_2d)):
            A[2 * i]     = u * P[2, :] - P[0, :]
            A[2 * i + 1] = v * P[2, :] - P[1, :]

        # Solve A * X = 0 via SVD
        _, _, vh = np.linalg.svd(A)
        X_hom = vh[-1]

        if abs(X_hom[3]) < 1e-9:
            return None, float("inf")

        pt_3d = X_hom[:3] / X_hom[3]

        # Compute mean reprojection error
        total_err = 0.0
        valid_cams = 0
        for P, (u, v) in zip(projection_matrices, points_2d):
            proj_h = P @ np.append(pt_3d, 1.0)
            if proj_h[2] <= 1e-6:
                continue
            u_proj = proj_h[0] / proj_h[2]
            v_proj = proj_h[1] / proj_h[2]
            err = np.hypot(u_proj - u, v_proj - v)
            total_err += err
            valid_cams += 1

        mean_err = total_err / max(1, valid_cams)
        return pt_3d, mean_err
