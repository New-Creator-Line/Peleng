import time
import numpy as np
import pytest

peleng_core = pytest.importorskip("peleng_core")


def test_cpp_triangulation():
    P1 = np.array([
        [0.0, 2828.28, 0.0, 0.0],
        [0.0, 0.0, -2828.28, 0.0],
        [1.0, 0.0, 0.0, 1000.0]
    ], dtype=np.float64)

    P2 = np.array([
        [-2828.28, 0.0, 0.0, 0.0],
        [0.0, 0.0, -2828.28, 0.0],
        [0.0, 1.0, 0.0, 1000.0]
    ], dtype=np.float64)

    pts = [(0.0, 0.0), (0.0, 0.0)]
    pt_3d, err = peleng_core.triangulate([P1, P2], pts)

    assert pt_3d is not None
    assert np.allclose(pt_3d, [0.0, 0.0, 0.0], atol=1e-3)
    assert err < 1e-4


def test_cpp_shape_validation():
    # Invalid matrix shape (3, 3 instead of 3, 4)
    P_bad = np.eye(3, 3, dtype=np.float64)
    with pytest.raises(ValueError):
        peleng_core.triangulate([P_bad, P_bad], [(0.0, 0.0), (0.0, 0.0)])

    # Length mismatch
    P_good = np.eye(3, 4, dtype=np.float64)
    res, err = peleng_core.triangulate([P_good, P_good], [(0.0, 0.0)])
    assert res is None


def test_cpp_kalman():
    tracker = peleng_core.KalmanTracker3D(1.0 / 30.0, 5.0, 2.0)
    init_pos = np.array([10.0, 20.0, 30.0])
    tracker.initialize(init_pos)

    pred = tracker.predict()
    assert np.allclose(pred, init_pos, atol=1e-3)

    meas = np.array([11.0, 20.5, 29.8])
    upd = tracker.update(meas, 1.0)
    assert len(upd) == 3

    # Test short vector validation
    with pytest.raises(ValueError):
        tracker.update(np.array([1.0, 2.0]), 1.0)


def test_cpp_speed_benchmark():
    P1 = np.eye(3, 4, dtype=np.float64)
    P2 = np.eye(3, 4, dtype=np.float64)
    pts = [(10.0, 20.0), (15.0, 22.0)]

    t0 = time.perf_counter()
    for _ in range(1000):
        peleng_core.triangulate([P1, P2], pts)
    elapsed = time.perf_counter() - t0

    fps = 1000.0 / elapsed
    print(f"\n[C++ Triangulation Benchmark] 1000 solves in {elapsed*1000:.2f} ms ({fps:.0f} FPS)")
    assert fps > 1000.0


def test_cpp_nvdec_capabilities():
    """Verify NVIDIA Video Codec SDK 13.1.24 NVDEC query binding in peleng_core."""
    assert hasattr(peleng_core, "query_nvdec_capabilities")
    caps = peleng_core.query_nvdec_capabilities()
    assert isinstance(caps, dict)
    assert "sdk_linked" in caps
    assert "h264_supported" in caps
    if caps["sdk_linked"] and caps["h264_supported"]:
        assert caps["num_nvdec_engines"] >= 1
        assert caps["max_width"] >= 1920
        assert caps["max_height"] >= 1080

