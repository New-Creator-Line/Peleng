#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "camera_model.hpp"
#include "kalman_filter.hpp"
#include "nvdec_probe.hpp"
#include "triangulator.hpp"

namespace py = pybind11;
using namespace peleng;

// Python-friendly triangulation function with strict shape checks and GIL release
py::tuple py_triangulate(const py::list& proj_matrices, const py::list& points_2d) {
    const size_t num_matrices = py::len(proj_matrices);
    const size_t num_points = py::len(points_2d);

    if (num_matrices < 2 || num_matrices != num_points) {
        return py::make_tuple(py::none(), std::numeric_limits<double>::infinity());
    }

    std::vector<Mat34> Ps;
    Ps.reserve(num_matrices);

    for (py::handle item : proj_matrices) {
        py::array_t<double> arr = item.cast<py::array_t<double>>();
        if (arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 4) {
            throw py::value_error("Projection matrix must have shape (3, 4)");
        }
        auto r = arr.unchecked<2>();
        Mat34 P{};
        for (py::ssize_t i = 0; i < 3; ++i) {
            for (py::ssize_t j = 0; j < 4; ++j) {
                P[static_cast<size_t>(i)][static_cast<size_t>(j)] = r(i, j);
            }
        }
        Ps.push_back(P);
    }

    std::vector<Vec2> pts;
    pts.reserve(num_points);
    for (py::handle item : points_2d) {
        py::tuple t = item.cast<py::tuple>();
        if (py::len(t) < 2) {
            throw py::value_error("Each 2D point must be a (u, v) tuple of at least 2 elements");
        }
        pts.push_back({ t[0].cast<double>(), t[1].cast<double>() });
    }

    std::optional<TriangulationResult> result;
    {
        // Release GIL during compute-intensive SVD and Jacobi solver
        py::gil_scoped_release release_gil;
        TriangulatorCpp triangulator;
        result = triangulator.triangulate(Ps, pts);
    }

    if (!result.has_value()) {
        return py::make_tuple(py::none(), std::numeric_limits<double>::infinity());
    }

    py::array_t<double> out_arr(3);
    auto mut = out_arr.mutable_unchecked<1>();
    mut(0) = result->point.x;
    mut(1) = result->point.y;
    mut(2) = result->point.z;

    return py::make_tuple(out_arr, result->mean_reprojection_error);
}

PYBIND11_MODULE(peleng_core, m) {
    m.doc() = "Peleng 3D Vision Core C++ Module (SVD Triangulation, Kalman Filtering & NVIDIA Video Codec SDK NVDEC)";

    m.def("triangulate", &py_triangulate, "Multi-view 3D Triangulation using SVD DLT");

    m.def("query_nvdec_capabilities", []() {
        const auto caps = query_nvdec_capabilities();
        py::dict d;
        d["sdk_linked"] = caps.sdk_linked;
        d["sdk_version"] = caps.sdk_version;
        d["h264_supported"] = caps.h264_supported;
        d["hevc_supported"] = caps.hevc_supported;
        d["num_nvdec_engines"] = caps.num_nvdec_engines;
        d["max_width"] = caps.max_width;
        d["max_height"] = caps.max_height;
        d["min_width"] = caps.min_width;
        d["min_height"] = caps.min_height;
        d["output_format_mask"] = caps.output_format_mask;
        return d;
    }, "Query NVIDIA Video Codec SDK 13.1.24 NVDEC hardware decoder capabilities on the active GPU");

    py::class_<KalmanFilter3DCpp>(m, "KalmanTracker3D")
        .def(py::init<double, double, double>(),
             py::arg("dt") = 1.0 / 30.0,
             py::arg("process_noise_std") = 5.0,
             py::arg("measurement_noise_std") = 2.0)
        .def("initialize", [](KalmanFilter3DCpp& self, py::array_t<double> pos, std::optional<py::array_t<double>> vel) {
            if (pos.size() < 3) {
                throw py::value_error("Position vector must have at least 3 elements");
            }
            auto p = pos.unchecked<1>();
            Vec3 pos_vec{ p(0), p(1), p(2) };
            Vec3 vel_vec{ 0.0, 0.0, 0.0 };
            if (vel.has_value()) {
                if (vel->size() < 3) {
                    throw py::value_error("Velocity vector must have at least 3 elements");
                }
                auto v = vel->unchecked<1>();
                vel_vec = { v(0), v(1), v(2) };
            }
            self.initialize(pos_vec, vel_vec);
        }, py::arg("pos"), py::arg("vel") = std::nullopt)
        .def("predict", [](KalmanFilter3DCpp& self) {
            Vec3 p = self.predict();
            py::array_t<double> arr(3);
            auto mut = arr.mutable_unchecked<1>();
            mut(0) = p.x; mut(1) = p.y; mut(2) = p.z;
            return arr;
        })
        .def("update", [](KalmanFilter3DCpp& self, py::array_t<double> meas, double noise_scale) {
            if (meas.size() < 3) {
                throw py::value_error("Measurement vector must have at least 3 elements");
            }
            auto m_arr = meas.unchecked<1>();
            Vec3 z{ m_arr(0), m_arr(1), m_arr(2) };
            Vec3 p = self.update(z, noise_scale);
            py::array_t<double> arr(3);
            auto mut = arr.mutable_unchecked<1>();
            mut(0) = p.x; mut(1) = p.y; mut(2) = p.z;
            return arr;
        }, py::arg("measurement"), py::arg("noise_scale") = 1.0)
        .def("get_position", [](const KalmanFilter3DCpp& self) {
            Vec3 p = self.get_position();
            py::array_t<double> arr(3);
            auto mut = arr.mutable_unchecked<1>();
            mut(0) = p.x; mut(1) = p.y; mut(2) = p.z;
            return arr;
        })
        .def("get_velocity", [](const KalmanFilter3DCpp& self) {
            Vec3 v = self.get_velocity();
            py::array_t<double> arr(3);
            auto mut = arr.mutable_unchecked<1>();
            mut(0) = v.x; mut(1) = v.y; mut(2) = v.z;
            return arr;
        });
}
