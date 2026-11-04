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
    m.doc() = "Peleng 3D Vision Core C++ Module (SVD Triangulation & Kalman Filtering)";

    m.def("triangulate", &py_triangulate, "Multi-view 3D Triangulation using SVD DLT");

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
        })
        .def("get_state", [](const KalmanFilter3DCpp& self) {
            const auto& s = self.get_state();
            py::array_t<double> arr(6);
            auto mut = arr.mutable_unchecked<1>();
            for (size_t i = 0; i < 6; ++i) mut(i) = s[i];
            return arr;
        })
        .def("set_state", [](KalmanFilter3DCpp& self, py::array_t<double> state) {
            if (state.size() < 6) throw py::value_error("State vector must have 6 elements");
            auto s = state.unchecked<1>();
            std::array<double, 6> arr{};
            for (size_t i = 0; i < 6; ++i) arr[i] = s(i);
            self.set_state(arr);
        });

    py::class_<CameraModelCpp>(m, "CameraModelCpp")
        .def(py::init([](int cam_id, py::array_t<double> loc, py::array_t<double> rot,
                         double focal_length_mm, double sensor_w_mm, double sensor_h_mm,
                         double img_w_px, double img_h_px) {
            if (loc.size() < 3 || rot.size() < 3) throw py::value_error("loc and rot must have 3 elements");
            auto l = loc.unchecked<1>();
            auto r = rot.unchecked<1>();
            return std::make_unique<CameraModelCpp>(
                cam_id,
                Vec3{l(0), l(1), l(2)},
                Vec3{r(0), r(1), r(2)},
                focal_length_mm, sensor_w_mm, sensor_h_mm, img_w_px, img_h_px
            );
        }),
        py::arg("cam_id"), py::arg("loc"), py::arg("rot"),
        py::arg("focal_length_mm") = 35.0, py::arg("sensor_w_mm") = 23.76,
        py::arg("sensor_h_mm") = 13.365, py::arg("img_w_px") = 1920.0, py::arg("img_h_px") = 1080.0)
        .def("project_point", [](const CameraModelCpp& self, py::array_t<double> pt) -> py::object {
            if (pt.size() < 3) throw py::value_error("Point must have 3 elements");
            auto p = pt.unchecked<1>();
            auto res = self.project_point(Vec3{p(0), p(1), p(2)});
            if (!res.has_value()) return py::none();
            return py::make_tuple(res->uv.u, res->uv.v, res->z_cam);
        })
        .def("project_ray_to_z", [](const CameraModelCpp& self, double u, double v, double target_z) -> py::object {
            auto res = self.project_ray_to_z(u, v, target_z);
            if (!res.has_value()) return py::none();
            py::array_t<double> arr(3);
            auto mut = arr.mutable_unchecked<1>();
            mut(0) = res->x; mut(1) = res->y; mut(2) = res->z;
            return arr;
        })
        .def("get_projection_matrix", [](const CameraModelCpp& self) {
            const auto& P = self.get_projection_matrix();
            py::array_t<double> arr({3, 4});
            auto mut = arr.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < 3; ++i) {
                for (py::ssize_t j = 0; j < 4; ++j) {
                    mut(i, j) = P[static_cast<size_t>(i)][static_cast<size_t>(j)];
                }
            }
            return arr;
        });
}
