#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <optional>

namespace peleng {

inline constexpr double deg_to_rad = std::numbers::pi / 180.0;

struct Vec2 {
    double u{0.0};
    double v{0.0};
};

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    [[nodiscard]] constexpr Vec3 operator+(const Vec3& o) const noexcept {
        return {x + o.x, y + o.y, z + o.z};
    }
    [[nodiscard]] constexpr Vec3 operator-(const Vec3& o) const noexcept {
        return {x - o.x, y - o.y, z - o.z};
    }
    [[nodiscard]] constexpr Vec3 operator*(double s) const noexcept {
        return {x * s, y * s, z * s};
    }
    [[nodiscard]] double norm() const noexcept {
        return std::sqrt(x * x + y * y + z * z);
    }
};

struct ProjectionResult {
    Vec2 uv;
    double z_cam{0.0};
};

using Mat3 = std::array<std::array<double, 3>, 3>;
using Mat34 = std::array<std::array<double, 4>, 3>;

[[nodiscard]] inline Vec3 mat3_mul_vec3(const Mat3& m, const Vec3& v) noexcept {
    return {
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
    };
}

// Computes camera rotation matrix from Unreal Engine FRotator
// Official Unreal Engine FRotationMatrix definition:
// Forward = [ cp*cy, cp*sy, sp ]
// Right   = [ sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, -sr*cp ]
// Up      = [ -(cr*sp*cy + sr*sy), -(cr*sp*sy - sr*cy), cr*cp ]
// In OpenCV camera coordinates: X_cv = Right, Y_cv = Down (-Up), Z_cv = Forward
[[nodiscard]] inline Mat3 get_ue_camera_rot_matrix(double roll_deg, double pitch_deg, double yaw_deg) noexcept {
    const double r = roll_deg * deg_to_rad;
    const double p = pitch_deg * deg_to_rad;
    const double y = yaw_deg * deg_to_rad;

    const double sp = std::sin(p), cp = std::cos(p);
    const double sy = std::sin(y), cy = std::cos(y);
    const double sr = std::sin(r), cr = std::cos(r);

    const std::array<double, 3> forward = { cp * cy, cp * sy, sp };
    const std::array<double, 3> right   = { sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp };
    const std::array<double, 3> up      = { -(cr * sp * cy + sr * sy), -(cr * sp * sy - sr * cy), cr * cp };

    Mat3 R_cam{};
    for (size_t j = 0; j < 3; ++j) {
        R_cam[0][j] = right[j];
        R_cam[1][j] = -up[j];
        R_cam[2][j] = forward[j];
    }
    return R_cam;
}

class CameraModelCpp {
public:
    CameraModelCpp(
        int cam_id,
        const Vec3& loc,
        const Vec3& rot,
        double focal_length_mm = 35.0,
        double sensor_w_mm = 23.76,
        double sensor_h_mm = 13.365,
        double img_w_px = 1920.0,
        double img_h_px = 1080.0
    ) noexcept
        : cam_id_(cam_id), loc_(loc), rot_(rot),
          focal_length_mm_(focal_length_mm), sensor_w_mm_(sensor_w_mm), sensor_h_mm_(sensor_h_mm),
          img_w_px_(img_w_px), img_h_px_(img_h_px)
    {
        compute_matrices();
    }

    [[nodiscard]] const Mat34& get_projection_matrix() const noexcept { return P_; }
    [[nodiscard]] int get_cam_id() const noexcept { return cam_id_; }

    [[nodiscard]] std::optional<ProjectionResult> project_point(const Vec3& pt_ue) const noexcept {
        const Vec3 pt_cam = mat3_mul_vec3(R_cam_, pt_ue) + t_cam_;
        if (pt_cam.z <= 1e-6) {
            return std::nullopt;
        }

        const double u_h = K_[0][0] * pt_cam.x + K_[0][2] * pt_cam.z;
        const double v_h = K_[1][1] * pt_cam.y + K_[1][2] * pt_cam.z;
        return ProjectionResult{
            .uv = { u_h / pt_cam.z, v_h / pt_cam.z },
            .z_cam = pt_cam.z
        };
    }

private:
    void compute_matrices() noexcept {
        const double fx = focal_length_mm_ * img_w_px_ / sensor_w_mm_;
        const double fy = focal_length_mm_ * img_h_px_ / sensor_h_mm_;
        const double cx = img_w_px_ / 2.0;
        const double cy = img_h_px_ / 2.0;

        K_ = {{
            {{ fx,  0.0, cx  }},
            {{ 0.0, fy,  cy  }},
            {{ 0.0, 0.0, 1.0 }}
        }};

        R_cam_ = get_ue_camera_rot_matrix(rot_.x, rot_.y, rot_.z);
        const Vec3 rotated_loc = mat3_mul_vec3(R_cam_, loc_);
        t_cam_ = { -rotated_loc.x, -rotated_loc.y, -rotated_loc.z };

        // P = K * [R | t]
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < 3; ++k) {
                    sum += K_[i][k] * R_cam_[k][j];
                }
                P_[i][j] = sum;
            }
        }
        const Vec3 Kt = mat3_mul_vec3(K_, t_cam_);
        P_[0][3] = Kt.x;
        P_[1][3] = Kt.y;
        P_[2][3] = Kt.z;
    }

    int cam_id_;
    Vec3 loc_;
    Vec3 rot_;
    double focal_length_mm_;
    double sensor_w_mm_;
    double sensor_h_mm_;
    double img_w_px_;
    double img_h_px_;

    Mat3 K_{};
    Mat3 R_cam_{};
    Vec3 t_cam_{};
    Mat34 P_{};
};

} // namespace peleng
