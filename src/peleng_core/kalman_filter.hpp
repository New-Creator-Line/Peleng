#pragma once

#include "camera_model.hpp"
#include <algorithm>
#include <array>

namespace peleng {

class KalmanFilter3DCpp {
public:
    explicit KalmanFilter3DCpp(
        double dt = 1.0 / 30.0,
        double process_noise_std = 5.0,
        double measurement_noise_std = 2.0
    ) noexcept
        : dt_(dt), process_noise_std_(process_noise_std),
          measurement_noise_std_(measurement_noise_std), is_initialized_(false)
    {
        init_matrices();
    }

    void initialize(const Vec3& pos, const Vec3& vel = {0.0, 0.0, 0.0}) noexcept {
        x_[0] = pos.x; x_[1] = pos.y; x_[2] = pos.z;
        x_[3] = vel.x; x_[4] = vel.y; x_[5] = vel.z;

        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                P_[i][j] = (i == j) ? 10.0 : 0.0;
            }
        }
        is_initialized_ = true;
    }

    Vec3 predict() noexcept {
        if (!is_initialized_) return {0.0, 0.0, 0.0};

        std::array<double, 6> x_new{};
        for (size_t i = 0; i < 6; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < 6; ++j) {
                sum += F_[i][j] * x_[j];
            }
            x_new[i] = sum;
        }
        x_ = x_new;

        std::array<std::array<double, 6>, 6> FP{};
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < 6; ++k) sum += F_[i][k] * P_[k][j];
                FP[i][j] = sum;
            }
        }

        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < 6; ++k) sum += FP[i][k] * F_[j][k];
                P_[i][j] = sum + Q_[i][j];
            }
        }

        return {x_[0], x_[1], x_[2]};
    }

    Vec3 update(const Vec3& z, double noise_scale = 1.0) noexcept {
        if (!is_initialized_) {
            initialize(z);
            return z;
        }

        const double r = (measurement_noise_std_ * measurement_noise_std_) * std::max(0.1, noise_scale);
        const std::array<double, 3> y = { z.x - x_[0], z.y - x_[1], z.z - x_[2] };

        // S = H * P * H^T + R (3x3)
        std::array<std::array<double, 3>, 3> S{};
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                S[i][j] = P_[i][j] + ((i == j) ? r : 0.0);
            }
        }

        const double det = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1])
                         - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0])
                         + S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);

        if (std::abs(det) < 1e-12) return {x_[0], x_[1], x_[2]};
        const double inv_det = 1.0 / det;

        std::array<std::array<double, 3>, 3> invS{};
        invS[0][0] = (S[1][1] * S[2][2] - S[1][2] * S[2][1]) * inv_det;
        invS[0][1] = (S[0][2] * S[2][1] - S[0][1] * S[2][2]) * inv_det;
        invS[0][2] = (S[0][1] * S[1][2] - S[0][2] * S[1][1]) * inv_det;

        invS[1][0] = (S[1][2] * S[2][0] - S[1][0] * S[2][2]) * inv_det;
        invS[1][1] = (S[0][0] * S[2][2] - S[0][2] * S[2][0]) * inv_det;
        invS[1][2] = (S[0][2] * S[1][0] - S[0][0] * S[1][2]) * inv_det;

        invS[2][0] = (S[1][0] * S[2][1] - S[1][1] * S[2][0]) * inv_det;
        invS[2][1] = (S[0][1] * S[2][0] - S[0][0] * S[2][1]) * inv_det;
        invS[2][2] = (S[0][0] * S[1][1] - S[0][1] * S[1][0]) * inv_det;

        // Kalman gain K = P * H^T * inv(S) (6x3)
        std::array<std::array<double, 3>, 6> K{};
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < 3; ++k) sum += P_[i][k] * invS[k][j];
                K[i][j] = sum;
            }
        }

        for (size_t i = 0; i < 6; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < 3; ++j) sum += K[i][j] * y[j];
            x_[i] += sum;
        }

        // Joseph form covariance update
        std::array<std::array<double, 6>, 6> I_KH{};
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                const double kh = (j < 3) ? K[i][j] : 0.0;
                I_KH[i][j] = ((i == j) ? 1.0 : 0.0) - kh;
            }
        }

        std::array<std::array<double, 6>, 6> temp{};
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < 6; ++k) sum += I_KH[i][k] * P_[k][j];
                temp[i][j] = sum;
            }
        }

        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                double sum1 = 0.0;
                for (size_t k = 0; k < 6; ++k) sum1 += temp[i][k] * I_KH[j][k];

                double sum2 = 0.0;
                for (size_t k = 0; k < 3; ++k) sum2 += K[i][k] * r * K[j][k];

                P_[i][j] = sum1 + sum2;
            }
        }

        return {x_[0], x_[1], x_[2]};
    }

    [[nodiscard]] Vec3 get_position() const noexcept {
        return {x_[0], x_[1], x_[2]};
    }

    [[nodiscard]] Vec3 get_velocity() const noexcept {
        return {x_[3], x_[4], x_[5]};
    }

    [[nodiscard]] std::array<double, 6> get_state() const noexcept {
        return x_;
    }

    void set_state(const std::array<double, 6>& state) noexcept {
        x_ = state;
    }

private:
    void init_matrices() noexcept {
        for (size_t i = 0; i < 6; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                F_[i][j] = (i == j) ? 1.0 : 0.0;
                P_[i][j] = (i == j) ? 10.0 : 0.0; // Consistent initialization baseline
                Q_[i][j] = 0.0;
            }
            if (i < 3) F_[i][i + 3] = dt_;
        }

        const double q = process_noise_std_ * process_noise_std_;
        const double dt2 = dt_ * dt_;
        const double dt3 = dt_ * dt_ * dt_ / 2.0;
        const double dt4 = dt_ * dt_ * dt_ * dt_ / 4.0;

        for (size_t i = 0; i < 3; ++i) {
            Q_[i][i] = dt4 * q;
            Q_[i][i + 3] = dt3 * q;
            Q_[i + 3][i] = dt3 * q;
            Q_[i + 3][i + 3] = dt2 * q;
        }

        for (double& v : x_) v = 0.0;
    }

    double dt_;
    double process_noise_std_;
    double measurement_noise_std_;
    bool is_initialized_;

    std::array<std::array<double, 6>, 6> F_{};
    std::array<std::array<double, 6>, 6> P_{};
    std::array<std::array<double, 6>, 6> Q_{};
    std::array<double, 6> x_{};
};

} // namespace peleng
