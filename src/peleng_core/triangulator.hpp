#pragma once

#include "camera_model.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace peleng {

struct TriangulationResult {
    Vec3 point;
    double mean_reprojection_error{0.0};
};

// Solves 4x4 symmetric matrix eigensystem via Jacobi rotations
inline void jacobi_eigen_4x4(std::array<std::array<double, 4>, 4>& S, std::array<std::array<double, 4>, 4>& V) noexcept {
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            V[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    constexpr int max_iter = 50;
    constexpr double epsilon = 1e-15;

    for (int iter = 0; iter < max_iter; ++iter) {
        double max_val = 0.0;
        size_t p = 0, q = 1;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = i + 1; j < 4; ++j) {
                const double abs_val = std::abs(S[i][j]);
                if (abs_val > max_val) {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_val < epsilon) {
            break;
        }

        const double diff = S[q][q] - S[p][p];
        double t = 0.0;
        if (std::abs(diff) < epsilon) {
            t = (S[p][q] > 0.0) ? 1.0 : -1.0;
        } else {
            const double theta = diff / (2.0 * S[p][q]);
            t = 1.0 / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
            if (theta < 0.0) t = -t;
        }

        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        const double tau = s / (1.0 + c);

        const double Spq = S[p][q];
        S[p][q] = 0.0;
        S[p][p] -= t * Spq;
        S[q][q] += t * Spq;

        for (size_t i = 0; i < p; ++i) {
            const double Sip = S[i][p];
            const double Siq = S[i][q];
            S[i][p] = Sip - s * (Siq + tau * Sip);
            S[i][q] = Siq + s * (Sip - tau * Siq);
        }
        for (size_t i = p + 1; i < q; ++i) {
            const double Spi = S[p][i];
            const double Siq = S[i][q];
            S[p][i] = Spi - s * (Siq + tau * Spi);
            S[i][q] = Siq + s * (Spi - tau * Siq);
        }
        for (size_t i = q + 1; i < 4; ++i) {
            const double Spi = S[p][i];
            const double Sqi = S[q][i];
            S[p][i] = Spi - s * (Sqi + tau * Spi);
            S[q][i] = Sqi + s * (Spi - tau * Sqi);
        }

        for (size_t i = 0; i < 4; ++i) {
            const double Vip = V[i][p];
            const double Viq = V[i][q];
            V[i][p] = Vip - s * (Viq + tau * Vip);
            V[i][q] = Viq + s * (Vip - tau * Viq);
        }
    }
}

class TriangulatorCpp {
public:
    explicit TriangulatorCpp(double reprojection_threshold = 50.0) noexcept
        : reprojection_threshold_(reprojection_threshold) {}

    [[nodiscard]] std::optional<TriangulationResult> triangulate(
        std::span<const Mat34> projection_matrices,
        std::span<const Vec2> points_2d
    ) const noexcept {
        const size_t num_views = projection_matrices.size();
        if (num_views < 2 || points_2d.size() != num_views) {
            return std::nullopt;
        }

        // Build 4x4 matrix S = A^T * A
        std::array<std::array<double, 4>, 4> S{};
        for (size_t v = 0; v < num_views; ++v) {
            const auto& P = projection_matrices[v];
            const auto& uv = points_2d[v];

            std::array<double, 4> a0{}, a1{};
            for (size_t j = 0; j < 4; ++j) {
                a0[j] = uv.u * P[2][j] - P[0][j];
                a1[j] = uv.v * P[2][j] - P[1][j];
            }

            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    S[i][j] += a0[i] * a0[j] + a1[i] * a1[j];
                }
            }
        }

        std::array<std::array<double, 4>, 4> V{};
        jacobi_eigen_4x4(S, V);

        size_t min_idx = 0;
        double min_eigen = S[0][0];
        for (size_t i = 1; i < 4; ++i) {
            if (S[i][i] < min_eigen) {
                min_eigen = S[i][i];
                min_idx = i;
            }
        }

        const double w_hom = V[3][min_idx];
        if (std::abs(w_hom) < 1e-9) {
            return std::nullopt;
        }

        const Vec3 pt_3d{
            V[0][min_idx] / w_hom,
            V[1][min_idx] / w_hom,
            V[2][min_idx] / w_hom
        };

        // Reprojection validation
        double total_err = 0.0;
        size_t valid_cams = 0;
        const std::array<double, 4> X_h = { pt_3d.x, pt_3d.y, pt_3d.z, 1.0 };

        for (size_t v = 0; v < num_views; ++v) {
            const auto& P = projection_matrices[v];
            const auto& uv = points_2d[v];

            const double z_proj = P[2][0] * X_h[0] + P[2][1] * X_h[1] + P[2][2] * X_h[2] + P[2][3] * X_h[3];
            if (z_proj <= 1e-6) continue;

            const double u_proj = (P[0][0] * X_h[0] + P[0][1] * X_h[1] + P[0][2] * X_h[2] + P[0][3] * X_h[3]) / z_proj;
            const double v_proj = (P[1][0] * X_h[0] + P[1][1] * X_h[1] + P[1][2] * X_h[2] + P[1][3] * X_h[3]) / z_proj;

            const double du = u_proj - uv.u;
            const double dv = v_proj - uv.v;
            total_err += std::sqrt(du * du + dv * dv);
            ++valid_cams;
        }

        // Triangulation requires at least 2 forward-facing cameras
        if (valid_cams < 2) {
            return std::nullopt;
        }

        const double mean_error = total_err / static_cast<double>(valid_cams);
        if (mean_error > reprojection_threshold_) {
            return std::nullopt;
        }

        return TriangulationResult{
            .point = pt_3d,
            .mean_reprojection_error = mean_error
        };
    }

private:
    double reprojection_threshold_;
};

} // namespace peleng
