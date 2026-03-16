#pragma once

#include <array>
#include <utility>
#include <vector>

#include "graphot/core_types.hpp"

namespace graphot::core {

class LogMeanOpsCpp {
  public:
    explicit LogMeanOpsCpp(MeanConfig config);

    double theta(double s, double t) const;
    double dtheta_ds(double s, double t) const;
    double dtheta_dt(double s, double t) const;
    bool origin_supergrad_contains(double z1, double z2) const;
    std::array<double, 3> project_k_top(double p1, double p2, double p3) const;

  private:
    MeanConfig config_;

    bool small_xi_mask(double xi) const;
    std::pair<double, double> beta_and_derivative(double xi) const;
    double beta(double xi) const;
    double beta_derivative(double xi) const;
    double theta_prime(double xi) const;
    std::pair<double, bool> invert_beta(double target) const;
    double top_surface_residual(const std::array<double, 3>& p, double xi) const;
    std::pair<double, double> top_surface_residual_and_deriv(const std::array<double, 3>& p, double xi) const;
};

std::array<double, 3> project_k_point(const LogMeanOpsCpp& mean, double p1, double p2, double p3);
void project_k(
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho_minus,
    const std::vector<double>& rho_plus,
    const std::vector<double>& vartheta,
    std::vector<double>& rho_minus_pr,
    std::vector<double>& rho_plus_pr,
    std::vector<double>& vartheta_pr
);

}  // namespace graphot::core
