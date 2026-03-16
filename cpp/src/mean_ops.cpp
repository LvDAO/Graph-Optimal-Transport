#include "graphot/mean_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "graphot/runtime.hpp"

#if defined(GRAPHOT_USE_OPENMP)
#include <omp.h>
#endif

namespace graphot::core {

LogMeanOpsCpp::LogMeanOpsCpp(MeanConfig config) : config_(config) {}

double LogMeanOpsCpp::theta(double s, double t) const {
    if (s == t && s >= 0.0) {
        return s;
    }
    if (!(s > 0.0 && t > 0.0)) {
        return 0.0;
    }
    if (std::abs(t - s) <= config_.eps_diag * std::max(s, t)) {
        const double m = 0.5 * (s + t);
        const double u = (t - s) / std::max(s + t, std::numeric_limits<double>::min());
        return m * (1.0 - (u * u) / 3.0);
    }
    return (t - s) / (std::log(t) - std::log(s));
}

double LogMeanOpsCpp::dtheta_ds(double s, double t) const {
    if (s == t && s > 0.0) {
        return 0.5;
    }
    if (!(s > 0.0 && t > 0.0)) {
        return 0.0;
    }
    if (std::abs(t - s) <= config_.eps_diag * std::max(s, t)) {
        const double m = 0.5 * (s + t);
        const double d = t - s;
        const double denom1 = std::max(6.0 * m, std::numeric_limits<double>::min());
        const double denom2 = std::max(24.0 * m * m, std::numeric_limits<double>::min());
        return 0.5 + d / denom1 + (d * d) / denom2;
    }
    const double lr = std::log(t) - std::log(s);
    return (((t - s) / s) - lr) / (lr * lr);
}

double LogMeanOpsCpp::dtheta_dt(double s, double t) const {
    if (s == t && s > 0.0) {
        return 0.5;
    }
    if (!(s > 0.0 && t > 0.0)) {
        return 0.0;
    }
    if (std::abs(t - s) <= config_.eps_diag * std::max(s, t)) {
        const double m = 0.5 * (s + t);
        const double d = t - s;
        const double denom1 = std::max(6.0 * m, std::numeric_limits<double>::min());
        const double denom2 = std::max(24.0 * m * m, std::numeric_limits<double>::min());
        return 0.5 - d / denom1 + (d * d) / denom2;
    }
    const double lr = std::log(t) - std::log(s);
    return (lr - (t - s) / t) / (lr * lr);
}

bool LogMeanOpsCpp::origin_supergrad_contains(double z1, double z2) const {
    const bool swap = z1 < z2;
    const double primary = swap ? z2 : z1;
    const double secondary = swap ? z1 : z2;
    const bool positive = (z1 > 0.0) && (z2 > 0.0);
    const auto [xi, in_range] = invert_beta(primary);
    const double s = std::exp(-0.5 * xi);
    const double t = std::exp(0.5 * xi);
    const double needed = dtheta_dt(s, t);
    return positive && in_range && (secondary >= needed - 1e-12);
}

std::array<double, 3> LogMeanOpsCpp::project_k_top(double p1, double p2, double p3) const {
    const std::array<double, 3> p = {p1, p2, p3};
    const double lo0 = -config_.xi_max;
    const double hi0 = config_.xi_max;
    const double f_lo = top_surface_residual(p, lo0);
    const double f_hi = top_surface_residual(p, hi0);
    const bool has_bracket = (std::abs(f_lo) <= 1e-12) || (std::abs(f_hi) <= 1e-12)
        || (sign_of(f_lo) != sign_of(f_hi));

    double left = lo0;
    double right = hi0;
    double left_residual = f_lo;
    double xi = 0.0;

    for (int iter = 0; iter < config_.newton_iters; ++iter) {
        const auto [value, deriv] = top_surface_residual_and_deriv(p, xi);
        const double safe_deriv = std::abs(deriv) > 1e-12 ? deriv : 1.0;
        const double cand = std::clamp(xi - value / safe_deriv, left, right);
        const double f_cand = top_surface_residual(p, cand);
        const bool same_side_as_left = sign_of(left_residual) * sign_of(f_cand) > 0;
        if (same_side_as_left) {
            left = cand;
            left_residual = f_cand;
        } else {
            right = cand;
        }
        xi = cand;
    }

    for (int iter = 0; iter < config_.bisect_iters; ++iter) {
        const double mid = 0.5 * (left + right);
        const double f_mid = top_surface_residual(p, mid);
        const bool same_side_as_left = sign_of(left_residual) * sign_of(f_mid) > 0;
        if (same_side_as_left) {
            left = mid;
            left_residual = f_mid;
        } else {
            right = mid;
        }
    }

    xi = 0.5 * (left + right);
    if (!has_bracket) {
        xi = (std::abs(f_lo) <= std::abs(f_hi)) ? lo0 : hi0;
    }
    const double s = std::exp(0.5 * xi);
    const double t = std::exp(-0.5 * xi);
    const std::array<double, 3> w = {s, t, theta(s, t)};
    const double denom = std::max(w[0] * w[0] + w[1] * w[1] + w[2] * w[2], kTiny);
    const double tau = std::max((p[0] * w[0] + p[1] * w[1] + p[2] * w[2]) / denom, 0.0);
    return {tau * w[0], tau * w[1], tau * w[2]};
}

bool LogMeanOpsCpp::small_xi_mask(double xi) const {
    return std::abs(xi) <= std::sqrt(config_.eps_diag);
}

std::pair<double, double> LogMeanOpsCpp::beta_and_derivative(double xi) const {
    const bool small = small_xi_mask(xi);
    const double xi2 = xi * xi;
    const double xi3 = xi2 * xi;
    const double xi4 = xi2 * xi2;
    const double em1 = std::expm1(xi);
    const double exp_x = em1 + 1.0;

    const double beta_raw = (em1 - xi) / (small ? 1.0 : xi2);
    const double beta_series = 0.5 + xi / 6.0 + xi2 / 24.0 + xi3 / 120.0 + xi4 / 720.0;
    const double beta = small ? beta_series : beta_raw;

    const double deriv_raw = (xi * (exp_x + 1.0) - 2.0 * em1) / (small ? 1.0 : xi3);
    const double deriv_series = 1.0 / 6.0 + xi / 12.0 + xi2 / 40.0 + xi3 / 180.0 + xi4 / 1008.0;
    const double deriv = small ? deriv_series : deriv_raw;
    return {beta, deriv};
}

double LogMeanOpsCpp::beta(double xi) const {
    return beta_and_derivative(xi).first;
}

double LogMeanOpsCpp::beta_derivative(double xi) const {
    return beta_and_derivative(xi).second;
}

double LogMeanOpsCpp::theta_prime(double xi) const {
    const bool small = small_xi_mask(xi);
    const double xi2 = xi * xi;
    const double xi3 = xi2 * xi;
    const double xi4 = xi2 * xi2;
    const double xi5 = xi4 * xi;
    const double em1 = std::expm1(xi);
    const double exp_x = em1 + 1.0;
    const double exp_half_neg = std::exp(-0.5 * xi);
    const double raw = exp_half_neg * (0.5 * xi * (exp_x + 1.0) - em1) / (small ? 1.0 : xi2);
    const double series = xi / 12.0 + xi3 / 480.0 + xi5 / 53760.0;
    return small ? series : raw;
}

std::pair<double, bool> LogMeanOpsCpp::invert_beta(double target) const {
    const double lo = 0.0;
    const double hi = config_.xi_max;
    const double f_lo = beta(lo) - target;
    const double f_hi = beta(hi) - target;
    const bool in_range = (f_lo <= 1e-12) && (f_hi >= -1e-12);

    double left = lo;
    double right = hi;
    double xi = 0.5 * (lo + hi);
    for (int iter = 0; iter < config_.newton_iters; ++iter) {
        const auto [beta_value, deriv] = beta_and_derivative(xi);
        const double step = (beta_value - target) / (std::abs(deriv) > 1e-12 ? deriv : 1.0);
        const double cand = std::clamp(xi - step, left, right);
        const double f_cand = beta(cand) - target;
        if (f_cand < 0.0) {
            left = cand;
        } else {
            right = cand;
        }
        xi = cand;
    }
    for (int iter = 0; iter < config_.bisect_iters; ++iter) {
        const double mid = 0.5 * (left + right);
        const double f_mid = beta(mid) - target;
        if (f_mid < 0.0) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return {0.5 * (left + right), in_range};
}

double LogMeanOpsCpp::top_surface_residual(const std::array<double, 3>& p, double xi) const {
    const double s = std::exp(0.5 * xi);
    const double t = std::exp(-0.5 * xi);
    const double theta_value = theta(s, t);
    const double dtheta_ds_value = dtheta_ds(s, t);
    const double dtheta_dt_value = dtheta_dt(s, t);
    return p[0] * (t + theta_value * dtheta_dt_value)
        - p[1] * (s + theta_value * dtheta_ds_value)
        + p[2] * (t * dtheta_ds_value - s * dtheta_dt_value);
}

std::pair<double, double> LogMeanOpsCpp::top_surface_residual_and_deriv(
    const std::array<double, 3>& p,
    double xi
) const {
    const double s = std::exp(0.5 * xi);
    const double t = std::exp(-0.5 * xi);
    const double theta_value = theta(s, t);
    const double dtheta_ds_value = dtheta_ds(s, t);
    const double dtheta_dt_value = dtheta_dt(s, t);
    const double residual = p[0] * (t + theta_value * dtheta_dt_value)
        - p[1] * (s + theta_value * dtheta_ds_value)
        + p[2] * (t * dtheta_ds_value - s * dtheta_dt_value);

    const double s_prime = 0.5 * s;
    const double t_prime = -0.5 * t;
    const double theta_prime_value = theta_prime(xi);
    const double dtheta_dt_prime = beta_derivative(xi);
    const double dtheta_ds_prime = -beta_derivative(-xi);
    const double deriv = p[0] * (t_prime + theta_prime_value * dtheta_dt_value + theta_value * dtheta_dt_prime)
        - p[1] * (s_prime + theta_prime_value * dtheta_ds_value + theta_value * dtheta_ds_prime)
        + p[2] * (
            t_prime * dtheta_ds_value
            + t * dtheta_ds_prime
            - s_prime * dtheta_dt_value
            - s * dtheta_dt_prime
        );
    return {residual, deriv};
}

std::array<double, 3> project_k_point(
    const LogMeanOpsCpp& mean,
    double p1,
    double p2,
    double p3
) {
    const double theta_value = mean.theta(p1, p2);
    const bool inside = (p1 >= 0.0) && (p2 >= 0.0) && (p3 >= 0.0) && (p3 <= theta_value + 1e-12);
    if (inside) {
        return {p1, p2, p3};
    }
    if (p3 <= 0.0) {
        return {std::max(p1, 0.0), std::max(p2, 0.0), 0.0};
    }
    const bool origin_case = (p1 <= 0.0) && (p2 <= 0.0)
        && mean.origin_supergrad_contains(-p1 / p3, -p2 / p3);
    if (origin_case) {
        return {0.0, 0.0, 0.0};
    }
    return mean.project_k_top(p1, p2, p3);
}

void project_k(
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho_minus,
    const std::vector<double>& rho_plus,
    const std::vector<double>& vartheta,
    std::vector<double>& rho_minus_pr,
    std::vector<double>& rho_plus_pr,
    std::vector<double>& vartheta_pr
) {
    rho_minus_pr.resize(rho_minus.size());
    rho_plus_pr.resize(rho_plus.size());
    vartheta_pr.resize(vartheta.size());
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(rho_minus.size()))
#endif
    for (std::size_t idx = 0; idx < rho_minus.size(); ++idx) {
        const auto proj = project_k_point(mean, rho_minus[idx], rho_plus[idx], vartheta[idx]);
        rho_minus_pr[idx] = proj[0];
        rho_plus_pr[idx] = proj[1];
        vartheta_pr[idx] = proj[2];
    }
}

}  // namespace graphot::core
