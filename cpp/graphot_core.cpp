#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#define GRAPHOT_HAS_TARGET_SIMD 1
#else
#define GRAPHOT_HAS_TARGET_SIMD 0
#endif

namespace py = pybind11;

namespace {

constexpr double kTiny = 1e-30;

enum class SimdLevel {
    scalar,
    avx2,
    avx512,
};

double dot_scalar(const double* left, const double* right, std::size_t size) {
    double total = 0.0;
    for (std::size_t idx = 0; idx < size; ++idx) {
        total += left[idx] * right[idx];
    }
    return total;
}

void axpy_scalar(double* dst, const double* src, double alpha, std::size_t size) {
    for (std::size_t idx = 0; idx < size; ++idx) {
        dst[idx] += alpha * src[idx];
    }
}

#if GRAPHOT_HAS_TARGET_SIMD
__attribute__((target("avx2,fma"))) double dot_avx2(const double* left, const double* right, std::size_t size) {
    std::size_t idx = 0;
    __m256d acc = _mm256_setzero_pd();
    for (; idx + 4 <= size; idx += 4) {
        const __m256d lhs = _mm256_loadu_pd(left + idx);
        const __m256d rhs = _mm256_loadu_pd(right + idx);
        acc = _mm256_fmadd_pd(lhs, rhs, acc);
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, acc);
    double total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; idx < size; ++idx) {
        total += left[idx] * right[idx];
    }
    return total;
}

__attribute__((target("avx2,fma"))) void axpy_avx2(double* dst, const double* src, double alpha, std::size_t size) {
    const __m256d scale = _mm256_set1_pd(alpha);
    std::size_t idx = 0;
    for (; idx + 4 <= size; idx += 4) {
        const __m256d y = _mm256_loadu_pd(dst + idx);
        const __m256d x = _mm256_loadu_pd(src + idx);
        _mm256_storeu_pd(dst + idx, _mm256_fmadd_pd(scale, x, y));
    }
    for (; idx < size; ++idx) {
        dst[idx] += alpha * src[idx];
    }
}

__attribute__((target("avx512f,avx512vl,avx512dq,fma"))) double dot_avx512(
    const double* left,
    const double* right,
    std::size_t size
) {
    std::size_t idx = 0;
    __m512d acc = _mm512_setzero_pd();
    for (; idx + 8 <= size; idx += 8) {
        const __m512d lhs = _mm512_loadu_pd(left + idx);
        const __m512d rhs = _mm512_loadu_pd(right + idx);
        acc = _mm512_fmadd_pd(lhs, rhs, acc);
    }
    alignas(64) double lanes[8];
    _mm512_store_pd(lanes, acc);
    double total = 0.0;
    for (double lane : lanes) {
        total += lane;
    }
    for (; idx < size; ++idx) {
        total += left[idx] * right[idx];
    }
    return total;
}

__attribute__((target("avx512f,avx512vl,avx512dq,fma"))) void axpy_avx512(
    double* dst,
    const double* src,
    double alpha,
    std::size_t size
) {
    const __m512d scale = _mm512_set1_pd(alpha);
    std::size_t idx = 0;
    for (; idx + 8 <= size; idx += 8) {
        const __m512d y = _mm512_loadu_pd(dst + idx);
        const __m512d x = _mm512_loadu_pd(src + idx);
        _mm512_storeu_pd(dst + idx, _mm512_fmadd_pd(scale, x, y));
    }
    for (; idx < size; ++idx) {
        dst[idx] += alpha * src[idx];
    }
}
#endif

struct SimdOps {
    double (*dot)(const double*, const double*, std::size_t);
    void (*axpy)(double*, const double*, double, std::size_t);
    SimdLevel level;
    const char* name;
};

SimdOps resolve_simd() {
    const char* raw = std::getenv("GRAPHOT_SIMD");
    std::string forced = raw == nullptr ? "auto" : std::string(raw);
    std::transform(forced.begin(), forced.end(), forced.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    auto scalar = SimdOps{dot_scalar, axpy_scalar, SimdLevel::scalar, "scalar"};

#if GRAPHOT_HAS_TARGET_SIMD
    auto avx2 = SimdOps{dot_avx2, axpy_avx2, SimdLevel::avx2, "avx2"};
    auto avx512 = SimdOps{dot_avx512, axpy_avx512, SimdLevel::avx512, "avx512"};

    if (forced == "scalar") {
        return scalar;
    }
    if (forced == "avx2") {
        return avx2;
    }
    if (forced == "avx512") {
        return avx512;
    }
    if (forced != "auto" && !forced.empty()) {
        throw std::invalid_argument("GRAPHOT_SIMD must be one of: auto, scalar, avx2, avx512");
    }

    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl")
        && __builtin_cpu_supports("avx512dq")) {
        return avx512;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return avx2;
    }
#else
    if (forced != "auto" && !forced.empty() && forced != "scalar") {
        throw std::invalid_argument("forced SIMD levels are only supported on x86_64 builds");
    }
#endif
    return scalar;
}

const SimdOps& simd_ops() {
    static const SimdOps ops = resolve_simd();
    return ops;
}

inline std::size_t row_major(int row, int col, int width) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + static_cast<std::size_t>(col);
}

void check_size(std::size_t actual, std::size_t expected, std::string_view name) {
    if (actual != expected) {
        throw std::invalid_argument(std::string(name) + " has unexpected size");
    }
}

void project_zero_mean(std::vector<double>& value) {
    if (value.empty()) {
        return;
    }
    double total = 0.0;
    for (double x : value) {
        total += x;
    }
    const double mean = total / static_cast<double>(value.size());
    for (double& x : value) {
        x -= mean;
    }
}

void vector_axpy(std::vector<double>& dst, const std::vector<double>& src, double alpha) {
    simd_ops().axpy(dst.data(), src.data(), alpha, dst.size());
}

double vector_dot(const std::vector<double>& left, const std::vector<double>& right) {
    return simd_ops().dot(left.data(), right.data(), left.size());
}

double vector_norm(const std::vector<double>& value) {
    return std::sqrt(std::max(vector_dot(value, value), 0.0));
}

int sign_of(double value) {
    if (value > 0.0) {
        return 1;
    }
    if (value < 0.0) {
        return -1;
    }
    return 0;
}

template <typename T>
py::array_t<T> make_array_from_vector(const std::vector<T>& data, std::vector<py::ssize_t> shape) {
    py::array_t<T> array(shape);
    std::memcpy(array.mutable_data(), data.data(), sizeof(T) * data.size());
    return array;
}

struct MeanConfig {
    double eps_diag;
    double xi_max;
    int newton_iters;
    int bisect_iters;
};

struct SolverConfig {
    double tau;
    double sigma;
    double relaxation;
    int max_iters;
    int check_every;
    double residual_tol;
    double feasibility_tol;
    int newton_iters;
    int cg_max_iters;
    double cg_tol;
    bool cg_warm_start;
    bool block_jacobi;
    bool linear_warm_start;
    bool record_debug_trace;
};

struct GraphData {
    int num_nodes;
    int num_edges;
    std::vector<int32_t> src;
    std::vector<int32_t> dst;
    std::vector<int32_t> rev;
    std::vector<double> q;
    std::vector<double> pi;
    std::vector<double> out_rate;
    std::vector<double> edge_weight;
    std::vector<double> one_plus_out_rate;

    GraphData(
        int num_nodes_,
        int num_edges_,
        const int32_t* src_ptr,
        const int32_t* dst_ptr,
        const int32_t* rev_ptr,
        const double* q_ptr,
        const double* pi_ptr,
        const double* out_rate_ptr
    )
        : num_nodes(num_nodes_),
          num_edges(num_edges_),
          src(src_ptr, src_ptr + num_edges_),
          dst(dst_ptr, dst_ptr + num_edges_),
          rev(rev_ptr, rev_ptr + num_edges_),
          q(q_ptr, q_ptr + num_edges_),
          pi(pi_ptr, pi_ptr + num_nodes_),
          out_rate(out_rate_ptr, out_rate_ptr + num_nodes_),
          edge_weight(num_edges_, 0.0),
          one_plus_out_rate(num_nodes_, 0.0) {
        for (int edge = 0; edge < num_edges; ++edge) {
            edge_weight[edge] = 0.5 * q[edge] * pi[src[edge]];
        }
        for (int node = 0; node < num_nodes; ++node) {
            one_plus_out_rate[node] = 1.0 + out_rate[node];
        }
    }
};

struct State {
    std::vector<double> rho;
    std::vector<double> m;
    std::vector<double> vartheta;
    std::vector<double> rho_minus;
    std::vector<double> rho_plus;
    std::vector<double> rho_bar;
    std::vector<double> q_node;

    State() = default;

    State(int num_steps, int num_nodes, int num_edges)
        : rho(static_cast<std::size_t>(num_steps + 1) * static_cast<std::size_t>(num_nodes), 0.0),
          m(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_edges), 0.0),
          vartheta(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_edges), 0.0),
          rho_minus(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_edges), 0.0),
          rho_plus(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_edges), 0.0),
          rho_bar(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_nodes), 0.0),
          q_node(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_nodes), 0.0) {}
};

struct CehStats {
    double cg_residual;
    int cg_iters;
};

struct Diagnostics {
    double primal_delta;
    double dual_delta;
    double continuity_residual;
    double k_violation;
    double endpoint_residual;
    double max_constraint_residual;
    double ceh_cg_residual;
    int ceh_cg_iters;
};

struct DebugTrace {
    std::vector<int32_t> iterations;
    std::vector<double> action;
    std::vector<double> continuity_residual;
    std::vector<double> primal_delta;
    std::vector<double> dual_delta;
    std::vector<double> max_constraint_residual;
    std::vector<double> ceh_cg_residual;
    std::vector<int32_t> ceh_cg_iters;
    std::vector<double> min_vartheta;
    int num_records = 0;
};

struct SolveResult {
    State state;
    Diagnostics diagnostics;
    int iterations_used = 0;
    bool converged = false;
    double action = 0.0;
    double distance = 0.0;
    std::optional<DebugTrace> debug_trace;
};

class LogMeanOpsCpp {
  public:
    explicit LogMeanOpsCpp(MeanConfig config) : config_(config) {}

    double theta(double s, double t) const {
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

    double dtheta_ds(double s, double t) const {
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

    double dtheta_dt(double s, double t) const {
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

    bool origin_supergrad_contains(double z1, double z2) const {
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

    std::array<double, 3> project_k_top(double p1, double p2, double p3) const {
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

  private:
    MeanConfig config_;

    bool small_xi_mask(double xi) const {
        return std::abs(xi) <= std::sqrt(config_.eps_diag);
    }

    std::pair<double, double> beta_and_derivative(double xi) const {
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

    double beta(double xi) const {
        return beta_and_derivative(xi).first;
    }

    double beta_derivative(double xi) const {
        return beta_and_derivative(xi).second;
    }

    double theta_prime(double xi) const {
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

    std::pair<double, bool> invert_beta(double target) const {
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

    double top_surface_residual(const std::array<double, 3>& p, double xi) const {
        const double s = std::exp(0.5 * xi);
        const double t = std::exp(-0.5 * xi);
        const double theta_value = theta(s, t);
        const double dtheta_ds_value = dtheta_ds(s, t);
        const double dtheta_dt_value = dtheta_dt(s, t);
        return p[0] * (t + theta_value * dtheta_dt_value)
            - p[1] * (s + theta_value * dtheta_ds_value)
            + p[2] * (t * dtheta_ds_value - s * dtheta_dt_value);
    }

    std::pair<double, double> top_surface_residual_and_deriv(const std::array<double, 3>& p, double xi) const {
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
};

State zero_state_like(const State& state) {
    State out;
    out.rho.assign(state.rho.size(), 0.0);
    out.m.assign(state.m.size(), 0.0);
    out.vartheta.assign(state.vartheta.size(), 0.0);
    out.rho_minus.assign(state.rho_minus.size(), 0.0);
    out.rho_plus.assign(state.rho_plus.size(), 0.0);
    out.rho_bar.assign(state.rho_bar.size(), 0.0);
    out.q_node.assign(state.q_node.size(), 0.0);
    return out;
}

State state_add_scaled(const State& left, const State& right, double scale) {
    State out = left;
    vector_axpy(out.rho, right.rho, scale);
    vector_axpy(out.m, right.m, scale);
    vector_axpy(out.vartheta, right.vartheta, scale);
    vector_axpy(out.rho_minus, right.rho_minus, scale);
    vector_axpy(out.rho_plus, right.rho_plus, scale);
    vector_axpy(out.rho_bar, right.rho_bar, scale);
    vector_axpy(out.q_node, right.q_node, scale);
    return out;
}

State state_subtract(const State& left, const State& right) {
    return state_add_scaled(left, right, -1.0);
}

double state_norm_weighted(const State& state, const GraphData& graph, int num_steps) {
    double total = 0.0;
    for (int step = 0; step <= num_steps; ++step) {
        for (int node = 0; node < graph.num_nodes; ++node) {
            const double weight = graph.pi[node];
            const double value = state.rho[row_major(step, node, graph.num_nodes)];
            total += weight * value * value;
        }
    }
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            const double weight = graph.edge_weight[edge];
            const auto edge_index = row_major(step, edge, graph.num_edges);
            total += weight * state.m[edge_index] * state.m[edge_index];
            total += weight * state.vartheta[edge_index] * state.vartheta[edge_index];
            total += weight * state.rho_minus[edge_index] * state.rho_minus[edge_index];
            total += weight * state.rho_plus[edge_index] * state.rho_plus[edge_index];
        }
        for (int node = 0; node < graph.num_nodes; ++node) {
            const double weight = graph.pi[node];
            const auto node_index = row_major(step, node, graph.num_nodes);
            total += weight * state.rho_bar[node_index] * state.rho_bar[node_index];
            total += weight * state.q_node[node_index] * state.q_node[node_index];
        }
    }
    return std::sqrt(total);
}

double dual_pair_norm_weighted(
    const std::vector<double>& m,
    const std::vector<double>& vartheta,
    const GraphData& graph,
    int num_steps
) {
    double total = 0.0;
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            const auto index = row_major(step, edge, graph.num_edges);
            const double weight = graph.edge_weight[edge];
            total += weight * (m[index] * m[index] + vartheta[index] * vartheta[index]);
        }
    }
    return std::sqrt(total);
}

void init_split_state(
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho,
    int num_steps,
    State& state
) {
    for (int step = 0; step < num_steps; ++step) {
        for (int node = 0; node < graph.num_nodes; ++node) {
            const double avg = 0.5 * (
                rho[row_major(step, node, graph.num_nodes)]
                + rho[row_major(step + 1, node, graph.num_nodes)]
            );
            state.rho_bar[row_major(step, node, graph.num_nodes)] = avg;
            state.q_node[row_major(step, node, graph.num_nodes)] = avg;
        }
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            const auto edge_index = row_major(step, edge, graph.num_edges);
            const double rho_minus = state.q_node[row_major(step, graph.src[edge], graph.num_nodes)];
            const double rho_plus = state.q_node[row_major(step, graph.dst[edge], graph.num_nodes)];
            state.rho_minus[edge_index] = rho_minus;
            state.rho_plus[edge_index] = rho_plus;
            state.vartheta[edge_index] = mean.theta(rho_minus, rho_plus);
        }
    }
}

State initialize_state(
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps
) {
    State state(num_steps, graph.num_nodes, graph.num_edges);
    for (int step = 0; step <= num_steps; ++step) {
        const double alpha = static_cast<double>(step) / static_cast<double>(num_steps);
        for (int node = 0; node < graph.num_nodes; ++node) {
            state.rho[row_major(step, node, graph.num_nodes)] = (1.0 - alpha) * rho_a[node] + alpha * rho_b[node];
        }
    }
    init_split_state(graph, mean, state.rho, num_steps, state);
    return state;
}

State build_trivial_state(
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho,
    int num_steps
) {
    State state(num_steps, graph.num_nodes, graph.num_edges);
    for (int step = 0; step <= num_steps; ++step) {
        std::copy(rho.begin(), rho.end(), state.rho.begin() + row_major(step, 0, graph.num_nodes));
    }
    init_split_state(graph, mean, state.rho, num_steps, state);
    return state;
}

double compute_action(const GraphData& graph, const State& state, int num_steps) {
    const double h = 1.0 / static_cast<double>(num_steps);
    double total = 0.0;
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            const auto index = row_major(step, edge, graph.num_edges);
            const double vartheta = state.vartheta[index];
            const double m = state.m[index];
            double safe = 0.0;
            if (vartheta > 0.0) {
                safe = (m * m) / vartheta;
            } else if (std::abs(m) > 1e-12) {
                return std::numeric_limits<double>::infinity();
            }
            total += h * graph.edge_weight[edge] * safe;
        }
    }
    return total;
}

void continuity_residual(
    const GraphData& graph,
    const std::vector<double>& rho,
    const std::vector<double>& m,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    std::vector<double>& out
) {
    out.assign(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(graph.num_nodes), 0.0);
    const double h = 1.0 / static_cast<double>(num_steps);
    for (int step = 0; step < num_steps; ++step) {
        for (int node = 0; node < graph.num_nodes; ++node) {
            double delta = rho[row_major(step + 1, node, graph.num_nodes)]
                - rho[row_major(step, node, graph.num_nodes)];
            if (step == 0) {
                delta = rho[row_major(1, node, graph.num_nodes)] - rho_a[node];
            }
            if (step == num_steps - 1) {
                delta = rho_b[node] - rho[row_major(num_steps - 1, node, graph.num_nodes)];
            }
            out[row_major(step, node, graph.num_nodes)] = delta / h;
        }
    }
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            const double contribution = 0.5 * graph.q[edge] * (
                m[row_major(step, graph.rev[edge], graph.num_edges)]
                - m[row_major(step, edge, graph.num_edges)]
            );
            out[row_major(step, graph.src[edge], graph.num_nodes)] += contribution;
        }
    }
}

void solve_tridiagonal(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs,
    std::vector<double>& out
) {
    const int size = static_cast<int>(rhs.size());
    std::vector<double> c_prime(size, 0.0);
    std::vector<double> d_prime(size, 0.0);
    c_prime[0] = size > 1 ? upper[0] / diag[0] : 0.0;
    d_prime[0] = rhs[0] / diag[0];
    for (int idx = 1; idx < size; ++idx) {
        const double denom = diag[idx] - lower[idx] * c_prime[idx - 1];
        c_prime[idx] = idx < size - 1 ? upper[idx] / denom : 0.0;
        d_prime[idx] = (rhs[idx] - lower[idx] * d_prime[idx - 1]) / denom;
    }
    out.assign(size, 0.0);
    out[size - 1] = d_prime[size - 1];
    for (int idx = size - 2; idx >= 0; --idx) {
        out[idx] = d_prime[idx] - c_prime[idx] * out[idx + 1];
    }
}

void solve_tridiagonal_javg(const std::vector<double>& rhs, int num_steps, int num_nodes, std::vector<double>& out) {
    std::vector<double> lower(num_steps, 0.25);
    std::vector<double> diag(num_steps, 1.5);
    std::vector<double> upper(num_steps, 0.25);
    lower[0] = 0.0;
    upper[num_steps - 1] = 0.0;
    diag[0] = 1.25;
    diag[num_steps - 1] = 1.25;

    out.assign(rhs.size(), 0.0);
    std::vector<double> rhs_node(num_steps, 0.0);
    std::vector<double> sol;
    for (int node = 0; node < num_nodes; ++node) {
        for (int step = 0; step < num_steps; ++step) {
            rhs_node[step] = rhs[row_major(step, node, num_nodes)];
        }
        solve_tridiagonal(lower, diag, upper, rhs_node, sol);
        for (int step = 0; step < num_steps; ++step) {
            out[row_major(step, node, num_nodes)] = sol[step];
        }
    }
}

void project_javg(
    const std::vector<double>& rho,
    const std::vector<double>& rho_bar,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    int num_nodes,
    std::vector<double>& rho_pr,
    std::vector<double>& rho_bar_pr
) {
    std::vector<double> rhs(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(num_nodes), 0.0);
    for (int step = 0; step < num_steps; ++step) {
        for (int node = 0; node < num_nodes; ++node) {
            double value = rho_bar[row_major(step, node, num_nodes)]
                - 0.5 * (
                    rho[row_major(step, node, num_nodes)]
                    + rho[row_major(step + 1, node, num_nodes)]
                );
            if (step == 0) {
                value = rho_bar[row_major(0, node, num_nodes)] - 0.5 * (
                    rho_a[node] + rho[row_major(1, node, num_nodes)]
                );
            }
            if (step == num_steps - 1) {
                value = rho_bar[row_major(num_steps - 1, node, num_nodes)] - 0.5 * (
                    rho_b[node] + rho[row_major(num_steps - 1, node, num_nodes)]
                );
            }
            rhs[row_major(step, node, num_nodes)] = value;
        }
    }

    std::vector<double> lam;
    solve_tridiagonal_javg(rhs, num_steps, num_nodes, lam);

    rho_pr = rho;
    for (int node = 0; node < num_nodes; ++node) {
        rho_pr[row_major(0, node, num_nodes)] = rho_a[node];
        rho_pr[row_major(num_steps, node, num_nodes)] = rho_b[node];
    }
    for (int step = 1; step < num_steps; ++step) {
        for (int node = 0; node < num_nodes; ++node) {
            rho_pr[row_major(step, node, num_nodes)] = rho[row_major(step, node, num_nodes)]
                + 0.5 * (
                    lam[row_major(step - 1, node, num_nodes)]
                    + lam[row_major(step, node, num_nodes)]
                );
        }
    }
    rho_bar_pr.resize(rho_bar.size());
    for (std::size_t idx = 0; idx < rho_bar.size(); ++idx) {
        rho_bar_pr[idx] = rho_bar[idx] - lam[idx];
    }
}

void prox_i_star_javg(
    const std::vector<double>& rho,
    const std::vector<double>& rho_bar,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    int num_nodes,
    std::vector<double>& rho_out,
    std::vector<double>& rho_bar_out
) {
    std::vector<double> rho_pr;
    std::vector<double> rho_bar_pr;
    project_javg(rho, rho_bar, rho_a, rho_b, num_steps, num_nodes, rho_pr, rho_bar_pr);
    rho_out.resize(rho.size());
    rho_bar_out.resize(rho_bar.size());
    for (std::size_t idx = 0; idx < rho.size(); ++idx) {
        rho_out[idx] = rho[idx] - rho_pr[idx];
    }
    for (std::size_t idx = 0; idx < rho_bar.size(); ++idx) {
        rho_bar_out[idx] = rho_bar[idx] - rho_bar_pr[idx];
    }
}

void project_jeq(
    const std::vector<double>& rho_bar,
    const std::vector<double>& q_node,
    std::vector<double>& rho_bar_pr,
    std::vector<double>& q_node_pr
) {
    rho_bar_pr.resize(rho_bar.size());
    q_node_pr.resize(q_node.size());
    for (std::size_t idx = 0; idx < rho_bar.size(); ++idx) {
        const double mid = 0.5 * (rho_bar[idx] + q_node[idx]);
        rho_bar_pr[idx] = mid;
        q_node_pr[idx] = mid;
    }
}

void project_jpm(
    const GraphData& graph,
    const std::vector<double>& q_node,
    const std::vector<double>& rho_minus,
    const std::vector<double>& rho_plus,
    int num_steps,
    std::vector<double>& q_node_pr,
    std::vector<double>& rho_minus_pr,
    std::vector<double>& rho_plus_pr
) {
    q_node_pr = q_node;
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            const double term = 0.5 * graph.q[edge] * (
                rho_minus[row_major(step, edge, graph.num_edges)]
                + rho_plus[row_major(step, graph.rev[edge], graph.num_edges)]
            );
            q_node_pr[row_major(step, graph.src[edge], graph.num_nodes)] += term;
        }
        for (int node = 0; node < graph.num_nodes; ++node) {
            q_node_pr[row_major(step, node, graph.num_nodes)] /= graph.one_plus_out_rate[node];
        }
    }
    rho_minus_pr.resize(rho_minus.size());
    rho_plus_pr.resize(rho_plus.size());
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            rho_minus_pr[row_major(step, edge, graph.num_edges)] =
                q_node_pr[row_major(step, graph.src[edge], graph.num_nodes)];
            rho_plus_pr[row_major(step, edge, graph.num_edges)] =
                q_node_pr[row_major(step, graph.dst[edge], graph.num_nodes)];
        }
    }
}

void prox_i_star_jpm(
    const GraphData& graph,
    const std::vector<double>& q_node,
    const std::vector<double>& rho_minus,
    const std::vector<double>& rho_plus,
    int num_steps,
    std::vector<double>& q_node_out,
    std::vector<double>& rho_minus_out,
    std::vector<double>& rho_plus_out
) {
    std::vector<double> q_pr;
    std::vector<double> rho_minus_pr;
    std::vector<double> rho_plus_pr;
    project_jpm(graph, q_node, rho_minus, rho_plus, num_steps, q_pr, rho_minus_pr, rho_plus_pr);
    q_node_out.resize(q_node.size());
    rho_minus_out.resize(rho_minus.size());
    rho_plus_out.resize(rho_plus.size());
    for (std::size_t idx = 0; idx < q_node.size(); ++idx) {
        q_node_out[idx] = q_node[idx] - q_pr[idx];
    }
    for (std::size_t idx = 0; idx < rho_minus.size(); ++idx) {
        rho_minus_out[idx] = rho_minus[idx] - rho_minus_pr[idx];
        rho_plus_out[idx] = rho_plus[idx] - rho_plus_pr[idx];
    }
}

void prox_a_star(
    const std::vector<double>& vartheta,
    const std::vector<double>& m,
    int newton_iters,
    std::vector<double>& vartheta_out,
    std::vector<double>& m_out
) {
    vartheta_out.resize(vartheta.size());
    m_out.resize(m.size());
    for (std::size_t idx = 0; idx < vartheta.size(); ++idx) {
        const bool feasible = vartheta[idx] + 0.25 * m[idx] * m[idx] <= 0.0;
        double v = m[idx];
        for (int iter = 0; iter < newton_iters; ++iter) {
            const double f = v * v * v + 4.0 * (vartheta[idx] + 2.0) * v - 8.0 * m[idx];
            const double df = 3.0 * v * v + 4.0 * (vartheta[idx] + 2.0);
            const double step = f / (std::abs(df) > 1e-12 ? df : 1.0);
            v -= step;
        }
        const double p_proj = -0.25 * v * v;
        const double q_proj = v;
        vartheta_out[idx] = feasible ? vartheta[idx] : p_proj;
        m_out[idx] = feasible ? m[idx] : q_proj;
    }
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
    for (std::size_t idx = 0; idx < rho_minus.size(); ++idx) {
        const auto proj = project_k_point(mean, rho_minus[idx], rho_plus[idx], vartheta[idx]);
        rho_minus_pr[idx] = proj[0];
        rho_plus_pr[idx] = proj[1];
        vartheta_pr[idx] = proj[2];
    }
}

void apply_ceh_constraint_zero_boundary(
    const GraphData& graph,
    const std::vector<double>& drho_int,
    const std::vector<double>& dm,
    int num_steps,
    std::vector<double>& out
) {
    out.assign(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(graph.num_nodes), 0.0);
    const double h = 1.0 / static_cast<double>(num_steps);

    for (int node = 0; node < graph.num_nodes; ++node) {
        out[row_major(0, node, graph.num_nodes)] = drho_int[row_major(0, node, graph.num_nodes)] / h;
    }
    for (int step = 1; step < num_steps - 1; ++step) {
        for (int node = 0; node < graph.num_nodes; ++node) {
            out[row_major(step, node, graph.num_nodes)] = (
                drho_int[row_major(step, node, graph.num_nodes)]
                - drho_int[row_major(step - 1, node, graph.num_nodes)]
            ) / h;
        }
    }
    for (int node = 0; node < graph.num_nodes; ++node) {
        out[row_major(num_steps - 1, node, graph.num_nodes)] =
            -drho_int[row_major(num_steps - 2, node, graph.num_nodes)] / h;
    }
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            const double contribution = 0.5 * graph.q[edge] * (
                dm[row_major(step, graph.rev[edge], graph.num_edges)]
                - dm[row_major(step, edge, graph.num_edges)]
            );
            out[row_major(step, graph.src[edge], graph.num_nodes)] += contribution;
        }
    }
}

void apply_ceh_constraint_transpose_zero_boundary(
    const GraphData& graph,
    const std::vector<double>& phi,
    int num_steps,
    std::vector<double>& drho_int_adj,
    std::vector<double>& dm_adj
) {
    drho_int_adj.assign(static_cast<std::size_t>(num_steps - 1) * static_cast<std::size_t>(graph.num_nodes), 0.0);
    dm_adj.assign(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(graph.num_edges), 0.0);
    const double h = 1.0 / static_cast<double>(num_steps);
    for (int step = 0; step < num_steps - 1; ++step) {
        for (int node = 0; node < graph.num_nodes; ++node) {
            drho_int_adj[row_major(step, node, graph.num_nodes)] = (
                phi[row_major(step, node, graph.num_nodes)]
                - phi[row_major(step + 1, node, graph.num_nodes)]
            ) / h;
        }
    }
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            dm_adj[row_major(step, edge, graph.num_edges)] = 0.5 * (
                graph.q[graph.rev[edge]] * phi[row_major(step, graph.dst[edge], graph.num_nodes)]
                - graph.q[edge] * phi[row_major(step, graph.src[edge], graph.num_nodes)]
            );
        }
    }
}

void apply_jacobi_preconditioner(
    const GraphData& graph,
    const std::vector<double>& value,
    int num_steps,
    std::vector<double>& out
) {
    out = value;
    project_zero_mean(out);
    const double h = 1.0 / static_cast<double>(num_steps);
    for (int step = 0; step < num_steps; ++step) {
        const double time_diag = (step == 0 || step == num_steps - 1) ? 1.0 / (h * h) : 2.0 / (h * h);
        for (int node = 0; node < graph.num_nodes; ++node) {
            const double pi_inv = 1.0 / std::max(graph.pi[node], kTiny);
            const double diag = std::max((time_diag + 0.5 * graph.out_rate[node]) * pi_inv, 1e-12);
            out[row_major(step, node, graph.num_nodes)] /= diag;
        }
    }
    project_zero_mean(out);
}

void apply_block_jacobi_preconditioner(
    const GraphData& graph,
    const std::vector<double>& value,
    int num_steps,
    std::vector<double>& out
) {
    out = value;
    project_zero_mean(out);
    const double h = 1.0 / static_cast<double>(num_steps);
    std::vector<double> lower(num_steps, -1.0 / (h * h));
    std::vector<double> upper(num_steps, -1.0 / (h * h));
    lower[0] = 0.0;
    upper[num_steps - 1] = 0.0;

    std::vector<double> diag(num_steps, 0.0);
    std::vector<double> rhs_node(num_steps, 0.0);
    std::vector<double> lower_node(num_steps, 0.0);
    std::vector<double> upper_node(num_steps, 0.0);
    std::vector<double> sol;

    for (int node = 0; node < graph.num_nodes; ++node) {
        const double pi_inv = 1.0 / std::max(graph.pi[node], kTiny);
        for (int step = 0; step < num_steps; ++step) {
            const double time_diag = (step == 0 || step == num_steps - 1) ? 1.0 / (h * h) : 2.0 / (h * h);
            diag[step] = std::max((time_diag + 0.5 * graph.out_rate[node]) * pi_inv, 1e-12);
            lower_node[step] = lower[step] * pi_inv;
            upper_node[step] = upper[step] * pi_inv;
            rhs_node[step] = out[row_major(step, node, graph.num_nodes)];
        }
        solve_tridiagonal(lower_node, diag, upper_node, rhs_node, sol);
        for (int step = 0; step < num_steps; ++step) {
            out[row_major(step, node, graph.num_nodes)] = sol[step];
        }
    }
    project_zero_mean(out);
}

struct CgResult {
    std::vector<double> x;
    double residual;
    int iters_used;
};

template <typename MatVec, typename Preconditioner>
CgResult conjugate_gradient(
    MatVec&& matvec,
    const std::vector<double>& b,
    int max_iters,
    double tol,
    const std::optional<std::vector<double>>& x0,
    Preconditioner&& preconditioner
) {
    std::vector<double> x = x0.has_value() ? *x0 : std::vector<double>(b.size(), 0.0);
    std::vector<double> ap;
    matvec(x, ap);

    std::vector<double> r(b.size(), 0.0);
    for (std::size_t idx = 0; idx < b.size(); ++idx) {
        r[idx] = b[idx] - ap[idx];
    }
    std::vector<double> z;
    preconditioner(r, z);
    std::vector<double> p = z;
    double rz_old = vector_dot(r, z);
    double residual = vector_norm(r);
    int iters_used = 0;
    if (residual <= tol) {
        return {std::move(x), residual, iters_used};
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        matvec(p, ap);
        const double denom = std::max(vector_dot(p, ap), kTiny);
        const double alpha = rz_old / denom;
        vector_axpy(x, p, alpha);
        vector_axpy(r, ap, -alpha);
        residual = vector_norm(r);
        ++iters_used;
        if (residual <= tol) {
            break;
        }
        preconditioner(r, z);
        const double rz_new = vector_dot(r, z);
        const double beta = rz_new / std::max(rz_old, kTiny);
        for (std::size_t idx = 0; idx < p.size(); ++idx) {
            p[idx] = z[idx] + beta * p[idx];
        }
        rz_old = rz_new;
    }
    return {std::move(x), residual, iters_used};
}

void project_ceh(
    const GraphData& graph,
    const std::vector<double>& rho,
    const std::vector<double>& m,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    int cg_max_iters,
    double cg_tol,
    const std::optional<std::vector<double>>& phi0,
    bool cg_warm_start,
    bool block_jacobi,
    std::vector<double>& rho_pr,
    std::vector<double>& m_pr,
    std::vector<double>& phi,
    CehStats& stats
) {
    auto matvec = [&](const std::vector<double>& value, std::vector<double>& out) {
        std::vector<double> phi_local = value;
        project_zero_mean(phi_local);
        std::vector<double> drho_int_adj;
        std::vector<double> dm_adj;
        apply_ceh_constraint_transpose_zero_boundary(graph, phi_local, num_steps, drho_int_adj, dm_adj);
        std::vector<double> drho_int(drho_int_adj.size(), 0.0);
        std::vector<double> dm(dm_adj.size(), 0.0);
        for (int step = 0; step < num_steps - 1; ++step) {
            for (int node = 0; node < graph.num_nodes; ++node) {
                drho_int[row_major(step, node, graph.num_nodes)] =
                    drho_int_adj[row_major(step, node, graph.num_nodes)] / std::max(graph.pi[node], kTiny);
            }
        }
        for (int step = 0; step < num_steps; ++step) {
            for (int edge = 0; edge < graph.num_edges; ++edge) {
                dm[row_major(step, edge, graph.num_edges)] =
                    dm_adj[row_major(step, edge, graph.num_edges)] / std::max(graph.edge_weight[edge], kTiny);
            }
        }
        apply_ceh_constraint_zero_boundary(graph, drho_int, dm, num_steps, out);
        project_zero_mean(out);
    };

    std::vector<double> rhs;
    continuity_residual(graph, rho, m, rho_a, rho_b, num_steps, rhs);
    project_zero_mean(rhs);

    auto preconditioner = [&](const std::vector<double>& value, std::vector<double>& out) {
        if (block_jacobi) {
            apply_block_jacobi_preconditioner(graph, value, num_steps, out);
        } else {
            apply_jacobi_preconditioner(graph, value, num_steps, out);
        }
    };

    const std::optional<std::vector<double>> x0 =
        (cg_warm_start && phi0.has_value()) ? phi0 : std::optional<std::vector<double>>{};
    CgResult cg = conjugate_gradient(matvec, rhs, cg_max_iters, cg_tol, x0, preconditioner);
    phi = std::move(cg.x);
    project_zero_mean(phi);

    std::vector<double> drho_int_adj;
    std::vector<double> dm_adj;
    apply_ceh_constraint_transpose_zero_boundary(graph, phi, num_steps, drho_int_adj, dm_adj);

    rho_pr = rho;
    for (int node = 0; node < graph.num_nodes; ++node) {
        rho_pr[row_major(0, node, graph.num_nodes)] = rho_a[node];
        rho_pr[row_major(num_steps, node, graph.num_nodes)] = rho_b[node];
    }
    for (int step = 1; step < num_steps; ++step) {
        for (int node = 0; node < graph.num_nodes; ++node) {
            rho_pr[row_major(step, node, graph.num_nodes)] =
                rho[row_major(step, node, graph.num_nodes)]
                - drho_int_adj[row_major(step - 1, node, graph.num_nodes)] / std::max(graph.pi[node], kTiny);
        }
    }
    m_pr = m;
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            m_pr[row_major(step, edge, graph.num_edges)] =
                m[row_major(step, edge, graph.num_edges)]
                - dm_adj[row_major(step, edge, graph.num_edges)] / std::max(graph.edge_weight[edge], kTiny);
        }
    }
    stats = {cg.residual, cg.iters_used};
}

State prox_f_star(
    const State& state,
    const GraphData& graph,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    int newton_iters
) {
    State out = state;
    prox_a_star(state.vartheta, state.m, newton_iters, out.vartheta, out.m);
    prox_i_star_jpm(
        graph,
        state.q_node,
        state.rho_minus,
        state.rho_plus,
        num_steps,
        out.q_node,
        out.rho_minus,
        out.rho_plus
    );
    prox_i_star_javg(
        state.rho,
        state.rho_bar,
        rho_a,
        rho_b,
        num_steps,
        graph.num_nodes,
        out.rho,
        out.rho_bar
    );
    return out;
}

State prox_g(
    const State& state,
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    const SolverConfig& config,
    const std::optional<std::vector<double>>& phi0,
    std::vector<double>& phi,
    CehStats& ceh_stats
) {
    std::vector<double> rho_pr;
    std::vector<double> m_pr;
    project_ceh(
        graph,
        state.rho,
        state.m,
        rho_a,
        rho_b,
        num_steps,
        config.cg_max_iters,
        config.cg_tol,
        phi0,
        config.cg_warm_start,
        config.block_jacobi,
        rho_pr,
        m_pr,
        phi,
        ceh_stats
    );

    std::vector<double> rho_minus_pr;
    std::vector<double> rho_plus_pr;
    std::vector<double> vartheta_pr;
    project_k(mean, state.rho_minus, state.rho_plus, state.vartheta, rho_minus_pr, rho_plus_pr, vartheta_pr);

    std::vector<double> rho_bar_pr;
    std::vector<double> q_node_pr;
    project_jeq(state.rho_bar, state.q_node, rho_bar_pr, q_node_pr);

    State out;
    out.rho = std::move(rho_pr);
    out.m = std::move(m_pr);
    out.vartheta = std::move(vartheta_pr);
    out.rho_minus = std::move(rho_minus_pr);
    out.rho_plus = std::move(rho_plus_pr);
    out.rho_bar = std::move(rho_bar_pr);
    out.q_node = std::move(q_node_pr);
    return out;
}

Diagnostics compute_diagnostics(
    const State& primal,
    const State& prev_primal,
    const State& dual,
    const State& prev_dual,
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    const CehStats& ceh_stats
) {
    const State primal_diff = state_subtract(primal, prev_primal);
    const double primal_delta = state_norm_weighted(primal_diff, graph, num_steps)
        / (1.0 + state_norm_weighted(prev_primal, graph, num_steps));

    std::vector<double> dual_m_diff = dual.m;
    std::vector<double> dual_vartheta_diff = dual.vartheta;
    for (std::size_t idx = 0; idx < dual_m_diff.size(); ++idx) {
        dual_m_diff[idx] -= prev_dual.m[idx];
        dual_vartheta_diff[idx] -= prev_dual.vartheta[idx];
    }
    const double dual_delta = dual_pair_norm_weighted(dual_m_diff, dual_vartheta_diff, graph, num_steps)
        / (1.0 + dual_pair_norm_weighted(prev_dual.m, prev_dual.vartheta, graph, num_steps));

    std::vector<double> continuity;
    continuity_residual(graph, primal.rho, primal.m, rho_a, rho_b, num_steps, continuity);
    double continuity_max = 0.0;
    for (double value : continuity) {
        continuity_max = std::max(continuity_max, std::abs(value));
    }

    double k_violation = 0.0;
    for (std::size_t idx = 0; idx < primal.vartheta.size(); ++idx) {
        const double theta_value = mean.theta(primal.rho_minus[idx], primal.rho_plus[idx]);
        const double upper_slack = std::max(primal.vartheta[idx] - theta_value, 0.0);
        const double lower_slack = std::max(-primal.vartheta[idx], 0.0);
        const double negativity = std::max(
            std::max(-primal.rho_minus[idx], 0.0),
            std::max(-primal.rho_plus[idx], 0.0)
        );
        k_violation = std::max(k_violation, std::max(std::max(upper_slack, lower_slack), negativity));
    }

    double endpoint_residual = 0.0;
    for (int node = 0; node < graph.num_nodes; ++node) {
        endpoint_residual = std::max(endpoint_residual, std::abs(primal.rho[row_major(0, node, graph.num_nodes)] - rho_a[node]));
        endpoint_residual = std::max(
            endpoint_residual,
            std::abs(primal.rho[row_major(num_steps, node, graph.num_nodes)] - rho_b[node])
        );
    }
    const double max_constraint = std::max(std::max(continuity_max, k_violation), endpoint_residual);
    return {
        primal_delta,
        dual_delta,
        continuity_max,
        k_violation,
        endpoint_residual,
        max_constraint,
        ceh_stats.cg_residual,
        ceh_stats.cg_iters,
    };
}

State build_linear_warm_start(
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    const SolverConfig& config
) {
    State base = initialize_state(graph, mean, rho_a, rho_b, num_steps);
    std::vector<double> rho_pr;
    std::vector<double> m_pr;
    std::vector<double> phi;
    CehStats stats{};
    project_ceh(
        graph,
        base.rho,
        base.m,
        rho_a,
        rho_b,
        num_steps,
        config.cg_max_iters,
        config.cg_tol,
        std::nullopt,
        false,
        config.block_jacobi,
        rho_pr,
        m_pr,
        phi,
        stats
    );
    State out(num_steps, graph.num_nodes, graph.num_edges);
    out.rho = std::move(rho_pr);
    out.m = std::move(m_pr);
    init_split_state(graph, mean, out.rho, num_steps, out);
    return out;
}

SolveResult run_solver(
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    const SolverConfig& config
) {
    const double max_diff = [&]() {
        double diff = 0.0;
        for (int node = 0; node < graph.num_nodes; ++node) {
            diff = std::max(diff, std::abs(rho_a[node] - rho_b[node]));
        }
        return diff;
    }();
    if (max_diff <= 1e-12) {
        SolveResult result;
        result.state = build_trivial_state(graph, mean, rho_a, num_steps);
        result.diagnostics = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0};
        result.iterations_used = 1;
        result.converged = true;
        result.action = 0.0;
        result.distance = 0.0;
        if (config.record_debug_trace) {
            DebugTrace trace;
            trace.iterations = {1};
            trace.action = {0.0};
            trace.continuity_residual = {0.0};
            trace.primal_delta = {0.0};
            trace.dual_delta = {0.0};
            trace.max_constraint_residual = {0.0};
            trace.ceh_cg_residual = {0.0};
            trace.ceh_cg_iters = {0};
            trace.min_vartheta = {*std::min_element(result.state.vartheta.begin(), result.state.vartheta.end())};
            trace.num_records = 1;
            result.debug_trace = std::move(trace);
        }
        return result;
    }

    State primal = config.linear_warm_start
        ? build_linear_warm_start(graph, mean, rho_a, rho_b, num_steps, config)
        : initialize_state(graph, mean, rho_a, rho_b, num_steps);
    State dual = zero_state_like(primal);
    State primal_bar = primal;
    std::vector<double> phi_cache(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(graph.num_nodes), 0.0);
    Diagnostics diagnostics = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        0,
    };

    DebugTrace trace;
    if (config.record_debug_trace) {
        const int trace_length = (config.max_iters + config.check_every - 1) / config.check_every;
        trace.iterations.assign(trace_length, 0);
        trace.action.assign(trace_length, 0.0);
        trace.continuity_residual.assign(trace_length, 0.0);
        trace.primal_delta.assign(trace_length, 0.0);
        trace.dual_delta.assign(trace_length, 0.0);
        trace.max_constraint_residual.assign(trace_length, 0.0);
        trace.ceh_cg_residual.assign(trace_length, 0.0);
        trace.ceh_cg_iters.assign(trace_length, 0);
        trace.min_vartheta.assign(trace_length, 0.0);
        trace.num_records = 0;
    }

    int iterations_used = 0;
    bool converged = false;
    while (!converged && iterations_used < config.max_iters) {
        State dual_trial = state_add_scaled(dual, primal_bar, config.sigma);
        State dual_next = prox_f_star(dual_trial, graph, rho_a, rho_b, num_steps, config.newton_iters);
        State primal_trial = state_add_scaled(primal, dual_next, -config.tau);

        std::vector<double> phi_next;
        CehStats ceh_stats{};
        State primal_next = prox_g(
            primal_trial,
            graph,
            mean,
            rho_a,
            rho_b,
            num_steps,
            config,
            phi_cache,
            phi_next,
            ceh_stats
        );
        State primal_bar_next = state_add_scaled(
            primal_next,
            state_subtract(primal_next, primal),
            config.relaxation
        );
        diagnostics = compute_diagnostics(
            primal_next,
            primal,
            dual_next,
            dual,
            graph,
            mean,
            rho_a,
            rho_b,
            num_steps,
            ceh_stats
        );
        ++iterations_used;
        const bool should_check = (iterations_used % config.check_every == 0) || (iterations_used == config.max_iters);
        converged = should_check
            && (diagnostics.primal_delta <= config.residual_tol)
            && (diagnostics.dual_delta <= config.residual_tol)
            && (diagnostics.max_constraint_residual <= config.feasibility_tol);

        if (config.record_debug_trace && should_check) {
            const int slot = trace.num_records;
            trace.iterations[slot] = static_cast<int32_t>(iterations_used);
            trace.action[slot] = compute_action(graph, primal_next, num_steps);
            trace.continuity_residual[slot] = diagnostics.continuity_residual;
            trace.primal_delta[slot] = diagnostics.primal_delta;
            trace.dual_delta[slot] = diagnostics.dual_delta;
            trace.max_constraint_residual[slot] = diagnostics.max_constraint_residual;
            trace.ceh_cg_residual[slot] = diagnostics.ceh_cg_residual;
            trace.ceh_cg_iters[slot] = static_cast<int32_t>(diagnostics.ceh_cg_iters);
            trace.min_vartheta[slot] = *std::min_element(primal_next.vartheta.begin(), primal_next.vartheta.end());
            ++trace.num_records;
        }

        primal = std::move(primal_next);
        dual = std::move(dual_next);
        primal_bar = std::move(primal_bar_next);
        phi_cache = std::move(phi_next);
    }

    const double action = compute_action(graph, primal, num_steps);
    const bool converged_flag = converged
        && std::isfinite(action)
        && diagnostics.max_constraint_residual <= config.feasibility_tol;

    SolveResult result;
    result.state = std::move(primal);
    result.diagnostics = diagnostics;
    result.iterations_used = iterations_used;
    result.converged = converged_flag;
    result.action = action;
    result.distance = std::sqrt(std::max(action, 0.0));
    if (config.record_debug_trace) {
        result.debug_trace = std::move(trace);
    }
    return result;
}

py::dict solve_ot_cpp(
    int num_nodes,
    int num_edges,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> dst,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> rev,
    py::array_t<double, py::array::c_style | py::array::forcecast> q,
    py::array_t<double, py::array::c_style | py::array::forcecast> pi,
    py::array_t<double, py::array::c_style | py::array::forcecast> out_rate,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho_b,
    int num_steps,
    double tau,
    double sigma,
    double relaxation,
    int max_iters,
    int check_every,
    double residual_tol,
    double feasibility_tol,
    int newton_iters,
    int cg_max_iters,
    double cg_tol,
    bool cg_warm_start,
    const std::string& cg_preconditioner,
    const std::string& warm_start,
    bool record_debug_trace,
    double mean_eps_diag,
    double mean_xi_max,
    int mean_newton_iters,
    int mean_bisect_iters
) {
    check_size(src.size(), static_cast<std::size_t>(num_edges), "src");
    check_size(dst.size(), static_cast<std::size_t>(num_edges), "dst");
    check_size(rev.size(), static_cast<std::size_t>(num_edges), "rev");
    check_size(q.size(), static_cast<std::size_t>(num_edges), "q");
    check_size(pi.size(), static_cast<std::size_t>(num_nodes), "pi");
    check_size(out_rate.size(), static_cast<std::size_t>(num_nodes), "out_rate");
    check_size(rho_a.size(), static_cast<std::size_t>(num_nodes), "rho_a");
    check_size(rho_b.size(), static_cast<std::size_t>(num_nodes), "rho_b");

    GraphData graph(
        num_nodes,
        num_edges,
        src.data(),
        dst.data(),
        rev.data(),
        q.data(),
        pi.data(),
        out_rate.data()
    );
    MeanConfig mean_config{mean_eps_diag, mean_xi_max, mean_newton_iters, mean_bisect_iters};
    SolverConfig solver_config{
        tau,
        sigma,
        relaxation,
        max_iters,
        check_every,
        residual_tol,
        feasibility_tol,
        newton_iters,
        cg_max_iters,
        cg_tol,
        cg_warm_start,
        cg_preconditioner == "block_jacobi",
        warm_start == "linear_path",
        record_debug_trace,
    };
    LogMeanOpsCpp mean(mean_config);
    std::vector<double> rho_a_vec(rho_a.data(), rho_a.data() + rho_a.size());
    std::vector<double> rho_b_vec(rho_b.data(), rho_b.data() + rho_b.size());
    SolveResult result = run_solver(graph, mean, rho_a_vec, rho_b_vec, num_steps, solver_config);

    py::dict payload;
    payload["rho"] = make_array_from_vector(result.state.rho, {num_steps + 1, num_nodes});
    payload["m"] = make_array_from_vector(result.state.m, {num_steps, num_edges});
    payload["vartheta"] = make_array_from_vector(result.state.vartheta, {num_steps, num_edges});
    payload["rho_minus"] = make_array_from_vector(result.state.rho_minus, {num_steps, num_edges});
    payload["rho_plus"] = make_array_from_vector(result.state.rho_plus, {num_steps, num_edges});
    payload["rho_bar"] = make_array_from_vector(result.state.rho_bar, {num_steps, num_nodes});
    payload["q_node"] = make_array_from_vector(result.state.q_node, {num_steps, num_nodes});
    payload["distance"] = result.distance;
    payload["action"] = result.action;
    payload["iterations_used"] = result.iterations_used;
    payload["converged"] = result.converged;
    payload["primal_delta"] = result.diagnostics.primal_delta;
    payload["dual_delta"] = result.diagnostics.dual_delta;
    payload["continuity_residual"] = result.diagnostics.continuity_residual;
    payload["k_violation"] = result.diagnostics.k_violation;
    payload["endpoint_residual"] = result.diagnostics.endpoint_residual;
    payload["max_constraint_residual"] = result.diagnostics.max_constraint_residual;
    payload["ceh_cg_residual"] = result.diagnostics.ceh_cg_residual;
    payload["ceh_cg_iters"] = result.diagnostics.ceh_cg_iters;
    payload["simd_level"] = std::string(simd_ops().name);

    if (result.debug_trace.has_value()) {
        const DebugTrace& trace = *result.debug_trace;
        py::dict trace_payload;
        trace_payload["iterations"] = make_array_from_vector(trace.iterations, {static_cast<py::ssize_t>(trace.iterations.size())});
        trace_payload["action"] = make_array_from_vector(trace.action, {static_cast<py::ssize_t>(trace.action.size())});
        trace_payload["continuity_residual"] = make_array_from_vector(
            trace.continuity_residual,
            {static_cast<py::ssize_t>(trace.continuity_residual.size())}
        );
        trace_payload["primal_delta"] = make_array_from_vector(trace.primal_delta, {static_cast<py::ssize_t>(trace.primal_delta.size())});
        trace_payload["dual_delta"] = make_array_from_vector(trace.dual_delta, {static_cast<py::ssize_t>(trace.dual_delta.size())});
        trace_payload["max_constraint_residual"] = make_array_from_vector(
            trace.max_constraint_residual,
            {static_cast<py::ssize_t>(trace.max_constraint_residual.size())}
        );
        trace_payload["ceh_cg_residual"] = make_array_from_vector(
            trace.ceh_cg_residual,
            {static_cast<py::ssize_t>(trace.ceh_cg_residual.size())}
        );
        trace_payload["ceh_cg_iters"] = make_array_from_vector(
            trace.ceh_cg_iters,
            {static_cast<py::ssize_t>(trace.ceh_cg_iters.size())}
        );
        trace_payload["min_vartheta"] = make_array_from_vector(
            trace.min_vartheta,
            {static_cast<py::ssize_t>(trace.min_vartheta.size())}
        );
        trace_payload["num_records"] = trace.num_records;
        payload["debug_trace"] = std::move(trace_payload);
    } else {
        payload["debug_trace"] = py::none();
    }
    return payload;
}

py::dict extension_info() {
    py::dict info;
    info["simd_level"] = std::string(simd_ops().name);
    info["compiled_with_x86_intrinsics"] =
#if defined(__x86_64__) || defined(_M_X64)
        true;
#else
        false;
#endif
    return info;
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "C++ CPU solver core for graphot";
    module.def("solve_ot_cpp", &solve_ot_cpp);
    module.def("extension_info", &extension_info);
}
