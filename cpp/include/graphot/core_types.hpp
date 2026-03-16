#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace graphot::core {

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
    bool verbose;
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
    std::vector<double> k_violation;
    std::vector<double> endpoint_residual;
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

}  // namespace graphot::core
