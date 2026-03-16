#include "graphot/prox_operators.hpp"

#include <algorithm>
#include <cmath>

#include "graphot/runtime.hpp"

#if defined(GRAPHOT_USE_OPENMP)
#include <omp.h>
#endif

namespace graphot::core {

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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(rhs.size()))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(static_cast<std::size_t>(num_nodes)))
#endif
    for (int node = 0; node < num_nodes; ++node) {
        rho_pr[row_major(0, node, num_nodes)] = rho_a[node];
        rho_pr[row_major(num_steps, node, num_nodes)] = rho_b[node];
    }
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(static_cast<std::size_t>(num_steps - 1) * num_nodes))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(rho_bar.size()))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(rho.size()))
#endif
    for (std::size_t idx = 0; idx < rho.size(); ++idx) {
        rho_out[idx] = rho[idx] - rho_pr[idx];
    }
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(rho_bar.size()))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(rho_bar.size()))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(static_cast<std::size_t>(num_steps) * (graph.num_nodes + graph.num_edges)))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(rho_minus.size()))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(q_node.size()))
#endif
    for (std::size_t idx = 0; idx < q_node.size(); ++idx) {
        q_node_out[idx] = q_node[idx] - q_pr[idx];
    }
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(rho_minus.size()))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(vartheta.size()))
#endif
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

}  // namespace graphot::core
