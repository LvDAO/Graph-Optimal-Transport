#include "graphot/state_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "graphot/runtime.hpp"

#if defined(GRAPHOT_USE_OPENMP)
#include <omp.h>
#endif

namespace graphot::core {

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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for reduction(+:total) if(should_parallelize(static_cast<std::size_t>(num_steps + 1) * graph.num_nodes))
#endif
    for (int step = 0; step <= num_steps; ++step) {
        for (int node = 0; node < graph.num_nodes; ++node) {
            const double weight = graph.pi[node];
            const double value = state.rho[row_major(step, node, graph.num_nodes)];
            total += weight * value * value;
        }
    }
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for reduction(+:total) if(should_parallelize(static_cast<std::size_t>(num_steps) * graph.num_edges))
#endif
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            const double weight = graph.edge_weight[edge];
            const auto edge_index = row_major(step, edge, graph.num_edges);
            total += weight * state.m[edge_index] * state.m[edge_index];
            total += weight * state.vartheta[edge_index] * state.vartheta[edge_index];
            total += weight * state.rho_minus[edge_index] * state.rho_minus[edge_index];
            total += weight * state.rho_plus[edge_index] * state.rho_plus[edge_index];
        }
    }
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for reduction(+:total) if(should_parallelize(static_cast<std::size_t>(num_steps) * graph.num_nodes))
#endif
    for (int step = 0; step < num_steps; ++step) {
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for reduction(+:total) if(should_parallelize(static_cast<std::size_t>(num_steps) * graph.num_edges))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(static_cast<std::size_t>(num_steps) * graph.num_nodes))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(static_cast<std::size_t>(num_steps + 1) * graph.num_nodes))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(static_cast<std::size_t>(num_steps + 1) * graph.num_nodes))
#endif
    for (int step = 0; step <= num_steps; ++step) {
        std::copy(rho.begin(), rho.end(), state.rho.begin() + row_major(step, 0, graph.num_nodes));
    }
    init_split_state(graph, mean, state.rho, num_steps, state);
    return state;
}

double compute_action(const GraphData& graph, const State& state, int num_steps) {
    const double h = 1.0 / static_cast<double>(num_steps);
    double total = 0.0;
    int infeasible = 0;
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for reduction(+:total) reduction(|:infeasible) if(should_parallelize(static_cast<std::size_t>(num_steps) * graph.num_edges))
#endif
    for (int step = 0; step < num_steps; ++step) {
        for (int edge = 0; edge < graph.num_edges; ++edge) {
            const auto index = row_major(step, edge, graph.num_edges);
            const double vartheta = state.vartheta[index];
            const double m = state.m[index];
            double safe = 0.0;
            if (vartheta > 0.0) {
                safe = (m * m) / vartheta;
            } else if (std::abs(m) > 1e-12) {
                infeasible = 1;
            }
            total += h * graph.edge_weight[edge] * safe;
        }
    }
    if (infeasible != 0) {
        return std::numeric_limits<double>::infinity();
    }
    return total;
}

}  // namespace graphot::core
