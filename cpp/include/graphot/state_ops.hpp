#pragma once

#include <vector>

#include "graphot/core_types.hpp"
#include "graphot/mean_ops.hpp"

namespace graphot::core {

State zero_state_like(const State& state);
State state_add_scaled(const State& left, const State& right, double scale);
State state_subtract(const State& left, const State& right);
double state_norm_weighted(const State& state, const GraphData& graph, int num_steps);
double dual_pair_norm_weighted(
    const std::vector<double>& m,
    const std::vector<double>& vartheta,
    const GraphData& graph,
    int num_steps
);
void init_split_state(
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho,
    int num_steps,
    State& state
);
State initialize_state(
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps
);
State build_trivial_state(
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho,
    int num_steps
);
double compute_action(const GraphData& graph, const State& state, int num_steps);

}  // namespace graphot::core
