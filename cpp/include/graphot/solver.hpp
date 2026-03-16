#pragma once

#include <optional>
#include <vector>

#include "graphot/mean_ops.hpp"

namespace graphot::core {

SolveResult run_solver(
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    const SolverConfig& config,
    const std::optional<State>& initial_state
);

}  // namespace graphot::core
