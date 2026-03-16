#pragma once

#include <optional>
#include <vector>

#include "graphot/ceh_linear.hpp"
#include "graphot/mean_ops.hpp"

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
);

void prox_i_star_javg(
    const std::vector<double>& rho,
    const std::vector<double>& rho_bar,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    int num_nodes,
    std::vector<double>& rho_out,
    std::vector<double>& rho_bar_out
);

void project_jeq(
    const std::vector<double>& rho_bar,
    const std::vector<double>& q_node,
    std::vector<double>& rho_bar_pr,
    std::vector<double>& q_node_pr
);

void project_jpm(
    const GraphData& graph,
    const std::vector<double>& q_node,
    const std::vector<double>& rho_minus,
    const std::vector<double>& rho_plus,
    int num_steps,
    std::vector<double>& q_node_pr,
    std::vector<double>& rho_minus_pr,
    std::vector<double>& rho_plus_pr
);

void prox_i_star_jpm(
    const GraphData& graph,
    const std::vector<double>& q_node,
    const std::vector<double>& rho_minus,
    const std::vector<double>& rho_plus,
    int num_steps,
    std::vector<double>& q_node_out,
    std::vector<double>& rho_minus_out,
    std::vector<double>& rho_plus_out
);

void prox_a_star(
    const std::vector<double>& vartheta,
    const std::vector<double>& m,
    int newton_iters,
    std::vector<double>& vartheta_out,
    std::vector<double>& m_out
);

State prox_f_star(
    const State& state,
    const GraphData& graph,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    int newton_iters
);

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
);

}  // namespace graphot::core
