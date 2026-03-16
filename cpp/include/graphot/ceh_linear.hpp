#pragma once

#include <optional>
#include <vector>

#include "graphot/core_types.hpp"

namespace graphot::core {

void continuity_residual(
    const GraphData& graph,
    const std::vector<double>& rho,
    const std::vector<double>& m,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    std::vector<double>& out
);

void solve_tridiagonal(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs,
    std::vector<double>& out
);

void solve_tridiagonal_javg(const std::vector<double>& rhs, int num_steps, int num_nodes, std::vector<double>& out);

void apply_ceh_constraint_zero_boundary(
    const GraphData& graph,
    const std::vector<double>& drho_int,
    const std::vector<double>& dm,
    int num_steps,
    std::vector<double>& out
);

void apply_ceh_constraint_transpose_zero_boundary(
    const GraphData& graph,
    const std::vector<double>& phi,
    int num_steps,
    std::vector<double>& drho_int_adj,
    std::vector<double>& dm_adj
);

void apply_jacobi_preconditioner(
    const GraphData& graph,
    const std::vector<double>& value,
    int num_steps,
    std::vector<double>& out
);

void apply_block_jacobi_preconditioner(
    const GraphData& graph,
    const std::vector<double>& value,
    int num_steps,
    std::vector<double>& out
);

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
);

}  // namespace graphot::core
