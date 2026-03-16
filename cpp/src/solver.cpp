#include "graphot/solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <limits>
#include <optional>
#include <vector>

#include "graphot/prox_operators.hpp"
#include "graphot/runtime.hpp"
#include "graphot/state_ops.hpp"

#if defined(GRAPHOT_USE_OPENMP)
#include <omp.h>
#endif

namespace graphot::core {
namespace {

constexpr int kDefaultDualBootstrapSteps = 8;

int dual_bootstrap_steps() {
    const char* raw = std::getenv("GRAPHOT_DUAL_BOOTSTRAP_STEPS");
    if (raw == nullptr || raw[0] == '\0') {
        return kDefaultDualBootstrapSteps;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return kDefaultDualBootstrapSteps;
    }
    return static_cast<int>(parsed);
}

bool vector_all_finite(const std::vector<double>& value) {
    return std::all_of(value.begin(), value.end(), [](double x) {
        return std::isfinite(x);
    });
}

bool state_all_finite(const State& state) {
    return vector_all_finite(state.rho)
        && vector_all_finite(state.m)
        && vector_all_finite(state.vartheta)
        && vector_all_finite(state.rho_minus)
        && vector_all_finite(state.rho_plus)
        && vector_all_finite(state.rho_bar)
        && vector_all_finite(state.q_node);
}

void log_solver_progress(
    const SolverConfig& config,
    int iteration,
    bool converged,
    const Diagnostics& diagnostics
) {
    if (!config.verbose) {
        return;
    }
    const int clamped_iteration = std::min(iteration, config.max_iters);
    const double fraction = static_cast<double>(clamped_iteration) / static_cast<double>(config.max_iters);
    constexpr int kBarWidth = 28;
    const int filled = static_cast<int>(std::round(fraction * static_cast<double>(kBarWidth)));

    std::ostringstream line;
    line << "\rgraphot [";
    for (int idx = 0; idx < kBarWidth; ++idx) {
        line << (idx < filled ? '=' : ' ');
    }
    line << "] "
         << std::setw(3) << static_cast<int>(std::round(100.0 * fraction)) << "% "
         << clamped_iteration << "/" << config.max_iters
         << " primal=" << std::scientific << std::setprecision(2) << diagnostics.primal_delta
         << " dual=" << diagnostics.dual_delta
         << " feas=" << diagnostics.max_constraint_residual;
    if (converged) {
        line << " converged";
    }
    if (clamped_iteration == config.max_iters && !converged) {
        line << " max_iters";
    }
    std::cerr << line.str();
    if (converged || clamped_iteration == config.max_iters) {
        std::cerr << '\n';
    }
    std::cerr.flush();
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if(should_parallelize(dual_m_diff.size()))
#endif
    for (std::size_t idx = 0; idx < dual_m_diff.size(); ++idx) {
        dual_m_diff[idx] -= prev_dual.m[idx];
        dual_vartheta_diff[idx] -= prev_dual.vartheta[idx];
    }
    const double dual_delta = dual_pair_norm_weighted(dual_m_diff, dual_vartheta_diff, graph, num_steps)
        / (1.0 + dual_pair_norm_weighted(prev_dual.m, prev_dual.vartheta, graph, num_steps));

    std::vector<double> continuity;
    continuity_residual(graph, primal.rho, primal.m, rho_a, rho_b, num_steps, continuity);
    double continuity_max = 0.0;
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for reduction(max:continuity_max) if(should_parallelize(continuity.size()))
#endif
    for (std::size_t idx = 0; idx < continuity.size(); ++idx) {
        continuity_max = std::max(continuity_max, std::abs(continuity[idx]));
    }

    double k_violation = 0.0;
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for reduction(max:k_violation) if(should_parallelize(primal.vartheta.size()))
#endif
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
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for reduction(max:endpoint_residual) if(should_parallelize(static_cast<std::size_t>(graph.num_nodes)))
#endif
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

void append_trace_record(
    DebugTrace& trace,
    int iteration,
    const State& primal,
    const Diagnostics& diagnostics,
    int num_steps,
    const GraphData& graph
) {
    const int slot = trace.num_records;
    trace.iterations[slot] = static_cast<int32_t>(iteration);
    trace.action[slot] = compute_action(graph, primal, num_steps);
    trace.continuity_residual[slot] = diagnostics.continuity_residual;
    trace.primal_delta[slot] = diagnostics.primal_delta;
    trace.dual_delta[slot] = diagnostics.dual_delta;
    trace.k_violation[slot] = diagnostics.k_violation;
    trace.endpoint_residual[slot] = diagnostics.endpoint_residual;
    trace.max_constraint_residual[slot] = diagnostics.max_constraint_residual;
    trace.ceh_cg_residual[slot] = diagnostics.ceh_cg_residual;
    trace.ceh_cg_iters[slot] = static_cast<int32_t>(diagnostics.ceh_cg_iters);
    trace.min_vartheta[slot] = *std::min_element(primal.vartheta.begin(), primal.vartheta.end());
    ++trace.num_records;
}

void bootstrap_dual_state(
    const State& primal,
    const GraphData& graph,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    const SolverConfig& config,
    State& dual
) {
    State candidate = zero_state_like(primal);
    const int bootstrap_steps = dual_bootstrap_steps();
    for (int step = 0; step < bootstrap_steps; ++step) {
        State dual_trial = state_add_scaled(candidate, primal, config.sigma);
        State dual_next = prox_f_star(
            dual_trial,
            graph,
            rho_a,
            rho_b,
            num_steps,
            config.newton_iters
        );
        if (!state_all_finite(dual_next)) {
            dual = zero_state_like(primal);
            return;
        }
        candidate = std::move(dual_next);
    }
    dual = std::move(candidate);
}

void bootstrap_phi_cache(
    const State& primal,
    const GraphData& graph,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    const SolverConfig& config,
    std::vector<double>& phi_cache,
    CehStats& stats
) {
    stats = {0.0, 0};
    if (!config.cg_warm_start) {
        return;
    }

    std::vector<double> rho_pr;
    std::vector<double> m_pr;
    std::vector<double> phi;
    CehStats phi_stats{};
    project_ceh(
        graph,
        primal.rho,
        primal.m,
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
        phi_stats
    );

    if (!vector_all_finite(phi)) {
        return;
    }

    phi_cache = std::move(phi);
    stats = phi_stats;
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

}  // namespace

SolveResult run_solver(
    const GraphData& graph,
    const LogMeanOpsCpp& mean,
    const std::vector<double>& rho_a,
    const std::vector<double>& rho_b,
    int num_steps,
    const SolverConfig& config,
    const std::optional<State>& initial_state
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
            trace.k_violation = {0.0};
            trace.endpoint_residual = {0.0};
            trace.max_constraint_residual = {0.0};
            trace.ceh_cg_residual = {0.0};
            trace.ceh_cg_iters = {0};
            trace.min_vartheta = {*std::min_element(result.state.vartheta.begin(), result.state.vartheta.end())};
            trace.num_records = 1;
            result.debug_trace = std::move(trace);
        }
        return result;
    }

    State primal = initial_state.has_value()
        ? *initial_state
        : (
            config.linear_warm_start
                ? build_linear_warm_start(graph, mean, rho_a, rho_b, num_steps, config)
                : initialize_state(graph, mean, rho_a, rho_b, num_steps)
        );
    State dual = zero_state_like(primal);
    State primal_bar = primal;
    std::vector<double> phi_cache(static_cast<std::size_t>(num_steps) * static_cast<std::size_t>(graph.num_nodes), 0.0);
    CehStats warm_start_ceh_stats{0.0, 0};

    if (initial_state.has_value()) {
        bootstrap_dual_state(primal, graph, rho_a, rho_b, num_steps, config, dual);
        bootstrap_phi_cache(
            primal,
            graph,
            rho_a,
            rho_b,
            num_steps,
            config,
            phi_cache,
            warm_start_ceh_stats
        );
    }
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
        const int trace_length = (config.max_iters + config.check_every - 1) / config.check_every
            + (initial_state.has_value() ? 1 : 0);
        trace.iterations.assign(trace_length, 0);
        trace.action.assign(trace_length, 0.0);
        trace.continuity_residual.assign(trace_length, 0.0);
        trace.primal_delta.assign(trace_length, 0.0);
        trace.dual_delta.assign(trace_length, 0.0);
        trace.k_violation.assign(trace_length, 0.0);
        trace.endpoint_residual.assign(trace_length, 0.0);
        trace.max_constraint_residual.assign(trace_length, 0.0);
        trace.ceh_cg_residual.assign(trace_length, 0.0);
        trace.ceh_cg_iters.assign(trace_length, 0);
        trace.min_vartheta.assign(trace_length, 0.0);
        trace.num_records = 0;

        if (initial_state.has_value()) {
            const Diagnostics warm_start_diagnostics = compute_diagnostics(
                primal,
                primal,
                dual,
                dual,
                graph,
                mean,
                rho_a,
                rho_b,
                num_steps,
                warm_start_ceh_stats
            );
            append_trace_record(trace, 0, primal, warm_start_diagnostics, num_steps, graph);
        }
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

        if (should_check) {
            log_solver_progress(config, iterations_used, converged, diagnostics);
        }

        if (config.record_debug_trace && should_check) {
            append_trace_record(trace, iterations_used, primal_next, diagnostics, num_steps, graph);
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

}  // namespace graphot::core
