#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "graphot/mean_ops.hpp"
#include "graphot/pybind_io.hpp"
#include "graphot/runtime.hpp"
#include "graphot/solver.hpp"

namespace py = pybind11;

namespace graphot::core {
namespace {

py::dict build_debug_trace_payload(const DebugTrace& trace) {
    py::dict trace_payload;
    trace_payload["iterations"] = make_array_from_vector(
        trace.iterations,
        {static_cast<py::ssize_t>(trace.iterations.size())}
    );
    trace_payload["action"] = make_array_from_vector(
        trace.action,
        {static_cast<py::ssize_t>(trace.action.size())}
    );
    trace_payload["continuity_residual"] = make_array_from_vector(
        trace.continuity_residual,
        {static_cast<py::ssize_t>(trace.continuity_residual.size())}
    );
    trace_payload["primal_delta"] = make_array_from_vector(
        trace.primal_delta,
        {static_cast<py::ssize_t>(trace.primal_delta.size())}
    );
    trace_payload["dual_delta"] = make_array_from_vector(
        trace.dual_delta,
        {static_cast<py::ssize_t>(trace.dual_delta.size())}
    );
    trace_payload["k_violation"] = make_array_from_vector(
        trace.k_violation,
        {static_cast<py::ssize_t>(trace.k_violation.size())}
    );
    trace_payload["endpoint_residual"] = make_array_from_vector(
        trace.endpoint_residual,
        {static_cast<py::ssize_t>(trace.endpoint_residual.size())}
    );
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
    return trace_payload;
}

}  // namespace

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
    bool verbose,
    double mean_eps_diag,
    double mean_xi_max,
    int mean_newton_iters,
    int mean_bisect_iters,
    py::object initial_state_payload
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
        verbose,
    };
    LogMeanOpsCpp mean(mean_config);
    std::vector<double> rho_a_vec(rho_a.data(), rho_a.data() + rho_a.size());
    std::vector<double> rho_b_vec(rho_b.data(), rho_b.data() + rho_b.size());
    const std::optional<State> initial_state =
        parse_initial_state(initial_state_payload, num_steps, num_nodes, num_edges);
    SolveResult result;
    {
        py::gil_scoped_release release;
        result = run_solver(
            graph,
            mean,
            rho_a_vec,
            rho_b_vec,
            num_steps,
            solver_config,
            initial_state
        );
    }

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
    payload["simd_level"] = std::string(simd_name());

    if (result.debug_trace.has_value()) {
        payload["debug_trace"] = build_debug_trace_payload(*result.debug_trace);
    } else {
        payload["debug_trace"] = py::none();
    }
    return payload;
}

py::dict extension_info() {
    py::dict info;
    info["simd_level"] = std::string(simd_name());
    info["compiled_with_x86_intrinsics"] =
#if defined(__x86_64__) || defined(_M_X64)
        true;
#else
        false;
#endif
    info["compiled_with_openmp"] =
#if defined(GRAPHOT_USE_OPENMP)
        true;
#else
        false;
#endif
    info["openmp_thread_source"] = std::string(configured_openmp_thread_source());
    info["configured_openmp_threads"] = configured_openmp_threads();
    info["default_openmp_threads"] = graphot_default_openmp_threads();
    info["openmp_max_threads"] = graphot_openmp_max_threads();
    info["openmp_num_procs"] = graphot_openmp_num_procs();
    return info;
}

}  // namespace graphot::core

PYBIND11_MODULE(_core, module) {
    module.doc() = "C++ CPU solver core for graphot";
    graphot::core::configure_openmp_threads();
    module.def("solve_ot_cpp", &graphot::core::solve_ot_cpp);
    module.def("extension_info", &graphot::core::extension_info);
}
