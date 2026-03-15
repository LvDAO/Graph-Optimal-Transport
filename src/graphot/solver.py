"""Top-level solve entrypoint and user-facing solver orchestration."""

from __future__ import annotations

import math

import numpy as np

from .means import LogMeanOps
from .types import OTConfig, OTDebugTrace, OTProblem, OTSolution, OTState

Array = np.ndarray

try:
    from . import _core as _core_backend
except ImportError:
    _core_backend = None


def _require_core_backend() -> None:
    """Ensure the compiled extension is available."""

    if _core_backend is None:
        raise RuntimeError(
            "the graphot._core extension is unavailable; reinstall graphot so the C++ backend builds correctly"
        )


def _validate_density(name: str, rho: Array, pi: Array, num_nodes: int) -> None:
    """Validate one endpoint density against the graph normalization rules."""

    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim != 1 or rho.shape[0] != num_nodes:
        raise ValueError(f"{name} must have shape ({num_nodes},)")
    if np.any(~np.isfinite(rho)):
        raise ValueError(f"{name} must be finite")
    if np.any(rho < -1e-12):
        raise ValueError(f"{name} must be nonnegative")
    mass = float(np.sum(np.asarray(pi, dtype=np.float64) * rho))
    if not math.isfinite(mass) or abs(mass - 1.0) > 1e-8:
        raise ValueError(f"{name} must satisfy sum(pi * rho) == 1")


def compute_action(problem: OTProblem, state: OTState) -> np.float64:
    """Compute the discrete transport action for a solved state."""

    h = problem.time.h
    graph = problem.graph
    m = np.asarray(state.m, dtype=np.float64)
    vartheta = np.asarray(state.vartheta, dtype=np.float64)
    weights = 0.5 * h * np.asarray(graph.q, dtype=np.float64)[None, :] * np.asarray(
        graph.pi,
        dtype=np.float64,
    )[np.asarray(graph.src, dtype=np.int32)][None, :]
    safe = np.where(
        vartheta > 0,
        (m * m) / vartheta,
        np.where(np.abs(m) <= 1e-12, 0.0, np.inf),
    )
    return np.float64(np.sum(weights * safe))


def _build_trivial_state(problem: OTProblem, rho: Array) -> OTState:
    """Build the exact zero-flow state used when the endpoints already match."""

    if not isinstance(problem.mean_ops, LogMeanOps):
        raise TypeError("only LogMeanOps is supported by the current runtime")

    graph = problem.graph
    num_steps = problem.time.num_steps
    rho_arr = np.asarray(rho, dtype=np.float64)
    rho_path = np.repeat(rho_arr[None, :], num_steps + 1, axis=0)
    rho_bar = 0.5 * (rho_path[:-1] + rho_path[1:])
    q_node = rho_bar.copy()
    rho_minus = q_node[:, np.asarray(graph.src, dtype=np.int32)]
    rho_plus = q_node[:, np.asarray(graph.dst, dtype=np.int32)]
    vartheta = np.asarray(problem.mean_ops.theta(rho_minus, rho_plus), dtype=np.float64)
    m = np.zeros((num_steps, graph.num_edges), dtype=np.float64)
    return OTState(
        rho=rho_path,
        m=m,
        vartheta=vartheta,
        rho_minus=rho_minus,
        rho_plus=rho_plus,
        rho_bar=rho_bar,
        q_node=q_node,
    )


def _wrap_cpp_result(payload: dict[str, object]) -> OTSolution:
    """Convert the raw extension payload into public Python dataclasses."""

    state = OTState(
        rho=np.asarray(payload["rho"], dtype=np.float64),
        m=np.asarray(payload["m"], dtype=np.float64),
        vartheta=np.asarray(payload["vartheta"], dtype=np.float64),
        rho_minus=np.asarray(payload["rho_minus"], dtype=np.float64),
        rho_plus=np.asarray(payload["rho_plus"], dtype=np.float64),
        rho_bar=np.asarray(payload["rho_bar"], dtype=np.float64),
        q_node=np.asarray(payload["q_node"], dtype=np.float64),
    )
    diagnostics = {
        "primal_delta": np.float64(payload["primal_delta"]),
        "dual_delta": np.float64(payload["dual_delta"]),
        "continuity_residual": np.float64(payload["continuity_residual"]),
        "k_violation": np.float64(payload["k_violation"]),
        "endpoint_residual": np.float64(payload["endpoint_residual"]),
        "max_constraint_residual": np.float64(payload["max_constraint_residual"]),
        "ceh_cg_residual": np.float64(payload["ceh_cg_residual"]),
        "ceh_cg_iters": np.int32(payload["ceh_cg_iters"]),
    }

    debug_trace = None
    trace_payload = payload["debug_trace"]
    if trace_payload is not None:
        debug_trace = OTDebugTrace(
            iterations=np.asarray(trace_payload["iterations"], dtype=np.int32),
            action=np.asarray(trace_payload["action"], dtype=np.float64),
            continuity_residual=np.asarray(trace_payload["continuity_residual"], dtype=np.float64),
            primal_delta=np.asarray(trace_payload["primal_delta"], dtype=np.float64),
            dual_delta=np.asarray(trace_payload["dual_delta"], dtype=np.float64),
            max_constraint_residual=np.asarray(
                trace_payload["max_constraint_residual"],
                dtype=np.float64,
            ),
            ceh_cg_residual=np.asarray(trace_payload["ceh_cg_residual"], dtype=np.float64),
            ceh_cg_iters=np.asarray(trace_payload["ceh_cg_iters"], dtype=np.int32),
            min_vartheta=np.asarray(trace_payload["min_vartheta"], dtype=np.float64),
            num_records=int(trace_payload["num_records"]),
        )

    return OTSolution(
        distance=np.float64(payload["distance"]),
        action=np.float64(payload["action"]),
        state=state,
        iterations_used=int(payload["iterations_used"]),
        converged=bool(payload["converged"]),
        diagnostics=diagnostics,
        debug_trace=debug_trace,
    )


def _solve_ot_cpp(problem: OTProblem, config: OTConfig) -> OTSolution:
    """Call the compiled backend and wrap the returned payload."""

    _require_core_backend()
    graph = problem.graph
    mean_ops = problem.mean_ops
    if not isinstance(mean_ops, LogMeanOps):
        raise TypeError("only LogMeanOps is supported by the current runtime")

    payload = _core_backend.solve_ot_cpp(
        int(graph.num_nodes),
        int(graph.num_edges),
        np.asarray(graph.src, dtype=np.int32),
        np.asarray(graph.dst, dtype=np.int32),
        np.asarray(graph.rev, dtype=np.int32),
        np.asarray(graph.q, dtype=np.float64),
        np.asarray(graph.pi, dtype=np.float64),
        np.asarray(graph.out_rate, dtype=np.float64),
        np.asarray(problem.rho_a, dtype=np.float64),
        np.asarray(problem.rho_b, dtype=np.float64),
        int(problem.time.num_steps),
        float(config.tau),
        float(config.sigma),
        float(config.relaxation),
        int(config.max_iters),
        int(config.check_every),
        float(config.residual_tol),
        float(config.feasibility_tol),
        int(config.newton_iters),
        int(config.cg_max_iters),
        float(config.cg_tol),
        bool(config.cg_warm_start),
        str(config.cg_preconditioner),
        str(config.warm_start),
        bool(config.record_debug_trace),
        float(mean_ops.eps_diag),
        float(mean_ops.xi_max),
        int(mean_ops.newton_iters),
        int(mean_ops.bisect_iters),
    )
    return _wrap_cpp_result(payload)


def solve_ot(problem: OTProblem, config: OTConfig = OTConfig()) -> OTSolution:
    """Solve the two-endpoint dynamic OT problem on a sparse reversible graph."""

    _validate_density("rho_a", problem.rho_a, problem.graph.pi, problem.graph.num_nodes)
    _validate_density("rho_b", problem.rho_b, problem.graph.pi, problem.graph.num_nodes)

    rho_a = np.asarray(problem.rho_a, dtype=np.float64)
    rho_b = np.asarray(problem.rho_b, dtype=np.float64)
    if float(np.max(np.abs(rho_a - rho_b))) <= 1e-12:
        state = _build_trivial_state(problem, rho_a)
        zero = np.float64(0.0)
        diagnostics = {
            "primal_delta": zero,
            "dual_delta": zero,
            "continuity_residual": zero,
            "k_violation": zero,
            "endpoint_residual": zero,
            "max_constraint_residual": zero,
            "ceh_cg_residual": zero,
            "ceh_cg_iters": np.int32(0),
        }
        debug_trace = None
        if config.record_debug_trace:
            debug_trace = OTDebugTrace(
                iterations=np.array([1], dtype=np.int32),
                action=np.array([0.0], dtype=np.float64),
                continuity_residual=np.array([0.0], dtype=np.float64),
                primal_delta=np.array([0.0], dtype=np.float64),
                dual_delta=np.array([0.0], dtype=np.float64),
                max_constraint_residual=np.array([0.0], dtype=np.float64),
                ceh_cg_residual=np.array([0.0], dtype=np.float64),
                ceh_cg_iters=np.array([0], dtype=np.int32),
                min_vartheta=np.array([np.min(state.vartheta)], dtype=np.float64),
                num_records=1,
            )
        return OTSolution(
            distance=zero,
            action=zero,
            state=state,
            iterations_used=1,
            converged=True,
            diagnostics=diagnostics,
            debug_trace=debug_trace,
        )

    return _solve_ot_cpp(problem, config)
