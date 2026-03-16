"""MOSEK harmonic-mean warm start construction for the log-mean solver."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .types import HarmonicWarmStartResult, OTProblem, OTState

Array = np.ndarray


def _validate_density(name: str, rho: Array, pi: Array, num_nodes: int) -> None:
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim != 1 or rho.shape[0] != num_nodes:
        raise ValueError(f"{name} must have shape ({num_nodes},)")
    if np.any(~np.isfinite(rho)):
        raise ValueError(f"{name} must be finite")
    if np.any(rho < -1e-12):
        raise ValueError(f"{name} must be nonnegative")
    mass = float(np.sum(np.asarray(pi, dtype=np.float64) * rho))
    if not np.isfinite(mass) or abs(mass - 1.0) > 1e-8:
        raise ValueError(f"{name} must satisfy sum(pi * rho) == 1")


def _require_mosek_fusion() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense, Var
    except ImportError as exc:
        raise RuntimeError(
            "harmonic MOSEK warm start requires mosek.fusion to be importable"
        ) from exc
    return Domain, Expr, Matrix, Model, ObjectiveSense, Var


def _build_continuity_matrix(problem: OTProblem) -> tuple[sparse.csr_matrix, Array, Array, Array]:
    graph = problem.graph
    edge_ids = np.arange(graph.num_edges, dtype=np.int32)
    rows = np.concatenate(
        [
            np.asarray(graph.src, dtype=np.int32),
            np.asarray(graph.dst, dtype=np.int32),
        ]
    )
    cols = np.concatenate([edge_ids, edge_ids])
    values = np.concatenate(
        [
            -0.5 * np.asarray(graph.q, dtype=np.float64),
            +0.5 * np.asarray(graph.q, dtype=np.float64),
        ]
    )
    continuity = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(graph.num_nodes, graph.num_edges),
        dtype=np.float64,
    ).tocsr()
    return continuity, rows, cols, values


def _harmonic_mean(s: Array, t: Array) -> Array:
    s_arr = np.asarray(s, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    denom = s_arr + t_arr
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(2.0 * s_arr * t_arr, denom, out=out, where=denom > 0.0)
    return out


def _build_harmonic_state(rho: Array, rho_bar: Array, m: Array, src: Array, dst: Array) -> OTState:
    q_node = np.asarray(rho_bar, dtype=np.float64).copy()
    rho_minus = q_node[:, np.asarray(src, dtype=np.int32)]
    rho_plus = q_node[:, np.asarray(dst, dtype=np.int32)]
    vartheta = _harmonic_mean(rho_minus, rho_plus)
    return OTState(
        rho=np.asarray(rho, dtype=np.float64),
        m=np.asarray(m, dtype=np.float64),
        vartheta=vartheta,
        rho_minus=rho_minus,
        rho_plus=rho_plus,
        rho_bar=q_node,
        q_node=q_node.copy(),
    )


def _build_trivial_result(problem: OTProblem) -> HarmonicWarmStartResult:
    graph = problem.graph
    num_steps = problem.time.num_steps
    rho = np.repeat(np.asarray(problem.rho_a, dtype=np.float64)[None, :], num_steps + 1, axis=0)
    rho_bar = rho[:-1].copy()
    state = _build_harmonic_state(
        rho,
        rho_bar,
        np.zeros((num_steps, graph.num_edges)),
        graph.src,
        graph.dst,
    )
    return HarmonicWarmStartResult(
        state=state,
        objective=0.0,
        continuity_residual=0.0,
        endpoint_residual=0.0,
        min_rho=float(np.min(state.rho)),
        min_rho_bar=float(np.min(state.rho_bar)),
        solve_status="trivial_identical_endpoints",
        export_path=None,
    )


def _extract_level(variable: Any, shape: tuple[int, ...]) -> Array:
    return np.asarray(variable.level(), dtype=np.float64).reshape(shape)


def _diagnostics(problem: OTProblem, state: OTState) -> tuple[float, float, float]:
    continuity, _, _, _ = _build_continuity_matrix(problem)
    rho = np.asarray(state.rho, dtype=np.float64)
    m = np.asarray(state.m, dtype=np.float64)
    h = problem.time.h
    continuity_residual = (rho[1:, :] - rho[:-1, :]) / h + np.asarray(m @ continuity.transpose())
    endpoint_residual = max(
        float(np.max(np.abs(rho[0, :] - np.asarray(problem.rho_a, dtype=np.float64)))),
        float(np.max(np.abs(rho[-1, :] - np.asarray(problem.rho_b, dtype=np.float64)))),
    )
    edge_density = np.asarray(problem.graph.pi, dtype=np.float64)[
        np.asarray(problem.graph.src, dtype=np.int32)
    ][None, :]
    action = 0.5 * h * float(
        np.sum(
            np.asarray(problem.graph.q, dtype=np.float64)[None, :]
            * edge_density
            * np.where(
                state.vartheta > 0.0,
                (state.m * state.m) / state.vartheta,
                np.where(np.abs(state.m) <= 1e-12, 0.0, np.inf),
            )
        )
    )
    return float(np.max(np.abs(continuity_residual))), endpoint_residual, action


def _normalize_export_path(export_path: str | PathLike[str] | None) -> Path | None:
    if export_path is None:
        return None
    path = Path(export_path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_result(path: Path, result: HarmonicWarmStartResult) -> None:
    np.savez(
        path,
        rho=np.asarray(result.state.rho, dtype=np.float64),
        m=np.asarray(result.state.m, dtype=np.float64),
        vartheta=np.asarray(result.state.vartheta, dtype=np.float64),
        rho_minus=np.asarray(result.state.rho_minus, dtype=np.float64),
        rho_plus=np.asarray(result.state.rho_plus, dtype=np.float64),
        rho_bar=np.asarray(result.state.rho_bar, dtype=np.float64),
        q_node=np.asarray(result.state.q_node, dtype=np.float64),
        objective=np.asarray(result.objective, dtype=np.float64),
        continuity_residual=np.asarray(result.continuity_residual, dtype=np.float64),
        endpoint_residual=np.asarray(result.endpoint_residual, dtype=np.float64),
        min_rho=np.asarray(result.min_rho, dtype=np.float64),
        min_rho_bar=np.asarray(result.min_rho_bar, dtype=np.float64),
        solve_status=np.asarray(result.solve_status),
    )


def solve_harmonic_socp_warm_start(
    problem: OTProblem,
    *,
    export_path: str | PathLike[str] | None = None,
) -> HarmonicWarmStartResult:
    """Solve the harmonic-mean SOCP warm start for one OT problem."""

    graph = problem.graph
    num_nodes = int(graph.num_nodes)
    num_edges = int(graph.num_edges)
    num_steps = int(problem.time.num_steps)
    h = problem.time.h
    rho_a = np.asarray(problem.rho_a, dtype=np.float64)
    rho_b = np.asarray(problem.rho_b, dtype=np.float64)

    _validate_density("rho_a", rho_a, graph.pi, num_nodes)
    _validate_density("rho_b", rho_b, graph.pi, num_nodes)

    if float(np.max(np.abs(rho_a - rho_b))) <= 1e-12:
        result = _build_trivial_result(problem)
        normalized_path = _normalize_export_path(export_path)
        if normalized_path is not None:
            _save_result(normalized_path, result)
            return HarmonicWarmStartResult(
                state=result.state,
                objective=result.objective,
                continuity_residual=result.continuity_residual,
                endpoint_residual=result.endpoint_residual,
                min_rho=result.min_rho,
                min_rho_bar=result.min_rho_bar,
                solve_status=result.solve_status,
                export_path=str(normalized_path),
            )
        return result

    Domain, Expr, Matrix, Model, ObjectiveSense, Var = _require_mosek_fusion()
    continuity, rows, cols, values = _build_continuity_matrix(problem)
    continuity_matrix = Matrix.sparse(
        num_nodes,
        num_edges,
        rows.tolist(),
        cols.tolist(),
        values.tolist(),
    )
    edge_weight = np.asarray(graph.q, dtype=np.float64) * np.asarray(graph.pi, dtype=np.float64)[
        np.asarray(graph.src, dtype=np.int32)
    ]
    objective_weight = np.tile(edge_weight, num_steps)

    try:
        with Model("harmonic_ot_socp") as model:
            rho = model.variable("rho", [num_steps + 1, num_nodes], Domain.greaterThan(0.0))
            rho_bar = model.variable("rho_bar", [num_steps, num_nodes], Domain.greaterThan(0.0))
            m = model.variable("m", [num_steps, num_edges], Domain.unbounded())
            u = model.variable("u", [num_steps, num_edges], Domain.greaterThan(0.0))
            v = model.variable("v", [num_steps, num_edges], Domain.greaterThan(0.0))

            model.constraint(
                "rho_a",
                rho.slice([0, 0], [1, num_nodes]),
                Domain.equalsTo(rho_a.reshape(1, num_nodes)),
            )
            model.constraint(
                "rho_b",
                rho.slice([num_steps, 0], [num_steps + 1, num_nodes]),
                Domain.equalsTo(rho_b.reshape(1, num_nodes)),
            )

            for step in range(num_steps):
                rho_curr = rho.slice([step, 0], [step + 1, num_nodes])
                rho_next = rho.slice([step + 1, 0], [step + 2, num_nodes])
                rho_bar_step = rho_bar.slice([step, 0], [step + 1, num_nodes])
                m_step = m.slice([step, 0], [step + 1, num_edges])

                model.constraint(
                    Expr.sub(Expr.mul(2.0, rho_bar_step), Expr.add(rho_curr, rho_next)),
                    Domain.equalsTo(0.0),
                )
                model.constraint(
                    Expr.add(
                        Expr.mul(1.0 / h, Expr.sub(rho_next, rho_curr)),
                        Expr.mul(m_step, continuity_matrix.transpose()),
                    ),
                    Domain.equalsTo(0.0),
                )

                for edge in range(num_edges):
                    src_node = int(graph.src[edge])
                    dst_node = int(graph.dst[edge])
                    model.constraint(
                        Var.vstack(
                            rho_bar.index(step, src_node),
                            u.index(step, edge),
                            m.index(step, edge),
                        ),
                        Domain.inRotatedQCone(),
                    )
                    model.constraint(
                        Var.vstack(
                            rho_bar.index(step, dst_node),
                            v.index(step, edge),
                            m.index(step, edge),
                        ),
                        Domain.inRotatedQCone(),
                    )

            model.objective(
                "harmonic_objective",
                ObjectiveSense.Minimize,
                Expr.mul(0.5 * h, Expr.dot(objective_weight, Expr.flatten(Expr.add(u, v)))),
            )
            model.solve()

            rho_sol = _extract_level(rho, (num_steps + 1, num_nodes))
            rho_bar_sol = _extract_level(rho_bar, (num_steps, num_nodes))
            m_sol = _extract_level(m, (num_steps, num_edges))
            solve_status = f"{model.getProblemStatus()} / {model.getPrimalSolutionStatus()}"
    except Exception as exc:
        raise RuntimeError(
            "MOSEK failed while solving the harmonic warm start; ensure mosek.fusion imports "
            "cleanly and that a valid MOSEK license is configured"
        ) from exc

    state = _build_harmonic_state(rho_sol, rho_bar_sol, m_sol, graph.src, graph.dst)
    continuity_residual, endpoint_residual, action = _diagnostics(problem, state)
    result = HarmonicWarmStartResult(
        state=state,
        objective=action,
        continuity_residual=continuity_residual,
        endpoint_residual=endpoint_residual,
        min_rho=float(np.min(state.rho)),
        min_rho_bar=float(np.min(state.rho_bar)),
        solve_status=solve_status,
        export_path=None,
    )

    normalized_path = _normalize_export_path(export_path)
    if normalized_path is not None:
        _save_result(normalized_path, result)
        return HarmonicWarmStartResult(
            state=result.state,
            objective=result.objective,
            continuity_residual=result.continuity_residual,
            endpoint_residual=result.endpoint_residual,
            min_rho=result.min_rho,
            min_rho_bar=result.min_rho_bar,
            solve_status=result.solve_status,
            export_path=str(normalized_path),
        )
    return result
