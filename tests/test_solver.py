from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import graphot.solver as solver_module
from graphot import (
    GraphSpec,
    LogMeanOps,
    OTConfig,
    OTProblem,
    TimeDiscretization,
    solve_harmonic_socp_warm_start,
    solve_ot,
)


def _two_node_problem(alpha: float, beta: float, *, num_steps: int = 28) -> OTProblem:
    graph = GraphSpec.from_undirected_weights(2, [0], [1], [1.0])
    rho_a = np.array([1.0 - alpha, 1.0 + alpha])
    rho_b = np.array([1.0 - beta, 1.0 + beta])
    return OTProblem(
        graph=graph,
        time=TimeDiscretization(num_steps),
        rho_a=rho_a,
        rho_b=rho_b,
        mean_ops=LogMeanOps(),
    )


def _reference_distance(alpha: float, beta: float, n: int = 4001) -> float:
    grid = np.linspace(alpha, beta, n)
    abs_grid = np.abs(grid)
    ratio = np.empty_like(grid)
    mask = abs_grid < 1e-10
    ratio[mask] = 1.0
    ratio[~mask] = np.arctanh(grid[~mask]) / grid[~mask]
    integrand = np.sqrt(ratio)
    return (1.0 / np.sqrt(2.0)) * np.trapezoid(integrand, grid)


def _cycle_problem(num_nodes: int, *, num_steps: int) -> OTProblem:
    u = list(range(num_nodes))
    v = [(i + 1) % num_nodes for i in range(num_nodes)]
    graph = GraphSpec.from_undirected_weights(num_nodes, u, v, [1.0] * num_nodes)
    rho_a = np.zeros(num_nodes)
    rho_b = np.zeros(num_nodes)
    rho_a[0] = num_nodes
    rho_b[1] = num_nodes
    return OTProblem(
        graph=graph,
        time=TimeDiscretization(num_steps),
        rho_a=rho_a,
        rho_b=rho_b,
        mean_ops=LogMeanOps(),
    )


def _directed_reversible_problem(*, num_steps: int) -> OTProblem:
    graph = GraphSpec.from_directed_rates(
        3,
        src=[0, 1, 1, 2],
        dst=[1, 0, 2, 1],
        q=[2.0, 1.0, 1.0, 2.0],
    )
    mass_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    mass_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return OTProblem(
        graph=graph,
        time=TimeDiscretization(num_steps),
        rho_a=mass_a / np.asarray(graph.pi),
        rho_b=mass_b / np.asarray(graph.pi),
        mean_ops=LogMeanOps(),
    )


def _extension_info_in_subprocess(
    *,
    graphot_num_threads: str | None,
    omp_num_threads: str | None,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if graphot_num_threads is None:
        env.pop("GRAPHOT_NUM_THREADS", None)
    else:
        env["GRAPHOT_NUM_THREADS"] = graphot_num_threads
    if omp_num_threads is None:
        env.pop("OMP_NUM_THREADS", None)
    else:
        env["OMP_NUM_THREADS"] = omp_num_threads
    script = """
import json
from graphot import _core
print(json.dumps(_core.extension_info(), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip())


@lru_cache(maxsize=1)
def _mosek_runtime_available() -> bool:
    try:
        from mosek.fusion import Domain, Expr, Model, ObjectiveSense
    except ImportError:
        return False

    try:
        with Model("mosek_test_license") as model:
            x = model.variable("x", 1, Domain.greaterThan(0.0))
            model.constraint(Expr.sub(x, 1.0), Domain.equalsTo(0.0))
            model.objective(ObjectiveSense.Minimize, Expr.sum(x))
            model.solve()
            level = np.asarray(x.level(), dtype=np.float64)
    except Exception:
        return False

    return level.shape == (1,) and np.isfinite(level[0]) and abs(float(level[0]) - 1.0) <= 1e-9


_MOSEK_SKIP_REASON = "MOSEK Fusion runtime or license unavailable"


@pytest.fixture(scope="module")
def debug_trace_solution():
    problem = _two_node_problem(-0.2, 0.2, num_steps=12)
    return solve_ot(
        problem,
        OTConfig(max_iters=20, check_every=5, cg_max_iters=64, record_debug_trace=True),
    )


def test_config_defaults_to_paper_mode() -> None:
    assert OTConfig().numerics_mode == "paper"
    assert OTConfig().verbose is False


def test_config_accepts_explicit_paper_mode() -> None:
    assert OTConfig(numerics_mode="paper").numerics_mode == "paper"


def test_config_accepts_block_jacobi_preconditioner() -> None:
    assert OTConfig(cg_preconditioner="block_jacobi").cg_preconditioner == "block_jacobi"


def test_config_accepts_harmonic_socp_warm_start() -> None:
    assert OTConfig(warm_start="harmonic_socp").warm_start == "harmonic_socp"


def test_config_rejects_legacy_mode_with_migration_message() -> None:
    with pytest.raises(ValueError, match="legacy mode has been removed; use numerics_mode='paper'"):
        OTConfig(numerics_mode="legacy")


def test_config_rejects_unknown_warm_start() -> None:
    with pytest.raises(
        ValueError,
        match="warm_start must be 'linear_path', 'zero', or 'harmonic_socp'",
    ):
        OTConfig(warm_start="not_a_mode")


def test_openmp_defaults_to_all_available_threads_when_thread_envs_are_unset() -> None:
    info = _extension_info_in_subprocess(graphot_num_threads=None, omp_num_threads=None)
    if not bool(info["compiled_with_openmp"]):
        assert info["configured_openmp_threads"] == 1
        assert info["openmp_max_threads"] == 1
        assert info["openmp_thread_source"] == "serial_build"
        return

    assert info["default_openmp_threads"] == info["openmp_num_procs"]
    assert info["configured_openmp_threads"] == info["openmp_num_procs"]
    assert info["openmp_max_threads"] == info["openmp_num_procs"]
    assert info["openmp_thread_source"] == "all_available_threads"


@pytest.mark.parametrize(
    ("graphot_num_threads", "omp_num_threads", "expected_threads", "expected_source"),
    [
        ("7", None, 7, "GRAPHOT_NUM_THREADS"),
        (None, "9", 9, "OMP_NUM_THREADS"),
    ],
)
def test_openmp_respects_thread_env_overrides(
    graphot_num_threads: str | None,
    omp_num_threads: str | None,
    expected_threads: int,
    expected_source: str,
) -> None:
    info = _extension_info_in_subprocess(
        graphot_num_threads=graphot_num_threads,
        omp_num_threads=omp_num_threads,
    )
    if not bool(info["compiled_with_openmp"]):
        assert info["configured_openmp_threads"] == 1
        assert info["openmp_max_threads"] == 1
        assert info["openmp_thread_source"] == "serial_build"
        return

    assert info["configured_openmp_threads"] == expected_threads
    assert info["openmp_max_threads"] == expected_threads
    assert info["openmp_thread_source"] == expected_source


def test_solver_zero_distance_for_identical_endpoints() -> None:
    problem = _two_node_problem(0.2, 0.2, num_steps=12)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=40,
            check_every=5,
            cg_max_iters=64,
        ),
    )
    assert float(solution.distance) < 1e-6
    assert solution.converged
    assert solution.iterations_used == 1


def test_solver_accepts_explicit_initial_state() -> None:
    seed_problem = _two_node_problem(-0.2, 0.2, num_steps=16)
    target_problem = _two_node_problem(-0.1, 0.3, num_steps=16)
    config = OTConfig(max_iters=240, check_every=10, tol=1e-8, cg_max_iters=96)

    seed_solution = solve_ot(seed_problem, config)
    direct_solution = solve_ot(target_problem, config)
    warm_started_solution = solve_ot(
        target_problem,
        config,
        initial_state=seed_solution.state,
    )

    assert direct_solution.converged
    assert warm_started_solution.converged
    assert np.isfinite(warm_started_solution.action)
    assert abs(float(direct_solution.distance) - float(warm_started_solution.distance)) < 1e-8


def test_explicit_initial_state_records_iteration_zero_debug_trace() -> None:
    seed_problem = _two_node_problem(-0.2, 0.2, num_steps=16)
    target_problem = _two_node_problem(-0.1, 0.3, num_steps=16)
    seed_solution = solve_ot(
        seed_problem,
        OTConfig(max_iters=240, check_every=10, tol=1e-8, cg_max_iters=96),
    )
    initial_action = float(solver_module.compute_action(target_problem, seed_solution.state))

    solution = solve_ot(
        target_problem,
        OTConfig(
            max_iters=1,
            check_every=1,
            residual_tol=1e-12,
            feasibility_tol=1e-12,
            cg_max_iters=96,
            record_debug_trace=True,
        ),
        initial_state=seed_solution.state,
    )

    trace = solution.debug_trace
    assert trace is not None
    valid_iterations = np.asarray(trace.iterations)[: trace.num_records]
    valid_action = np.asarray(trace.action)[: trace.num_records]
    valid_ceh_cg_iters = np.asarray(trace.ceh_cg_iters)[: trace.num_records]

    assert solution.iterations_used == 1
    assert trace.num_records == 2
    assert valid_iterations.tolist() == [0, 1]
    assert valid_action[0] == pytest.approx(initial_action)
    assert np.isfinite(valid_action[0])
    assert int(valid_ceh_cg_iters[0]) > 0


def test_explicit_initial_state_skips_phi_bootstrap_when_cg_warm_start_disabled() -> None:
    seed_problem = _two_node_problem(-0.2, 0.2, num_steps=16)
    target_problem = _two_node_problem(-0.1, 0.3, num_steps=16)
    seed_solution = solve_ot(
        seed_problem,
        OTConfig(max_iters=240, check_every=10, tol=1e-8, cg_max_iters=96),
    )
    initial_action = float(solver_module.compute_action(target_problem, seed_solution.state))

    solution = solve_ot(
        target_problem,
        OTConfig(
            max_iters=1,
            check_every=1,
            residual_tol=1e-12,
            feasibility_tol=1e-12,
            cg_max_iters=96,
            cg_warm_start=False,
            record_debug_trace=True,
        ),
        initial_state=seed_solution.state,
    )

    trace = solution.debug_trace
    assert trace is not None
    valid_iterations = np.asarray(trace.iterations)[: trace.num_records]
    valid_action = np.asarray(trace.action)[: trace.num_records]
    valid_ceh_cg_iters = np.asarray(trace.ceh_cg_iters)[: trace.num_records]
    valid_ceh_cg_residual = np.asarray(trace.ceh_cg_residual)[: trace.num_records]

    assert solution.iterations_used == 1
    assert trace.num_records == 2
    assert valid_iterations.tolist() == [0, 1]
    assert valid_action[0] == pytest.approx(initial_action)
    assert int(valid_ceh_cg_iters[0]) == 0
    assert float(valid_ceh_cg_residual[0]) == 0.0


@pytest.mark.skipif(not _mosek_runtime_available(), reason=_MOSEK_SKIP_REASON)
def test_harmonic_socp_warm_start_returns_valid_state() -> None:
    problem = _two_node_problem(-0.2, 0.2, num_steps=10)
    result = solve_harmonic_socp_warm_start(problem)

    assert result.state.rho.shape == (problem.time.num_steps + 1, problem.graph.num_nodes)
    assert result.state.m.shape == (problem.time.num_steps, problem.graph.num_edges)
    assert result.state.rho_bar.shape == (problem.time.num_steps, problem.graph.num_nodes)
    assert np.isfinite(result.objective)
    assert result.continuity_residual < 1e-7
    assert result.endpoint_residual < 1e-9
    assert result.min_rho > -1e-8
    assert result.min_rho_bar > -1e-8
    assert "Feasible" in result.solve_status or "Optimal" in result.solve_status


@pytest.mark.skipif(not _mosek_runtime_available(), reason=_MOSEK_SKIP_REASON)
def test_harmonic_socp_warm_start_can_export_npz(tmp_path: Path) -> None:
    problem = _two_node_problem(-0.1, 0.3, num_steps=8)
    export_path = tmp_path / "harmonic_warm_start"
    result = solve_harmonic_socp_warm_start(problem, export_path=export_path)

    assert result.export_path is not None
    saved_path = Path(result.export_path)
    assert saved_path.exists()

    with np.load(saved_path) as payload:
        assert {
            "rho",
            "m",
            "vartheta",
            "rho_minus",
            "rho_plus",
            "rho_bar",
            "q_node",
            "objective",
            "continuity_residual",
            "endpoint_residual",
            "min_rho",
            "min_rho_bar",
            "solve_status",
        }.issubset(payload.files)


@pytest.mark.skipif(not _mosek_runtime_available(), reason=_MOSEK_SKIP_REASON)
def test_solver_supports_harmonic_socp_warm_start() -> None:
    problem = _two_node_problem(-0.2, 0.2, num_steps=16)
    direct_config = OTConfig(max_iters=240, check_every=10, tol=1e-8, cg_max_iters=96)
    harmonic_config = OTConfig(
        max_iters=240,
        check_every=10,
        tol=1e-8,
        cg_max_iters=96,
        warm_start="harmonic_socp",
    )

    direct_solution = solve_ot(problem, direct_config)
    harmonic_solution = solve_ot(problem, harmonic_config)

    assert direct_solution.converged
    assert harmonic_solution.converged
    assert np.isfinite(harmonic_solution.action)
    assert abs(float(direct_solution.distance) - float(harmonic_solution.distance)) < 1e-8


@pytest.mark.skipif(not _mosek_runtime_available(), reason=_MOSEK_SKIP_REASON)
def test_explicit_initial_state_takes_precedence_over_harmonic_socp(monkeypatch) -> None:
    seed_problem = _two_node_problem(-0.2, 0.2, num_steps=12)
    target_problem = _two_node_problem(-0.1, 0.3, num_steps=12)
    seed_solution = solve_ot(seed_problem, OTConfig(max_iters=240, check_every=10, tol=1e-8))

    def _unexpected_harmonic_call(problem: OTProblem):  # pragma: no cover - defensive guard
        raise AssertionError(f"unexpected harmonic warm-start call for {problem}")

    monkeypatch.setattr(
        solver_module,
        "solve_harmonic_socp_warm_start",
        _unexpected_harmonic_call,
    )

    solution = solve_ot(
        target_problem,
        OTConfig(
            max_iters=240,
            check_every=10,
            tol=1e-8,
            cg_max_iters=96,
            warm_start="harmonic_socp",
        ),
        initial_state=seed_solution.state,
    )

    assert solution.converged


@pytest.mark.parametrize(
    ("rho_a", "rho_b", "message"),
    [
        (np.array([np.nan, np.nan]), np.array([1.0, 1.0]), "rho_a must be finite"),
        (np.array([1.0, 1.0]), np.array([np.inf, 0.0]), "rho_b must be finite"),
    ],
)
def test_solver_rejects_nonfinite_endpoint_densities(
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    message: str,
) -> None:
    graph = GraphSpec.from_undirected_weights(2, [0], [1], [1.0])
    problem = OTProblem(
        graph=graph,
        time=TimeDiscretization(12),
        rho_a=rho_a,
        rho_b=rho_b,
        mean_ops=LogMeanOps(),
    )
    with pytest.raises(ValueError, match=message):
        solve_ot(problem)


@pytest.mark.slow
def test_solver_is_symmetric_on_two_node_problem() -> None:
    forward = _two_node_problem(-0.4, 0.4, num_steps=24)
    backward = _two_node_problem(0.4, -0.4, num_steps=24)
    config = OTConfig(max_iters=240, check_every=10, tol=1e-7, cg_max_iters=96)
    dist_forward = float(solve_ot(forward, config).distance)
    dist_backward = float(solve_ot(backward, config).distance)
    assert abs(dist_forward - dist_backward) < 1e-6


@pytest.mark.slow
def test_solver_matches_two_node_reference_reasonably() -> None:
    alpha = -0.3
    beta = 0.3
    problem = _two_node_problem(alpha, beta, num_steps=28)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=240,
            check_every=10,
            tol=1e-7,
            cg_max_iters=96,
        ),
    )
    reference = _reference_distance(alpha, beta)
    rel_err = abs(float(solution.distance) - reference) / reference
    assert solution.converged
    assert rel_err < 1e-3
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8
    assert float(solution.diagnostics["max_constraint_residual"]) < 1e-8


@pytest.mark.slow
def test_three_cycle_routes_positive_mass_through_third_node() -> None:
    problem = _cycle_problem(3, num_steps=16)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=600,
            check_every=20,
            residual_tol=1e-7,
            feasibility_tol=1e-7,
            cg_max_iters=96,
        ),
    )
    midpoint = np.asarray(solution.state.rho)[problem.time.num_steps // 2]
    assert midpoint[2] > 1e-4
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8


@pytest.mark.slow
def test_four_cycle_keeps_long_path_mass_small() -> None:
    problem = _cycle_problem(4, num_steps=16)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=1200,
            check_every=20,
            residual_tol=1e-7,
            feasibility_tol=1e-7,
            cg_max_iters=96,
        ),
    )
    midpoint = np.asarray(solution.state.rho)[problem.time.num_steps // 2]
    assert midpoint[2] + midpoint[3] < 1e-2
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8


@pytest.mark.slow
def test_default_linear_warm_start_stays_stable_in_subprocess_on_linux() -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux-specific regression for the historical CE_h abort path")

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    script = """
import numpy as np
from graphot import GraphSpec, LogMeanOps, OTConfig, OTProblem, TimeDiscretization, solve_ot

graph = GraphSpec.from_undirected_weights(4, [0, 1, 2, 3], [1, 2, 3, 0], [1.0] * 4)
rho_a = np.array([4.0, 0.0, 0.0, 0.0], dtype=np.float64)
rho_b = np.array([0.0, 4.0, 0.0, 0.0], dtype=np.float64)
problem = OTProblem(
    graph=graph,
    time=TimeDiscretization(16),
    rho_a=rho_a,
    rho_b=rho_b,
    mean_ops=LogMeanOps(),
)
solution = solve_ot(
    problem,
    OTConfig(
        max_iters=1,
        check_every=1,
        residual_tol=1e-7,
        feasibility_tol=1e-7,
        cg_max_iters=96,
    ),
)
print(float(solution.action))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, (
        f"subprocess exited with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.slow
def test_directed_reversible_solver_preserves_constraints() -> None:
    problem = _directed_reversible_problem(num_steps=16)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=1200,
            check_every=20,
            residual_tol=1e-7,
            feasibility_tol=1e-7,
            cg_max_iters=128,
        ),
    )
    assert np.isfinite(float(solution.distance))
    assert np.isfinite(float(solution.action))
    assert float(solution.diagnostics["endpoint_residual"]) == 0.0
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8
    assert float(solution.diagnostics["max_constraint_residual"]) < 1e-8


@pytest.mark.slow
def test_solver_block_jacobi_runs_on_two_node_problem() -> None:
    alpha = -0.3
    beta = 0.3
    problem = _two_node_problem(alpha, beta, num_steps=20)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=240,
            check_every=10,
            tol=1e-7,
            cg_max_iters=64,
            cg_preconditioner="block_jacobi",
        ),
    )
    reference = _reference_distance(alpha, beta)
    rel_err = abs(float(solution.distance) - reference) / reference
    assert solution.converged
    assert rel_err < 5e-3
    assert float(solution.diagnostics["endpoint_residual"]) == 0.0
    assert float(solution.diagnostics["continuity_residual"]) < 1e-8
    assert float(solution.diagnostics["max_constraint_residual"]) < 1e-8


def test_solver_returns_debug_trace_when_enabled(debug_trace_solution) -> None:
    trace = debug_trace_solution.debug_trace
    assert trace is not None
    assert trace.num_records > 0
    lengths = {
        len(np.asarray(trace.iterations)),
        len(np.asarray(trace.action)),
        len(np.asarray(trace.continuity_residual)),
        len(np.asarray(trace.primal_delta)),
        len(np.asarray(trace.dual_delta)),
        len(np.asarray(trace.k_violation)),
        len(np.asarray(trace.endpoint_residual)),
        len(np.asarray(trace.max_constraint_residual)),
        len(np.asarray(trace.ceh_cg_residual)),
        len(np.asarray(trace.ceh_cg_iters)),
        len(np.asarray(trace.min_vartheta)),
    }
    assert len(lengths) == 1
    valid_iterations = np.asarray(trace.iterations)[: trace.num_records]
    assert np.all(np.diff(valid_iterations) > 0)
    action = np.asarray(trace.action)[: trace.num_records]
    continuity = np.asarray(trace.continuity_residual)[: trace.num_records]
    assert action.shape == continuity.shape
    assert np.all(np.isfinite(continuity))


def test_solver_omits_debug_trace_when_disabled() -> None:
    problem = _two_node_problem(-0.2, -0.2, num_steps=12)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=20,
            check_every=5,
            cg_max_iters=64,
            record_debug_trace=False,
        ),
    )
    assert solution.debug_trace is None


def test_solver_verbose_logs_checkpoint_iterations(capfd: pytest.CaptureFixture[str]) -> None:
    problem = _two_node_problem(-0.2, 0.2, num_steps=12)
    solution = solve_ot(
        problem,
        OTConfig(
            max_iters=20,
            check_every=5,
            cg_max_iters=64,
            verbose=True,
        ),
    )
    assert solution.iterations_used > 0
    captured = capfd.readouterr()
    matches = re.findall(r"(\d+)/20", captured.err)
    assert matches
    assert all(int(step) % 5 == 0 for step in matches)
    assert "graphot [" in captured.err
