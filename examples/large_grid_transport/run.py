from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _bootstrap_examples_dir() -> None:
    examples_dir = Path(__file__).resolve().parents[1]
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))


def _block_mass(node_mass: np.ndarray, side: int, *, rows: range, cols: range) -> float:
    total = 0.0
    for row in rows:
        for col in cols:
            total += float(node_mass[row * side + col])
    return total


def _float_tag(value: float) -> str:
    text = f"{value:.8g}"
    return text.replace("-", "m").replace(".", "p")


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in raw.split(","):
        stripped = token.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise ValueError("expected at least one comma-separated float value")
    return tuple(values)


def _normalize_continuation_epsilons(epsilons: tuple[float, ...]) -> tuple[float, ...]:
    normalized: list[float] = []
    previous = float("inf")
    for epsilon in epsilons:
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("continuation epsilons must lie in [0, 1)")
        if epsilon > previous + 1e-15:
            raise ValueError("continuation epsilons must be listed in nonincreasing order")
        normalized.append(float(epsilon))
        previous = float(epsilon)
    if normalized[-1] != 0.0:
        normalized.append(0.0)
    return tuple(normalized)


def _allocate_iteration_budget(total_iters: int, num_stages: int) -> tuple[int, ...]:
    if total_iters <= 0:
        raise ValueError("total_iters must be positive")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    if total_iters < num_stages:
        raise ValueError("total_iters must be at least the number of stages")

    base = total_iters // num_stages
    remainder = total_iters % num_stages
    return tuple(base + (1 if idx < remainder else 0) for idx in range(num_stages))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


@dataclass(frozen=True)
class SweepCase:
    name: str
    cg_preconditioner: str
    relaxation: float
    continuation_epsilons: tuple[float, ...]
    warm_start: str = "linear_path"


def _default_sweep_cases(continuation_epsilons: tuple[float, ...]) -> dict[str, SweepCase]:
    return {
        "large_jacobi": SweepCase(
            name="large_jacobi",
            cg_preconditioner="jacobi",
            relaxation=1.0,
            continuation_epsilons=(0.0,),
        ),
        "large_block_jacobi": SweepCase(
            name="large_block_jacobi",
            cg_preconditioner="block_jacobi",
            relaxation=1.0,
            continuation_epsilons=(0.0,),
        ),
        "large_block_jacobi_relaxed": SweepCase(
            name="large_block_jacobi_relaxed",
            cg_preconditioner="block_jacobi",
            relaxation=0.9,
            continuation_epsilons=(0.0,),
        ),
        "large_continuation_block_jacobi": SweepCase(
            name="large_continuation_block_jacobi",
            cg_preconditioner="block_jacobi",
            relaxation=1.0,
            continuation_epsilons=continuation_epsilons,
        ),
        "large_continuation_block_jacobi_relaxed": SweepCase(
            name="large_continuation_block_jacobi_relaxed",
            cg_preconditioner="block_jacobi",
            relaxation=0.9,
            continuation_epsilons=continuation_epsilons,
        ),
    }


def main() -> None:
    _bootstrap_examples_dir()

    from _common import (
        block_density,
        estimate_state_memory_bytes,
        grid_graph,
        grid_layout,
        probability_mass,
        regularize_density,
        save_debug_trace_npz,
        save_debug_trace_plot,
        save_edge_flow_heatmap,
        save_graph_snapshot_series,
        save_node_mass_heatmap,
        save_solution,
        solve_problem,
        summarize_solution,
    )
    from graphot import OTConfig

    parser = argparse.ArgumentParser(
        description=(
            "Run and compare demanding 32x32-style large-grid transport solves, including "
            "warm-started continuation over endpoint relaxations."
        )
    )
    parser.add_argument("--side", type=int, default=32, help="Side length of the square grid.")
    parser.add_argument("--steps", type=int, default=32, help="Number of time intervals.")
    parser.add_argument(
        "--blob-size",
        type=int,
        default=6,
        help="Side length of the source and target corner mass blocks.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=409600,
        help="Total iteration budget per sweep case. Continuation cases split this across stages.",
    )
    parser.add_argument(
        "--check-every",
        type=int,
        default=256,
        help="Checkpoint interval for solver convergence checks.",
    )
    parser.add_argument(
        "--cg-max-iters",
        type=int,
        default=512,
        help="Maximum inner CG iterations for the CE_h projection.",
    )
    parser.add_argument(
        "--newton-iters",
        type=int,
        default=16,
        help="Newton iterations for the dual proximal step.",
    )
    parser.add_argument(
        "--cg-tol",
        type=float,
        default=1e-10,
        help="CG tolerance used by the CE_h projection.",
    )
    parser.add_argument(
        "--continuation-epsilons",
        type=str,
        default="0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.0",
        help=(
            "Nonincreasing endpoint-relaxation schedule for continuation cases. "
            "Each stage uses (1 - eps) * rho + eps * I."
        ),
    )
    parser.add_argument(
        "--case-names",
        type=str,
        default="",
        help=(
            "Comma-separated subset of sweep cases to run. "
            "Default: all available cases."
        ),
    )
    parser.set_defaults(debug_trace=True)
    parser.add_argument(
        "--debug-trace",
        dest="debug_trace",
        action="store_true",
        help="Record checkpointed solver metrics and save per-stage trace plots.",
    )
    parser.add_argument(
        "--no-debug-trace",
        dest="debug_trace",
        action="store_false",
        help="Disable debug-trace recording and plotting.",
    )
    args = parser.parse_args()

    if args.side < 2:
        raise ValueError("side must be at least 2")
    if args.steps < 2:
        raise ValueError("steps must be at least 2")
    if args.blob_size < 1 or args.blob_size > args.side:
        raise ValueError("blob-size must lie in [1, side]")

    continuation_epsilons = _normalize_continuation_epsilons(
        _parse_float_list(args.continuation_epsilons)
    )
    sweep_cases = _default_sweep_cases(continuation_epsilons)
    available_case_names = tuple(sweep_cases.keys())
    requested_case_names = (
        [name.strip() for name in args.case_names.split(",") if name.strip()]
        if args.case_names.strip()
        else list(available_case_names)
    )
    unknown_case_names = sorted(set(requested_case_names) - set(available_case_names))
    if unknown_case_names:
        raise ValueError(
            f"unknown case names: {unknown_case_names}; available cases are {available_case_names}"
        )

    graph = grid_graph(args.side)
    source_rows = range(args.blob_size)
    source_cols = range(args.blob_size)
    target_rows = range(args.side - args.blob_size, args.side)
    target_cols = range(args.side - args.blob_size, args.side)
    rho_a_base = block_density(graph, args.side, rows=source_rows, cols=source_cols)
    rho_b_base = block_density(graph, args.side, rows=target_rows, cols=target_cols)
    pi = np.asarray(graph.pi)
    memory_mb = estimate_state_memory_bytes(graph, args.steps) / (1024.0 * 1024.0)
    output_dir = OUTPUT_DIR

    print(
        f"grid={args.side}x{args.side}, num_nodes={graph.num_nodes}, num_edges={graph.num_edges}, "
        f"num_steps={args.steps}, blob_size={args.blob_size}"
    )
    print(f"estimated_persistent_memory_mb={memory_mb:.2f}")
    print(f"pi_min={float(np.min(pi)):.8e}, pi_max={float(np.max(pi)):.8e}")
    print(f"continuation_epsilons={continuation_epsilons}")
    print(f"available_cases={available_case_names}")
    print(f"selected_cases={tuple(requested_case_names)}")

    summary_rows: list[dict[str, object]] = []

    for case_name in requested_case_names:
        case = sweep_cases[case_name]
        case_base_name = (
            f"large_grid_{args.side}x{args.side}_blob{args.blob_size}_steps{args.steps}_{case.name}"
        )
        stage_budgets = _allocate_iteration_budget(args.max_iters, len(case.continuation_epsilons))
        stage_rows: list[dict[str, object]] = []
        initial_state = None
        final_solution = None
        final_stage_trace_npz: Path | None = None
        final_stage_trace_plot: Path | None = None

        print(
            f"case={case.name}, cg_preconditioner={case.cg_preconditioner}, "
            f"solver_relaxation={case.relaxation:.3f}, stage_budgets={stage_budgets}"
        )

        for stage_index, (epsilon, stage_max_iters) in enumerate(
            zip(case.continuation_epsilons, stage_budgets, strict=True),
            start=1,
        ):
            rho_a = regularize_density(graph, rho_a_base, epsilon)
            rho_b = regularize_density(graph, rho_b_base, epsilon)
            config = OTConfig(
                relaxation=case.relaxation,
                warm_start=case.warm_start,
                max_iters=stage_max_iters,
                check_every=args.check_every,
                newton_iters=args.newton_iters,
                cg_max_iters=args.cg_max_iters,
                cg_tol=args.cg_tol,
                cg_preconditioner=case.cg_preconditioner,
                record_debug_trace=args.debug_trace,
            )
            print(
                f"case={case.name} stage={stage_index}/{len(stage_budgets)} "
                f"endpoint_epsilon={epsilon:.8g} stage_max_iters={stage_max_iters}"
            )
            final_solution = solve_problem(
                graph,
                rho_a,
                rho_b,
                num_steps=args.steps,
                config=config,
                initial_state=initial_state,
            )
            initial_state = final_solution.state
            print(
                summarize_solution(
                    f"{case.name}[stage={stage_index},eps={epsilon:.8g}]",
                    final_solution,
                )
            )

            stage_trace_npz = None
            stage_trace_plot = None
            if final_solution.debug_trace is not None:
                stage_trace_base_name = (
                    f"{case_base_name}_stage{stage_index:02d}_eps{_float_tag(epsilon)}"
                )
                stage_trace_npz = save_debug_trace_npz(
                    output_dir,
                    stage_trace_base_name,
                    final_solution.debug_trace,
                )
                stage_trace_plot = save_debug_trace_plot(
                    output_dir,
                    stage_trace_base_name,
                    final_solution.debug_trace,
                    title=(
                        f"Large grid transport ({args.side}x{args.side}, {case.name}): "
                        f"stage {stage_index} trace at epsilon={epsilon:.8g}"
                    ),
                )
                final_stage_trace_npz = stage_trace_npz
                final_stage_trace_plot = stage_trace_plot

            stage_rows.append(
                {
                    "case_name": case.name,
                    "stage_index": stage_index,
                    "num_stages": len(stage_budgets),
                    "endpoint_epsilon": epsilon,
                    "stage_max_iters": stage_max_iters,
                    "iterations_used": final_solution.iterations_used,
                    "converged": final_solution.converged,
                    "distance": float(final_solution.distance),
                    "action": float(final_solution.action),
                    "continuity_residual": float(final_solution.diagnostics["continuity_residual"]),
                    "max_constraint_residual": float(
                        final_solution.diagnostics["max_constraint_residual"]
                    ),
                    "ceh_cg_residual": float(final_solution.diagnostics["ceh_cg_residual"]),
                    "ceh_cg_iters": int(final_solution.diagnostics["ceh_cg_iters"]),
                    "debug_trace_npz": str(stage_trace_npz) if stage_trace_npz is not None else "",
                    "debug_trace_plot": (
                        str(stage_trace_plot) if stage_trace_plot is not None else ""
                    ),
                }
            )

        if final_solution is None:
            raise RuntimeError(f"case {case.name} produced no solution")

        rho = np.asarray(final_solution.state.rho)
        state_path = save_solution(output_dir, case_base_name, final_solution)
        node_plot = save_node_mass_heatmap(
            output_dir,
            case_base_name,
            graph,
            final_solution,
            title=(
                f"Large grid transport ({args.side}x{args.side}, {case.name}): "
                "probability mass over time"
            ),
        )
        flow_plot = save_edge_flow_heatmap(
            output_dir,
            case_base_name,
            graph,
            final_solution,
            title=(
                f"Large grid transport ({args.side}x{args.side}, {case.name}): "
                "edge flow over time"
            ),
        )
        graph_plot = save_graph_snapshot_series(
            output_dir,
            case_base_name,
            graph,
            final_solution,
            positions=grid_layout(args.side),
            title=(
                f"Large grid transport ({args.side}x{args.side}, {case.name}): "
                "graph snapshots (node area scales with density)"
            ),
            snapshot_indices=(0, rho.shape[0] // 2, rho.shape[0] - 1),
        )

        stage_summary_path = _write_csv(
            output_dir / f"{case_base_name}_stage_summary.csv",
            [
                "case_name",
                "stage_index",
                "num_stages",
                "endpoint_epsilon",
                "stage_max_iters",
                "iterations_used",
                "converged",
                "distance",
                "action",
                "continuity_residual",
                "max_constraint_residual",
                "ceh_cg_residual",
                "ceh_cg_iters",
                "debug_trace_npz",
                "debug_trace_plot",
            ],
            stage_rows,
        )

        node_mass = probability_mass(graph, final_solution)
        midpoint_mass = node_mass[node_mass.shape[0] // 2]
        source_midpoint_mass = _block_mass(
            midpoint_mass,
            args.side,
            rows=source_rows,
            cols=source_cols,
        )
        target_midpoint_mass = _block_mass(
            midpoint_mass,
            args.side,
            rows=target_rows,
            cols=target_cols,
        )
        total_iterations_used = int(sum(int(row["iterations_used"]) for row in stage_rows))
        summary_rows.append(
            {
                "case_name": case.name,
                "cg_preconditioner": case.cg_preconditioner,
                "solver_relaxation": case.relaxation,
                "continuation_epsilons": "|".join(
                    f"{epsilon:.8g}" for epsilon in case.continuation_epsilons
                ),
                "total_iteration_budget": args.max_iters,
                "total_iterations_used": total_iterations_used,
                "final_converged": final_solution.converged,
                "final_distance": float(final_solution.distance),
                "final_action": float(final_solution.action),
                "final_continuity_residual": float(
                    final_solution.diagnostics["continuity_residual"]
                ),
                "final_max_constraint_residual": float(
                    final_solution.diagnostics["max_constraint_residual"]
                ),
                "final_ceh_cg_residual": float(final_solution.diagnostics["ceh_cg_residual"]),
                "final_ceh_cg_iters": int(final_solution.diagnostics["ceh_cg_iters"]),
                "midpoint_source_mass": source_midpoint_mass,
                "midpoint_target_mass": target_midpoint_mass,
                "saved_state": str(state_path),
                "saved_node_plot": str(node_plot),
                "saved_flow_plot": str(flow_plot),
                "saved_graph_plot": str(graph_plot),
                "saved_stage_summary": str(stage_summary_path),
                "saved_final_debug_trace_npz": (
                    str(final_stage_trace_npz) if final_stage_trace_npz is not None else ""
                ),
                "saved_final_debug_trace_plot": (
                    str(final_stage_trace_plot) if final_stage_trace_plot is not None else ""
                ),
            }
        )

        print(
            "midpoint_corner_masses="
            f"{{'case': '{case.name}', 'source': {source_midpoint_mass:.8f}, "
            f"'target': {target_midpoint_mass:.8f}}}"
        )
        print(f"saved_state={state_path}")
        print(f"saved_node_plot={node_plot}")
        print(f"saved_flow_plot={flow_plot}")
        print(f"saved_graph_plot={graph_plot}")
        print(f"saved_stage_summary={stage_summary_path}")
        if final_stage_trace_npz is not None:
            print(f"saved_final_debug_trace_npz={final_stage_trace_npz}")
        if final_stage_trace_plot is not None:
            print(f"saved_final_debug_trace_plot={final_stage_trace_plot}")

    sweep_summary_path = _write_csv(
        output_dir
        / (
            f"large_grid_{args.side}x{args.side}_blob{args.blob_size}"
            f"_steps{args.steps}_sweep_summary.csv"
        ),
        [
            "case_name",
            "cg_preconditioner",
            "solver_relaxation",
            "continuation_epsilons",
            "total_iteration_budget",
            "total_iterations_used",
            "final_converged",
            "final_distance",
            "final_action",
            "final_continuity_residual",
            "final_max_constraint_residual",
            "final_ceh_cg_residual",
            "final_ceh_cg_iters",
            "midpoint_source_mass",
            "midpoint_target_mass",
            "saved_state",
            "saved_node_plot",
            "saved_flow_plot",
            "saved_graph_plot",
            "saved_stage_summary",
            "saved_final_debug_trace_npz",
            "saved_final_debug_trace_plot",
        ],
        summary_rows,
    )

    best_row = min(
        summary_rows,
        key=lambda row: (
            not bool(row["final_converged"]),
            float("inf")
            if not np.isfinite(float(row["final_max_constraint_residual"]))
            else float(row["final_max_constraint_residual"]),
            float("inf")
            if not np.isfinite(float(row["final_action"]))
            else float(row["final_action"]),
        ),
    )
    print(f"best_case={best_row['case_name']}")
    print(f"saved_sweep_summary={sweep_summary_path}")


if __name__ == "__main__":
    main()
