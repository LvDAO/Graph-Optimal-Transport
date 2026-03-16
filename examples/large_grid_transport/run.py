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


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.split(","):
        stripped = token.strip()
        if stripped:
            values.append(int(stripped))
    if not values:
        raise ValueError("expected at least one comma-separated integer value")
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


def _normalize_sweep_steps(step_values: tuple[int, ...]) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for step in step_values:
        if step < 2:
            raise ValueError("swept step counts must be at least 2")
        if step not in seen:
            normalized.append(int(step))
            seen.add(int(step))
    return tuple(normalized)


def _warm_start_suffix(warm_start: str) -> str:
    return "" if warm_start == "linear_path" else f"_warm_{warm_start}"


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
    num_steps: int
    endpoint_epsilon: float


def _default_sweep_cases(
    side: int,
    blob_size: int,
    step_values: tuple[int, ...],
    epsilons: tuple[float, ...],
) -> dict[str, SweepCase]:
    return {
        (
            f"large_grid_{side}x{side}_blob{blob_size}_steps"
            f"{num_steps}_eps{_float_tag(epsilon)}"
        ): SweepCase(
            name=(
                f"large_grid_{side}x{side}_blob{blob_size}_steps"
                f"{num_steps}_eps{_float_tag(epsilon)}"
            ),
            num_steps=num_steps,
            endpoint_epsilon=epsilon,
        )
        for num_steps in step_values
        for epsilon in epsilons
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
            "Run demanding 32x32-style large-grid transport sweeps over endpoint "
            "regularization epsilon and optional time-step counts."
        )
    )
    parser.add_argument("--side", type=int, default=32, help="Side length of the square grid.")
    parser.add_argument("--steps", type=int, default=32, help="Number of time intervals.")
    parser.add_argument(
        "--sweep-steps",
        type=str,
        default="",
        help=(
            "Comma-separated time-step counts to sweep. "
            "Default: use the single value from --steps."
        ),
    )
    parser.add_argument(
        "--blob-size",
        type=int,
        default=6,
        help="Side length of the source and target corner mass blocks.",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        choices=("linear_path", "zero", "harmonic_socp"),
        default="linear_path",
        help="Warm-start mode passed through to OTConfig.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=409600,
        help="Maximum solver iterations for each sweep case.",
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
        "--cg-preconditioner",
        type=str,
        default="block_jacobi",
        choices=("jacobi", "block_jacobi"),
        help="Fixed CG preconditioner used for every sweep case.",
    )
    parser.add_argument(
        "--relaxation",
        type=float,
        default=1.0,
        help="Fixed PDHG relaxation used for every sweep case.",
    )
    parser.add_argument(
        "--continuation-epsilons",
        type=str,
        default="0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.0",
        help=(
            "Nonincreasing endpoint-relaxation values for the sweep. "
            "Each case uses (1 - eps) * rho + eps * I."
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
    parser.set_defaults(verbose=True)
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
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Show the C++ solver progress bar during each sweep case.",
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Disable the C++ solver progress bar for the sweep.",
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
    sweep_steps = _normalize_sweep_steps(
        _parse_int_list(args.sweep_steps) if args.sweep_steps.strip() else (args.steps,)
    )
    sweep_cases = _default_sweep_cases(
        args.side,
        args.blob_size,
        sweep_steps,
        continuation_epsilons,
    )
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
    output_dir = OUTPUT_DIR
    estimated_memory_mb = {
        num_steps: estimate_state_memory_bytes(graph, num_steps) / (1024.0 * 1024.0)
        for num_steps in sweep_steps
    }

    print(
        f"grid={args.side}x{args.side}, num_nodes={graph.num_nodes}, num_edges={graph.num_edges}, "
        f"default_num_steps={args.steps}, sweep_steps={sweep_steps}, blob_size={args.blob_size}"
    )
    print(f"estimated_persistent_memory_mb_by_steps={estimated_memory_mb}")
    print(f"pi_min={float(np.min(pi)):.8e}, pi_max={float(np.max(pi)):.8e}")
    print(f"sweep_epsilons={continuation_epsilons}")
    print(
        f"fixed_solver_settings={{'cg_preconditioner': '{args.cg_preconditioner}', "
        f"'relaxation': {args.relaxation:.3f}, 'warm_start': '{args.warm_start}', "
        f"'verbose': {args.verbose}}}"
    )
    print(f"available_cases={available_case_names}")
    print(f"selected_cases={tuple(requested_case_names)}")

    summary_rows: list[dict[str, object]] = []
    warm_start_suffix = _warm_start_suffix(args.warm_start)

    for case_name in requested_case_names:
        case = sweep_cases[case_name]
        case_output_dir = output_dir / f"{case.name}{warm_start_suffix}"
        rho_a = regularize_density(graph, rho_a_base, case.endpoint_epsilon)
        rho_b = regularize_density(graph, rho_b_base, case.endpoint_epsilon)
        config = OTConfig(
            relaxation=args.relaxation,
            warm_start=args.warm_start,
            max_iters=args.max_iters,
            check_every=args.check_every,
            newton_iters=args.newton_iters,
            cg_max_iters=args.cg_max_iters,
            cg_tol=args.cg_tol,
            cg_preconditioner=args.cg_preconditioner,
            record_debug_trace=args.debug_trace,
            verbose=args.verbose,
        )
        print(
            f"case={case.name}, num_steps={case.num_steps}, "
            f"endpoint_epsilon={case.endpoint_epsilon:.8g}, warm_start={args.warm_start}, "
            f"max_iters={args.max_iters}"
        )
        final_solution = solve_problem(
            graph,
            rho_a,
            rho_b,
            num_steps=case.num_steps,
            config=config,
        )
        print(
            summarize_solution(
                f"{case.name}[steps={case.num_steps},eps={case.endpoint_epsilon:.8g}]",
                final_solution,
            )
        )

        debug_trace_npz = None
        debug_trace_plot = None
        if final_solution.debug_trace is not None:
            debug_trace_npz = save_debug_trace_npz(
                case_output_dir,
                case.name,
                final_solution.debug_trace,
            )
            debug_trace_plot = save_debug_trace_plot(
                case_output_dir,
                case.name,
                final_solution.debug_trace,
                title=(
                    f"Large grid transport ({args.side}x{args.side}, steps={case.num_steps}, "
                    f"epsilon={case.endpoint_epsilon:.8g})"
                ),
            )

        rho = np.asarray(final_solution.state.rho)
        state_path = save_solution(case_output_dir, case.name, final_solution)
        node_plot = save_node_mass_heatmap(
            case_output_dir,
            case.name,
            graph,
            final_solution,
            title=(
                f"Large grid transport ({args.side}x{args.side}, steps={case.num_steps}, "
                f"epsilon={case.endpoint_epsilon:.8g}): regularized probability mass over time"
            ),
        )
        flow_plot = save_edge_flow_heatmap(
            case_output_dir,
            case.name,
            graph,
            final_solution,
            title=(
                f"Large grid transport ({args.side}x{args.side}, steps={case.num_steps}, "
                f"epsilon={case.endpoint_epsilon:.8g}): edge flow over time"
            ),
        )
        graph_plot = save_graph_snapshot_series(
            case_output_dir,
            case.name,
            graph,
            final_solution,
            positions=grid_layout(args.side),
            title=(
                f"Large grid transport ({args.side}x{args.side}, steps={case.num_steps}, "
                f"epsilon={case.endpoint_epsilon:.8g}): regularized endpoint and midpoint snapshots"
            ),
            snapshot_indices=(0, rho.shape[0] // 2, rho.shape[0] - 1),
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
        case_row = {
            "case_name": case.name,
            "warm_start": args.warm_start,
            "num_steps": case.num_steps,
            "endpoint_epsilon": case.endpoint_epsilon,
            "cg_preconditioner": args.cg_preconditioner,
            "solver_relaxation": args.relaxation,
            "max_iters": args.max_iters,
            "iterations_used": final_solution.iterations_used,
            "converged": final_solution.converged,
            "distance": float(final_solution.distance),
            "action": float(final_solution.action),
            "continuity_residual": float(final_solution.diagnostics["continuity_residual"]),
            "max_constraint_residual": float(final_solution.diagnostics["max_constraint_residual"]),
            "ceh_cg_residual": float(final_solution.diagnostics["ceh_cg_residual"]),
            "ceh_cg_iters": int(final_solution.diagnostics["ceh_cg_iters"]),
            "midpoint_source_mass": source_midpoint_mass,
            "midpoint_target_mass": target_midpoint_mass,
            "case_output_dir": str(case_output_dir),
            "saved_state": str(state_path),
            "saved_node_plot": str(node_plot),
            "saved_flow_plot": str(flow_plot),
            "saved_graph_plot": str(graph_plot),
            "saved_debug_trace_npz": str(debug_trace_npz) if debug_trace_npz is not None else "",
            "saved_debug_trace_plot": str(debug_trace_plot) if debug_trace_plot is not None else "",
        }
        case_summary_path = _write_csv(
            case_output_dir / "case_summary.csv",
            list(case_row.keys()),
            [case_row],
        )
        case_row["saved_case_summary"] = str(case_summary_path)
        summary_rows.append(case_row)

        print(
            "midpoint_corner_masses="
            f"{{'case': '{case.name}', 'source': {source_midpoint_mass:.8f}, "
            f"'target': {target_midpoint_mass:.8f}}}"
        )
        print(f"case_output_dir={case_output_dir}")
        print(f"saved_state={state_path}")
        print(f"saved_node_plot={node_plot}")
        print(f"saved_flow_plot={flow_plot}")
        print(f"saved_graph_plot={graph_plot}")
        print(f"saved_case_summary={case_summary_path}")
        if debug_trace_npz is not None:
            print(f"saved_debug_trace_npz={debug_trace_npz}")
        if debug_trace_plot is not None:
            print(f"saved_debug_trace_plot={debug_trace_plot}")

    sweep_summary_path = _write_csv(
        output_dir
        / (
            f"large_grid_{args.side}x{args.side}_blob{args.blob_size}"
            f"{warm_start_suffix}_sweep_summary.csv"
        ),
        [
            "case_name",
            "warm_start",
            "num_steps",
            "endpoint_epsilon",
            "cg_preconditioner",
            "solver_relaxation",
            "max_iters",
            "iterations_used",
            "converged",
            "distance",
            "action",
            "continuity_residual",
            "max_constraint_residual",
            "ceh_cg_residual",
            "ceh_cg_iters",
            "midpoint_source_mass",
            "midpoint_target_mass",
            "case_output_dir",
            "saved_state",
            "saved_node_plot",
            "saved_flow_plot",
            "saved_graph_plot",
            "saved_debug_trace_npz",
            "saved_debug_trace_plot",
            "saved_case_summary",
        ],
        summary_rows,
    )

    best_row = min(
        summary_rows,
        key=lambda row: (
            not bool(row["converged"]),
            float("inf")
            if not np.isfinite(float(row["max_constraint_residual"]))
            else float(row["max_constraint_residual"]),
            float("inf")
            if not np.isfinite(float(row["action"]))
            else float(row["action"]),
        ),
    )
    print(f"best_case={best_row['case_name']}")
    print(f"saved_sweep_summary={sweep_summary_path}")


if __name__ == "__main__":
    main()
