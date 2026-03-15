from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _bootstrap_examples_dir() -> None:
    examples_dir = Path(__file__).resolve().parents[1]
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.split(","):
        stripped = token.strip()
        if stripped:
            values.append(int(stripped))
    if not values:
        raise ValueError("expected at least one comma-separated integer value")
    if any(value <= 0 for value in values):
        raise ValueError("thread counts must be positive")
    return tuple(values)


def _default_threads() -> tuple[int, ...]:
    cpu_count = os.cpu_count() or 1
    candidates = (1, 2, 4, 8)
    selected = tuple(candidate for candidate in candidates if candidate <= cpu_count)
    return selected if selected else (1,)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the OpenMP-enabled large-grid solver against the single-thread baseline "
            "using separate subprocesses with different OMP_NUM_THREADS values."
        )
    )
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--blob-size", type=int, default=6)
    parser.add_argument("--max-iters", type=int, default=128)
    parser.add_argument("--check-every", type=int, default=32)
    parser.add_argument("--cg-max-iters", type=int, default=256)
    parser.add_argument("--newton-iters", type=int, default=16)
    parser.add_argument("--cg-tol", type=float, default=1e-10)
    parser.add_argument("--cg-preconditioner", type=str, default="block_jacobi")
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--warm-start", type=str, default="linear_path")
    parser.add_argument("--threads", type=str, default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output-name", type=str, default="openmp_large_grid_benchmark")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-threads", type=int, default=1, help=argparse.SUPPRESS)
    return parser


def _run_worker(args: argparse.Namespace) -> None:
    _bootstrap_examples_dir()

    from _common import block_density, grid_graph, solve_problem
    from graphot import OTConfig
    from graphot import _core as _core_backend

    graph = grid_graph(args.side)
    source_rows = range(args.blob_size)
    source_cols = range(args.blob_size)
    target_rows = range(args.side - args.blob_size, args.side)
    target_cols = range(args.side - args.blob_size, args.side)
    rho_a = block_density(graph, args.side, rows=source_rows, cols=source_cols)
    rho_b = block_density(graph, args.side, rows=target_rows, cols=target_cols)
    config = OTConfig(
        relaxation=args.relaxation,
        warm_start=args.warm_start,
        max_iters=args.max_iters,
        check_every=args.check_every,
        newton_iters=args.newton_iters,
        cg_max_iters=args.cg_max_iters,
        cg_tol=args.cg_tol,
        cg_preconditioner=args.cg_preconditioner,
        record_debug_trace=False,
    )

    for _ in range(args.warmup_runs):
        solve_problem(graph, rho_a, rho_b, num_steps=args.steps, config=config)

    started = time.perf_counter()
    solution = solve_problem(graph, rho_a, rho_b, num_steps=args.steps, config=config)
    elapsed_seconds = time.perf_counter() - started
    payload = {
        "threads": args.worker_threads,
        "elapsed_seconds": elapsed_seconds,
        "distance": float(solution.distance),
        "action": float(solution.action),
        "iterations_used": int(solution.iterations_used),
        "converged": bool(solution.converged),
        "continuity_residual": float(solution.diagnostics["continuity_residual"]),
        "max_constraint_residual": float(solution.diagnostics["max_constraint_residual"]),
        "ceh_cg_iters": int(solution.diagnostics["ceh_cg_iters"]),
        "extension_info": _core_backend.extension_info(),
    }
    print(json.dumps(payload, sort_keys=True))


def _run_driver(args: argparse.Namespace) -> None:
    thread_counts = _parse_int_list(args.threads) if args.threads.strip() else _default_threads()
    benchmark_rows: list[dict[str, object]] = []
    medians: dict[int, float] = {}

    for thread_count in thread_counts:
        elapsed_values: list[float] = []
        print(f"benchmark_threads={thread_count}")
        for repeat in range(1, args.repeats + 1):
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(thread_count)
            env.setdefault("OMP_PROC_BIND", "true")
            env.setdefault("OMP_PLACES", "cores")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-threads",
                str(thread_count),
                "--side",
                str(args.side),
                "--steps",
                str(args.steps),
                "--blob-size",
                str(args.blob_size),
                "--max-iters",
                str(args.max_iters),
                "--check-every",
                str(args.check_every),
                "--cg-max-iters",
                str(args.cg_max_iters),
                "--newton-iters",
                str(args.newton_iters),
                "--cg-tol",
                str(args.cg_tol),
                "--cg-preconditioner",
                args.cg_preconditioner,
                "--relaxation",
                str(args.relaxation),
                "--warm-start",
                args.warm_start,
                "--warmup-runs",
                str(args.warmup_runs),
            ]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError(f"worker produced no output for threads={thread_count}")
            payload = json.loads(lines[-1])
            elapsed_seconds = float(payload["elapsed_seconds"])
            elapsed_values.append(elapsed_seconds)
            row = {
                "threads": thread_count,
                "repeat": repeat,
                "elapsed_seconds": elapsed_seconds,
                "distance": float(payload["distance"]),
                "action": float(payload["action"]),
                "iterations_used": int(payload["iterations_used"]),
                "converged": bool(payload["converged"]),
                "continuity_residual": float(payload["continuity_residual"]),
                "max_constraint_residual": float(payload["max_constraint_residual"]),
                "ceh_cg_iters": int(payload["ceh_cg_iters"]),
                "compiled_with_openmp": bool(payload["extension_info"]["compiled_with_openmp"]),
                "openmp_max_threads": int(payload["extension_info"]["openmp_max_threads"]),
                "openmp_num_procs": int(payload["extension_info"]["openmp_num_procs"]),
                "simd_level": str(payload["extension_info"]["simd_level"]),
            }
            benchmark_rows.append(row)
            print(
                f"threads={thread_count} repeat={repeat} elapsed_seconds={elapsed_seconds:.6f} "
                f"iterations={row['iterations_used']} converged={row['converged']}"
            )
        medians[thread_count] = statistics.median(elapsed_values)

    baseline = medians[min(thread_counts)]
    summary_rows = []
    for thread_count in thread_counts:
        thread_rows = [row for row in benchmark_rows if int(row["threads"]) == thread_count]
        elapsed_values = [float(row["elapsed_seconds"]) for row in thread_rows]
        median_elapsed = medians[thread_count]
        summary_rows.append(
            {
                "threads": thread_count,
                "median_elapsed_seconds": median_elapsed,
                "min_elapsed_seconds": min(elapsed_values),
                "max_elapsed_seconds": max(elapsed_values),
                "speedup_vs_thread1": baseline / median_elapsed,
                "compiled_with_openmp": bool(thread_rows[0]["compiled_with_openmp"]),
                "openmp_max_threads": int(thread_rows[0]["openmp_max_threads"]),
                "openmp_num_procs": int(thread_rows[0]["openmp_num_procs"]),
                "simd_level": str(thread_rows[0]["simd_level"]),
            }
        )

    output_dir = OUTPUT_DIR
    detail_path = _write_csv(
        output_dir / f"{args.output_name}_detail.csv",
        [
            "threads",
            "repeat",
            "elapsed_seconds",
            "distance",
            "action",
            "iterations_used",
            "converged",
            "continuity_residual",
            "max_constraint_residual",
            "ceh_cg_iters",
            "compiled_with_openmp",
            "openmp_max_threads",
            "openmp_num_procs",
            "simd_level",
        ],
        benchmark_rows,
    )
    summary_path = _write_csv(
        output_dir / f"{args.output_name}_summary.csv",
        [
            "threads",
            "median_elapsed_seconds",
            "min_elapsed_seconds",
            "max_elapsed_seconds",
            "speedup_vs_thread1",
            "compiled_with_openmp",
            "openmp_max_threads",
            "openmp_num_procs",
            "simd_level",
        ],
        summary_rows,
    )

    print(f"detail_csv={detail_path}")
    print(f"summary_csv={summary_path}")
    for row in summary_rows:
        print(
            f"summary threads={row['threads']} median_elapsed_seconds="
            f"{float(row['median_elapsed_seconds']):.6f} "
            f"speedup_vs_thread1={float(row['speedup_vs_thread1']):.3f}"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.worker:
        _run_worker(args)
        return
    _run_driver(args)


if __name__ == "__main__":
    main()
