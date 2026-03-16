# Examples

This folder contains runnable examples for `graphot`.

If you are using a local checkout, install the example dependencies first:

```bash
uv sync --extra examples
```

Then run examples with:

```bash
uv run python examples/...
```

## Included Scripts

- `two_node_benchmark/`: smallest benchmark and best first example.
- `cycle_neighbor_transport/`: transport on small cycle graphs.
- `line_chain_transport/`: transport along a path graph.
- `directed_reversible_transport/`: directed but reversible graph example.
- `large_grid_transport/`: larger grid sweeps, traces, warm-start comparisons,
  and `benchmark_openmp.py` for thread-scaling benchmarks.

## Recommended First Commands

```bash
uv run python examples/two_node_benchmark/run.py
uv run python examples/cycle_neighbor_transport/run.py
```

For a single traced large-grid case:

```bash
uv run python examples/large_grid_transport/run.py \
  --side 32 \
  --steps 4 \
  --sweep-steps 4 \
  --blob-size 6 \
  --continuation-epsilons 0 \
  --case-names large_grid_32x32_blob6_steps4_eps0 \
  --warm-start linear_path \
  --max-iters 409600 \
  --check-every 256 \
  --debug-trace \
  --verbose
```

This is the recommended first large-grid command. Running the script with a
broader sweep configuration is substantially heavier.

## Warm-Start Modes In `large_grid_transport/run.py`

The large-grid example supports:
- `--warm-start linear_path`
- `--warm-start zero`
- `--warm-start harmonic_socp`

Use `harmonic_socp` only if MOSEK Fusion is installed and licensed.

## Output Files

Most scripts save:
- a state `.npz` file,
- a node plot,
- an edge-flow plot,
- a graph snapshot.

The large-grid scripts also save:
- debug-trace artifacts,
- per-case summaries,
- and benchmark CSV files when applicable.

## CPU Threads

If your build includes OpenMP, the examples use all available CPU threads by
default. Override that before Python starts:

```bash
GRAPHOT_NUM_THREADS=16 uv run python examples/large_grid_transport/run.py
```

or:

```bash
OMP_NUM_THREADS=16 uv run python examples/large_grid_transport/run.py
```

For more detail on what each example is for, see
[`docs/examples-guide.md`](../docs/examples-guide.md).
