# Examples Guide

The repository ships runnable examples under `examples/`.

If you are using a local checkout, install the example dependencies first:

```bash
uv sync --extra examples
```

Then run examples with:

```bash
uv run python examples/...
```

## Recommended Order

If you are new to `graphot`, use this order:

1. `two_node_benchmark`
2. `cycle_neighbor_transport`
3. `line_chain_transport`
4. `directed_reversible_transport`
5. `large_grid_transport`

This goes from easiest to most demanding.

## Two-Node Benchmark

Path:
- `examples/two_node_benchmark/run.py`

Use it for:
- the first successful run,
- checking the reported distance on the smallest possible graph,
- understanding `rho` and `m` with minimal complexity.

Run:

```bash
uv run python examples/two_node_benchmark/run.py
```

## Cycle Transport

Path:
- `examples/cycle_neighbor_transport/run.py`

Use it for:
- small routing examples,
- visual intuition on a tiny reversible graph,
- confirming that plotting works in your environment.

Run:

```bash
uv run python examples/cycle_neighbor_transport/run.py
```

## Line Chain Transport

Path:
- `examples/line_chain_transport/run.py`

Use it for:
- one-dimensional transport,
- simple path-graph experiments,
- inspecting edge fluxes along a single corridor.

Run:

```bash
uv run python examples/line_chain_transport/run.py
```

## Directed Reversible Transport

Path:
- `examples/directed_reversible_transport/run.py`

Use it for:
- directed graphs that are still reversible,
- learning when `GraphSpec.from_directed_rates(...)` is the right constructor.

Run:

```bash
uv run python examples/directed_reversible_transport/run.py
```

## Large-Grid Transport

Path:
- `examples/large_grid_transport/run.py`

Use it for:
- larger `32x32`-style experiments,
- debug traces,
- continuation schedules,
- warm-start comparisons,
- solver tuning on harder cases.

This is the hardest example in the repository.

### One Case With Trace

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

This is the recommended first large-grid command because it runs one case and
still saves a full trace.

### Full Sweep Behavior

If you run `examples/large_grid_transport/run.py` with its broader defaults, it
can launch a much heavier sweep over multiple endpoint regularization levels and
save a full set of plots and summaries for each case.

Use the single-case command above before trying a broader sweep.

### Warm-Start Modes

The large-grid example now supports:
- `--warm-start linear_path`
- `--warm-start zero`
- `--warm-start harmonic_socp`

Use `harmonic_socp` only if MOSEK Fusion is installed and licensed.

### Benchmarking OpenMP

Path:
- `examples/large_grid_transport/benchmark_openmp.py`

Use it for:
- thread scaling,
- comparing `OMP_NUM_THREADS` or `GRAPHOT_NUM_THREADS`,
- benchmarking different warm-start modes on the same case.

### Useful Flags

- `--warm-start ...`: choose `linear_path`, `zero`, or `harmonic_socp`
- `--continuation-epsilons ...`: choose which endpoint-regularization levels to sweep
- `--no-debug-trace`: skip debug-trace output when you only want the final state
- `--no-verbose`: disable checkpoint progress printing

## Typical Outputs

Most example scripts write:
- a saved state `.npz`,
- a node heatmap,
- an edge-flow heatmap,
- a graph snapshot plot.

The large-grid example also writes:
- a per-case CSV summary,
- a debug-trace `.npz`,
- a debug-trace plot.

## CPU Threads

If your build includes OpenMP, examples use all available CPU threads by
default. Override that before Python starts:

```bash
GRAPHOT_NUM_THREADS=16 uv run python examples/large_grid_transport/run.py
```

or:

```bash
OMP_NUM_THREADS=16 uv run python examples/large_grid_transport/run.py
```
