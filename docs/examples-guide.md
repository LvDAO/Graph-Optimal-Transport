# Examples Guide

The repository includes runnable examples under `examples/`.

Each example writes its outputs to a local `output/` directory.

## Start With These

If you are new to `graphot`, use this order:

1. two-node benchmark
2. cycle transport
3. directed reversible transport
4. line chain transport
5. large-grid transport

This goes from simplest to most demanding.

## Two-Node Benchmark

Path:

- `examples/two_node_benchmark/run.py`

Use it for:

- a first successful run,
- checking the reported distance against a known reference,
- understanding the meaning of `rho` and `m` on the smallest possible graph.

## Cycle Transport

Path:

- `examples/cycle_neighbor_transport/run.py`

Use it for:

- seeing how mass chooses between nearby routes,
- understanding the qualitative shape of a transport path on a small graph.

## Line Chain Transport

Path:

- `examples/line_chain_transport/run.py`

Use it for:

- simple path-graph transport,
- inspecting how mass moves along a one-dimensional graph.

## Directed Reversible Transport

Path:

- `examples/directed_reversible_transport/run.py`

Use it for:

- examples where the graph is directed but still reversible,
- learning when `GraphSpec.from_directed_rates(...)` is appropriate.

## Large-Grid Transport

Path:

- `examples/large_grid_transport/run.py`

Use it for:

- testing the solver on a larger problem,
- generating richer plots,
- exploring debug traces on harder cases,
- comparing large-solver settings and continuation schedules on the `32x32` setup.

This is the most demanding example in the repository.

## Running The Examples

After installing the optional plotting dependencies, run scripts from the repo
root with:

```bash
python examples/two_node_benchmark/run.py
python examples/cycle_neighbor_transport/run.py
python examples/line_chain_transport/run.py
python examples/directed_reversible_transport/run.py
python examples/large_grid_transport/run.py
```

If your build includes OpenMP, the examples use all available CPU threads by
default. To run with a different thread count, prefix the command with an
environment variable:

```bash
GRAPHOT_NUM_THREADS=16 python examples/large_grid_transport/run.py
```

`OMP_NUM_THREADS=16 ...` also works if you already manage OpenMP that way.

## Typical Outputs

Most example scripts write:

- a saved state file,
- a node plot,
- an edge-flow plot,
- a graph snapshot.

The large-grid example saves debug traces by default. Use `--no-debug-trace`
if you want to skip them.
