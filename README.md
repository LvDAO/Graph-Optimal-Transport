# graphot

`graphot` computes dynamic optimal transport on sparse reversible graphs.

Give it:
- a graph,
- a starting distribution,
- an ending distribution,
- and a time discretization,

and it returns:
- node densities over time,
- edge fluxes over time,
- a transport distance,
- and optional checkpointed debug traces.

`graphot` is a good fit when you want the full transport path on a graph, not
just a static coupling.

## What It Supports Today

The current public API targets:
- sparse reversible graphs,
- Python-first workflows with NumPy arrays,
- small and medium problems by default,
- exploratory large-grid experiments with diagnostics.

It does not target:
- nonreversible graphs,
- GPU execution,
- entropy-regularized OT,
- or production-scale graph OT without user tuning.

## Install

Supported environment:
- Python `3.11+`
- NumPy/SciPy-based Python workflow
- CPU execution

Install from PyPI:

```bash
pip install graphot
```

If you want the plotting examples:

```bash
pip install "graphot[examples]"
```

If you are working from a local checkout and want the repository-managed
environment, use `uv`:

```bash
uv sync --extra examples
```

Then run scripts with:

```bash
uv run python your_script.py
```

Notes:
- The package depends on the Python `mosek` package.
- A working MOSEK license is only required if you use `warm_start="harmonic_socp"`
  or call `solve_harmonic_socp_warm_start(...)`.

If you build from source instead of using a wheel, expect to need:
- a working C++ toolchain,
- CMake,
- and Conan via the build backend declared in `pyproject.toml`.

## Quick Start

```python
import numpy as np

from graphot import (
    GraphSpec,
    LogMeanOps,
    OTConfig,
    OTProblem,
    TimeDiscretization,
    solve_ot,
)

graph = GraphSpec.from_undirected_weights(
    num_nodes=2,
    edge_u=[0],
    edge_v=[1],
    weight=[1.0],
)

mass_a = np.array([1.0, 0.0], dtype=np.float64)
mass_b = np.array([0.0, 1.0], dtype=np.float64)

rho_a = mass_a / graph.pi
rho_b = mass_b / graph.pi

problem = OTProblem(
    graph=graph,
    time=TimeDiscretization(num_steps=64),
    rho_a=rho_a,
    rho_b=rho_b,
    mean_ops=LogMeanOps(),
)

solution = solve_ot(problem, OTConfig())

print("distance:", float(solution.distance))
print("converged:", solution.converged)
print("iterations:", solution.iterations_used)
```

The one rule that matters most:

- `graphot` expects endpoint densities with respect to `graph.pi`, not raw masses.
- Convert masses with `rho = mass / graph.pi`.
- Check `solution.converged` before treating a harder run as final.

## Warm Starts

`OTConfig.warm_start` supports:
- `"linear_path"`: default warm start for normal use,
- `"zero"`: disable the built-in linear warm start,
- `"harmonic_socp"`: build a harmonic-mean SOCP warm start with MOSEK, then
  start the log-mean solver from that state.

You can also pass `initial_state=previous_solution.state` to `solve_ot(...)`.
That takes precedence over `config.warm_start`.

## Example Commands

Run the smallest example:

```bash
uv run python examples/two_node_benchmark/run.py
```

Run one large-grid case with a trace:

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

The large-grid example writes its outputs under:

```text
examples/large_grid_transport/output/
```

The large-grid scripts are intentionally user-facing but computationally
heavier than the small examples. Start with one case before trying a full sweep.

## CPU Threads

If your build includes OpenMP, `graphot` uses all available CPU threads by
default.

To choose a different thread count, set an environment variable before Python
starts:

```bash
GRAPHOT_NUM_THREADS=16 uv run python your_script.py
```

or:

```bash
OMP_NUM_THREADS=16 uv run python your_script.py
```

If the extension was built without OpenMP, solves stay single-threaded.

## Read Next

- [Getting Started](docs/getting-started.md)
- [Graph Model](docs/graph-model.md)
- [API Reference](docs/api-reference.md)
- [Examples Guide](docs/examples-guide.md)
- [Debugging and Diagnostics](docs/debugging-and-diagnostics.md)
- [Numerical Limitations](docs/numerical-limitations.md)

The example index is here:

- [examples/README.md](examples/README.md)
