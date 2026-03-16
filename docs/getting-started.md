# Getting Started

This page is the shortest path to a correct first solve.

## Install

Supported environment:
- Python `3.11+`
- CPU execution
- sparse reversible graph problems

Install from PyPI:

```bash
pip install graphot
```

If you want the plotting examples:

```bash
pip install "graphot[examples]"
```

If you are working from a local checkout, the repository is set up to work well
with `uv`:

```bash
uv sync --extra examples
```

Then run scripts with:

```bash
uv run python your_script.py
```

Notes:
- The package depends on the Python `mosek` package.
- A MOSEK license is only needed when you use the harmonic SOCP warm start.
- If you build from source, expect to need a working C++ toolchain, CMake, and
  Conan through the configured build backend.

## First Complete Example

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

## The Input Rule You Must Remember

`graphot` expects endpoint densities with respect to `graph.pi`.

If you start from ordinary node masses, convert them with:

```python
rho = mass / graph.pi
```

The required normalization is:

```python
np.sum(graph.pi * rho) == 1
```

If this conversion is wrong, the solver will either reject the input or solve a
different problem from the one you intended.

## Reading The Result

Most users inspect:
- `solution.distance`
- `solution.converged`
- `solution.iterations_used`
- `solution.state.rho`
- `solution.state.m`

Use `solution.converged` as the first reality check. If it is `False`, the
state may still be useful for inspection, but the reported distance should be
treated cautiously.

## Good First Settings

For a first run:
- start with a small graph,
- use `num_steps` between `32` and `64`,
- leave `OTConfig()` at its defaults,
- and check `solution.converged`.

If you want a large-grid command that already records a trace, start with:

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

## Next Steps

- [Graph Model](graph-model.md) explains how to build the graph input correctly.
- [Examples Guide](examples-guide.md) points to runnable scripts and larger commands.
- [API Reference](api-reference.md) lists the main classes and result fields.
- [Debugging and Diagnostics](debugging-and-diagnostics.md) explains what to
  inspect when a run does not converge.
