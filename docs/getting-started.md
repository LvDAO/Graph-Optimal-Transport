# Getting Started

This page shows the shortest path to a correct first solve.

## Install

Install from PyPI:

```bash
pip install graphot
```

If you want the plotting examples:

```bash
pip install "graphot[examples]"
```

If you are running from a local checkout, use `pip install .` or
`pip install ".[examples]"` from the repository root instead.

The public import path used in these docs is:

```python
from graphot import solve_ot
```

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

## The One Input Rule You Must Remember

`graphot` expects endpoint densities with respect to `graph.pi`.

If you start from ordinary node masses, convert them like this:

```python
rho = mass / graph.pi
```

Your endpoint arrays must satisfy:

```python
np.sum(graph.pi * rho) == 1
```

If this rule is wrong, the solver will reject the input or return a misleading
result.

## Reading The Result

The fields most users inspect are:

- `solution.distance`
- `solution.converged`
- `solution.iterations_used`
- `solution.state.rho`
- `solution.state.m`

Use `solution.converged` as a reality check. If it is `False`, the state may
still be useful for inspection, but the reported distance should be treated
with caution.

## Good First Settings

For a first run, the defaults are usually enough. If you want a safe starting
point:

- use a small graph,
- use `num_steps` between `32` and `64`,
- leave `OTConfig()` at its defaults,
- check `solution.converged` before doing further analysis.

## Next Steps

- [Graph Model](graph-model.md) explains how to build the graph input correctly.
- [API Reference](api-reference.md) lists the main classes and result fields.
- [Examples Guide](examples-guide.md) points you to ready-to-run scripts.
