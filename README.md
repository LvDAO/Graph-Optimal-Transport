# graphot

`graphot` solves dynamic optimal transport on sparse reversible graphs.

Use it when you want:
- a transport path between two endpoint distributions on a graph,
- node densities over time,
- edge fluxes over time,
- a solver that runs from Python and returns NumPy arrays.

## Install

Install from PyPI:

```bash
pip install graphot
```

If you want to run the plotting examples:

```bash
pip install "graphot[examples]"
```

If you are working from a local checkout instead, run `pip install .` or
`pip install ".[examples]"` from the repository root.

These docs use the public import path `graphot`:

```python
from graphot import solve_ot
```

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

## Three Things To Remember

- `graphot` expects endpoint densities with respect to `graph.pi`, not raw masses.
- Convert ordinary masses with `rho = mass / graph.pi`.
- Check `solution.converged` on harder problems before treating the result as final.

## What You Get Back

The most useful outputs are:
- `solution.distance`
- `solution.state.rho`
- `solution.state.m`
- `solution.diagnostics`

## Start Here

- [Getting Started](docs/getting-started.md)
- [Graph Model](docs/graph-model.md)
- [API Reference](docs/api-reference.md)
- [Examples Guide](docs/examples-guide.md)
- [Debugging and Diagnostics](docs/debugging-and-diagnostics.md)
- [Numerical Limitations](docs/numerical-limitations.md)

## Examples

The repository includes ready-to-run scripts for:
- two-node transport,
- cycle transport,
- line transport,
- directed reversible transport,
- large-grid transport.

See [examples/README.md](examples/README.md) for the example index.
