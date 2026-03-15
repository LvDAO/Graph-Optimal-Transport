# API Reference

This page summarizes the public API most users need.

## Main Import

```python
from graphot import (
    GraphSpec,
    LogMeanOps,
    OTConfig,
    OTProblem,
    TimeDiscretization,
    solve_ot,
)
```

## `GraphSpec`

Represents the graph.

Most users should build it with one of these constructors:

- `GraphSpec.from_undirected_weights(...)`
- `GraphSpec.from_directed_rates(...)`

Useful fields:

- `graph.num_nodes`
- `graph.num_edges`
- `graph.pi`
- `graph.src`
- `graph.dst`
- `graph.q`

Use `graph.pi` when converting ordinary masses to solver densities.

## `TimeDiscretization`

```python
TimeDiscretization(num_steps: int)
```

This sets how finely the transport path is split in time.

- smaller `num_steps` is cheaper,
- larger `num_steps` gives a finer path.

For a first run, `32` to `64` steps is usually a sensible range.

## `LogMeanOps`

```python
LogMeanOps()
```

This is the mean model used by the current solver. In normal use, create it and
pass it into `OTProblem`.

## `OTConfig`

```python
OTConfig(...)
```

Controls solver behavior.

Most users only need these fields:

- `max_iters`
- `check_every`
- `cg_max_iters`
- `record_debug_trace`

The defaults are a good starting point. Tune further only if a problem is slow
or does not converge.

## `OTProblem`

```python
OTProblem(
    graph=graph,
    time=TimeDiscretization(...),
    rho_a=rho_a,
    rho_b=rho_b,
    mean_ops=LogMeanOps(),
)
```

Bundles all inputs for one solve.

Requirements for `rho_a` and `rho_b`:

- shape `(num_nodes,)`,
- finite,
- nonnegative,
- normalized so that `sum(graph.pi * rho) == 1`.

## `solve_ot`

```python
solution = solve_ot(problem, config=OTConfig())
```

This is the main entry point.

It returns an `OTSolution`.

## `OTSolution`

The fields most users inspect are:

- `solution.distance`
- `solution.converged`
- `solution.iterations_used`
- `solution.state`
- `solution.diagnostics`
- `solution.debug_trace`

### `solution.state`

The state contains the full time-dependent result.

Most users look at:

- `solution.state.rho`
- `solution.state.m`

Both are NumPy arrays.

### `solution.diagnostics`

This dictionary contains compact status values from the final checkpoint.

The most useful keys are:

- `continuity_residual`
- `primal_delta`
- `dual_delta`
- `max_constraint_residual`
- `endpoint_residual`

If a solve is slow or unstable, look here first.

## `OTDebugTrace`

Available when `record_debug_trace=True`.

Useful fields:

- `trace.iterations`
- `trace.action`
- `trace.continuity_residual`
- `trace.primal_delta`
- `trace.dual_delta`
- `trace.min_vartheta`
- `trace.num_records`

Only the first `trace.num_records` entries are valid.

## Related Pages

- [Getting Started](getting-started.md)
- [Graph Model](graph-model.md)
- [Debugging and Diagnostics](debugging-and-diagnostics.md)
- [Examples Guide](examples-guide.md)
