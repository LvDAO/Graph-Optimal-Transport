# API Reference

This page summarizes the public API most users need.

## Main Import

```python
from graphot import (
    GraphSpec,
    LogMeanOps,
    OTConfig,
    OTProblem,
    OTState,
    TimeDiscretization,
    solve_harmonic_socp_warm_start,
    solve_ot,
)
```

## `GraphSpec`

Represents the graph.

Most users should construct it with:
- `GraphSpec.from_undirected_weights(...)`
- `GraphSpec.from_directed_rates(...)`

Fields users often inspect:
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

This sets the temporal resolution of the path.

Practical guidance:
- smaller `num_steps`: cheaper and usually more stable,
- larger `num_steps`: finer path, more memory, harder optimization.

For a first run, `32` to `64` steps is usually a reasonable range.

## `LogMeanOps`

```python
LogMeanOps()
```

This is the mean model used by the current public solver.

In normal use, create it once and pass it into `OTProblem`.

## `OTConfig`

```python
OTConfig(...)
```

Controls the PDHG-based log-mean solve.

Fields most users care about first:
- `max_iters`
- `check_every`
- `cg_max_iters`
- `cg_tol`
- `warm_start`
- `record_debug_trace`
- `verbose`

Less commonly tuned fields:
- `tau`
- `sigma`
- `relaxation`
- `residual_tol`
- `feasibility_tol`
- `newton_iters`
- `cg_preconditioner`

The defaults are a reasonable starting point for small and medium problems.

### `warm_start`

`warm_start` accepts:
- `"linear_path"`: default linear warm start,
- `"zero"`: disable the built-in linear warm start,
- `"harmonic_socp"`: solve a harmonic-mean SOCP in Python with MOSEK and use
  the resulting state to initialize the log-mean solver.

The `"harmonic_socp"` mode requires:
- `mosek.fusion` to be importable,
- a valid MOSEK license,
- and is mainly intended for harder experiments rather than the smallest examples.

### Trace And Progress

- `record_debug_trace=True` stores checkpointed history in `solution.debug_trace`
- `verbose=True` prints C++ progress lines at the same checkpoint interval as
  `check_every`

### `tol`

If you pass `tol=...`, it becomes the residual stopping tolerance by setting
`residual_tol` internally.

## CPU Thread Control

`OTConfig` does not control CPU threading.

If the installed build includes OpenMP, the solver uses all available CPU
threads by default. Override that before Python starts:

```bash
GRAPHOT_NUM_THREADS=16 python your_script.py
```

or:

```bash
OMP_NUM_THREADS=16 python your_script.py
```

If the extension was built without OpenMP, solves stay single-threaded.

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

Bundles the inputs for one solve.

Requirements for `rho_a` and `rho_b`:
- shape `(num_nodes,)`
- finite
- nonnegative
- normalized so that `sum(graph.pi * rho) == 1`

## `solve_ot`

```python
solution = solve_ot(problem, config=OTConfig(), initial_state=None)
```

This is the main entry point.

Use `initial_state` when you want to warm-start from an existing state, for
example from:
- a previous nearby solve,
- a continuation scheme,
- or `solve_harmonic_socp_warm_start(...).state`.

If `initial_state` is provided, it takes precedence over `config.warm_start`.

## `solve_harmonic_socp_warm_start`

```python
result = solve_harmonic_socp_warm_start(problem, export_path=None)
```

This helper solves the harmonic-mean SOCP warm start directly and returns a
`HarmonicWarmStartResult`.

Use it when you want to:
- inspect the harmonic warm-start state,
- save the warm start to `.npz`,
- or pass `result.state` into `solve_ot(..., initial_state=result.state)`.

Useful fields on the result:
- `result.state`
- `result.objective`
- `result.continuity_residual`
- `result.endpoint_residual`
- `result.min_rho`
- `result.min_rho_bar`
- `result.solve_status`
- `result.export_path`

## `OTState`

`OTState` is the full split primal state.

Fields most users inspect:
- `state.rho`
- `state.m`

The remaining fields are the solver’s split variables:
- `state.vartheta`
- `state.rho_minus`
- `state.rho_plus`
- `state.rho_bar`
- `state.q_node`

If you only want the physical path, start with `rho` and `m`.

### Array Shapes And Units

For a problem with `num_steps`, `num_nodes`, and `num_edges`:
- `state.rho` has shape `(num_steps + 1, num_nodes)`
- `state.m` has shape `(num_steps, num_edges)`

`state.rho` stores densities with respect to `graph.pi`, not ordinary masses.
If you want the node mass at one time slice, convert with:

```python
mass_t = graph.pi * state.rho[t]
```

## `OTSolution`

Fields most users inspect first:
- `solution.distance`
- `solution.converged`
- `solution.iterations_used`
- `solution.state`
- `solution.diagnostics`
- `solution.debug_trace`

### `solution.diagnostics`

This dictionary stores compact final-checkpoint diagnostics.

The most useful keys are:
- `continuity_residual`
- `primal_delta`
- `dual_delta`
- `max_constraint_residual`
- `endpoint_residual`
- `ceh_cg_residual`
- `ceh_cg_iters`

## `OTDebugTrace`

Available when `record_debug_trace=True`.

Useful fields:
- `trace.iterations`
- `trace.action`
- `trace.continuity_residual`
- `trace.primal_delta`
- `trace.dual_delta`
- `trace.k_violation`
- `trace.endpoint_residual`
- `trace.max_constraint_residual`
- `trace.ceh_cg_residual`
- `trace.ceh_cg_iters`
- `trace.min_vartheta`

Only the first `trace.num_records` entries are valid.

For solves started from `initial_state`, the first valid trace record may be
iteration `0`. That record is the loaded warm-start state before the first PDHG
step. `solution.iterations_used` still counts only PDHG iterations.
