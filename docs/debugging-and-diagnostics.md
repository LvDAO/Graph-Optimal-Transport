# Debugging and Diagnostics

This page is for the common question: "Why did my solve not converge?"

## Start With These Checks

Before tuning anything, check:

1. Are `rho_a` and `rho_b` densities with respect to `graph.pi`?
2. Do they satisfy `sum(graph.pi * rho) == 1`?
3. Is the graph connected and reversible?
4. Is the problem small enough to be a reasonable first test?

Most early failures come from input normalization, not from the solver itself.

## The Main Status Fields

Look at `solution.diagnostics`.

The most useful entries are:

- `continuity_residual`
- `primal_delta`
- `dual_delta`
- `max_constraint_residual`
- `endpoint_residual`
- `ceh_cg_residual`
- `ceh_cg_iters`

Rough interpretation:

- large `continuity_residual`: mass conservation is still off,
- large `primal_delta` or `dual_delta`: the solver is still moving,
- nonzero `endpoint_residual`: the endpoint constraints are not yet settled,
- large `ceh_cg_residual`: the inner linear solve is too loose.

## What `converged=False` Usually Means

It usually means one of these:

- the run hit `max_iters`,
- the problem is too sharp or too large for the current settings,
- the inner solve needs more work,
- the iterate became numerically unstable.

`converged=False` does not always mean the returned state is useless, but it
does mean you should inspect the diagnostics before trusting the distance.

## Enabling The Debug Trace

```python
from graphot import OTConfig, solve_ot

solution = solve_ot(problem, OTConfig(record_debug_trace=True))
trace = solution.debug_trace
```

This gives you checkpointed history over the run.

Useful trace fields:

- `trace.iterations`
- `trace.action`
- `trace.continuity_residual`
- `trace.primal_delta`
- `trace.dual_delta`
- `trace.min_vartheta`

Only the first `trace.num_records` entries are valid.

## Signs Of An Unstable Run

Common warning signs are:

- `action` becomes `inf`,
- `min_vartheta` keeps shrinking toward `0`,
- `continuity_residual` is small but `converged` is still `False`.

That usually means the run is becoming singular rather than simply needing a
few extra iterations.

## What To Try Next

Try these in order:

1. increase `cg_max_iters`,
2. increase `max_iters`,
3. reduce `num_steps`,
4. use smoother endpoint masses,
5. enable `record_debug_trace=True`.

If you are testing on a large grid, step back to a smaller example first and
make sure that converges cleanly.
