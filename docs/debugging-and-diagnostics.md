# Debugging and Diagnostics

This page is for the common question: "Why did my solve not converge?"

## Start With These Checks

Before tuning solver parameters, check:

1. Are `rho_a` and `rho_b` densities with respect to `graph.pi`?
2. Do they satisfy `sum(graph.pi * rho) == 1`?
3. Is the graph connected and reversible?
4. Is the problem small enough to be a reasonable first test?

Many apparent solver failures are actually input-normalization problems.

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
- nonzero `endpoint_residual`: endpoint constraints are not settled,
- large `ceh_cg_residual`: the inner linear solve is too loose or too short.

## What `converged=False` Usually Means

It usually means one of these:
- the run hit `max_iters`,
- the problem is too sharp or too large for the current settings,
- the inner `CE_h` solve needs more work,
- the iterate became numerically singular.

`converged=False` does not always mean the returned state is useless, but it
does mean you should inspect the diagnostics before trusting the distance.

## Enabling The Debug Trace

```python
from graphot import OTConfig, solve_ot

solution = solve_ot(problem, OTConfig(record_debug_trace=True))
trace = solution.debug_trace
```

Useful trace fields:
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

If you also want live C++ progress while the solve runs, use `verbose=True`.
Progress prints use the same checkpoint interval as `check_every`.

For solves started from `initial_state`, the first valid trace record may be
iteration `0`. That record is the loaded warm-start state before the first PDHG
step. `solution.iterations_used` still excludes this initialization record.

## How To Read `action` And `min_vartheta`

`min_vartheta` reaching `0` is not automatically a crash.

The important distinction is:
- `vartheta = 0` and the corresponding flow is also `0`: this can still be a
  meaningful iterate,
- `vartheta = 0` while a corresponding edge flow remains nonzero: the primal
  action becomes non-finite.

So in practice:
- finite `action` means the current primal iterate is still a valid candidate
  for the original action minimization problem,
- `action = inf` means the iterate has become singular from the primal point of
  view, even if some residuals are still small.

## Common Signs Of An Unstable Run

Watch for:
- `action` becoming `inf`,
- `min_vartheta` collapsing toward `0`,
- `ceh_cg_iters` repeatedly hitting `cg_max_iters`,
- `continuity_residual` becoming tiny while the action is already non-finite.

That usually means the run is entering a singular region rather than simply
needing a few extra iterations.

## What To Try Next

Try these in roughly this order:

1. increase `cg_max_iters`,
2. increase `max_iters`,
3. reduce `num_steps`,
4. smooth the endpoint masses,
5. enable `record_debug_trace=True`,
6. compare `warm_start="linear_path"` against `warm_start="zero"` or a saved
   `initial_state`.

For hard large-grid runs, it is usually better to get a smaller or smoother
version converging first and only then scale up.

## Symptom To Knob Map

- `continuity_residual` stays large: increase `cg_max_iters`, then inspect the debug trace
- `primal_delta` and `dual_delta` stay large: increase `max_iters` or reduce `num_steps`
- `ceh_cg_iters` repeatedly equals `cg_max_iters`: raise `cg_max_iters`
- `action` becomes `inf`: inspect `min_vartheta`, compare warm starts, or reduce problem sharpness
- large-grid run is too slow from the start: lower `num_steps` or run one case instead of a sweep
