# Numerical Limitations

`graphot` is usable today, but it is still an early-stage scientific solver.

This page is here to set expectations clearly before a long run.

## What It Handles Best

The current package is a good fit for:
- sparse reversible graphs,
- small and medium examples,
- research and exploratory workflows,
- runs where you want the full time-dependent path.

## What It Does Not Target

The current public solver does not target:
- nonreversible graphs,
- GPU execution,
- entropy-regularized OT,
- static coupling OT,
- or very large production workloads with default settings.

## Where You Should Be Careful

The hardest problems today are usually:
- larger grids,
- many time steps,
- very sharp endpoint masses,
- runs whose debug trace drives `min_vartheta` toward zero.

Typical warning signs are:
- `solution.converged` stays `False`,
- `solution.distance` becomes non-finite,
- `solution.debug_trace.action` becomes `inf`,
- `ceh_cg_iters` repeatedly hits `cg_max_iters`.

## Large-Grid Runs

Large-grid runs are supported as exploratory examples, not as guaranteed
out-of-the-box workloads.

In practice, you should expect to tune:
- `max_iters`,
- `cg_max_iters`,
- `num_steps`,
- and sometimes the warm-start strategy.

For release-facing user expectations, the safest message is:
- small examples should be the first experience,
- large-grid experiments are possible,
- but they are the most numerically demanding part of the package.

## Harmonic Warm Start

`warm_start="harmonic_socp"` can be useful on harder problems, but it is not a
guaranteed fix for large-grid instability.

It also requires:
- `mosek.fusion`,
- and a valid MOSEK license.

Use it as an optional advanced tool, not as the baseline path for first-time
users.

## Practical Advice

If a problem is unstable:

1. reduce `num_steps`,
2. increase `cg_max_iters`,
3. increase `max_iters`,
4. smooth the endpoint masses,
5. enable a debug trace,
6. compare warm-start modes on the same case.

It is usually better to get a smaller or smoother version of the same problem
to converge cleanly before scaling up.

## Scaling Intuition

Runtime and memory both grow quickly as you increase:
- `num_steps`
- `num_nodes`
- `num_edges`

In practical terms:
- doubling `num_steps` makes both the path state and the inner linear work noticeably heavier,
- larger sparse graphs are usually manageable only when you keep `num_steps` conservative,
- the large-grid example is best treated as a tuning workflow, not as a default one-shot solve.
