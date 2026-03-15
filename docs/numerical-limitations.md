# Numerical Limitations

`graphot` is useful today, but it is not a solver for every graph transport
setting.

## What It Handles Well

The current solver is a good fit for:

- sparse reversible graphs,
- small and medium examples,
- exploratory transport studies,
- runs where you want the full path, not just a static coupling.

## What It Does Not Handle

The current project does not target:

- nonreversible graphs,
- GPU execution,
- entropy-regularized OT,
- static coupling OT,
- very large production-scale graph problems by default.

## Where You Should Be Careful

Large graphs and sharp endpoint distributions can still be difficult.

Typical warning signs are:

- `solution.converged` stays `False`,
- `solution.distance` becomes non-finite,
- the debug trace shows `min_vartheta` collapsing toward zero.

This usually means the run is becoming numerically singular.

## Practical Advice For Harder Problems

If a larger problem is unstable:

- reduce `num_steps`,
- increase `max_iters`,
- increase `cg_max_iters`,
- smooth the endpoint masses,
- enable `record_debug_trace=True`.

It is better to get a smaller version of the problem converging first and only
then scale it up.

## Recommended Expectation

Treat large-grid runs as exploratory rather than guaranteed-default workloads.

For the cleanest first experience, start with the small examples shipped in the
repository.
