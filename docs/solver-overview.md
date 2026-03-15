# Solver Overview

This page explains, at a high level, what `graphot` computes.

## What The Solver Does

`graphot` solves a two-endpoint dynamic transport problem on a graph.

You provide:

- a graph,
- a starting distribution,
- an ending distribution,
- a number of time steps.

The solver returns:

- a sequence of node densities over time,
- a sequence of edge fluxes over time,
- a distance value summarizing the transport cost.

## What Changes When You Increase `num_steps`

`num_steps` controls the time resolution of the transport path.

- fewer steps: faster and cheaper,
- more steps: finer path, but more work and more memory.

If you are exploring a new problem, start small and increase the number of
steps only when you need a finer path.

## What A Solve Produces

The main result lives in:

- `solution.state.rho`
- `solution.state.m`

Think of them as:

- `rho`: how mass is distributed over nodes at each time slice,
- `m`: how mass moves along edges between slices.

## What A Successful Run Looks Like

On a stable problem, you should expect:

- `solution.converged == True`,
- a finite `solution.distance`,
- small residuals in `solution.diagnostics`.

If `converged` is `False`, the result may still be useful for inspection, but
you should not treat it as a fully settled answer.

## What Makes A Problem Harder

Problems usually become harder when you increase:

- the number of nodes,
- the number of edges,
- the number of time steps,
- the sharpness of the endpoint distributions.

Large grid problems are often much harder than small toy graphs.

## Practical Advice

If you are new to the solver:

1. start with a small graph,
2. use `OTConfig()` defaults,
3. check `solution.converged`,
4. only then scale up the graph or time resolution.
