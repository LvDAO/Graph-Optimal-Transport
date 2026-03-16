# graphot Documentation

This documentation is written for users who want to solve dynamic optimal
transport problems on sparse reversible graphs from Python.

If you are new to the project, the fastest path is:

1. [Getting Started](getting-started.md)
2. [Graph Model](graph-model.md)
3. [Examples Guide](examples-guide.md)
4. [API Reference](api-reference.md)

If you are already running larger problems, these pages matter next:

5. [Debugging and Diagnostics](debugging-and-diagnostics.md)
6. [Numerical Limitations](numerical-limitations.md)

## Choose A Starting Point

Read [Getting Started](getting-started.md) if you want:
- installation instructions,
- a first complete solve,
- the endpoint normalization rule,
- a safe first large-grid command.

Read [Graph Model](graph-model.md) if you want:
- to build graphs correctly,
- to understand `graph.pi`,
- to decide between `from_undirected_weights(...)` and
  `from_directed_rates(...)`.

Read [Examples Guide](examples-guide.md) if you want:
- runnable scripts,
- recommended example order,
- large-grid commands,
- trace and warm-start example usage.

Read [API Reference](api-reference.md) if you want:
- the public import surface,
- the main config fields,
- warm-start modes,
- result and trace fields.

Read [Debugging and Diagnostics](debugging-and-diagnostics.md) if:
- `converged` is `False`,
- `action` becomes non-finite,
- the trace starts looking unstable,
- or you are tuning large-grid runs.

Read [Numerical Limitations](numerical-limitations.md) if:
- you are planning larger experiments,
- you want to know where the current solver is reliable,
- or you need realistic expectations before a longer run.

## Current Scope

`graphot` currently focuses on:
- sparse reversible graphs,
- CPU solves from Python,
- NumPy-friendly outputs,
- and explicit transport paths over time.

Large-grid runs are supported as exploratory workloads, but they are still the
hardest part of the current package. Use the diagnostics and examples when you
scale up.
