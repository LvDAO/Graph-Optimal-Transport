# graphot Documentation

Welcome to the user guide for `graphot`.

`graphot` computes dynamic optimal transport on sparse reversible graphs. You give
it a graph, two endpoint distributions, and a time discretization. It returns a
transport path, edge flows, and a distance value.

## Read In This Order

1. [Getting Started](getting-started.md)
2. [Graph Model](graph-model.md)
3. [API Reference](api-reference.md)
4. [Examples Guide](examples-guide.md)
5. [Debugging and Diagnostics](debugging-and-diagnostics.md)
6. [Numerical Limitations](numerical-limitations.md)

## What Each Page Covers

- [Getting Started](getting-started.md): install, first solve, and the most important input rule.
- [Graph Model](graph-model.md): how to build a graph and when to use each constructor.
- [API Reference](api-reference.md): the main classes, functions, and result fields.
- [Examples Guide](examples-guide.md): what each example script is good for.
- [Solver Overview](solver-overview.md): a high-level view of what the solver computes.
- [Debugging and Diagnostics](debugging-and-diagnostics.md): what to check when a run does not converge.
- [Numerical Limitations](numerical-limitations.md): where the current solver is strong and where you should be cautious.
