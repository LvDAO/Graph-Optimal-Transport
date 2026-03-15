# Graph Model

`graphot` works with sparse, connected, reversible graphs.

You usually build a graph in one of two ways:
- from undirected weights,
- from directed reversible rates.

## What A Valid Graph Looks Like

Every graph used by `graphot` must be:

- finite,
- connected,
- reversible,
- stored with an explicit reverse edge for every directed edge.

Each graph also carries a stationary distribution `pi`. Endpoint inputs are
densities with respect to that `pi`.

## Use `from_undirected_weights(...)` When Your Graph Is Symmetric

This is the easiest and most common path.

```python
from graphot import GraphSpec

graph = GraphSpec.from_undirected_weights(
    num_nodes=4,
    edge_u=[0, 1, 2],
    edge_v=[1, 2, 3],
    weight=[1.0, 2.0, 1.5],
)
```

Use this constructor when you naturally have:

- an undirected graph,
- a symmetric affinity graph,
- conductances or edge weights.

The constructor builds the reversible directed representation for you and
computes `graph.pi` automatically.

## Use `from_directed_rates(...)` When You Already Know The Rates

```python
from graphot import GraphSpec

graph = GraphSpec.from_directed_rates(
    num_nodes=3,
    src=[0, 1, 1, 2],
    dst=[1, 0, 2, 1],
    q=[2.0, 1.0, 1.0, 2.0],
)
```

Use this constructor when you already have a reversible directed rate model.

Important:

- every directed edge must have its reverse edge,
- the graph must still be connected,
- the rates must be reversible,
- `pi` can be supplied explicitly or inferred.

## Understanding `pi`

`graph.pi` is the stationary distribution associated with the graph.

`graphot` does not take raw endpoint masses directly. It takes densities relative to
`pi`.

If `mass` is an ordinary probability vector on the nodes, convert it with:

```python
rho = mass / graph.pi
```

Then the required normalization is:

```python
np.sum(graph.pi * rho) == 1
```

## Which Constructor Should You Pick?

- If your graph is symmetric, use `from_undirected_weights(...)`.
- If your graph is already a reversible directed rate model, use `from_directed_rates(...)`.
- If you are unsure, start with `from_undirected_weights(...)`.

## Practical Tip

Many input mistakes come from mixing up masses and densities. If the solver says
your endpoints are invalid, check the conversion to `rho = mass / graph.pi`
first.
