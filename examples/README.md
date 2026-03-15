# Examples

This folder contains runnable examples for `graphot`.

Install the package with plotting dependencies:

```bash
pip install "graphot[examples]"
```

If you are running from a local checkout, `pip install ".[examples]"` from the
repository root works too.

## Included Scripts

- `two_node_benchmark/`: the smallest benchmark and best first example.
- `cycle_neighbor_transport/`: transport on 3-cycle and 4-cycle graphs.
- `line_chain_transport/`: transport along a simple path graph.
- `directed_reversible_transport/`: a directed but reversible graph example.
- `large_grid_transport/`: a larger grid example for more demanding runs.

## Run

```bash
python examples/two_node_benchmark/run.py
python examples/cycle_neighbor_transport/run.py
python examples/line_chain_transport/run.py
python examples/directed_reversible_transport/run.py
python examples/large_grid_transport/run.py
```

## Output Files

Most scripts save:

- a state `.npz` file,
- a node plot,
- an edge-flow plot,
- a graph snapshot.

For more detail on what each example shows, see
[`docs/examples-guide.md`](../docs/examples-guide.md).
