"""Graph validation and :class:`graphot.GraphSpec` construction."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .types import GraphSpec


def _as_int_array(values: Iterable[int]) -> np.ndarray:
    """Coerce a one-dimensional integer-like input into a NumPy array."""

    arr = np.asarray(values, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("expected a one-dimensional integer array")
    return arr


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    """Coerce a one-dimensional float-like input into a NumPy array."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("expected a one-dimensional float array")
    return arr


def _build_reverse_edge_map(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Build the reverse-edge lookup and enforce paired directed edges."""

    edge_index: dict[tuple[int, int], int] = {}
    for idx, (x, y) in enumerate(zip(src.tolist(), dst.tolist(), strict=True)):
        if x == y:
            raise ValueError("self-loops are not supported")
        key = (x, y)
        if key in edge_index:
            raise ValueError(f"duplicate directed edge {key}")
        edge_index[key] = idx

    rev = np.empty_like(src)
    for idx, (x, y) in enumerate(zip(src.tolist(), dst.tolist(), strict=True)):
        key = (y, x)
        if key not in edge_index:
            raise ValueError(f"missing reverse edge for {(x, y)}")
        rev[idx] = edge_index[key]

    if not np.all(rev[rev] == np.arange(src.size, dtype=rev.dtype)):
        raise ValueError("reverse-edge mapping is not involutive")
    return rev


def _check_connected(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> None:
    """Ensure the undirected graph support is connected."""

    if src.size == 0:
        raise ValueError("graph must have at least one edge")

    support = sparse.coo_matrix(
        (np.ones(src.size, dtype=np.float64), (src, dst)),
        shape=(num_nodes, num_nodes),
    )
    num_components, _ = csgraph.connected_components(support, directed=False)
    if num_components != 1:
        raise ValueError("graph must be connected")


def _build_bfs_tree_edge_order(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Build a root-0 BFS tree as a fixed directed-edge order."""

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_idx, x in enumerate(src.tolist()):
        adjacency[x].append(edge_idx)

    seen = np.zeros(num_nodes, dtype=bool)
    seen[0] = True
    queue: deque[int] = deque([0])
    tree_edge_order = np.empty(num_nodes - 1, dtype=np.int32)
    num_tree_edges = 0

    while queue:
        node = queue.popleft()
        for edge_idx in adjacency[node]:
            nbr = int(dst[edge_idx])
            if not seen[nbr]:
                seen[nbr] = True
                tree_edge_order[num_tree_edges] = edge_idx
                num_tree_edges += 1
                queue.append(nbr)

    if num_tree_edges != num_nodes - 1 or not np.all(seen):
        raise ValueError("graph must be connected")
    return tree_edge_order


def _normalize_pi(pi: np.ndarray) -> np.ndarray:
    """Normalize a positive stationary distribution candidate to unit mass."""

    if pi.ndim != 1:
        raise ValueError("pi must be one-dimensional")
    if np.any(pi <= 0):
        raise ValueError("pi must be strictly positive")
    total = float(np.sum(pi))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("pi must have positive finite mass")
    return pi / total


def _out_rate(num_nodes: int, src: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the outgoing rate sum at each node."""

    out = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(out, src, q)
    return out


def _validate_stationarity(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    q: np.ndarray,
    pi: np.ndarray,
    out_rate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Validate that ``pi`` is stationary for the supplied directed rates."""

    inflow = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(inflow, dst, pi[src] * q)
    residual = inflow - pi * out_rate
    scale = max(1.0, float(np.max(np.abs(pi[src] * q), initial=0.0)))
    tol = atol + rtol * scale
    if float(np.max(np.abs(residual), initial=0.0)) > tol:
        raise ValueError("pi is not stationary for the supplied rates")


def _validate_reversibility(
    src: np.ndarray,
    dst: np.ndarray,
    rev: np.ndarray,
    q: np.ndarray,
    pi: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Validate the detailed-balance condition edge by edge."""

    edge_residual = pi[src] * q - pi[dst] * q[rev]
    scale = max(1.0, float(np.max(np.abs(pi[src] * q), initial=0.0)))
    tol = atol + rtol * scale
    if float(np.max(np.abs(edge_residual), initial=0.0)) > tol:
        raise ValueError("graph is not reversible under the supplied stationary distribution")


def _infer_pi_from_reversible_rates_kernel(
    src: np.ndarray,
    dst: np.ndarray,
    rev: np.ndarray,
    q: np.ndarray,
    tree_edge_order: np.ndarray,
) -> tuple[np.ndarray, np.float64]:
    """Infer ``pi`` numerically from a fixed BFS tree order."""

    src = np.asarray(src, dtype=np.int32)
    dst = np.asarray(dst, dtype=np.int32)
    rev = np.asarray(rev, dtype=np.int32)
    q = np.asarray(q, dtype=np.float64)
    tree_edge_order = np.asarray(tree_edge_order, dtype=np.int32)

    num_nodes = tree_edge_order.shape[0] + 1
    log_pi = np.zeros((num_nodes,), dtype=np.float64)
    log_q = np.log(q)
    for edge_idx in tree_edge_order.tolist():
        x = int(src[edge_idx])
        y = int(dst[edge_idx])
        log_pi[y] = log_pi[x] + log_q[edge_idx] - log_q[rev[edge_idx]]
    shifted = log_pi - np.max(log_pi)
    weights = np.exp(shifted)
    pi = weights / np.sum(weights)
    edge_residual = (log_pi[dst] - log_pi[src]) - (log_q - log_q[rev])
    max_abs_edge_residual = np.float64(np.max(np.abs(edge_residual)))
    return pi, max_abs_edge_residual


def _infer_pi_from_reversible_rates_kernel_jit(
    src: np.ndarray,
    dst: np.ndarray,
    rev: np.ndarray,
    q: np.ndarray,
    tree_edge_order: np.ndarray,
) -> tuple[np.ndarray, np.float64]:
    """Compatibility wrapper kept for tests after JAX removal."""

    return _infer_pi_from_reversible_rates_kernel(src, dst, rev, q, tree_edge_order)


def _infer_pi_from_reversible_rates(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    rev: np.ndarray,
    q: np.ndarray,
    *,
    tol_ratio: float,
) -> np.ndarray:
    """Infer ``pi`` from reversible directed rates."""

    tree_edge_order = _build_bfs_tree_edge_order(num_nodes, src, dst)
    pi, max_abs_edge_residual = _infer_pi_from_reversible_rates_kernel(
        src,
        dst,
        rev,
        q,
        tree_edge_order,
    )
    if float(max_abs_edge_residual) > tol_ratio:
        raise ValueError("cycle inconsistency detected while inferring stationary distribution")
    return _normalize_pi(np.asarray(pi, dtype=np.float64))


def _finalize_graph(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    rev: np.ndarray,
    q: np.ndarray,
    pi: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> GraphSpec:
    """Run final graph invariants and convert validated arrays into ``GraphSpec``."""

    out_rate = _out_rate(num_nodes, src, q)
    _validate_reversibility(src, dst, rev, q, pi, atol=atol, rtol=rtol)
    _validate_stationarity(num_nodes, src, dst, q, pi, out_rate, atol=atol, rtol=rtol)
    return GraphSpec(
        num_nodes=num_nodes,
        num_edges=int(src.size),
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        rev=np.asarray(rev, dtype=np.int32),
        q=np.asarray(q, dtype=np.float64),
        pi=np.asarray(pi, dtype=np.float64),
        out_rate=np.asarray(out_rate, dtype=np.float64),
    )


def build_graph_from_undirected_weights(
    num_nodes: int,
    edge_u: Iterable[int],
    edge_v: Iterable[int],
    weight: Iterable[float],
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> GraphSpec:
    """Construct a validated graph from undirected conductances."""

    u = _as_int_array(edge_u)
    v = _as_int_array(edge_v)
    w = _as_float_array(weight)
    if not (u.size == v.size == w.size):
        raise ValueError("edge_u, edge_v, and weight must have the same length")
    if num_nodes <= 1:
        raise ValueError("num_nodes must be at least 2")
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive")
    if np.any((u < 0) | (u >= num_nodes) | (v < 0) | (v >= num_nodes)):
        raise ValueError("edge endpoints are out of range")
    if np.any(u == v):
        raise ValueError("self-loops are not supported")

    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    _check_connected(num_nodes, src, dst)
    degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(degree, u, w)
    np.add.at(degree, v, w)
    pi = _normalize_pi(degree)
    q = np.concatenate([w / degree[u], w / degree[v]])
    rev = _build_reverse_edge_map(src, dst)
    return _finalize_graph(num_nodes, src, dst, rev, q, pi, atol=atol, rtol=rtol)


def build_graph_from_directed_rates(
    num_nodes: int,
    src: Iterable[int],
    dst: Iterable[int],
    q: Iterable[float],
    *,
    pi: Iterable[float] | None = None,
    check_reversible: bool = True,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    tol_ratio: float = 1e-10,
) -> GraphSpec:
    """Construct a graph from directed rates and an optional stationary law."""

    src_arr = _as_int_array(src)
    dst_arr = _as_int_array(dst)
    q_arr = _as_float_array(q)
    if not (src_arr.size == dst_arr.size == q_arr.size):
        raise ValueError("src, dst, and q must have the same length")
    if num_nodes <= 1:
        raise ValueError("num_nodes must be at least 2")
    if np.any((src_arr < 0) | (src_arr >= num_nodes) | (dst_arr < 0) | (dst_arr >= num_nodes)):
        raise ValueError("edge endpoints are out of range")
    if not np.all(np.isfinite(q_arr)):
        raise ValueError("rates must be finite")
    if np.any(q_arr <= 0):
        raise ValueError("rates must be strictly positive")

    rev = _build_reverse_edge_map(src_arr, dst_arr)
    _check_connected(num_nodes, src_arr, dst_arr)

    if pi is None:
        if not check_reversible:
            raise ValueError("pi must be supplied when check_reversible is False")
        pi_arr = _infer_pi_from_reversible_rates(
            num_nodes,
            src_arr,
            dst_arr,
            rev,
            q_arr,
            tol_ratio=tol_ratio,
        )
    else:
        pi_arr = _normalize_pi(_as_float_array(pi))
        if pi_arr.size != num_nodes:
            raise ValueError("pi must have length num_nodes")

    if check_reversible:
        return _finalize_graph(
            num_nodes,
            src_arr,
            dst_arr,
            rev,
            q_arr,
            pi_arr,
            atol=atol,
            rtol=rtol,
        )

    out_rate = _out_rate(num_nodes, src_arr, q_arr)
    return GraphSpec(
        num_nodes=num_nodes,
        num_edges=int(src_arr.size),
        src=np.asarray(src_arr, dtype=np.int32),
        dst=np.asarray(dst_arr, dtype=np.int32),
        rev=np.asarray(rev, dtype=np.int32),
        q=np.asarray(q_arr, dtype=np.float64),
        pi=np.asarray(pi_arr, dtype=np.float64),
        out_rate=np.asarray(out_rate, dtype=np.float64),
    )
