"""Public data containers and configuration types for the dynamic OT solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .means import MeanOps

Array = np.ndarray


@dataclass(frozen=True)
class GraphSpec:
    """Sparse reversible graph stored as a directed edge list.

    The solver stores graphs as paired directed edges rather than as a dense
    matrix. Every directed edge must have an explicit reverse edge recorded by
    ``rev``. Most users should construct graphs through
    :meth:`from_undirected_weights` or :meth:`from_directed_rates` instead of
    instantiating this dataclass manually.

    Attributes:
        num_nodes: Number of graph nodes.
        num_edges: Number of directed edges.
        src: Source node for each directed edge, shape ``(E,)``.
        dst: Destination node for each directed edge, shape ``(E,)``.
        rev: Reverse-edge index for each directed edge, shape ``(E,)``.
        q: Directed edge rates, shape ``(E,)``.
        pi: Stationary distribution, shape ``(X,)``.
        out_rate: Outgoing rate sum at each node, shape ``(X,)``.
    """

    num_nodes: int
    num_edges: int
    src: Array
    dst: Array
    rev: Array
    q: Array
    pi: Array
    out_rate: Array

    @classmethod
    def from_undirected_weights(
        cls,
        num_nodes: int,
        edge_u: Any,
        edge_v: Any,
        weight: Any,
        *,
        atol: float = 1e-12,
        rtol: float = 1e-10,
    ) -> "GraphSpec":
        """Build a sparse reversible graph from undirected conductances.

        Each entry ``(edge_u[k], edge_v[k], weight[k])`` defines an undirected
        conductance. The constructor expands the graph into paired directed
        edges, computes ``pi`` from the weighted-degree normalization of the
        associated unit-exit-rate random walk, and returns a graph that is
        reversible by construction.

        Args:
            num_nodes: Total number of nodes in the graph.
            edge_u: Source endpoint for each undirected edge.
            edge_v: Destination endpoint for each undirected edge.
            weight: Positive conductance assigned to each undirected edge.
            atol: Absolute tolerance used by the final invariant checks.
            rtol: Relative tolerance used by the final invariant checks.

        Returns:
            A validated :class:`GraphSpec` instance.
        """

        from .graph import build_graph_from_undirected_weights

        return build_graph_from_undirected_weights(
            num_nodes,
            edge_u,
            edge_v,
            weight,
            atol=atol,
            rtol=rtol,
        )

    @classmethod
    def from_directed_rates(
        cls,
        num_nodes: int,
        src: Any,
        dst: Any,
        q: Any,
        *,
        pi: Any | None = None,
        check_reversible: bool = True,
        atol: float = 1e-12,
        rtol: float = 1e-10,
        tol_ratio: float = 1e-10,
    ) -> "GraphSpec":
        """Build a sparse graph from directed rates ``Q(x, y)``.

        The inputs define a directed rate graph through ``src``, ``dst``, and
        ``q``. If ``pi`` is omitted, the constructor infers the stationary
        distribution under the reversibility assumption. Every positive edge
        must have an explicit reverse edge. When reversibility checks are
        enabled, nonreversible inputs raise ``ValueError``.
        """

        from .graph import build_graph_from_directed_rates

        return build_graph_from_directed_rates(
            num_nodes,
            src,
            dst,
            q,
            pi=pi,
            check_reversible=check_reversible,
            atol=atol,
            rtol=rtol,
            tol_ratio=tol_ratio,
        )


@dataclass(frozen=True)
class TimeDiscretization:
    """Time grid for the two-endpoint dynamic OT problem."""

    num_steps: int

    def __post_init__(self) -> None:
        if self.num_steps < 2:
            raise ValueError("num_steps must be at least 2")

    @property
    def h(self) -> float:
        """Return the uniform time step size ``1 / num_steps``."""

        return 1.0 / float(self.num_steps)


@dataclass(frozen=True)
class OTConfig:
    """Solver configuration for the PDHG-based dynamic OT solve.

    ``numerics_mode`` is retained only as a compatibility field and accepts
    only ``"paper"``. ``warm_start="harmonic_socp"`` uses a Python-side
    MOSEK harmonic-mean SOCP to build an initial state before the C++ log-mean
    PDHG iterations begin.
    """

    tau: float = 0.95
    sigma: float = 0.95
    relaxation: float = 1.0
    warm_start: str = "linear_path"
    max_iters: int = 400
    check_every: int = 10
    tol: float | None = None
    residual_tol: float = 1e-8
    feasibility_tol: float = 1e-8
    newton_iters: int = 12
    bisect_iters: int = 20
    cg_max_iters: int = 200
    cg_tol: float = 1e-10
    cg_warm_start: bool = True
    cg_preconditioner: str = "jacobi"
    numerics_mode: str = "paper"
    record_debug_trace: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.tau <= 0 or self.sigma <= 0:
            raise ValueError("tau and sigma must be positive")
        if self.tau * self.sigma >= 1.0:
            raise ValueError("tau * sigma must be strictly less than 1")
        if not 0.0 <= self.relaxation <= 1.0:
            raise ValueError("relaxation must lie in [0, 1]")
        if self.warm_start not in {"linear_path", "zero", "harmonic_socp"}:
            raise ValueError("warm_start must be 'linear_path', 'zero', or 'harmonic_socp'")
        if self.max_iters <= 0 or self.check_every <= 0:
            raise ValueError("max_iters and check_every must be positive")
        if self.tol is not None:
            object.__setattr__(self, "residual_tol", self.tol)
        if self.residual_tol <= 0 or self.feasibility_tol <= 0:
            raise ValueError("residual_tol and feasibility_tol must be positive")
        if self.newton_iters <= 0 or self.bisect_iters <= 0:
            raise ValueError("newton_iters and bisect_iters must be positive")
        if self.cg_max_iters <= 0:
            raise ValueError("cg_max_iters must be positive")
        if self.cg_preconditioner not in {"jacobi", "block_jacobi"}:
            raise ValueError("cg_preconditioner must be 'jacobi' or 'block_jacobi'")
        if self.numerics_mode != "paper":
            raise ValueError("legacy mode has been removed; use numerics_mode='paper'")


@dataclass(frozen=True)
class OTProblem:
    """Bundle the inputs for one dynamic OT solve."""

    graph: GraphSpec
    time: TimeDiscretization
    rho_a: Array
    rho_b: Array
    mean_ops: "MeanOps"


@dataclass(frozen=True)
class OTState:
    """Full split state returned by the solver."""

    rho: Array
    m: Array
    vartheta: Array
    rho_minus: Array
    rho_plus: Array
    rho_bar: Array
    q_node: Array

    def primal_norm(self) -> np.float64:
        """Compute the Euclidean norm across all split state blocks."""

        terms = (
            np.sum(self.rho**2)
            + np.sum(self.m**2)
            + np.sum(self.vartheta**2)
            + np.sum(self.rho_minus**2)
            + np.sum(self.rho_plus**2)
            + np.sum(self.rho_bar**2)
            + np.sum(self.q_node**2)
        )
        return np.sqrt(terms, dtype=np.float64)


@dataclass(frozen=True)
class OTDebugTrace:
    """Checkpointed diagnostic trace recorded during a solve."""

    iterations: Array
    action: Array
    continuity_residual: Array
    primal_delta: Array
    dual_delta: Array
    k_violation: Array
    endpoint_residual: Array
    max_constraint_residual: Array
    ceh_cg_residual: Array
    ceh_cg_iters: Array
    min_vartheta: Array
    num_records: int


@dataclass(frozen=True)
class OTSolution:
    """Result returned by :func:`graphot.solve_ot`."""

    distance: Any
    action: Any
    state: OTState
    iterations_used: int
    converged: bool
    diagnostics: dict[str, Any]
    debug_trace: OTDebugTrace | None = None


@dataclass(frozen=True)
class HarmonicWarmStartResult:
    """Harmonic-mean SOCP warm start and its compact diagnostics."""

    state: OTState
    objective: float
    continuity_residual: float
    endpoint_residual: float
    min_rho: float
    min_rho_bar: float
    solve_status: str
    export_path: str | None = None
