from __future__ import annotations

import numpy as np

from graphot import GraphSpec, LogMeanOps, OTConfig, OTProblem, TimeDiscretization, solve_ot


def main() -> None:
    graph = GraphSpec.from_undirected_weights(2, [0], [1], [1.0])
    problem = OTProblem(
        graph=graph,
        time=TimeDiscretization(4),
        rho_a=np.array([2.0, 0.0], dtype=np.float64),
        rho_b=np.array([0.0, 2.0], dtype=np.float64),
        mean_ops=LogMeanOps(),
    )
    solution = solve_ot(problem, OTConfig(max_iters=2, check_every=1, cg_max_iters=16))
    assert np.isfinite(float(solution.distance))
    assert solution.iterations_used >= 1


if __name__ == "__main__":
    main()
