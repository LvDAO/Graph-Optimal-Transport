from __future__ import annotations

import graphot


def test_graphot_exports_public_api() -> None:
    assert callable(graphot.solve_ot)
    assert callable(graphot.solve_harmonic_socp_warm_start)
    assert graphot.GraphSpec.__name__ == "GraphSpec"
    assert graphot.HarmonicWarmStartResult.__name__ == "HarmonicWarmStartResult"
    assert graphot.OTConfig.__name__ == "OTConfig"
