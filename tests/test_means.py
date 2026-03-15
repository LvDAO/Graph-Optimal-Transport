from __future__ import annotations

import numpy as np

from graphot import LogMeanOps


def _central_difference(fn, x: float, *, eps: float) -> float:
    return (float(fn(x + eps)) - float(fn(x - eps))) / (2.0 * eps)


def test_log_mean_defaults_to_float64() -> None:
    mean = LogMeanOps()
    value = np.asarray(mean.theta(0.7, 1.3))
    assert value.dtype == np.float64


def test_log_mean_is_symmetric_and_stable_near_diagonal() -> None:
    mean = LogMeanOps()
    s = np.array([1.0, 1.0])
    t = np.array([2.0, 1.0 + 1e-9])
    val1 = np.asarray(mean.theta(s, t))
    val2 = np.asarray(mean.theta(t, s))
    np.testing.assert_allclose(val1, val2)
    np.testing.assert_allclose(val1[1], 1.0 + 0.5e-9, atol=1e-9)


def test_log_mean_derivatives_match_finite_difference() -> None:
    mean = LogMeanOps()
    s = 0.7
    t = 1.3
    eps = 1e-6
    fd_s = (float(mean.theta(s + eps, t)) - float(mean.theta(s - eps, t))) / (2 * eps)
    fd_t = (float(mean.theta(s, t + eps)) - float(mean.theta(s, t - eps))) / (2 * eps)
    assert abs(float(mean.dtheta_ds(s, t)) - fd_s) < 5e-5
    assert abs(float(mean.dtheta_dt(s, t)) - fd_t) < 5e-5


def test_origin_supergradient_membership() -> None:
    mean = LogMeanOps()
    assert bool(mean.origin_supergrad_contains(0.6, 0.5))
    assert not bool(mean.origin_supergrad_contains(0.4, 0.4))
    assert not bool(mean.origin_supergrad_contains(-1.0, 1.0))


def test_project_k_top_lands_on_log_mean_surface() -> None:
    mean = LogMeanOps()
    q1, q2, q3 = map(float, mean.project_k_top(1.4, 0.2, 1.0))
    assert q1 >= 0.0
    assert q2 >= 0.0
    assert q3 >= 0.0
    assert abs(q3 - float(mean.theta(q1, q2))) < 1e-8

def test_beta_derivative_matches_central_difference() -> None:
    mean = LogMeanOps()
    for x, eps in ((1e-4, 1e-7), (0.5, 1e-6)):
        actual = float(mean._beta_derivative(x))
        expected = _central_difference(mean._beta, x, eps=eps)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_top_surface_residual_derivative_matches_central_difference() -> None:
    mean = LogMeanOps()
    p = np.array([0.7, 1.3, 0.9], dtype=np.float64)
    for x, eps in ((1e-4, 1e-7), (0.5, 1e-6)):
        actual = float(mean._top_surface_residual_and_deriv(p, x)[1])
        expected = _central_difference(
            lambda arg: mean._top_surface_residual(p, arg),
            x,
            eps=eps,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)
