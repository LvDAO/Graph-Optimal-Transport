"""Averaging functions used by the transport action and ``K`` projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

Array = np.ndarray


class MeanOps(Protocol):
    """Abstract interface required by the solver's mean-dependent operations."""

    def theta(self, s: Array | float, t: Array | float) -> Array | np.float64:
        ...

    def dtheta_ds(self, s: Array | float, t: Array | float) -> Array | np.float64:
        ...

    def dtheta_dt(self, s: Array | float, t: Array | float) -> Array | np.float64:
        ...

    def origin_supergrad_contains(self, z1: Array | float, z2: Array | float) -> Array | np.bool_:
        ...

    def project_k_top(
        self,
        p1: Array | float,
        p2: Array | float,
        p3: Array | float,
    ) -> tuple[Array | np.float64, Array | np.float64, Array | np.float64]:
        ...


def _to_numpy(*values: Array | float) -> list[Array]:
    dtype = np.result_type(*values, np.float64)
    return [np.asarray(value, dtype=dtype) for value in values]


@dataclass(frozen=True)
class LogMeanOps:
    """Logarithmic mean and its helper operations for the current solver."""

    eps_diag: float = 1e-6
    xi_max: float = 18.0
    newton_iters: int = 3
    bisect_iters: int = 24

    def _small_xi_mask(self, xi: Array | float) -> Array:
        xi_arr = np.asarray(xi, dtype=np.float64)
        return np.abs(xi_arr) <= np.sqrt(self.eps_diag)

    def theta(self, s: Array | float, t: Array | float) -> Array | np.float64:
        s_arr, t_arr = _to_numpy(s, t)
        positive = (s_arr > 0) & (t_arr > 0)
        close = positive & (np.abs(t_arr - s_arr) <= self.eps_diag * np.maximum(s_arr, t_arr))
        m = 0.5 * (s_arr + t_arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (t_arr - s_arr) / np.maximum(s_arr + t_arr, np.finfo(m.dtype).tiny)
            series = m * (1.0 - (u * u) / 3.0)
            raw = (t_arr - s_arr) / (np.log(t_arr) - np.log(s_arr))
        out = np.where(close, series, raw)
        out = np.where(positive, out, 0.0)
        out = np.where((s_arr == t_arr) & (s_arr >= 0), s_arr, out)
        return out

    def dtheta_ds(self, s: Array | float, t: Array | float) -> Array | np.float64:
        s_arr, t_arr = _to_numpy(s, t)
        positive = (s_arr > 0) & (t_arr > 0)
        close = positive & (np.abs(t_arr - s_arr) <= self.eps_diag * np.maximum(s_arr, t_arr))
        m = 0.5 * (s_arr + t_arr)
        d = t_arr - s_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            series = (
                0.5
                + d / np.maximum(6.0 * m, np.finfo(m.dtype).tiny)
                + (d * d) / np.maximum(24.0 * m * m, np.finfo(m.dtype).tiny)
            )
            lr = np.log(t_arr) - np.log(s_arr)
            raw = (((t_arr - s_arr) / s_arr) - lr) / (lr * lr)
        out = np.where(close, series, raw)
        out = np.where(positive, out, 0.0)
        out = np.where((s_arr == t_arr) & (s_arr > 0), 0.5, out)
        return out

    def dtheta_dt(self, s: Array | float, t: Array | float) -> Array | np.float64:
        s_arr, t_arr = _to_numpy(s, t)
        positive = (s_arr > 0) & (t_arr > 0)
        close = positive & (np.abs(t_arr - s_arr) <= self.eps_diag * np.maximum(s_arr, t_arr))
        m = 0.5 * (s_arr + t_arr)
        d = t_arr - s_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            series = (
                0.5
                - d / np.maximum(6.0 * m, np.finfo(m.dtype).tiny)
                + (d * d) / np.maximum(24.0 * m * m, np.finfo(m.dtype).tiny)
            )
            lr = np.log(t_arr) - np.log(s_arr)
            raw = (lr - (t_arr - s_arr) / t_arr) / (lr * lr)
        out = np.where(close, series, raw)
        out = np.where(positive, out, 0.0)
        out = np.where((s_arr == t_arr) & (s_arr > 0), 0.5, out)
        return out

    def _beta(self, xi: Array | float) -> Array | np.float64:
        beta, _ = self._beta_and_derivative(xi)
        return beta

    def _beta_and_derivative(self, xi: Array | float) -> tuple[Array, Array]:
        xi_arr = np.asarray(xi, dtype=np.float64)
        small = self._small_xi_mask(xi_arr)
        xi2 = xi_arr * xi_arr
        xi3 = xi2 * xi_arr
        xi4 = xi2 * xi2
        em1 = np.expm1(xi_arr)
        exp_x = em1 + 1.0

        beta_raw = (em1 - xi_arr) / np.where(small, 1.0, xi2)
        beta_series = 0.5 + xi_arr / 6.0 + xi2 / 24.0 + xi3 / 120.0 + xi4 / 720.0
        beta = np.where(small, beta_series, beta_raw)

        deriv_raw = (xi_arr * (exp_x + 1.0) - 2.0 * em1) / np.where(small, 1.0, xi3)
        deriv_series = 1.0 / 6.0 + xi_arr / 12.0 + xi2 / 40.0 + xi3 / 180.0 + xi4 / 1008.0
        deriv = np.where(small, deriv_series, deriv_raw)
        return beta, deriv

    def _beta_derivative(self, xi: Array | float) -> Array | np.float64:
        _, deriv = self._beta_and_derivative(xi)
        return deriv

    def _theta_prime(self, xi: Array | float) -> Array | np.float64:
        xi_arr = np.asarray(xi, dtype=np.float64)
        small = self._small_xi_mask(xi_arr)
        xi2 = xi_arr * xi_arr
        xi3 = xi2 * xi_arr
        xi4 = xi2 * xi2
        xi5 = xi4 * xi_arr
        em1 = np.expm1(xi_arr)
        exp_x = em1 + 1.0
        exp_half_neg = np.exp(-0.5 * xi_arr)
        raw = exp_half_neg * (0.5 * xi_arr * (exp_x + 1.0) - em1) / np.where(small, 1.0, xi2)
        series = xi_arr / 12.0 + xi3 / 480.0 + xi5 / 53760.0
        return np.where(small, series, raw)

    def _invert_beta(self, target: Array | float) -> tuple[Array | np.float64, Array | np.bool_]:
        target_arr = np.asarray(target, dtype=np.float64)
        lo = np.zeros_like(target_arr)
        hi = np.full_like(target_arr, self.xi_max)
        f_lo = self._beta(lo) - target_arr
        f_hi = self._beta(hi) - target_arr
        in_range = (f_lo <= 1e-12) & (f_hi >= -1e-12)

        left = lo
        right = hi
        xi = 0.5 * (lo + hi)
        for _ in range(self.newton_iters):
            beta, deriv = self._beta_and_derivative(xi)
            value = beta - target_arr
            step = value / np.where(np.abs(deriv) > 1e-12, deriv, 1.0)
            cand = np.clip(xi - step, left, right)
            f_cand = self._beta(cand) - target_arr
            move_left = f_cand < 0
            left = np.where(move_left, cand, left)
            right = np.where(move_left, right, cand)
            xi = cand

        for _ in range(self.bisect_iters):
            mid = 0.5 * (left + right)
            f_mid = self._beta(mid) - target_arr
            move_left = f_mid < 0
            left = np.where(move_left, mid, left)
            right = np.where(move_left, right, mid)

        return 0.5 * (left + right), in_range

    def _top_surface_residual(self, p: Array | float, xi: Array | float) -> Array | np.float64:
        p_arr = np.asarray(p, dtype=np.float64)
        xi_arr = np.asarray(xi, dtype=np.float64)
        s = np.exp(0.5 * xi_arr)
        t = np.exp(-0.5 * xi_arr)
        theta = self.theta(s, t)
        dtheta_ds = self.dtheta_ds(s, t)
        dtheta_dt = self.dtheta_dt(s, t)
        return (
            p_arr[0] * (t + theta * dtheta_dt)
            - p_arr[1] * (s + theta * dtheta_ds)
            + p_arr[2] * (t * dtheta_ds - s * dtheta_dt)
        )

    def _top_surface_residual_and_deriv(
        self,
        p: Array | float,
        xi: Array | float,
    ) -> tuple[Array | np.float64, Array | np.float64]:
        p_arr = np.asarray(p, dtype=np.float64)
        xi_arr = np.asarray(xi, dtype=np.float64)
        s = np.exp(0.5 * xi_arr)
        t = np.exp(-0.5 * xi_arr)
        theta = self.theta(s, t)
        dtheta_ds = self.dtheta_ds(s, t)
        dtheta_dt = self.dtheta_dt(s, t)
        residual = (
            p_arr[0] * (t + theta * dtheta_dt)
            - p_arr[1] * (s + theta * dtheta_ds)
            + p_arr[2] * (t * dtheta_ds - s * dtheta_dt)
        )

        s_prime = 0.5 * s
        t_prime = -0.5 * t
        theta_prime = self._theta_prime(xi_arr)
        dtheta_dt_prime = self._beta_derivative(xi_arr)
        dtheta_ds_prime = -self._beta_derivative(-xi_arr)
        deriv = (
            p_arr[0] * (t_prime + theta_prime * dtheta_dt + theta * dtheta_dt_prime)
            - p_arr[1] * (s_prime + theta_prime * dtheta_ds + theta * dtheta_ds_prime)
            + p_arr[2]
            * (
                t_prime * dtheta_ds
                + t * dtheta_ds_prime
                - s_prime * dtheta_dt
                - s * dtheta_dt_prime
            )
        )
        return residual, deriv

    def origin_supergrad_contains(self, z1: Array | float, z2: Array | float) -> Array | np.bool_:
        z1_arr, z2_arr = _to_numpy(z1, z2)
        swap = z1_arr < z2_arr
        primary = np.where(swap, z2_arr, z1_arr)
        secondary = np.where(swap, z1_arr, z2_arr)
        positive = (z1_arr > 0) & (z2_arr > 0)

        xi, in_range = self._invert_beta(primary)
        s = np.exp(-0.5 * xi)
        t = np.exp(0.5 * xi)
        needed = self.dtheta_dt(s, t)
        return positive & in_range & (secondary >= needed - 1e-12)

    def project_k_top(
        self,
        p1: Array | float,
        p2: Array | float,
        p3: Array | float,
    ) -> tuple[Array | np.float64, Array | np.float64, Array | np.float64]:
        p = np.asarray([p1, p2, p3], dtype=np.float64)
        lo0 = np.array(-self.xi_max, dtype=np.float64)
        hi0 = np.array(self.xi_max, dtype=np.float64)
        f_lo = self._top_surface_residual(p, lo0)
        f_hi = self._top_surface_residual(p, hi0)
        has_bracket = (
            (np.abs(f_lo) <= 1e-12)
            | (np.abs(f_hi) <= 1e-12)
            | (np.sign(f_lo) != np.sign(f_hi))
        )

        left = lo0
        right = hi0
        f_left = f_lo
        f_right = f_hi
        xi = np.array(0.0, dtype=np.float64)
        for _ in range(self.newton_iters):
            value, deriv = self._top_surface_residual_and_deriv(p, xi)
            step = value / np.where(np.abs(deriv) > 1e-12, deriv, 1.0)
            cand = np.clip(xi - step, left, right)
            f_cand = self._top_surface_residual(p, cand)
            same_side_as_left = np.sign(f_left) * np.sign(f_cand) > 0
            left = np.where(same_side_as_left, cand, left)
            f_left = np.where(same_side_as_left, f_cand, f_left)
            right = np.where(same_side_as_left, right, cand)
            f_right = np.where(same_side_as_left, f_right, f_cand)
            xi = cand

        for _ in range(self.bisect_iters):
            mid = 0.5 * (left + right)
            f_mid = self._top_surface_residual(p, mid)
            same_side_as_left = np.sign(f_left) * np.sign(f_mid) > 0
            left = np.where(same_side_as_left, mid, left)
            f_left = np.where(same_side_as_left, f_mid, f_left)
            right = np.where(same_side_as_left, right, mid)
            f_right = np.where(same_side_as_left, f_right, f_mid)

        xi = 0.5 * (left + right)
        xi = np.where(
            has_bracket,
            xi,
            np.where(np.abs(f_lo) <= np.abs(f_hi), lo0, hi0),
        )
        s = np.exp(0.5 * xi)
        t = np.exp(-0.5 * xi)
        w = np.asarray([s, t, self.theta(s, t)], dtype=np.float64)
        tau = np.maximum(np.dot(p, w) / np.maximum(np.dot(w, w), 1e-30), 0.0)
        proj = tau * w
        return proj[0], proj[1], proj[2]
