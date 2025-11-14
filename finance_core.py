# src/finance_core.py
from __future__ import annotations

import math
from typing import Iterable, Sequence, Union

import numpy as np
from scipy.stats import norm


def _as_years_scalar(t: Union[float, int, np.number, Iterable[float]]) -> float:
    if isinstance(t, (int, float, np.floating, np.integer)):
        return float(t)
    try:
        return float(np.asarray(t).ravel()[0])
    except Exception as e:
        raise TypeError("t must be a scalar or 1D array-like of times in years") from e


def _ensure_1d_array(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=float).ravel()
    if a.ndim != 1:
        raise ValueError("Expected a 1D array-like.")
    return a


def expectation_probability(
    E_0: float, mu_E: float, lambda_E: float, sigma_E: float,
    t_o: Union[float, Sequence[float]], E_bar: float
) -> float:
    t = _as_years_scalar(t_o)
    if t <= 0 or E_0 <= 0 or E_bar <= 0 or sigma_E <= 0:
        raise ValueError("Invalid input: t_o, E_0, E_bar, sigma_E must be positive.")
    num = math.log(E_0 / E_bar) + (mu_E - lambda_E * sigma_E - 0.5 * sigma_E**2) * t
    den = sigma_E * math.sqrt(t)
    return float(norm.cdf(num / den))


def historical_probability(
    E_0: float, mu_E: float, sigma_E: float, t_o: float, E_bar: float
) -> float:
    if t_o <= 0 or E_0 <= 0 or E_bar <= 0 or sigma_E <= 0:
        raise ValueError("Invalid input.")
    num = math.log(E_0 / E_bar) + (mu_E - 0.5 * sigma_E**2) * t_o
    den = sigma_E * math.sqrt(t_o)
    return float(norm.cdf(num / den))


def vasicek_price(r: float, kappa: float, theta: float, sigma_r: float, T: float) -> float:
    if T < 0 or kappa <= 0:
        raise ValueError("Invalid maturity or kappa.")
    if T == 0:
        return 1.0
    B = (1.0 - math.exp(-kappa * T)) / kappa
    A = (T - B) * (theta - sigma_r**2 / (2.0 * kappa**2)) + (sigma_r**2 / (4.0 * kappa)) * (B**2)
    return math.exp(-A - B * r)


def calculate_V_B(r, delta, kappa, theta, sigma_r, c, t_i) -> float:
    t_i = _ensure_1d_array(t_i)
    if len(t_i) == 0 or t_i.min() <= 0:
        raise ValueError("Invalid coupon times.")
    w = t_i[0]
    first = w * c * math.exp(-delta * t_i[0]) * vasicek_price(r, kappa, theta, sigma_r, t_i[0])
    rest = sum(math.exp(-delta * ti) * c * vasicek_price(r, kappa, theta, sigma_r, ti) for ti in t_i[1:])
    fv = 100.0 * math.exp(-delta * t_i[-1]) * vasicek_price(r, kappa, theta, sigma_r, t_i[-1])
    return float(first + rest + fv)


def calculate_SLB_P1(r, delta, kappa, theta, sigma_r, t_o, c, t_i, k_factor) -> float:
    t_o = _ensure_1d_array(t_o)
    return float(sum(math.exp(-delta * ti) * k_factor * vasicek_price(r, kappa, theta, sigma_r, ti) for ti in t_o))


def calculate_V_SLB(r, delta, kappa, theta, sigma_r, E_0, mu_E, lambda_E, sigma_E, t_o, E_bar, c, t_i, k_factor) -> float:
    t_o = _ensure_1d_array(t_o)
    t_i = _ensure_1d_array(t_i)
    if len(t_o) == 0 or len(t_i) == 0 or t_o.min() <= 0 or t_i.min() <= 0:
        raise ValueError("Invalid times.")
    w = t_i[0]
    first = w * c * math.exp(-delta * t_i[0]) * vasicek_price(r, kappa, theta, sigma_r, t_i[0])
    rest = sum(math.exp(-delta * ti) * c * vasicek_price(r, kappa, theta, sigma_r, ti) for ti in t_i[1:])
    prob = expectation_probability(E_0, mu_E, lambda_E, sigma_E, t_o, E_bar)
    fv = 100.0 * math.exp(-delta * t_i[-1]) * vasicek_price(r, kappa, theta, sigma_r, t_i[-1])
    extra = calculate_SLB_P1(r, delta, kappa, theta, sigma_r, t_o, c, t_i, k_factor)
    return float(first + rest + prob * extra + fv)