# src/calibration.py
from .finance_core import calculate_V_SLB, calculate_V_B
from scipy.optimize import minimize
import numpy as np


def find_lambda_E(target_V_CB_0, r, delta, kappa, theta, sigma_r, E_0, mu_E, sigma_E, t_o, E_bar, c, t_i, k_factor):
    def obj(lmbd):
        V = calculate_V_SLB(r, delta, kappa, theta, sigma_r, E_0, mu_E, lmbd[0], sigma_E, t_o, E_bar, c, t_i, k_factor)
        return (V - target_V_CB_0) ** 2
    res = minimize(obj, x0=np.array([0.0]), method="BFGS")
    return float(res.x[0])


def find_delta(target_V_B_0, r, kappa, theta, sigma_r, c, t_i):
    def obj(d):
        V = calculate_V_B(r, d[0], kappa, theta, sigma_r, c, t_i)
        return (V - target_V_B_0) ** 2
    res = minimize(obj, x0=np.array([0.01]), method="BFGS")
    return float(res.x[0])
