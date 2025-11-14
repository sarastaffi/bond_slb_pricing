# src/calculations.py
import pandas as pd
import numpy as np
from .calibration import find_delta, find_lambda_E
from .finance_core import calculate_V_B, calculate_V_SLB, calculate_SLB_P1


def calculate_delta_and_model(db: pd.DataFrame, bond_type: str = 'standard') -> pd.DataFrame:
    delta_sols, model_prices, lambda_sols = [], [], []
    for _, row in db.iterrows():
        try:
            r = row.get('r0_star', row.get('r', 0))
            kappa, theta, sigma_r = row['kappa_star'], row['theta_star'], row['sigma_star']
            c = row.get('Coupon', 0) / 100.0
            t_i = np.array(eval(row['coupon_ti'])) if isinstance(row['coupon_ti'], str) else np.array(row['coupon_ti'])
            t_i = t_i[t_i > 0]
            target = row['Mid Price']

            delta_sol = find_delta(target, r, kappa, theta, sigma_r, c, t_i)
            model_price = calculate_V_B(r, delta_sol, kappa, theta, sigma_r, c, t_i)

            delta_sols.append(delta_sol)
            model_prices.append(model_price)

            if bond_type == 'slb':
                t_o = np.array(eval(row['c_coupon_ti'])) if 'c_coupon_ti' in row else t_i
                t_o = t_o[t_o > 0]
                lambda_sol = find_lambda_E(target, r, delta_sol, kappa, theta, sigma_r,
                                           row['lag_emission'], row['mu_E'], row['sigma_E'],
                                           t_o, row['SPT Threshold'], c, t_i, row['Pay'])
                lambda_sols.append(lambda_sol)
            else:
                lambda_sols.append(np.nan)

        except Exception as e:
            print(f"Error at {row['date']}: {e}")
            delta_sols.append(np.nan)
            model_prices.append(np.nan)
            lambda_sols.append(np.nan)

    db['delta'] = delta_sols
    db['model_price'] = model_prices
    if bond_type == 'sl_ui':
        db['lambda_E'] = lambda_sols
    return db


def calculate_omega(green_db: pd.DataFrame, interp_df: pd.DataFrame) -> pd.DataFrame:
    merged = green_db.merge(interp_df, on='date', how='left')
    merged['omega'] = merged['delta'] - merged['Interpolated_SLB_Spread']
    return merged
