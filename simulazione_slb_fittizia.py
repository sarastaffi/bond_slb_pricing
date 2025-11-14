# simulazione_slb_fittizia.py
"""
Simulazione completa di un Sustainability-Linked Bond (SLB) con dati fittizi.
Usa il modulo src/ del progetto riorganizzato.
Esegui con: python simulazione_slb_fittizia.py
"""

import sys
from pathlib import Path
# Ensure the project `code` folder and its parent are on sys.path so `from src import ...` works
HERE = Path(__file__).resolve().parent
# Prefer local `code/src` (we copy project modules here) so it shadows progetto_bond/src
LOCAL_SRC = HERE / 'src'
if LOCAL_SRC.exists():
    sys.path.insert(0, str(LOCAL_SRC))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# 1. IMPORT DAL TUO PROGETTO (progetto_bond/src)
# -------------------------------------------------
# Import from local `src` package (copied into `code/src`)
from src import (
    vasicek_price,
    calculate_V_B,
    calculate_V_SLB,
    calculate_SLB_P1,
    expectation_probability,
)
from src.calibration import find_delta, find_lambda_E

# -------------------------------------------------
# 2. DATI FITTIZI (tutto generato qui!)
# -------------------------------------------------

# Parametri Vasicek (fittizi, ma realistici)
r0 = 0.02          # tasso spot iniziale
kappa = 0.3
theta = 0.02
sigma_r = 0.01

# Parametri SLB
issue_date = pd.Timestamp("2021-01-01")
maturity = pd.Timestamp("2026-01-01")
coupon_rate = 0.03   # 3% annuo
face_value = 100.0

# Date di osservazione (SPT)
observation_dates = [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")]

# Parametri emissioni (GBM)
E_0 = 300.0          # emissioni iniziali (tCO2)
mu_E = -0.05         # drift negativo (riduzione)
sigma_E = 0.15
E_bar = 200.0        # soglia SPT
k_factor = 0.25      # penalty: +25 bp se fallisce

# Genera date di valutazione (ogni 6 mesi)
eval_dates = pd.date_range("2021-06-30", maturity, freq="6ME")
n_dates = len(eval_dates)

# Simula prezzi di mercato (rumore realistico)
np.random.seed(42)
market_prices = 100 + np.cumsum(np.random.normal(0, 0.5, n_dates))  # prezzo pulito
market_prices = np.clip(market_prices, 95, 110)

# Simula r0 variabile nel tempo
r0_series = r0 + np.cumsum(np.random.normal(0, 0.001, n_dates))
r0_series = np.clip(r0_series, 0.015, 0.03)

# -------------------------------------------------
# 3. PREPARA IL DATABASE FITTIZIO
# -------------------------------------------------

def build_coupon_schedule(eval_date):
    """Calcola tempi coupon futuri da eval_date"""
    all_coupons = pd.date_range(issue_date, maturity, freq="12ME")
    future = [d for d in all_coupons if d > eval_date]
    t_i = [(d - eval_date).days / 365.25 for d in future]
    return np.array(t_i) if t_i else np.array([0.01])


def build_observation_times(eval_date):
    """Tempi di osservazione SPT"""
    future_obs = [d for d in observation_dates if d > eval_date]
    t_o = [(d - eval_date).days / 365.25 for d in future_obs]
    return np.array(t_o) if t_o else np.array([0.01])

# Costruisci database
data = []
for i, date in enumerate(eval_dates):
    t_i = build_coupon_schedule(date)
    t_o = build_observation_times(date)
    
    data.append({
        'date': date,
        'r0_star': r0_series[i],
        'kappa_star': kappa,
        'theta_star': theta,
        'sigma_star': sigma_r,
        'coupon_ti': t_i.tolist(),
        'c_coupon_ti': t_o.tolist(),
        'Mid Price': market_prices[i],
        'lag_emission': E_0 * np.exp((mu_E - 0.5*sigma_E**2) * (date - issue_date).days / 365.25 
                                  + sigma_E * np.random.normal()),
        'mu_E': mu_E,
        'sigma_E': sigma_E,
        'Pay': k_factor,
        'SPT Threshold': E_bar,
    })

db = pd.DataFrame(data)

# -------------------------------------------------
# 4. CALCOLI: delta, lambda_E, prezzi modello
# -------------------------------------------------

delta_vals = []
lambda_vals = []
model_slb = []
model_bond = []

for _, row in db.iterrows():
    r = row['r0_star']
    t_i = np.array(row['coupon_ti'])
    t_o = np.array(row['c_coupon_ti'])
    target = row['Mid Price']
    c = coupon_rate
    E0 = row['lag_emission']
    
    # Filtra tempi > 0
    t_i = t_i[t_i > 1e-6]
    t_o = t_o[t_o > 1e-6]
    
    if len(t_i) == 0:
        delta_vals.append(np.nan)
        lambda_vals.append(np.nan)
        model_slb.append(np.nan)
        model_bond.append(np.nan)
        continue
    
    # Calibra delta
    delta = find_delta(target, r, kappa, theta, sigma_r, c, t_i)
    V_B = calculate_V_B(r, delta, kappa, theta, sigma_r, c, t_i)
    
    # Calibra lambda_E
    lambda_E = find_lambda_E(target, r, delta, kappa, theta, sigma_r,
                             E0, mu_E, sigma_E, t_o, E_bar, c, t_i, k_factor)
    V_SLB = calculate_V_SLB(r, delta, kappa, theta, sigma_r,
                            E0, mu_E, lambda_E, sigma_E, t_o, E_bar, c, t_i, k_factor)
    
    delta_vals.append(delta)
    lambda_vals.append(lambda_E)
    model_slb.append(V_SLB)
    model_bond.append(V_B)

db['delta'] = delta_vals
db['lambda_E'] = lambda_vals
db['model_SLB'] = model_slb
db['model_bond'] = model_bond

# -------------------------------------------------
# 5. GRAFICI
# -------------------------------------------------

plt.figure(figsize=(14, 10))

# Prezzo di mercato vs modello
plt.subplot(2, 2, 1)
plt.plot(db['date'], db['Mid Price'], 'o-', label='Mercato', color='black')
plt.plot(db['date'], db['model_SLB'], 's-', label='Modello SLB', color='red')
plt.plot(db['date'], db['model_bond'], '--', label='Vanilla Bond', color='blue')
plt.title('Prezzo SLB: Mercato vs Modello')
plt.legend()
plt.grid(True)

# Delta
plt.subplot(2, 2, 2)
plt.plot(db['date'], db['delta'], 'o-', color='purple')
plt.title('Delta Calibrato')
plt.ylabel('Spread (delta)')
plt.grid(True)

# Lambda_E
plt.subplot(2, 2, 3)
plt.plot(db['date'], db['lambda_E'], 'o-', color='green')
plt.title('Lambda_E (Risk Adjustment)')
plt.ylabel('lambda_E')
plt.grid(True)

# Probabilità di fallire SPT
probs = []
for _, row in db.iterrows():
    t_o = np.array(row['c_coupon_ti'])
    if len(t_o) == 0 or t_o[0] <= 0:
        probs.append(np.nan)
        continue
    p = expectation_probability(row['lag_emission'], mu_E, row['lambda_E'], sigma_E, t_o, E_bar)
    probs.append(p)
db['prob_fail'] = probs

plt.subplot(2, 2, 4)
plt.plot(db['date'], db['prob_fail'], 'o-', color='orange')
plt.title('Probabilità di Fallire SPT')
plt.ylabel('P(E_t > E_bar)')
plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# 6. STAMPA RISULTATI
# -------------------------------------------------

print("\nSIMULAZIONE COMPLETATA!")
print(f"Date simulate: {len(db)}")
print(f"Prezzo medio mercato: {db['Mid Price'].mean():.2f}")
print(f"Prezzo medio modello SLB: {db['model_SLB'].mean():.2f}")
print(f"Delta medio: {db['delta'].mean():.4f}")
print(f"Lambda_E medio: {db['lambda_E'].mean():.4f}")
print(f"Probabilità media fallimento SPT: {db['prob_fail'].mean():.1%}")
