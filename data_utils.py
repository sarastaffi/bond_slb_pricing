# src/data_utils.py
import pandas as pd
import numpy as np
from typing import Dict, Iterable, List
from pathlib import Path


def _is_bday(d): return d.weekday() < 5
def _following(d):
    d = pd.Timestamp(d).normalize()
    while not _is_bday(d): d += pd.Timedelta(days=1)
    return d
def _add_bdays(start, n):
    d = pd.Timestamp(start).normalize()
    while n > 0:
        d += pd.Timedelta(days=1)
        if _is_bday(d): n -= 1
    return d


def build_annual_schedule(start, end, freq="Y", adjust=True):
    sched = pd.date_range(start=start, end=end, freq=freq).tolist()
    if sched and sched[-1] != end: sched.append(end)
    return [_following(d) for d in sched] if adjust else sched


def future_coupon_dates(date_t, schedule):
    if pd.isna(date_t) or not schedule: return []
    sched = [_following(pd.Timestamp(d).normalize()) for d in schedule if pd.notna(d)]
    trade = _following(pd.Timestamp(date_t).normalize())
    settlement = _add_bdays(trade, 2)
    return [d for d in sched if d >= settlement]


def compute_emission_features(row: pd.Series, emissions_df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(row["date"])
    cutoff_year = date.year - 1
    filtered = emissions_df[emissions_df.index.year == cutoff_year]
    return pd.Series({
        "mu_E": filtered["mu_E"].iloc[0] if not filtered.empty else np.nan,
        "sigma_E": filtered["sigma_E"].iloc[0] if not filtered.empty else np.nan,
        "lag_emission": filtered["emissions"].iloc[0] if not filtered.empty else np.nan
    })


def compute_interpolated_spread(maturity_to_match, df_param, isin_frames, standard_isins=None, roll_med_win=0, ewm_halflife=0):
    maturity_to_match = pd.to_datetime(maturity_to_match)
    standard_isins = standard_isins or list(isin_frames.keys())
    rows = df_param.loc[df_param['ISIN'].isin(standard_isins), ['ISIN', 'Maturity']].dropna()
    rows['Maturity'] = pd.to_datetime(rows['Maturity'])
    ref = rows['Maturity'].min()
    T_map = {r.ISIN: (r.Maturity - ref).days / 365.25 for _, r in rows.iterrows()}
    T_star = (maturity_to_match - ref).days / 365.25

    frames = []
    for isin in standard_isins:
        df = isin_frames.get(isin)
        if df is None or df.empty: continue
        tmp = df[['date', 'delta']].copy()
        tmp['date'] = pd.to_datetime(tmp['date'])
        tmp = tmp.dropna().sort_values('date')
        frames.append(tmp.rename(columns={'delta': f'delta_{isin}'}))

    all_dates = pd.concat([f[['date']] for f in frames]).drop_duplicates().sort_values('date')
    wide = all_dates.copy()
    for tmp in frames: wide = wide.merge(tmp, on='date', how='left')

    def two_nearest_linear(ts_vals):
        if not ts_vals: return np.nan
        if len(ts_vals) == 1: return float(ts_vals[0][1])
        ts_vals.sort(key=lambda x: abs(x[0] - T_star))
        (t1, v1), (t2, v2) = ts_vals[:2]
        return float((v1 + v2) / 2) if t1 == t2 else float(v1 + (T_star - t1) / (t2 - t1) * (v2 - v1))

    def estimate_row(row):
        ts_vals = [(T_map[isin], row[f'delta_{isin}']) for isin in standard_isins
                   if f'delta_{isin}' in row and pd.notna(row[f'delta_{isin}']) and isin in T_map]
        return two_nearest_linear(ts_vals)

    wide['Interpolated_SLB_Spread_raw'] = wide.apply(estimate_row, axis=1)
    out = wide[['date', 'Interpolated_SLB_Spread_raw']].rename(columns={'Interpolated_SLB_Spread_raw': 'Interpolated_SLB_Spread'})
    if roll_med_win > 0:
        out['Interpolated_SLB_Spread'] = out['Interpolated_SLB_Spread'].rolling(window=roll_med_win, center=True, min_periods=1).median()
    if ewm_halflife > 0:
        out['Interpolated_SLB_Spread'] = out['Interpolated_SLB_Spread'].ewm(halflife=ewm_halflife, min_periods=1).mean()
    return out
