from __future__ import annotations

import numpy as np
import pandas as pd


def transaction_costs(
    weights: pd.DataFrame, returns: pd.DataFrame, dollar_vol: pd.DataFrame,
    *, nav: float = 100_000.0, fixed_bps: float = 0.001, impact_coeff: float = 0.1
) -> pd.Series:
    turnover = weights.reindex(returns.index).diff().abs()
    cost = pd.Series(0.0, index=returns.index)
    sig = returns.rolling(21).std().fillna(0)
    adv = dollar_vol.rolling(21).median().reindex(returns.index).clip(lower=5_000_000).fillna(5_000_000)
    for s in weights.columns:
        fixed = turnover[s] * fixed_bps
        trade_val = (turnover[s] * nav).fillna(0)
        impact = impact_coeff * sig[s] * np.sqrt((trade_val / adv[s]).clip(lower=0))
        cost += fixed + impact.clip(upper=0.02)
    return cost


def check_no_missing_held_returns(day_w: pd.Series, ret_row: pd.Series, date) -> None:
    bad = (day_w.abs() > 1e-9) & ret_row.isna()
    if bad.any():
        raise RuntimeError(
            f"Missing return on {pd.Timestamp(date).date()} for held names "
            f"{day_w.index[bad].tolist()}"
        )
