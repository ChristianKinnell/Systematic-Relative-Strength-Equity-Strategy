from __future__ import annotations

import numpy as np
import pandas as pd


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return 2 * (df.rank(axis=1, pct=True, na_option="keep") - 0.5)


def cs_zscore(df: pd.DataFrame, winsor: float = 0.02) -> pd.DataFrame:
    lo = df.quantile(winsor, axis=1)
    hi = df.quantile(1 - winsor, axis=1)
    clipped = df.clip(lower=lo, upper=hi, axis=0)
    mu = clipped.mean(axis=1)
    sd = clipped.std(axis=1).replace(0, np.nan)
    return clipped.sub(mu, axis=0).div(sd, axis=0)


def build_momentum_signal(
    price: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_price: pd.Series,
    *,
    skip_days: int = 5,
    ewm_halflife: int = 5,
) -> dict[str, pd.DataFrame]:
    m1 = price.shift(skip_days) / price.shift(skip_days + 21) - 1
    m3 = price.shift(skip_days) / price.shift(skip_days + 63) - 1
    m6 = price.shift(skip_days) / price.shift(skip_days + 126) - 1
    m12 = price.shift(skip_days) / price.shift(skip_days + 252) - 1
    vol63 = returns.rolling(63).std().replace(0, np.nan)
    raw = 0.15 * m1 + 0.30 * m3 + 0.30 * m6 + 0.15 * m12
    vol_adj = raw / vol63
    sig_mom = cs_rank(vol_adj)
    bench6 = benchmark_price.pct_change(126)
    rs = price.pct_change(126, fill_method=None).sub(bench6, axis=0)
    sig_rs = cs_rank(rs)
    alpha = (0.90 * sig_mom.fillna(0) + 0.10 * sig_rs.fillna(0))
    alpha = alpha.ewm(halflife=ewm_halflife).mean().reindex(returns.index).fillna(0)
    return {
        "mom_1m": m1, "mom_3m": m3, "mom_6m": m6, "mom_12m": m12,
        "vol_63": vol63, "mom_raw": raw, "mom_vol_adj": vol_adj,
        "sig_mom": sig_mom, "rel_strength": rs, "sig_rs": sig_rs,
        "alpha_final": alpha,
    }


def confirmation_gate(
    price: pd.DataFrame, rel_strength: pd.DataFrame, vol_63: pd.DataFrame,
    mom_3m: pd.DataFrame, mom_6m: pd.DataFrame, *, vol_ceiling: float = 0.65
) -> pd.DataFrame:
    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()
    vol_ann = vol_63 * np.sqrt(252)
    return (
        (price > ma50) & (ma50 > ma200) & (rel_strength > 0) &
        (vol_ann <= vol_ceiling) & (mom_3m > 0) & (mom_6m > 0)
    )
