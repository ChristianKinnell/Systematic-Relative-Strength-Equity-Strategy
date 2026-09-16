from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(ret: pd.Series, ann: int = 252) -> dict[str, float]:
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    downside = ret[ret < 0]
    years = len(ret) / ann
    total = eq.iloc[-1] - 1 if len(eq) else np.nan
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and 1 + total > 0 else np.nan
    sharpe = ret.mean() / ret.std() * np.sqrt(ann) if ret.std() > 0 else np.nan
    down_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(ann) if len(downside) else np.nan
    sortino = ret.mean() * ann / down_dev if down_dev and down_dev > 0 else np.nan
    max_dd = dd.min() if len(dd) else np.nan
    calmar = cagr / abs(max_dd) if np.isfinite(max_dd) and max_dd < 0 else np.nan
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd, "calmar": calmar}


def relative_wealth(port_ret: pd.Series, bench_ret: pd.Series) -> pd.Series:
    idx = port_ret.index.intersection(bench_ret.index)
    peq = (1 + port_ret.reindex(idx)).cumprod()
    beq = (1 + bench_ret.reindex(idx)).cumprod()
    return peq / beq


def rank_ic_series(signal: pd.DataFrame, price: pd.DataFrame, eligible: pd.DataFrame, horizon: int) -> pd.Series:
    fwd = price.pct_change(horizon, fill_method=None).shift(-horizon).reindex(signal.index)
    masked = signal.reindex_like(eligible).where(eligible)
    return masked.corrwith(fwd, axis=1, method="spearman").dropna()
