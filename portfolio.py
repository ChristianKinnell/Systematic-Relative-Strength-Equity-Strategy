from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol_weights(ret_window) -> np.ndarray:
    r = ret_window.values if hasattr(ret_window, "values") else np.asarray(ret_window)
    vol = r.std(axis=0)
    pos = vol[vol > 0]
    fallback = pos.mean() if len(pos) else 1.0
    vol = np.where(vol > 0, vol, fallback)
    inv = 1.0 / vol
    return inv / inv.sum()


def cap_and_redistribute(weights: pd.Series, max_weight: float) -> pd.Series:
    w = weights.astype(float).copy()
    for _ in range(20):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w.loc[over] = max_weight
        under = w < max_weight - 1e-12
        if excess <= 1e-12 or not under.any():
            break
        room = (max_weight - w[under]).clip(lower=0)
        if room.sum() <= 0:
            break
        w.loc[under] += excess * room / room.sum()
    return w


def enforce_sector_cap(weights: pd.Series, symbols: list[str], sectors: dict[str, str], sector_cap: float) -> pd.Series:
    w = weights.copy().astype(float)
    if w.empty:
        return w
    for _ in range(20):
        sector_totals = {}
        for s in symbols:
            if s in w.index:
                sec = sectors.get(s, "Unknown")
                sector_totals[sec] = sector_totals.get(sec, 0.0) + float(w[s])
        breaches = {sec: val for sec, val in sector_totals.items() if val > sector_cap + 1e-12}
        if not breaches:
            break
        freed = 0.0
        for sec, total in breaches.items():
            members = [s for s in w.index if sectors.get(s, "Unknown") == sec]
            scale = sector_cap / total
            before = float(w[members].sum())
            w.loc[members] *= scale
            freed += before - float(w[members].sum())
        if freed <= 1e-12:
            break
        eligible = [s for s in w.index if sector_totals.get(sectors.get(s, "Unknown"), 0.0) < sector_cap - 1e-12]
        if not eligible:
            break
        base = w[eligible].sum()
        if base > 0:
            w.loc[eligible] += freed * w[eligible] / base
    return w


def apply_rebalance_band(previous: pd.Series, candidate: pd.Series, band: float) -> pd.Series:
    """Mandatory exits bypass the band; continuing names use hysteresis."""
    result = previous.copy()
    relevant = previous[previous.abs() > 1e-9].index.union(candidate[candidate.abs() > 1e-9].index)
    for s in relevant:
        if previous.get(s, 0.0) > 0 and candidate.get(s, 0.0) == 0:
            result[s] = 0.0
        elif abs(candidate.get(s, 0.0) - previous.get(s, 0.0)) > band:
            result[s] = candidate.get(s, 0.0)
    return result
