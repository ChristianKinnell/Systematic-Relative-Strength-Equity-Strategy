from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def last_trading_session_each_week(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index.to_series().resample("W-FRI").last().dropna().values)


def save_snapshot(raw: pd.DataFrame, universe: dict, directory: str | Path, manifest: dict) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(d / "prices.parquet")
    pd.DataFrame([
        {"ticker": s, "sector": v["sector"], "cost_bps": v["cost_bps"]}
        for s, v in universe.items()
    ]).to_csv(d / "universe.csv", index=False)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_frozen_universe(directory: str | Path) -> dict:
    df = pd.read_csv(Path(directory) / "universe.csv")
    return {row.ticker: {"sector": row.sector, "cost_bps": row.cost_bps} for row in df.itertuples()}
