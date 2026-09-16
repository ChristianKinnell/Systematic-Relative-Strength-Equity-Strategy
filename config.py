from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class StrategyConfig:
    start: str = "2018-01-01"
    warmup_days: int = 260
    top_n: int = 7
    score_threshold: float = 0.0
    sizing_method: str = "inverse_vol"
    rebal_band: float = 0.08
    dd_stop: float = -0.08
    dd_cooldown: int = 3
    kelly_fraction: float = 0.25
    max_weight: float = 0.15
    min_weight: float = 0.10
    vol_low: float = 0.14
    vol_high: float = 0.24
    stability_n_splits: int = 6
    stability_k_test: int = 2
    stability_buffer_days: int = 5
    hmm_states: int = 3
    hmm_restarts: int = 8
    hmm_min_train: int = 252
    hmm_refit_every: int = 63
    cvar_confidence: float = 0.95
    cvar_limit: float = 0.030
    lambda_turnover: float = 0.035
    lambda_alpha: float = 0.10
    sector_cap: float = 0.45
    stress_scale: float = 0.50
    atr_period: int = 14
    atr_stop_mult: float = 2.5
    backtest_nav: float = 100_000.0

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = StrategyConfig()
