# Changelog

## v5.0 — 2026-09-17 — Research freeze
### Correctness and reproducibility
- Removed equity OHLCV forward-fill; missing observations remain missing.
- Added phantom market-holiday-row guard and minimum-eligible-universe sanity gate.
- Added hard failure for missing returns in held names.
- Frozen price snapshot now freezes candidate universe and metadata as well.
- Fixed mandatory exits so they bypass the rebalance band.
- Weekly rebalances use the final actual trading session of W-FRI weeks.
- HMM labels use in-training-window forward benchmark returns; model and scaler are retained atomically on failed refits.
- OOS segment states use causal forward filtering rather than Viterbi smoothing.
- Portfolio drawdown circuit breaker now uses net NAV, exact cooldown length and peak reset on resume.
- Volatility hysteresis is live in portfolio construction.
- Production and ablation engines both use net-of-cost NAV logic.
- Active return changed to relative-wealth mathematics.
- IC and quintile diagnostics are masked to the investable liquidity universe.
- Renamed misleading validation terminology; time-slice analysis is not claimed as CPCV/PBO.
- Updated transaction-cost terminology to fixed spread + square-root impact approximation.

### Freeze policy
Historical results through the freeze date are development evidence. Only unchanged-model results after 2026-09-17 are prospective OOS.

## Pre-v5
Earlier versions were iterative research builds and should not be presented as independently validated production models. The archived v5 source documents important historical bug fixes in detail.
