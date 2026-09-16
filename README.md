# Systematic Relative-Strength Equity Strategy

A research-grade, long-only U.S. equity momentum framework combining multi-horizon momentum, relative strength, confirmation filters, expanding-window HMM regime detection, inverse-volatility sizing, portfolio risk controls, transaction-cost modelling, and walk-forward validation.

> **Status:** v5.0 research freeze. Strategy parameters are frozen from **2026-09-17**. Historical results are development evidence, not prospective out-of-sample evidence.

## Why this repository exists

This repository documents the full investment-research process rather than presenting a single backtest statistic. The emphasis is on causal signal construction, explicit limitations, reproducibility, risk controls, and evidence that each layer contributes something measurable.

## Strategy architecture

```text
Yahoo US large-cap screen
        ↓
Liquidity / coverage / history filter
        ↓
1m / 3m / 6m / 12m momentum + 6m relative strength
        ↓
Trend + absolute-momentum + volatility confirmation
        ↓
Top-7 selection with positive-score threshold
        ↓
Inverse-volatility sizing + name/sector caps
        ↓
HMM / macro / stress exposure controls
        ↓
Volatility hysteresis + drawdown circuit breaker
        ↓
T+1 portfolio path + transaction costs
        ↓
IC / decay / ablations / walk-forward / concentration diagnostics
```

## Core design

- **Universe:** current Yahoo Finance U.S.-domiciled NYSE/NASDAQ companies above $25bn market cap.
- **Alpha:** 1m/3m/6m/12m momentum with a 5-trading-day skip, volatility adjustment, cross-sectional ranking, 10% relative-strength overlay, and 5-day EWM smoothing.
- **Confirmation:** price > 50DMA > 200DMA, positive 6m relative strength, M3/M6 > 0, annualised volatility <= 65%.
- **Selection:** Top 7 positive-score names; unfilled slots remain cash.
- **Sizing:** inverse volatility; 15% per-name cap; 45% sector cap.
- **Regime:** 3-state expanding-window Gaussian HMM on SPX/VIX/LQD, causally forward-filtered and labelled by in-window forward SPX return; macro and fast-stress overlays.
- **Risk:** stress exposure 50%; 14%/24% realised-volatility hysteresis; -8% drawdown circuit breaker with 3-session time-based freeze.
- **Costs:** 10 bps fixed cost per trade plus square-root market-impact approximation.
- **Timing:** decisions made at close are applied to the next trading session's return.

## Repository layout

```text
src/nyse_momentum/     reusable research components
tests/                 invariant and unit tests
docs/                  methodology, validation and limitations
research_snapshot/     frozen code/config fingerprint
archive/                exact v5 monolithic implementation
scripts/                run and snapshot utilities
```

The exact frozen model is deliberately preserved in `archive/frozen_v5_monolith.py`. The modular package is a staged refactor. The monolith remains the canonical implementation until parity tests demonstrate identical weights, NAV, turnover, risk events, and targets.

## Reproducibility

The model can freeze prices, universe membership, sector metadata, and a manifest. Large market-data files are not committed by default. See `docs/REPRODUCIBILITY.md`.

## Validation

The project reports portfolio metrics and alpha-specific diagnostics: rank IC, IC decay, forward-return quintiles, confirmation/regime/sizing/risk ablations, parameter sensitivity, combinatorial time-slice stability, fixed-config expanding-year checks, and a scoped train→optimise→freeze→test grid. See `docs/VALIDATION.md`.

## Important limitation

The historical universe is selected from **today's** Yahoo Finance screen, so historical results remain exposed to survivorship bias. This is disclosed rather than disguised as point-in-time membership. See `docs/LIMITATIONS.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/run_frozen.py
```

To create a frozen Yahoo snapshot before a research run:

```bash
NYSE_ALPHA_FREEZE_SNAPSHOT=1 python scripts/run_frozen.py
```

To replay the frozen snapshot:

```bash
NYSE_ALPHA_USE_SNAPSHOT=1 python scripts/run_frozen.py
```

## Research freeze policy

The v5 parameter set is frozen as of 2026-09-17. Future data may be used to evaluate it, but changing parameters in response to future results creates a new version and resets the prospective OOS clock.

## Disclaimer

Research software only. This repository is not investment advice and does not submit brokerage orders.
