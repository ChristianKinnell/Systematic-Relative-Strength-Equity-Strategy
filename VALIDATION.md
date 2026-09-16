# Validation Framework

The objective is to test whether the strategy's behaviour is stable and causally defensible, not merely to maximise one backtest Sharpe ratio.

## Portfolio-level evidence
- CAGR, annualised volatility, Sharpe, Sortino, Calmar, maximum drawdown.
- Daily CVaR(95%), turnover and estimated transaction costs.
- Market-factor beta, idiosyncratic volatility and R².
- Pairwise correlation, variance contribution and first-principal-component concentration.

## Alpha-specific evidence
- Spearman rank IC at 1-week, 1-month and 3-month horizons.
- IC decay at 5/10/20/40/60 trading days.
- Forward-return quintiles and monotonicity.
- Rank versus winsorised-z-score comparison.
- Momentum-horizon weighting diagnostics.

## Ablations
The research script compares confirmation gates, regime variants, HMM state count, top-N, sizing approaches, risk thresholds, ATR stops, volatility controls, and a stacked module path from momentum-only through the full risk-controlled portfolio.

## Temporal robustness
1. **Combinatorial time-slice stability:** slices the already net-of-cost strategy return path into combinations of test periods. It is not labelled CPCV/PBO.
2. **Fixed-config expanding-year check:** evaluates the same frozen configuration across successive annual periods. This is temporal consistency, not independent evidence if the configuration was shaped using those dates.
3. **Scoped walk-forward optimisation:** within each fold, searches only top-N, vol ceiling and ATR multiplier on the training history, freezes the selected candidate, then measures the next year. Candidate equity is already net of costs; costs must not be subtracted a second time.
4. **Recent-period robustness:** the last 12 months are reported descriptively but explicitly acknowledged as development data.

## Prospective OOS rule
The v5 research freeze is 2026-09-17. Only data strictly after the freeze, evaluated without parameter changes, qualifies as prospective OOS evidence. If future evidence causes a parameter change, preserve v5 and create a new version.
