# Methodology

## 1. Universe
The research universe is generated from the current Yahoo Finance U.S. equity screen: NYSE/NASDAQ listings, market capitalisation above $25bn, then a U.S.-domicile metadata filter. A historical price/ADV/coverage/history screen is applied point-in-time within that surviving universe. This does **not** make membership itself point-in-time; see `LIMITATIONS.md`.

## 2. Alpha signal
For each stock, momentum is measured over approximately 1, 3, 6 and 12 months. Every leg skips the most recent 5 trading sessions. The weighted raw signal is:

`0.15*M1 + 0.30*M3 + 0.30*M6 + 0.15*M12`

The result is divided by trailing 63-session volatility and transformed to a cross-sectional percentile rank scaled to [-1, 1]. Six-month relative strength versus the S&P 500 is ranked separately. Final score:

`0.90*rank(vol-adjusted momentum) + 0.10*rank(relative strength)`

The score is exponentially smoothed with 5-session half-life.

## 3. Confirmation
A security must satisfy all of: price > 50DMA; 50DMA > 200DMA; six-month relative strength > 0; M3 > 0; M6 > 0; annualised 63-session volatility <= 65%.

## 4. Selection and sizing
Eligible positive-score securities are ranked and the top seven are selected. Fewer than seven qualifiers implies cash. Relative weights are inverse-volatility. The portfolio applies a 15% name cap and 45% sector cap. An 8 percentage-point rebalance band applies to continuing positions; mandatory exits bypass the band.

## 5. Regime model
A 3-state Gaussian HMM is fitted on daily SPX, VIX and LQD changes using an expanding training window, minimum 252 observations, and roughly quarterly refits. Features are standardised using training-window moments. Multiple deterministic restarts are tried and the best training likelihood is retained.

State labels are assigned using forward SPX returns that remain inside the training window. OOS segment states are generated with a causal forward filter P(S_t | X_1..X_t), not a full-segment Viterbi decode. Macro and fast-stress overlays can override the HMM label. Only `stress` changes exposure in v5, scaling gross exposure to 50%.

## 6. Portfolio risk
Trailing 21-session realised portfolio volatility uses 14%/24% hysteresis. Above 24%, exposure scales by `24% / realised_vol` until volatility falls below 14%. A drawdown circuit breaker triggers below -8% from the active peak, freezes exposure at 0% for exactly three sessions, resets the reference peak at resume, and rebalances at the final frozen day's close.

## 7. Transaction costs and timing
Signal decisions use information available at a session close and affect returns starting the next session. Costs are 10 bps per weight change plus a square-root market-impact term based on realised volatility, trade value and ADV. The risk engine operates on net-of-cost NAV.

## 8. Security-level risk and fundamentals
ATR(14) x 2.5 is shown as a manual reference stop; no broker order is placed. A current-fundamentals sanity/veto layer is applied only to final live targets and is excluded from historical performance because point-in-time fundamentals are unavailable in the Yahoo data used here.
