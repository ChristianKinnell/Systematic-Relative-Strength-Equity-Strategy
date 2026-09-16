# Limitations

## 1. Survivorship bias — material
The historical simulation begins with companies returned by today's Yahoo Finance large-cap screen. Companies that were historically eligible but later delisted, were acquired, changed domicile, or fell below the current cap threshold are absent. The historical liquidity filter is point-in-time **within the surviving set**, not a point-in-time reconstruction of membership. Reported historical performance must therefore be described as research evidence, not an institutional-quality unbiased track record.

## 2. Data source
Yahoo Finance is convenient but not an institutional security master. Downloads can be incomplete or revised. The model retries downloads, rejects suspiciously small eligible universes, preserves per-name missing observations, fails if a held name has a missing return, and supports frozen price/universe snapshots.

## 3. Adjusted prices
Corporate-action-adjusted OHLC is used consistently for historical signal and P&L research. These values are not literal historical executable tape prices.

## 4. Transaction-cost approximation
The cost model is not a full execution simulator. It uses a fixed 10 bps per trade plus a square-root impact approximation. Queue position, intraday spread variation, auction liquidity, order type, partial fills and market microstructure are outside scope.

## 5. Current fundamentals
The fundamental veto uses current Yahoo metadata and is therefore excluded from historical backtesting. It should not be interpreted as a historically tested alpha source.

## 6. Regime-model uncertainty
HMM states are latent statistical classifications, not economic truths. Labels are based on forward SPX return ranking within the training sample. The macro/stress overlay is rule-based.

## 7. Small concentrated portfolio
Top-7 selection with a 15% cap creates a deliberately concentrated book. Sector caps reduce labelled sector concentration but do not guarantee low correlation. PCA/correlation diagnostics are monitoring tools rather than automatic trade vetoes.

## 8. Researcher degrees of freedom
The strategy was iterated during development. Historical folds that were inspected while changing the model are not independent OOS evidence. The freeze policy exists specifically to prevent this from being obscured.
