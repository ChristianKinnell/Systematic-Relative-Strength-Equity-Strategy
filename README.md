# Systematic-Relative-Strength-Equity-Strategy

Minimal Python research and paper-execution framework for a long-only U.S. equity momentum strategy using:

- liquidity-filtered universe selection
- volatility-adjusted multi-horizon momentum and benchmark-relative strength
- trend confirmation before inclusion
- constrained inverse-volatility position sizing
- dynamic gross exposure from market regime and portfolio risk
- transaction-cost aware rebalance planning
- signal IC, ablations, parameter sensitivity, and walk-forward evaluation

## Files

- `momentum_framework.py` – core strategy, portfolio construction, execution plan, and robustness analytics
- `tests/test_momentum_framework.py` – focused regression tests for the framework

## Run tests

```bash
python -m unittest discover -s tests
```
