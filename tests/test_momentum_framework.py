import unittest

from momentum_framework import (
    SecuritySnapshot,
    StrategyConfig,
    WalkForwardWindow,
    construct_inverse_volatility_portfolio,
    dynamic_gross_exposure,
    generate_rebalance_plan,
    parameter_sensitivity,
    rank_universe,
    run_ablation_study,
    signal_information_coefficient,
    walk_forward_validation,
)


def price_series(daily_return: float, days: int = 260, start: float = 100.0, shock_day: int | None = None, shock: float = 0.0):
    prices = [start]
    for day in range(1, days):
        value = prices[-1] * (1 + daily_return)
        if shock_day is not None and day == shock_day:
            value *= 1 + shock
        prices.append(value)
    return tuple(prices)


class MomentumFrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = StrategyConfig(
            liquidity_threshold=5_000_000.0,
            max_positions=2,
            max_weight=0.6,
            trend_window=50,
            market_trend_window=100,
            volatility_lookback=21,
            momentum_lookbacks=(21, 63, 126),
            momentum_weights=(0.5, 0.3, 0.2),
        )
        benchmark = price_series(0.0015)
        self.universe = (
            SecuritySnapshot(
                symbol="LEADER",
                closes=price_series(0.0022),
                average_dollar_volume=15_000_000.0,
                benchmark_closes=benchmark,
            ),
            SecuritySnapshot(
                symbol="STEADY",
                closes=price_series(0.0019, shock_day=200, shock=-0.01),
                average_dollar_volume=12_000_000.0,
                benchmark_closes=benchmark,
            ),
            SecuritySnapshot(
                symbol="ILLQ",
                closes=price_series(0.0030),
                average_dollar_volume=1_000_000.0,
                benchmark_closes=benchmark,
            ),
            SecuritySnapshot(
                symbol="BROKEN",
                closes=price_series(-0.0002),
                average_dollar_volume=20_000_000.0,
                benchmark_closes=benchmark,
            ),
        )
        self.future_returns = {"LEADER": 0.05, "STEADY": 0.03, "ILLQ": 0.04, "BROKEN": -0.02}

    def test_rank_universe_filters_and_orders_candidates(self):
        ranked = rank_universe(self.universe, self.config)

        self.assertEqual([security.symbol for security in ranked], ["LEADER", "STEADY"])
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_inverse_volatility_portfolio_respects_caps(self):
        ranked = rank_universe(self.universe, self.config)

        weights = construct_inverse_volatility_portfolio(ranked, self.config, gross_exposure=0.8)

        self.assertAlmostEqual(sum(weights.values()), 0.8, places=8)
        self.assertLessEqual(max(weights.values()), 0.8 * self.config.max_weight + 1e-9)
        self.assertGreater(weights["LEADER"], 0.0)
        self.assertGreater(weights["STEADY"], 0.0)

    def test_dynamic_exposure_reduces_in_bearish_high_risk_regime(self):
        bullish_market = price_series(0.001, days=260)
        bearish_market = price_series(-0.0005, days=260)

        bullish = dynamic_gross_exposure(bullish_market, portfolio_volatility=0.12, config=self.config)
        bearish = dynamic_gross_exposure(bearish_market, portfolio_volatility=0.30, config=self.config)

        self.assertGreater(bullish, bearish)
        self.assertGreaterEqual(bearish, self.config.min_exposure)

    def test_rebalance_plan_estimates_turnover_and_costs(self):
        market = price_series(0.001)

        plan = generate_rebalance_plan(
            self.universe,
            market,
            self.config,
            current_weights={"STEADY": 0.2},
        )

        self.assertGreater(plan.gross_exposure, 0.0)
        self.assertAlmostEqual(sum(plan.target_weights.values()), plan.gross_exposure, places=8)
        self.assertGreater(plan.estimated_turnover, 0.0)
        self.assertGreater(plan.estimated_transaction_cost, 0.0)

    def test_signal_diagnostics_and_robustness_outputs(self):
        ranked = rank_universe(self.universe, self.config)
        scores = {security.symbol: security.score for security in ranked}

        ic = signal_information_coefficient(scores, self.future_returns)
        ablations = run_ablation_study(self.universe, self.future_returns, self.config)
        sensitivity = parameter_sensitivity(
            self.universe,
            self.future_returns,
            self.config,
            parameter_grid={"trend_window": [30, 50], "max_positions": [1, 2]},
        )

        self.assertGreater(ic, 0.0)
        self.assertIn("baseline", ablations)
        self.assertEqual(len(sensitivity), 4)

    def test_walk_forward_validation_reports_out_of_sample_metrics(self):
        market = price_series(0.0012)
        window = WalkForwardWindow(
            in_sample_universe=self.universe[:2],
            in_sample_future_returns={"LEADER": 0.04, "STEADY": 0.02},
            out_of_sample_universe=self.universe[:2],
            out_of_sample_future_returns={"LEADER": 0.03, "STEADY": 0.01},
            market_closes=market,
        )

        results = walk_forward_validation((window,), self.config)

        self.assertGreaterEqual(results["average_in_sample_ic"], 0.0)
        self.assertGreaterEqual(results["average_out_of_sample_ic"], 0.0)
        self.assertGreater(results["average_gross_exposure"], 0.0)


if __name__ == "__main__":
    unittest.main()
