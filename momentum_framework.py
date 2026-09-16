from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class StrategyConfig:
    momentum_lookbacks: Tuple[int, ...] = (21, 63, 126)
    momentum_weights: Tuple[float, ...] = (0.5, 0.3, 0.2)
    volatility_lookback: int = 63
    trend_window: int = 100
    market_trend_window: int = 200
    liquidity_threshold: float = 1_000_000.0
    max_positions: int = 10
    min_weight: float = 0.0
    max_weight: float = 0.15
    target_portfolio_volatility: float = 0.18
    min_exposure: float = 0.25
    max_exposure: float = 1.0
    bearish_exposure: float = 0.5
    transaction_cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if len(self.momentum_lookbacks) != len(self.momentum_weights):
            raise ValueError("Momentum lookbacks and weights must have the same length.")
        if sum(self.momentum_weights) <= 0:
            raise ValueError("Momentum weights must sum to a positive value.")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive.")


@dataclass(frozen=True)
class SecuritySnapshot:
    symbol: str
    closes: Tuple[float, ...]
    average_dollar_volume: float
    benchmark_closes: Tuple[float, ...] = ()


@dataclass(frozen=True)
class RankedSecurity:
    symbol: str
    score: float
    risk_adjusted_momentum: float
    relative_strength: float
    volatility: float
    passes_trend: bool


@dataclass(frozen=True)
class RebalancePlan:
    target_weights: Dict[str, float]
    gross_exposure: float
    estimated_turnover: float
    estimated_transaction_cost: float


@dataclass(frozen=True)
class SensitivityResult:
    parameter: str
    value: object
    signal_ic: float


@dataclass(frozen=True)
class WalkForwardWindow:
    in_sample_universe: Tuple[SecuritySnapshot, ...]
    in_sample_future_returns: Mapping[str, float]
    out_of_sample_universe: Tuple[SecuritySnapshot, ...]
    out_of_sample_future_returns: Mapping[str, float]
    market_closes: Tuple[float, ...]


def _validate_prices(prices: Sequence[float], minimum_points: int) -> None:
    if len(prices) < minimum_points:
        raise ValueError(f"Expected at least {minimum_points} price points, received {len(prices)}.")


def _daily_returns(prices: Sequence[float]) -> List[float]:
    _validate_prices(prices, 2)
    returns: List[float] = []
    for previous, current in zip(prices[:-1], prices[1:]):
        if previous <= 0 or current <= 0:
            raise ValueError("Prices must be positive.")
        returns.append((current / previous) - 1.0)
    return returns


def _total_return(prices: Sequence[float], lookback: int) -> float:
    _validate_prices(prices, lookback + 1)
    return (prices[-1] / prices[-lookback - 1]) - 1.0


def realized_volatility(prices: Sequence[float], lookback: int) -> float:
    _validate_prices(prices, lookback + 1)
    trailing_returns = _daily_returns(prices[-(lookback + 1) :])
    if len(trailing_returns) < 2:
        return 0.0
    return pstdev(trailing_returns) * sqrt(TRADING_DAYS_PER_YEAR)


def market_regime_signal(market_closes: Sequence[float], lookback: int) -> bool:
    _validate_prices(market_closes, lookback)
    moving_average = mean(market_closes[-lookback:])
    return market_closes[-1] >= moving_average


def trend_confirmation(prices: Sequence[float], lookback: int) -> bool:
    _validate_prices(prices, lookback)
    moving_average = mean(prices[-lookback:])
    return prices[-1] >= moving_average


def _risk_adjusted_momentum(prices: Sequence[float], config: StrategyConfig) -> float:
    volatility = max(realized_volatility(prices, config.volatility_lookback), 1e-9)
    weighted_score = 0.0
    for lookback, weight in zip(config.momentum_lookbacks, config.momentum_weights):
        weighted_score += weight * (_total_return(prices, lookback) / volatility)
    return weighted_score


def _relative_strength(prices: Sequence[float], benchmark_closes: Sequence[float], config: StrategyConfig) -> float:
    if not benchmark_closes:
        return 0.0
    relative_score = 0.0
    for lookback, weight in zip(config.momentum_lookbacks, config.momentum_weights):
        relative_score += weight * (
            _total_return(prices, lookback) - _total_return(benchmark_closes, lookback)
        )
    return relative_score


def rank_universe(universe: Sequence[SecuritySnapshot], config: StrategyConfig) -> List[RankedSecurity]:
    ranked: List[RankedSecurity] = []
    for security in universe:
        if security.average_dollar_volume < config.liquidity_threshold:
            continue
        passes_trend = trend_confirmation(security.closes, config.trend_window)
        if not passes_trend:
            continue
        volatility = realized_volatility(security.closes, config.volatility_lookback)
        risk_adjusted_momentum = _risk_adjusted_momentum(security.closes, config)
        relative_strength = _relative_strength(security.closes, security.benchmark_closes, config)
        ranked.append(
            RankedSecurity(
                symbol=security.symbol,
                score=risk_adjusted_momentum + relative_strength,
                risk_adjusted_momentum=risk_adjusted_momentum,
                relative_strength=relative_strength,
                volatility=volatility,
                passes_trend=passes_trend,
            )
        )
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def _cap_weights(raw_weights: Mapping[str, float], config: StrategyConfig) -> Dict[str, float]:
    remaining = dict(raw_weights)
    capped: Dict[str, float] = {}
    remaining_weight = 1.0

    while remaining:
        total_raw = sum(remaining.values())
        if total_raw <= 0:
            break
        newly_capped = {
            symbol: remaining_weight * value / total_raw
            for symbol, value in remaining.items()
            if remaining_weight * value / total_raw > config.max_weight
        }
        if not newly_capped:
            break
        for symbol in newly_capped:
            capped[symbol] = config.max_weight
            remaining_weight -= config.max_weight
            remaining.pop(symbol)
        if remaining_weight <= 0:
            return capped

    if remaining:
        total_raw = sum(remaining.values())
        for symbol, value in remaining.items():
            capped[symbol] = remaining_weight * value / total_raw

    if config.min_weight <= 0:
        return capped

    filtered = {symbol: weight for symbol, weight in capped.items() if weight >= config.min_weight}
    if not filtered:
        return capped
    scale = 1.0 / sum(filtered.values())
    return _cap_weights({symbol: weight * scale for symbol, weight in filtered.items()}, replace(config, min_weight=0.0))


def construct_inverse_volatility_portfolio(
    ranked_universe: Sequence[RankedSecurity], config: StrategyConfig, gross_exposure: float = 1.0
) -> Dict[str, float]:
    selected = list(ranked_universe[: config.max_positions])
    if not selected:
        return {}
    inverse_vol = {
        security.symbol: 1.0 / max(security.volatility, 1e-9)
        for security in selected
    }
    normalized = _cap_weights(inverse_vol, config)
    return {symbol: weight * gross_exposure for symbol, weight in normalized.items()}


def dynamic_gross_exposure(
    market_closes: Sequence[float], portfolio_volatility: float, config: StrategyConfig
) -> float:
    bullish = market_regime_signal(market_closes, config.market_trend_window)
    base_exposure = config.max_exposure if bullish else config.bearish_exposure
    if portfolio_volatility <= 0:
        scaled_exposure = base_exposure
    else:
        scaled_exposure = base_exposure * min(1.0, config.target_portfolio_volatility / portfolio_volatility)
    return min(config.max_exposure, max(config.min_exposure, scaled_exposure))


def estimated_turnover(current_weights: Mapping[str, float], target_weights: Mapping[str, float]) -> float:
    symbols = set(current_weights) | set(target_weights)
    return sum(abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in symbols) / 2.0


def estimated_transaction_cost(turnover: float, config: StrategyConfig) -> float:
    return turnover * (config.transaction_cost_bps / 10_000.0)


def generate_rebalance_plan(
    universe: Sequence[SecuritySnapshot],
    market_closes: Sequence[float],
    config: StrategyConfig,
    current_weights: Mapping[str, float] | None = None,
) -> RebalancePlan:
    ranked = rank_universe(universe, config)
    gross_exposure = dynamic_gross_exposure(
        market_closes,
        portfolio_volatility=mean([security.volatility for security in ranked[: config.max_positions]]) if ranked else 0.0,
        config=config,
    )
    target_weights = construct_inverse_volatility_portfolio(ranked, config, gross_exposure=gross_exposure)
    current_weights = current_weights or {}
    turnover = estimated_turnover(current_weights, target_weights)
    return RebalancePlan(
        target_weights=target_weights,
        gross_exposure=gross_exposure,
        estimated_turnover=turnover,
        estimated_transaction_cost=estimated_transaction_cost(turnover, config),
    )


def _ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        tie_end = position
        while tie_end + 1 < len(indexed) and indexed[tie_end + 1][1] == indexed[position][1]:
            tie_end += 1
        average_rank = (position + tie_end + 2) / 2.0
        for tie_position in range(position, tie_end + 1):
            ranks[indexed[tie_position][0]] = average_rank
        position = tie_end + 1
    return ranks


def signal_information_coefficient(predicted_scores: Mapping[str, float], future_returns: Mapping[str, float]) -> float:
    overlapping_symbols = [symbol for symbol in predicted_scores if symbol in future_returns]
    if len(overlapping_symbols) < 2:
        return 0.0
    ranked_scores = _ranks([predicted_scores[symbol] for symbol in overlapping_symbols])
    ranked_returns = _ranks([future_returns[symbol] for symbol in overlapping_symbols])
    mean_scores = mean(ranked_scores)
    mean_returns = mean(ranked_returns)
    numerator = sum(
        (score - mean_scores) * (ret - mean_returns)
        for score, ret in zip(ranked_scores, ranked_returns)
    )
    denominator_left = sqrt(sum((score - mean_scores) ** 2 for score in ranked_scores))
    denominator_right = sqrt(sum((ret - mean_returns) ** 2 for ret in ranked_returns))
    denominator = denominator_left * denominator_right
    if denominator == 0:
        return 0.0
    return numerator / denominator


def run_ablation_study(
    universe: Sequence[SecuritySnapshot], future_returns: Mapping[str, float], config: StrategyConfig
) -> Dict[str, float]:
    baseline = rank_universe(universe, config)
    baseline_scores = {security.symbol: security.score for security in baseline}

    no_relative_strength_scores = {
        security.symbol: security.risk_adjusted_momentum for security in baseline
    }
    no_vol_adjustment_scores = {}
    for security in universe:
        if security.average_dollar_volume >= config.liquidity_threshold and trend_confirmation(
            security.closes, config.trend_window
        ):
            no_vol_adjustment_scores[security.symbol] = sum(
                weight * _total_return(security.closes, lookback)
                for lookback, weight in zip(config.momentum_lookbacks, config.momentum_weights)
            )

    relaxed_trend_config = replace(config, trend_window=min(config.trend_window, min(len(s.closes) for s in universe)))
    no_trend_scores = {}
    for security in universe:
        if security.average_dollar_volume < config.liquidity_threshold:
            continue
        score = _risk_adjusted_momentum(security.closes, relaxed_trend_config) + _relative_strength(
            security.closes, security.benchmark_closes, relaxed_trend_config
        )
        no_trend_scores[security.symbol] = score

    return {
        "baseline": signal_information_coefficient(baseline_scores, future_returns),
        "no_relative_strength": signal_information_coefficient(no_relative_strength_scores, future_returns),
        "no_volatility_adjustment": signal_information_coefficient(no_vol_adjustment_scores, future_returns),
        "no_trend_filter": signal_information_coefficient(no_trend_scores, future_returns),
    }


def parameter_sensitivity(
    universe: Sequence[SecuritySnapshot],
    future_returns: Mapping[str, float],
    base_config: StrategyConfig,
    parameter_grid: Mapping[str, Iterable[object]],
) -> List[SensitivityResult]:
    results: List[SensitivityResult] = []
    for parameter, values in parameter_grid.items():
        for value in values:
            config = replace(base_config, **{parameter: value})
            ranked = rank_universe(universe, config)
            scores = {security.symbol: security.score for security in ranked}
            results.append(
                SensitivityResult(
                    parameter=parameter,
                    value=value,
                    signal_ic=signal_information_coefficient(scores, future_returns),
                )
            )
    return results


def walk_forward_validation(windows: Sequence[WalkForwardWindow], config: StrategyConfig) -> Dict[str, float]:
    if not windows:
        return {"average_in_sample_ic": 0.0, "average_out_of_sample_ic": 0.0, "average_gross_exposure": 0.0}

    in_sample_ics: List[float] = []
    out_of_sample_ics: List[float] = []
    exposures: List[float] = []

    for window in windows:
        in_sample_ranked = rank_universe(window.in_sample_universe, config)
        out_of_sample_ranked = rank_universe(window.out_of_sample_universe, config)
        in_sample_ics.append(
            signal_information_coefficient(
                {security.symbol: security.score for security in in_sample_ranked},
                window.in_sample_future_returns,
            )
        )
        out_of_sample_ics.append(
            signal_information_coefficient(
                {security.symbol: security.score for security in out_of_sample_ranked},
                window.out_of_sample_future_returns,
            )
        )
        exposures.append(
            dynamic_gross_exposure(
                window.market_closes,
                portfolio_volatility=mean(
                    [security.volatility for security in out_of_sample_ranked[: config.max_positions]]
                )
                if out_of_sample_ranked
                else 0.0,
                config=config,
            )
        )

    return {
        "average_in_sample_ic": mean(in_sample_ics),
        "average_out_of_sample_ic": mean(out_of_sample_ics),
        "average_gross_exposure": mean(exposures),
    }
