from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrawdownCircuitBreaker:
    stop: float = -0.08
    cooldown_days: int = 3
    peak_ref: float = 1.0
    frozen_days: int = 0
    resume_rebalance: bool = False

    def update_peak(self, equity: float) -> None:
        self.peak_ref = max(self.peak_ref, equity)

    def trigger_if_needed(self, equity: float) -> bool:
        if self.frozen_days > 0 or self.resume_rebalance:
            return False
        dd = equity / self.peak_ref - 1
        if dd < self.stop:
            self.frozen_days = self.cooldown_days
            return True
        return False

    def advance_frozen_day(self, equity: float) -> bool:
        if self.frozen_days <= 0:
            return False
        self.frozen_days -= 1
        if self.frozen_days == 0:
            self.resume_rebalance = True
            self.peak_ref = equity
            return True
        return False

    def consume_resume(self) -> None:
        self.resume_rebalance = False


@dataclass
class VolatilityHysteresis:
    low: float = 0.14
    high: float = 0.24
    derisked: bool = False

    def scale(self, realized_vol: float) -> float:
        if not self.derisked and realized_vol > self.high:
            self.derisked = True
        elif self.derisked and realized_vol < self.low:
            self.derisked = False
        return min(1.0, self.high / realized_vol) if self.derisked and realized_vol > 0 else 1.0
