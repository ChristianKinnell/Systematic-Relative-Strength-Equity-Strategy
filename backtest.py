"""Backtest orchestration boundary.

The pure primitives have been extracted into package modules, but the canonical
portfolio loop remains in archive/frozen_v5_monolith.py until parity tests cover
weights, NAV, circuit-breaker dates, turnover, and target holdings. This staged
approach prevents a software refactor from becoming an unrecorded strategy change.
"""
