"""Tests for Vectorised Backtester module (Look-Ahead Bias & Friction Verification)."""

import numpy as np
import pandas as pd
import pytest

from src.backtester import Backtester
from src.strategy import Strategy


class DummyJumpStrategy(Strategy):
    """Strategy that generates a buy signal only AFTER observing a price jump on day T."""

    def __init__(self, jump_threshold: float = 0.5):
        self.jump_threshold = jump_threshold

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        daily_ret = prices.pct_change(fill_method=None).fillna(0.0)
        # Generates signal at close of day t where return > jump_threshold
        signal = (daily_ret > self.jump_threshold).astype(float)
        return signal


def test_look_ahead_prevention_on_price_jump():
    """Verify that an algorithm reacting to a Day T jump earns 0% on Day T.

    Mathematical Invariant:
    A price jump occurring between Day T-1 and Day T cannot be captured by an algorithm
    that generates its signal at the close of Day T. The executed weight on Day T MUST
    be based on Day T-1 target weight.
    """
    dates = pd.bdate_range("2021-01-01", periods=6)
    # Day 0: 100
    # Day 1: 100
    # Day 2: 100
    # Day 3: 200  (100% price jump occurs on Day 3!)
    # Day 4: 200
    # Day 5: 200
    prices = pd.DataFrame({"STOCK_JUMP.NS": [100.0, 100.0, 100.0, 200.0, 200.0, 200.0]}, index=dates)

    strat = DummyJumpStrategy(jump_threshold=0.50)
    # Target signal on Day 3 will be 1.0 because return on Day 3 is 100%
    target_weights = strat.generate_signals(prices)
    assert target_weights["STOCK_JUMP.NS"].iloc[3] == 1.0
    assert target_weights["STOCK_JUMP.NS"].iloc[2] == 0.0

    engine = Backtester(cost_per_side=0.0, rf_annual=0.0)
    result = engine.run(prices, strat)
    hist = result.portfolio_history

    # Executed weight on Day 3 MUST be 0.0 (from target weight on Day 2)
    assert result.executed_weights["STOCK_JUMP.NS"].iloc[3] == 0.0, "Executed weight on Day 3 must be 0!"

    # Gross & Net return on Day 3 MUST be exactly 0.0 (DID NOT capture the jump in hindsight)
    assert hist["gross_return"].iloc[3] == 0.0, "Day 3 return must be 0.0 (no look-ahead allowed)"
    assert hist["net_return"].iloc[3] == 0.0

    # Executed weight on Day 4 becomes 1.0 (traded after seeing Day 3 close)
    assert result.executed_weights["STOCK_JUMP.NS"].iloc[4] == 1.0
    # Day 4 price is flat (200 -> 200), so Day 4 return is 0.0
    assert hist["gross_return"].iloc[4] == 0.0

    # Total return over the entire backtest must be 0.0!
    assert hist["equity_curve"].iloc[-1] == engine.initial_capital


def test_transaction_cost_arithmetic_on_100_percent_reallocation():
    """Verify that an immediate 100% reallocation from Stock A to Stock B charges exactly 200% turnover * fee."""
    dates = pd.bdate_range("2021-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "STOCK_A.NS": [100.0, 100.0, 100.0, 100.0, 100.0],
            "STOCK_B.NS": [50.0, 50.0, 50.0, 50.0, 50.0],
        },
        index=dates,
    )

    # Construct target weights:
    # Day 0: 100% Stock A, 0% Stock B
    # Day 1: 0% Stock A, 100% Stock B (Reallocation signal)
    # Day 2: 0% Stock A, 100% Stock B
    # Day 3: 0% Stock A, 100% Stock B
    # Day 4: 0% Stock A, 100% Stock B
    target_weights = pd.DataFrame(
        [
            [1.0, 0.0],  # Day 0
            [0.0, 1.0],  # Day 1: signal flip
            [0.0, 1.0],  # Day 2
            [0.0, 1.0],  # Day 3
            [0.0, 1.0],  # Day 4
        ],
        index=dates,
        columns=["STOCK_A.NS", "STOCK_B.NS"],
    )

    fee_per_side = 0.0015  # 15 bps
    engine = Backtester(cost_per_side=fee_per_side, rf_annual=0.0)
    result = engine.run(prices, target_weights)
    hist = result.portfolio_history

    # Executed weights:
    # Day 1: [1.0, 0.0] (from Day 0) -> Turnover on Day 1: 1.0 (buying Stock A)
    # Day 2: [0.0, 1.0] (from Day 1) -> Turnover on Day 2: |0 - 1| + |1 - 0| = 2.0 (200%!)
    turnover_day2 = hist["turnover"].iloc[2]
    assert np.isclose(turnover_day2, 2.0), f"Expected turnover of 2.0 (200%), got {turnover_day2}"

    cost_day2 = hist["transaction_cost"].iloc[2]
    expected_cost = 2.0 * fee_per_side
    assert np.isclose(cost_day2, expected_cost), f"Expected cost of {expected_cost}, got {cost_day2}"

    # Net return on Day 2 (since prices are flat) must equal -cost_day2
    assert np.isclose(hist["net_return"].iloc[2], -expected_cost)


def test_execution_lag_alignment():
    """Verify executed_weights is exactly target_weights.shift(1).fillna(0.0)."""
    dates = pd.bdate_range("2021-01-01", periods=10)
    prices = pd.DataFrame({"A.NS": np.full(10, 100.0), "B.NS": np.full(10, 50.0)}, index=dates)

    target_weights = pd.DataFrame(np.random.rand(10, 2), index=dates, columns=["A.NS", "B.NS"])
    target_weights = target_weights.div(target_weights.sum(axis=1), axis=0)

    engine = Backtester()
    result = engine.run(prices, target_weights)

    expected_executed = target_weights.shift(1).fillna(0.0)
    pd.testing.assert_frame_equal(result.executed_weights, expected_executed)

