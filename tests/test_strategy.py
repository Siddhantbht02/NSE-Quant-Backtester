"""Tests for Strategy Framework (Momentum & Mean Reversion)."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import MomentumStrategy, MeanReversionStrategy


def test_momentum_strategy_weight_constraints():
    """Verify MomentumStrategy weights never exceed 1.0 row sum and have correct shape."""
    dates = pd.bdate_range("2018-01-01", periods=320)
    rng = np.random.default_rng(42)
    tickers = [f"STOCK_{i}.NS" for i in range(15)]
    data = {}
    for t in tickers:
        ret = rng.normal(0.0005, 0.015, size=320)
        data[t] = 100.0 * np.exp(np.cumsum(ret))
    prices = pd.DataFrame(data, index=dates)

    strat = MomentumStrategy(lookback_days=252, skip_days=21, top_quantile=0.20, rebalance_freq="monthly")
    weights = strat.generate_signals(prices)

    # Shape verification
    assert weights.shape == prices.shape
    assert (weights.index == prices.index).all()
    assert (weights.columns == prices.columns).all()

    # Before lookback_days, weights must be 0
    assert (weights.iloc[:252] == 0.0).all().all()

    # After lookback_days, row sums must be <= 1.000001
    row_sums = weights.abs().sum(axis=1)
    assert (row_sums <= 1.000001).all()

    # Active weights should be non-negative (long only)
    assert (weights >= 0.0).all().all()


def test_momentum_strategy_selection_logic():
    """Verify that stock with highest 12-1 momentum is indeed chosen in the top quintile."""
    dates = pd.bdate_range("2018-01-01", periods=280)
    # Create 5 stocks. Stock A grows 500% over the period, others stay flat
    data = {
        "STOCK_WINNER.NS": np.linspace(100, 600, 280),
        "STOCK_FLAT1.NS": np.full(280, 100.0),
        "STOCK_FLAT2.NS": np.full(280, 100.0),
        "STOCK_FLAT3.NS": np.full(280, 100.0),
        "STOCK_FLAT4.NS": np.full(280, 100.0),
    }
    prices = pd.DataFrame(data, index=dates)

    strat = MomentumStrategy(lookback_days=252, skip_days=21, top_quantile=0.20, rebalance_freq="monthly")
    weights = strat.generate_signals(prices)

    # After month-end rebalance following day 252, WINNER must have weight > 0
    # Top 20% of 5 stocks = 1 stock
    active_rows = weights.iloc[265:]
    assert (active_rows["STOCK_WINNER.NS"] == 1.0).any()


def test_mean_reversion_entry_and_exit():
    """Verify MeanReversionStrategy triggers at Z < -1.5 and exits at Z >= 0.0."""
    dates = pd.bdate_range("2020-01-01", periods=60)
    # Construct synthetic price path:
    # Days 0-24: flat at 100 (SMA ~ 100, std small)
    # Day 25: sharp plunge to 80 (Z drops well below -1.5)
    # Day 26-28: stays low
    # Day 29: bounces back to 105 (Z >= 0)
    p = np.full(60, 100.0)
    # Add minor noise so std is not 0
    p[:25] = 100.0 + np.sin(np.arange(25)) * 0.5
    p[25:29] = 85.0
    p[29:] = 105.0

    prices = pd.DataFrame({"STOCK_TEST.NS": p}, index=dates)
    strat = MeanReversionStrategy(window=20, entry_z=-1.5, exit_z=0.0, weighting="equal")
    weights = strat.generate_signals(prices)

    # Before plunge (days 0-24), no trigger
    assert weights["STOCK_TEST.NS"].iloc[:24].sum() == 0.0

    # At plunge (day 25), Z drops below -1.5 -> weight must become 1.0
    assert weights["STOCK_TEST.NS"].iloc[25] == 1.0
    # Must persist on days 26-28 while price stays depressed
    assert weights["STOCK_TEST.NS"].iloc[26] == 1.0
    assert weights["STOCK_TEST.NS"].iloc[27] == 1.0

    # When price rebounds to 105 at day 29, Z becomes positive -> exit (weight becomes 0.0)
    assert weights["STOCK_TEST.NS"].iloc[30] == 0.0


def test_mean_reversion_row_sums():
    """Verify MeanReversionStrategy respects sum(|w_i|) <= 1.0 across multi-asset universe."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(77)
    tickers = [f"STOCK_{i}.NS" for i in range(8)]
    data = {t: 100.0 + np.cumsum(rng.normal(0, 2, size=100)) for t in tickers}
    prices = pd.DataFrame(data, index=dates)

    strat = MeanReversionStrategy(window=20, entry_z=-1.0, exit_z=0.0, weighting="equal")
    weights = strat.generate_signals(prices)

    row_sums = weights.abs().sum(axis=1)
    assert (row_sums <= 1.000001).all()

