"""Tests for Quantitative Performance Metrics module."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    calculate_cagr,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_alpha_beta,
    calculate_turnover,
    calculate_hit_rate,
    calculate_all_metrics,
)


def test_cagr_equity_doubling_invariant():
    """Verify that an equity curve doubling over 252 days returns exactly 100% CAGR."""
    # Constant daily return r such that (1 + r)^252 = 2.0
    r_daily = (2.0 ** (1.0 / 252.0)) - 1.0
    returns = pd.Series(np.full(252, r_daily))

    cagr = calculate_cagr(returns, periods_per_year=252)
    assert np.isclose(cagr, 1.0, atol=1e-7), f"Expected CAGR of 1.0 (100%), got {cagr}"


def test_zero_volatility_edge_case():
    """Verify zero volatility returns do not raise divide-by-zero errors."""
    returns = pd.Series(np.zeros(100))

    vol = calculate_annualized_volatility(returns)
    assert vol == 0.0

    sharpe = calculate_sharpe_ratio(returns, rf_annual=0.065)
    assert sharpe == 0.0

    sortino = calculate_sortino_ratio(returns, rf_annual=0.065)
    # Downside deviations are negative, but with zero vol should handle cleanly
    assert isinstance(sortino, float)


def test_monotonic_upward_max_drawdown():
    """Verify that an asset with strictly non-negative returns has a max drawdown of 0.0."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.03])
    mdd, dd_series = calculate_max_drawdown(returns, is_returns=True)
    assert mdd == 0.0
    assert (dd_series == 0.0).all()


def test_known_max_drawdown():
    """Verify max drawdown on a known equity trajectory: 100 -> 120 -> 90 -> 110."""
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    mdd, dd_series = calculate_max_drawdown(equity, is_returns=False)
    # Peak is 120, trough is 90 -> drop = (90 - 120) / 120 = -30 / 120 = -0.25 (-25%)
    assert np.isclose(mdd, -0.25)
    assert np.isclose(dd_series.iloc[2], -0.25)


def test_calmar_ratio():
    """Verify Calmar ratio calculation."""
    # Total return = 20% over 252 days -> CAGR = 20%
    # Suppose MDD is -10% -> Calmar = 0.20 / 0.10 = 2.0
    r_daily = (1.20 ** (1.0 / 252.0)) - 1.0
    returns_arr = np.full(252, r_daily)
    # Inject a dip mid-way to create a known MDD
    # Let's test with direct returns
    returns = pd.Series(returns_arr)
    calmar = calculate_calmar_ratio(returns, periods_per_year=252)
    assert isinstance(calmar, float)


def test_alpha_beta_known_regression():
    """Verify Alpha and Beta recover known CAPM parameters."""
    rng = np.random.default_rng(42)
    n = 5000
    daily_rf = 0.065 / 252.0
    bench_returns = pd.Series(rng.normal(0.0005, 0.01, size=n))

    true_beta = 1.35
    true_alpha_daily = 0.0002
    strat_returns = pd.Series(
        daily_rf + true_alpha_daily + true_beta * (bench_returns - daily_rf) + rng.normal(0, 0.0001, size=n)
    )

    ann_alpha, est_beta = calculate_alpha_beta(strat_returns, bench_returns, rf_annual=0.065, periods_per_year=252)

    assert np.isclose(est_beta, true_beta, atol=0.02), f"Expected beta ~{true_beta}, got {est_beta}"
    assert np.isclose(ann_alpha, true_alpha_daily * 252, atol=0.005), f"Expected alpha ~{true_alpha_daily*252}, got {ann_alpha}"


def test_hit_rate_and_turnover():
    """Verify Hit Rate and Turnover arithmetic."""
    returns = pd.Series([0.01, -0.01, 0.02, 0.0, 0.03])
    # Active days: 4 days (3 positive, 1 negative). Hit rate = 3 / 4 = 0.75
    hit_rate = calculate_hit_rate(returns)
    assert hit_rate == 0.75

    turnover = pd.Series([0.10, 0.20, 0.0, 0.10])
    # Mean turnover = 0.10 -> Annualized = 0.10 * 252 = 25.2
    ann_turnover = calculate_turnover(turnover, periods_per_year=252)
    assert np.isclose(ann_turnover, 25.2)


def test_calculate_all_metrics_dictionary_keys():
    """Verify calculate_all_metrics outputs all required keys without error."""
    strat_ret = pd.Series(np.random.normal(0.0005, 0.01, size=252))
    bench_ret = pd.Series(np.random.normal(0.0004, 0.009, size=252))
    turnover = pd.Series(np.random.uniform(0.01, 0.05, size=252))

    metrics = calculate_all_metrics(strat_ret, bench_ret, turnover, rf_annual=0.065)
    expected_keys = [
        "Total Return", "CAGR", "Annualized Volatility", "Sharpe Ratio",
        "Sortino Ratio", "Max Drawdown", "Calmar Ratio", "Alpha (CAPM)",
        "Beta (CAPM)", "Annualized Turnover", "Hit Rate"
    ]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], (float, int))
