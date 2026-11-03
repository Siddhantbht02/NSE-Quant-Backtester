"""Quantitative & Performance Metrics module for nse-backtester.

Provides standalone, purely functional calculation routines:
- CAGR (Compound Annual Growth Rate)
- Annualized Volatility
- Sharpe Ratio (calibrated to Indian Risk-Free Rate, default 6.5%)
- Sortino Ratio (downside semivariance)
- Maximum Drawdown (MDD) & Drawdown Series
- Calmar Ratio
- CAPM Alpha & Beta against Benchmark (^NSEI)
- Annualized Turnover & Daily Hit Rate
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


def calculate_cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).

    Formula:
        Total Return = prod(1 + R_t) - 1.0
        CAGR = (1 + Total Return) ** (periods_per_year / N) - 1.0

    Args:
        returns: Daily net returns series.
        periods_per_year: Trading days per year (default 252).

    Returns:
        CAGR as a float (e.g. 0.15 for 15%).
    """
    clean_r = returns.dropna()
    n = len(clean_r)
    if n == 0:
        return 0.0

    total_return = np.prod(1.0 + clean_r.to_numpy()) - 1.0
    if total_return <= -1.0:
        return -1.0

    cagr = (1.0 + total_return) ** (periods_per_year / n) - 1.0
    return float(cagr)


def calculate_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility of returns.

    Formula:
        sigma_annual = std(R_daily, ddof=1) * sqrt(periods_per_year)

    Args:
        returns: Daily net returns series.
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualized volatility as a float.
    """
    clean_r = returns.dropna()
    if len(clean_r) < 2:
        return 0.0

    daily_std = clean_r.std(ddof=1)
    if np.isnan(daily_std) or daily_std <= 0.0:
        return 0.0

    return float(daily_std * np.sqrt(periods_per_year))


def calculate_sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.065,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe Ratio using an annualized risk-free rate.

    Formula:
        Sharpe = ((mean(R_daily) - R_f / periods_per_year) / std(R_daily)) * sqrt(periods_per_year)

    Args:
        returns: Daily net returns series.
        rf_annual: Annualized Indian Risk-Free Rate (default 0.065 = 6.5%).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualized Sharpe Ratio as float.
    """
    clean_r = returns.dropna()
    if len(clean_r) < 2:
        return 0.0

    daily_rf = rf_annual / periods_per_year
    daily_std = clean_r.std(ddof=1)

    if daily_std <= 1e-12 or np.isnan(daily_std):
        return 0.0

    excess_mean = clean_r.mean() - daily_rf
    sharpe = (excess_mean / daily_std) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: pd.Series,
    rf_annual: float = 0.065,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino Ratio based on downside deviation below the risk-free rate.

    Formula:
        D_t = R_t - R_f,daily
        Downside Vol = sqrt( (1 / N) * sum( min(0, D_t)^2 ) ) * sqrt(periods_per_year)
        Sortino = ( (mean(R_t) - R_f,daily) * sqrt(periods_per_year) ) / Downside Vol

    Args:
        returns: Daily net returns series.
        rf_annual: Annualized Indian Risk-Free Rate (default 0.065).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualized Sortino Ratio as float.
    """
    clean_r = returns.dropna()
    n = len(clean_r)
    if n < 2:
        return 0.0

    daily_rf = rf_annual / periods_per_year
    excess_r = clean_r.to_numpy() - daily_rf
    downside_deviations = np.minimum(0.0, excess_r)

    downside_variance = np.sum(downside_deviations**2) / n
    if downside_variance <= 1e-14:
        # If no downside deviation and mean excess is positive
        mean_excess = np.mean(excess_r)
        return float(np.inf if mean_excess > 0 else 0.0)

    downside_vol = np.sqrt(downside_variance) * np.sqrt(periods_per_year)
    sortino = (np.mean(excess_r) * np.sqrt(periods_per_year)) / (downside_vol / np.sqrt(periods_per_year))
    # Equivalently: (mean_excess * periods_per_year) / downside_vol
    sortino_ratio = (np.mean(excess_r) * periods_per_year) / downside_vol
    return float(sortino_ratio)


def calculate_max_drawdown(
    equity_or_returns: Union[pd.Series, np.ndarray],
    is_returns: bool = False,
) -> Tuple[float, pd.Series]:
    """Calculate Maximum Drawdown (MDD) and underwater drawdown series.

    Formula:
        Peak_t = max_{s <= t}(V_s)
        Drawdown_t = (V_t - Peak_t) / Peak_t
        MDD = min(Drawdown_t)  (negative decimal, e.g. -0.22)

    Args:
        equity_or_returns: Equity curve or daily return series.
        is_returns: If True, treats input as daily returns and computes equity curve.

    Returns:
        Tuple of (max_drawdown: float, drawdown_series: pd.Series).
    """
    if isinstance(equity_or_returns, np.ndarray):
        series = pd.Series(equity_or_returns)
    else:
        series = equity_or_returns.copy()

    series = series.dropna()
    if series.empty:
        return 0.0, pd.Series(dtype=float)

    if is_returns:
        equity = (1.0 + series).cumprod()
    else:
        equity = series

    peaks = equity.cummax()
    drawdown = (equity - peaks) / peaks.replace(0.0, np.nan)
    drawdown = drawdown.fillna(0.0)

    mdd = float(drawdown.min())
    return mdd, drawdown


def calculate_calmar_ratio(
    returns: pd.Series,
    rf_annual: float = 0.065,
    periods_per_year: int = 252,
) -> float:
    """Calculate Calmar Ratio: CAGR / |Max Drawdown|.

    Args:
        returns: Daily net returns series.
        rf_annual: Risk free rate (unused directly in standard Calmar, kept for signature consistency).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Calmar Ratio as float.
    """
    cagr = calculate_cagr(returns, periods_per_year=periods_per_year)
    mdd, _ = calculate_max_drawdown(returns, is_returns=True)
    abs_mdd = abs(mdd)

    if abs_mdd <= 1e-12:
        return float(np.inf if cagr > 0 else 0.0)

    return float(cagr / abs_mdd)


def calculate_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = 0.065,
    periods_per_year: int = 252,
) -> Tuple[float, float]:
    """Calculate CAPM Annualized Alpha and Beta against the benchmark.

    Regression model:
        (R_strat,t - R_f,daily) = alpha_daily + beta * (R_bench,t - R_f,daily) + epsilon_t
        Annualized Alpha = alpha_daily * periods_per_year

    Args:
        strategy_returns: Strategy daily net returns.
        benchmark_returns: Benchmark daily returns (^NSEI).
        rf_annual: Annualized Indian Risk-Free Rate (default 0.065).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Tuple of (annualized_alpha: float, beta: float).
    """
    df = pd.DataFrame(
        {"strategy": strategy_returns, "benchmark": benchmark_returns}
    ).dropna()

    if len(df) < 5:
        return 0.0, 1.0

    daily_rf = rf_annual / periods_per_year
    y = df["strategy"].to_numpy() - daily_rf
    x = df["benchmark"].to_numpy() - daily_rf

    var_x = np.var(x, ddof=1)
    if var_x <= 1e-14:
        return 0.0, 0.0

    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    beta = float(cov_xy / var_x)

    daily_alpha = float(np.mean(y) - beta * np.mean(x))
    annualized_alpha = float(daily_alpha * periods_per_year)

    return annualized_alpha, beta


def calculate_turnover(turnover_series: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized portfolio turnover.

    Formula:
        Annualized Turnover = mean(Daily Turnover) * periods_per_year

    Args:
        turnover_series: Daily turnover series (sum(|w_{i, t} - w_{i, t-1}|)).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualized turnover as float (e.g. 2.4 = 240% annual turnover).
    """
    clean_t = turnover_series.dropna()
    if clean_t.empty:
        return 0.0
    return float(clean_t.mean() * periods_per_year)


def calculate_hit_rate(returns: pd.Series) -> float:
    """Calculate percentage of positive return days over active trading days.

    Args:
        returns: Daily net returns series.

    Returns:
        Hit rate between 0.0 and 1.0.
    """
    clean_r = returns.dropna()
    active_days = clean_r[clean_r != 0.0]
    if len(active_days) == 0:
        return 0.0

    win_days = (active_days > 0.0).sum()
    return float(win_days / len(active_days))


def calculate_all_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    turnover_series: Optional[pd.Series] = None,
    rf_annual: float = 0.065,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Calculate comprehensive suite of quantitative performance metrics.

    Args:
        strategy_returns: Strategy daily net returns.
        benchmark_returns: Optional benchmark returns (^NSEI).
        turnover_series: Optional daily turnover series.
        rf_annual: Annualized risk-free rate (default 0.065 = 6.5%).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Dictionary mapping metric names to quantitative values.
    """
    clean_strat = strategy_returns.dropna()
    total_return = float(np.prod(1.0 + clean_strat.to_numpy()) - 1.0) if len(clean_strat) > 0 else 0.0

    cagr = calculate_cagr(clean_strat, periods_per_year)
    ann_vol = calculate_annualized_volatility(clean_strat, periods_per_year)
    sharpe = calculate_sharpe_ratio(clean_strat, rf_annual, periods_per_year)
    sortino = calculate_sortino_ratio(clean_strat, rf_annual, periods_per_year)
    mdd, _ = calculate_max_drawdown(clean_strat, is_returns=True)
    calmar = calculate_calmar_ratio(clean_strat, rf_annual, periods_per_year)
    hit_rate = calculate_hit_rate(clean_strat)

    if benchmark_returns is not None:
        alpha, beta = calculate_alpha_beta(clean_strat, benchmark_returns, rf_annual, periods_per_year)
    else:
        alpha, beta = 0.0, 1.0

    ann_turnover = calculate_turnover(turnover_series, periods_per_year) if turnover_series is not None else 0.0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": mdd,
        "Calmar Ratio": calmar,
        "Alpha (CAPM)": alpha,
        "Beta (CAPM)": beta,
        "Annualized Turnover": ann_turnover,
        "Hit Rate": hit_rate,
    }


def generate_performance_table(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    turnover_series: Optional[pd.Series] = None,
    rf_annual: float = 0.065,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Generate a structured performance table comparing Strategy vs Benchmark metrics.

    Args:
        strategy_returns: Daily returns of the strategy.
        benchmark_returns: Daily returns of the benchmark (^NSEI).
        turnover_series: Daily turnover series.
        rf_annual: Annualized risk-free rate.
        periods_per_year: Trading days per year.

    Returns:
        DataFrame comparing Strategy, Benchmark, and Delta across all quantitative metrics.
    """
    strat_metrics = calculate_all_metrics(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        turnover_series=turnover_series,
        rf_annual=rf_annual,
        periods_per_year=periods_per_year,
    )

    if benchmark_returns is not None:
        bench_metrics = calculate_all_metrics(
            strategy_returns=benchmark_returns,
            benchmark_returns=benchmark_returns,
            turnover_series=pd.Series(0.0, index=benchmark_returns.index),
            rf_annual=rf_annual,
            periods_per_year=periods_per_year,
        )
    else:
        bench_metrics = {}

    rows = []
    for k, strat_v in strat_metrics.items():
        bench_v = bench_metrics.get(k, np.nan)
        delta_v = strat_v - bench_v if not np.isnan(bench_v) else np.nan
        rows.append(
            {
                "Metric": k,
                "Strategy": strat_v,
                "Benchmark (^NSEI)": bench_v,
                "Delta": delta_v,
            }
        )
    return pd.DataFrame(rows).set_index("Metric")


