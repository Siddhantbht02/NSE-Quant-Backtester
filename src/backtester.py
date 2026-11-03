"""Vectorised Simulation & Execution Engine module for nse-backtester.

Implements vectorised execution with:
- Forced 1-day execution lag (weights.shift(1)) to prevent look-ahead bias
- Friction modeling calibrated to Indian markets (STT, stamp duty, exchange fees, GST, slippage)
- Turnovers, transaction costs, and comprehensive metric outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from src.metrics import calculate_all_metrics, calculate_max_drawdown
from src.strategy import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Encapsulates comprehensive output of a backtest simulation run."""

    strategy_name: str
    portfolio_history: pd.DataFrame
    target_weights: pd.DataFrame
    executed_weights: pd.DataFrame
    metrics: Dict[str, float]
    benchmark_metrics: Dict[str, float]

    def summary(self) -> pd.DataFrame:
        """Return a formatted DataFrame comparing Strategy vs Benchmark metrics."""
        data = []
        for key in self.metrics:
            strat_val = self.metrics[key]
            bench_val = self.benchmark_metrics.get(key, np.nan)
            diff_val = strat_val - bench_val if not np.isnan(bench_val) else np.nan
            data.append(
                {
                    "Metric": key,
                    "Strategy": strat_val,
                    "Benchmark (^NSEI)": bench_val,
                    "Delta (Strategy - Benchmark)": diff_val,
                }
            )
        return pd.DataFrame(data).set_index("Metric")

    def __repr__(self) -> str:
        cagr = self.metrics.get("CAGR", 0.0) * 100.0
        sharpe = self.metrics.get("Sharpe Ratio", 0.0)
        mdd = self.metrics.get("Max Drawdown", 0.0) * 100.0
        return (
            f"<BacktestResult strategy='{self.strategy_name}' "
            f"CAGR={cagr:.2f}% Sharpe={sharpe:.2f} MDD={mdd:.2f}%>"
        )


class Backtester:
    """Vectorised backtest simulation engine with realistic execution lag and friction."""

    def __init__(
        self,
        cost_per_side: float = 0.0015,
        rf_annual: float = 0.065,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        """Initialize Backtester.

        Args:
            cost_per_side: One-way transaction friction fraction (default 0.0015 = 15 bps / 0.15%).
                Calibrated to Indian equities:
                - STT (Securities Transaction Tax: 0.10% on delivery)
                - Stamp Duty (0.015%)
                - Exchange Turnover + SEBI turnover fee (~0.0035%)
                - GST (18% on charges)
                - Bid-ask slippage (~0.03%)
                Total round-trip friction ~30 bps (15 bps per side).
            rf_annual: Annualized Indian Risk-Free Rate (default 0.065 = 6.5%).
            initial_capital: Starting capital in INR (default 1,000,000 INR).
        """
        if cost_per_side < 0:
            raise ValueError(f"cost_per_side ({cost_per_side}) cannot be negative.")
        if rf_annual < 0:
            raise ValueError(f"rf_annual ({rf_annual}) cannot be negative.")

        self.cost_per_side = cost_per_side
        self.rf_annual = rf_annual
        self.initial_capital = initial_capital

    def run(
        self,
        prices: pd.DataFrame,
        strategy_or_weights: Union[Strategy, pd.DataFrame],
        benchmark: Optional[pd.Series] = None,
        strategy_name: Optional[str] = None,
    ) -> BacktestResult:
        """Run vectorised backtest on historical adjusted close prices.

        Execution Rules:
        - Strict 1-period shift: Executed weights on day t = Target weights computed at close of day t-1.
        - Turnover on day t = sum_i |w_exec_{i, t} - w_exec_{i, t-1}|.
        - Cost on day t = Turnover_t * cost_per_side.
        - Net daily return = Gross daily return - Cost_t.

        Args:
            prices: Adjusted close prices DataFrame (dates x tickers).
            strategy_or_weights: Strategy instance or pre-computed target weights DataFrame.
            benchmark: Optional benchmark prices series (^NSEI).
            strategy_name: Optional name for reporting.

        Returns:
            BacktestResult object containing simulation timeline, weights, and metrics.
        """
        if prices.empty:
            raise ValueError("Prices DataFrame cannot be empty.")

        # Determine strategy name and target weights
        if isinstance(strategy_or_weights, Strategy):
            name = strategy_name or strategy_or_weights.__class__.__name__
            target_weights = strategy_or_weights.generate_signals(prices)
        else:
            name = strategy_name or "CustomStrategy"
            target_weights = strategy_or_weights.copy()

        # Align target weights columns and index with prices
        target_weights = target_weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

        # CRITICAL LOOK-AHEAD PREVENTION:
        # A signal generated at close of day t can only be traded and earn returns starting day t+1.
        # Therefore, executed weights are shifted by 1 trading day.
        executed_weights = target_weights.shift(1).fillna(0.0)

        # Asset returns: R_{i, t} = (P_{i, t} / P_{i, t-1}) - 1.0
        asset_returns = prices.pct_change(fill_method=None).fillna(0.0)

        # Gross daily return: sum_i (w_exec_{i, t} * R_{i, t})
        # NumPy dot product / row sum for speed
        w_exec_arr = executed_weights.to_numpy(dtype=np.float64)
        r_asset_arr = asset_returns.to_numpy(dtype=np.float64)
        gross_return_arr = np.sum(w_exec_arr * r_asset_arr, axis=1)

        # Turnover calculation:
        # Turnover_t = sum_i |w_exec_{i, t} - w_exec_{i, t-1}|
        # For day 0, previous weights are assumed 0.0
        prev_w_exec = np.zeros_like(w_exec_arr)
        prev_w_exec[1:] = w_exec_arr[:-1]
        turnover_arr = np.sum(np.abs(w_exec_arr - prev_w_exec), axis=1)

        # Daily transaction cost: Turnover_t * cost_per_side
        cost_arr = turnover_arr * self.cost_per_side

        # Net daily return: R_net,t = R_gross,t - Cost_t
        net_return_arr = gross_return_arr - cost_arr

        # Cumulative equity curves
        equity_progression = np.cumprod(1.0 + net_return_arr)
        strategy_equity = self.initial_capital * equity_progression

        # Strategy Drawdown
        _, strategy_drawdown = calculate_max_drawdown(pd.Series(equity_progression, index=prices.index))

        # Benchmark calculation
        if benchmark is not None:
            aligned_bench = benchmark.reindex(prices.index).ffill()
            bench_returns = aligned_bench.pct_change(fill_method=None).fillna(0.0)
            bench_equity = self.initial_capital * np.cumprod(1.0 + bench_returns.to_numpy())
            _, bench_drawdown = calculate_max_drawdown(pd.Series(bench_equity, index=prices.index))
        else:
            bench_returns = pd.Series(0.0, index=prices.index)
            bench_equity = pd.Series(self.initial_capital, index=prices.index)
            bench_drawdown = pd.Series(0.0, index=prices.index)

        # Construct portfolio history DataFrame
        portfolio_history = pd.DataFrame(
            {
                "gross_return": gross_return_arr,
                "turnover": turnover_arr,
                "transaction_cost": cost_arr,
                "net_return": net_return_arr,
                "equity_curve": strategy_equity,
                "drawdown": strategy_drawdown,
                "benchmark_return": bench_returns,
                "benchmark_equity_curve": bench_equity,
                "benchmark_drawdown": bench_drawdown,
            },
            index=prices.index,
        )

        # Calculate metrics
        strat_net_series = portfolio_history["net_return"]
        bench_ret_series = portfolio_history["benchmark_return"]
        turnover_series = portfolio_history["turnover"]

        metrics = calculate_all_metrics(
            strategy_returns=strat_net_series,
            benchmark_returns=bench_ret_series if benchmark is not None else None,
            turnover_series=turnover_series,
            rf_annual=self.rf_annual,
        )

        if benchmark is not None:
            bench_metrics = calculate_all_metrics(
                strategy_returns=bench_ret_series,
                benchmark_returns=bench_ret_series,
                turnover_series=pd.Series(0.0, index=prices.index),
                rf_annual=self.rf_annual,
            )
        else:
            bench_metrics = {}

        return BacktestResult(
            strategy_name=name,
            portfolio_history=portfolio_history,
            target_weights=target_weights,
            executed_weights=executed_weights,
            metrics=metrics,
            benchmark_metrics=bench_metrics,
        )

