"""Strategy Framework module for nse-backtester.

Defines the abstract Strategy interface and concrete implementations:
- MomentumStrategy (12-1 Month Cross-Sectional Momentum)
- MeanReversionStrategy (Rolling 20-day SMA Z-Score Mean Reversion)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Abstract Base Class for all quantitative portfolio strategies."""

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute target portfolio weights given an adjusted close price DataFrame.

        Args:
            prices: Adjusted close prices DataFrame (index=dates, columns=tickers).

        Returns:
            Target weights DataFrame (index=dates, columns=tickers) with row sums
            satisfying sum(|w_i|) <= 1.0.
        """
        pass


class MomentumStrategy(Strategy):
    """12-1 Month Cross-Sectional Momentum Strategy.

    Academic Logic:
    Ranks the universe cross-sectionally based on past performance over a 12-month
    (252 trading days) lookback, skipping the most recent 1-month (21 trading days)
    to eliminate short-term microstructure reversal effects (Jegadeesh & Titman, 1993).

    Selection & Rebalancing:
    Rebalances monthly (or every N days). Selects the top quintile (top 20%) of eligible
    stocks and assigns equal weights. Holds target weights constant between rebalance dates.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        top_quantile: float = 0.20,
        rebalance_freq: Literal["monthly", "21_days"] = "monthly",
    ) -> None:
        """Initialize Momentum Strategy.

        Args:
            lookback_days: Total lookback window in trading days (default 252).
            skip_days: Microstructure reversal skip window in trading days (default 21).
            top_quantile: Top fraction of universe to select (default 0.20 for top quintile).
            rebalance_freq: 'monthly' (month-end) or '21_days' (every 21 trading days).
        """
        if lookback_days <= skip_days:
            raise ValueError(f"lookback_days ({lookback_days}) must be greater than skip_days ({skip_days})")
        if not (0.0 < top_quantile <= 1.0):
            raise ValueError(f"top_quantile ({top_quantile}) must be in range (0.0, 1.0]")

        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.top_quantile = top_quantile
        self.rebalance_freq = rebalance_freq

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate target portfolio weights based on 12-1 month cross-sectional momentum.

        Args:
            prices: Adjusted close prices DataFrame (index=dates, columns=tickers).

        Returns:
            Target weights DataFrame (index=dates, columns=tickers).
        """
        if prices.empty or len(prices) <= self.lookback_days:
            # Not enough data for lookback; return zero weights
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # 12-1 Momentum score: P_{t - skip} / P_{t - lookback} - 1.0
        # Both shifts strictly look backward from current day t
        p_t_minus_skip = prices.shift(self.skip_days)
        p_t_minus_lookback = prices.shift(self.lookback_days)
        momentum_scores = (p_t_minus_skip / p_t_minus_lookback) - 1.0

        # Identify rebalance dates
        dates = prices.index
        n_dates = len(dates)
        rebalance_flags = np.zeros(n_dates, dtype=bool)

        if self.rebalance_freq == "monthly":
            # Calendar month-end trading days
            periods = dates.to_period("M")
            is_month_end = np.array([periods[i] != periods[i + 1] for i in range(n_dates - 1)] + [True])
            rebalance_flags = is_month_end
        else:
            # Every 21 trading days starting from lookback_days
            for i in range(self.lookback_days, n_dates, 21):
                rebalance_flags[i] = True

        weights = pd.DataFrame(np.nan, index=dates, columns=prices.columns, dtype=float)

        # Before warmup period, portfolio is 100% cash (weights = 0)
        weights.iloc[: self.lookback_days] = 0.0

        for i in range(self.lookback_days, n_dates):
            if rebalance_flags[i]:
                row_scores = momentum_scores.iloc[i]
                current_prices = prices.iloc[i]

                # Asset must have a valid current price AND a valid momentum score
                valid_mask = (~row_scores.isna()) & (~current_prices.isna()) & np.isfinite(row_scores)
                valid_tickers = row_scores[valid_mask]

                row_weights = pd.Series(0.0, index=prices.columns, dtype=float)
                if len(valid_tickers) > 0:
                    # Select top quintile
                    k = max(1, math.ceil(len(valid_tickers) * self.top_quantile))
                    top_assets = valid_tickers.nlargest(k).index
                    weight_per_asset = 1.0 / len(top_assets)
                    row_weights[top_assets] = weight_per_asset

                weights.iloc[i] = row_weights

        # Forward-fill target weights between rebalance dates
        weights = weights.ffill().fillna(0.0)

        # Verify row sums <= 1.0
        row_sums = weights.abs().sum(axis=1)
        over_mask = row_sums > 1.0000001
        if over_mask.any():
            weights[over_mask] = weights[over_mask].div(row_sums[over_mask], axis=0)

        return weights


class MeanReversionStrategy(Strategy):
    """Time-series Z-Score Mean Reversion Strategy.

    Logic:
    Computes rolling 20-day Simple Moving Average (SMA) and rolling standard deviation
    for each asset.
    Z-Score = (Price_t - SMA_20_t) / Rolling_Std_20_t.

    Rules:
    - Long Entry: Enter when Z < -1.5 (oversold condition).
    - Long Exit: Exit when Z >= 0.0 (price has reverted to rolling mean).
    - Allocation: Normalized equal weight or inverse-volatility weight among all
      currently active positions.
    """

    def __init__(
        self,
        window: int = 20,
        entry_z: float = -1.5,
        exit_z: float = 0.0,
        weighting: Literal["equal", "inverse_vol"] = "equal",
        vol_window: int = 20,
    ) -> None:
        """Initialize Mean Reversion Strategy.

        Args:
            window: Rolling window for SMA and standard deviation (default 20).
            entry_z: Z-score threshold to enter long position (default -1.5).
            exit_z: Z-score threshold to exit long position (default 0.0).
            weighting: 'equal' or 'inverse_vol' (risk-parity across active signals).
            vol_window: Rolling window for volatility estimation if inverse_vol is selected.
        """
        if entry_z >= exit_z:
            raise ValueError(f"entry_z ({entry_z}) must be strictly less than exit_z ({exit_z})")
        if window < 2:
            raise ValueError(f"window ({window}) must be at least 2")

        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.weighting = weighting
        self.vol_window = vol_window

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate target weights based on rolling Z-score mean reversion.

        Args:
            prices: Adjusted close prices DataFrame (index=dates, columns=tickers).

        Returns:
            Target weights DataFrame (index=dates, columns=tickers).
        """
        if prices.empty or len(prices) < self.window:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Vectorised rolling statistics
        rolling_mean = prices.rolling(window=self.window, min_periods=self.window).mean()
        rolling_std = prices.rolling(window=self.window, min_periods=self.window).std(ddof=1)
        z_scores = (prices - rolling_mean) / (rolling_std.replace(0.0, np.nan) + 1e-8)

        # Precalculate daily return volatility for inverse-vol weighting
        if self.weighting == "inverse_vol":
            daily_returns = prices.pct_change(fill_method=None)
            rolling_vol = daily_returns.rolling(window=self.vol_window, min_periods=5).std(ddof=1)
        else:
            rolling_vol = None

        n_rows, n_cols = prices.shape
        z_arr = z_scores.to_numpy()
        price_arr = prices.to_numpy()

        # State matrix: 1 = active long position, 0 = flat
        state_matrix = np.zeros((n_rows, n_cols), dtype=np.int8)

        # Fast sequential state-machine per ticker
        for j in range(n_cols):
            curr_state = 0
            for i in range(self.window - 1, n_rows):
                z = z_arr[i, j]
                p = price_arr[i, j]

                if np.isnan(p) or np.isnan(z):
                    curr_state = 0
                elif curr_state == 0:
                    if z < self.entry_z:
                        curr_state = 1
                elif curr_state == 1:
                    if z >= self.exit_z:
                        curr_state = 0

                state_matrix[i, j] = curr_state

        # Compute portfolio target weights
        weights_arr = np.zeros((n_rows, n_cols), dtype=np.float64)

        if self.weighting == "equal":
            for i in range(self.window - 1, n_rows):
                active_idx = np.where(state_matrix[i] == 1)[0]
                n_active = len(active_idx)
                if n_active > 0:
                    weights_arr[i, active_idx] = 1.0 / n_active
        else:
            # Inverse-volatility weighting
            vol_arr = rolling_vol.to_numpy() if rolling_vol is not None else None
            for i in range(self.window - 1, n_rows):
                active_idx = np.where(state_matrix[i] == 1)[0]
                if len(active_idx) > 0 and vol_arr is not None:
                    vols = vol_arr[i, active_idx]
                    # Filter out NaN or non-positive vols
                    valid_sub = ~np.isnan(vols) & (vols > 1e-6)
                    if np.any(valid_sub):
                        inv_vols = np.zeros_like(vols)
                        inv_vols[valid_sub] = 1.0 / vols[valid_sub]
                        inv_sum = np.sum(inv_vols)
                        if inv_sum > 0:
                            weights_arr[i, active_idx] = inv_vols / inv_sum
                        else:
                            weights_arr[i, active_idx] = 1.0 / len(active_idx)
                    else:
                        weights_arr[i, active_idx] = 1.0 / len(active_idx)

        weights = pd.DataFrame(weights_arr, index=prices.index, columns=prices.columns)

        # Enforce sum(|w_i|) <= 1.0
        row_sums = weights.abs().sum(axis=1)
        over_mask = row_sums > 1.0000001
        if over_mask.any():
            weights[over_mask] = weights[over_mask].div(row_sums[over_mask], axis=0)

        return weights

