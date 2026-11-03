"""Portfolio & Risk Engine module for nse-backtester.

Handles position sizing, maximum asset weight capping, cash buffering,
turnover damping, and portfolio constraint normalization.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_portfolio_constraints(
    weights: pd.DataFrame,
    max_weight: float = 0.15,
    cash_buffer: float = 0.05,
    max_gross_exposure: Optional[float] = None,
    min_weight_threshold: float = 0.001,
) -> pd.DataFrame:
    """Apply institutional risk and portfolio constraints to target weights.

    Enforces:
    1. Cash buffering: Maximum equity exposure capped at `max_gross_exposure`
       (defaults to 1.0 - cash_buffer, e.g. 95% equity, 5% cash).
    2. Single-stock allocation cap: No asset exceeds `max_weight` (e.g. 15%).
       Excess weight from capped assets is iteratively redistributed to
       uncapped active positions.
    3. Non-negativity / sign preservation and threshold filtering.
    4. Exact row normalization satisfying sum(|w_i|) <= max_gross_exposure <= 1.0.

    Args:
        weights: DataFrame of unconstrained target weights (dates x tickers).
        max_weight: Maximum allowed weight for any individual asset (default 0.15 / 15%).
        cash_buffer: Minimum mandatory cash allocation (default 0.05 / 5%).
        max_gross_exposure: Maximum total portfolio equity exposure. If None,
            equals (1.0 - cash_buffer).
        min_weight_threshold: Tiny positions below this threshold are zeroed out (default 0.001 / 10 bps).

    Returns:
        Constrained and normalized weights DataFrame.
    """
    if weights.empty:
        return weights.copy()

    if max_gross_exposure is None:
        max_gross_exposure = max(0.0, 1.0 - cash_buffer)

    target_max_gross = min(1.0, max_gross_exposure)
    w = weights.copy()

    # Zero out negligible positions below minimum threshold
    w[w.abs() < min_weight_threshold] = 0.0

    w_arr = w.to_numpy(dtype=np.float64, copy=True)
    n_rows, n_cols = w_arr.shape

    for i in range(n_rows):
        row = w_arr[i]
        total_w = np.sum(np.abs(row))
        if total_w <= 1e-8:
            continue

        # Scale down if row exceeds maximum gross exposure
        if total_w > target_max_gross:
            row = row * (target_max_gross / total_w)

        # Iterative weight capping and redistribution
        # Up to 10 iterations to redistribute excess weight to uncapped active assets
        active_mask = row > 0
        if not np.any(active_mask):
            w_arr[i] = row
            continue

        permanently_capped = np.zeros(n_cols, dtype=bool)

        for _ in range(10):
            over_cap = (row > max_weight) & (~permanently_capped)
            if not np.any(over_cap):
                break

            permanently_capped |= over_cap
            excess = float(np.sum(row[over_cap] - max_weight))
            row[over_cap] = max_weight

            # Only assets that have not hit the cap can receive redistributed weight
            eligible = active_mask & (~permanently_capped)
            eligible_sum = float(np.sum(row[eligible]))

            if eligible_sum > 1e-8 and excess > 0:
                # Proportional redistribution of excess weight
                row[eligible] += excess * (row[eligible] / eligible_sum)
            else:
                # Excess cannot be redistributed without violating caps; remains in cash
                break

        # Clip any minor floating point overshoot
        row = np.minimum(row, max_weight)

        # Final safety check: ensure total weight <= target_max_gross
        curr_sum = np.sum(np.abs(row))
        if curr_sum > target_max_gross + 1e-8:
            row = row * (target_max_gross / curr_sum)

        w_arr[i] = row

    result = pd.DataFrame(w_arr, index=weights.index, columns=weights.columns)
    return result


class PortfolioManager:
    """Manages strategy weight transformations, capital allocation, and risk limits."""

    def __init__(
        self,
        max_weight: float = 0.15,
        cash_buffer: float = 0.05,
        max_gross_exposure: float = 0.95,
        min_weight_threshold: float = 0.005,
    ) -> None:
        """Initialize PortfolioManager.

        Args:
            max_weight: Maximum weight per single asset (default 0.15).
            cash_buffer: Cash buffer reservation (default 0.05).
            max_gross_exposure: Maximum total equity exposure (default 0.95).
            min_weight_threshold: Minimum position threshold to prevent micro-turnover (default 0.005).
        """
        self.max_weight = max_weight
        self.cash_buffer = cash_buffer
        self.max_gross_exposure = max_gross_exposure
        self.min_weight_threshold = min_weight_threshold

    def process_weights(self, raw_weights: pd.DataFrame) -> pd.DataFrame:
        """Apply all risk and sizing constraints to generate production target weights.

        Args:
            raw_weights: Raw signal weights from strategy.

        Returns:
            Constrained portfolio weights.
        """
        return apply_portfolio_constraints(
            weights=raw_weights,
            max_weight=self.max_weight,
            cash_buffer=self.cash_buffer,
            max_gross_exposure=self.max_gross_exposure,
            min_weight_threshold=self.min_weight_threshold,
        )

    @staticmethod
    def get_exposure_summary(weights: pd.DataFrame) -> pd.DataFrame:
        """Compute portfolio exposure diagnostics over time.

        Args:
            weights: Executed or target portfolio weights.

        Returns:
            DataFrame containing Gross Exposure, Cash Weight, Active Asset Count, and Max Weight.
        """
        gross_exposure = weights.abs().sum(axis=1)
        cash_weight = 1.0 - gross_exposure
        active_assets = (weights.abs() > 1e-4).sum(axis=1)
        max_asset_weight = weights.max(axis=1)

        summary = pd.DataFrame(
            {
                "gross_exposure": gross_exposure,
                "cash_weight": cash_weight,
                "active_assets": active_assets,
                "max_asset_weight": max_asset_weight,
            },
            index=weights.index,
        )
        return summary
