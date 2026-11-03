"""Tests for Portfolio Management and Risk Engine."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import PortfolioManager, apply_portfolio_constraints


def test_portfolio_constraints_capping_and_redistribution():
    """Verify single-stock max_weight cap (15%) and cash buffering (95% equity)."""
    dates = pd.bdate_range("2021-01-01", periods=5)
    # 3 active stocks with equal unconstrained weights = 0.333 each
    raw_weights = pd.DataFrame(
        {
            "A.NS": [0.50, 0.40, 0.33, 0.0, 0.80],
            "B.NS": [0.50, 0.40, 0.33, 0.5, 0.10],
            "C.NS": [0.00, 0.20, 0.33, 0.5, 0.10],
        },
        index=dates,
    )

    constrained = apply_portfolio_constraints(
        weights=raw_weights,
        max_weight=0.15,
        cash_buffer=0.05,
        max_gross_exposure=0.95,
    )

    # Verification 1: No individual asset weight exceeds max_weight (0.15)
    assert (constrained <= 0.150001).all().all()

    # Verification 2: Total gross exposure <= max_gross_exposure (0.95)
    row_sums = constrained.sum(axis=1)
    assert (row_sums <= 0.950001).all()

    # Verification 3: Non-negativity preserved
    assert (constrained >= 0.0).all().all()


def test_portfolio_manager_class():
    """Verify PortfolioManager wrapper and exposure diagnostics."""
    pm = PortfolioManager(max_weight=0.15, cash_buffer=0.05)
    dates = pd.bdate_range("2021-01-01", periods=10)
    raw = pd.DataFrame(np.random.uniform(0.1, 0.5, size=(10, 10)), index=dates)
    raw = raw.div(raw.sum(axis=1), axis=0)

    processed = pm.process_weights(raw)
    assert (processed.max(axis=1) <= 0.150001).all()

    summary = pm.get_exposure_summary(processed)
    assert "gross_exposure" in summary.columns
    assert "cash_weight" in summary.columns
    assert (summary["gross_exposure"] <= 0.950001).all()
    assert (summary["cash_weight"] >= 0.049999).all()

