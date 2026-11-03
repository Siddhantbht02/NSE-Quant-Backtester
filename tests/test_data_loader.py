"""Tests for DataLoader module."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader, clean_prices, DEFAULT_NIFTY50_TICKERS, BENCHMARK_TICKER


def test_clean_prices_preserves_leading_nans():
    """Verify that leading NaNs (prior to IPO/listing) are preserved, not backfilled."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    # Stock A has data from day 0; Stock B lists on day 4 (leading NaNs for days 0-3)
    data = {
        "STOCK_A.NS": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "STOCK_B.NS": [np.nan, np.nan, np.nan, np.nan, 50, 51, np.nan, 53, 54, 55],
    }
    raw_df = pd.DataFrame(data, index=dates)
    cleaned = clean_prices(raw_df)

    # First 4 days of STOCK_B must strictly remain NaN
    assert cleaned["STOCK_B.NS"].iloc[:4].isna().all(), "Leading NaNs must NOT be filled!"

    # Day 4 onwards must be valid
    assert cleaned["STOCK_B.NS"].iloc[4] == 50.0

    # Day 6 was an isolated holiday/gap (NaN between 51 and 53), must be forward-filled with 51
    assert cleaned["STOCK_B.NS"].iloc[6] == 51.0
    assert cleaned["STOCK_B.NS"].iloc[7] == 53.0


def test_clean_prices_drops_entirely_nan_columns():
    """Verify that assets with completely missing data are dropped."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    data = {
        "STOCK_A.NS": [10, 11, 12, 13, 14],
        "STOCK_EMPTY.NS": [np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    cleaned = clean_prices(pd.DataFrame(data, index=dates))
    assert "STOCK_A.NS" in cleaned.columns
    assert "STOCK_EMPTY.NS" not in cleaned.columns


def test_data_loader_cache_roundtrip(tmp_path):
    """Verify DataLoader saves to disk cache and loads without network call."""
    loader = DataLoader(cache_dir=tmp_path)
    dates = pd.bdate_range("2021-01-01", periods=50)

    # Generate synthetic prices and save to cache
    synth_prices, synth_bench = DataLoader.generate_synthetic_data(
        tickers=["STOCK_X.NS", "STOCK_Y.NS"], n_days=50, start_date="2021-01-01"
    )

    cache_file = tmp_path / "nse_adjusted_close.parquet"
    bench_file = tmp_path / "benchmark_adjusted_close.parquet"
    synth_prices.to_parquet(cache_file)
    pd.DataFrame({"^NSEI": synth_bench}).to_parquet(bench_file)

    # Calling fetch_data should load from cache
    loaded_prices, loaded_bench = loader.fetch_data(
        tickers=["STOCK_X.NS", "STOCK_Y.NS"],
        start_date="2021-01-01",
        end_date="2021-03-01",
        force_reload=False,
    )

    assert not loaded_prices.empty
    assert "STOCK_X.NS" in loaded_prices.columns
    assert len(loaded_prices) <= 50
    assert not loaded_bench.empty


def test_survivorship_disclaimer():
    """Verify survivorship disclaimer is well-formed and explicit."""
    disclaimer = DataLoader.survivorship_disclaimer()
    assert "SURVIVORSHIP BIAS DISCLOSURE" in disclaimer
    assert "point-in-time" in disclaimer.lower()


def test_default_universe_definition():
    """Verify default universe contains at least 25 primary liquid NSE tickers."""
    assert len(DEFAULT_NIFTY50_TICKERS) >= 25
    for ticker in DEFAULT_NIFTY50_TICKERS:
        assert ticker.endswith(".NS")
    assert BENCHMARK_TICKER == "^NSEI"

