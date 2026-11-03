"""Pytest configuration and shared fixtures for nse-backtester."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Ensure mimetypes patch on Windows
import mimetypes
_orig_read = getattr(mimetypes.MimeTypes, "read_windows_registry", None)
if _orig_read is not None:
    def _safe_read(self, *args, **kwargs):
        try:
            return _orig_read(self, *args, **kwargs)
        except (PermissionError, OSError):
            pass
    mimetypes.MimeTypes.read_windows_registry = _safe_read


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Generate 300 days of synthetic prices for 5 assets."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(123)
    tickers = ["STOCK_A.NS", "STOCK_B.NS", "STOCK_C.NS", "STOCK_D.NS", "STOCK_E.NS"]
    data = {}
    for t in tickers:
        ret = rng.normal(0.0005, 0.015, size=300)
        p = 100.0 * np.exp(np.cumsum(ret))
        data[t] = p
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def benchmark_series() -> pd.Series:
    """Generate 300 days of synthetic benchmark prices."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(999)
    ret = rng.normal(0.0004, 0.012, size=300)
    p = 10000.0 * np.exp(np.cumsum(ret))
    return pd.Series(p, index=dates, name="^NSEI")

