"""Data Loader module for historical NSE equity and benchmark price data.

Handles automated downloading via Yahoo Finance, disk caching (Parquet/CSV),
timestamped metadata tracking, survivorship bias disclosure, and robust data cleaning
preserving leading NaNs for late listings.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Primary liquid NSE universe (30 major Nifty 50 constituents across diverse sectors)
DEFAULT_NIFTY50_TICKERS: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
    "HINDUNILVR.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "M&M.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "TATASTEEL.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ASIANPAINT.NS",
    "ULTRACEMCO.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TATAMOTORS.NS",
    "COALINDIA.NS",
    "ONGC.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BAJAJFINSV.NS",
]

BENCHMARK_TICKER: str = "^NSEI"

SURVIVORSHIP_DISCLAIMER: str = (
    "SURVIVORSHIP BIAS DISCLOSURE:\n"
    "This dataset utilizes current constituents of the NSE Nifty index. Backtesting current\n"
    "index members historically introduces survivorship bias, as companies that were removed,\n"
    "delisted, or went bankrupt during the backtest window are not included in the sample.\n"
    "This structural bias tends to inflate historical performance metrics. For institutional\n"
    "production deployment, point-in-time constituent lists (e.g., from NSE India official\n"
    "historical archives or Bloomberg/Refinitiv) must be integrated."
)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Clean adjusted close prices DataFrame without introducing look-ahead or artificial volatility.

    Rules applied:
    1. Forward-fill isolated missing dates (trading holidays / missing ticks), preserving
       the last observed market transaction.
    2. Crucially, LEAVE LEADING NaNs INTACT until an asset's true inception / listing date.
       Do not backfill (bfill) as that creates artificial flat prices and false zero-volatility.
    3. Drop columns that contain exclusively NaN values.
    4. Sort index monotonically by datetime.

    Args:
        prices: Raw DataFrame of prices (dates as index, tickers as columns).

    Returns:
        Cleaned DataFrame of prices.
    """
    if prices.empty:
        return prices.copy()

    df = prices.copy()
    # Normalize index to timezone-naive datetime normalized to midnight
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df.sort_index()

    # Drop columns that are completely NaN
    df = df.dropna(axis=1, how="all")

    # Forward-fill isolated gaps; leading NaNs remain NaN naturally because ffill does not backfill
    df = df.ffill()

    return df


class DataLoader:
    """Production-grade data loader for NSE equities and benchmark indices."""

    def __init__(self, cache_dir: Union[str, Path] = "data") -> None:
        """Initialize DataLoader with local cache directory.

        Args:
            cache_dir: Path to directory for caching parquet/csv and metadata.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"

    @staticmethod
    def survivorship_disclaimer() -> str:
        """Return quantitative bias disclosure regarding survivorship."""
        return SURVIVORSHIP_DISCLAIMER

    def _get_cache_path(self, name: str, extension: str = "parquet") -> Path:
        return self.cache_dir / f"{name}.{extension}"

    def _save_metadata(self, meta: Dict[str, object]) -> None:
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write metadata cache: {e}")

    def _read_metadata(self) -> Dict[str, object]:
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read metadata cache: {e}")
        return {}

    def fetch_data(
        self,
        tickers: Optional[List[str]] = None,
        benchmark: str = BENCHMARK_TICKER,
        start_date: str = "2016-01-01",
        end_date: Optional[str] = None,
        force_reload: bool = False,
        cache_format: str = "parquet",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Fetch adjusted close prices for tickers and benchmark.

        Checks local disk cache first before performing network calls.

        Args:
            tickers: List of NSE ticker symbols (with '.NS' suffix). If None, defaults to DEFAULT_NIFTY50_TICKERS.
            benchmark: Benchmark ticker symbol (default '^NSEI').
            start_date: ISO start date string (e.g. '2016-01-01').
            end_date: ISO end date string (e.g. '2025-12-31'). If None, fetches up to latest.
            force_reload: If True, bypasses disk cache and downloads fresh data.
            cache_format: 'parquet' (preferred) or 'csv'.

        Returns:
            Tuple of (prices_df, benchmark_series), both cleaned and index-aligned.
        """
        if tickers is None:
            tickers = list(DEFAULT_NIFTY50_TICKERS)
        else:
            tickers = list(tickers)

        prices_cache_file = self._get_cache_path("nse_adjusted_close", cache_format)
        bench_cache_file = self._get_cache_path("benchmark_adjusted_close", cache_format)

        # Check cache validity
        if not force_reload and prices_cache_file.exists() and bench_cache_file.exists():
            try:
                if cache_format == "parquet":
                    prices = pd.read_parquet(prices_cache_file)
                    bench_df = pd.read_parquet(bench_cache_file)
                else:
                    prices = pd.read_csv(prices_cache_file, index_col=0, parse_dates=True)
                    bench_df = pd.read_csv(bench_cache_file, index_col=0, parse_dates=True)

                prices = clean_prices(prices)
                bench_df = clean_prices(bench_df)
                bench_series = bench_df.iloc[:, 0]

                # Filter tickers and date range if cached data covers requested subset
                avail_tickers = [t for t in tickers if t in prices.columns]
                if len(avail_tickers) >= len(tickers) * 0.8:
                    sub_prices = prices[avail_tickers]
                    sub_prices = sub_prices.loc[start_date:end_date]
                    sub_bench = bench_series.loc[start_date:end_date]
                    logger.info(f"Loaded {len(avail_tickers)} tickers from cache: {prices_cache_file}")
                    return sub_prices, sub_bench
            except Exception as e:
                logger.warning(f"Cache read error: {e}. Falling back to live download.")

        # Download via yfinance
        prices, bench_series = self._download_from_yfinance(
            tickers=tickers,
            benchmark=benchmark,
            start_date=start_date,
            end_date=end_date,
        )

        # Save to disk cache
        try:
            if cache_format == "parquet":
                prices.to_parquet(prices_cache_file)
                pd.DataFrame({benchmark: bench_series}).to_parquet(bench_cache_file)
            else:
                prices.to_csv(prices_cache_file)
                pd.DataFrame({benchmark: bench_series}).to_csv(bench_cache_file)

            metadata = {
                "download_timestamp": datetime.now(timezone.utc).isoformat(),
                "start_date": start_date,
                "end_date": end_date,
                "universe_count": len(prices.columns),
                "tickers": list(prices.columns),
                "benchmark": benchmark,
                "num_rows": len(prices),
                "survivorship_disclaimer": SURVIVORSHIP_DISCLAIMER,
            }
            self._save_metadata(metadata)
            logger.info(f"Cached price data to {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Could not write cache files: {e}")

        return prices, bench_series

    def _download_from_yfinance(
        self,
        tickers: List[str],
        benchmark: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Download raw adjusted close prices via yfinance."""
        import yfinance as yf

        all_symbols = list(set(tickers + [benchmark]))
        logger.info(f"Downloading {len(all_symbols)} symbols from Yahoo Finance...")

        # Batch download with auto_adjust=True guarantees adjustment for dividends, splits, and bonus issues
        raw = yf.download(
            tickers=all_symbols,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )

        if raw.empty:
            raise ValueError("No data returned from Yahoo Finance download.")

        # Handle multi-level columns if multiple metrics were returned
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.levels[0]:
                prices = raw["Close"]
            else:
                prices = raw.xs("Close", axis=1, level=0, drop_level=True)
        else:
            prices = raw

        prices = clean_prices(prices)

        if benchmark in prices.columns:
            bench_series = prices[benchmark].dropna()
            stock_prices = prices.drop(columns=[benchmark], errors="ignore")
        else:
            logger.warning(f"Benchmark {benchmark} missing in download; attempting single download.")
            bench_raw = yf.download(benchmark, start=start_date, end=end_date, auto_adjust=True, progress=False)
            bench_series = bench_raw["Close"].squeeze() if "Close" in bench_raw.columns else bench_raw.squeeze()
            bench_series = clean_prices(pd.DataFrame({benchmark: bench_series}))[benchmark]
            stock_prices = prices

        # Filter stock prices to requested tickers
        stock_cols = [c for c in tickers if c in stock_prices.columns]
        return stock_prices[stock_cols], bench_series

    @staticmethod
    def generate_synthetic_data(
        tickers: Optional[List[str]] = None,
        n_days: int = 2520,
        start_date: str = "2016-01-01",
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate mathematically rigorous synthetic price paths for offline testing.

        Simulates Geometric Brownian Motion with realistic drift, volatility, cross-correlation,
        and simulated delayed listings (late IPOs).

        Args:
            tickers: Optional list of tickers. Defaults to first 10 Nifty tickers.
            n_days: Number of trading days to simulate (~10 years).
            start_date: Starting date string.
            seed: Random seed for repeatability.

        Returns:
            Tuple of (prices_df, benchmark_series).
        """
        rng = np.random.default_rng(seed)
        if tickers is None:
            tickers = DEFAULT_NIFTY50_TICKERS[:12]

        dates = pd.bdate_range(start=start_date, periods=n_days)
        n_assets = len(tickers)

        # Generate market factor (benchmark)
        bench_drift = 0.12 / 252.0
        bench_vol = 0.15 / np.sqrt(252.0)
        market_shocks = rng.normal(bench_drift, bench_vol, size=n_days)
        bench_prices = 10000.0 * np.exp(np.cumsum(market_shocks))
        bench_series = pd.Series(bench_prices, index=dates, name=BENCHMARK_TICKER)

        # Generate asset returns correlated with market
        asset_prices_dict: Dict[str, np.ndarray] = {}
        for i, ticker in enumerate(tickers):
            beta = 0.7 + 0.6 * rng.random()
            idio_vol = (0.18 + 0.10 * rng.random()) / np.sqrt(252.0)
            idio_shocks = rng.normal(0.02 / 252.0, idio_vol, size=n_days)
            daily_returns = beta * market_shocks + idio_shocks
            initial_price = 500.0 + rng.random() * 2000.0
            p = initial_price * np.exp(np.cumsum(daily_returns))

            # Simulate delayed listing for the last 2 assets
            if i >= n_assets - 2:
                ipo_offset = int(n_days * 0.4)
                p[:ipo_offset] = np.nan

            asset_prices_dict[ticker] = p

        prices_df = pd.DataFrame(asset_prices_dict, index=dates)
        return clean_prices(prices_df), bench_series

