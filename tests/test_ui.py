"""Unit and smoke tests for UI components, Plotly chart generators, and configuration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ui.plots import (
    create_allocation_area_chart,
    create_drawdown_plot,
    create_equity_curve,
    create_rolling_sharpe_chart,
)
from ui.sidebar import SidebarConfig


@pytest.fixture
def mock_results_df() -> pd.DataFrame:
    """Generate mock backtest results DataFrame for chart testing."""
    dates = pd.bdate_range("2021-01-01", periods=150)
    rng = np.random.default_rng(42)

    net_ret = rng.normal(0.0008, 0.012, size=150)
    bench_ret = rng.normal(0.0005, 0.010, size=150)

    strat_eq = 1_000_000.0 * np.cumprod(1.0 + net_ret)
    bench_eq = 1_000_000.0 * np.cumprod(1.0 + bench_ret)

    strat_peak = np.maximum.accumulate(strat_eq)
    strat_dd = (strat_eq - strat_peak) / strat_peak

    bench_peak = np.maximum.accumulate(bench_eq)
    bench_dd = (bench_eq - bench_peak) / bench_peak

    return pd.DataFrame(
        {
            "net_return": net_ret,
            "benchmark_return": bench_ret,
            "equity_curve": strat_eq,
            "benchmark_equity_curve": bench_eq,
            "drawdown": strat_dd,
            "benchmark_drawdown": bench_dd,
        },
        index=dates,
    )


@pytest.fixture
def mock_weights_df() -> pd.DataFrame:
    """Generate mock portfolio weights DataFrame."""
    dates = pd.bdate_range("2021-01-01", periods=150)
    data = {
        "RELIANCE.NS": np.full(150, 0.15),
        "TCS.NS": np.full(150, 0.15),
        "INFY.NS": np.full(150, 0.10),
        "ZERO_STOCK.NS": np.zeros(150),
    }
    return pd.DataFrame(data, index=dates)


def test_create_equity_curve(mock_results_df: pd.DataFrame):
    """Verify equity curve figure creation and trace configurations."""
    fig = create_equity_curve(mock_results_df, use_log_scale=False, normalize=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Strategy + Benchmark
    assert fig.data[0].name == "Strategy (Net of Friction)"
    assert fig.data[1].name == "Benchmark (Nifty 50)"

    # Test log scale option
    fig_log = create_equity_curve(mock_results_df, use_log_scale=True, normalize=True)
    assert fig_log.layout.yaxis.type == "log"


def test_create_drawdown_plot(mock_results_df: pd.DataFrame):
    """Verify underwater drawdown figure creation."""
    fig = create_drawdown_plot(mock_results_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    # Strategy trace must have tozeroy fill
    assert fig.data[0].fill == "tozeroy"
    assert "Strategy Drawdown" in fig.data[0].name


def test_create_allocation_area_chart(mock_weights_df: pd.DataFrame):
    """Verify stacked allocation chart creation and filtering of inactive stocks."""
    fig = create_allocation_area_chart(mock_weights_df)
    assert isinstance(fig, go.Figure)
    # ZERO_STOCK.NS should be filtered out because total weight is 0
    trace_names = [t.name for t in fig.data]
    assert "ZERO_STOCK" not in trace_names
    assert "RELIANCE" in trace_names
    assert fig.data[0].stackgroup == "one"


def test_create_rolling_sharpe_chart(mock_results_df: pd.DataFrame):
    """Verify rolling Sharpe ratio chart generation."""
    fig = create_rolling_sharpe_chart(mock_results_df, window=63, rf_annual=0.065)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    # Should contain horizontal zero reference line
    assert any("Zero Excess Return" in str(ann) for ann in fig.layout.annotations)


def test_sidebar_config_initialization():
    """Verify SidebarConfig dataclass initialization and types."""
    cfg = SidebarConfig(
        start_date="2016-01-01",
        end_date="2025-12-31",
        strategy_type="momentum",
        lookback_days=252,
        skip_days=21,
        top_quantile=0.20,
        rebalance_freq="monthly",
        mr_window=20,
        entry_z=-1.5,
        exit_z=0.0,
        mr_weighting="inverse_vol",
        cost_bps=15.0,
        rf_annual=0.065,
        max_weight=0.15,
        cash_buffer=0.05,
        force_reload=False,
    )
    assert cfg.strategy_type == "momentum"
    assert cfg.lookback_days == 252
    assert cfg.cost_bps == 15.0


def test_empty_dataframe_plots():
    """Verify plot generators handle empty DataFrames gracefully without crashing."""
    empty_df = pd.DataFrame()
    fig1 = create_equity_curve(empty_df)
    fig2 = create_drawdown_plot(empty_df)
    fig3 = create_allocation_area_chart(empty_df)
    fig4 = create_rolling_sharpe_chart(empty_df)

    assert isinstance(fig1, go.Figure)
    assert isinstance(fig2, go.Figure)
    assert isinstance(fig3, go.Figure)
    assert isinstance(fig4, go.Figure)

