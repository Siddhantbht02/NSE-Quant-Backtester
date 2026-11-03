"""Plotly figure generators for the nse-backtester research dashboard.

Constructs institutional-grade, high-contrast, dark-mode-first interactive visualizations:
- Equity Curve Progression (Linear / Log Scale)
- Underwater Drawdown (% drop from historical peak)
- Dynamic Portfolio Allocation (Stacked Area Chart)
- Rolling 6-Month (126-Day) Annualized Sharpe Ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _apply_institutional_theme(
    fig: go.Figure,
    title: str,
    yaxis_title: str,
    xaxis_title: str = "Date",
    hovermode: str = "x unified",
) -> go.Figure:
    """Apply consistent high-contrast, dark-mode-first institutional layout styling."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Arial, sans-serif", "color": "#F1F5F9"},
        title={
            "text": f"<b>{title}</b>",
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 15, "color": "#F8FAFC"},
        },
        xaxis={
            "title": {"text": xaxis_title, "font": {"color": "#CBD5E1", "size": 12}},
            "tickfont": {"color": "#94A3B8", "size": 11},
            "showgrid": True,
            "gridcolor": "rgba(255, 255, 255, 0.08)",
            "zeroline": False,
            "showline": True,
            "linecolor": "rgba(255, 255, 255, 0.25)",
        },
        yaxis={
            "title": {"text": yaxis_title, "font": {"color": "#CBD5E1", "size": 12}},
            "tickfont": {"color": "#94A3B8", "size": 11},
            "showgrid": True,
            "gridcolor": "rgba(255, 255, 255, 0.08)",
            "zeroline": True,
            "zerolinecolor": "rgba(255, 255, 255, 0.25)",
            "showline": True,
            "linecolor": "rgba(255, 255, 255, 0.25)",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(15, 23, 42, 0.85)",
            "bordercolor": "rgba(255, 255, 255, 0.15)",
            "borderwidth": 1,
            "font": {"color": "#E2E8F0", "size": 11},
        },
        hovermode=hovermode,
        hoverlabel={
            "bgcolor": "#0F172A",
            "font_size": 12,
            "font_family": "Inter, Arial, sans-serif",
            "font_color": "#F8FAFC",
            "bordercolor": "#38BDF8",
        },
        margin={"l": 55, "r": 30, "t": 60, "b": 45},
        height=450,
    )
    return fig


def create_equity_curve(
    results: pd.DataFrame,
    use_log_scale: bool = False,
    normalize: bool = True,
) -> go.Figure:
    """Generate an interactive cumulative equity progression chart.

    Args:
        results: DataFrame containing 'equity_curve' and 'benchmark_equity_curve'.
        use_log_scale: Whether to set y-axis to logarithmic scale.
        normalize: If True, re-bases starting value to 1.0 for clean comparison.

    Returns:
        Plotly Graph Objects Figure.
    """
    fig = go.Figure()

    if results.empty:
        return _apply_institutional_theme(fig, "Cumulative Equity Progression", "Portfolio Wealth")

    strat_eq = results["equity_curve"].copy()
    bench_eq = results.get("benchmark_equity_curve", pd.Series(1.0, index=results.index)).copy()

    if normalize:
        strat_base = strat_eq.iloc[0] if strat_eq.iloc[0] != 0 else 1.0
        bench_base = bench_eq.iloc[0] if bench_eq.iloc[0] != 0 else 1.0
        strat_plot = strat_eq / strat_base
        bench_plot = bench_eq / bench_base
        yaxis_label = "Normalized Wealth (Base = 1.00)"
        val_hover_fmt = ":.3f"
    else:
        strat_plot = strat_eq
        bench_plot = bench_eq
        yaxis_label = "Portfolio Value (INR)"
        val_hover_fmt = ":,.2f"

    # Primary Strategy Trace (Bright Electric Cyan - 100% visible on black)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=strat_plot,
            name="Strategy (Net of Friction)",
            mode="lines",
            line={"color": "#38BDF8", "width": 2.4},
            hovertemplate=f"Strategy: %{{y{val_hover_fmt}}}<extra></extra>",
        )
    )

    # Benchmark Trace (Bright Silver Gray)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=bench_plot,
            name="Benchmark (Nifty 50)",
            mode="lines",
            line={"color": "#94A3B8", "width": 1.6, "dash": "dash"},
            hovertemplate=f"Nifty 50: %{{y{val_hover_fmt}}}<extra></extra>",
        )
    )

    if use_log_scale:
        fig.update_yaxes(type="log")

    return _apply_institutional_theme(fig, "Cumulative Equity Progression: Strategy vs Nifty 50", yaxis_label)


def create_drawdown_plot(results: pd.DataFrame) -> go.Figure:
    """Generate an interactive underwater drawdown plot.

    Args:
        results: DataFrame containing 'drawdown' and 'benchmark_drawdown'.

    Returns:
        Plotly Graph Objects Figure.
    """
    fig = go.Figure()

    if results.empty:
        return _apply_institutional_theme(fig, "Underwater Drawdown Analysis", "Drawdown (%)")

    strat_dd = results["drawdown"] * 100.0
    bench_dd = results.get("benchmark_drawdown", pd.Series(0.0, index=results.index)) * 100.0

    # Strategy Area Drawdown (Vibrant Coral/Rose Red with translucent fill)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=strat_dd,
            name="Strategy Drawdown",
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(244, 63, 94, 0.22)",
            line={"color": "#F43F5E", "width": 2.0},
            hovertemplate="Strategy Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    # Superimposed Benchmark Line (Bright Amber/Gold - High Contrast on Dark/Black)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=bench_dd,
            name="Nifty 50 Drawdown",
            mode="lines",
            line={"color": "#FBBF24", "width": 1.5, "dash": "dot"},
            hovertemplate="Nifty 50 Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_yaxes(ticksuffix="%")
    return _apply_institutional_theme(fig, "Underwater Drawdown Comparison (% Drop from Peak)", "Drawdown (%)")


def create_allocation_area_chart(weights: pd.DataFrame) -> go.Figure:
    """Generate a stacked area chart displaying asset weight allocations over time.

    Args:
        weights: DataFrame of target or executed portfolio weights (dates x tickers).

    Returns:
        Plotly Graph Objects Figure.
    """
    fig = go.Figure()

    if weights.empty:
        return _apply_institutional_theme(fig, "Historical Portfolio Asset Allocation", "Allocation Weight (%)")

    # Filter out columns that have strictly 0 weight everywhere
    active_cols = [c for c in weights.columns if (weights[c].abs().sum() > 1e-4)]
    if not active_cols:
        active_cols = list(weights.columns[:5])

    # Dynamic vibrant color palette calibrated for black/dark themes
    palette = [
        "#38BDF8",  # Sky Blue
        "#34D399",  # Emerald Green
        "#FBBF24",  # Amber Gold
        "#F43F5E",  # Rose Red
        "#A78BFA",  # Violet
        "#F472B6",  # Pink
        "#FB923C",  # Orange
        "#2DD4BF",  # Teal
        "#818CF8",  # Indigo
        "#4ADE80",  # Light Green
        "#E879F9",  # Fuchsia
        "#67E8F9",  # Cyan
        "#C084FC",  # Purple
        "#FDE047",  # Yellow
    ]

    for idx, col in enumerate(active_cols):
        clean_name = col.replace(".NS", "")
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[col] * 100.0,
                name=clean_name,
                mode="lines",
                stackgroup="one",
                line={"width": 0.5, "color": color},
                fillcolor=color,
                hovertemplate=f"{clean_name}: %{{y:.1f}}%<extra></extra>",
            )
        )

    fig.update_yaxes(ticksuffix="%", range=[0, 105])
    return _apply_institutional_theme(
        fig,
        f"Dynamic Portfolio Allocation ({len(active_cols)} Active Assets)",
        "Allocation Weight (%)",
    )


def create_rolling_sharpe_chart(
    results: pd.DataFrame,
    window: int = 126,
    rf_annual: float = 0.065,
) -> go.Figure:
    """Generate rolling 6-month (126-day) annualized Sharpe ratio comparison.

    Args:
        results: DataFrame with 'net_return' and 'benchmark_return'.
        window: Rolling observation window in trading days (default 126 = ~6 months).
        rf_annual: Annualized risk-free rate (default 6.5%).

    Returns:
        Plotly Graph Objects Figure.
    """
    fig = go.Figure()

    if results.empty or len(results) < window:
        return _apply_institutional_theme(fig, f"Rolling {window}-Day Sharpe Ratio", "Sharpe Ratio")

    daily_rf = rf_annual / 252.0
    strat_ret = results["net_return"]
    bench_ret = results.get("benchmark_return", pd.Series(0.0, index=results.index))

    # Rolling annualized Sharpe calculations
    strat_mean = strat_ret.rolling(window=window, min_periods=window).mean()
    strat_std = strat_ret.rolling(window=window, min_periods=window).std(ddof=1)
    strat_rolling_sharpe = ((strat_mean - daily_rf) / strat_std.replace(0.0, np.nan)) * np.sqrt(252.0)

    bench_mean = bench_ret.rolling(window=window, min_periods=window).mean()
    bench_std = bench_ret.rolling(window=window, min_periods=window).std(ddof=1)
    bench_rolling_sharpe = ((bench_mean - daily_rf) / bench_std.replace(0.0, np.nan)) * np.sqrt(252.0)

    # Strategy Rolling Sharpe Trace (Electric Cyan)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=strat_rolling_sharpe,
            name=f"Strategy Rolling Sharpe ({window}d)",
            mode="lines",
            line={"color": "#38BDF8", "width": 2.2},
            hovertemplate="Strategy Sharpe: %{y:.2f}<extra></extra>",
        )
    )

    # Benchmark Rolling Sharpe Trace (Silver Gray)
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=bench_rolling_sharpe,
            name=f"Nifty 50 Rolling Sharpe ({window}d)",
            mode="lines",
            line={"color": "#94A3B8", "width": 1.6, "dash": "dash"},
            hovertemplate="Nifty 50 Sharpe: %{y:.2f}<extra></extra>",
        )
    )

    # Reference Zero Line (High-contrast red dashed line)
    fig.add_hline(
        y=0.0,
        line={"color": "#EF4444", "width": 1.4, "dash": "dot"},
        annotation_text="Zero Excess Return",
        annotation_position="bottom right",
        annotation_font={"color": "#FCA5A5", "size": 11},
    )

    return _apply_institutional_theme(
        fig,
        f"Rolling {window}-Day Annualized Sharpe Ratio (Rf = {rf_annual*100:.1f}%)",
        "Annualized Sharpe Ratio",
    )
