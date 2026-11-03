"""Reusable UI components, styling helpers, and interactive info popovers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.backtester import BacktestResult
from src.data_loader import DataLoader
from src.report import compute_monthly_returns_matrix


def inject_custom_css() -> None:
    """Inject institutional styling CSS calibrated for dark/black and high-contrast themes."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 96%;
        }

        /* Top Header Card - Dark Institutional Theme */
        .header-container {
            background: linear-gradient(135deg, #0B1120 0%, #1E293B 100%);
            border-radius: 12px;
            padding: 22px 28px;
            color: #FFFFFF !important;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .header-title {
            font-size: 1.85rem;
            font-weight: 800;
            color: #FFFFFF !important;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .header-subtitle {
            font-size: 0.95rem;
            color: #94A3B8 !important;
            margin-top: 5px;
            margin-bottom: 14px;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            background: rgba(30, 41, 59, 0.95);
            border: 1px solid #475569;
            border-radius: 20px;
            padding: 5px 12px;
            font-size: 0.8rem;
            color: #F1F5F9 !important;
            margin-right: 8px;
            margin-bottom: 6px;
            font-weight: 500;
        }
        .status-pill b {
            color: #38BDF8 !important;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10B981;
            margin-right: 6px;
            box-shadow: 0 0 6px #10B981;
        }

        /* Metric Card Accents - High Contrast on Dark/Black Themes */
        div[data-testid="stMetric"] {
            background-color: rgba(30, 41, 59, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
            border-radius: 10px;
            padding: 14px 18px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #38BDF8 !important;
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.84rem !important;
            font-weight: 600 !important;
            color: #94A3B8 !important; /* Crisp light silver */
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        div[data-testid="stMetricValue"] div {
            font-size: 1.7rem !important;
            font-weight: 800 !important;
            color: #FFFFFF !important; /* Pure white text */
            letter-spacing: -0.02em;
        }

        /* Tab Bar Styling */
        button[data-baseweb="tab"] {
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            color: #94A3B8 !important;
            padding-top: 10px !important;
            padding-bottom: 10px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38BDF8 !important; /* Bright cyan active tab */
            border-bottom: 3px solid #38BDF8 !important;
            background-color: transparent !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #FFFFFF !important;
        }

        /* Section Container Cards */
        .chart-card {
            background-color: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }

        /* Popover Content Styling */
        div[data-testid="stPopoverBody"] {
            background-color: #0F172A !important;
            border: 1px solid #334155 !important;
            color: #F8FAFC !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
        }
        .info-popover-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #38BDF8 !important; /* Bright Cyan header */
            margin-bottom: 6px;
        }
        .info-popover-text {
            font-size: 0.88rem;
            line-height: 1.45;
            color: #E2E8F0 !important; /* Crisp light slate */
            margin-bottom: 8px;
        }
        .info-formula-box {
            background-color: rgba(15, 23, 42, 0.9) !important;
            border-left: 3px solid #38BDF8 !important;
            border: 1px solid #334155 !important;
            border-left-width: 3px !important;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.84rem;
            color: #38BDF8 !important;
            margin: 8px 0;
            font-family: monospace;
        }

        /* Disclaimer Card */
        .disclaimer-box {
            background-color: rgba(30, 41, 59, 0.65) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-left: 4px solid #38BDF8 !important;
            padding: 12px 18px;
            border-radius: 6px;
            font-size: 0.86rem;
            color: #E2E8F0 !important; /* Crisp light white/slate */
            margin-top: 1.2rem;
            margin-bottom: 1rem;
        }
        .disclaimer-box b {
            color: #38BDF8 !important;
        }
        .disclaimer-box code {
            background-color: rgba(15, 23, 42, 0.9) !important;
            color: #38BDF8 !important;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header_banner(
    strategy_name: str,
    start_date: str,
    end_date: str,
    n_assets: int,
    cost_bps: float,
) -> None:
    """Render top institutional header card with live status badges."""
    st.markdown(
        f"""
        <div class="header-container">
            <h1 class="header-title">NSE Quantitative Research Dashboard</h1>
            <div class="header-subtitle">
                Institutional-Grade Vectorised Strategy Simulation Engine • National Stock Exchange of India
            </div>
            <div>
                <span class="status-pill"><span class="status-dot"></span>Engine: <b>Active</b></span>
                <span class="status-pill">📊 Model: <b>{strategy_name}</b></span>
                <span class="status-pill">📅 Period: <b>{start_date}</b> to <b>{end_date}</b></span>
                <span class="status-pill">🏢 Universe: <b>{n_assets} Liquid Bluechips</b></span>
                <span class="status-pill">⚡ Friction: <b>{cost_bps:.1f} bps/side</b></span>
                <span class="status-pill">🛡️ Bias Defense: <b>Shift(1) Lag Enforced</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_cached_market_data(
    start_date: str,
    end_date: str,
    force_reload: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Streamlit cached wrapper around DataLoader to prevent redundant network or disk reads."""
    loader = DataLoader(cache_dir="data")
    try:
        return loader.fetch_data(
            start_date=start_date,
            end_date=end_date,
            force_reload=force_reload,
        )
    except Exception as e:
        st.warning(f"Live market data fetch encountered an issue: {e}. Generating offline synthetic market paths.")
        return DataLoader.generate_synthetic_data(start_date=start_date)


def render_kpi_ribbon(result: BacktestResult) -> None:
    """Render the top executive KPI ribbon with built-in (i) description buttons."""
    cagr = result.metrics.get("CAGR", 0.0)
    bench_cagr = result.benchmark_metrics.get("CAGR", 0.0)
    cagr_delta = cagr - bench_cagr

    sharpe = result.metrics.get("Sharpe Ratio", 0.0)
    bench_sharpe = result.benchmark_metrics.get("Sharpe Ratio", 0.0)
    sharpe_delta = sharpe - bench_sharpe

    mdd = result.metrics.get("Max Drawdown", 0.0)
    bench_mdd = result.benchmark_metrics.get("Max Drawdown", 0.0)
    # For drawdown: if strategy MDD is -20% and benchmark is -38%, delta is +18% (improvement)
    mdd_improvement = abs(bench_mdd) - abs(mdd)

    alpha = result.metrics.get("Alpha (CAPM)", 0.0)
    beta = result.metrics.get("Beta (CAPM)", 1.0)

    # Row 1: Primary Metrics with full help tooltips
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="CAGR (Strategy)",
            value=f"{cagr * 100.0:+.2f}%",
            delta=f"{cagr_delta * 100.0:+.2f}% vs Nifty 50",
            delta_color="normal",
            help=(
                "Compound Annual Growth Rate (CAGR):\n\n"
                "The annualized geometric return rate of the portfolio assuming profits are continuously reinvested.\n\n"
                "Formula: CAGR = (1 + Total Return)^(252 / N) - 1\n\n"
                "Higher is better. Shows the true annualized compounding speed after friction."
            ),
        )

    with col2:
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}",
            delta=f"{sharpe_delta:+.2f} vs Nifty 50",
            delta_color="normal",
            help=(
                "Sharpe Ratio (Risk-Adjusted Return):\n\n"
                "Measures the excess return generated per unit of total annualized volatility relative to the Indian Risk-Free Rate (6.5%).\n\n"
                "Formula: Sharpe = [Mean(R - Rf) / Std(R)] * sqrt(252)\n\n"
                "> 1.0 is considered institutional grade. > 0.5 outperforms broader market."
            ),
        )

    with col3:
        st.metric(
            label="Max Drawdown",
            value=f"{mdd * 100.0:.2f}%",
            delta=f"{mdd_improvement * 100.0:+.2f}% vs Nifty 50" if bench_mdd != 0 else None,
            delta_color="normal" if mdd_improvement >= 0 else "inverse",
            help=(
                "Maximum Drawdown (MDD):\n\n"
                "The largest peak-to-trough equity decline experienced during the simulation.\n\n"
                "Formula: MDD = Min_t [ (Equity_t - Peak_t) / Peak_t ]\n\n"
                "Represents capital risk and pain tolerance. Lower absolute magnitude is better."
            ),
        )

    with col4:
        st.metric(
            label="CAPM Alpha (Annualized)",
            value=f"{alpha * 100.0:+.2f}%",
            delta=f"Beta: {beta:.2f} to Nifty 50",
            delta_color="off",
            help=(
                "CAPM Annualized Alpha & Beta:\n\n"
                "Alpha is the true risk-adjusted excess return unexplained by broader market movements (Nifty 50).\n\n"
                "Beta measures market correlation & sensitivity: Beta = Cov(R_strat, R_bench) / Var(R_bench).\n\n"
                "Beta < 1.0 indicates lower volatility exposure than the Nifty 50 index."
            ),
        )

    # Expandable Secondary Quantitative Diagnostics
    with st.expander("🔍 View Secondary Quantitative Diagnostics (Sortino, Calmar, Turnover, Hit Rate)", expanded=False):
        sortino = result.metrics.get("Sortino Ratio", 0.0)
        calmar = result.metrics.get("Calmar Ratio", 0.0)
        turnover = result.metrics.get("Annualized Turnover", 0.0)
        hit_rate = result.metrics.get("Hit Rate", 0.0)

        scol1, scol2, scol3, scol4 = st.columns(4)
        with scol1:
            st.metric(
                label="Sortino Ratio",
                value=f"{sortino:.2f}",
                help=(
                    "Sortino Ratio:\n\n"
                    "Similar to Sharpe, but only penalizes downside volatility below the risk-free rate, ignoring upside volatility.\n\n"
                    "Formula: Sortino = (Mean Excess Return) / Downside Semivariance"
                ),
            )
        with scol2:
            st.metric(
                label="Calmar Ratio",
                value=f"{calmar:.2f}",
                help=(
                    "Calmar Ratio:\n\n"
                    "Ratio of CAGR to Maximum Drawdown magnitude: CAGR / |MDD|.\n\n"
                    "Measures return earned per unit of catastrophic tail risk."
                ),
            )
        with scol3:
            st.metric(
                label="Annualized Turnover",
                value=f"{turnover * 100.0:.1f}%",
                help=(
                    "Annualized Turnover:\n\n"
                    "Sum of all absolute weight shifts over 252 trading days: 252 * Mean(Turnover_daily).\n\n"
                    "High turnover increases STT, GST, and slippage drag."
                ),
            )
        with scol4:
            st.metric(
                label="Daily Hit Rate",
                value=f"{hit_rate * 100.0:.1f}%",
                help=(
                    "Daily Hit Rate (Win %):\n\n"
                    "Percentage of active trading days with positive net returns after transaction costs."
                ),
            )


def render_info_popover(
    title: str,
    summary: str,
    formula: Optional[str] = None,
    interpretation: Optional[str] = None,
) -> None:
    """Render a compact (i) Info popover button explaining what a chart or section represents."""
    with st.popover("ℹ️ What is this?"):
        st.markdown(f"<div class='info-popover-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-popover-text'>{summary}</div>", unsafe_allow_html=True)
        if formula:
            st.markdown(f"<div class='info-formula-box'>{formula}</div>", unsafe_allow_html=True)
        if interpretation:
            st.markdown(f"<div class='info-popover-text'><b>How to interpret:</b> {interpretation}</div>", unsafe_allow_html=True)


def render_performance_table_component(result: BacktestResult) -> None:
    """Render formatted comparison metrics table with styled percentages and deltas."""
    summary_df = result.summary().reset_index()

    pct_metrics = {
        "Total Return",
        "CAGR",
        "Annualized Volatility",
        "Max Drawdown",
        "Alpha (CAPM)",
        "Hit Rate",
        "Annualized Turnover",
    }

    formatted_rows = []
    for _, row in summary_df.iterrows():
        m_name = row["Metric"]
        s_val = row["Strategy"]
        b_val = row["Benchmark (^NSEI)"]
        d_val = row["Delta (Strategy - Benchmark)"]

        if m_name in pct_metrics:
            s_str = f"{s_val * 100.0:+.2f}%" if not np.isnan(s_val) else "-"
            b_str = f"{b_val * 100.0:+.2f}%" if not np.isnan(b_val) else "-"
            d_str = f"{d_val * 100.0:+.2f}%" if not np.isnan(d_val) else "-"
        else:
            s_str = f"{s_val:.2f}" if not np.isnan(s_val) else "-"
            b_str = f"{b_val:.2f}" if not np.isnan(b_val) else "-"
            d_str = f"{d_val:+.2f}" if not np.isnan(d_val) else "-"

        formatted_rows.append(
            {
                "Quantitative Metric": m_name,
                f"Strategy ({result.strategy_name})": s_str,
                "Benchmark (Nifty 50)": b_str,
                "Delta (Strategy - Bench)": d_str,
            }
        )

    display_df = pd.DataFrame(formatted_rows)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_monthly_matrix_heatmap(returns: pd.Series) -> None:
    """Render styled interactive heatmap grid for Year x Month returns with summary stats."""
    matrix = compute_monthly_returns_matrix(returns)
    if matrix.empty:
        st.info("Insufficient history to calculate monthly returns matrix.")
        return

    # Color formatter calibrated for dark/black themes
    def color_returns(val: object) -> str:
        if isinstance(val, (int, float)) and not np.isnan(val):
            if val > 0.05:
                return "background-color: rgba(16, 185, 129, 0.35); color: #34D399; font-weight: bold;"
            elif val > 0:
                return "background-color: rgba(16, 185, 129, 0.18); color: #6EE7B7;"
            elif val < -0.05:
                return "background-color: rgba(239, 68, 68, 0.35); color: #F87171; font-weight: bold;"
            elif val < 0:
                return "background-color: rgba(239, 68, 68, 0.18); color: #FCA5A5;"
        return ""

    styled = matrix.style.format("{:+.2%}", na_rep="-").map(color_returns)
    st.dataframe(styled, use_container_width=True)

    # Quick Summary Statistics for Monthly Distribution
    clean_m = matrix.drop(columns=["YTD"], errors="ignore").values.flatten()
    clean_m = clean_m[~np.isnan(clean_m)]

    if len(clean_m) > 0:
        pos_months = (clean_m > 0).sum()
        total_months = len(clean_m)
        best_m = np.max(clean_m)
        worst_m = np.min(clean_m)

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(f"<span style='color:#CBD5E1'>Positive Months:</span> <b style='color:#34D399'>{pos_months} / {total_months} ({pos_months/total_months*100:.1f}%)</b>", unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"<span style='color:#CBD5E1'>Best Month:</span> <b style='color:#34D399'>{best_m*100:+.2f}%</b>", unsafe_allow_html=True)
        with mcol3:
            st.markdown(f"<span style='color:#CBD5E1'>Worst Month:</span> <b style='color:#F87171'>{worst_m*100:+.2f}%</b>", unsafe_allow_html=True)
        with mcol4:
            st.markdown(f"<span style='color:#CBD5E1'>Median Monthly:</span> <b style='color:#38BDF8'>{np.median(clean_m)*100:+.2f}%</b>", unsafe_allow_html=True)
