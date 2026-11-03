"""Sidebar controls, quick presets, and interactive parameter guidance for the nse-backtester dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Tuple

import streamlit as st


@dataclass
class SidebarConfig:
    """Encapsulates all validated user parameters from sidebar inputs."""

    start_date: str
    end_date: str
    strategy_type: Literal["momentum", "mean_reversion"]
    # Momentum Params
    lookback_days: int
    skip_days: int
    top_quantile: float
    rebalance_freq: Literal["monthly", "21_days"]
    # Mean Reversion Params
    mr_window: int
    entry_z: float
    exit_z: float
    mr_weighting: Literal["equal", "inverse_vol"]
    # Friction & Risk Params
    cost_bps: float
    rf_annual: float
    max_weight: float
    cash_buffer: float
    force_reload: bool
    llm_provider: str = "auto"
    llm_api_key: str = ""


def render_sidebar() -> Tuple[SidebarConfig, bool]:
    """Render sidebar widgets with comprehensive (i) tooltip guidance and preset triggers.

    Returns:
        Tuple of (SidebarConfig, run_button_clicked: bool).
    """
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 14px 0;">
            <h2 style="margin: 0; color: #38BDF8; font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em;">
                NSE Quantitative Lab
            </h2>
            <p style="margin: 3px 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                Dynamic Parameter Tuning & Risk Configuration
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # Quick Horizon Presets
    # -------------------------------------------------------------
    with st.sidebar.expander("⚡ Quick Horizon Presets", expanded=False):
        st.caption("Click a preset to automatically populate common research horizons:")
        pcol1, pcol2 = st.columns(2)
        preset_start = None
        preset_end = None
        with pcol1:
            if st.button("Full 10Y (2016-25)", use_container_width=True):
                st.session_state["start_date_val"] = date(2016, 1, 1)
                st.session_state["end_date_val"] = date(2025, 12, 31)
            if st.button("Pre-COVID (16-19)", use_container_width=True):
                st.session_state["start_date_val"] = date(2016, 1, 1)
                st.session_state["end_date_val"] = date(2019, 12, 31)
        with pcol2:
            if st.button("Post-COVID (20-25)", use_container_width=True):
                st.session_state["start_date_val"] = date(2020, 1, 1)
                st.session_state["end_date_val"] = date(2025, 12, 31)
            if st.button("Recent 3Y (22-25)", use_container_width=True):
                st.session_state["start_date_val"] = date(2022, 1, 1)
                st.session_state["end_date_val"] = date(2025, 12, 31)

    # -------------------------------------------------------------
    # 1. Backtest Horizon (Date Selection)
    # -------------------------------------------------------------
    st.sidebar.subheader("📅 Backtest Period")
    def_start = st.session_state.get("start_date_val", date(2016, 1, 1))
    def_end = st.session_state.get("end_date_val", date(2025, 12, 31))

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_d = st.date_input(
            "Start Date",
            value=def_start,
            min_value=date(2010, 1, 1),
            max_value=date(2026, 12, 31),
            help="Starting date for historical price ingestion. Data cached in local Parquet storage.",
        )
    with col2:
        end_d = st.date_input(
            "End Date",
            value=def_end,
            min_value=date(2012, 1, 1),
            max_value=date(2026, 12, 31),
            help="Ending date for the backtest window.",
        )

    if start_d >= end_d:
        st.sidebar.error("Start Date must precede End Date.")

    # -------------------------------------------------------------
    # 2. Strategy Hyperparameters
    # -------------------------------------------------------------
    st.sidebar.subheader("🧠 Quantitative Strategy")
    strat_choice = st.sidebar.radio(
        "Strategy Model",
        options=["12-1 Month Momentum", "20-Day Mean Reversion"],
        index=0,
        help=(
            "Select algorithmic logic:\n\n"
            "• Momentum: Cross-sectional relative strength selecting top performers over past 12 months.\n\n"
            "• Mean Reversion: Statistical arbitrage buying oversold stocks relative to their 20-day rolling SMA."
        ),
    )

    # Defaults
    lookback_days = 252
    skip_days = 21
    top_quantile = 0.20
    rebalance_freq: Literal["monthly", "21_days"] = "monthly"
    mr_window = 20
    entry_z = -1.5
    exit_z = 0.0
    mr_weighting: Literal["equal", "inverse_vol"] = "inverse_vol"

    if strat_choice == "12-1 Month Momentum":
        strategy_type = "momentum"
        with st.sidebar.expander("⚙️ Momentum Hyperparameters", expanded=True):
            lookback_days = st.slider(
                "Lookback Formation (Days)",
                min_value=126,
                max_value=504,
                value=252,
                step=21,
                help=(
                    "Lookback Period:\n\n"
                    "Total historical window used to calculate cross-sectional momentum return.\n"
                    "Default: 252 trading days (~12 calendar months)."
                ),
            )
            skip_days = st.slider(
                "Skip Lag (Days)",
                min_value=5,
                max_value=42,
                value=21,
                step=1,
                help=(
                    "Microstructure Reversal Skip Window:\n\n"
                    "Skips the most recent N days (t - skip to t) to eliminate short-term bid-ask bounce and liquidity provision reversal (Jegadeesh & Titman 1993).\n"
                    "Default: 21 trading days (~1 month)."
                ),
            )
            top_q_pct = st.slider(
                "Top Selection Cutoff (%)",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help=(
                    "Top Quantile Fraction:\n\n"
                    "Fraction of universe selected into the active portfolio.\n"
                    "20% = Top Quintile (~6 stocks in a 30-stock universe)."
                ),
            )
            top_quantile = top_q_pct / 100.0

            freq_choice = st.selectbox(
                "Rebalancing Schedule",
                options=["Monthly (Month-End)", "Every 21 Trading Days"],
                index=0,
                help=(
                    "Portfolio Rebalancing Frequency:\n\n"
                    "• Monthly: Rebalance on the last trading session of each calendar month.\n"
                    "• Every 21 Days: Fixed periodic 21-day trading cycle."
                ),
            )
            rebalance_freq = "monthly" if "Monthly" in freq_choice else "21_days"

    else:
        strategy_type = "mean_reversion"
        with st.sidebar.expander("⚙️ Mean Reversion Hyperparameters", expanded=True):
            mr_window = st.slider(
                "Rolling SMA Window",
                min_value=10,
                max_value=60,
                value=20,
                step=2,
                help=(
                    "Rolling Equilibrium Window:\n\n"
                    "Number of trading days used to calculate rolling mean and standard deviation.\n"
                    "Default: 20 trading days (~1 month)."
                ),
            )
            entry_z = st.slider(
                "Entry Z-Score (Oversold)",
                min_value=-3.0,
                max_value=-0.5,
                value=-1.5,
                step=0.1,
                help=(
                    "Long Entry Trigger:\n\n"
                    "Enters long position when price drops below (SMA - Z * Std).\n"
                    "Z = -1.5 corresponds to the 6.7th percentile of a standard normal distribution."
                ),
            )
            exit_z = st.slider(
                "Exit Z-Score (Target Reversion)",
                min_value=-0.5,
                max_value=1.5,
                value=0.0,
                step=0.1,
                help=(
                    "Liquidation Trigger:\n\n"
                    "Exits position once price rebounds back to rolling equilibrium.\n"
                    "Z = 0.0 represents exact return to the 20-day moving average."
                ),
            )
            weight_type = st.selectbox(
                "Allocation Weighting",
                options=["Inverse Volatility (Risk-Parity)", "Equal Weight"],
                index=0,
                help=(
                    "Asset Sizing Scheme:\n\n"
                    "• Inverse Volatility: Allocates weight proportional to 1/volatility, dampening risk on turbulent stocks.\n"
                    "• Equal Weight: Equal division among all active triggers."
                ),
            )
            mr_weighting = "inverse_vol" if "Inverse" in weight_type else "equal"

    # -------------------------------------------------------------
    # 3. Execution Friction & Indian Market Microstructure
    # -------------------------------------------------------------
    st.sidebar.subheader("⚡ Friction & Execution Drag")
    with st.sidebar.expander("🇮🇳 Indian Market Friction Details", expanded=False):
        cost_bps = st.slider(
            "Transaction Cost (bps / side)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help=(
                "One-Way Transaction Friction (in Basis Points, 1 bp = 0.01%):\n\n"
                "Reflects Indian statutory charges on equity delivery:\n"
                "• STT (Securities Transaction Tax): 0.10% on turnover\n"
                "• Stamp Duty: 0.015%\n"
                "• Exchange Turnover + SEBI fees: ~0.0035%\n"
                "• GST (18% on brokerage/charges): ~0.005%\n"
                "• Bid-Ask Spread Slippage: ~0.025%\n\n"
                "Total default: 15 bps per side (30 bps round-trip)."
            ),
        )
        rf_pct = st.number_input(
            "Risk-Free Rate (% p.a.)",
            min_value=0.0,
            max_value=15.0,
            value=6.5,
            step=0.25,
            help="Annualized benchmark risk-free rate (calibrated to the Reserve Bank of India Repo Rate, 6.5%).",
        )

    # -------------------------------------------------------------
    # 4. Institutional Portfolio & Risk Controls
    # -------------------------------------------------------------
    st.sidebar.subheader("🛡️ Risk Limits & Constraints")
    with st.sidebar.expander("Portfolio Diversification Limits", expanded=False):
        max_weight_pct = st.slider(
            "Single-Stock Max Cap (%)",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            help=(
                "Single-Stock Allocation Limit:\n\n"
                "Guarantees that no individual stock exceeds this percentage of total equity.\n"
                "Excess weight is iteratively redistributed across uncapped active assets (default 15%)."
            ),
        )
        cash_buffer_pct = st.slider(
            "Mandatory Cash Reserve (%)",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
            help=(
                "Cash Liquidity Buffer:\n\n"
                "Mandatory uninvested cash buffer reserved for liquidity.\n"
                "Maximum gross equity exposure = 100% - Cash Reserve (default 5% cash, 95% max equity)."
            ),
        )

    # -------------------------------------------------------------
    # 5. Data Cache Control
    # -------------------------------------------------------------
    force_reload = st.sidebar.checkbox(
        "Bypass Cache (Force YFinance)",
        value=False,
        help="Check to force a live download from Yahoo Finance instead of reading from the local Parquet cache.",
    )

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # 6. AI Performance Teardown Settings
    # -------------------------------------------------------------
    with st.sidebar.expander("🤖 AI Commentary Settings", expanded=False):
        llm_provider = st.selectbox(
            "AI Provider",
            options=["auto", "gemini", "openai", "fallback"],
            format_func=lambda x: {
                "auto": "⚡ Auto (Gemini / OpenAI / Rule-Engine)",
                "gemini": "✨ Google Gemini (google-genai)",
                "openai": "🧠 OpenAI (GPT-4o-mini)",
                "fallback": "📊 Rule-Based Quant Engine",
            }.get(x, x),
            help="Select the AI engine to generate qualitative performance analysis.",
        )
        llm_api_key = st.text_input(
            "API Key (Optional)",
            value="",
            type="password",
            help="Enter your Gemini or OpenAI API key. If left blank, reads GEMINI_API_KEY/OPENAI_API_KEY env vars or uses the algorithmic engine.",
        )

    # -------------------------------------------------------------
    # 7. Action Button
    # -------------------------------------------------------------
    st.sidebar.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    run_btn = st.sidebar.button("🚀 Run Backtest Simulation", type="primary", use_container_width=True)

    config = SidebarConfig(
        start_date=start_d.strftime("%Y-%m-%d"),
        end_date=end_d.strftime("%Y-%m-%d"),
        strategy_type=strategy_type,
        lookback_days=lookback_days,
        skip_days=skip_days,
        top_quantile=top_quantile,
        rebalance_freq=rebalance_freq,
        mr_window=mr_window,
        entry_z=entry_z,
        exit_z=exit_z,
        mr_weighting=mr_weighting,
        cost_bps=cost_bps,
        rf_annual=rf_pct / 100.0,
        max_weight=max_weight_pct / 100.0,
        cash_buffer=cash_buffer_pct / 100.0,
        force_reload=force_reload,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
    )

    return config, run_btn
