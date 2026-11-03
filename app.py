"""Main Streamlit application entry point for nse-backtester quantitative research dashboard."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.backtester import Backtester, BacktestResult
from src.data_loader import DataLoader
from src.portfolio import PortfolioManager
from src.report import generate_excel_report, plot_performance
from src.strategy import MeanReversionStrategy, MomentumStrategy
from ui import (
    create_allocation_area_chart,
    create_drawdown_plot,
    create_equity_curve,
    create_rolling_sharpe_chart,
    inject_custom_css,
    load_cached_market_data,
    render_header_banner,
    render_info_popover,
    render_kpi_ribbon,
    render_monthly_matrix_heatmap,
    render_performance_table_component,
    render_sidebar,
)

# 1. Page Configuration
st.set_page_config(
    page_title="NSE Quant Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Institutional Styling
inject_custom_css()

# 3. Render Sidebar Controls
config, run_clicked = render_sidebar()


# 4. Simulation Pipeline
def run_simulation() -> BacktestResult | None:
    """Execute backtest simulation pipeline strictly using core engine modules."""
    with st.spinner("🔄 Ingesting historical adjusted price data & simulating portfolio execution..."):
        prices, benchmark = load_cached_market_data(
            start_date=config.start_date,
            end_date=config.end_date,
            force_reload=config.force_reload,
        )

        # Slice data to requested date range
        prices = prices.loc[config.start_date : config.end_date]
        benchmark = benchmark.loc[config.start_date : config.end_date]

        if prices.empty or len(prices) < 30:
            st.error("Selected date range contains insufficient trading days. Please choose a broader window.")
            return None

        # Check warmup requirement for Momentum Strategy
        if config.strategy_type == "momentum" and len(prices) <= config.lookback_days:
            st.warning(
                f"Date range contains {len(prices)} trading days, which is less than the momentum lookback "
                f"period ({config.lookback_days} days). Please expand your date range or reduce the lookback."
            )
            return None

        # Setup Strategy
        if config.strategy_type == "momentum":
            strategy = MomentumStrategy(
                lookback_days=config.lookback_days,
                skip_days=config.skip_days,
                top_quantile=config.top_quantile,
                rebalance_freq=config.rebalance_freq,
            )
            strategy_name = f"Momentum ({config.lookback_days}-{config.skip_days}d)"
        else:
            strategy = MeanReversionStrategy(
                window=config.mr_window,
                entry_z=config.entry_z,
                exit_z=config.exit_z,
                weighting=config.mr_weighting,
            )
            strategy_name = f"Mean Reversion (Z {config.entry_z}/{config.exit_z})"

        # Generate signals
        raw_weights = strategy.generate_signals(prices)

        # Apply Risk & Sizing Constraints via PortfolioManager
        pm = PortfolioManager(
            max_weight=config.max_weight,
            cash_buffer=config.cash_buffer,
            max_gross_exposure=1.0 - config.cash_buffer,
        )
        constrained_weights = pm.process_weights(raw_weights)

        # Execute Backtest Simulation Engine (Enforces weights.shift(1) execution lag)
        cost_fraction = config.cost_bps / 10000.0
        engine = Backtester(
            cost_per_side=cost_fraction,
            rf_annual=config.rf_annual,
            initial_capital=1_000_000.0,
        )

        result = engine.run(
            prices=prices,
            strategy_or_weights=constrained_weights,
            benchmark=benchmark,
            strategy_name=strategy_name,
        )
        return result


# 5. Session State Management
if "backtest_result" not in st.session_state or run_clicked:
    res = run_simulation()
    if res is not None:
        st.session_state["backtest_result"] = res
        st.session_state["current_config"] = config

current_result: BacktestResult | None = st.session_state.get("backtest_result", None)

if current_result is not None:
    hist = current_result.portfolio_history
    n_assets = current_result.executed_weights.shape[1]

    # 6. Render Institutional Header Banner
    render_header_banner(
        strategy_name=current_result.strategy_name,
        start_date=config.start_date,
        end_date=config.end_date,
        n_assets=n_assets,
        cost_bps=config.cost_bps,
    )

    # 7. Render Top Metric Ribbon with (i) Buttons
    render_kpi_ribbon(current_result)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 8. Interactive Tabbed Layout
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Performance & Risk Visuals",
            "📊 Metric Deep Dive & Heatmap",
            "💼 Dynamic Asset Allocation",
            "📥 Institutional Export Center",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: Performance & Risk Visuals
    # -------------------------------------------------------------
    with tab1:
        # Chart 1: Cumulative Equity Progression
        col_hdr1, col_info1 = st.columns([6, 1])
        with col_hdr1:
            st.markdown("### 📈 Cumulative Wealth Growth: Strategy vs Nifty 50")
        with col_info1:
            render_info_popover(
                title="Cumulative Wealth Progression",
                summary="Tracks the geometric growth of portfolio equity starting from ₹10 Lakhs (or normalized 1.00) net of Indian friction charges compared to the Nifty 50 (^NSEI) index.",
                formula="Equity_t = Initial_Capital * prod(1 + R_net,s)",
                interpretation="A higher trajectory indicates outperformance. When the Strategy line diverges upwards from the Nifty 50 line, positive alpha is being generated.",
            )

        col_scale, _ = st.columns([2, 5])
        with col_scale:
            use_log = st.checkbox("Logarithmic Equity Scale (Log Y-Axis)", value=False, help="View percentage growth equally at all wealth levels without exponential distortion.")

        fig_equity = create_equity_curve(hist, use_log_scale=use_log, normalize=True)
        st.plotly_chart(fig_equity, use_container_width=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Chart 2: Underwater Drawdown
        col_hdr2, col_info2 = st.columns([6, 1])
        with col_hdr2:
            st.markdown("### 🌊 Underwater Drawdown Comparison (% Drop from Peak)")
        with col_info2:
            render_info_popover(
                title="Underwater Drawdown",
                summary="Quantifies the percentage drop from the highest historical portfolio value at any point in time.",
                formula="Drawdown_t = (Value_t - Peak_t) / Peak_t",
                interpretation="Shows psychological pain, downside risk, and recovery time. A strategy that stays close to 0% and rebounds quickly from dips possesses resilient risk management.",
            )

        fig_dd = create_drawdown_plot(hist)
        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Chart 3: Rolling Sharpe Ratio
        col_hdr3, col_info3 = st.columns([6, 1])
        with col_hdr3:
            st.markdown("### ⚖️ Rolling Risk-Adjusted Return (Annualized Sharpe)")
        with col_info3:
            render_info_popover(
                title="Rolling Sharpe Ratio",
                summary="Computes annualized Sharpe ratio over moving historical observation windows to assess consistency across changing economic cycles.",
                formula="Rolling_Sharpe = [Mean(R - Rf) / Std(R)] * sqrt(252)",
                interpretation="Periods above the red dotted zero line indicate superior returns relative to risk-free Indian interest rates (6.5%). Values > 1.0 represent excellent risk efficiency.",
            )

        col_win, _ = st.columns([2, 5])
        with col_win:
            roll_win = st.select_slider(
                "Rolling Observation Window",
                options=[63, 126, 252],
                value=126,
                format_func=lambda w: f"{w} Trading Days (~{w//21} Months)",
                help="Adjust the lookback length for rolling Sharpe calculation (e.g. 126 days = 6 months).",
            )
        fig_sharpe = create_rolling_sharpe_chart(hist, window=roll_win, rf_annual=config.rf_annual)
        st.plotly_chart(fig_sharpe, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: Metrics Deep Dive & Monthly Heatmap
    # -------------------------------------------------------------
    with tab2:
        col_mhdr, col_minfo = st.columns([6, 1])
        with col_mhdr:
            st.markdown("### 📊 Comprehensive Quantitative Metric Breakdown")
        with col_minfo:
            render_info_popover(
                title="Quantitative Performance Table",
                summary="Provides an audit-grade comparison between Strategy net results and the Nifty 50 benchmark across standard quantitative risk and return ratios.",
                formula="CAGR, Volatility, Sharpe, Sortino, Calmar, MDD, Alpha, Beta, Turnover, Hit Rate",
                interpretation="Positive Delta values indicate outperformance on return and risk-adjusted metrics, while lower values on Drawdown and Volatility indicate defensive stability.",
            )

        render_performance_table_component(current_result)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        col_hthdr, col_htinfo = st.columns([6, 1])
        with col_hthdr:
            st.markdown("### 🗓️ Compounding Monthly Returns Matrix (Year × Month)")
        with col_htinfo:
            render_info_popover(
                title="Monthly Returns Matrix",
                summary="Aggregates daily returns into geometric monthly compounding performance, plus Year-to-Date (YTD) full-year return.",
                formula="R_month = prod(1 + R_daily) - 1.0",
                interpretation="Dark green cells highlight strong monthly alpha (> 5%), while red cells identify market drawdown months. Useful for identifying seasonal trends in Indian equities.",
            )

        render_monthly_matrix_heatmap(hist["net_return"])

        # Quantitative Bias Defense Note
        st.markdown(
            f"""
            <div class="disclaimer-box">
                <b>🛡️ Survivorship Bias & Data Feed Disclosure:</b><br>
                Historical price feeds from Yahoo Finance utilize current Nifty 50 members. Companies delisted or liquidated 
                prior to 2025 are omitted. To counter look-ahead bias, this framework strictly enforces 
                <code>weights.shift(1)</code> execution lag. In institutional deployment, point-in-time constituent datasets 
                from official NSE archives should be integrated.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # TAB 3: Dynamic Asset Allocation & Risk Limits
    # -------------------------------------------------------------
    with tab3:
        col_ahdr, col_ainfo = st.columns([6, 1])
        with col_ahdr:
            st.markdown("### 💼 Dynamic Portfolio Composition & Single-Stock Limits")
        with col_ainfo:
            render_info_popover(
                title="Portfolio Allocation Stack",
                summary="Displays historical target weight allocation across active equities and cash reserves over time.",
                formula="Single-Stock Cap <= 15% | Cash Reserve >= 5%",
                interpretation="Visualizes how the portfolio reallocates capital between different sectors and names. Unallocated white space represents cash buffer for risk management.",
            )

        fig_alloc = create_allocation_area_chart(current_result.executed_weights)
        st.plotly_chart(fig_alloc, use_container_width=True)

        # Diagnostics row
        active_counts = (current_result.executed_weights.abs() > 1e-4).sum(axis=1)
        gross_exposures = current_result.executed_weights.abs().sum(axis=1) * 100.0

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            st.metric(
                "Avg Active Stocks",
                f"{active_counts.mean():.1f} equities",
                help="Average number of non-zero stock holdings held in portfolio daily.",
            )
        with dcol2:
            st.metric(
                "Avg Equity Exposure",
                f"{gross_exposures.mean():.1f}%",
                help="Average percentage of total portfolio capital invested in equities.",
            )
        with dcol3:
            st.metric(
                "Avg Cash Buffer",
                f"{100.0 - gross_exposures.mean():.1f}%",
                help="Average uninvested liquidity maintained to cushion drawdowns.",
            )
        with dcol4:
            st.metric(
                "Max Single-Stock Cap",
                f"{config.max_weight * 100.0:.0f}%",
                help="Hard cap preventing single-name concentration.",
            )

    # -------------------------------------------------------------
    # TAB 4: Institutional Export Center
    # -------------------------------------------------------------
    with tab4:
        col_ehdr, col_einfo = st.columns([6, 1])
        with col_ehdr:
            st.markdown("### 📥 Automated Institutional Report Generation")
        with col_einfo:
            render_info_popover(
                title="Institutional Reports & Export Assets",
                summary="Generates presentation-ready artifacts directly in your browser without requiring external software.",
                interpretation="Download the 3-tab formatted Excel report with embedded native charts for client presentations, or export raw CSV logs for custom quantitative analysis.",
            )

        st.caption("Export executive Excel workbooks, high-resolution 300 DPI publication plots, and complete daily execution timelines.")

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        excel_out_path = reports_dir / "dashboard_export.xlsx"
        chart_out_path = reports_dir / "dashboard_performance_chart.png"

        # Generate export artifacts only if not already generated for current result
        if (
            "cached_export_name" not in st.session_state
            or st.session_state.get("cached_export_name") != current_result.strategy_name
            or run_clicked
        ):
            generate_excel_report(current_result, output_path=excel_out_path)
            plot_performance(current_result, output_path=chart_out_path)
            st.session_state["cached_export_name"] = current_result.strategy_name

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            st.markdown("#### 📑 Executive Excel Workbook")
            st.caption("3-tab openpyxl workbook: Executive Summary, Monthly Heatmap, and native Excel LineCharts.")
            if excel_out_path.exists():
                with open(excel_out_path, "rb") as f:
                    excel_bytes = f.read()
                st.download_button(
                    label="📥 Download Excel (.xlsx)",
                    data=excel_bytes,
                    file_name=f"nse_backtest_{current_result.strategy_name.lower().replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        with col_exp2:
            st.markdown("#### 📊 Publication Plot (300 DPI)")
            st.caption("High-resolution 2-panel chart showing Cumulative Equity Progression and Underwater Drawdown.")
            if chart_out_path.exists():
                with open(chart_out_path, "rb") as f:
                    png_bytes = f.read()
                st.download_button(
                    label="📥 Download Chart (.png)",
                    data=png_bytes,
                    file_name="nse_performance_plot.png",
                    mime="image/png",
                    use_container_width=True,
                )

        with col_exp3:
            st.markdown("#### 📄 Daily Execution Log (CSV)")
            st.caption("Complete time-series containing daily gross return, turnover, friction costs, net return, and drawdowns.")
            csv_data = hist.to_csv().encode("utf-8")
            st.download_button(
                label="📥 Download Data (.csv)",
                data=csv_data,
                file_name="portfolio_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("<hr style='margin: 22px 0;'>", unsafe_allow_html=True)
        st.markdown("##### Preview of Generated Publication Chart (300 DPI Matplotlib)")
        if chart_out_path.exists():
            st.image(str(chart_out_path), caption="Automated Matplotlib Export (300 DPI)", use_container_width=True)

else:
    st.info("Click **Run Simulation** in the sidebar to initiate vectorised backtesting.")
