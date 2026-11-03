"""Automated Reporting module for nse-backtester.

Generates:
1. Multi-tab styled Excel Workbooks (openpyxl):
   - Sheet 1: Executive Summary with styled metrics table
   - Sheet 2: Monthly Returns Matrix (Year x Month heatmap table)
   - Sheet 3: Daily Curves with embedded native Excel LineCharts
2. High-resolution 2-panel Matplotlib charts (Cumulative Equity & Underwater Drawdown).
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Optional, Union

# Handle Windows sandbox registry restriction for mimetypes
_orig_read = getattr(mimetypes.MimeTypes, "read_windows_registry", None)
if _orig_read is not None:
    def _safe_read_windows_registry(self: mimetypes.MimeTypes, *args: object, **kwargs: object) -> None:
        try:
            _orig_read(self, *args, **kwargs)
        except (PermissionError, OSError):
            pass

    mimetypes.MimeTypes.read_windows_registry = _safe_read_windows_registry  # type: ignore[assignment]

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

from src.backtester import BacktestResult

logger = logging.getLogger(__name__)


def compute_monthly_returns_matrix(daily_returns: pd.Series) -> pd.DataFrame:
    """Compute Year x Month matrix of compounding monthly returns and YTD returns.

    Args:
        daily_returns: Daily returns series indexed by DatetimeIndex.

    Returns:
        DataFrame with Year as index, Jan-Dec + YTD as columns.
    """
    clean_r = daily_returns.dropna()
    if clean_r.empty:
        return pd.DataFrame()

    df = clean_r.to_frame(name="ret")
    df["Year"] = df.index.year
    df["Month"] = df.index.month

    # Compound monthly return: prod(1 + r) - 1
    monthly = df.groupby(["Year", "Month"])["ret"].apply(lambda s: np.prod(1.0 + s.to_numpy()) - 1.0).unstack(level="Month")

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    monthly = monthly.rename(columns=month_names)

    # Ensure all 12 columns exist
    for m in month_names.values():
        if m not in monthly.columns:
            monthly[m] = np.nan
    monthly = monthly[list(month_names.values())]

    # Compute YTD return per year
    ytd = df.groupby("Year")["ret"].apply(lambda s: np.prod(1.0 + s.to_numpy()) - 1.0)
    monthly["YTD"] = ytd

    return monthly


def generate_excel_report(
    result: BacktestResult,
    output_path: Union[str, Path] = "reports/backtest_report.xlsx",
) -> Path:
    """Generate professional, multi-tab styled Excel report using openpyxl.

    Tabs created:
    1. Executive Summary: Formatted table of performance metrics (Strategy vs Benchmark)
    2. Monthly Returns Matrix: Compounding monthly returns table with conditional heatmap
    3. Daily Curves: Daily time-series with embedded native Excel LineChart

    Args:
        result: BacktestResult object from simulation.
        output_path: Target path for the workbook.

    Returns:
        Path to the saved Excel workbook.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Common Styles
    navy_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    subhead_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    white_bold = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    navy_bold = Font(name="Calibri", size=11, bold=True, color="1F4E78")
    title_font = Font(name="Calibri", size=14, bold=True, color="1F4E78")
    regular_font = Font(name="Calibri", size=11)
    bold_font = Font(name="Calibri", size=11, bold=True)

    thin_border_side = Side(border_style="thin", color="D9D9D9")
    thin_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
    thick_bottom = Border(bottom=Side(border_style="medium", color="1F4E78"))

    # -------------------------------------------------------------
    # SHEET 1: Executive Summary
    # -------------------------------------------------------------
    ws_exec = wb.create_sheet(title="Executive Summary")
    ws_exec.views.sheetView[0].showGridLines = True

    # Title Banner
    ws_exec.cell(row=2, column=2, value="NSE QUANTITATIVE BACKTEST EXECUTIVE REPORT").font = title_font
    ws_exec.cell(row=3, column=2, value=f"Strategy: {result.strategy_name} | Benchmark: Nifty 50 (^NSEI)").font = navy_bold

    headers = ["Metric", f"Strategy ({result.strategy_name})", "Benchmark (^NSEI)", "Delta (Strategy - Bench)"]
    start_row = 5
    for col_idx, h in enumerate(headers, start=2):
        cell = ws_exec.cell(row=start_row, column=col_idx, value=h)
        cell.fill = navy_fill
        cell.font = white_bold
        cell.alignment = Alignment(horizontal="center" if col_idx > 2 else "left", vertical="center")
    ws_exec.row_dimensions[start_row].height = 24

    percentage_metrics = {
        "Total Return", "CAGR", "Annualized Volatility", "Max Drawdown",
        "Alpha (CAPM)", "Hit Rate", "Annualized Turnover"
    }

    summary_df = result.summary()
    curr_row = start_row + 1

    for metric_name, row_data in summary_df.iterrows():
        strat_v = row_data["Strategy"]
        bench_v = row_data["Benchmark (^NSEI)"]
        delta_v = row_data["Delta (Strategy - Benchmark)"]

        c_name = ws_exec.cell(row=curr_row, column=2, value=str(metric_name))
        c_strat = ws_exec.cell(row=curr_row, column=3, value=float(strat_v) if not np.isnan(strat_v) else "-")
        c_bench = ws_exec.cell(row=curr_row, column=4, value=float(bench_v) if not np.isnan(bench_v) else "-")
        c_delta = ws_exec.cell(row=curr_row, column=5, value=float(delta_v) if not np.isnan(delta_v) else "-")

        c_name.font = regular_font
        c_strat.font = bold_font
        c_bench.font = regular_font
        c_delta.font = bold_font

        # Formatting
        fmt = "0.00%" if metric_name in percentage_metrics else "0.00"
        if isinstance(c_strat.value, float):
            c_strat.number_format = fmt
        if isinstance(c_bench.value, float):
            c_bench.number_format = fmt
        if isinstance(c_delta.value, float):
            c_delta.number_format = fmt
            if metric_name in ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Alpha (CAPM)"]:
                if c_delta.value > 0:
                    c_delta.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
                elif c_delta.value < 0:
                    c_delta.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

        for col_idx in range(2, 6):
            ws_exec.cell(row=curr_row, column=col_idx).border = thin_border
            if col_idx > 2:
                ws_exec.cell(row=curr_row, column=col_idx).alignment = Alignment(horizontal="right")

        curr_row += 1

    # Auto-adjust column widths
    ws_exec.column_dimensions["B"].width = 28
    ws_exec.column_dimensions["C"].width = 26
    ws_exec.column_dimensions["D"].width = 24
    ws_exec.column_dimensions["E"].width = 26

    # -------------------------------------------------------------
    # SHEET 2: Monthly Returns Matrix
    # -------------------------------------------------------------
    ws_month = wb.create_sheet(title="Monthly Returns Matrix")
    ws_month.views.sheetView[0].showGridLines = True

    ws_month.cell(row=2, column=2, value="STRATEGY MONTHLY COMPOUND RETURNS").font = title_font

    monthly_matrix = compute_monthly_returns_matrix(result.portfolio_history["net_return"])
    if not monthly_matrix.empty:
        col_headers = ["Year"] + list(monthly_matrix.columns)
        m_start_row = 4
        for col_idx, ch in enumerate(col_headers, start=2):
            cell = ws_month.cell(row=m_start_row, column=col_idx, value=ch)
            cell.fill = navy_fill
            cell.font = white_bold
            cell.alignment = Alignment(horizontal="center")
        ws_month.row_dimensions[m_start_row].height = 22

        m_row = m_start_row + 1
        pos_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")  # soft green
        neg_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")  # soft red

        for year, row in monthly_matrix.iterrows():
            c_yr = ws_month.cell(row=m_row, column=2, value=int(year))
            c_yr.font = bold_font
            c_yr.alignment = Alignment(horizontal="center")
            c_yr.border = thin_border

            for col_idx, val in enumerate(row, start=3):
                cell = ws_month.cell(row=m_row, column=col_idx)
                if not np.isnan(val):
                    cell.value = float(val)
                    cell.number_format = "0.00%"
                    cell.font = regular_font
                    cell.alignment = Alignment(horizontal="right")
                    if val > 0:
                        cell.fill = pos_fill
                    elif val < 0:
                        cell.fill = neg_fill
                else:
                    cell.value = "-"
                    cell.alignment = Alignment(horizontal="center")
                cell.border = thin_border

            m_row += 1

        for col_letter in [get_column_letter(c) for c in range(2, len(col_headers) + 2)]:
            ws_month.column_dimensions[col_letter].width = 11

    # -------------------------------------------------------------
    # SHEET 3: Daily Curves & Native Excel LineChart
    # -------------------------------------------------------------
    ws_daily = wb.create_sheet(title="Daily Curves")
    ws_daily.views.sheetView[0].showGridLines = True

    daily_headers = [
        "Date", "Strategy Daily Return", "Benchmark Daily Return",
        "Strategy Equity", "Benchmark Equity", "Strategy Drawdown", "Benchmark Drawdown"
    ]
    for col_idx, dh in enumerate(daily_headers, start=1):
        cell = ws_daily.cell(row=1, column=col_idx, value=dh)
        cell.fill = navy_fill
        cell.font = white_bold
        cell.alignment = Alignment(horizontal="center")
    ws_daily.row_dimensions[1].height = 22

    hist = result.portfolio_history
    n_records = len(hist)

    for idx, (dt, row) in enumerate(hist.iterrows(), start=2):
        c_dt = ws_daily.cell(row=idx, column=1, value=dt.strftime("%Y-%m-%d"))
        c_sr = ws_daily.cell(row=idx, column=2, value=float(row["net_return"]))
        c_br = ws_daily.cell(row=idx, column=3, value=float(row["benchmark_return"]))
        c_se = ws_daily.cell(row=idx, column=4, value=float(row["equity_curve"]))
        c_be = ws_daily.cell(row=idx, column=5, value=float(row["benchmark_equity_curve"]))
        c_sd = ws_daily.cell(row=idx, column=6, value=float(row["drawdown"]))
        c_bd = ws_daily.cell(row=idx, column=7, value=float(row["benchmark_drawdown"]))

        c_sr.number_format = "0.0000%"
        c_br.number_format = "0.0000%"
        c_se.number_format = "#,##0.00"
        c_be.number_format = "#,##0.00"
        c_sd.number_format = "0.00%"
        c_bd.number_format = "0.00%"

    ws_daily.column_dimensions["A"].width = 13
    ws_daily.column_dimensions["B"].width = 20
    ws_daily.column_dimensions["C"].width = 22
    ws_daily.column_dimensions["D"].width = 18
    ws_daily.column_dimensions["E"].width = 18
    ws_daily.column_dimensions["F"].width = 18
    ws_daily.column_dimensions["G"].width = 18

    # Add Native Openpyxl LineChart for Equity Progression
    if n_records > 1:
        chart_eq = LineChart()
        chart_eq.title = f"Equity Progression: {result.strategy_name} vs Benchmark"
        chart_eq.style = 13
        chart_eq.y_axis.title = "Portfolio Value (INR)"
        chart_eq.x_axis.title = "Date"
        chart_eq.width = 18
        chart_eq.height = 12

        # Data from columns 4 (Strategy Equity) and 5 (Benchmark Equity)
        data_ref = Reference(ws_daily, min_col=4, min_row=1, max_col=5, max_row=min(n_records + 1, 2500))
        cats_ref = Reference(ws_daily, min_col=1, min_row=2, max_row=min(n_records + 1, 2500))

        chart_eq.add_data(data_ref, titles_from_data=True)
        chart_eq.set_categories(cats_ref)
        ws_daily.add_chart(chart_eq, "I2")

        # Second Chart for Underwater Drawdown
        chart_dd = LineChart()
        chart_dd.title = "Underwater Drawdown Comparison"
        chart_dd.style = 10
        chart_dd.y_axis.title = "Drawdown"
        chart_dd.x_axis.title = "Date"
        chart_dd.width = 18
        chart_dd.height = 10

        dd_data_ref = Reference(ws_daily, min_col=6, min_row=1, max_col=7, max_row=min(n_records + 1, 2500))
        chart_dd.add_data(dd_data_ref, titles_from_data=True)
        chart_dd.set_categories(cats_ref)
        ws_daily.add_chart(chart_dd, "I24")

    wb.save(out_file)
    logger.info(f"Excel report successfully generated: {out_file}")
    return out_file


def plot_performance(
    result: BacktestResult,
    output_path: Union[str, Path] = "reports/performance_chart.png",
    dpi: int = 300,
) -> Path:
    """Generate high-resolution 2-panel publication-grade performance plot.

    Top panel: Cumulative Equity Progression (Strategy vs Benchmark ^NSEI)
    Bottom panel: Underwater Drawdown Comparison (% drop from peak)

    Args:
        result: BacktestResult object.
        output_path: Path to save the PNG/PDF chart.
        dpi: Resolution (default 300 dpi).

    Returns:
        Path to the saved chart image.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    hist = result.portfolio_history
    dates = hist.index

    # Setup 2-panel figure with custom styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.0]}
    )

    # Panel 1: Equity Curve
    strat_equity = hist["equity_curve"] / hist["equity_curve"].iloc[0]
    bench_equity = hist["benchmark_equity_curve"] / hist["benchmark_equity_curve"].iloc[0]

    ax1.plot(dates, strat_equity, label=f"Strategy: {result.strategy_name}", color="#1f77b4", linewidth=2.0)
    ax1.plot(dates, bench_equity, label="Benchmark: Nifty 50 (^NSEI)", color="#7f7f7f", linewidth=1.6, linestyle="--")

    ax1.set_title(
        f"Quantitative Backtest Performance: {result.strategy_name} vs Nifty 50\n"
        f"CAGR: {result.metrics.get('CAGR', 0)*100:.1f}% | Sharpe: {result.metrics.get('Sharpe Ratio', 0):.2f} | "
        f"MDD: {result.metrics.get('Max Drawdown', 0)*100:.1f}% | Alpha: {result.metrics.get('Alpha (CAPM)', 0)*100:.1f}%",
        fontsize=13, fontweight="bold", pad=12
    )
    ax1.set_ylabel("Normalized Wealth (Base = 1.0)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Underwater Drawdown
    strat_dd = hist["drawdown"] * 100.0
    bench_dd = hist["benchmark_drawdown"] * 100.0

    ax2.plot(dates, strat_dd, label=f"{result.strategy_name} Drawdown", color="#d62728", linewidth=1.4)
    ax2.fill_between(dates, strat_dd, 0, color="#d62728", alpha=0.25)
    ax2.plot(dates, bench_dd, label="Nifty 50 Drawdown", color="#7f7f7f", linewidth=1.2, linestyle=":")

    ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.set_ylim(min(strat_dd.min(), bench_dd.min()) * 1.15, 2.0)
    ax2.legend(loc="lower left", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Format x-axis dates
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Performance chart exported to {out_file}")
    return out_file

