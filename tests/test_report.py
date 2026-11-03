"""Tests for Automated Reporting module."""

from pathlib import Path
import openpyxl
import pandas as pd
import pytest

from src.backtester import Backtester
from src.data_loader import DataLoader
from src.strategy import MomentumStrategy
from src.report import generate_excel_report, plot_performance, compute_monthly_returns_matrix


def test_monthly_returns_matrix():
    """Verify computation of monthly returns matrix."""
    dates = pd.bdate_range("2021-01-01", periods=100)
    returns = pd.Series(0.001, index=dates)
    matrix = compute_monthly_returns_matrix(returns)
    assert not matrix.empty
    assert "Jan" in matrix.columns
    assert "YTD" in matrix.columns


def test_excel_and_plot_generation(tmp_path):
    """Verify Excel report and Matplotlib chart generation end-to-end."""
    prices, bench = DataLoader.generate_synthetic_data(n_days=300, start_date="2020-01-01")
    strat = MomentumStrategy(lookback_days=252, skip_days=21)
    engine = Backtester()
    result = engine.run(prices, strat, benchmark=bench)

    excel_path = tmp_path / "test_report.xlsx"
    chart_path = tmp_path / "test_chart.png"

    gen_excel = generate_excel_report(result, excel_path)
    gen_chart = plot_performance(result, chart_path)

    assert Path(gen_excel).exists()
    assert Path(gen_chart).exists()

    # Inspect Excel workbook sheets
    wb = openpyxl.load_workbook(gen_excel)
    sheet_names = wb.sheetnames
    assert "Executive Summary" in sheet_names
    assert "Monthly Returns Matrix" in sheet_names
    assert "Daily Curves" in sheet_names

