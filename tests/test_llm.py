"""Tests for LLM Quantitative Performance Commentary engine."""

from __future__ import annotations

from pathlib import Path
import pytest
import openpyxl

from src.backtester import Backtester, BacktestResult
from src.llm_commentary import (
    build_quant_prompt,
    generate_fallback_commentary,
    generate_llm_commentary,
)
from src.report import generate_excel_report
from src.strategy import MomentumStrategy


@pytest.fixture
def sample_result(synthetic_prices, benchmark_series) -> BacktestResult:
    """Fixture providing a simulated BacktestResult."""
    strat = MomentumStrategy(lookback_days=40, skip_days=5, top_quantile=0.3)
    weights = strat.generate_signals(synthetic_prices)
    engine = Backtester(cost_per_side=0.0015, rf_annual=0.065, initial_capital=1_000_000.0)
    return engine.run(
        prices=synthetic_prices,
        strategy_or_weights=weights,
        benchmark=benchmark_series,
        strategy_name="Momentum (Test)",
    )


def test_build_quant_prompt(sample_result: BacktestResult):
    """Verify prompt formatting includes all institutional risk and return inputs."""
    prompt = build_quant_prompt(sample_result)
    assert "Momentum (Test)" in prompt
    assert "Nifty 50 (^NSEI)" in prompt
    assert "Execution Lag: t+1 day" in prompt
    assert "15.0 bps per side" in prompt
    assert "CAGR" in prompt
    assert "Sharpe Ratio" in prompt
    assert "Max Drawdown" in prompt


def test_generate_fallback_commentary(sample_result: BacktestResult):
    """Verify fallback commentary produces multi-section institutional teardown."""
    commentary = generate_fallback_commentary(sample_result)
    assert "Quantitative Performance Teardown" in commentary
    assert "1. Executive Alpha Assessment" in commentary
    assert "2. Risk-Adjusted Efficiency & Asymmetry" in commentary
    assert "3. Drawdown Duration & Tail Risk Profile" in commentary
    assert "4. Execution Scalability & Strategic Recommendations" in commentary
    assert "Sharpe Ratio" in commentary


def test_generate_llm_commentary_fallback_mode(sample_result: BacktestResult):
    """Verify generate_llm_commentary with explicit provider='fallback'."""
    commentary = generate_llm_commentary(sample_result, provider="fallback")
    assert isinstance(commentary, str)
    assert len(commentary) > 200
    assert "CAGR" in commentary or "alpha" in commentary.lower()


def test_generate_llm_commentary_auto_without_keys(sample_result: BacktestResult, monkeypatch):
    """Verify that when no environment keys exist, it gracefully defaults to fallback."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    commentary = generate_llm_commentary(sample_result, provider="auto")
    assert isinstance(commentary, str)
    assert len(commentary) > 200
    assert "Quantitative Performance Teardown" in commentary


def test_excel_report_with_commentary(sample_result: BacktestResult, tmp_path: Path):
    """Verify that generate_excel_report embeds LLM commentary into the workbook."""
    out_file = tmp_path / "test_report_with_llm.xlsx"
    custom_commentary = "### 🤖 Test Commentary\nThis is an institutional strategy teardown."

    generate_excel_report(sample_result, output_path=out_file, commentary=custom_commentary)
    assert out_file.exists()

    wb = openpyxl.load_workbook(out_file, data_only=True)
    ws = wb["Executive Summary"]
    
    # Check that commentary header exists
    found_comm_header = False
    for row in range(1, 30):
        val = ws.cell(row=row, column=2).value
        if val and "LLM QUANTITATIVE PERFORMANCE COMMENTARY" in str(val):
            found_comm_header = True
            break
    assert found_comm_header, "Commentary section header was not found in Executive Summary sheet"
