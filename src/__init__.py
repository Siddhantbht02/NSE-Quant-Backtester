"""nse-backtester: Production-Grade Quantitative Backtesting Framework for NSE (Indian Equities).

A modular, mathematically rigorous, vectorised backtesting platform designed for historical
testing of quantitative equity strategies on the National Stock Exchange of India.
"""

from __future__ import annotations

import mimetypes

# Workaround for Windows sandboxed/restricted registry environments where mimetypes.init()
# fails with PermissionError trying to read HKEY_CLASSES_ROOT
_orig_read = getattr(mimetypes.MimeTypes, "read_windows_registry", None)
if _orig_read is not None:
    def _safe_read_windows_registry(self: mimetypes.MimeTypes, *args: object, **kwargs: object) -> None:
        try:
            _orig_read(self, *args, **kwargs)
        except (PermissionError, OSError):
            pass

    mimetypes.MimeTypes.read_windows_registry = _safe_read_windows_registry  # type: ignore[assignment]

from src.data_loader import DataLoader
from src.strategy import Strategy, MomentumStrategy, MeanReversionStrategy
from src.portfolio import PortfolioManager, apply_portfolio_constraints
from src.backtester import Backtester, BacktestResult
from src.metrics import (
    calculate_cagr,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_alpha_beta,
    calculate_turnover,
    calculate_hit_rate,
    calculate_all_metrics,
    generate_performance_table,
)
from src.report import generate_excel_report, plot_performance
from src.llm_commentary import generate_llm_commentary, build_quant_prompt

__all__ = [
    "DataLoader",
    "Strategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "PortfolioManager",
    "apply_portfolio_constraints",
    "Backtester",
    "BacktestResult",
    "calculate_cagr",
    "calculate_annualized_volatility",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    "calculate_alpha_beta",
    "calculate_turnover",
    "calculate_hit_rate",
    "calculate_all_metrics",
    "generate_performance_table",
    "generate_excel_report",
    "plot_performance",
    "generate_llm_commentary",
    "build_quant_prompt",
]
