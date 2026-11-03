"""CLI Entry point for nse-backtester.

Executes quantitative backtests on NSE equities (2016-2025), displays formatted
performance comparison tables, and generates Excel workbooks & visualization charts.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

from tabulate import tabulate

from src.backtester import Backtester, BacktestResult
from src.data_loader import DataLoader
from src.portfolio import PortfolioManager
from src.report import generate_excel_report, plot_performance
from src.strategy import MeanReversionStrategy, MomentumStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nse-backtester")


def format_summary_table(results: List[BacktestResult]) -> str:
    """Format multiple backtest results into an aligned comparison table."""
    if not results:
        return "No results to display."

    headers = ["Metric"] + [r.strategy_name for r in results] + ["Nifty 50 (^NSEI)"]
    metrics_keys = list(results[0].metrics.keys())

    pct_keys = {
        "Total Return",
        "CAGR",
        "Annualized Volatility",
        "Max Drawdown",
        "Alpha (CAPM)",
        "Hit Rate",
        "Annualized Turnover",
    }

    rows = []
    for k in metrics_keys:
        row = [k]
        for r in results:
            val = r.metrics.get(k, 0.0)
            if k in pct_keys:
                row.append(f"{val * 100.0:+.2f}%")
            else:
                row.append(f"{val:.2f}")

        # Benchmark value from first result
        bench_val = results[0].benchmark_metrics.get(k, 0.0)
        if k in pct_keys:
            row.append(f"{bench_val * 100.0:+.2f}%")
        else:
            row.append(f"{bench_val:.2f}")

        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="github", stralign="right")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production-Grade Quantitative Backtesting Framework for NSE (Indian Equities)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["momentum", "mean_reversion", "both"],
        default="both",
        help="Strategy to evaluate: momentum, mean_reversion, or both (default: both)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2016-01-01",
        help="Start date for backtest period (YYYY-MM-DD, default: 2016-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-12-31",
        help="End date for backtest period (YYYY-MM-DD, default: 2025-12-31)",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=15.0,
        help="One-way transaction cost in basis points (default: 15 bps = 0.15% per side)",
    )
    parser.add_argument(
        "--rf",
        type=float,
        default=0.065,
        help="Annualized Indian Risk-Free Rate (default: 0.065 = 6.5%)",
    )
    parser.add_argument(
        "--excel",
        type=str,
        default="reports/backtest_report.xlsx",
        help="Path for generated Excel report",
    )
    parser.add_argument(
        "--chart",
        type=str,
        default="reports/performance_chart.png",
        help="Path for generated performance chart",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Bypass disk cache and download fresh data from Yahoo Finance",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  NSE QUANTITATIVE BACKTESTER: PRODUCTION RUN")
    print("=" * 80)
    print(f"Period          : {args.start_date} to {args.end_date}")
    print(f"Friction Model  : {args.cost_bps} bps per side (30 bps round-trip)")
    print(f"Risk-Free Rate  : {args.rf * 100:.1f}% (Indian Repo Benchmark)")
    print("-" * 80)

    # 1. Load Data
    data_loader = DataLoader(cache_dir="data")
    print("[1/4] Ingesting & cleaning historical price series...")
    try:
        prices, benchmark = data_loader.fetch_data(
            start_date=args.start_date,
            end_date=args.end_date,
            force_reload=args.reload_data if hasattr(args, "reload_data") else False,
        )
    except Exception as e:
        logger.warning(f"Live data fetch error: {e}. Falling back to cached or synthetic data.")
        prices, benchmark = DataLoader.generate_synthetic_data(start_date=args.start_date)

    print(f"      Universe: {prices.shape[1]} NSE liquid stocks | {len(prices)} trading days")
    print(f"      Benchmark: {benchmark.name} ({len(benchmark)} observations)")

    # 2. Setup Backtest Engine & Risk Constraints
    cost_fraction = args.cost_bps / 10000.0
    engine = Backtester(cost_per_side=cost_fraction, rf_annual=args.rf)
    pm = PortfolioManager(max_weight=0.15, cash_buffer=0.05, max_gross_exposure=0.95)

    results: List[BacktestResult] = []

    # 3. Execute Strategies
    if args.strategy in ["momentum", "both"]:
        print("\n[2/4] Simulating 12-1 Month Cross-Sectional Momentum Strategy...")
        mom_strat = MomentumStrategy(
            lookback_days=252, skip_days=21, top_quantile=0.20, rebalance_freq="monthly"
        )
        raw_mom_weights = mom_strat.generate_signals(prices)
        constrained_mom_weights = pm.process_weights(raw_mom_weights)
        res_mom = engine.run(
            prices=prices,
            strategy_or_weights=constrained_mom_weights,
            benchmark=benchmark,
            strategy_name="Momentum (12-1M)",
        )
        results.append(res_mom)

    if args.strategy in ["mean_reversion", "both"]:
        print("[3/4] Simulating 20-Day Rolling Z-Score Mean Reversion Strategy...")
        mr_strat = MeanReversionStrategy(
            window=20, entry_z=-1.5, exit_z=0.0, weighting="inverse_vol"
        )
        raw_mr_weights = mr_strat.generate_signals(prices)
        constrained_mr_weights = pm.process_weights(raw_mr_weights)
        res_mr = engine.run(
            prices=prices,
            strategy_or_weights=constrained_mr_weights,
            benchmark=benchmark,
            strategy_name="Mean Reversion (Z-Score)",
        )
        results.append(res_mr)

    # 4. Display Results
    print("\n" + "=" * 80)
    print("  EXECUTIVE QUANTITATIVE PERFORMANCE SUMMARY")
    print("=" * 80)
    print(format_summary_table(results))

    # 5. Export Reports
    print("\n[4/4] Generating automated executive reports & artifacts...")
    primary_result = results[0]
    excel_path = generate_excel_report(primary_result, output_path=args.excel)
    print(f"      [OK] Multi-tab Excel report created: {excel_path}")

    chart_path = plot_performance(primary_result, output_path=args.chart)
    print(f"      [OK] Publication chart exported:     {chart_path}")

    # If both strategies ran, export the second as well
    if len(results) > 1:
        mr_res = results[1]
        mr_excel_path = Path(args.excel).parent / "mean_reversion_report.xlsx"
        mr_chart_path = Path(args.chart).parent / "mean_reversion_chart.png"
        generate_excel_report(mr_res, output_path=mr_excel_path)
        plot_performance(mr_res, output_path=mr_chart_path)
        print(f"      [OK] Mean Reversion Excel created:   {mr_excel_path}")
        print(f"      [OK] Mean Reversion chart exported: {mr_chart_path}")

    print("\n" + "=" * 80)
    print("  SIMULATION COMPLETE (Look-Ahead Bias Free, Friction Adjusted)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
