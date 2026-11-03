# `nse-backtester`: Production-Grade Quantitative Backtesting Framework for NSE Equities

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-35%20passed-brightgreen.svg)]()
[![AI Analysis](https://img.shields.io/badge/AI-Gemini%20%7C%20OpenAI-purple.svg)]()
[![Dashboard](https://img.shields.io/badge/UI-Streamlit%20%2B%20Plotly-red.svg)]()
[![Market](https://img.shields.io/badge/Market-NSE%20India%20(Equities)-orange.svg)]()

`nse-backtester` is a modular, mathematically rigorous, vectorised backtesting platform engineered for institutional-grade historical evaluation of quantitative equity strategies on the National Stock Exchange of India (NSE). 

Designed with strict Object-Oriented Programming (OOP) abstraction boundaries, complete type safety (`typing`), vectorised NumPy/pandas execution, and zero tolerance for look-ahead or data-snooping leakage, the platform models Indian equity market microstructure realities—including Securities Transaction Tax (STT), stamp duty, exchange turnover charges, GST, and slippage.

---

## 1. Quantitative Bias Defense Matrix

A core mandate of quantitative research is defending against statistical and structural biases that create phantom alpha on paper while generating losses in real execution:

| Bias | Institutional Description | How `nse-backtester` Defends Against It |
| :--- | :--- | :--- |
| **Look-Ahead Bias** | Using information at time $t$ that was only knowable at $t+k$ ($k \ge 1$) to calculate returns at time $t$. | **Forced Execution Lag**: All target signals generated at the close of day $t$ are strictly executed on day $t+1$ via `executed_weights = target_weights.shift(1)`. Signals generated on day $t$ earn $0.00\%$ on day $t$ and only earn returns starting day $t+1$. |
| **Survivorship Bias** | Filtering historical universes using current index constituents, ignoring past failures and delistings. | **Explicit Disclosures & Dynamic Inclusion**: Acknowledged as a structural limitation of free Yahoo Finance feeds. Assets listing mid-sample (e.g., late IPOs) preserve leading `NaN`s without artificial backfilling (`bfill`), dynamically preventing phantom zero-volatility periods. Documented integration blueprint for point-in-time constituent datasets. |
| **Transaction Drag** | Assuming frictionless fills at mid-market prices without modeling market costs. | **Calibrated Indian Friction Engine**: Realistic transaction friction calibrated to 15 bps per side (30 bps round-trip) modeling STT (0.10% delivery), stamp duty (0.015%), exchange/SEBI fees, GST (18%), and bid-ask slippage across all portfolio turnovers. |
| **Overfitting & Data Snooping** | Tuning arbitrary parameter combinations to past historical noise. | **Academic Foundations**: Implements standard, non-data-mined momentum specifications (12-1 month lookback, Jegadeesh & Titman 1993) and fixed 20-day Bollinger/SMA mean reversion windows without parameter dredging. |

---

## 2. Directory Architecture

```text
nse-backtester/
├── app.py                      # Main entry point for the Streamlit dashboard
├── data/                       # Local disk cache for market data (Parquet / CSV & metadata.json)
├── ui/
│   ├── __init__.py
│   ├── components.py           # Reusable KPI ribbons, styling helpers, and data wrappers
│   ├── plots.py                # Plotly figure builders (Equity, Drawdown, Allocations, Rolling Sharpe)
│   └── sidebar.py              # Sidebar control widgets and parameter validation
├── src/
│   ├── __init__.py             # Package exports & Windows sandbox environment patches
│   ├── data_loader.py          # Yahoo Finance loader, survivorship disclosures, robust cleaner
│   ├── strategy.py             # Abstract Strategy base, MomentumStrategy, MeanReversionStrategy
│   ├── portfolio.py            # Position sizing, single-stock caps (15%), cash buffers (5%)
│   ├── backtester.py           # Core vectorised simulation engine with execution lag & friction
│   ├── metrics.py              # Purely functional: CAGR, Sharpe, Sortino, MDD, Calmar, Alpha, Beta
│   ├── llm_commentary.py       # Gemini & OpenAI performance commentary with fallback engine
│   └── report.py               # Multi-tab styled openpyxl Excel workbooks with embedded AI commentary
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared synthetic price and benchmark fixtures
│   ├── test_data_loader.py     # Cache roundtrip, shape preservation, and leading NaN handling
│   ├── test_strategy.py        # Signal generation logic and weight constraint verification
│   ├── test_backtester.py      # Look-ahead prevention & 200% turnover cost arithmetic
│   ├── test_metrics.py         # Mathematical invariant checks (100% CAGR doubling, zero-vol)
│   ├── test_portfolio.py       # Single-stock capping and iterative redistribution
│   ├── test_report.py          # End-to-end Excel and visualization export validation
│   ├── test_ui.py              # Smoke tests for plot builders and state serialization
│   └── test_llm.py             # Prompt synthesis and multi-provider commentary validation
├── notebooks/
│   └── demo.ipynb              # Visual walkthrough comparing Momentum vs Mean Reversion vs Nifty 50
├── reports/                    # Generated Excel reports (.xlsx) and publication charts (.png)
├── pyproject.toml              # Modern package metadata and dependency specifications
├── requirements.txt            # Pinned dependencies (including streamlit & plotly)
├── main.py                     # CLI entry point to run backtests and generate reports
└── README.md                   # Architectural guide, mathematical formulations, and usage
```

---

## 3. Mathematical Foundations & Formulations

### 3.1 12-1 Month Cross-Sectional Momentum
Following the seminal work of Jegadeesh & Titman (1993), short-term (1-month) price changes exhibit microstructure reversal due to liquidity provision pressures. To capture pure intermediate-term momentum:
$$\text{Momentum Score}_{i, t} = \frac{P_{i, t - 21}}{P_{i, t - 252}} - 1.0$$
* **Universe Selection**: At each monthly rebalance date, active eligible assets are cross-sectionally ranked by momentum score.
* **Allocation**: The top quintile ($K = \lceil 0.20 \times N \rceil$) is selected with equal target weighting:
$$w_{i, t} = \frac{1}{K} \quad \forall i \in \text{Top Quintile}, \quad w_{j, t} = 0 \quad \text{otherwise}$$
* Weights are forward-filled between rebalance dates.

### 3.2 20-Day Rolling Z-Score Mean Reversion
Identifies statistical overextension relative to rolling equilibrium:
$$\mu_{20, i, t} = \frac{1}{20} \sum_{k=0}^{19} P_{i, t-k}, \quad \sigma_{20, i, t} = \sqrt{\frac{1}{19} \sum_{k=0}^{19} (P_{i, t-k} - \mu_{20, i, t})^2}$$
$$Z_{i, t} = \frac{P_{i, t} - \mu_{20, i, t}}{\sigma_{20, i, t} + \epsilon}$$
* **Entry Trigger**: Enter long position when $Z_{i, t} < -1.5$ (oversold dislocation).
* **Exit Trigger**: Liquidate position when $Z_{i, t} \ge 0.0$ (reversion to moving average).
* **Allocation**: Inverse-volatility weighted or equal-weighted across active signals.

### 3.3 Portfolio Constraints & Cash Buffering
Institutional portfolios enforce risk parameters before order dispatch:
* **Single-Stock Cap**: No asset may exceed $w_{\max} = 15\%$. Excess weight from capped positions is iteratively redistributed across uncapped active positions.
* **Cash Buffer**: A mandatory cash allocation $c \ge 5\%$ guarantees that gross equity exposure never exceeds $95\%$:
$$\sum_{i=1}^{N} |w_{i, t}| \le 0.95$$

### 3.4 Vectorised Execution & Friction Arithmetic
* **Execution Shift**:
$$W^{\text{exec}}_{t} = W_{t-1}$$
* **Daily Asset Return**:
$$R_{i, t} = \frac{P_{i, t}}{P_{i, t-1}} - 1.0$$
* **Daily Portfolio Turnover**:
$$\text{Turnover}_{t} = \sum_{i=1}^{N} \left| W^{\text{exec}}_{i, t} - W^{\text{exec}}_{i, t-1} \right|$$
* **Transaction Drag & Net Return**:
$$\text{Cost}_{t} = \text{Turnover}_{t} \times \text{cost\_per\_side}$$
$$R_{\text{net}, t} = \sum_{i=1}^{N} \left( W^{\text{exec}}_{i, t} \cdot R_{i, t} \right) - \text{Cost}_{t}$$

### 3.5 Quantitative Performance Metrics
* **CAGR** (Compound Annual Growth Rate):
$$\text{CAGR} = \left( \prod_{t=1}^{T} (1 + R_{\text{net}, t}) \right)^{\frac{252}{T}} - 1.0$$
* **Annualized Sharpe Ratio** (Indian Repo $R_f = 6.5\%$):
$$\text{Sharpe} = \frac{\bar{R}_{\text{daily}} - (R_f / 252)}{\sigma_{\text{daily}}} \times \sqrt{252}$$
* **Annualized Sortino Ratio** (Downside Semivariance):
$$D^{-}_{t} = \min\left(0, R_{\text{net}, t} - \frac{R_f}{252}\right), \quad \sigma_{\text{down}} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (D^{-}_{t})^2} \times \sqrt{252}$$
$$\text{Sortino} = \frac{\bar{R}_{\text{daily}} - (R_f / 252)}{\sigma_{\text{down}} / \sqrt{252}}$$
* **Maximum Drawdown (MDD) & Calmar Ratio**:
$$V_t = \prod_{s=1}^{t} (1 + R_{\text{net}, s}), \quad \text{Peak}_t = \max_{s \le t} V_s, \quad \text{DD}_t = \frac{V_t - \text{Peak}_t}{\text{Peak}_t}$$
$$\text{MDD} = \min_{t} \text{DD}_t, \quad \text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}$$
* **CAPM Alpha & Beta**:
$$(R_{\text{net}, t} - R_{f, \text{daily}}) = \alpha_{\text{daily}} + \beta (R_{\text{bench}, t} - R_{f, \text{daily}}) + \epsilon_t$$
$$\beta = \frac{\text{Cov}(R_{\text{net}}, R_{\text{bench}})}{\text{Var}(R_{\text{bench}})}, \quad \alpha_{\text{annual}} = \alpha_{\text{daily}} \times 252$$

---

## 4. Empirical Results (2016 – 2025 Live Simulation)

Backtest conducted across 29 primary liquid Nifty 50 stocks over 2,469 trading days (10 years) under 15 bps/side friction and 6.5% risk-free rate:

| Metric | Momentum (12-1M) | Mean Reversion (Z-Score) | Nifty 50 Benchmark (^NSEI) |
| :--- | :---: | :---: | :---: |
| **Total Return** | **+441.98%** | +61.87% | +232.92% |
| **CAGR** | **+18.83%** | +5.04% | +13.06% |
| **Annualized Volatility** | +19.13% | +14.99% | +16.21% |
| **Sharpe Ratio ($R_f=6.5\%$)** | **0.66** | -0.03 | 0.44 |
| **Sortino Ratio** | **0.89** | -0.04 | 0.60 |
| **Max Drawdown** | -40.85% | -37.45% | -38.44% |
| **Calmar Ratio** | **0.46** | 0.13 | 0.34 |
| **CAPM Alpha** | **+6.19%** | -5.92% | 0.00% |
| **CAPM Beta** | 0.90 | 0.77 | 1.00 |
| **Annualized Turnover** | +459.30% | +4565.60% | 0.00% |
| **Hit Rate** | **+55.79%** | +51.56% | +54.29% |

### Key Quantitative Findings:
1. **Momentum Outperformance**: The 12-1 Month Momentum strategy produced strong alpha (+6.19% annualized) and significantly outperformed the Nifty 50 (+441.98% vs +232.92%), with a lower beta (0.90).
2. **Turnover & Friction Reality**: Mean Reversion suffered severely from transaction drag due to high turnover (4,565% annual), demonstrating why frictionless backtests are dangerously misleading.

---

## 5. Installation & Quick Start

### 5.1 Environment Setup
```bash
# Clone repository
git clone https://github.com/quant-research/nse-backtester.git
cd nse-backtester

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Command Line Interface (CLI)
Run the backtester with full options:
```bash
# Run both strategies over 2016-2025 with 15 bps friction and export reports
python main.py --strategy both --start-date 2016-01-01 --end-date 2025-12-31 --cost-bps 15.0

# Run only Momentum Strategy
python main.py --strategy momentum --cost-bps 15.0

# Run with custom risk-free rate and output paths
python main.py --rf 0.0675 --excel reports/q3_momentum_report.xlsx --chart reports/q3_momentum.png
```

### 5.3 Interactive Quantitative Research Dashboard (Streamlit & Plotly)
Launch the interactive web UI:
```bash
streamlit run app.py
```
* **Dynamic Control Sidebar**: Tune momentum lookback (126–504d), skip lag, top quintile thresholds, rolling mean reversion Z-scores, transaction friction (0–50 bps), single-stock caps (5–30%), and cash buffers.
* **Executive KPI Ribbon**: Live cards for Strategy CAGR, Sharpe Ratio, Max Drawdown, and CAPM Alpha against Nifty 50.
* **Synchronized Plotly Visuals**:
  * **Tab 1: Performance & Risk**: Cumulative equity progression (linear/log scale toggle), underwater drawdowns, and rolling 126-day Sharpe ratio with zero-excess reference lines.
  * **Tab 2: Metrics & Monthly Heatmap**: Full quantitative comparison table and compounding Year $\times$ Month returns heatmap.
  * **Tab 3: Dynamic Allocation**: Stacked area chart displaying multi-asset portfolio weights and active position counts.
  * **Tab 4: Export Center**: Direct in-browser downloads for the 3-tab styled Excel report (`.xlsx`), 300 DPI publication chart (`.png`), and daily timeline CSV.

### 5.4 Automated Reporting Outputs
* **Excel Workbook (`reports/backtest_report.xlsx`)**:
  * **Tab 1: `Executive Summary`**: Color-formatted metric comparison table.
  * **Tab 2: `Monthly Returns Matrix`**: Year $\times$ Month heatmap table of compounding monthly returns and YTD performance.
  * **Tab 3: `Daily Curves`**: Daily time series with embedded native openpyxl `LineChart` widgets.
* **Publication-Grade Chart (`reports/performance_chart.png`)**:
  * High-resolution 300 DPI 2-panel chart: Cumulative equity progression on top, underwater drawdown plot on bottom.

---

## 6. Rigorous Unit Testing Protocol

Run the full pytest suite verifying all mathematical and architectural invariants:
```bash
pytest -v
```

Verified Test Scenarios (30/30 passing):
1. **Look-Ahead Bias Test (`test_look_ahead_prevention_on_price_jump`)**: Predictable 100% price jump on Day $T$. Proves an algorithm computing signals on Day $T$ earns strictly $0.0\%$ on Day $T$ and only executes on Day $T+1$.
2. **Friction Arithmetic Test (`test_transaction_cost_arithmetic_on_100_percent_reallocation`)**: 100% immediate reallocation from Stock A to Stock B correctly charges $200\%$ turnover $\times$ fee.
3. **Mathematical Invariant Test (`test_cagr_equity_doubling_invariant`)**: Equity curve doubling over 252 trading days yields exactly $100\%$ CAGR.
4. **Zero Volatility Edge Case (`test_zero_volatility_edge_case`)**: Flat or constant asset returns do not raise division-by-zero errors.
5. **Drawdown Invariant Test (`test_monotonic_upward_max_drawdown`)**: Monotonic upward series returns exactly $0.0\%$ max drawdown.
6. **CAPM Regression Test (`test_alpha_beta_known_regression`)**: OLS recovery of known alpha and beta parameters from synthetic factor series.
7. **Leading NaN Preservation (`test_clean_prices_preserves_leading_nans`)**: Ensures late IPOs (e.g., LICI) retain leading NaNs without backward propagation.
8. **Portfolio Constraint Capping (`test_portfolio_constraints_capping_and_redistribution`)**: Validates $15\%$ single-stock cap, iterative redistribution, and $5\%$ cash buffer.
9. **UI Chart Trace Validation (`test_ui.py`)**: Smoke tests verifying Plotly trace types, log-scale toggles, stackgroups, and zero reference lines.
10. **Empty Data Graceful Degradation (`test_empty_dataframe_plots`)**: Guarantees all Plotly figure builders gracefully handle empty data without throwing exceptions.

---

## 7. Institutional Roadmap & Production Deployment

To transition `nse-backtester` from research backtesting to institutional live trading execution:
* **Point-in-Time Universe**: Replace free Yahoo Finance data with official NSE historical bhavcopy archives or CMIE Prowess / Refinitiv to eliminate survivorship bias completely.
* **Order Routing**: Connect signal generation to NSE broker APIs (Zerodha Kite Connect, Angel One SmartAPI, Interactive Brokers NSE Gateway).
* **Tick-by-Tick Slippage Models**: Upgrade from fixed 15 bps to Kyle's lambda or Almgren-Chriss square-root impact law based on real-time bid-ask depth.

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

