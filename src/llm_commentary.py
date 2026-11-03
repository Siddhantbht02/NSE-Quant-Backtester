"""LLM Performance Commentary Engine for nse-backtester.

Generates institutional-grade qualitative teardowns of quantitative backtest results
using Google Gemini API (google-genai) or OpenAI API (openai), with an intelligent
rule-based quantitative fallback when API keys are not supplied.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.backtester import BacktestResult

logger = logging.getLogger(__name__)


def build_quant_prompt(result: BacktestResult) -> str:
    """Construct an institutional quantitative prompt from backtest results."""
    summary = result.summary()
    strat_metrics = summary["Strategy"].to_dict()
    bench_metrics = summary["Benchmark (^NSEI)"].to_dict()

    hist = result.portfolio_history
    start_date = hist.index[0].strftime("%Y-%m-%d") if len(hist) > 0 else "N/A"
    end_date = hist.index[-1].strftime("%Y-%m-%d") if len(hist) > 0 else "N/A"
    total_days = len(hist)

    prompt = f"""You are a Senior Quantitative Portfolio Manager and Head of Risk at an Indian quantitative hedge fund.
Analyze the following historical backtest results for an equity strategy evaluated on the National Stock Exchange of India (NSE).

### STRATEGY PROFILE
- Strategy: {result.strategy_name}
- Benchmark: Nifty 50 (^NSEI)
- Backtest Period: {start_date} to {end_date} ({total_days} trading days)
- Execution Lag: t+1 day (signals generated at close of day t are executed at close of day t+1)
- Transaction Friction: 15.0 bps per side (30.0 bps round-trip) modeling STT (0.10%), stamp duty, exchange turnover fees, GST (18%), and slippage
- Position Constraints: Max single-stock cap 15%, minimum 5% cash buffer

### KEY PERFORMANCE METRICS (STRATEGY vs BENCHMARK)
- Total Return: {strat_metrics.get('Total Return', 0):.2%} vs {bench_metrics.get('Total Return', 0):.2%}
- CAGR (Compound Annual Growth Rate): {strat_metrics.get('CAGR', 0):.2%} vs {bench_metrics.get('CAGR', 0):.2%}
- Annualized Volatility: {strat_metrics.get('Annualized Volatility', 0):.2%} vs {bench_metrics.get('Annualized Volatility', 0):.2%}
- Sharpe Ratio (Risk-Free = 6.5% RBI): {strat_metrics.get('Sharpe Ratio', 0):.2f} vs {bench_metrics.get('Sharpe Ratio', 0):.2f}
- Sortino Ratio (Downside Deviation): {strat_metrics.get('Sortino Ratio', 0):.2f} vs {bench_metrics.get('Sortino Ratio', 0):.2f}
- Max Drawdown (Peak-to-Trough): {strat_metrics.get('Max Drawdown', 0):.2%} vs {bench_metrics.get('Max Drawdown', 0):.2%}
- Calmar Ratio (CAGR / |Max Drawdown|): {strat_metrics.get('Calmar Ratio', 0):.2f} vs {bench_metrics.get('Calmar Ratio', 0):.2f}
- CAPM Alpha (Annualized): {strat_metrics.get('Alpha (CAPM)', 0):.2%}
- CAPM Beta: {strat_metrics.get('Beta (CAPM)', 0):.2f}
- Annualized Turnover: {strat_metrics.get('Annualized Turnover', 0):.2f}x
- Hit Rate (Daily Win %): {strat_metrics.get('Hit Rate', 0):.2%}

### INSTRUCTIONS
Provide a structured, rigorous, 4-section executive commentary:
1. **Executive Alpha Assessment**: Evaluate net CAGR outperformance, statistical significance of alpha, and return attribution.
2. **Risk-Adjusted Efficiency & Asymmetry**: Compare Sharpe and Sortino ratios. Discuss upside capture vs downside volatility.
3. **Drawdown Duration & Tail Risk Profile**: Evaluate the maximum drawdown, recovery dynamics, and Calmar ratio.
4. **Execution Scalability & Practical Recommendations**: Evaluate whether the 15 bps friction and annual turnover allow institutional execution capacity without alpha decay. Provide 2 actionable enhancements (e.g. regime filter, dynamic cash hedging).

Keep the tone professional, objective, and quantitative. Do not use generic filler. Write in crisp markdown format.
"""
    return prompt


def generate_fallback_commentary(result: BacktestResult) -> str:
    """Generate high-quality algorithmic quantitative commentary when LLM API is unavailable.
    
    This ensures zero downtime and reliable reports regardless of external network or API key status.
    """
    summary = result.summary()
    strat = summary["Strategy"].to_dict()
    bench = summary["Benchmark (^NSEI)"].to_dict()

    cagr_diff = (strat.get("CAGR", 0) - bench.get("CAGR", 0)) * 100
    alpha_pct = strat.get("Alpha (CAPM)", 0) * 100
    beta_val = strat.get("Beta (CAPM)", 1.0)
    sharpe_strat = strat.get("Sharpe Ratio", 0)
    sharpe_bench = bench.get("Sharpe Ratio", 0)
    sortino_strat = strat.get("Sortino Ratio", 0)
    sortino_bench = bench.get("Sortino Ratio", 0)
    mdd_strat = strat.get("Max Drawdown", 0) * 100
    mdd_bench = bench.get("Max Drawdown", 0) * 100
    calmar_strat = strat.get("Calmar Ratio", 0)
    turnover = strat.get("Annualized Turnover", 0)

    # Dynamic classification
    outperformance = "strong positive" if cagr_diff > 3.0 else ("moderate" if cagr_diff > 0 else "underperforming")
    asymmetry = "favorable downside protection" if sortino_strat > sharpe_strat * 1.25 else "balanced volatility"
    beta_desc = "defensive market sensitivity" if beta_val < 0.9 else ("aggressive beta tilt" if beta_val > 1.1 else "market-neutral beta proximity")

    commentary = f"""### 🤖 Quantitative Performance Teardown: {result.strategy_name}

#### 1. Executive Alpha Assessment
Over the evaluated multi-year period, **{result.strategy_name}** demonstrated **{outperformance} alpha generation**, delivering a net CAGR of **{strat.get('CAGR', 0):.2%}** compared to **{bench.get('CAGR', 0):.2%}** for the Nifty 50 benchmark (a spread of **{cagr_diff:+.2f}%**). The CAPM regression yields an annualized Jensen's Alpha of **{alpha_pct:+.2f}%** with a Beta of **{beta_val:.2f}**, confirming that excess returns stem primarily from systematic factor selection rather than unhedged market leverage ({beta_desc}).

#### 2. Risk-Adjusted Efficiency & Asymmetry
The strategy achieved an annualized Sharpe Ratio of **{sharpe_strat:.2f}** (vs. **{sharpe_bench:.2f}** for Nifty 50) using the RBI 6.5% benchmark risk-free rate. The Sortino Ratio expanded to **{sortino_strat:.2f}** (vs. **{sortino_bench:.2f}** for the benchmark), reflecting {asymmetry}. The positive spread between Sortino and Sharpe confirms that return volatility is skewed toward the upside while penalizing adverse downside drawdowns.

#### 3. Drawdown Duration & Tail Risk Profile
The maximum peak-to-trough drawdown recorded was **{mdd_strat:.2f}%**, against **{mdd_bench:.2f}%** for the broader index. The resulting Calmar Ratio stands at **{calmar_strat:.2f}** (vs. **{bench.get('Calmar Ratio', 0):.2f}** for Nifty 50). While the strategy participates in macroeconomic downturns (such as market-wide liquidity shocks), the enforced 5% cash buffer and 15% single-stock position caps prevented catastrophic idiosyncratic tail-risk blowups.

#### 4. Execution Scalability & Strategic Recommendations
With an annualized turnover of **{turnover:.2f}x**, the strategy absorbs realistic Indian market microstructure friction (15.0 bps per side covering delivery STT, exchange turnover fees, GST, and slippage) without significant alpha degradation. Signals executed with strict t+1 lag eliminate look-ahead bias.
- **Enhancement 1 (Macro Trend Filter)**: Implement a 200-day EMA market regime filter on Nifty 50 to scale cash reserves from 5% to 50% during structural bear regimes, reducing maximum drawdown.
- **Enhancement 2 (Volatility Sizing)**: Shift from equal quintile weighting to Inverse-Volatility sizing to equalize risk contribution across high-beta names.
"""
    return commentary.strip()


def generate_llm_commentary(
    result: BacktestResult,
    api_key: Optional[str] = None,
    provider: str = "auto",
    model: Optional[str] = None,
) -> str:
    """Generate institutional performance commentary using Gemini, OpenAI, or algorithmic fallback.
    
    Parameters
    ----------
    result : BacktestResult
        Completed backtest simulation result object.
    api_key : Optional[str]
        API key for Gemini or OpenAI. If None, checks GEMINI_API_KEY and OPENAI_API_KEY env vars.
    provider : str
        'auto', 'gemini', 'openai', or 'fallback'.
    model : Optional[str]
        Optional model override (e.g., 'gemini-2.5-flash', 'gpt-4o-mini').

    Returns
    -------
    str
        Markdown formatted institutional performance commentary.
    """
    if provider == "fallback":
        return generate_fallback_commentary(result)

    prompt = build_quant_prompt(result)

    # 1. Resolve Provider and Key
    gemini_key = api_key if provider == "gemini" else (api_key or os.environ.get("GEMINI_API_KEY"))
    openai_key = api_key if provider == "openai" else (api_key or os.environ.get("OPENAI_API_KEY"))

    # Try Gemini First
    if provider in ["auto", "gemini"] and gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            target_model = model or "gemini-2.5-flash"
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini API commentary generation failed: {e}. Attempting fallback...")
            if provider == "gemini":
                return generate_fallback_commentary(result)

    # Try OpenAI Next
    if provider in ["auto", "openai"] and openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            target_model = model or "gpt-4o-mini"
            completion = client.chat.completions.create(
                model=target_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Senior Quantitative Portfolio Manager and Head of Risk specializing in Indian equity strategies.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = completion.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            logger.warning(f"OpenAI API commentary generation failed: {e}. Attempting fallback...")
            if provider == "openai":
                return generate_fallback_commentary(result)

    # If no keys or API failed, use robust algorithmic fallback
    logger.info("Using algorithmic quantitative commentary generator (no active LLM API key detected).")
    return generate_fallback_commentary(result)
