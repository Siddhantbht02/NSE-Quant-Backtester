"""UI layer for nse-backtester quantitative research dashboard."""

from __future__ import annotations

from ui.components import (
    inject_custom_css,
    load_cached_market_data,
    render_header_banner,
    render_info_popover,
    render_kpi_ribbon,
    render_llm_commentary_card,
    render_monthly_matrix_heatmap,
    render_performance_table_component,
)
from ui.plots import (
    create_allocation_area_chart,
    create_drawdown_plot,
    create_equity_curve,
    create_rolling_sharpe_chart,
)
from ui.sidebar import SidebarConfig, render_sidebar

__all__ = [
    "inject_custom_css",
    "load_cached_market_data",
    "render_header_banner",
    "render_info_popover",
    "render_kpi_ribbon",
    "render_llm_commentary_card",
    "render_monthly_matrix_heatmap",
    "render_performance_table_component",
    "create_allocation_area_chart",
    "create_drawdown_plot",
    "create_equity_curve",
    "create_rolling_sharpe_chart",
    "SidebarConfig",
    "render_sidebar",
]
