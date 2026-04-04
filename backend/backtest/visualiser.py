"""
StockSense AI — backtest/visualiser.py
========================================
Chapter 5.4 — Strategy vs Buy-and-Hold comparison, visual equity
curves, and the complete backtesting report that powers the website's
performance section.

This file owns:
  - Equity curve generation   (strategy vs buy-and-hold on the same chart)
  - Drawdown visualisation    (running drawdown over time)
  - Rolling Sharpe chart      (60-day rolling risk-adjusted returns)
  - Monthly returns heatmap   (calendar-style profitability grid)
  - Trade scatter plot        (individual trade P&L vs date)
  - Full "performance page"   (multi-panel figure for the website)
  - JSON-ready chart data     (arrays for the React frontend)

It does NOT own:
  - Trade simulation          → backtest/engine.py
  - Metric computation        → backtest/metrics.py
  - Prediction orchestration  → backtest/backtester.py
  - ML model metrics          → models/evaluator.py

Why compare against buy-and-hold?
─────────────────────────────────────────────────────────────
  Buy-and-hold is the simplest possible investment strategy.
  It captures the market's average return with zero effort.
  Any active trading strategy must demonstrate clear advantage
  over this baseline, otherwise you should just buy an index fund.

  We use the EXACT same test period and starting capital so the
  comparison is completely fair — no survivorship bias, no cherry-
  picking start/end dates, same transaction cost model.

Why both visual + JSON output?
─────────────────────────────────────────────────────────────
  Visuals (matplotlib): for Jupyter notebooks and printed reports
  JSON arrays:          for the React frontend (Chart.js / Recharts)
  Both render the same underlying data. The matplotlib figures are
  generated once and saved as PNG. The JSON arrays stream live to
  the frontend whenever a user views a stock's performance page.

Architecture:
─────────────────────────────────────────────────────────────
  backtester.py  →  engine.py   →  metrics.py  →  visualiser.py
  (orchestrate)     (simulate)     (measure)      (display)

  The website calls:
    1. backtester.backtest_pipeline()  →  BacktestResult
    2. metrics.compute_metrics()       →  PerformanceReport
    3. visualiser.build_report()       →  BacktestReport (this file)

  The React frontend receives the BacktestReport as JSON and renders
  the equity curve, drawdown chart, monthly heatmap, and metric cards.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from backtest.engine import BacktestResult, Trade
from backtest.metrics import (
    PerformanceReport,
    compute_metrics,
    report_to_dict,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# StockSense brand colours — consistent across all charts
COLOUR_STRATEGY   = "#2563EB"   # primary blue  — our model's equity curve
COLOUR_BENCHMARK  = "#94A3B8"   # muted slate   — buy-and-hold benchmark
COLOUR_DRAWDOWN   = "#EF4444"   # red           — drawdown area fill
COLOUR_WIN        = "#22C55E"   # green         — profitable trades
COLOUR_LOSS       = "#EF4444"   # red           — losing trades
COLOUR_ROLLING_SH = "#8B5CF6"   # purple        — rolling Sharpe line
COLOUR_GRID       = "#F1F5F9"   # light grey    — chart gridlines
COLOUR_BG         = "#FFFFFF"   # white         — chart background

# Chart layout
DEFAULT_FIG_WIDTH  = 14          # inches — fits 1080p and above
DEFAULT_FIG_HEIGHT = 8           # inches — 16:9 ratio
DPI                = 150         # resolution for saved PNGs
FONT_SIZE_TITLE    = 14          # section titles on performance page
FONT_SIZE_LABEL    = 11          # axis labels
FONT_SIZE_TICK     = 9           # tick mark labels
ROLLING_SHARPE_WINDOW = 60       # days — matches metrics.py


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURE — Complete Backtest Report
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestReport:
    """
    The final output of the entire backtesting pipeline.
    Contains everything the website needs to render the performance page.

    Data flow:
      BacktestResult (engine.py)
        → PerformanceReport (metrics.py)
          → BacktestReport (this file)
            → JSON for React frontend

    Why wrap PerformanceReport instead of extending it?
    ─────────────────────────────────────────────────────────────
    PerformanceReport holds computed numbers.
    BacktestReport adds chart-ready data arrays AND the figures
    themselves. Separation keeps metrics.py free of matplotlib
    dependency — important for lightweight API containers.

    Fields
    ──────
    report        : PerformanceReport with all computed metrics
    equity_data   : dict with 'dates', 'strategy', 'benchmark' arrays
    drawdown_data : dict with 'dates', 'drawdown' arrays
    monthly_data  : dict with 'year_months', 'returns' arrays
    trade_data    : dict with 'dates', 'returns', 'is_win' arrays
    rolling_data  : dict with 'dates', 'sharpe' arrays
    figures       : dict mapping figure_name → matplotlib Figure
    """
    # Core metrics
    report:           PerformanceReport

    # Chart-ready data (for React frontend — JSON serialisable)
    equity_data:      Dict               # dates, strategy values, benchmark values
    drawdown_data:    Dict               # dates, drawdown fractions
    monthly_data:     Dict               # year-month labels, monthly returns
    trade_data:       Dict               # per-trade: date, return, win/loss flag
    rolling_data:     Dict               # dates, rolling Sharpe values

    # Matplotlib figures (for Jupyter / PNG export)
    figures:          Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS — Chart styling
# ══════════════════════════════════════════════════════════════════════════════

def _apply_stocksense_style(ax: plt.Axes, title: str = "") -> None:
    """
    Apply consistent StockSense styling to any matplotlib Axes.

    Why centralise styling?
    ─────────────────────────────────────────────────────────────
    Every chart on the website should look like it belongs together.
    Centralising font sizes, colours, and grid style ensures this.
    Changing a colour here updates every chart simultaneously.
    """
    ax.set_facecolor(COLOUR_BG)
    ax.grid(True, alpha=0.3, color=COLOUR_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_TITLE,
                      fontweight="bold", pad=12)


def _format_date_axis(ax: plt.Axes) -> None:
    """
    Auto-format the x-axis for date display.
    Uses intelligent month/year formatting based on date range.
    """
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def _format_dollar_axis(ax: plt.Axes) -> None:
    """Format y-axis with dollar signs and thousands separators."""
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )


def _format_pct_axis(ax: plt.Axes) -> None:
    """Format y-axis as percentages."""
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%")
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 1 — Equity Curve (Strategy vs Buy-and-Hold)
# ══════════════════════════════════════════════════════════════════════════════

def plot_equity_curve(
    result:  BacktestResult,
    report:  PerformanceReport,
    ax:      Optional[plt.Axes] = None,
    show:    bool = False,
) -> Tuple[plt.Figure, Dict]:
    """
    Plot strategy equity curve vs buy-and-hold benchmark.

    This is the HERO chart on the stock performance page.
    It answers the single most important question:
      "Did my model make more money than just holding the stock?"

    The strategy line (blue) vs the benchmark line (grey) makes
    the answer immediately visual — no numbers needed.

    Parameters
    ----------
    result : BacktestResult from engine.py
    report : PerformanceReport from metrics.py
    ax     : Optional axes to plot on (for multi-panel figures)
    show   : If True, plt.show() immediately (for Jupyter)

    Returns
    -------
    (fig, chart_data) where chart_data is JSON-serialisable dict
    """
    # ── Extract data ──────────────────────────────────────────────────────
    strategy_pv  = result.daily_portfolio["portfolio_value"]
    benchmark_pv = result.benchmark_portfolio["portfolio_value"]

    # Align indices — benchmark may have slightly different dates
    # Use inner join to keep only overlapping dates
    common_dates = strategy_pv.index.intersection(benchmark_pv.index)
    strategy_pv  = strategy_pv.loc[common_dates]
    benchmark_pv = benchmark_pv.loc[common_dates]

    # ── Build chart data dict for React frontend ──────────────────────────
    chart_data = {
        "dates":      [str(d.date()) for d in common_dates],
        "strategy":   strategy_pv.round(2).tolist(),
        "benchmark":  benchmark_pv.round(2).tolist(),
    }

    # ── Create figure ─────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
    else:
        fig = ax.figure

    # ── Plot both equity curves ───────────────────────────────────────────
    ax.plot(common_dates, strategy_pv.values,
            color=COLOUR_STRATEGY, linewidth=2.0,
            label=f"AI Strategy ({report.total_return*100:+.1f}%)",
            zorder=3)

    ax.plot(common_dates, benchmark_pv.values,
            color=COLOUR_BENCHMARK, linewidth=1.5, linestyle="--",
            label=f"Buy & Hold ({report.benchmark_total_return*100:+.1f}%)",
            zorder=2)

    # ── Shade the area between the two curves ─────────────────────────────
    # Green where strategy > benchmark, red where strategy < benchmark
    # This immediately shows WHEN the strategy outperforms
    ax.fill_between(
        common_dates,
        strategy_pv.values,
        benchmark_pv.values,
        where=(strategy_pv.values >= benchmark_pv.values),
        color=COLOUR_WIN, alpha=0.08,
        interpolate=True,
        label="_nolegend_",
    )
    ax.fill_between(
        common_dates,
        strategy_pv.values,
        benchmark_pv.values,
        where=(strategy_pv.values < benchmark_pv.values),
        color=COLOUR_LOSS, alpha=0.08,
        interpolate=True,
        label="_nolegend_",
    )

    # ── Mark the starting capital line ────────────────────────────────────
    initial_capital = result.config["initial_capital"]
    ax.axhline(y=initial_capital, color="#CBD5E1", linewidth=0.8,
               linestyle=":", alpha=0.7, zorder=1)
    ax.text(common_dates[0], initial_capital * 1.01,
            f"  ${initial_capital:,.0f} start",
            fontsize=FONT_SIZE_TICK, color="#94A3B8", va="bottom")

    # ── Styling ───────────────────────────────────────────────────────────
    beat_text = "✅ Beats" if report.beats_benchmark else "❌ Trails"
    _apply_stocksense_style(
        ax,
        f"{result.ticker} — Equity Curve  "
        f"({beat_text} buy-and-hold by "
        f"{report.alpha*100:+.1f}%/yr)"
    )
    _format_date_axis(ax)
    _format_dollar_axis(ax)
    ax.set_xlabel("Date", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Portfolio Value ($)", fontsize=FONT_SIZE_LABEL)
    ax.legend(loc="upper left", fontsize=FONT_SIZE_LABEL, framealpha=0.9)

    if standalone:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, chart_data


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 2 — Drawdown Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_drawdown(
    result:  BacktestResult,
    report:  PerformanceReport,
    ax:      Optional[plt.Axes] = None,
    show:    bool = False,
) -> Tuple[plt.Figure, Dict]:
    """
    Plot the running drawdown (peak-to-trough decline) over time.

    Drawdown answers: "What was the worst loss from a portfolio peak?"
    Every investor cares about this — it determines whether you would
    have stopped trading and abandoned the strategy.

    A strategy with 15% max drawdown looks great on paper.
    Experiencing it live (watching $10,000 drop to $8,500) tests
    your conviction. The drawdown chart shows you exactly when
    those painful moments occur.

    Parameters
    ----------
    result : BacktestResult
    report : PerformanceReport (contains rolling_drawdown)
    ax     : Optional axes
    show   : Display immediately

    Returns
    -------
    (fig, chart_data)
    """
    # ── Compute running drawdown from portfolio values ────────────────────
    pv       = result.daily_portfolio["portfolio_value"]
    peak     = pv.cummax()
    drawdown = (pv - peak) / (peak + 1e-10)

    chart_data = {
        "dates":    [str(d.date()) for d in drawdown.index],
        "drawdown": drawdown.round(4).tolist(),
    }

    # ── Create figure ─────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, 4))
    else:
        fig = ax.figure

    # ── Plot drawdown as filled area ──────────────────────────────────────
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color=COLOUR_DRAWDOWN, alpha=0.3, zorder=2)
    ax.plot(drawdown.index, drawdown.values,
            color=COLOUR_DRAWDOWN, linewidth=1.0, alpha=0.8, zorder=3)

    # ── Mark the maximum drawdown point ───────────────────────────────────
    max_dd_idx  = drawdown.idxmin()
    max_dd_val  = drawdown.min()
    ax.scatter([max_dd_idx], [max_dd_val],
               color=COLOUR_DRAWDOWN, s=60, zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.annotate(
        f"Max DD: {max_dd_val*100:.1f}%",
        xy=(max_dd_idx, max_dd_val),
        xytext=(15, -15),
        textcoords="offset points",
        fontsize=FONT_SIZE_TICK,
        color=COLOUR_DRAWDOWN,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOUR_DRAWDOWN, lw=1.0),
    )

    # ── Zero line ─────────────────────────────────────────────────────────
    ax.axhline(y=0, color="#64748B", linewidth=0.5, zorder=1)

    # ── Styling ───────────────────────────────────────────────────────────
    _apply_stocksense_style(
        ax,
        f"{result.ticker} — Drawdown  "
        f"(max: {report.max_drawdown*100:.1f}%, "
        f"duration: {report.max_drawdown_duration}d)"
    )
    _format_date_axis(ax)
    _format_pct_axis(ax)
    ax.set_xlabel("Date", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Drawdown (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(min(max_dd_val * 1.3, -0.01), 0.01)

    if standalone:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, chart_data


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 3 — Rolling Sharpe Ratio
# ══════════════════════════════════════════════════════════════════════════════

def plot_rolling_sharpe(
    report:  PerformanceReport,
    ax:      Optional[plt.Axes] = None,
    show:    bool = False,
) -> Tuple[plt.Figure, Dict]:
    """
    Plot the 60-day rolling Sharpe ratio over time.

    A stable rolling Sharpe above 0.5 is much better than a volatile
    one that averages 1.0 — consistency matters for deployment.

    This chart reveals:
      - Regime sensitivity: does the model work only in bull markets?
      - Degradation: is performance decaying over time?
      - Seasonality: does it work better in certain months?

    The reference lines at 0.0 and 1.0 help contextualise the values:
      Sharpe > 1.0 = strong risk-adjusted performance
      Sharpe > 0.0 = positive expected return
      Sharpe < 0.0 = losing money on a risk-adjusted basis

    Parameters
    ----------
    report : PerformanceReport (contains rolling_sharpe Series)
    ax     : Optional axes
    show   : Display immediately

    Returns
    -------
    (fig, chart_data)
    """
    rolling = report.rolling_sharpe

    chart_data = {
        "dates":  [str(d.date()) for d in rolling.index],
        "sharpe": rolling.round(4).tolist(),
    }

    # ── Create figure ─────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, 4))
    else:
        fig = ax.figure

    # ── Plot rolling Sharpe ───────────────────────────────────────────────
    ax.plot(rolling.index, rolling.values,
            color=COLOUR_ROLLING_SH, linewidth=1.5, zorder=3)

    # ── Shade regions by Sharpe quality ───────────────────────────────────
    ax.fill_between(rolling.index, rolling.values, 0,
                    where=(rolling.values >= 0),
                    color=COLOUR_WIN, alpha=0.08, zorder=2)
    ax.fill_between(rolling.index, rolling.values, 0,
                    where=(rolling.values < 0),
                    color=COLOUR_LOSS, alpha=0.08, zorder=2)

    # ── Reference lines ───────────────────────────────────────────────────
    ax.axhline(y=0.0, color="#64748B", linewidth=0.8,
               linestyle="--", alpha=0.6, zorder=1)
    ax.axhline(y=1.0, color=COLOUR_WIN, linewidth=0.8,
               linestyle=":", alpha=0.5, zorder=1)
    ax.text(rolling.index[0], 1.05, "  Sharpe = 1.0 (good)",
            fontsize=FONT_SIZE_TICK - 1, color=COLOUR_WIN, alpha=0.8)

    # ── Styling ───────────────────────────────────────────────────────────
    overall_sharpe = report.sharpe_ratio
    _apply_stocksense_style(
        ax,
        f"{report.ticker} — Rolling Sharpe "
        f"({ROLLING_SHARPE_WINDOW}d window, "
        f"overall: {overall_sharpe:.2f})"
    )
    _format_date_axis(ax)
    ax.set_xlabel("Date", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Sharpe Ratio", fontsize=FONT_SIZE_LABEL)

    if standalone:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, chart_data


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 4 — Monthly Returns Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_monthly_returns(
    result:  BacktestResult,
    ax:      Optional[plt.Axes] = None,
    show:    bool = False,
) -> Tuple[plt.Figure, Dict]:
    """
    Plot a calendar-style monthly returns heatmap.

    Each cell shows one month's total return. Green = profit, red = loss.
    This answers: "Which months did the strategy work best?"
    Helps detect seasonality and regime-dependent performance.

    Layout: rows = years, columns = months (Jan → Dec)
    This mirrors the standard hedge-fund tear-sheet format.

    Parameters
    ----------
    result : BacktestResult
    ax     : Optional axes
    show   : Display immediately

    Returns
    -------
    (fig, chart_data) where chart_data has year-month labels + returns
    """
    # ── Compute monthly returns from portfolio values ─────────────────────
    pv      = result.daily_portfolio["portfolio_value"]
    monthly = pv.resample("ME").last().pct_change().dropna()

    # Build a year × month pivot table
    years   = sorted(set(monthly.index.year))
    months  = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    grid = np.full((len(years), 12), np.nan)
    for date, ret in monthly.items():
        row = years.index(date.year)
        col = date.month - 1
        grid[row, col] = ret

    # ── Chart data for frontend ───────────────────────────────────────────
    chart_data = {
        "years":       years,
        "months":      month_names,
        "returns":     [[None if np.isnan(v) else round(v * 100, 2)
                         for v in row] for row in grid],
    }

    # ── Create figure ─────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        height = max(2.5, len(years) * 0.8 + 1.5)
        fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, height))
    else:
        fig = ax.figure

    # ── Draw heatmap ──────────────────────────────────────────────────────
    # Use red-white-green colourmap centred at zero
    vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 0.01)
    im   = ax.imshow(grid, aspect="auto",
                     cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                     interpolation="nearest")

    # ── Label cells with return percentages ───────────────────────────────
    for i in range(len(years)):
        for j in range(12):
            val = grid[i, j]
            if not np.isnan(val):
                colour = "white" if abs(val) > vmax * 0.7 else "black"
                ax.text(j, i, f"{val*100:+.1f}%",
                        ha="center", va="center",
                        fontsize=FONT_SIZE_TICK - 1,
                        fontweight="bold", color=colour)

    # ── Axis labels ───────────────────────────────────────────────────────
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=FONT_SIZE_TICK)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=FONT_SIZE_TICK)

    # ── Colour bar ────────────────────────────────────────────────────────
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK - 1)
    cbar.set_label("Monthly Return (%)", fontsize=FONT_SIZE_TICK)

    # ── Styling ───────────────────────────────────────────────────────────
    total_positive = np.nansum(grid > 0)
    total_months   = np.sum(~np.isnan(grid))
    pct_positive   = total_positive / total_months * 100 if total_months > 0 else 0
    _apply_stocksense_style(
        ax,
        f"{result.ticker} — Monthly Returns  "
        f"({pct_positive:.0f}% profitable months)"
    )
    ax.grid(False)   # disable grid for heatmap
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if standalone:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, chart_data


# ══════════════════════════════════════════════════════════════════════════════
#  CHART 5 — Trade Scatter Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_trade_scatter(
    result:  BacktestResult,
    ax:      Optional[plt.Axes] = None,
    show:    bool = False,
) -> Tuple[plt.Figure, Dict]:
    """
    Scatter plot of individual trades: date (x) vs return % (y).

    Green dots = winning trades, red dots = losing trades.
    Dot size proportional to absolute P&L.

    This reveals:
      - Are wins consistently small or are there fat-tail profits?
      - Are losses well-contained (tight cluster near zero)?
      - Is there a time trend in trade quality?

    A healthy strategy shows: many small losses + occasional big wins.
    An unhealthy one shows: many small wins + occasional huge losses.

    Parameters
    ----------
    result : BacktestResult (uses result.trades list)
    ax     : Optional axes
    show   : Display immediately

    Returns
    -------
    (fig, chart_data)
    """
    trades = result.trades

    if not trades:
        # No trades — create empty chart
        chart_data = {"dates": [], "returns": [], "is_win": []}
        fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, 4))
        ax.text(0.5, 0.5, "No trades executed",
                ha="center", va="center", fontsize=14, color="#94A3B8",
                transform=ax.transAxes)
        return fig, chart_data

    dates      = [t.entry_date for t in trades]
    returns    = [t.return_pct for t in trades]
    is_win     = [t.return_pct > 0 for t in trades]
    abs_pnl    = [abs(t.net_pnl) for t in trades]

    chart_data = {
        "dates":    [str(d.date()) for d in dates],
        "returns":  [round(r * 100, 3) for r in returns],
        "is_win":   is_win,
        "net_pnl":  [round(t.net_pnl, 2) for t in trades],
    }

    # ── Create figure ─────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(DEFAULT_FIG_WIDTH, 4))
    else:
        fig = ax.figure

    # ── Scale dot sizes: bigger dot = bigger P&L ──────────────────────────
    max_pnl = max(abs_pnl) if abs_pnl else 1.0
    sizes   = [max(20, min(200, (p / max_pnl) * 150)) for p in abs_pnl]

    # ── Plot wins and losses separately for colour ────────────────────────
    win_dates   = [d for d, w in zip(dates, is_win) if w]
    win_rets    = [r for r, w in zip(returns, is_win) if w]
    win_sizes   = [s for s, w in zip(sizes, is_win) if w]

    loss_dates  = [d for d, w in zip(dates, is_win) if not w]
    loss_rets   = [r for r, w in zip(returns, is_win) if not w]
    loss_sizes  = [s for s, w in zip(sizes, is_win) if not w]

    ax.scatter(win_dates, [r * 100 for r in win_rets],
               s=win_sizes, color=COLOUR_WIN, alpha=0.6,
               edgecolors="white", linewidths=0.5,
               label=f"Wins ({len(win_dates)})", zorder=3)

    ax.scatter(loss_dates, [r * 100 for r in loss_rets],
               s=loss_sizes, color=COLOUR_LOSS, alpha=0.6,
               edgecolors="white", linewidths=0.5,
               label=f"Losses ({len(loss_dates)})", zorder=3)

    # ── Zero line ─────────────────────────────────────────────────────────
    ax.axhline(y=0, color="#64748B", linewidth=0.8,
               linestyle="--", alpha=0.6, zorder=1)

    # ── Styling ───────────────────────────────────────────────────────────
    n_trades = len(trades)
    win_rate = len(win_dates) / n_trades if n_trades > 0 else 0
    _apply_stocksense_style(
        ax,
        f"{result.ticker} — Individual Trades  "
        f"({n_trades} trades, {win_rate*100:.1f}% win rate)"
    )
    _format_date_axis(ax)
    ax.set_xlabel("Date", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Trade Return (%)", fontsize=FONT_SIZE_LABEL)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_TICK, framealpha=0.9)

    if standalone:
        fig.tight_layout()
    if show:
        plt.show()

    return fig, chart_data


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PERFORMANCE PAGE — Multi-panel figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_performance_page(
    result:  BacktestResult,
    report:  PerformanceReport,
    save_path: Optional[str] = None,
    show:      bool = False,
) -> Tuple[plt.Figure, Dict[str, Dict]]:
    """
    Generate the complete multi-panel performance page figure.

    This is the main figure shown on the StockSense stock page.
    Contains 4 vertically stacked panels:
      1. Equity curve (strategy vs buy-and-hold)   — hero chart
      2. Drawdown chart                             — risk view
      3. Rolling Sharpe ratio                       — consistency view
      4. Trade scatter plot                         — trade quality view

    (Monthly returns heatmap is rendered separately on the frontend
     because it needs a different aspect ratio.)

    Parameters
    ----------
    result    : BacktestResult from engine.py
    report    : PerformanceReport from metrics.py
    save_path : If provided, saves figure as PNG at this path
    show      : If True, displays figure immediately (Jupyter)

    Returns
    -------
    (fig, all_chart_data) where all_chart_data maps panel names
    to their JSON-serialisable chart data dicts.
    """
    # ── Create 4-panel figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(
        4, 1,
        figsize=(DEFAULT_FIG_WIDTH, 22),
        gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1.2]},
    )

    fig.suptitle(
        f"StockSense AI — {result.ticker} Backtest Performance",
        fontsize=16, fontweight="bold", y=0.995,
    )

    # ── Panel 1: Equity Curve ─────────────────────────────────────────────
    _, equity_data = plot_equity_curve(result, report, ax=axes[0])

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    _, drawdown_data = plot_drawdown(result, report, ax=axes[1])

    # ── Panel 3: Rolling Sharpe ───────────────────────────────────────────
    _, rolling_data = plot_rolling_sharpe(report, ax=axes[2])

    # ── Panel 4: Trade Scatter ────────────────────────────────────────────
    _, trade_data = plot_trade_scatter(result, ax=axes[3])

    # ── Layout ────────────────────────────────────────────────────────────
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # ── Save if requested ─────────────────────────────────────────────────
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")

    if show:
        plt.show()

    all_chart_data = {
        "equity":   equity_data,
        "drawdown": drawdown_data,
        "rolling":  rolling_data,
        "trades":   trade_data,
    }

    return fig, all_chart_data


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD COMPLETE REPORT — Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_report(
    result:     BacktestResult,
    save_dir:   Optional[str] = None,
    verbose:    bool = True,
) -> BacktestReport:
    """
    Build the complete backtest report: metrics + charts + JSON data.

    This is THE function called by the website. It:
      1. Computes all metrics via metrics.compute_metrics()
      2. Generates all chart data arrays for React
      3. Creates matplotlib figures for Jupyter/PNG export
      4. Wraps everything in a BacktestReport dataclass

    Parameters
    ----------
    result   : BacktestResult from engine.py or backtester.py
    save_dir : If provided, saves all figures as PNGs in this directory
    verbose  : Print progress messages

    Returns
    -------
    BacktestReport with metrics, chart data, and matplotlib figures.

    Usage example (website API):
        result = backtester.backtest_pipeline(pipeline, X_test, "AAPL")
        report = visualiser.build_report(result)
        return report_to_json(report)  # → React frontend

    Usage example (Jupyter notebook):
        result = backtester.backtest_pipeline(pipeline, X_test, "AAPL")
        report = visualiser.build_report(result, save_dir="./figures")
        # figures are saved as PNGs and displayed inline
    """
    ticker = result.ticker

    if verbose:
        print(f"\n{'═'*60}")
        print(f"Building backtest report: {ticker}")
        print(f"{'═'*60}")

    # ── Step 1: Compute metrics ───────────────────────────────────────────
    if verbose:
        print(f"  Computing performance metrics...")
    perf_report = compute_metrics(result, verbose=verbose)

    # ── Step 2: Generate individual charts + data arrays ──────────────────
    if verbose:
        print(f"  Generating charts...")

    fig_equity,   equity_data   = plot_equity_curve(result, perf_report)
    fig_dd,       drawdown_data = plot_drawdown(result, perf_report)
    fig_rolling,  rolling_data  = plot_rolling_sharpe(perf_report)
    fig_monthly,  monthly_data  = plot_monthly_returns(result)
    fig_trades,   trade_data    = plot_trade_scatter(result)

    # ── Step 3: Generate full performance page ────────────────────────────
    save_page_path = None
    if save_dir:
        save_page_path = str(Path(save_dir) / f"{ticker}_performance.png")

    fig_page, _ = plot_performance_page(
        result, perf_report,
        save_path=save_page_path,
        show=False,
    )

    # ── Step 4: Save individual charts if save_dir provided ───────────────
    if save_dir:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        fig_equity.savefig(
            save_dir_path / f"{ticker}_equity_curve.png",
            dpi=DPI, bbox_inches="tight", facecolor="white",
        )
        fig_dd.savefig(
            save_dir_path / f"{ticker}_drawdown.png",
            dpi=DPI, bbox_inches="tight", facecolor="white",
        )
        fig_rolling.savefig(
            save_dir_path / f"{ticker}_rolling_sharpe.png",
            dpi=DPI, bbox_inches="tight", facecolor="white",
        )
        fig_monthly.savefig(
            save_dir_path / f"{ticker}_monthly_returns.png",
            dpi=DPI, bbox_inches="tight", facecolor="white",
        )
        fig_trades.savefig(
            save_dir_path / f"{ticker}_trades.png",
            dpi=DPI, bbox_inches="tight", facecolor="white",
        )

        if verbose:
            print(f"  Figures saved to: {save_dir_path}")

    # ── Step 5: Assemble report ───────────────────────────────────────────
    figures = {
        "equity_curve":    fig_equity,
        "drawdown":        fig_dd,
        "rolling_sharpe":  fig_rolling,
        "monthly_returns": fig_monthly,
        "trade_scatter":   fig_trades,
        "performance_page": fig_page,
    }

    bt_report = BacktestReport(
        report=perf_report,
        equity_data=equity_data,
        drawdown_data=drawdown_data,
        monthly_data=monthly_data,
        trade_data=trade_data,
        rolling_data=rolling_data,
        figures=figures,
    )

    if verbose:
        print(f"  Report built: {len(figures)} figures, "
              f"{len(equity_data['dates'])} equity points")
        print(f"{'═'*60}")

    # Close all figures to free memory (they're stored in the report)
    plt.close("all")

    return bt_report


# ══════════════════════════════════════════════════════════════════════════════
#  JSON SERIALISATION — For the React frontend
# ══════════════════════════════════════════════════════════════════════════════

def report_to_json(bt_report: BacktestReport) -> Dict:
    """
    Convert the full BacktestReport to a JSON-serialisable dict.

    This is the function called by FastAPI's GET /backtest?ticker=AAPL
    endpoint. The React frontend receives this and renders:
      - Metric cards (Sharpe, drawdown, win rate, verdict)
      - Equity curve chart (Chart.js line chart)
      - Drawdown chart (area chart)
      - Monthly returns heatmap (custom component)
      - Trade scatter plot (scatter chart)
      - Rolling Sharpe chart (line chart)

    Parameters
    ----------
    bt_report : BacktestReport from build_report()

    Returns
    -------
    Dict ready for json.dumps() / FastAPI's JSONResponse
    """
    return {
        # All numeric metrics (from metrics.py)
        "metrics":        report_to_dict(bt_report.report),

        # Chart data arrays (each is a dict with lists)
        "charts": {
            "equity":          bt_report.equity_data,
            "drawdown":        bt_report.drawdown_data,
            "monthly_returns": bt_report.monthly_data,
            "trades":          bt_report.trade_data,
            "rolling_sharpe":  bt_report.rolling_data,
        },

        # Summary fields for quick display
        "summary": {
            "ticker":           bt_report.report.ticker,
            "period":           (f"{bt_report.report.period_start.date()} → "
                                 f"{bt_report.report.period_end.date()}"),
            "total_return_pct": round(bt_report.report.total_return * 100, 2),
            "sharpe_ratio":     bt_report.report.sharpe_ratio,
            "max_drawdown_pct": round(bt_report.report.max_drawdown * 100, 2),
            "win_rate_pct":     round(bt_report.report.win_rate * 100, 1),
            "beats_benchmark":  bt_report.report.beats_benchmark,
            "verdict":          bt_report.report.verdict,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON UTILITIES — Multi-stock / multi-config comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_strategies(
    reports:  Dict[str, BacktestReport],
    metric:   str = "total_return",
    save_path: Optional[str] = None,
    show:      bool = False,
) -> Tuple[plt.Figure, Dict]:
    """
    Bar chart comparing a single metric across multiple strategies.

    Useful for comparing:
      - Same model on different stocks (which stocks work best?)
      - Same stock with different configs (which config wins?)
      - Different models on the same stock (model A vs model B)

    Parameters
    ----------
    reports   : Dict mapping label → BacktestReport
    metric    : Which metric to compare. Options:
                'total_return', 'sharpe_ratio', 'max_drawdown',
                'win_rate', 'profit_factor', 'alpha'
    save_path : Optional save location for PNG
    show      : Display immediately

    Returns
    -------
    (fig, chart_data) for frontend rendering
    """
    # ── Extract metric values ─────────────────────────────────────────────
    labels = list(reports.keys())

    # Map metric names to PerformanceReport attributes
    metric_map = {
        "total_return":  lambda r: r.report.total_return * 100,
        "sharpe_ratio":  lambda r: r.report.sharpe_ratio,
        "max_drawdown":  lambda r: r.report.max_drawdown * 100,
        "win_rate":      lambda r: r.report.win_rate * 100,
        "profit_factor": lambda r: r.report.profit_factor,
        "alpha":         lambda r: r.report.alpha * 100,
    }

    if metric not in metric_map:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Available: {list(metric_map.keys())}"
        )

    extractor = metric_map[metric]
    values    = [extractor(r) for r in reports.values()]

    chart_data = {
        "labels":  labels,
        "values":  [round(v, 2) for v in values],
        "metric":  metric,
    }

    # ── Create figure ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))

    colours = [COLOUR_WIN if v > 0 else COLOUR_LOSS for v in values]
    bars    = ax.bar(labels, values, color=colours, alpha=0.8,
                     edgecolor="white", linewidth=1)

    # ── Label each bar with its value ─────────────────────────────────────
    for bar, val in zip(bars, values):
        ypos = bar.get_height()
        va   = "bottom" if ypos >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.2f}", ha="center", va=va,
                fontsize=FONT_SIZE_TICK, fontweight="bold",
                color=bar.get_facecolor())

    # ── Styling ───────────────────────────────────────────────────────────
    metric_labels = {
        "total_return":  "Total Return (%)",
        "sharpe_ratio":  "Sharpe Ratio",
        "max_drawdown":  "Max Drawdown (%)",
        "win_rate":      "Win Rate (%)",
        "profit_factor": "Profit Factor",
        "alpha":         "Alpha (%/year)",
    }
    _apply_stocksense_style(
        ax, f"Strategy Comparison — {metric_labels[metric]}"
    )
    ax.set_ylabel(metric_labels[metric], fontsize=FONT_SIZE_LABEL)
    ax.axhline(y=0, color="#64748B", linewidth=0.5, zorder=1)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    if show:
        plt.show()

    plt.close(fig)
    return fig, chart_data


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from backtest.backtester import quick_backtest

    print("Running quick backtest for AAPL...")
    result = quick_backtest("AAPL", period="2y", verbose=False)

    print("Building full report...")
    report = build_report(result, save_dir="./figures", verbose=True)

    print("\nJSON output (summary):")
    json_data = report_to_json(report)
    for key, value in json_data["summary"].items():
        print(f"  {key}: {value}")

    print(f"\nChart data keys: {list(json_data['charts'].keys())}")
    print(f"Equity curve points: {len(json_data['charts']['equity']['dates'])}")
    print(f"Trade data points: {len(json_data['charts']['trades']['dates'])}")
