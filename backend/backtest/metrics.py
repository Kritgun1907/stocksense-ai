"""
StockSense AI — backtest/metrics.py
=====================================
Financial performance metrics for backtesting.

This file owns:
  - Portfolio-level metrics (Sharpe, Sortino, Calmar, max drawdown)
  - Trade-level metrics (win rate, profit factor, expectancy, streaks)
  - Benchmark-relative metrics (alpha, information ratio, beta)
  - Rolling metrics (rolling Sharpe, rolling drawdown)
  - Full performance report generation
  - Metric interpretation (human-readable labels)

It does NOT own:
  - Trade simulation        → backtest/engine.py
  - Backtest orchestration  → backtest/backtester.py
  - Visualisation           → backtest/visualiser.py (Chapter 5.4)
  - ML metrics              → models/evaluator.py

Why separate from evaluator.py?
─────────────────────────────────────────────────────────────
  evaluator.py computes metrics from prediction arrays.
  This module computes metrics from BacktestResult objects —
  which contain a full portfolio simulation with compounding,
  realistic costs, and a complete trade log.
  The distinction matters: a strategy with 53% accuracy can have
  a Sharpe of 1.2 (evaluator shows poor ML score, this shows
  good financial performance). Both perspectives are needed.

Why Sortino alongside Sharpe?
─────────────────────────────────────────────────────────────
  Sharpe penalises upside and downside volatility equally.
  A strategy with occasional large wins (good) is penalised
  identically to one with occasional large losses (bad).
  Sortino only penalises downside — more appropriate for
  asymmetric strategies like ours (long only, sit out on DOWN).

Why include rolling metrics?
─────────────────────────────────────────────────────────────
  A strategy with annual Sharpe 1.0 might have Sharpe 2.0 in
  bull markets and 0.1 in bear markets. Rolling Sharpe reveals
  this regime sensitivity — critical for deployment decisions.
  A stable rolling Sharpe is far more deployable than one with
  high variance across periods.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from backtest.engine import BacktestResult, Trade

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_DAILY  = 0.0   # simplified: 0% daily risk-free rate
                               # real: ~5% annual → 0.019% daily (negligible)

# Metric interpretation thresholds — mirrors SHAP_STRENGTH_LABELS pattern
SHARPE_THRESHOLDS = [
    (2.0,  "🟢 Excellent",  "Institutional quality — verify not overfitted"),
    (1.0,  "🟢 Good",       "Strong risk-adjusted return"),
    (0.5,  "🟡 Acceptable", "Marginal — worth deploying with caution"),
    (0.0,  "🟠 Poor",       "Negative risk-adjusted return"),
    (-999, "🔴 Negative",   "Strategy destroys value"),
]

DRAWDOWN_THRESHOLDS = [
    (-0.05,  "🟢 Low",      "< 5% — low risk strategy"),
    (-0.15,  "🟡 Moderate", "5-15% — manageable for most investors"),
    (-0.30,  "🟠 High",     "15-30% — significant psychological pressure"),
    (-999,   "🔴 Extreme",  "> 30% — most investors would stop trading"),
]

WIN_RATE_THRESHOLDS = [
    (0.60,  "🟢 Strong",     "Consistent edge demonstrated"),
    (0.55,  "🟡 Decent",     "Slight edge — needs profit factor > 1.2"),
    (0.50,  "🟠 Weak",       "Coin flip — profit factor must compensate"),
    (-999,  "🔴 Below random", "Worse than random — review model"),
]

# Minimum trades for statistically reliable metrics
MIN_TRADES = 20
ROLLING_WINDOW_DAYS = 60


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceReport:
    """
    Complete performance report for one backtest run.
    Passed to visualiser.py for charting and to FastAPI for the UI.

    Why a dataclass rather than a plain dict?
    ─────────────────────────────────────────────────────────────
    Performance reports are serialised to JSON for the API response.
    A dataclass makes the schema explicit and IDE-navigable.
    Fields match the StockSense UI "backtest performance" section.
    """
    # Identification
    ticker:              str
    period_start:        pd.Timestamp
    period_end:          pd.Timestamp
    n_trading_days:      int
    n_trades:            int
    trade_rate:          float           # fraction of days traded

    # Portfolio-level metrics
    total_return:        float           # decimal (0.15 = 15%)
    annualised_return:   float
    sharpe_ratio:        float
    sortino_ratio:       float
    calmar_ratio:        float
    max_drawdown:        float           # decimal, always negative
    max_drawdown_duration: int           # trading days
    volatility_annual:   float           # annualised daily std

    # Benchmark comparison
    benchmark_total_return:  float
    benchmark_sharpe:        float
    alpha:                   float       # excess annualised return
    information_ratio:       float
    beats_benchmark:         bool

    # Trade-level metrics
    win_rate:            float
    avg_win_pct:         float
    avg_loss_pct:        float
    win_loss_ratio:      float
    profit_factor:       float
    expectancy_pct:      float          # expected return per trade
    max_win_streak:      int
    max_loss_streak:     int
    avg_holding_days:    float

    # Rolling metrics
    rolling_sharpe:      pd.Series      # 60-day rolling Sharpe
    rolling_drawdown:    pd.Series      # running drawdown series

    # Interpretations (human-readable)
    sharpe_label:        str
    drawdown_label:      str
    win_rate_label:      str
    verdict:             str            # overall assessment


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _daily_returns(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Compute daily portfolio returns from portfolio value series.
    Uses percentage change: (V_t - V_{t-1}) / V_{t-1}.
    """
    values = np.array(portfolio_values, dtype=float)
    return np.diff(values) / (values[:-1] + 1e-10)


def _annualised_return(total_return: float, n_days: int) -> float:
    """
    Compound annualised return from total return over n trading days.

    Formula: (1 + total_return)^(252/n_days) - 1
    This is the CAGR (Compound Annual Growth Rate) applied to daily data.
    """
    if n_days == 0:
        return 0.0
    n_years = n_days / TRADING_DAYS_PER_YEAR
    return float((1 + total_return) ** (1 / n_years) - 1)


def _sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe ratio. Returns 0.0 if std is zero."""
    if len(returns) < 2:
        return 0.0
    std = np.std(returns)
    if std == 0:
        return 0.0
    return float((np.mean(returns) - RISK_FREE_RATE_DAILY) / std
                 * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(returns: np.ndarray, target: float = 0.0) -> float:
    """
    Sortino ratio — uses only downside volatility.

    Why target=0?
    ─────────────────────────────────────────────────────────────
    We penalise any negative return (below 0).
    Some implementations use the risk-free rate as target.
    For simplicity and consistency with Sharpe (Rf=0), we use 0.
    """
    downside = returns[returns < target]
    if len(downside) < 2:
        return 0.0
    downside_std = np.std(downside)
    if downside_std == 0:
        return float('inf')
    return float(np.mean(returns) / downside_std
                 * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(portfolio_values: np.ndarray) -> Tuple[float, int]:
    """
    Compute maximum drawdown and its duration in trading days.

    Returns
    -------
    (max_drawdown_fraction, max_duration_days) tuple.
    max_drawdown is always <= 0 (loss from peak).
    """
    values    = np.array(portfolio_values, dtype=float)
    peak      = np.maximum.accumulate(values)
    drawdowns = (values - peak) / (peak + 1e-10)
    max_dd    = float(drawdowns.min())

    # Duration: longest consecutive run below a previous peak
    in_dd   = (values < peak).astype(int)
    max_dur = 0
    current = 0
    for d in in_dd:
        current = (current + 1) if d else 0
        max_dur = max(max_dur, current)

    return max_dd, max_dur


def _calmar(annualised_return: float, max_drawdown: float) -> float:
    """
    Calmar ratio = annualised_return / |max_drawdown|.
    Returns 0 if max_drawdown is 0 (no drawdown experienced).
    """
    if max_drawdown == 0:
        return 0.0
    return float(annualised_return / abs(max_drawdown))


def _information_ratio(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """
    Information ratio = mean(active returns) / std(active returns)
    Active returns = strategy_returns - benchmark_returns.

    Measures whether active management adds value over passive holding.
    IR > 0.5 is considered good active management.
    """
    if len(strategy_returns) != len(benchmark_returns):
        n   = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns  = strategy_returns[:n]
        benchmark_returns = benchmark_returns[:n]

    active = strategy_returns - benchmark_returns
    if np.std(active) == 0:
        return 0.0
    return float(np.mean(active) / np.std(active)
                 * np.sqrt(TRADING_DAYS_PER_YEAR))


def _label_metric(value: float, thresholds: list) -> str:
    """Look up a human-readable label for a metric value."""
    for threshold, label, _ in thresholds:
        if value >= threshold:
            return label
    return thresholds[-1][1]


def _describe_metric(value: float, thresholds: list) -> str:
    """Look up a human-readable description for a metric value."""
    for threshold, _, description in thresholds:
        if value >= threshold:
            return description
    return thresholds[-1][2]


# ══════════════════════════════════════════════════════════════════════════════
#  TRADE-LEVEL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _trade_metrics(trades: List[Trade]) -> Dict:
    """
    Compute all trade-level statistics from a list of Trade objects.

    Why compute from Trade objects rather than returns array?
    ─────────────────────────────────────────────────────────────
    The Trade object contains per-trade direction, gross/net P&L,
    and entry/exit dates — needed for streak analysis and holding
    period calculation. A simple returns array loses this information.
    """
    if not trades:
        return {
            "win_rate":       0.0,
            "avg_win_pct":    0.0,
            "avg_loss_pct":   0.0,
            "win_loss_ratio": 0.0,
            "profit_factor":  0.0,
            "expectancy_pct": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "avg_holding_days": 0.0,
        }

    returns = [t.return_pct for t in trades]
    wins    = [r for r in returns if r > 0]
    losses  = [r for r in returns if r < 0]
    n       = len(returns)

    win_rate      = len(wins) / n
    avg_win       = float(np.mean(wins))    if wins   else 0.0
    avg_loss      = float(np.mean(losses))  if losses else 0.0  # negative
    win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')

    # Profit factor: total gross profit / total gross loss
    total_profit = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
    total_loss   = sum(abs(t.gross_pnl) for t in trades if t.gross_pnl < 0)
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    # Expectancy: expected P&L per trade
    # = win_rate × avg_win + loss_rate × avg_loss
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Win/loss streaks
    is_win = [1 if r > 0 else 0 for r in returns]
    max_win_streak  = _max_streak(is_win, target=1)
    max_loss_streak = _max_streak(is_win, target=0)

    # Average holding period (entry to exit)
    holding_days = []
    for trade in trades:
        delta = (trade.exit_date - trade.entry_date).days
        holding_days.append(max(delta, 1))   # minimum 1 day

    return {
        "win_rate":         round(win_rate,       4),
        "avg_win_pct":      round(avg_win,        4),
        "avg_loss_pct":     round(avg_loss,       4),
        "win_loss_ratio":   round(win_loss_ratio, 4),
        "profit_factor":    round(min(profit_factor, 999.0), 4),
        "expectancy_pct":   round(expectancy,     4),
        "max_win_streak":   max_win_streak,
        "max_loss_streak":  max_loss_streak,
        "avg_holding_days": round(float(np.mean(holding_days)), 1),
    }


def _max_streak(binary_list: List[int], target: int) -> int:
    """Count the longest consecutive run of target values."""
    max_run = 0
    current = 0
    for val in binary_list:
        current = (current + 1) if val == target else 0
        max_run = max(max_run, current)
    return max_run


# ══════════════════════════════════════════════════════════════════════════════
#  ROLLING METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _rolling_sharpe(
    portfolio_values: pd.Series,
    window:           int = ROLLING_WINDOW_DAYS,
) -> pd.Series:
    """
    Compute rolling Sharpe ratio over a sliding window.

    Why 60 days (default)?
    ─────────────────────────────────────────────────────────────
    Shorter windows (20d) are too noisy — Sharpe fluctuates wildly.
    Longer windows (120d) are too slow to detect regime changes.
    60 days (3 calendar months) balances responsiveness and stability.
    """
    returns = portfolio_values.pct_change().fillna(0)
    roll    = returns.rolling(window, min_periods=window // 2)
    sharpe  = (roll.mean() / (roll.std() + 1e-10)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe.fillna(0)


def _running_drawdown(portfolio_values: pd.Series) -> pd.Series:
    """
    Compute the running drawdown series (drawdown at each point in time).
    Used for the drawdown chart on the stock page.
    """
    peak      = portfolio_values.cummax()
    drawdown  = (portfolio_values - peak) / (peak + 1e-10)
    return drawdown


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    result:  BacktestResult,
    verbose: bool = True,
) -> PerformanceReport:
    """
    Compute the full performance report from a BacktestResult.

    This is the primary function called after backtester.py
    generates a BacktestResult.

    Parameters
    ----------
    result  : BacktestResult from backtest/engine.py.
    verbose : Print formatted performance report.

    Returns
    -------
    PerformanceReport dataclass with all metrics and interpretations.

    Raises
    ------
    ValueError : If result has fewer than MIN_TRADES trades.
    """
    if result.n_trades < MIN_TRADES:
        raise ValueError(
            f"Backtest has only {result.n_trades} trades "
            f"(minimum {MIN_TRADES} required for reliable metrics). "
            f"Consider extending the test period or lowering the "
            f"classification threshold."
        )

    # ── Portfolio series
    pv     = result.daily_portfolio["portfolio_value"]
    bv     = result.benchmark_portfolio["portfolio_value"]
    pv_arr = pv.values
    bv_arr = bv.values

    # ── Returns
    strategy_returns  = _daily_returns(pv_arr)
    benchmark_returns = _daily_returns(bv_arr)

    n = min(len(strategy_returns), len(benchmark_returns))
    strategy_returns  = strategy_returns[:n]
    benchmark_returns = benchmark_returns[:n]

    # ── Core portfolio metrics
    initial      = pv_arr[0]
    final        = pv_arr[-1]
    total_ret    = (final - initial) / initial
    n_days       = len(pv_arr)
    ann_ret      = _annualised_return(total_ret, n_days)
    sharpe       = _sharpe(strategy_returns)
    sortino      = _sortino(strategy_returns)
    max_dd, max_dd_dur = _max_drawdown(pv_arr)
    calmar       = _calmar(ann_ret, max_dd)
    vol_annual   = float(np.std(strategy_returns) * np.sqrt(TRADING_DAYS_PER_YEAR))

    # ── Benchmark metrics
    bh_initial   = bv_arr[0]
    bh_final     = bv_arr[-1]
    bh_total_ret = (bh_final - bh_initial) / bh_initial
    bh_ann_ret   = _annualised_return(bh_total_ret, len(bv_arr))
    bh_sharpe    = _sharpe(benchmark_returns)
    alpha        = ann_ret - bh_ann_ret
    info_ratio   = _information_ratio(strategy_returns, benchmark_returns)
    beats_bh     = sharpe > bh_sharpe

    # ── Trade-level metrics
    trade_stats  = _trade_metrics(result.trades)
    trade_rate   = result.n_trades / max(n_days - 1, 1)

    # ── Rolling metrics
    rolling_sh   = _rolling_sharpe(pv)
    running_dd   = _running_drawdown(pv)

    # ── Interpretations
    sharpe_label    = _label_metric(sharpe,    SHARPE_THRESHOLDS)
    drawdown_label  = _label_metric(max_dd,    DRAWDOWN_THRESHOLDS)
    win_rate_label  = _label_metric(
        trade_stats["win_rate"], WIN_RATE_THRESHOLDS
    )

    # ── Verdict
    verdict = _build_verdict(
        sharpe, max_dd, trade_stats["profit_factor"],
        beats_bh, info_ratio
    )

    report = PerformanceReport(
        # Identification
        ticker=result.ticker,
        period_start=result.date_range[0],
        period_end=result.date_range[1],
        n_trading_days=n_days,
        n_trades=result.n_trades,
        trade_rate=round(trade_rate, 4),
        # Portfolio-level
        total_return=round(total_ret,   4),
        annualised_return=round(ann_ret, 4),
        sharpe_ratio=round(sharpe,      4),
        sortino_ratio=round(sortino,    4),
        calmar_ratio=round(calmar,      4),
        max_drawdown=round(max_dd,      4),
        max_drawdown_duration=max_dd_dur,
        volatility_annual=round(vol_annual, 4),
        # Benchmark
        benchmark_total_return=round(bh_total_ret, 4),
        benchmark_sharpe=round(bh_sharpe,          4),
        alpha=round(alpha,                          4),
        information_ratio=round(info_ratio,         4),
        beats_benchmark=beats_bh,
        # Trade-level
        win_rate=trade_stats["win_rate"],
        avg_win_pct=trade_stats["avg_win_pct"],
        avg_loss_pct=trade_stats["avg_loss_pct"],
        win_loss_ratio=trade_stats["win_loss_ratio"],
        profit_factor=trade_stats["profit_factor"],
        expectancy_pct=trade_stats["expectancy_pct"],
        max_win_streak=trade_stats["max_win_streak"],
        max_loss_streak=trade_stats["max_loss_streak"],
        avg_holding_days=trade_stats["avg_holding_days"],
        # Rolling
        rolling_sharpe=rolling_sh,
        rolling_drawdown=running_dd,
        # Interpretations
        sharpe_label=sharpe_label,
        drawdown_label=drawdown_label,
        win_rate_label=win_rate_label,
        verdict=verdict,
    )

    if verbose:
        _print_report(report)

    return report


def _build_verdict(
    sharpe:        float,
    max_dd:        float,
    profit_factor: float,
    beats_bh:      bool,
    info_ratio:    float,
) -> str:
    """
    Build a one-sentence verdict on overall strategy quality.
    Used on the StockSense stock page backtest section.
    """
    checks_passed = 0
    total_checks  = 5

    if sharpe > 0.5:         checks_passed += 1
    if max_dd > -0.15:       checks_passed += 1
    if profit_factor > 1.2:  checks_passed += 1
    if beats_bh:             checks_passed += 1
    if info_ratio > 0.3:     checks_passed += 1

    rate = checks_passed / total_checks

    if rate >= 0.80:
        return (f"🟢 Strong performance — strategy passes {checks_passed}/{total_checks} "
                f"quality checks. Suitable for paper trading.")
    elif rate >= 0.60:
        return (f"🟡 Acceptable — {checks_passed}/{total_checks} checks passed. "
                f"Consider further tuning before deployment.")
    elif rate >= 0.40:
        return (f"🟠 Weak — only {checks_passed}/{total_checks} checks passed. "
                f"Strategy needs improvement.")
    else:
        return (f"🔴 Poor — {checks_passed}/{total_checks} checks passed. "
                f"Strategy does not demonstrate consistent edge.")


def _print_report(report: PerformanceReport) -> None:
    """Print a formatted performance report to console."""
    beat = "✅" if report.beats_benchmark else "❌"

    print(f"\n{'═'*65}")
    print(f"Performance Report: {report.ticker}")
    print(f"  Period: {report.period_start.date()} → "
          f"{report.period_end.date()} "
          f"({report.n_trading_days} days)")
    print(f"{'═'*65}")

    print(f"\n── Portfolio Metrics ───────────────────────────────────────")
    print(f"  {'Total Return':<28} {report.total_return*100:>+10.2f}%")
    print(f"  {'Annualised Return':<28} {report.annualised_return*100:>+10.2f}%")
    print(f"  {'Volatility (annual)':<28} {report.volatility_annual*100:>10.2f}%")
    print(f"  {'Sharpe Ratio':<28} {report.sharpe_ratio:>11.3f}  "
          f"{report.sharpe_label}")
    print(f"  {'Sortino Ratio':<28} {report.sortino_ratio:>11.3f}")
    print(f"  {'Calmar Ratio':<28} {report.calmar_ratio:>11.3f}")
    print(f"  {'Max Drawdown':<28} {report.max_drawdown*100:>10.2f}%  "
          f"{report.drawdown_label}")
    print(f"  {'Max Drawdown Duration':<28} {report.max_drawdown_duration:>9}d")

    print(f"\n── Benchmark Comparison ────────────────────────────────────")
    print(f"  {'Benchmark Return':<28} {report.benchmark_total_return*100:>+10.2f}%")
    print(f"  {'Benchmark Sharpe':<28} {report.benchmark_sharpe:>11.3f}")
    print(f"  {'Alpha':<28} {report.alpha*100:>+10.2f}%/year")
    print(f"  {'Information Ratio':<28} {report.information_ratio:>11.3f}")
    print(f"  {'Beats Benchmark':<28} {'Yes' if report.beats_benchmark else 'No':>11}  "
          f"{beat}")

    print(f"\n── Trade Statistics ─────────────────────────────────────────")
    print(f"  {'Trades':<28} {report.n_trades:>11}  "
          f"({report.trade_rate*100:.1f}% of days)")
    print(f"  {'Win Rate':<28} {report.win_rate*100:>10.1f}%  "
          f"{report.win_rate_label}")
    print(f"  {'Avg Win':<28} {report.avg_win_pct*100:>+10.2f}%")
    print(f"  {'Avg Loss':<28} {report.avg_loss_pct*100:>+10.2f}%")
    print(f"  {'Win/Loss Ratio':<28} {report.win_loss_ratio:>11.3f}")
    print(f"  {'Profit Factor':<28} {report.profit_factor:>11.3f}")
    print(f"  {'Expectancy per Trade':<28} {report.expectancy_pct*100:>+10.3f}%")
    print(f"  {'Max Win Streak':<28} {report.max_win_streak:>11}")
    print(f"  {'Max Loss Streak':<28} {report.max_loss_streak:>11}")
    print(f"  {'Avg Holding Period':<28} {report.avg_holding_days:>10.1f}d")

    print(f"\n{'═'*65}")
    print(f"VERDICT: {report.verdict}")
    print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-STOCK AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_multi_stock_metrics(
    reports:  Dict[str, PerformanceReport],
    verbose:  bool = True,
) -> Dict:
    """
    Aggregate performance reports across multiple stocks.

    Used to evaluate the universal model's performance across
    all stocks in the screener — not just one.

    Parameters
    ----------
    reports : Dict mapping ticker → PerformanceReport.
    verbose : Print aggregated summary.

    Returns
    -------
    Dict with median, mean, std, and win rate of each key metric
    across all stocks.

    Why median rather than mean as the primary metric?
    ─────────────────────────────────────────────────────────────
    One stock with an extreme Sharpe (good or bad) skews the mean.
    Median is robust to outliers — better represents the typical stock.
    Report both so the reader can detect extreme outliers.
    """
    if not reports:
        return {}

    # Collect key metrics across all stocks
    sharpes    = [r.sharpe_ratio    for r in reports.values()]
    max_dds    = [r.max_drawdown    for r in reports.values()]
    win_rates  = [r.win_rate        for r in reports.values()]
    alphas     = [r.alpha           for r in reports.values()]
    p_factors  = [r.profit_factor   for r in reports.values()
                  if r.profit_factor < 999]   # exclude infinite
    beats_bh   = [r.beats_benchmark for r in reports.values()]

    agg = {
        "n_stocks":             len(reports),
        "stocks_beat_benchmark": sum(beats_bh),
        "beat_rate":            round(sum(beats_bh) / len(beats_bh), 4),

        "sharpe_median":        round(float(np.median(sharpes)),   4),
        "sharpe_mean":          round(float(np.mean(sharpes)),     4),
        "sharpe_std":           round(float(np.std(sharpes)),      4),
        "sharpe_min":           round(float(np.min(sharpes)),      4),
        "sharpe_max":           round(float(np.max(sharpes)),      4),

        "max_dd_median":        round(float(np.median(max_dds)),   4),
        "max_dd_worst":         round(float(np.min(max_dds)),      4),

        "win_rate_median":      round(float(np.median(win_rates)), 4),
        "win_rate_mean":        round(float(np.mean(win_rates)),   4),

        "alpha_median":         round(float(np.median(alphas)),    4),
        "alpha_mean":           round(float(np.mean(alphas)),      4),

        "profit_factor_median": round(float(np.median(p_factors)), 4)
                                if p_factors else 0.0,
    }

    if verbose:
        beat_pct = agg["beat_rate"] * 100
        print(f"\n{'═'*60}")
        print(f"Multi-Stock Backtest Summary ({agg['n_stocks']} stocks)")
        print(f"{'═'*60}")
        print(f"  Beats buy-and-hold:  "
              f"{agg['stocks_beat_benchmark']}/{agg['n_stocks']} "
              f"({beat_pct:.1f}%)")
        print(f"\n  {'Metric':<22} {'Median':>8} {'Mean':>8} {'Std':>8}")
        print(f"  {'─'*52}")
        print(f"  {'Sharpe Ratio':<22} "
              f"{agg['sharpe_median']:>8.3f} "
              f"{agg['sharpe_mean']:>8.3f} "
              f"{agg['sharpe_std']:>8.3f}")
        print(f"  {'Max Drawdown':<22} "
              f"{agg['max_dd_median']*100:>7.1f}%  "
              f"(worst: {agg['max_dd_worst']*100:.1f}%)")
        print(f"  {'Win Rate':<22} "
              f"{agg['win_rate_median']*100:>7.1f}%  "
              f"{agg['win_rate_mean']*100:>7.1f}%")
        print(f"  {'Alpha (annual)':<22} "
              f"{agg['alpha_median']*100:>+7.1f}%  "
              f"{agg['alpha_mean']*100:>+7.1f}%")
        print(f"  {'Profit Factor':<22} "
              f"{agg['profit_factor_median']:>8.3f}")
        print(f"{'═'*60}\n")

    return agg


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT TO DICT (for JSON serialisation in FastAPI)
# ══════════════════════════════════════════════════════════════════════════════

def report_to_dict(report: PerformanceReport) -> Dict:
    """
    Convert PerformanceReport to a JSON-serialisable dict.

    Called by the FastAPI endpoint GET /backtest?ticker=AAPL
    to return backtest results to the frontend.

    Excludes rolling_sharpe and rolling_drawdown Series
    (these are returned as separate arrays for charting).
    """
    return {
        "ticker":                 report.ticker,
        "period_start":           str(report.period_start.date()),
        "period_end":             str(report.period_end.date()),
        "n_trading_days":         report.n_trading_days,
        "n_trades":               report.n_trades,
        "trade_rate":             report.trade_rate,
        "total_return_pct":       round(report.total_return * 100, 2),
        "annualised_return_pct":  round(report.annualised_return * 100, 2),
        "sharpe_ratio":           report.sharpe_ratio,
        "sortino_ratio":          report.sortino_ratio,
        "calmar_ratio":           report.calmar_ratio,
        "max_drawdown_pct":       round(report.max_drawdown * 100, 2),
        "max_drawdown_duration":  report.max_drawdown_duration,
        "volatility_annual_pct":  round(report.volatility_annual * 100, 2),
        "benchmark_return_pct":   round(report.benchmark_total_return * 100, 2),
        "benchmark_sharpe":       report.benchmark_sharpe,
        "alpha_pct":              round(report.alpha * 100, 2),
        "information_ratio":      report.information_ratio,
        "beats_benchmark":        report.beats_benchmark,
        "win_rate_pct":           round(report.win_rate * 100, 1),
        "avg_win_pct":            round(report.avg_win_pct * 100, 2),
        "avg_loss_pct":           round(report.avg_loss_pct * 100, 2),
        "win_loss_ratio":         report.win_loss_ratio,
        "profit_factor":          report.profit_factor,
        "expectancy_pct":         round(report.expectancy_pct * 100, 3),
        "max_win_streak":         report.max_win_streak,
        "max_loss_streak":        report.max_loss_streak,
        "avg_holding_days":       report.avg_holding_days,
        "sharpe_label":           report.sharpe_label,
        "drawdown_label":         report.drawdown_label,
        "win_rate_label":         report.win_rate_label,
        "verdict":                report.verdict,
        "rolling_sharpe":         report.rolling_sharpe.fillna(0).tolist(),
        "rolling_drawdown":       report.rolling_drawdown.fillna(0).tolist(),
        "rolling_dates":          [str(d.date()) for d in report.rolling_sharpe.index],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from backtest.backtester import quick_backtest

    print("Running quick backtest for AAPL...")
    result = quick_backtest("AAPL", period="2y", verbose=False)

    print("Computing metrics...")
    report = compute_metrics(result, verbose=True)

    print("\nJSON-serialisable dict (first 5 keys):")
    d = report_to_dict(report)
    for k, v in list(d.items())[:5]:
        print(f"  {k}: {v}")
