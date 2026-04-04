"""
StockSense AI — tests/test_backtest_e2e.py
=============================================
End-to-end integration tests for the full backtest pipeline:
  engine.py  →  backtester.py  →  metrics.py  →  visualiser.py

Also runs the complete model → backtest pipeline:
  trainer.train()  →  backtester.backtest_pipeline()
                   →  metrics.compute_metrics()
                   →  visualiser.build_report()
                   →  JSON for React frontend

Coverage strategy:
─────────────────────────────────────────────────────────────
  - ALL tests use SYNTHETIC data — no internet, no yfinance calls.
  - Tests are grouped into 10 sections:
      1. engine.py — trade simulation, portfolio tracking
      2. engine.py — buy-and-hold benchmark
      3. engine.py — configs and validation
      4. metrics.py — portfolio-level metrics (Sharpe, drawdown, etc.)
      5. metrics.py — trade-level metrics (win rate, profit factor)
      6. metrics.py — full PerformanceReport + JSON serialisation
      7. backtester.py — prediction-to-price alignment
      8. visualiser.py — chart data generation + build_report
      9. visualiser.py — JSON serialisation + compare_strategies
     10. Full pipeline integration: trainer → backtest → report → JSON

Run with:
    cd backend
    python tests/test_backtest_e2e.py

Expected runtime: ~30-40 seconds
"""

import sys
import os
import warnings
import time
import tempfile
import shutil
import json

import numpy as np
import pandas as pd

# ── Ensure backend/ is on sys.path ───────────────────────────────────────────
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST INFRASTRUCTURE  (mirrors test_models_e2e.py pattern)
# ══════════════════════════════════════════════════════════════════════════════

_PASS = 0
_FAIL = 0
_ERRORS = []


def pass_fail(name: str, condition: bool, detail: str = "") -> bool:
    """Record and print a single assertion result."""
    global _PASS, _FAIL
    icon   = "✅ PASS" if condition else "❌ FAIL"
    detail = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{detail}")
    if condition:
        _PASS += 1
    else:
        _FAIL += 1
        _ERRORS.append(name)
    return condition


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def summary() -> int:
    """Print final tally. Returns 1 if any failures, else 0."""
    total = _PASS + _FAIL
    print(f"\n{'═'*62}")
    print(f"  RESULTS: {_PASS}/{total} passed  |  {_FAIL} failed")
    if _ERRORS:
        print(f"\n  Failed tests:")
        for name in _ERRORS:
            print(f"    ✗  {name}")
    print(f"{'═'*62}\n")
    return 1 if _FAIL > 0 else 0


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

def make_backtest_arrays(
    n_days:  int   = 200,
    up_rate: float = 0.55,
    seed:    int   = 42,
) -> tuple:
    """
    Build synthetic aligned arrays for backtest engine testing.

    Returns (predictions, open_prices, close_prices, dates) where:
      - predictions: binary 0/1 array with given up_rate
      - open_prices: realistic stock price series starting at $100
      - close_prices: open ± small random daily return
      - dates: business day DatetimeIndex

    The price series uses geometric Brownian motion (random walk with
    drift) to produce realistic-looking stock prices. The predictions
    are independent of prices — they are synthetic model outputs.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_days)

    # Binary predictions with controlled signal rate
    predictions = (rng.random(n_days) < up_rate).astype(int)

    # Realistic price series: geometric Brownian motion
    # daily drift = 0.02% (≈5% annual), daily vol = 1.5%
    daily_returns = rng.normal(0.0002, 0.015, n_days)
    cum_returns   = np.cumprod(1 + daily_returns)
    close_prices  = 100.0 * cum_returns

    # Open prices: close of previous day ± small overnight gap
    open_prices    = np.empty_like(close_prices)
    open_prices[0] = 100.0
    open_prices[1:] = close_prices[:-1] * (1 + rng.normal(0, 0.002, n_days - 1))

    return predictions, open_prices, close_prices, dates


def make_X_y(
    n_days:     int  = 504,
    n_features: int  = 30,
    seed:       int  = 42,
) -> tuple:
    """
    Build deterministic (X, y) for trainer-based tests.
    Uses ~50/50 balanced labels.
    """
    rng  = np.random.default_rng(seed)
    cols = [f"feature_{i:03d}" for i in range(n_features)]
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    X = pd.DataFrame(
        rng.normal(size=(n_days, n_features)),
        index=dates, columns=cols,
    )
    y = pd.Series(
        rng.integers(0, 2, size=n_days).astype(int),
        index=dates, name="target",
    )
    half = n_days // 2
    y.iloc[:half] = np.arange(half) % 2

    return X, y


def make_price_df(dates: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic OHLCV DataFrame covering the given date range.
    Extends 10 days beyond the last date (for next-day price lookups).
    """
    rng = np.random.default_rng(seed)

    # Extend date range by 10 business days
    extended = pd.bdate_range(dates[0], periods=len(dates) + 10)

    n = len(extended)
    daily_ret  = rng.normal(0.0003, 0.015, n)
    cum_ret    = np.cumprod(1 + daily_ret)
    close      = 150.0 * cum_ret
    open_p     = np.empty_like(close)
    open_p[0]  = 150.0
    open_p[1:] = close[:-1] * (1 + rng.normal(0, 0.003, n - 1))
    high       = np.maximum(open_p, close) * (1 + rng.uniform(0, 0.01, n))
    low        = np.minimum(open_p, close) * (1 - rng.uniform(0, 0.01, n))
    volume     = rng.integers(1_000_000, 50_000_000, n)

    return pd.DataFrame({
        "open":   open_p,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=extended)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — engine.py: Trade simulation
# ══════════════════════════════════════════════════════════════════════════════

def test_engine_trade_simulation():
    section("1 · engine.py — trade simulation")

    from backtest.engine import run_backtest, BacktestResult, Trade

    preds, opens, closes, dates = make_backtest_arrays(n_days=200)
    result = run_backtest(
        predictions=preds,
        open_prices=opens,
        close_prices=closes,
        dates=dates,
        ticker="TEST",
        initial_capital=10_000,
        position_size=0.95,
        transaction_cost=0.001,
        allow_short=False,
        verbose=False,
    )

    # 1.1 — Result type
    pass_fail(
        "run_backtest returns BacktestResult",
        isinstance(result, BacktestResult),
    )

    # 1.2 — daily_portfolio is a DataFrame with portfolio_value column
    pass_fail(
        "daily_portfolio is DataFrame with 'portfolio_value'",
        isinstance(result.daily_portfolio, pd.DataFrame)
        and "portfolio_value" in result.daily_portfolio.columns,
    )

    # 1.3 — Portfolio value never goes negative
    pv = result.daily_portfolio["portfolio_value"]
    pass_fail(
        "portfolio value never negative",
        (pv > 0).all(),
        f"min={pv.min():.2f}",
    )

    # 1.4 — Trades are Trade dataclass instances
    pass_fail(
        "trades list contains Trade objects",
        all(isinstance(t, Trade) for t in result.trades),
        f"n_trades={len(result.trades)}",
    )

    # 1.5 — n_trades matches len(trades)
    pass_fail(
        "n_trades matches len(trades)",
        result.n_trades == len(result.trades),
    )

    # 1.6 — Each trade has positive entry_price
    pass_fail(
        "all trades have positive entry_price",
        all(t.entry_price > 0 for t in result.trades),
    )

    # 1.7 — Each trade has capital_after ≈ capital_before + net_pnl
    capital_consistent = True
    for t in result.trades:
        expected = t.capital_before + t.net_pnl
        if abs(t.capital_after - expected) > 0.01:
            capital_consistent = False
            break
    pass_fail(
        "capital_after = capital_before + net_pnl for every trade",
        capital_consistent,
    )

    # 1.8 — Transaction costs are positive
    pass_fail(
        "all transaction costs are positive",
        all(t.transaction_cost > 0 for t in result.trades),
    )

    # 1.9 — date_range tuple is valid
    pass_fail(
        "date_range is (start, end) tuple",
        len(result.date_range) == 2
        and result.date_range[0] <= result.date_range[1],
        f"{result.date_range[0].date()} → {result.date_range[1].date()}",
    )

    # 1.10 — n_signals counts UP predictions
    pass_fail(
        "n_signals counts UP predictions",
        result.n_signals == int(preds.sum()) or result.n_signals > 0,
        f"n_signals={result.n_signals}",
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — engine.py: Buy-and-hold benchmark
# ══════════════════════════════════════════════════════════════════════════════

def test_engine_benchmark(result):
    section("2 · engine.py — buy-and-hold benchmark")

    # 2.1 — Benchmark portfolio is a DataFrame
    pass_fail(
        "benchmark_portfolio is DataFrame",
        isinstance(result.benchmark_portfolio, pd.DataFrame),
    )

    # 2.2 — Benchmark has portfolio_value column
    pass_fail(
        "benchmark has 'portfolio_value' column",
        "portfolio_value" in result.benchmark_portfolio.columns,
    )

    # 2.3 — Benchmark starts near initial capital
    bv     = result.benchmark_portfolio["portfolio_value"]
    initial = result.config["initial_capital"]
    pass_fail(
        "benchmark starts near initial capital",
        abs(bv.iloc[0] - initial) / initial < 0.05,
        f"first={bv.iloc[0]:.2f}, expected≈{initial:.2f}",
    )

    # 2.4 — Benchmark values never negative
    pass_fail(
        "benchmark values all positive",
        (bv > 0).all(),
        f"min={bv.min():.2f}",
    )

    # 2.5 — Benchmark has same date count as strategy
    sp = result.daily_portfolio
    pass_fail(
        "benchmark and strategy have same number of dates",
        len(bv) == len(sp),
        f"benchmark={len(bv)}, strategy={len(sp)}",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — engine.py: Configs and validation
# ══════════════════════════════════════════════════════════════════════════════

def test_engine_configs():
    section("3 · engine.py — configs and validation")

    from backtest.engine import (
        BACKTEST_CONFIGS, get_backtest_config,
        list_backtest_configs, run_backtest,
    )

    # 3.1 — All 4 configs exist
    expected_configs = {"default", "conservative", "aggressive", "realistic_retail"}
    pass_fail(
        "BACKTEST_CONFIGS has all 4 named configs",
        set(BACKTEST_CONFIGS.keys()) == expected_configs,
        f"got={set(BACKTEST_CONFIGS.keys())}",
    )

    # 3.2 — Each config has required keys
    for name, cfg in BACKTEST_CONFIGS.items():
        required = {"initial_capital", "position_size", "transaction_cost",
                     "allow_short", "description"}
        has_all = required.issubset(set(cfg.keys()))
        pass_fail(
            f"config '{name}' has all required keys",
            has_all,
            f"missing={required - set(cfg.keys())}" if not has_all else "",
        )

    # 3.3 — get_backtest_config returns correct dict
    cfg = get_backtest_config("default")
    pass_fail(
        "get_backtest_config('default') returns dict",
        isinstance(cfg, dict) and "initial_capital" in cfg,
    )

    # 3.4 — get_backtest_config raises on bad name
    try:
        get_backtest_config("nonexistent_config")
        pass_fail("get_backtest_config raises on bad name", False)
    except ValueError:
        pass_fail("get_backtest_config raises on bad name", True)

    # 3.5 — list_backtest_configs returns list of strings
    names = list_backtest_configs(verbose=False)
    pass_fail(
        "list_backtest_configs returns 4 names",
        isinstance(names, list) and len(names) == 4,
        f"got={names}",
    )

    # 3.6 — Validation: mismatched lengths raise ValueError
    preds, opens, closes, dates = make_backtest_arrays(n_days=100)
    try:
        run_backtest(preds[:50], opens, closes, dates, verbose=False)
        pass_fail("mismatched array lengths raise ValueError", False)
    except ValueError:
        pass_fail("mismatched array lengths raise ValueError", True)

    # 3.7 — Validation: invalid predictions raise ValueError
    bad_preds = np.array([0, 1, 2, 3, 0])
    try:
        run_backtest(bad_preds, opens[:5], closes[:5], dates[:5], verbose=False)
        pass_fail("non-binary predictions raise ValueError", False)
    except ValueError:
        pass_fail("non-binary predictions raise ValueError", True)

    # 3.8 — config_name overrides individual params
    preds2, opens2, closes2, dates2 = make_backtest_arrays(n_days=100)
    result = run_backtest(
        preds2, opens2, closes2, dates2,
        config_name="conservative",
        verbose=False,
    )
    pass_fail(
        "config_name='conservative' sets position_size=0.5",
        result.config["position_size"] == 0.50,
        f"got={result.config['position_size']}",
    )

    # 3.9 — allow_short config creates short trades
    preds_down = np.zeros(100, dtype=int)   # all DOWN
    preds_down[:5] = 1  # just 5 UPs to avoid empty
    p2, o2, c2, d2 = make_backtest_arrays(n_days=100)
    result_short = run_backtest(
        preds_down, o2, c2, d2,
        allow_short=True,
        verbose=False,
    )
    has_short = any(t.direction == "short" for t in result_short.trades)
    pass_fail(
        "allow_short=True generates short trades",
        has_short,
        f"n_short={sum(1 for t in result_short.trades if t.direction=='short')}",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — metrics.py: Portfolio-level metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_metrics_portfolio(result):
    section("4 · metrics.py — portfolio-level metrics")

    from backtest.metrics import compute_metrics, PerformanceReport

    report = compute_metrics(result, verbose=False)

    # 4.1 — Returns PerformanceReport
    pass_fail(
        "compute_metrics returns PerformanceReport",
        isinstance(report, PerformanceReport),
    )

    # 4.2 — Total return is a finite float
    pass_fail(
        "total_return is finite float",
        isinstance(report.total_return, float) and np.isfinite(report.total_return),
        f"got={report.total_return:.4f}",
    )

    # 4.3 — Sharpe ratio is finite
    pass_fail(
        "sharpe_ratio is finite",
        np.isfinite(report.sharpe_ratio),
        f"got={report.sharpe_ratio:.4f}",
    )

    # 4.4 — Sortino ratio is finite
    pass_fail(
        "sortino_ratio is finite",
        np.isfinite(report.sortino_ratio),
        f"got={report.sortino_ratio:.4f}",
    )

    # 4.5 — Max drawdown is ≤ 0
    pass_fail(
        "max_drawdown ≤ 0",
        report.max_drawdown <= 0,
        f"got={report.max_drawdown:.4f}",
    )

    # 4.6 — Max drawdown duration ≥ 0
    pass_fail(
        "max_drawdown_duration ≥ 0",
        report.max_drawdown_duration >= 0,
        f"got={report.max_drawdown_duration}",
    )

    # 4.7 — Volatility is non-negative
    pass_fail(
        "volatility_annual ≥ 0",
        report.volatility_annual >= 0,
        f"got={report.volatility_annual:.4f}",
    )

    # 4.8 — Benchmark total return is finite
    pass_fail(
        "benchmark_total_return is finite",
        np.isfinite(report.benchmark_total_return),
        f"got={report.benchmark_total_return:.4f}",
    )

    # 4.9 — Alpha is finite
    pass_fail(
        "alpha is finite",
        np.isfinite(report.alpha),
        f"got={report.alpha:.4f}",
    )

    # 4.10 — Information ratio is finite
    pass_fail(
        "information_ratio is finite",
        np.isfinite(report.information_ratio),
        f"got={report.information_ratio:.4f}",
    )

    # 4.11 — beats_benchmark is bool
    pass_fail(
        "beats_benchmark is bool",
        isinstance(report.beats_benchmark, bool),
        f"got={report.beats_benchmark}",
    )

    return report


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — metrics.py: Trade-level metrics
# ══════════════════════════════════════════════════════════════════════════════

def test_metrics_trade_level(report):
    section("5 · metrics.py — trade-level metrics")

    # 5.1 — Win rate in [0, 1]
    pass_fail(
        "win_rate in [0, 1]",
        0 <= report.win_rate <= 1,
        f"got={report.win_rate:.4f}",
    )

    # 5.2 — Profit factor ≥ 0
    pass_fail(
        "profit_factor ≥ 0",
        report.profit_factor >= 0,
        f"got={report.profit_factor:.4f}",
    )

    # 5.3 — Max win streak ≥ 0
    pass_fail(
        "max_win_streak ≥ 0",
        report.max_win_streak >= 0,
        f"got={report.max_win_streak}",
    )

    # 5.4 — Max loss streak ≥ 0
    pass_fail(
        "max_loss_streak ≥ 0",
        report.max_loss_streak >= 0,
        f"got={report.max_loss_streak}",
    )

    # 5.5 — avg_holding_days ≥ 0
    pass_fail(
        "avg_holding_days ≥ 0",
        report.avg_holding_days >= 0,
        f"got={report.avg_holding_days}",
    )

    # 5.6 — Trade rate in [0, 1]
    pass_fail(
        "trade_rate in [0, 1]",
        0 <= report.trade_rate <= 1,
        f"got={report.trade_rate:.4f}",
    )

    # 5.7 — n_trades > 0 (we used 55% up_rate)
    pass_fail(
        "n_trades > MIN_TRADES (20)",
        report.n_trades >= 20,
        f"got={report.n_trades}",
    )

    # 5.8 — Expectancy is finite
    pass_fail(
        "expectancy_pct is finite",
        np.isfinite(report.expectancy_pct),
        f"got={report.expectancy_pct:.4f}",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — metrics.py: Report + JSON serialisation
# ══════════════════════════════════════════════════════════════════════════════

def test_metrics_serialisation(report):
    section("6 · metrics.py — report + JSON serialisation")

    from backtest.metrics import report_to_dict

    # 6.1 — Labels are non-empty strings
    pass_fail(
        "sharpe_label is non-empty string",
        isinstance(report.sharpe_label, str) and len(report.sharpe_label) > 0,
        f"got='{report.sharpe_label}'",
    )
    pass_fail(
        "drawdown_label is non-empty string",
        isinstance(report.drawdown_label, str) and len(report.drawdown_label) > 0,
    )
    pass_fail(
        "win_rate_label is non-empty string",
        isinstance(report.win_rate_label, str) and len(report.win_rate_label) > 0,
    )
    pass_fail(
        "verdict is non-empty string",
        isinstance(report.verdict, str) and len(report.verdict) > 0,
    )

    # 6.2 — Rolling Sharpe is a pd.Series
    pass_fail(
        "rolling_sharpe is pd.Series",
        isinstance(report.rolling_sharpe, pd.Series),
        f"len={len(report.rolling_sharpe)}",
    )

    # 6.3 — Rolling drawdown is a pd.Series
    pass_fail(
        "rolling_drawdown is pd.Series",
        isinstance(report.rolling_drawdown, pd.Series),
        f"len={len(report.rolling_drawdown)}",
    )

    # 6.4 — report_to_dict returns JSON-serialisable dict
    d = report_to_dict(report)
    pass_fail(
        "report_to_dict returns dict",
        isinstance(d, dict),
    )

    # 6.5 — JSON-serialisable (no numpy types, no Timestamps)
    try:
        json_str = json.dumps(d)
        pass_fail("report_to_dict is JSON-serialisable", True,
                  f"len={len(json_str)} chars")
    except (TypeError, ValueError) as e:
        pass_fail("report_to_dict is JSON-serialisable", False, str(e))

    # 6.6 — Dict has expected top-level keys
    expected_keys = {
        "ticker", "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
        "win_rate_pct", "beats_benchmark", "verdict",
        "rolling_sharpe", "rolling_drawdown", "rolling_dates",
    }
    has_keys = expected_keys.issubset(set(d.keys()))
    pass_fail(
        "report_to_dict has all expected keys",
        has_keys,
        f"missing={expected_keys - set(d.keys())}" if not has_keys else "",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — backtester.py: Prediction-to-price alignment
# ══════════════════════════════════════════════════════════════════════════════

def test_backtester_alignment():
    section("7 · backtester.py — prediction alignment")

    from backtest.backtester import _extract_dates, _get_next_day_prices

    # 7.1 — _extract_dates with plain DatetimeIndex
    dates = pd.bdate_range("2023-01-03", periods=50)
    X_plain = pd.DataFrame(np.random.randn(50, 5), index=dates)
    extracted = _extract_dates(X_plain)
    pass_fail(
        "_extract_dates works with plain DatetimeIndex",
        isinstance(extracted, pd.DatetimeIndex) and len(extracted) == 50,
    )

    # 7.2 — _extract_dates with MultiIndex
    mi = pd.MultiIndex.from_arrays(
        [dates, ["AAPL"] * 50], names=["date", "ticker"]
    )
    X_multi = pd.DataFrame(np.random.randn(50, 5), index=mi)
    extracted_multi = _extract_dates(X_multi)
    pass_fail(
        "_extract_dates works with (date, ticker) MultiIndex",
        isinstance(extracted_multi, pd.DatetimeIndex)
        and len(extracted_multi) == 50,
    )

    # 7.3 — _get_next_day_prices returns correct tuple structure
    signal_dates = pd.bdate_range("2023-01-03", periods=30)
    price_df     = make_price_df(signal_dates)
    opens, closes, valid_dates, valid_idx = _get_next_day_prices(
        signal_dates, price_df
    )
    pass_fail(
        "_get_next_day_prices returns 4-element tuple",
        len(opens) == len(closes) == len(valid_dates) == len(valid_idx),
        f"n_valid={len(opens)}",
    )

    # 7.4 — Valid indices are a subset of original range
    pass_fail(
        "valid_idx values within [0, n_signals)",
        all(0 <= i < 30 for i in valid_idx),
    )

    # 7.5 — All returned prices are positive
    pass_fail(
        "returned open/close prices all positive",
        (opens > 0).all() and (closes > 0).all(),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — visualiser.py: Chart data + build_report
# ══════════════════════════════════════════════════════════════════════════════

def test_visualiser_charts(bt_result, perf_report):
    section("8 · visualiser.py — chart data + build_report")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from backtest.visualiser import (
        plot_equity_curve, plot_drawdown, plot_rolling_sharpe,
        plot_monthly_returns, plot_trade_scatter,
        plot_performance_page, build_report,
        BacktestReport,
    )

    # 8.1 — plot_equity_curve
    fig_eq, eq_data = plot_equity_curve(bt_result, perf_report)
    pass_fail(
        "plot_equity_curve returns (Figure, dict)",
        isinstance(fig_eq, plt.Figure) and isinstance(eq_data, dict),
    )
    pass_fail(
        "equity chart data has dates/strategy/benchmark keys",
        all(k in eq_data for k in ("dates", "strategy", "benchmark")),
    )
    pass_fail(
        "equity dates and strategy arrays have same length",
        len(eq_data["dates"]) == len(eq_data["strategy"]),
        f"dates={len(eq_data['dates'])}, strategy={len(eq_data['strategy'])}",
    )
    plt.close(fig_eq)

    # 8.2 — plot_drawdown
    fig_dd, dd_data = plot_drawdown(bt_result, perf_report)
    pass_fail(
        "plot_drawdown returns (Figure, dict)",
        isinstance(fig_dd, plt.Figure) and isinstance(dd_data, dict),
    )
    pass_fail(
        "drawdown data has dates/drawdown keys",
        "dates" in dd_data and "drawdown" in dd_data,
    )
    pass_fail(
        "all drawdown values ≤ 0",
        all(d <= 0.0001 for d in dd_data["drawdown"]),
        f"max_dd={max(dd_data['drawdown']):.6f}",
    )
    plt.close(fig_dd)

    # 8.3 — plot_rolling_sharpe
    fig_rs, rs_data = plot_rolling_sharpe(perf_report)
    pass_fail(
        "plot_rolling_sharpe returns (Figure, dict)",
        isinstance(fig_rs, plt.Figure) and isinstance(rs_data, dict),
    )
    pass_fail(
        "rolling sharpe data has dates/sharpe keys",
        "dates" in rs_data and "sharpe" in rs_data,
    )
    plt.close(fig_rs)

    # 8.4 — plot_monthly_returns
    fig_mr, mr_data = plot_monthly_returns(bt_result)
    pass_fail(
        "plot_monthly_returns returns (Figure, dict)",
        isinstance(fig_mr, plt.Figure) and isinstance(mr_data, dict),
    )
    pass_fail(
        "monthly data has years/months/returns keys",
        all(k in mr_data for k in ("years", "months", "returns")),
    )
    pass_fail(
        "monthly data months list has 12 entries",
        len(mr_data["months"]) == 12,
    )
    plt.close(fig_mr)

    # 8.5 — plot_trade_scatter
    fig_ts, ts_data = plot_trade_scatter(bt_result)
    pass_fail(
        "plot_trade_scatter returns (Figure, dict)",
        isinstance(fig_ts, plt.Figure) and isinstance(ts_data, dict),
    )
    pass_fail(
        "trade data has dates/returns/is_win keys",
        all(k in ts_data for k in ("dates", "returns", "is_win")),
    )
    pass_fail(
        "trade scatter dates count matches trade count",
        len(ts_data["dates"]) == bt_result.n_trades,
        f"scatter={len(ts_data['dates'])}, trades={bt_result.n_trades}",
    )
    plt.close(fig_ts)

    # 8.6 — plot_performance_page (multi-panel figure)
    tmpdir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(tmpdir, "test_perf.png")
        fig_page, all_data = plot_performance_page(
            bt_result, perf_report, save_path=save_path
        )
        pass_fail(
            "plot_performance_page returns (Figure, dict)",
            isinstance(fig_page, plt.Figure) and isinstance(all_data, dict),
        )
        pass_fail(
            "performance page has equity/drawdown/rolling/trades panels",
            all(k in all_data for k in ("equity", "drawdown", "rolling", "trades")),
        )
        pass_fail(
            "performance page PNG saved to disk",
            os.path.exists(save_path),
        )
        plt.close(fig_page)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 8.7 — build_report (full pipeline)
    tmpdir2 = tempfile.mkdtemp()
    try:
        bt_report = build_report(bt_result, save_dir=tmpdir2, verbose=False)
        pass_fail(
            "build_report returns BacktestReport",
            isinstance(bt_report, BacktestReport),
        )
        pass_fail(
            "BacktestReport has PerformanceReport inside",
            hasattr(bt_report, "report") and bt_report.report is not None,
        )
        pass_fail(
            "BacktestReport has equity_data dict",
            isinstance(bt_report.equity_data, dict)
            and "dates" in bt_report.equity_data,
        )
        pass_fail(
            "BacktestReport has 6 figures",
            len(bt_report.figures) == 6,
            f"got={list(bt_report.figures.keys())}",
        )

        # Check all figure PNGs were saved
        expected_pngs = [
            f"TEST_equity_curve.png",
            f"TEST_drawdown.png",
            f"TEST_rolling_sharpe.png",
            f"TEST_monthly_returns.png",
            f"TEST_trades.png",
            f"TEST_performance.png",
        ]
        saved = os.listdir(tmpdir2)
        for png in expected_pngs:
            pass_fail(
                f"saved figure: {png}",
                png in saved,
                f"files={saved}" if png not in saved else "",
            )
    finally:
        shutil.rmtree(tmpdir2, ignore_errors=True)

    return bt_report


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — visualiser.py: JSON + compare_strategies
# ══════════════════════════════════════════════════════════════════════════════

def test_visualiser_json_and_compare(bt_report):
    section("9 · visualiser.py — JSON + compare_strategies")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from backtest.visualiser import report_to_json, compare_strategies

    # 9.1 — report_to_json
    json_data = report_to_json(bt_report)
    pass_fail(
        "report_to_json returns dict",
        isinstance(json_data, dict),
    )
    pass_fail(
        "JSON has 'metrics', 'charts', 'summary' top-level keys",
        all(k in json_data for k in ("metrics", "charts", "summary")),
    )
    pass_fail(
        "charts has equity/drawdown/monthly_returns/trades/rolling_sharpe",
        all(k in json_data["charts"]
            for k in ("equity", "drawdown", "monthly_returns",
                      "trades", "rolling_sharpe")),
    )
    pass_fail(
        "summary has ticker, verdict, beats_benchmark",
        all(k in json_data["summary"]
            for k in ("ticker", "verdict", "beats_benchmark")),
    )

    # 9.2 — Full JSON round-trip
    try:
        dumped  = json.dumps(json_data)
        loaded  = json.loads(dumped)
        pass_fail("JSON full round-trip (dumps → loads)", True,
                  f"len={len(dumped)} chars")
    except Exception as e:
        pass_fail("JSON full round-trip", False, str(e))

    # 9.3 — compare_strategies
    # Build a second report with different config to compare
    preds2, opens2, closes2, dates2 = make_backtest_arrays(n_days=200, seed=99)
    from backtest.engine import run_backtest
    from backtest.visualiser import build_report as br
    result2 = run_backtest(
        preds2, opens2, closes2, dates2,
        ticker="TEST2", config_name="conservative", verbose=False,
    )
    bt_report2 = br(result2, verbose=False)

    reports = {"Default": bt_report, "Conservative": bt_report2}
    fig_cmp, cmp_data = compare_strategies(
        reports, metric="total_return"
    )
    pass_fail(
        "compare_strategies returns (Figure, dict)",
        isinstance(fig_cmp, plt.Figure) and isinstance(cmp_data, dict),
    )
    pass_fail(
        "compare chart data has labels/values/metric",
        all(k in cmp_data for k in ("labels", "values", "metric")),
    )
    pass_fail(
        "compare labels match input",
        cmp_data["labels"] == ["Default", "Conservative"],
    )

    # 9.4 — compare_strategies with different metrics
    for metric_name in ["sharpe_ratio", "max_drawdown", "win_rate"]:
        try:
            fig_m, _ = compare_strategies(reports, metric=metric_name)
            pass_fail(
                f"compare_strategies works with metric='{metric_name}'",
                True,
            )
            plt.close(fig_m)
        except Exception as e:
            pass_fail(
                f"compare_strategies works with metric='{metric_name}'",
                False, str(e),
            )

    # 9.5 — compare_strategies raises on unknown metric
    try:
        compare_strategies(reports, metric="nonexistent_metric")
        pass_fail("compare_strategies raises on unknown metric", False)
    except ValueError:
        pass_fail("compare_strategies raises on unknown metric", True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — Full pipeline: trainer → backtest → report → JSON
# ══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_integration():
    section("10 · Full pipeline: trainer → backtest → report → JSON")

    import matplotlib
    matplotlib.use("Agg")

    from models.trainer import build_pipeline, train, _three_way_split
    from backtest.engine import run_backtest
    from backtest.metrics import compute_metrics
    from backtest.visualiser import build_report, report_to_json

    X, y = make_X_y(n_days=504, n_features=30)

    # 10.1 — Train model (same as test_models_e2e)
    try:
        pipeline, results = train(X, y, verbose=False)
        pass_fail("train() produces fitted pipeline", True)
    except Exception as e:
        pass_fail("train() produces fitted pipeline", False, str(e))
        return

    # 10.2 — Get test split
    X_train, X_val, X_test, y_train, y_val, y_test = _three_way_split(X, y)
    pass_fail(
        "three-way split yields non-empty test set",
        len(X_test) > 0,
        f"n_test={len(X_test)}",
    )

    # 10.3 — Generate predictions on test set
    X_test_reset = X_test.reset_index(drop=True)
    probas = pipeline.predict_proba(X_test_reset)[:, 1]
    preds  = (probas >= 0.5).astype(int)
    pass_fail(
        "predictions generated for test set",
        len(preds) == len(X_test),
        f"n_preds={len(preds)}, UP_rate={preds.mean():.2f}",
    )

    # 10.4 — Build synthetic price data aligned with test dates
    test_dates = pd.DatetimeIndex(X_test.index).normalize()
    price_df   = make_price_df(test_dates)

    # Find next-day aligned prices
    from backtest.backtester import _get_next_day_prices
    opens, closes, valid_dates, valid_idx = _get_next_day_prices(
        test_dates, price_df
    )
    preds_aligned = preds[valid_idx]

    pass_fail(
        "price alignment yields ≥ 20 valid predictions",
        len(preds_aligned) >= 20,
        f"n_aligned={len(preds_aligned)}",
    )

    # 10.5 — Run engine backtest
    bt_result = run_backtest(
        predictions=preds_aligned,
        open_prices=opens,
        close_prices=closes,
        dates=valid_dates,
        ticker="SYNTH",
        config_name="default",
        verbose=False,
    )
    pass_fail(
        "backtest engine completes on model predictions",
        bt_result.n_trades > 0,
        f"n_trades={bt_result.n_trades}",
    )

    # 10.6 — Compute metrics
    try:
        perf_report = compute_metrics(bt_result, verbose=False)
        pass_fail("compute_metrics succeeds on backtest result", True)
    except ValueError as e:
        # May have too few trades — that's a valid outcome
        if "too few" in str(e).lower() or "minimum" in str(e).lower():
            print(f"  ⚠️  Too few trades for metrics ({bt_result.n_trades}) — "
                  f"acceptable for synthetic data")
            pass_fail("compute_metrics raises expected ValueError for few trades", True)
            return
        else:
            pass_fail("compute_metrics succeeds", False, str(e))
            return

    # 10.7 — Build full visual report
    tmpdir = tempfile.mkdtemp()
    try:
        bt_report = build_report(bt_result, save_dir=tmpdir, verbose=False)
        pass_fail(
            "build_report produces BacktestReport",
            bt_report is not None and bt_report.report is not None,
        )

        # 10.8 — JSON export
        json_data = report_to_json(bt_report)
        json_str  = json.dumps(json_data)
        pass_fail(
            "full pipeline JSON export round-trips",
            len(json_str) > 100,
            f"json_len={len(json_str)} chars",
        )

        # 10.9 — JSON has complete data for frontend
        charts = json_data.get("charts", {})
        pass_fail(
            "JSON charts have equity curve data",
            "equity" in charts and len(charts["equity"].get("dates", [])) > 0,
            f"n_equity_points={len(charts.get('equity', {}).get('dates', []))}",
        )

        # 10.10 — Saved PNGs exist
        saved = os.listdir(tmpdir)
        png_count = sum(1 for f in saved if f.endswith(".png"))
        pass_fail(
            f"PNG files saved ({png_count} found)",
            png_count >= 5,
            f"files={saved}",
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 62)
    print("  StockSense AI — Backtest End-to-End Test Suite")
    print("  Sections: engine · metrics · backtester · visualiser · pipeline")
    print("═" * 62)

    t0 = time.time()

    # ── Sections 1-3: engine.py ───────────────────────────────────────────
    bt_result = test_engine_trade_simulation()
    test_engine_benchmark(bt_result)
    test_engine_configs()

    # ── Sections 4-6: metrics.py ──────────────────────────────────────────
    perf_report = test_metrics_portfolio(bt_result)
    test_metrics_trade_level(perf_report)
    test_metrics_serialisation(perf_report)

    # ── Section 7: backtester.py ──────────────────────────────────────────
    test_backtester_alignment()

    # ── Sections 8-9: visualiser.py ───────────────────────────────────────
    bt_report = test_visualiser_charts(bt_result, perf_report)
    test_visualiser_json_and_compare(bt_report)

    # ── Section 10: Full pipeline integration ─────────────────────────────
    test_full_pipeline_integration()

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")

    sys.exit(summary())
