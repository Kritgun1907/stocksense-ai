"""
StockSense AI — backtest/backtester.py
========================================
High-level backtesting pipeline.

This file owns:
  - Connecting trained pipeline to the backtest engine
  - Solving the prediction-to-price alignment problem
  - Walk-forward backtest (optional rigorous mode)
  - Multi-stock portfolio backtest orchestration
  - Generating prediction arrays from feature matrices

It does NOT own:
  - Trade simulation mechanics  → backtest/engine.py
  - Financial metrics           → backtest/metrics.py (Chapter 5.3)
  - Visualisation               → backtest/visualiser.py (Chapter 5.4)
  - Model training              → models/trainer.py

The alignment problem:
─────────────────────────────────────────────────────────────
  Feature matrix X has dates where rows weren't dropped (non-neutral).
  Prices have continuous trading days (no gaps except weekends).
  After labelling with threshold, some dates disappear from X.
  The backtester must correctly find the NEXT TRADING DAY's prices
  for each prediction date — not the current day's prices.
  This is non-trivial because 'next trading day' is not always 'date + 1'.

Walk-forward vs fixed model:
─────────────────────────────────────────────────────────────
  Fixed model: train once, predict on held-out test.
               Fast, honest if test was truly held out.
               This is the default mode.
  Walk-forward: retrain model at each time step.
               Slowest, most rigorous, best simulation of real trading.
               Optional — enabled with walk_forward=True.
               Warning: for 500 stocks × 250 test days = 125,000 refits.
               Use sparingly — single stock walk-forward takes ~10 minutes.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from sklearn.pipeline import Pipeline

from backtest.engine import (
    run_backtest, run_portfolio_backtest,
    BacktestResult, BACKTEST_CONFIGS,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Minimum trades required for a meaningful backtest.
# Below this, metrics are statistically unreliable.
MIN_TRADES_FOR_VALIDITY = 20


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_dates(X: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Extract a clean DatetimeIndex from either a MultiIndex or plain index.
    Handles both (date, ticker) MultiIndex from assembler.py
    and plain DatetimeIndex from single-stock pipelines.
    """
    if isinstance(X.index, pd.MultiIndex):
        dates = pd.DatetimeIndex(
            X.index.get_level_values("date")
        ).normalize()
    else:
        dates = pd.DatetimeIndex(X.index).normalize()
    return dates


def _get_next_day_prices(
    signal_dates: pd.DatetimeIndex,
    price_df:     pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """
    For each signal date, find the next available trading day's
    open and close prices.

    Why 'next trading day' rather than 'signal date + 1 calendar day'?
    ─────────────────────────────────────────────────────────────────────
    Signal generated on Friday → next trading day is Monday.
    signal_date + 1 day = Saturday → not in price_df → KeyError.
    This function correctly skips weekends and holidays.

    Parameters
    ----------
    signal_dates : Dates when predictions were generated.
    price_df     : OHLCV DataFrame with 'open' and 'close' columns.
                   Must cover the period AFTER signal_dates.

    Returns
    -------
    (open_prices, close_prices, valid_signal_dates, valid_indices) tuple.
    Skips signal dates with no available next trading day.
    valid_indices: positions in original signal_dates that were kept.
    """
    trading_days = price_df.index.normalize()
    open_list    = []
    close_list   = []
    date_list    = []
    idx_list     = []

    for i, sig_date in enumerate(signal_dates):
        sig_date   = pd.Timestamp(sig_date).normalize()
        future     = trading_days[trading_days > sig_date]

        if len(future) == 0:
            continue   # no next trading day — skip (last row in test)

        next_day = future[0]

        try:
            open_price  = float(price_df.loc[next_day, "open"])
            close_price = float(price_df.loc[next_day, "close"])
        except KeyError:
            continue   # next day not in price_df — skip

        if np.isnan(open_price) or np.isnan(close_price):
            continue   # bad data — skip

        open_list.append(open_price)
        close_list.append(close_price)
        date_list.append(sig_date)
        idx_list.append(i)

    return (
        np.array(open_list, dtype=float),
        np.array(close_list, dtype=float),
        pd.DatetimeIndex(date_list),
        np.array(idx_list, dtype=int),
    )


def _fetch_price_data(
    ticker: str,
    start:  pd.Timestamp,
    end:    pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker between start and end dates.
    Extends end by 5 days to ensure next-day prices are available
    for signals generated near the end of the test period.
    """
    # Extend end by 5 trading days to capture next-day prices
    extended_end = end + pd.Timedelta(days=7)

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=extended_end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No price data fetched for {ticker}")

    raw.columns = [c.lower() for c in raw.columns]

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.DatetimeIndex(raw.index).normalize()

    return raw[["open", "high", "low", "close", "volume"]]


# ══════════════════════════════════════════════════════════════════════════════
#  CORE BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

def backtest_pipeline(
    pipeline:         Pipeline,
    X_test:           pd.DataFrame,
    ticker:           str,
    price_df:         Optional[pd.DataFrame] = None,
    threshold:        float = 0.5,
    config_name:      str   = "default",
    verbose:          bool  = True,
) -> BacktestResult:
    """
    Run a backtest by connecting a trained pipeline to the engine.

    This is the main entry point for single-stock backtesting.
    Handles prediction generation, price alignment, and engine call.

    Parameters
    ----------
    pipeline    : Fitted sklearn Pipeline from trainer.py.
    X_test      : Test feature matrix (held-out, never used in training).
                  Can have (date, ticker) MultiIndex or plain DatetimeIndex.
    ticker      : Stock symbol. Used to fetch prices if price_df is None.
    price_df    : Pre-fetched OHLCV DataFrame. If None, fetches via yfinance.
                  Providing price_df is faster for repeated calls.
    threshold   : Classification threshold for UP signal.
                  Use find_optimal_threshold() from trainer.py.
    config_name : Backtest configuration name from BACKTEST_CONFIGS.
    verbose     : Print backtest progress.

    Returns
    -------
    BacktestResult with full portfolio history, trade log, and benchmark.

    Raises
    ------
    ValueError : If too few trades generated (< MIN_TRADES_FOR_VALIDITY).
    """
    # ── Step 1: Extract dates from test set ───────────────────────────────
    signal_dates = _extract_dates(X_test)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"Backtesting: {ticker}")
        print(f"  Test period: {signal_dates[0].date()} → "
              f"{signal_dates[-1].date()}")
        print(f"  Test rows:   {len(X_test):,}")
        print(f"  Config:      {config_name}")
        print(f"  Threshold:   {threshold:.3f}")

    # ── Step 2: Fetch price data if not provided ───────────────────────────
    if price_df is None:
        if verbose:
            print(f"  Fetching price data from yfinance...")
        price_df = _fetch_price_data(
            ticker,
            start=signal_dates[0],
            end=signal_dates[-1],
        )

    # ── Step 3: Generate predictions ──────────────────────────────────────
    X_reset   = X_test.reset_index(drop=True)
    probas    = pipeline.predict_proba(X_reset)[:, 1]
    preds     = (probas >= threshold).astype(int)

    if verbose:
        print(f"  UP signals:  {preds.sum()} / {len(preds)} "
              f"({preds.mean()*100:.1f}%)")

    # ── Step 4: Align predictions with next-day prices ─────────────────────
    open_prices, close_prices, valid_dates, valid_idx = _get_next_day_prices(
        signal_dates, price_df
    )

    # Keep only predictions that have corresponding next-day prices
    preds_aligned = preds[valid_idx]
    probas_aligned = probas[valid_idx]

    if len(preds_aligned) < MIN_TRADES_FOR_VALIDITY:
        raise ValueError(
            f"Too few valid predictions after alignment: "
            f"{len(preds_aligned)} (need ≥ {MIN_TRADES_FOR_VALIDITY}). "
            f"Check that price_df covers the test period."
        )

    if verbose:
        print(f"  Aligned:     {len(preds_aligned)} / {len(preds)} predictions "
              f"({len(preds)-len(preds_aligned)} dropped — no next-day price)")

    # ── Step 5: Run engine ─────────────────────────────────────────────────
    result = run_backtest(
        predictions=preds_aligned,
        open_prices=open_prices,
        close_prices=close_prices,
        dates=valid_dates,
        ticker=ticker,
        config_name=config_name,
        verbose=verbose,
    )

    return result


def backtest_multiple_tickers(
    pipeline:    Pipeline,
    X_test_dict: Dict[str, pd.DataFrame],
    price_dict:  Optional[Dict[str, pd.DataFrame]] = None,
    threshold:   float = 0.5,
    config_name: str   = "default",
    verbose:     bool  = True,
) -> Dict[str, BacktestResult]:
    """
    Backtest a single trained pipeline across multiple stocks.

    The same universal model (trained on all stocks combined) is used
    to generate predictions for each individual stock's test set.

    Parameters
    ----------
    pipeline     : Fitted universal pipeline from trainer.py.
    X_test_dict  : Dict mapping ticker → test feature DataFrame.
    price_dict   : Dict mapping ticker → OHLCV DataFrame (optional).
                   If None, fetches from yfinance per ticker.
    threshold    : Classification threshold.
    config_name  : Backtest config name.
    verbose      : Print per-ticker progress.

    Returns
    -------
    Dict mapping ticker → BacktestResult.
    Tickers that fail alignment or have too few trades are skipped.
    """
    results = {}
    total   = len(X_test_dict)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"Multi-Ticker Backtest: {total} stocks")
        print(f"Config: {config_name} | Threshold: {threshold:.3f}")
        print(f"{'═'*60}")

    for i, (ticker, X_test) in enumerate(X_test_dict.items()):
        if verbose:
            print(f"\n[{i+1}/{total}] {ticker}")

        price_df = price_dict.get(ticker) if price_dict else None

        try:
            result = backtest_pipeline(
                pipeline=pipeline,
                X_test=X_test,
                ticker=ticker,
                price_df=price_df,
                threshold=threshold,
                config_name=config_name,
                verbose=False,
            )
            results[ticker] = result

            if verbose:
                pv    = result.daily_portfolio["portfolio_value"]
                ret   = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
                bh    = result.benchmark_portfolio["portfolio_value"]
                bh_ret = (bh.iloc[-1] / bh.iloc[0] - 1) * 100
                beat  = "✅" if ret > bh_ret else "❌"
                print(f"  Strategy: {ret:+.1f}% | "
                      f"BuyHold: {bh_ret:+.1f}% {beat} | "
                      f"Trades: {result.n_trades}")

        except ValueError as e:
            if verbose:
                print(f"  ⚠️  Skipped: {e}")
            continue
        except Exception as e:
            if verbose:
                print(f"  ❌ Error: {e}")
            continue

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Completed: {len(results)}/{total} stocks backtested")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_backtest(
    X:              pd.DataFrame,
    y:              pd.Series,
    ticker:         str,
    price_df:       pd.DataFrame,
    n_splits:       int   = 5,
    gap_days:       int   = 20,
    xgb_params:     Optional[Dict] = None,
    threshold:      float = 0.5,
    config_name:    str   = "default",
    verbose:        bool  = True,
) -> List[BacktestResult]:
    """
    Walk-forward backtest: retrain model at each fold, predict next fold.

    This is the most rigorous backtesting approach. At each step:
      1. Train pipeline on all data up to fold boundary
      2. Generate predictions on next fold
      3. Run backtest on those predictions
      4. Move to next fold

    Parameters
    ----------
    X           : Full feature matrix (not pre-split).
    y           : Full target series.
    ticker      : Stock symbol.
    price_df    : Full OHLCV DataFrame covering entire period.
    n_splits    : Number of walk-forward folds.
    gap_days    : Trading day gap between train and test.
    xgb_params  : XGBoost parameters (uses DEFAULT_XGB_PARAMS if None).
    threshold   : Classification threshold.
    config_name : Backtest config.
    verbose     : Print per-fold progress.

    Returns
    -------
    List of BacktestResult, one per fold.
    Combine them with combine_walk_forward_results() for aggregate metrics.

    Warning:
    ─────────────────────────────────────────────────────────────
    Walk-forward is SLOW. Each fold requires a full model refit.
    For 500 stocks × 5 folds = 2,500 model training runs.
    Only use for your primary research stock or a small subset.
    """
    from models.timeseries_cv import TimeSeriesSplitWithGap
    from models.trainer import build_pipeline

    cv      = TimeSeriesSplitWithGap(n_splits=n_splits, gap_days=gap_days)
    results = []

    if verbose:
        print(f"\n{'═'*60}")
        print(f"Walk-Forward Backtest: {ticker}")
        print(f"  Folds: {n_splits} | Gap: {gap_days} days")
        print(f"{'═'*60}")

    for fold_info in cv.split_with_info(X):
        fold = fold_info["fold"]

        if verbose:
            train_start, train_end = fold_info["train_dates"]
            test_start,  test_end  = fold_info["test_dates"]
            print(f"\nFold {fold}:")
            print(f"  Train: {train_start.date()} → {train_end.date()} "
                  f"({fold_info['n_train']:,} rows)")
            print(f"  Test:  {test_start.date()} → {test_end.date()} "
                  f"({fold_info['n_test']:,} rows)")

        # ── Get fold data
        X_train_f = X.iloc[fold_info["train_idx"]]
        X_test_f  = X.iloc[fold_info["test_idx"]]
        y_train_f = y.iloc[fold_info["train_idx"]]

        # ── Train fresh pipeline on this fold's training data
        from models.trainer import _calculate_scale_pos_weight
        spw      = _calculate_scale_pos_weight(y_train_f)
        pipeline = build_pipeline(xgb_params=xgb_params,
                                  scale_pos_weight=spw)

        # Simple train without early stopping for walk-forward
        # (no separate val set within each fold)
        X_tr_r = X_train_f.reset_index(drop=True)
        y_tr_r = y_train_f.reset_index(drop=True)
        pipeline.fit(X_tr_r, y_tr_r)

        if verbose:
            print(f"  Pipeline fitted.")

        # ── Backtest this fold
        try:
            result = backtest_pipeline(
                pipeline=pipeline,
                X_test=X_test_f,
                ticker=ticker,
                price_df=price_df,
                threshold=threshold,
                config_name=config_name,
                verbose=False,
            )
            results.append(result)

            pv  = result.daily_portfolio["portfolio_value"]
            ret = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
            if verbose:
                print(f"  Fold return: {ret:+.1f}% | "
                      f"Trades: {result.n_trades}")

        except ValueError as e:
            if verbose:
                print(f"  ⚠️  Fold skipped: {e}")
            continue

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Walk-forward complete: {len(results)}/{n_splits} folds")

    return results


def combine_walk_forward_results(
    results: List[BacktestResult],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Combine portfolio histories from multiple walk-forward folds
    into a single continuous equity curve.

    Each fold's portfolio starts where the previous fold ended.
    This simulates continuous capital deployment across all folds.

    Parameters
    ----------
    results : List of BacktestResult from walk_forward_backtest().
    verbose : Print combined summary.

    Returns
    -------
    pd.DataFrame with combined portfolio_value indexed by date.
    """
    if not results:
        return pd.DataFrame()

    combined_rows = []
    running_capital = results[0].config["initial_capital"]

    for result in results:
        pv     = result.daily_portfolio["portfolio_value"]
        # Scale this fold's portfolio to start from running_capital
        scale  = running_capital / pv.iloc[0]
        scaled = pv * scale

        for date, val in scaled.items():
            combined_rows.append({"date": date, "portfolio_value": val})

        running_capital = scaled.iloc[-1]

    combined = pd.DataFrame(combined_rows).set_index("date")
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    if verbose:
        initial  = results[0].config["initial_capital"]
        final    = combined["portfolio_value"].iloc[-1]
        total_ret = (final / initial - 1) * 100
        n_trades  = sum(r.n_trades for r in results)

        print(f"\nWalk-Forward Combined Results:")
        print(f"  Total return:  {total_ret:+.1f}%")
        print(f"  Total trades:  {n_trades}")
        print(f"  Date range:    {combined.index[0].date()} → "
              f"{combined.index[-1].date()}")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def quick_backtest(
    ticker:       str,
    period:       str   = "2y",
    config_name:  str   = "default",
    threshold:    float = 0.5,
    verbose:      bool  = True,
) -> BacktestResult:
    """
    One-function convenience wrapper: ticker → backtest result.

    Fetches data, engineers features, trains model, backtests.
    Useful for quick exploration and the website's "demo" mode.

    Parameters
    ----------
    ticker      : Stock symbol e.g. "AAPL".
    period      : yfinance period string for data download.
    config_name : Backtest configuration.
    threshold   : Classification threshold.
    verbose     : Print all progress.

    Returns
    -------
    BacktestResult for the test period.

    Warning: This trains a fresh model every call.
    For production, use backtest_pipeline() with a pre-trained pipeline.
    """
    import yfinance as yf
    from data.cleaner import clean_stock_data
    from features.engineer import build_features
    from features.indicators import get_model_features
    from data.labeller import create_labels
    from models.trainer import train, _three_way_split

    if verbose:
        print(f"\nQuick backtest: {ticker} ({period})")

    # ── Fetch and prepare
    raw = yf.download(ticker, period=period,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    raw.columns = [c.lower() for c in raw.columns]

    from data.cleaner import clean_stock_data
    clean    = clean_stock_data(raw, ticker=ticker)
    featured = build_features(clean).dropna()
    labelled = create_labels(featured, horizon=1,
                             threshold=0.003, verbose=False)
    X = get_model_features(labelled, extra_drop=["target"]).fillna(0)
    y = labelled["target"]

    # ── Train
    pipeline, _ = train(X, y, verbose=False)

    # ── Split
    _, _, X_test, _, _, _ = _three_way_split(X, y)

    # ── Backtest
    result = backtest_pipeline(
        pipeline=pipeline,
        X_test=X_test,
        ticker=ticker,
        price_df=clean,
        threshold=threshold,
        config_name=config_name,
        verbose=verbose,
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    # ── Quick single-stock backtest
    print("Running quick backtest for AAPL...")
    result = quick_backtest("AAPL", period="2y", verbose=True)

    # ── Show trade log
    print(f"\nFirst 5 trades:")
    for trade in result.trades[:5]:
        print(f"  {trade.entry_date.date()} → "
              f"net P&L: ${trade.net_pnl:+.2f} "
              f"({trade.return_pct*100:+.2f}%)")

    # ── Show portfolio tail
    print(f"\nPortfolio value (last 5 days):")
    print(result.daily_portfolio["portfolio_value"].tail())
