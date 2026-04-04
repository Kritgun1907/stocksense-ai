"""
StockSense AI — backtest/engine.py
=====================================
Portfolio-level backtesting engine.

This file owns:
  - Trade simulation with realistic timing (signal at close, trade at open)
  - Portfolio value tracking with compounding
  - Transaction cost simulation
  - Position sizing (fixed fraction + optional Kelly)
  - Benchmark comparison (buy-and-hold)

It does NOT own:
  - Metric calculation         → backtest/metrics.py (Chapter 5.3)
  - Result visualisation       → backtest/visualiser.py (Chapter 5.4)
  - Model predictions          → models/trainer.py
  - Feature engineering        → features/engineer.py

Why open-to-close returns (not close-to-close)?
─────────────────────────────────────────────────────────────
  Your signal is generated at the CLOSE of day t using that day's
  features. The earliest you can act on it is the OPEN of day t+1.
  The return you capture is therefore open_t+1 → close_t+1.
  Close-to-close includes the overnight gap (close_t → open_t+1)
  which happens BEFORE your trade — you cannot capture it.
  Using close-to-close would be a subtle form of lookahead bias.

Why compound returns (not simple P&L)?
─────────────────────────────────────────────────────────────
  Simple P&L: each trade risks the same fixed dollar amount.
  Compounding: each trade risks a fraction of current portfolio value.
  Compounding is more realistic — successful traders reinvest profits.
  It also reveals portfolio risk more honestly: a 10% drawdown on a
  $20,000 portfolio (after gains) feels different to a 10% drawdown
  on the original $10,000.

Why position_size < 1.0?
─────────────────────────────────────────────────────────────
  Deploying 100% of capital on each trade means a single bad trade
  can permanently impair the portfolio. 95% leaves a cash buffer
  for costs and prevents rounding errors. In practice, position
  sizing (Kelly, volatility targeting) is a research area in itself.
  For this project, fixed fraction with Half-Kelly as an option
  covers the key concepts without over-engineering.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_INITIAL_CAPITAL  = 10_000.0   # starting portfolio value
DEFAULT_POSITION_SIZE    = 0.95       # fraction of capital per trade
DEFAULT_TRANSACTION_COST = 0.001      # 0.1% one-way (0.2% round trip)
TRADING_DAYS_PER_YEAR    = 252

# Backtest configs — mirrors LABEL_CONFIGS / ASSEMBLY_CONFIGS pattern
BACKTEST_CONFIGS = {
    "default": {
        "initial_capital":   10_000.0,
        "position_size":     0.95,
        "transaction_cost":  0.001,
        "allow_short":       False,
        "description": "Long-only, 95% position size, 0.1% one-way cost",
    },
    "conservative": {
        "initial_capital":   10_000.0,
        "position_size":     0.50,
        "transaction_cost":  0.002,
        "allow_short":       False,
        "description": "Long-only, 50% position size, 0.2% one-way cost",
    },
    "aggressive": {
        "initial_capital":   10_000.0,
        "position_size":     0.95,
        "transaction_cost":  0.0005,
        "allow_short":       True,
        "description": "Long-short, 95% position, 0.05% cost (institutional)",
    },
    "realistic_retail": {
        "initial_capital":   10_000.0,
        "position_size":     0.80,
        "transaction_cost":  0.002,
        "allow_short":       False,
        "description": "Long-only, 80% position, 0.2% cost (typical retail)",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """
    Records one complete trade (entry + exit).

    Why a dataclass rather than a dict?
    ─────────────────────────────────────────────────────────────
    Type-checked, auto-documented, IDE-friendly.
    The trade log is central to all downstream analysis.
    Explicit field types prevent subtle bugs from dict key typos.
    """
    entry_date:      pd.Timestamp
    exit_date:       pd.Timestamp
    ticker:          str
    direction:       str              # 'long' or 'short'
    entry_price:     float
    exit_price:      float
    shares:          float
    gross_pnl:       float
    transaction_cost: float
    net_pnl:         float
    return_pct:      float            # net return as fraction
    capital_before:  float
    capital_after:   float


@dataclass
class BacktestResult:
    """
    Complete backtest output — passed to metrics.py and visualiser.py.

    Why include both daily_portfolio and trades?
    ─────────────────────────────────────────────────────────────
    daily_portfolio: time series of portfolio value (for charts, Sharpe)
    trades:          individual trade log (for win rate, profit factor)
    Both are needed — portfolio view and trade view reveal different things.
    """
    daily_portfolio:    pd.DataFrame      # date, portfolio_value, cash, in_position
    trades:             List[Trade]        # individual trade records
    benchmark_portfolio: pd.DataFrame     # buy-and-hold comparison
    config:             Dict              # backtest parameters used
    ticker:             str
    n_signals:          int               # total UP signals generated
    n_trades:           int               # trades executed (= signals if long-only)
    date_range:         Tuple[pd.Timestamp, pd.Timestamp]


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _validate_backtest_inputs(
    predictions:  np.ndarray,
    open_prices:  np.ndarray,
    close_prices: np.ndarray,
    dates:        pd.DatetimeIndex,
) -> None:
    """
    Validate all backtest inputs before running.
    Raises ValueError with a clear message on any problem.

    Why validate explicitly?
    ─────────────────────────────────────────────────────────────
    Backtests with misaligned arrays silently produce wrong results.
    An off-by-one between predictions and prices is the most common
    bug and the hardest to detect — the backtest runs and produces
    plausible-looking numbers that are completely wrong.
    """
    arrays  = [predictions, open_prices, close_prices]
    lengths = [len(a) for a in arrays]

    if len(set(lengths)) != 1:
        raise ValueError(
            f"All arrays must have the same length. "
            f"Got: predictions={lengths[0]}, "
            f"open={lengths[1]}, close={lengths[2]}"
        )

    if len(dates) != len(predictions):
        raise ValueError(
            f"dates length ({len(dates)}) must match "
            f"predictions length ({len(predictions)})"
        )

    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError(
            f"predictions must contain only 0 and 1. "
            f"Found values: {np.unique(predictions)}"
        )

    if np.any(open_prices <= 0) or np.any(close_prices <= 0):
        raise ValueError(
            "All prices must be positive. "
            "Check for missing/corrupt price data."
        )

    if np.any(np.isnan(open_prices)) or np.any(np.isnan(close_prices)):
        raise ValueError(
            "NaN values found in price data. "
            "Run cleaner.py before backtesting."
        )


def _calculate_kelly_position_size(
    win_rate:     float,
    avg_win:      float,
    avg_loss:     float,
    kelly_fraction: float = 0.5,
) -> float:
    """
    Calculate optimal position size using (Half-)Kelly Criterion.

    Kelly = W - (1-W)/R where R = avg_win/avg_loss
    Half-Kelly = Kelly × kelly_fraction (default 0.5)

    Parameters
    ----------
    win_rate        : Historical fraction of profitable trades.
    avg_win         : Average return on winning trades (as fraction).
    avg_loss        : Average loss on losing trades (as fraction, positive).
    kelly_fraction  : Multiplier on full Kelly. 0.5 = Half-Kelly.

    Returns
    -------
    float position size as fraction of capital (clipped to [0.05, 0.95]).

    Why clip to 0.05 minimum?
    ─────────────────────────────────────────────────────────────
    Very low win rates or win/loss ratios can produce negative Kelly
    (don't trade this system). Clipping to 0.05 prevents complete
    inaction and surfaces the issue in the results rather than
    silently producing no trades.
    """
    if avg_loss == 0 or avg_win == 0:
        return 0.5  # default if insufficient history

    R     = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / R
    size  = kelly * kelly_fraction

    return float(np.clip(size, 0.05, 0.95))


# ══════════════════════════════════════════════════════════════════════════════
#  CORE BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    predictions:      np.ndarray,
    open_prices:      np.ndarray,
    close_prices:     np.ndarray,
    dates:            pd.DatetimeIndex,
    ticker:           str   = "UNKNOWN",
    initial_capital:  float = DEFAULT_INITIAL_CAPITAL,
    position_size:    float = DEFAULT_POSITION_SIZE,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    allow_short:      bool  = False,
    config_name:      Optional[str] = None,
    verbose:          bool  = True,
) -> BacktestResult:
    """
    Run a portfolio-level backtest on model predictions.

    Timing model:
      Signal generated at CLOSE of day t (using that day's features).
      Trade executed at OPEN of day t+1.
      Position closed at CLOSE of day t+1.
      Return captured: (close_t+1 - open_t+1) / open_t+1

    Parameters
    ----------
    predictions      : Array of 0/1 predictions (0=DOWN, 1=UP).
    open_prices      : Array of next-day open prices aligned with predictions.
    close_prices     : Array of next-day close prices aligned with predictions.
    dates            : DatetimeIndex aligned with predictions.
    ticker           : Stock symbol (for logging and result labelling).
    initial_capital  : Starting portfolio value in dollars.
    position_size    : Fraction of portfolio to deploy per trade (0-1).
    transaction_cost : One-way cost as fraction (0.001 = 0.1%).
    allow_short      : If True, DOWN signal → short the stock.
    config_name      : Override all params with a named BACKTEST_CONFIGS entry.
    verbose          : Print trade-by-trade summary.

    Returns
    -------
    BacktestResult dataclass with full portfolio history and trade log.
    """
    # ── Apply named config if provided ────────────────────────────────────
    if config_name is not None:
        if config_name not in BACKTEST_CONFIGS:
            raise ValueError(
                f"Config '{config_name}' not found. "
                f"Available: {list(BACKTEST_CONFIGS.keys())}"
            )
        cfg              = BACKTEST_CONFIGS[config_name]
        initial_capital  = cfg["initial_capital"]
        position_size    = cfg["position_size"]
        transaction_cost = cfg["transaction_cost"]
        allow_short      = cfg["allow_short"]

    # ── Validate inputs ────────────────────────────────────────────────────
    _validate_backtest_inputs(predictions, open_prices, close_prices, dates)

    # ── Initialise state ───────────────────────────────────────────────────
    capital        = float(initial_capital)
    trades         = []
    daily_records  = []
    n_signals      = 0

    if verbose:
        print(f"\n{'═'*60}")
        print(f"Backtest: {ticker}")
        print(f"  Period:  {dates[0].date()} → {dates[-1].date()}")
        print(f"  Days:    {len(predictions)}")
        print(f"  Capital: ${initial_capital:,.0f}")
        print(f"  Config:  pos_size={position_size:.0%} | "
              f"cost={transaction_cost:.3%} | "
              f"short={'yes' if allow_short else 'no'}")
        print(f"{'═'*60}")

    # ── Main backtest loop ─────────────────────────────────────────────────
    # Signal on day i → trade from open[i+1] to close[i+1]
    # So we iterate over days 0 to n-2 (leave last day without a trade)
    for i in range(len(predictions) - 1):
        signal     = int(predictions[i])
        trade_date = dates[i + 1]   # day we actually trade
        signal_date = dates[i]      # day signal was generated

        # Record portfolio state at start of day
        daily_records.append({
            "date":            dates[i],
            "portfolio_value": capital,
            "signal":          signal,
            "in_position":     signal != 0 if allow_short else signal == 1,
        })

        if signal == 1:  # UP signal → go long
            n_signals    += 1
            direction     = "long"
            entry_price   = float(open_prices[i + 1])
            exit_price    = float(close_prices[i + 1])
            capital_used  = capital * position_size
            shares        = capital_used / entry_price

            gross_pnl     = shares * (exit_price - entry_price)
            cost          = capital_used * transaction_cost * 2  # round trip
            net_pnl       = gross_pnl - cost

            cap_before    = capital
            capital      += net_pnl

            trades.append(Trade(
                entry_date=signal_date,
                exit_date=trade_date,
                ticker=ticker,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                shares=shares,
                gross_pnl=gross_pnl,
                transaction_cost=cost,
                net_pnl=net_pnl,
                return_pct=(exit_price - entry_price) / entry_price,
                capital_before=cap_before,
                capital_after=capital,
            ))

        elif signal == 0 and allow_short:  # DOWN signal → go short
            n_signals    += 1
            direction     = "short"
            entry_price   = float(open_prices[i + 1])
            exit_price    = float(close_prices[i + 1])
            capital_used  = capital * position_size
            shares        = capital_used / entry_price

            # Short: profit when price falls
            gross_pnl     = shares * (entry_price - exit_price)
            cost          = capital_used * transaction_cost * 2
            net_pnl       = gross_pnl - cost

            cap_before    = capital
            capital      += net_pnl

            trades.append(Trade(
                entry_date=signal_date,
                exit_date=trade_date,
                ticker=ticker,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                shares=shares,
                gross_pnl=gross_pnl,
                transaction_cost=cost,
                net_pnl=net_pnl,
                return_pct=(entry_price - exit_price) / entry_price,
                capital_before=cap_before,
                capital_after=capital,
            ))

    # Record final day
    daily_records.append({
        "date":            dates[-1],
        "portfolio_value": capital,
        "signal":          int(predictions[-1]),
        "in_position":     False,
    })

    daily_df = pd.DataFrame(daily_records).set_index("date")

    # ── Benchmark: buy and hold ────────────────────────────────────────────
    benchmark_df = _run_buy_and_hold(
        close_prices, dates, initial_capital, transaction_cost
    )

    config_used = {
        "initial_capital":   initial_capital,
        "position_size":     position_size,
        "transaction_cost":  transaction_cost,
        "allow_short":       allow_short,
    }

    result = BacktestResult(
        daily_portfolio=daily_df,
        trades=trades,
        benchmark_portfolio=benchmark_df,
        config=config_used,
        ticker=ticker,
        n_signals=n_signals,
        n_trades=len(trades),
        date_range=(dates[0], dates[-1]),
    )

    if verbose:
        final_val = daily_df["portfolio_value"].iloc[-1]
        bh_final  = benchmark_df["portfolio_value"].iloc[-1]
        print(f"  Strategy final:    ${final_val:>10,.2f}  "
              f"({(final_val/initial_capital-1)*100:+.1f}%)")
        print(f"  Buy & Hold final:  ${bh_final:>10,.2f}  "
              f"({(bh_final/initial_capital-1)*100:+.1f}%)")
        print(f"  Total trades:      {len(trades)}")
        print(f"  UP signals:        {n_signals}/{len(predictions)}")
        print(f"{'═'*60}")

    return result


def _run_buy_and_hold(
    close_prices:    np.ndarray,
    dates:           pd.DatetimeIndex,
    initial_capital: float,
    transaction_cost: float,
) -> pd.DataFrame:
    """
    Simulate buying at first close and holding until last close.
    Used as the benchmark every strategy must beat.
    """
    entry_price = float(close_prices[0])
    shares      = (initial_capital * (1 - transaction_cost)) / entry_price
    cost        = initial_capital * transaction_cost

    portfolio_values = shares * close_prices.astype(float)

    return pd.DataFrame({
        "portfolio_value": portfolio_values,
    }, index=dates)


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-STOCK BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_portfolio_backtest(
    ticker_predictions: Dict[str, np.ndarray],
    ticker_prices:      Dict[str, pd.DataFrame],
    initial_capital:    float = DEFAULT_INITIAL_CAPITAL,
    position_size:      float = 0.20,   # smaller per stock when trading many
    transaction_cost:   float = DEFAULT_TRANSACTION_COST,
    allow_short:        bool  = False,
    max_concurrent:     int   = 5,      # max positions at once
    verbose:            bool  = True,
) -> Dict[str, BacktestResult]:
    """
    Run backtest for multiple stocks simultaneously.

    Parameters
    ----------
    ticker_predictions : Dict mapping ticker → predictions array.
    ticker_prices      : Dict mapping ticker → DataFrame with open/close columns.
    initial_capital    : Total portfolio capital split across tickers.
    position_size      : Fraction of TOTAL capital per position.
    transaction_cost   : One-way cost per trade.
    allow_short        : Short on DOWN signals.
    max_concurrent     : Maximum simultaneous positions.
    verbose            : Print progress.

    Returns
    -------
    Dict mapping ticker → BacktestResult.

    Why smaller position_size for multi-stock?
    ─────────────────────────────────────────────────────────────
    With 5 concurrent positions at 20% each, 100% of capital is
    deployed. With single-stock at 95%, you risk the full portfolio
    on one trade. Multi-stock naturally diversifies risk.
    """
    results = {}
    tickers = list(ticker_predictions.keys())

    if verbose:
        print(f"\nPortfolio Backtest: {len(tickers)} stocks")
        print(f"Capital: ${initial_capital:,.0f} | "
              f"Per position: {position_size:.0%}")

    for ticker in tickers:
        if ticker not in ticker_prices:
            if verbose:
                print(f"  [{ticker}] No price data — skipping")
            continue

        prices = ticker_prices[ticker]
        preds  = ticker_predictions[ticker]

        if "open" not in prices.columns or "close" not in prices.columns:
            if verbose:
                print(f"  [{ticker}] Missing open/close columns — skipping")
            continue

        # Align predictions with price data
        n    = min(len(preds), len(prices))
        preds_aligned  = preds[:n]
        open_aligned   = prices["open"].values[:n]
        close_aligned  = prices["close"].values[:n]
        dates_aligned  = prices.index[:n]

        result = run_backtest(
            predictions=preds_aligned,
            open_prices=open_aligned,
            close_prices=close_aligned,
            dates=dates_aligned,
            ticker=ticker,
            initial_capital=initial_capital / len(tickers),
            position_size=position_size,
            transaction_cost=transaction_cost,
            allow_short=allow_short,
            verbose=False,
        )

        results[ticker] = result

        if verbose:
            final = result.daily_portfolio["portfolio_value"].iloc[-1]
            ret   = (final / (initial_capital / len(tickers)) - 1) * 100
            print(f"  [{ticker}] Return: {ret:+.1f}% | "
                  f"Trades: {result.n_trades}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_backtest_config(name: str) -> Dict:
    """Retrieve a named backtest configuration."""
    if name not in BACKTEST_CONFIGS:
        raise ValueError(
            f"Config '{name}' not found. "
            f"Available: {list(BACKTEST_CONFIGS.keys())}"
        )
    return dict(BACKTEST_CONFIGS[name])


def list_backtest_configs(verbose: bool = True) -> List[str]:
    """List all available backtest configurations."""
    names = list(BACKTEST_CONFIGS.keys())
    if verbose:
        print(f"\n{'Config':<20} {'PosSize':>8} {'Cost':>8} "
              f"{'Short':>6}  Description")
        print("─" * 75)
        for name, cfg in BACKTEST_CONFIGS.items():
            print(
                f"  {name:<18} {cfg['position_size']:>7.0%} "
                f"{cfg['transaction_cost']:>7.3%} "
                f"{'yes' if cfg['allow_short'] else 'no':>6}  "
                f"{cfg['description']}"
            )
    return names


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    import yfinance as yf
    from data.cleaner import clean_stock_data
    from features.engineer import build_features
    from features.indicators import get_model_features
    from data.labeller import create_labels
    from models.trainer import train, _three_way_split

    # ── List configs
    list_backtest_configs()

    # ── Prepare data
    raw = yf.download("AAPL", period="2y", auto_adjust=True, progress=False)
    raw.columns = [c.lower() for c in raw.columns]
    clean    = clean_stock_data(raw, ticker="AAPL")
    featured = build_features(clean).dropna()
    labelled = create_labels(featured, horizon=1,
                             threshold=0.003, verbose=False)
    X = get_model_features(labelled, extra_drop=["target"]).fillna(0)
    y = labelled["target"]

    # ── Train
    pipeline, _ = train(X, y, verbose=False)

    # ── Get test split
    _, _, X_test, _, _, y_test = _three_way_split(X, y)
    preds, _ = pipeline.predict(X_test.reset_index(drop=True)), None
    preds    = pipeline.predict(X_test.reset_index(drop=True))

    # Get aligned prices for test period
    test_dates = (pd.DatetimeIndex(
        X_test.index.get_level_values("date")
        if isinstance(X_test.index, pd.MultiIndex)
        else X_test.index
    ))

    price_test = clean.loc[
        clean.index.isin(test_dates)
    ][["open", "close"]].values

    open_prices  = price_test[:len(preds), 0]
    close_prices = price_test[:len(preds), 1]
    dates        = pd.DatetimeIndex(test_dates[:len(preds)])

    # ── Run backtest
    result = run_backtest(
        predictions=preds,
        open_prices=open_prices,
        close_prices=close_prices,
        dates=dates,
        ticker="AAPL",
        config_name="default",
        verbose=True,
    )

    print(f"\nTrade log (first 5 trades):")
    for trade in result.trades[:5]:
        print(f"  {trade.entry_date.date()} → {trade.exit_date.date()}: "
              f"net P&L ${trade.net_pnl:+.2f} "
              f"({trade.return_pct*100:+.2f}%)")
