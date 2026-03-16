"""
StockSense AI — Technical Indicators
====================================
Adds moving average, trend, and momentum-based features.

Uses pandas_ta library for efficient calculation of:
  - SMA, EMA (moving averages)
  - RSI (relative strength index)
  - ROC (rate of change)
  - Golden/Death crosses
  - Overbought/oversold signals

All features are designed to be stationary and model-friendly.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


# ─── Moving Averages & Trend Features ────────────────────────
def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add moving average and trend-based features.

    Why these features?
    - SMAs capture different trend timescales (short/medium/long)
    - Price distance from MA = stationary, comparable across stocks
    - MA ratios detect trend reversals (uptrend vs downtrend)
    - Golden/Death crosses = classic reversal signals
    - Rising MA = trend confirmation

    Features added:
      sma_20, sma_50, sma_200   — simple moving averages
      ema_9, ema_21, ema_50     — exponential (more responsive)
      price_vs_sma20/50/200     — % distance from MA (stationary)
      price_vs_ema21            — price vs fast EMA
      sma20_sma50_ratio         — short vs medium trend strength
      sma50_sma200_ratio        — medium vs long trend strength
      ema9_ema21_ratio          — fast vs medium momentum
      golden_cross              — SMA 50 > 200 cross up (bullish)
      death_cross               — SMA 50 < 200 cross down (bearish)
      ema_crossover             — EMA 9 > 21 cross up (fast signal)
      sma20_rising, sma50_rising, ema21_rising — MA momentum
    """
    df = df.copy()

    # ── Raw Moving Averages ────────────────────────────────────
    df["sma_20"]  = ta.sma(df["close"], length=20)
    df["sma_50"]  = ta.sma(df["close"], length=50)
    df["sma_200"] = ta.sma(df["close"], length=200)
    df["ema_9"]   = ta.ema(df["close"], length=9)
    df["ema_21"]  = ta.ema(df["close"], length=21)
    df["ema_50"]  = ta.ema(df["close"], length=50)

    # ── Price Distance from MAs (% based — stationary) ─────────
    # Positive = price above MA (bullish)
    # Negative = price below MA (bearish)
    # Example: if close=100, sma_20=95 → price_vs_sma20 = +5.26%
    df["price_vs_sma20"]  = (df["close"] - df["sma_20"])  / df["sma_20"]  * 100
    df["price_vs_sma50"]  = (df["close"] - df["sma_50"])  / df["sma_50"]  * 100
    df["price_vs_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"] * 100
    df["price_vs_ema21"]  = (df["close"] - df["ema_21"])  / df["ema_21"]  * 100

    # ── MA Ratios (trend strength) ──────────────────────────────
    # > 1 = short term above long term = uptrend
    # < 1 = short term below long term = downtrend
    # More stable than raw MA prices, model learns trend direction
    df["sma20_sma50_ratio"]  = df["sma_20"] / df["sma_50"]
    df["sma50_sma200_ratio"] = df["sma_50"] / df["sma_200"]
    df["ema9_ema21_ratio"]   = df["ema_9"]  / df["ema_21"]

    # ── Golden Cross / Death Cross (binary flags) ───────────────
    # Goldencross = classic bullish reversal signal
    # Detected when SMA 50 crosses above SMA 200
    df["golden_cross"] = (
        (df["sma_50"] > df["sma_200"]) &
        (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
    ).astype(int)

    # Death cross = classic bearish reversal signal
    df["death_cross"] = (
        (df["sma_50"] < df["sma_200"]) &
        (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
    ).astype(int)

    # ── EMA Crossover (faster signal) ──────────────────────────
    # EMA 9 crossing above EMA 21 = early bullish signal
    df["ema_crossover"] = (
        (df["ema_9"] > df["ema_21"]) &
        (df["ema_9"].shift(1) <= df["ema_21"].shift(1))
    ).astype(int)

    # ── Trend Direction (is the MA itself rising?) ─────────────
    # Model learns: rising MA = continuing trend
    df["sma20_rising"]  = (df["sma_20"]  > df["sma_20"].shift(1)).astype(int)
    df["sma50_rising"]  = (df["sma_50"]  > df["sma_50"].shift(1)).astype(int)
    df["ema21_rising"]  = df["ema_21"].diff().apply(
                          lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    return df


# ─── RSI & Momentum Features ──────────────────────────────────
def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI and momentum-based features.

    Why RSI?
    - RSI 14 = standard momentum oscillator, 0–100 range
    - RSI > 70 = overbought (often before pullback)
    - RSI < 30 = oversold (often before bounce)
    - RSI momentum = direction of RSI itself (not just value)
    - Multiple timeframes = catch fast & slow momentum

    Features added:
      rsi_7, rsi_14, rsi_21     — standard RSI at 3 timeframes
      rsi_overbought            — RSI > 70 (binary flag)
      rsi_oversold              — RSI < 30 (binary flag)
      rsi_neutral               — 40 < RSI < 60 (neutral zone)
      rsi_momentum              — 1-day change in RSI
      rsi_momentum_3d           — 3-day change in RSI
      rsi_normalized            — RSI / 100 (0–1 scale)
      roc_5, roc_10, roc_20     — rate of change (price momentum %)
      rsi_fast_slow_diff        — RSI 7 minus RSI 21 (momentum acceleration)
    """
    df = df.copy()

    # ── RSI at multiple timeframes ─────────────────────────────
    df["rsi_7"]  = ta.rsi(df["close"], length=7)   # fast — more signals
    df["rsi_14"] = ta.rsi(df["close"], length=14)  # standard
    df["rsi_21"] = ta.rsi(df["close"], length=21)  # slow — fewer false signals

    # ── RSI Zone Flags (binary — model learns optimal thresholds) ───
    # Standard levels: overbought=70, oversold=30
    # But model can learn better thresholds for this specific stock
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)
    df["rsi_neutral"]    = (
        (df["rsi_14"] >= 40) & (df["rsi_14"] <= 60)
    ).astype(int)

    # ── RSI Momentum (is RSI rising or falling?) ───────────────
    # Even if RSI is at 50 (neutral), if it's rising rapidly → momentum
    # This captures divergences (price falling, RSI rising = bullish divergence)
    df["rsi_momentum"]    = df["rsi_14"].diff(1)   # 1-day change in RSI
    df["rsi_momentum_3d"] = df["rsi_14"].diff(3)   # 3-day change in RSI

    # ── RSI normalized to 0-1 range (cleaner for neural networks) ────
    df["rsi_normalized"] = df["rsi_14"] / 100

    # ── Rate of Change (momentum in price terms) ───────────────
    # ROC = (close_today - close_N_days_ago) / close_N_days_ago
    # Measures how fast the stock is moving (positive = rallying, negative = falling)
    df["roc_5"]  = ta.roc(df["close"], length=5)   # 1 week momentum
    df["roc_10"] = ta.roc(df["close"], length=10)  # 2 weeks momentum
    df["roc_20"] = ta.roc(df["close"], length=20)  # 1 month momentum

    # ── RSI fast-slow diff (momentum acceleration) ──────────────
    # When RSI_7 >> RSI_21, momentum is accelerating
    # Useful for catching trend changes early
    df["rsi_fast_slow_diff"] = df["rsi_7"] - df["rsi_21"]

    return df

# ─── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf

    # Download price data
    print("Downloading AAPL data...")
    df = yf.download("AAPL", period="1y", auto_adjust=True, progress=False)
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    # Apply features
    print("Calculating trend features...")
    df = add_trend_features(df)

    print("Calculating momentum features...")
    df = add_momentum_features(df)

    # Drop NaN rows from rolling windows
    df = df.dropna()

    # ── Display Results ─────────────────────────────────────────
    print("\n=== TREND FEATURES ===")
    print(df[
        ["close", "sma_20", "sma_50", "price_vs_sma20",
         "sma20_sma50_ratio", "golden_cross"]
    ].tail(10).to_string())

    print("\n=== MOMENTUM FEATURES ===")
    print(df[
        ["close", "rsi_14", "rsi_overbought",
         "rsi_oversold", "rsi_momentum", "roc_10"]
    ].tail(10).to_string())

    print(f"\n✅ Total features added: {len(df.columns) - 6}")  # -6 for original OHLCV
    print(f"📊 Columns: {list(df.columns)}")
