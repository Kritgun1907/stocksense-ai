"""
Test suite for technical indicator features.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
from indicators import add_all_indicators


# ─── Educational: RSI from scratch ────────────────────────────
def calculate_rsi_scratch(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI from scratch — educational/reference version."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ─── Test Suite ───────────────────────────────────────────────
if __name__ == "__main__":
    # Download AAPL data
    print("Downloading AAPL data (1 year)...")
    df = yf.download("AAPL", period="1y", auto_adjust=True, progress=False)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    # Test 1: Educational RSI from scratch
    print("\n=== Test 1: RSI from scratch ===")
    df["rsi_scratch"] = calculate_rsi_scratch(df["close"], 14)
    print(df[["close", "rsi_scratch"]].tail(5))

    # Test 2: All indicators
    print("\n=== Test 2: All technical indicators ===")
    df = add_all_indicators(df)
    df = df.dropna()

    print(f"✅ Features added: {len(df.columns) - 6}")  # -6 for original OHLCV
    print(f"\n📊 Last 5 rows of indicator data:")
    print(df[[
        "close", "sma_20", "rsi_14", "macd_line", "macd_histogram"
    ]].tail(5).to_string())

    # Test 3: Verification — compare pandas_ta RSI with scratch RSI
    print("\n=== Test 3: RSI comparison (pandas_ta vs scratch) ===")
    comparison = df[["rsi_14", "rsi_scratch"]].tail(10)
    comparison["diff"] = (comparison["rsi_14"] - comparison["rsi_scratch"]).abs()
    print(comparison.to_string())
    max_diff = comparison["diff"].max()
    print(f"\nMax difference: {max_diff:.4f} (should be < 0.5)")

    # Test 4: Feature statistics
    print("\n=== Test 4: Feature statistics ===")
    indicators = [
        "sma_20", "rsi_14", "roc_10", "rsi_momentum",
        "macd_line", "macd_histogram"
    ]
    print(df[indicators].describe().round(3).to_string())

    # Test 5: MACD feature columns
    # Standard MACD (12, 26, 9)
    print("\n=== Test 5: MACD feature columns ===")
    macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
    print(macd_data.columns)
    # ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
    #   ↑ MACD line      ↑ Histogram       ↑ Signal line

    # Add to DataFrame
    df['macd_line']      = macd_data['MACD_12_26_9']
    df['macd_signal']    = macd_data['MACDs_12_26_9']
    df['macd_histogram'] = macd_data['MACDh_12_26_9']
    print(df[["close", "macd_line", "macd_signal", "macd_histogram"]].tail(10).to_string())
    

    # Reading the output:
    # macd_line positive  → short term trend bullish
    # macd_line > signal  → momentum accelerating
    # histogram growing   → momentum strengthening
    # histogram shrinking → momentum fading (watch for reversal)
    
    
    
# Simply put: Volatility measures how wildly a stock's price swings up and down over time.

# Everyday Analogy
# Think of two cars on a road:

# Low volatility = smooth highway, steady speed, predictable
# High volatility = mountain road with sharp turns, speed changes constantly, unpredictable


# What it looks like in practice
# Stock A (Low Volatility - e.g. Coca-Cola):
# $100 → $101 → $100 → $102 → $101   (small, calm moves)

# Stock B (High Volatility - e.g. a crypto or small tech stock):
# $100 → $120 → $85 → $130 → $90    (big, wild swings)