"""
Test suite for technical indicator features.
"""
import pandas as pd
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
        "close", "sma_20", "rsi_14", "atr_14", "bb_position"
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
        "sma_20", "rsi_14", "atr_14", "bb_position",
        "volatility_20d", "roc_10", "rsi_momentum"
    ]
    print(df[indicators].describe().round(3).to_string()) 