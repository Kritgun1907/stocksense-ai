"""
Test engineer.py → run full feature engineering pipeline through indicators.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os

# Add backend to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.engineer import build_features, get_feature_columns, FEATURE_GROUPS


def test_full_pipeline():
    """Run the complete feature engineering pipeline on real data."""
    print("=" * 80)
    print("STOCKSENSE AI — FEATURE ENGINEERING PIPELINE TEST")
    print("=" * 80)

    # ── Download sample data ──────────────────────────────────────────────────
    print("\n📥 Downloading AAPL 1-year data...")
    df = yf.download("AAPL", period="1y", auto_adjust=True, progress=False)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    print(f"   Initial shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    # ── Run the full pipeline ─────────────────────────────────────────────────
    print("\n🔧 Building features through engineer.py...")
    df_features = build_features(df)

    print(f"   After build_features: {df_features.shape}")
    print(f"   New columns added: {df_features.shape[1] - df.shape[1]}")

    # ── Drop NaN from rolling windows ─────────────────────────────────────────
    print("\n🧹 Dropping NaN from rolling windows...")
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    dropped = initial_rows - len(df_features)
    print(f"   Rows before: {initial_rows}")
    print(f"   Rows after:  {len(df_features)}")
    print(f"   Dropped:     {dropped} (from rolling windows)")

    # ── Get model-ready features ──────────────────────────────────────────────
    print("\n🎯 Extracting model features (non-stationary cols dropped)...")
    model_features = get_feature_columns(df_features)
    print(f"   Model-ready features: {len(model_features)}")

    # ── Display feature breakdown by group ─────────────────────────────────────
    print("\n📊 FEATURE BREAKDOWN BY GROUP")
    print("-" * 80)
    total_grouped = 0
    for group_name in sorted(FEATURE_GROUPS.keys()):
        cols = [c for c in FEATURE_GROUPS[group_name] if c in df_features.columns]
        count = len(cols)
        total_grouped += count
        pct = (count / len(model_features) * 100) if model_features else 0
        print(f"  {group_name:15} {count:3} features  ({pct:5.1f}%)")
        if count <= 5:  # Show small groups
            for c in cols:
                print(f"    └─ {c}")

    print(f"\n  TOTAL (grouped): {total_grouped} / {len(model_features)}")

    # ── Sample of each group ──────────────────────────────────────────────────
    print("\n📈 SAMPLE VALUES — LAST 5 ROWS")
    print("=" * 80)

    sample_groups = {
        "Trend": ["price_vs_sma20", "price_vs_sma50", "trend_agreement", "golden_cross"],
        "Momentum": ["rsi_14", "rsi_normalized", "roc_10"],
        "MACD": ["macd_line_pct", "macd_bullish_cross", "bullish_divergence"],
        "Volatility": ["bb_percent", "atr_pct", "high_volatility"],
        "Volume": ["volume_ratio", "obv_change_5d", "dist_to_vwap"],
        "Geometry": ["body_ratio", "wick_imbalance", "candle_direction"],
        "Patterns": ["pattern_signal", "bullish_pattern_count", "pat_hammer"],
        "Interactions": ["hammer_oversold", "confirmed_bull_divergence"],
        "Sequence": ["dir_streak", "dir_balance_5"],
        "Returns": ["ret_1d", "ret_5d", "max_drawdown_10d"],
    }

    for section_name, cols in sample_groups.items():
        cols = [c for c in cols if c in df_features.columns]
        if not cols:
            continue
        print(f"\n{section_name}:")
        print(df_features[cols].tail(5).to_string())

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n\n📐 FEATURE STATISTICS")
    print("=" * 80)
    print(f"\nTotal columns (after build_features): {df_features.shape[1]}")
    print(f"Model-ready features:                 {len(model_features)}")
    print(f"Rows (after dropna):                  {len(df_features)}")

    # Count NaN, inf per feature
    nan_counts = df_features[model_features].isna().sum()
    inf_counts = df_features[model_features].isin([np.inf, -np.inf]).sum().sum()
    print(f"\nData quality:")
    print(f"  Features with NaN:  {(nan_counts > 0).sum()} / {len(model_features)}")
    print(f"  Total NaN values:   {nan_counts.sum()}")
    print(f"  Inf values:         {inf_counts}")

    if (nan_counts > 0).sum() > 0:
        print(f"\n  NaN-containing features:")
        for col in nan_counts[nan_counts > 0].head(10).index:
            print(f"    {col:30} {nan_counts[col]:6} NaN")

    # ── Correlation check ─────────────────────────────────────────────────────
    print("\n\n🔗 FEATURE CORRELATION SAMPLE")
    print("=" * 80)
    correlation_sample = [
        "price_vs_sma20", "rsi_14", "macd_line_pct", "bb_percent",
        "volume_ratio", "pattern_signal", "ret_1d"
    ]
    correlation_sample = [c for c in correlation_sample if c in df_features.columns]
    if len(correlation_sample) >= 2:
        print("\nCorrelation matrix (sample features):")
        print(df_features[correlation_sample].corr().round(3).to_string())

    # ── Distribution check ────────────────────────────────────────────────────
    print("\n\n📉 FEATURE RANGE CHECK (first 15 model features)")
    print("=" * 80)
    for col in model_features[:15]:
        vals = df_features[col].dropna()
        if len(vals) > 0:
            print(
                f"  {col:30} "
                f"min={vals.min():9.2f}  mean={vals.mean():9.2f}  "
                f"max={vals.max():9.2f}  std={vals.std():9.2f}"
            )

    # ── Final validation ──────────────────────────────────────────────────────
    print("\n\n✅ VALIDATION SUMMARY")
    print("=" * 80)
    checks = {
        "Shape (rows × cols)": f"{df_features.shape[0]} × {df_features.shape[1]}",
        "All numeric features": all(
            pd.api.types.is_numeric_dtype(df_features[col])
            for col in model_features
        ),
        "No inf values": df_features[model_features].isin([np.inf, -np.inf]).sum().sum() == 0,
        "Model features count": len(model_features),
        "OHLCV removed": "close" not in model_features,
    }

    for check_name, result in checks.items():
        status = "✓" if (isinstance(result, bool) and result) or isinstance(result, (int, str)) else "✗"
        print(f"  {status} {check_name:25} {result}")

    print("\n" + "=" * 80)
    print("✅ PIPELINE TEST COMPLETE!")
    print("=" * 80)

    return df_features, model_features


if __name__ == "__main__":
    df_engineered, features = test_full_pipeline()
