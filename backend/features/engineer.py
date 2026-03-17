"""
StockSense AI — features/engineer.py
======================================
Master feature engineering pipeline — the single conductor.

This file owns ALL orchestration logic:
  - What functions run
  - In what order they run
  - Why each step is where it is

indicators.py contains ONLY the feature calculation logic.
This file is the only place that decides the call sequence.

Usage
─────
    from features.engineer import build_features, get_feature_columns

    df = build_features(raw_ohlcv_df)          # full pipeline
    feature_cols = get_feature_columns(df)      # cols to feed the model
"""

import pandas as pd
import numpy as np
from typing import List

from features.indicators import (
    # ── Indicator modules (must run before candlestick — patterns read these)
    add_trend_features,        # 1  SMA/EMA, crosses, slopes, ADX, agreement
    add_momentum_features,     # 2  RSI multi-period, ROC
    add_macd_features,         # 3  MACD + divergence
    add_volatility_features,   # 4  Bollinger Bands, ATR, Keltner, vol regime
    add_volume_features,       # 5  volume ratio, OBV, VWAP

    # ── Candlestick modules (depend on indicators above)
    add_candle_geometry,       # 6  continuous candle anatomy ratios
    add_pattern_features,      # 7  binary pattern flags (reads 1–6)
    add_pattern_strength,      # 8  continuous quality scores per pattern
    add_interaction_features,  # 9  pattern × regime cross-products
    add_sequence_features,     # 10 direction streaks, pattern clustering
    add_sr_proximity,          # 11 distance to rolling S/R levels

    # ── Memory / return modules (run last — lag the final feature set)
    add_lag_features,          # 12 lagged copies of key indicators
    add_return_features,       # 13 log returns, rolling stats, drawdown

    # ── Utilities
    get_model_features,
    RAW_COLUMNS_TO_DROP,
    FEATURE_GROUPS,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline.

    Takes a clean OHLCV DataFrame (+ optional sentiment columns)
    and returns a fully engineered ML-ready DataFrame.

    Called once per stock during training,
    and once per stock during live prediction.

    Dependency order explained
    ──────────────────────────
    Steps 1–5  (indicators) must run BEFORE step 6–11 (candlestick).
    Reason: add_pattern_features uses trend_agreement (from step 1)
            and rsi_14 (step 2) and bb_percent (step 4) for context.
            Without these, patterns fall back to naive 5d price comparison.

    Step 6 (geometry) must run BEFORE steps 7–11 (patterns).
    Reason: pattern functions read body_ratio, body_abs, upper_wick etc.

    Step 8 (strength) must run AFTER step 7 (patterns).
    Reason: reads pat_hammer, pat_bull_engulf etc.

    Step 9 (interactions) must run AFTER steps 7–8.
    Reason: reads all pattern flags + all indicator values.

    Step 10 (sequence) must run AFTER step 7.
    Reason: reads bullish_pattern_count, pattern_signal, candle_direction.

    Step 12 (lags) must run AFTER ALL other features.
    Reason: lags the final feature values — must be computed last.

    Step 13 (returns) runs after lags so ret_1d can itself be lagged
    in future iterations if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close
        Optional         : volume, sentiment_* (passed through untouched)

    Returns
    -------
    pd.DataFrame with all original columns + ~340 new feature columns.
    Call df.dropna() after this to remove rows with NaN from rolling windows.
    """
    df = df.copy()

    # ── Step 1: Trend (MAs, slopes, ADX, golden/death cross) ─────────────────
    # Produces: sma_*, ema_*, price_vs_*, trend_agreement, adx_proxy, etc.
    # Needed by: add_pattern_features (trend_agreement)
    df = add_trend_features(df)

    # ── Step 2: Momentum (RSI, ROC) ───────────────────────────────────────────
    # Produces: rsi_14, rsi_oversold, rsi_overbought, roc_*, etc.
    # Needed by: add_pattern_features (rsi_14 for context)
    #            add_interaction_features (rsi_oversold, rsi_overbought)
    df = add_momentum_features(df)

    # ── Step 3: MACD (line, signal, histogram, divergence) ────────────────────
    # Produces: macd_*, bullish_divergence, bearish_divergence
    # Needed by: add_interaction_features (macd_bullish_cross)
    df = add_macd_features(df)

    # ── Step 4: Volatility (BB, ATR, Keltner, regime) ─────────────────────────
    # Produces: bb_*, atr_14, atr_pct, rv5, rv20, kc_position, etc.
    # Needed by: add_candle_geometry (atr_14 for body_vs_atr)
    #            add_interaction_features (bb_percent, bb_squeeze)
    #            add_return_features (atr_pct, rv5)
    df = add_volatility_features(df)

    # ── Step 5: Volume (ratio, OBV, VWAP, money flow) ─────────────────────────
    # Produces: volume_ratio, obv_*, dist_to_vwap, etc.
    # Needed by: add_interaction_features (volume_ratio)
    df = add_volume_features(df)

    # ── Step 6: Candle Geometry (continuous anatomy ratios) ───────────────────
    # Produces: body_ratio, body_abs, upper_wick, lower_wick,
    #           candle_direction, wick_imbalance, body_vs_atr, etc.
    # Needed by: add_pattern_features (all geometry columns)
    #            add_pattern_strength (body_ratio, lower/upper wick)
    df = add_candle_geometry(df)

    # ── Step 7: Pattern Flags (binary signals) ────────────────────────────────
    # Produces: pat_hammer, pat_bull_engulf, pat_morning_star, etc.
    #           bullish_pattern_count, bearish_pattern_count, pattern_signal
    # Context used from steps 1+2: trend_agreement, rsi_14
    # Needed by: add_pattern_strength, add_interaction_features,
    #            add_sequence_features
    df = add_pattern_features(df)

    # ── Step 8: Pattern Strength (continuous quality scores) ──────────────────
    # Produces: hammer_quality, star_quality, bull_engulf_strength, etc.
    # Turns binary flags into graded signals — XGBoost learns its own threshold
    df = add_pattern_strength(df)

    # ── Step 9: Interaction Features (pattern × regime cross-products) ─────────
    # Produces: hammer_oversold, bull_pat_at_bb_low, confirmed_bull_divergence,
    #           pat_*_vol, pat_*_adx, doji_in_squeeze, etc.
    # Encodes compound signals that would need many tree splits to reconstruct
    df = add_interaction_features(df)

    # ── Step 10: Sequence Features (temporal memory) ──────────────────────────
    # Produces: dir_streak, dir_balance_*, bull/bear_pat_5d_cluster,
    #           pct_from_high_*, pct_from_low_*, etc.
    # Injects short-horizon trajectory context that XGBoost lacks natively
    df = add_sequence_features(df)

    # ── Step 11: S/R Proximity (distance to rolling highs/lows) ───────────────
    # Produces: dist_to_high_20/50/100, dist_to_low_*, range_pos_*
    # Reversal patterns firing at S/R levels are more reliable
    df = add_sr_proximity(df)

    # ── Step 12: Lag Features (recent trajectory memory) ──────────────────────
    # Produces: rsi_14_lag1..lag10, trend_agreement_lag1..lag10, etc.
    # Must run AFTER all features are computed — lags the final values
    df = add_lag_features(df)

    # ── Step 13: Return Features (price returns at multiple horizons) ──────────
    # Produces: ret_1d..ret_20d, ret_mean_*, ret_std_*, max_drawdown_10d, etc.
    # Raw close is non-stationary — returns are comparable across all stocks
    df = add_return_features(df)

    # ── Final cleanup ─────────────────────────────────────────────────────────
    # Replace any inf that snuck through (div-by-zero on edge cases)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Caller drops NaN rows — we don't dropna() here because the caller may
    # need to align with a target column or other data first
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of feature columns to feed to the ML model.
    Excludes raw OHLCV, ticker, target, and non-stationary intermediate columns.

    Parameters
    ----------
    df : output of build_features()

    Returns
    -------
    List[str] of column names safe to pass to XGBoost / SHAP.
    """
    exclude = set(RAW_COLUMNS_TO_DROP) | {
        "open", "high", "low", "close", "volume",
        "ticker", "target",
    }
    return [col for col in df.columns if col not in exclude
            and pd.api.types.is_numeric_dtype(df[col])]