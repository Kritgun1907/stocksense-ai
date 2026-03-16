import pandas as pd
import numpy as np
from typing import List
from features.indicators import (
    add_trend_features,
    add_momentum_features
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline.
    Takes clean OHLCV + sentiment DataFrame.
    Returns DataFrame with all features ready for ML model.

    Called once per stock during training,
    and once per stock during live prediction.
    """
    df = df.copy()

    # Each of these functions gets built in Chapters 2.2-2.6
    df = add_trend_features(df)        # Chapter 2.2 — moving averages, EMA
    df = add_momentum_features(df)     # Chapter 2.2 — RSI, rate of change
    df = add_macd_features(df)         # Chapter 2.3 — MACD
    df = add_volatility_features(df)   # Chapter 2.4 — Bollinger, ATR
    df = add_volume_features(df)       # Chapter 2.4 — volume signals
    df = add_pattern_features(df)      # Chapter 2.5 — candlestick patterns
    df = add_lag_features(df)          # Chapter 2.6 — memory features
    df = add_return_features(df)       # Chapter 2.6 — pct change features

    # Drop rows with NaN from rolling windows
    # (first N rows will have NaN until rolling window fills up)
    df = df.dropna()

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return list of feature column names only.
    Excludes raw OHLCV, ticker, and target columns.
    These are the columns fed into the ML model.
    """
    exclude = ['open', 'high', 'low', 'close', 'volume',
               'ticker', 'target']
    return [col for col in df.columns if col not in exclude]


# Placeholder functions — each gets fully implemented
# in their respective chapters

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    return df  # implemented in Chapter 2.2

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    return df  # implemented in Chapter 2.2

def add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    return df  # implemented in Chapter 2.3

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    return df  # implemented in Chapter 2.4

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    return df  # implemented in Chapter 2.4

def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    return df  # implemented in Chapter 2.5

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    return df  # implemented in Chapter 2.6

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    return df  # implemented in Chapter 2.6

# This master file is the **conductor.** Every chapter from 2.2 to 2.6 will fill in one of these functions. By the end of Phase 2, calling `build_features(df)` on any stock will give you a fully engineered ML-ready dataset in one line.

