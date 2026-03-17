"""
StockSense AI — features/indicators.py
=======================================
Pure feature logic library.

This file contains ONLY feature calculation functions.
It has NO orchestration, NO pipeline, NO chaining of modules.

ALL call ordering lives in engineer.py → build_features().
This file is imported by engineer.py for its functions only.

Public functions (imported by engineer.py):
─────────────────────────────────────────────────────────────
  Indicators (must run first — patterns depend on these):
    add_trend_features        SMA/EMA, crosses, slopes, ADX, trend agreement
    add_momentum_features     RSI multi-period, ROC, acceleration
    add_macd_features         MACD line/signal/histogram, divergence
    add_volatility_features   Bollinger Bands, ATR, Keltner, vol regime
    add_volume_features       Volume ratio, OBV, money flow, VWAP

  Candlestick (run after indicators so context is available):
    add_candle_geometry       Continuous candle anatomy ratios
    add_pattern_features      All pattern flags (single/two/three candle)
    add_pattern_strength      Continuous quality scores per pattern
    add_interaction_features  Pattern × regime cross-products
    add_sequence_features     Direction streaks, pattern clustering
    add_sr_proximity          Distance to rolling S/R levels

  Memory / return (run last):
    add_lag_features          Lagged copies of key indicators
    add_return_features       Log returns, rolling stats, drawdown

Utilities (used by engineer.py after build_features):
    get_model_features        Drop non-stationary cols → XGBoost/SHAP ready
    RAW_COLUMNS_TO_DROP       List of raw cols removed by get_model_features
    FEATURE_GROUPS            Dict: group name → list of feature columns

Required input columns : open, high, low, close
Optional               : volume  (unlocks all volume features)
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_div(a: pd.Series, b, fill: float = 0.0) -> pd.Series:
    """Division with zero / NaN safety."""
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
    elif b == 0:
        return pd.Series(fill, index=a.index)
    return a.div(b).fillna(fill)


def _rolling_rank(s: pd.Series, window: int) -> pd.Series:
    """Percentile rank of each value within its rolling window (0–1)."""
    return s.rolling(window, min_periods=max(1, window // 2)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


# ══════════════════════════════════════════════════════════════════════════════
#  1. TREND FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moving average and trend features.

    Columns added
    ─────────────
    Raw MAs (dropped before training — needed by other features as intermediates):
      sma_20, sma_50, sma_200, ema_9, ema_21, ema_50

    Stationary features (% distance from MA):
      price_vs_sma20, price_vs_sma50, price_vs_sma200
      price_vs_ema9,  price_vs_ema21, price_vs_ema50

    Ratio features (trend direction):
      sma20_sma50_ratio    >1 = short above medium = uptrend
      sma50_sma200_ratio   >1 = medium above long  = uptrend
      ema9_ema21_ratio     >1 = fast above medium  = momentum

    Event crosses (1 on crossover day, 0 otherwise):
      golden_cross, death_cross, ema9_ema21_cross

    Direction flags:
      sma20_rising, sma50_rising, ema21_rising

    Slope features (momentum of the MA itself):
      ma5_slope, ma10_slope, ma20_slope, ma50_slope
      lr_slope_5, lr_slope_10, lr_slope_20   (linear regression)

    Directional strength (ADX-style, no TA-Lib):
      di_plus, di_minus, adx_proxy

    Swing structure:
      hh_streak   higher-high streak length
      ll_streak   lower-low streak length

    Consensus:
      trend_agreement   fraction of MAs agreeing on direction (−1 to +1)
    """
    df    = df.copy()
    close = df["close"]

    # ── Raw MAs ───────────────────────────────────────────────────────────────
    df["sma_20"]  = ta.sma(close, length=20)
    df["sma_50"]  = ta.sma(close, length=50)
    df["sma_200"] = ta.sma(close, length=200)
    df["ema_9"]   = ta.ema(close, length=9)
    df["ema_21"]  = ta.ema(close, length=21)
    df["ema_50"]  = ta.ema(close, length=50)

    # ── % Distance from MA ────────────────────────────────────────────────────
    for ma_col, out_col in [
        ("sma_20",  "price_vs_sma20"),
        ("sma_50",  "price_vs_sma50"),
        ("sma_200", "price_vs_sma200"),
        ("ema_9",   "price_vs_ema9"),
        ("ema_21",  "price_vs_ema21"),
        ("ema_50",  "price_vs_ema50"),
    ]:
        df[out_col] = _safe_div(close - df[ma_col], df[ma_col]) * 100

    # ── Ratios ────────────────────────────────────────────────────────────────
    df["sma20_sma50_ratio"]  = _safe_div(df["sma_20"], df["sma_50"],  fill=1.0)
    df["sma50_sma200_ratio"] = _safe_div(df["sma_50"], df["sma_200"], fill=1.0)
    df["ema9_ema21_ratio"]   = _safe_div(df["ema_9"],  df["ema_21"],  fill=1.0)

    # ── Crosses ───────────────────────────────────────────────────────────────
    df["golden_cross"] = (
        (df["sma_50"] > df["sma_200"]) &
        (df["sma_50"].shift(1) <= df["sma_200"].shift(1))
    ).astype(int)

    df["death_cross"] = (
        (df["sma_50"] < df["sma_200"]) &
        (df["sma_50"].shift(1) >= df["sma_200"].shift(1))
    ).astype(int)

    df["ema9_ema21_cross"] = (
        (df["ema_9"] > df["ema_21"]) &
        (df["ema_9"].shift(1) <= df["ema_21"].shift(1))
    ).astype(int)

    # ── Direction flags ───────────────────────────────────────────────────────
    df["sma20_rising"] = (df["sma_20"] > df["sma_20"].shift(1)).astype(int)
    df["sma50_rising"] = (df["sma_50"] > df["sma_50"].shift(1)).astype(int)
    df["ema21_rising"] = np.sign(df["ema_21"].diff()).fillna(0).astype(int)

    # ── MA slopes ─────────────────────────────────────────────────────────────
    for w, src in [(5, "sma_20"), (10, "sma_50"), (20, "sma_200"), (50, "sma_200")]:
        df[f"ma{w}_slope"] = _safe_div(df[src].diff(3), df[src].shift(3)) * 100

    # ── Linear-regression price slope ─────────────────────────────────────────
    for w in [5, 10, 20]:
        df[f"lr_slope_{w}"] = (
            close.rolling(w, min_periods=w).apply(
                lambda y: np.polyfit(np.arange(len(y)), y, 1)[0]
                          / (y.mean() + 1e-10) * 100,
                raw=True,
            )
        )

    # ── ADX-style directional strength ────────────────────────────────────────
    high, low = df["high"], df["low"]
    dh = high.diff()
    dl = low.diff().abs()
    dm_plus  = dh.where(dh > dl, 0.0).clip(lower=0)
    dm_minus = dl.where(dl > dh, 0.0).clip(lower=0)
    atr14    = (high - low).rolling(14).mean().replace(0, np.nan)
    df["di_plus"]   = _safe_div(dm_plus.rolling(14).mean(),  atr14) * 100
    df["di_minus"]  = _safe_div(dm_minus.rolling(14).mean(), atr14) * 100
    df["adx_proxy"] = (
        _safe_div(
            (df["di_plus"] - df["di_minus"]).abs(),
            df["di_plus"] + df["di_minus"],
        ) * 100
    ).rolling(14).mean()

    # ── Swing streaks ─────────────────────────────────────────────────────────
    hh = (high > high.shift(1)).astype(int)
    ll = (low  < low.shift(1) ).astype(int)
    df["hh_streak"] = hh.groupby((hh == 0).cumsum()).cumsum()
    df["ll_streak"] = ll.groupby((ll == 0).cumsum()).cumsum()

    # ── Trend agreement score ─────────────────────────────────────────────────
    sign_cols = ["price_vs_sma20", "price_vs_sma50",
                 "price_vs_ema21", "price_vs_ema50"]
    df["trend_agreement"] = (
        df[sign_cols].apply(np.sign).sum(axis=1) / len(sign_cols)
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2. MOMENTUM FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI and rate-of-change features.

    Columns added
    ─────────────
    rsi_7, rsi_14, rsi_21
    rsi_overbought (>70), rsi_oversold (<30), rsi_neutral (40–60)
    rsi_momentum (1d diff), rsi_momentum_3d (3d diff)
    rsi_normalized (÷100, 0–1 scale)
    rsi_fast_slow_diff  RSI7 − RSI21 = momentum acceleration
    roc_3, roc_5, roc_10, roc_20
    """
    df    = df.copy()
    close = df["close"]

    df["rsi_7"]  = ta.rsi(close, length=7)
    df["rsi_14"] = ta.rsi(close, length=14)
    df["rsi_21"] = ta.rsi(close, length=21)

    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)
    df["rsi_neutral"]    = (
        (df["rsi_14"] >= 40) & (df["rsi_14"] <= 60)
    ).astype(int)

    df["rsi_momentum"]       = df["rsi_14"].diff(1)
    df["rsi_momentum_3d"]    = df["rsi_14"].diff(3)
    df["rsi_normalized"]     = df["rsi_14"] / 100
    df["rsi_fast_slow_diff"] = df["rsi_7"] - df["rsi_21"]

    for w in [3, 5, 10, 20]:
        df[f"roc_{w}"] = ta.roc(close, length=w)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  3. MACD FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD (12, 26, 9) features.

    Columns added
    ─────────────
    Raw (dropped before training):
      macd_line, macd_signal, macd_histogram

    Stationary (normalised by price):
      macd_line_pct, macd_signal_pct, macd_histogram_pct

    Crosses:
      macd_bullish_cross, macd_bearish_cross
      macd_zero_cross_up, macd_zero_cross_down

    Position flags:
      macd_above_signal, macd_above_zero

    Histogram dynamics:
      histogram_momentum (1d diff), histogram_growing (binary)

    Divergence:
      bullish_divergence, bearish_divergence
    """
    df    = df.copy()
    close = df["close"]

    macd_data = ta.macd(close, fast=12, slow=26, signal=9)
    df["macd_line"]      = macd_data["MACD_12_26_9"]
    df["macd_signal"]    = macd_data["MACDs_12_26_9"]
    df["macd_histogram"] = macd_data["MACDh_12_26_9"]

    for raw, pct in [
        ("macd_line",      "macd_line_pct"),
        ("macd_signal",    "macd_signal_pct"),
        ("macd_histogram", "macd_histogram_pct"),
    ]:
        df[pct] = _safe_div(df[raw], close) * 100

    df["macd_bullish_cross"] = (
        (df["macd_line"] > df["macd_signal"]) &
        (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)

    df["macd_bearish_cross"] = (
        (df["macd_line"] < df["macd_signal"]) &
        (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
    ).astype(int)

    df["macd_zero_cross_up"] = (
        (df["macd_line"] > 0) & (df["macd_line"].shift(1) <= 0)
    ).astype(int)

    df["macd_zero_cross_down"] = (
        (df["macd_line"] < 0) & (df["macd_line"].shift(1) >= 0)
    ).astype(int)

    df["macd_above_signal"] = (df["macd_line"] > df["macd_signal"]).astype(int)
    df["macd_above_zero"]   = (df["macd_line"] > 0).astype(int)

    df["histogram_momentum"] = df["macd_histogram"].diff(1)
    df["histogram_growing"]  = (
        df["macd_histogram"].abs() > df["macd_histogram"].abs().shift(1)
    ).astype(int)

    # ── Divergence ────────────────────────────────────────────────────────────
    w  = 10
    mp = max(1, w // 2)
    p_min = close.rolling(w, min_periods=mp).min().shift(1)
    p_max = close.rolling(w, min_periods=mp).max().shift(1)
    h_min = df["macd_histogram"].rolling(w, min_periods=mp).min().shift(1)
    h_max = df["macd_histogram"].rolling(w, min_periods=mp).max().shift(1)

    df["bullish_divergence"] = (
        (close < p_min) & (df["macd_histogram"] > h_min)
    ).astype(int)
    df["bearish_divergence"] = (
        (close > p_max) & (df["macd_histogram"] < h_max)
    ).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  4. VOLATILITY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger Bands, ATR, Keltner Channel, volatility regime.

    Columns added
    ─────────────
    Raw band levels (dropped before training):
      bb_upper, bb_middle, bb_lower

    Stationary Bollinger features:
      bb_bandwidth, bb_percent
      bb_near_upper, bb_near_lower, bb_above_upper, bb_below_lower
      bb_squeeze     bottom-20% of 252d bandwidth history
      bb_bandwidth_change (3d pct change), bb_expanding

    ATR:
      atr_14     raw (dropped before training)
      atr_pct    normalised by price — comparable across stocks
      atr_ratio  current vs 50d average ATR

    Realised volatility:
      rv5, rv20          annualised %, 5d and 20d windows
      vol_regime         rv5 ÷ rv20  (>1 = vol expanding)

    Keltner Channel:
      kc_position        0 = lower band, 1 = upper band

    Regime flags:
      high_volatility, low_volatility

    Range percentile:
      range_percentile   today's range vs 20d history (0–1)
    """
    df    = df.copy()
    close = df["close"]
    high, low = df["high"], df["low"]

    bbands  = ta.bbands(close, length=20, std=2)
    bb_cols = bbands.columns.tolist()
    df["bb_upper"]     = bbands[bb_cols[0]]
    df["bb_middle"]    = bbands[bb_cols[1]]
    df["bb_lower"]     = bbands[bb_cols[2]]
    df["bb_bandwidth"] = bbands[bb_cols[3]]
    df["bb_percent"]   = bbands[bb_cols[4]]

    df["bb_near_upper"]  = (df["bb_percent"] > 0.80).astype(int)
    df["bb_near_lower"]  = (df["bb_percent"] < 0.20).astype(int)
    df["bb_above_upper"] = (df["bb_percent"] > 1.00).astype(int)
    df["bb_below_lower"] = (df["bb_percent"] < 0.00).astype(int)

    bb_q20           = df["bb_bandwidth"].rolling(252, min_periods=50).quantile(0.20)
    df["bb_squeeze"] = (df["bb_bandwidth"] < bb_q20).astype(int)

    df["bb_bandwidth_change"] = df["bb_bandwidth"].pct_change(3)
    df["bb_expanding"]        = (
        df["bb_bandwidth"] > df["bb_bandwidth"].shift(1)
    ).astype(int)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    df["atr_14"]    = tr.rolling(14).mean()
    df["atr_pct"]   = _safe_div(df["atr_14"], close) * 100
    df["atr_ratio"] = _safe_div(df["atr_14"], df["atr_14"].rolling(50).mean(), fill=1.0)

    log_ret      = np.log(close / close.shift(1))
    df["rv5"]    = log_ret.rolling(5).std()  * np.sqrt(252) * 100
    df["rv20"]   = log_ret.rolling(20).std() * np.sqrt(252) * 100
    df["vol_regime"] = _safe_div(df["rv5"], df["rv20"], fill=1.0)

    ma20     = close.rolling(20).mean()
    kc_upper = ma20 + df["atr_14"] * 2
    kc_lower = ma20 - df["atr_14"] * 2
    df["kc_position"] = _safe_div(close - kc_lower, kc_upper - kc_lower)

    atr_75 = df["atr_pct"].rolling(252, min_periods=50).quantile(0.75)
    atr_25 = df["atr_pct"].rolling(252, min_periods=50).quantile(0.25)
    df["high_volatility"] = (df["atr_pct"] > atr_75).astype(int)
    df["low_volatility"]  = (df["atr_pct"] < atr_25).astype(int)

    df["range_percentile"] = _rolling_rank((high - low).fillna(0), 20)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  5. VOLUME FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume ratio, OBV, money flow, VWAP.
    Returns df unchanged if 'volume' column is absent.

    Columns added
    ─────────────
    volume_ma20  (dropped before training)
    volume_ratio      today ÷ 20d average
    volume_spike (>2×), volume_extreme_spike (>3×), volume_low (<0.5×)
    volume_trend      MA rising over 5d
    obv_change_5d, obv_change_20d, obv_above_ma
    price_vol_confirm  price moved + volume > 1.2×
    close_position     0 = closed at low, 1 = closed at high
    bullish_vol_conf, bearish_vol_conf
    vol_percentile     0–1 rank in 20d window
    money_flow_ma10    (raw money_flow dropped)
    vwap_20  (dropped before training), dist_to_vwap
    """
    if "volume" not in df.columns:
        return df

    df    = df.copy()
    close = df["close"]
    vol   = df["volume"].replace(0, np.nan)

    df["volume_ma20"]  = vol.rolling(20).mean()
    df["volume_ratio"] = _safe_div(vol, df["volume_ma20"], fill=1.0)

    df["volume_spike"]         = (df["volume_ratio"] > 2.0).astype(int)
    df["volume_extreme_spike"] = (df["volume_ratio"] > 3.0).astype(int)
    df["volume_low"]           = (df["volume_ratio"] < 0.5).astype(int)
    df["volume_trend"]         = (
        df["volume_ma20"] > df["volume_ma20"].shift(5)
    ).astype(int)

    obv = ta.obv(close, vol.fillna(0))
    df["obv_change_5d"]  = obv.pct_change(5)
    df["obv_change_20d"] = obv.pct_change(20)
    df["obv_above_ma"]   = (obv > obv.rolling(20).mean()).astype(int)

    daily_ret = close.pct_change()
    df["price_vol_confirm"] = (
        ((daily_ret > 0) | (daily_ret < 0)) & (df["volume_ratio"] > 1.2)
    ).astype(int)

    df["close_position"] = _safe_div(
        close - df["low"], df["high"] - df["low"], fill=0.5
    )

    direction = np.sign(daily_ret).fillna(0)
    df["bullish_vol_conf"] = (direction ==  1).astype(float) * df["volume_ratio"].fillna(1)
    df["bearish_vol_conf"] = (direction == -1).astype(float) * df["volume_ratio"].fillna(1)

    df["vol_percentile"] = _rolling_rank(vol.fillna(0), 20)

    typical            = (df["high"] + df["low"] + close) / 3
    df["money_flow"]   = typical * vol.fillna(0)
    df["money_flow_ma10"] = df["money_flow"].rolling(10).mean()

    vwap_num      = (typical * vol.fillna(0)).rolling(20).sum()
    vwap_den      = vol.fillna(0).rolling(20).sum().replace(0, np.nan)
    df["vwap_20"] = vwap_num / vwap_den
    df["dist_to_vwap"] = _safe_div(close - df["vwap_20"], df["vwap_20"]) * 100

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  6. CANDLE GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def add_candle_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Continuous structural measurements of each candle.
    Must run before add_pattern_features.

    Why continuous instead of binary?
    XGBoost finds its own thresholds. Giving it body_ratio=0.72 is more
    informative than just hammer=1.

    Columns added
    ─────────────
    Intermediates (dropped before training):
      body_size, body_abs, upper_wick, lower_wick, candle_range, body_abs_ma5

    Ratios (0–1 within the candle's range):
      body_ratio          body ÷ range
      upper_wick_ratio    upper wick ÷ range
      lower_wick_ratio    lower wick ÷ range
      wick_imbalance      (upper − lower) ÷ range  (+ = top heavy)
      body_position       0 = body at bottom of range, 1 = top
      body_midpoint_pos   midpoint of body within range

    Direction and magnitude:
      candle_direction    +1 bull / 0 doji / −1 bear
      close_open_pct      % move from open to close

    Volatility-scaled:
      body_vs_atr         body size ÷ ATR  (requires add_volatility_features first)
      range_vs_atr        candle range ÷ ATR

    Trend:
      body_size_trend     today's body ÷ 5d average body − 1  (growing conviction?)
    """
    df = df.copy()

    body       = df["close"] - df["open"]
    body_abs   = body.abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    rng        = (df["high"] - df["low"]).replace(0, np.nan)

    # ── Intermediates (needed by pattern functions) ────────────────────────────
    df["body_size"]     = body
    df["body_abs"]      = body_abs
    df["upper_wick"]    = upper_wick
    df["lower_wick"]    = lower_wick
    df["candle_range"]  = rng.fillna(0)

    # ── Continuous ratios ─────────────────────────────────────────────────────
    df["body_ratio"]        = _safe_div(body_abs, rng)
    df["upper_wick_ratio"]  = _safe_div(upper_wick, rng)
    df["lower_wick_ratio"]  = _safe_div(lower_wick, rng)
    df["wick_imbalance"]    = _safe_div(upper_wick - lower_wick, rng)
    df["body_position"]     = _safe_div(
        df[["open", "close"]].min(axis=1) - df["low"], rng
    )
    df["body_midpoint_pos"] = _safe_div(
        (df["open"] + df["close"]) / 2 - df["low"], rng
    )

    df["candle_direction"] = np.sign(body).fillna(0).astype(int)
    df["close_open_pct"]   = _safe_div(body, df["open"]) * 100

    # ── Volatility-scaled (uses atr_14 if available) ──────────────────────────
    if "atr_14" in df.columns:
        atr = df["atr_14"].replace(0, np.nan)
        df["body_vs_atr"]  = _safe_div(body_abs, atr)
        df["range_vs_atr"] = _safe_div(df["candle_range"], atr)
    else:
        df["body_vs_atr"]  = np.nan
        df["range_vs_atr"] = np.nan

    # ── Body conviction trend ─────────────────────────────────────────────────
    df["body_abs_ma5"]    = body_abs.rolling(5).mean()
    df["body_size_trend"] = _safe_div(body_abs, df["body_abs_ma5"]) - 1

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  7. PATTERN FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary candlestick pattern flags.
    Requires add_candle_geometry to have run first.
    Context-aware: uses trend_agreement + rsi_14 when available.

    Columns added
    ─────────────
    Single-candle:
      pat_doji, pat_gravestone_doji, pat_dragonfly_doji
      pat_hammer, pat_inv_hammer, pat_shooting_star, pat_hanging_man
      pat_spinning_top
      pat_bull_marubozu, pat_bear_marubozu

    Two-candle:
      pat_bull_engulf, pat_bear_engulf
      pat_piercing, pat_dark_cloud
      pat_bull_harami, pat_bear_harami
      pat_tweezer_top, pat_tweezer_bottom

    Three-candle:
      pat_morning_star, pat_evening_star
      pat_3white_soldiers, pat_3black_crows

    Structure:
      pat_inside_bar, pat_outside_bar

    Summary:
      bullish_pattern_count, bearish_pattern_count, pattern_signal
    """
    df = df.copy()

    body_abs   = df["body_abs"]
    upper_wick = df["upper_wick"]
    lower_wick = df["lower_wick"]
    rng        = df["candle_range"].replace(0, np.nan)
    is_bull    = (df["candle_direction"] == 1)
    is_bear    = (df["candle_direction"] == -1)

    # ── Rich context when earlier modules have run ────────────────────────────
    if "trend_agreement" in df.columns and "rsi_14" in df.columns:
        downtrend = (df["trend_agreement"] < -0.25) & (df["rsi_14"].fillna(50) < 50)
        uptrend   = (df["trend_agreement"] >  0.25) & (df["rsi_14"].fillna(50) > 50)
    elif "trend_agreement" in df.columns:
        downtrend = df["trend_agreement"] < -0.25
        uptrend   = df["trend_agreement"] >  0.25
    else:
        downtrend = df["close"] < df["close"].shift(5)
        uptrend   = df["close"] > df["close"].shift(5)

    # ── Single-candle patterns ────────────────────────────────────────────────
    df["pat_doji"]            = (df["body_ratio"] <= 0.10).astype(int)
    df["pat_gravestone_doji"] = (
        (df["pat_doji"] == 1) & (lower_wick <= rng * 0.05)
    ).astype(int)
    df["pat_dragonfly_doji"]  = (
        (df["pat_doji"] == 1) & (upper_wick <= rng * 0.05)
    ).astype(int)

    hammer_shape = (
        (lower_wick >= 2 * body_abs) &
        (upper_wick <= body_abs * 0.3) &
        (df["body_ratio"] <= 0.35) &
        (rng > df["close"] * 0.001)
    )
    inv_hammer_shape = (
        (upper_wick >= 2 * body_abs) &
        (lower_wick <= body_abs * 0.3) &
        (df["body_ratio"] <= 0.35)
    )

    df["pat_hammer"]        = (hammer_shape     & downtrend).astype(int)
    df["pat_inv_hammer"]    = (inv_hammer_shape & downtrend).astype(int)
    df["pat_shooting_star"] = (inv_hammer_shape & uptrend  ).astype(int)
    df["pat_hanging_man"]   = (hammer_shape     & uptrend  ).astype(int)

    df["pat_spinning_top"]  = (
        (df["body_ratio"] <= 0.30) &
        (upper_wick >= rng * 0.20) &
        (lower_wick >= rng * 0.20) &
        (df["body_ratio"] > 0.10)
    ).astype(int)

    df["pat_bull_marubozu"] = (is_bull & (df["body_ratio"] >= 0.95)).astype(int)
    df["pat_bear_marubozu"] = (is_bear & (df["body_ratio"] >= 0.95)).astype(int)

    # ── Two-candle patterns ───────────────────────────────────────────────────
    pb_abs  = body_abs.shift(1)
    pb_dir  = df["candle_direction"].shift(1)
    p_open  = df["open"].shift(1)
    p_close = df["close"].shift(1)
    p_bull  = (pb_dir ==  1)
    p_bear  = (pb_dir == -1)

    df["pat_bull_engulf"]    = (
        p_bear & is_bull &
        (df["open"] < p_close) & (df["close"] > p_open) &
        (body_abs > pb_abs * 1.1)
    ).astype(int)
    df["pat_bear_engulf"]    = (
        p_bull & is_bear &
        (df["open"] > p_close) & (df["close"] < p_open) &
        (body_abs > pb_abs * 1.1)
    ).astype(int)
    df["pat_piercing"]       = (
        p_bear & is_bull &
        (df["open"] < p_close) &
        (df["close"] > (p_open + p_close) / 2) &
        (df["close"] < p_open)
    ).astype(int)
    df["pat_dark_cloud"]     = (
        p_bull & is_bear &
        (df["open"] > p_close) &
        (df["close"] < (p_open + p_close) / 2) &
        (df["close"] > p_open)
    ).astype(int)
    df["pat_bull_harami"]    = (
        p_bear & is_bull &
        (df["open"] > p_close) & (df["close"] < p_open) &
        (body_abs < pb_abs * 0.5)
    ).astype(int)
    df["pat_bear_harami"]    = (
        p_bull & is_bear &
        (df["open"] < p_close) & (df["close"] > p_open) &
        (body_abs < pb_abs * 0.5)
    ).astype(int)
    df["pat_tweezer_top"]    = (
        p_bull & is_bear &
        ((df["high"] - df["high"].shift(1)).abs() <= rng * 0.05)
    ).astype(int)
    df["pat_tweezer_bottom"] = (
        p_bear & is_bull &
        ((df["low"] - df["low"].shift(1)).abs() <= rng * 0.05)
    ).astype(int)

    # ── Three-candle patterns ─────────────────────────────────────────────────
    p2_open  = df["open"].shift(2)
    p2_close = df["close"].shift(2)
    p2_bull  = (df["candle_direction"].shift(2) ==  1)
    p2_bear  = (df["candle_direction"].shift(2) == -1)
    p2_babs  = body_abs.shift(2)

    df["pat_morning_star"] = (
        p2_bear & (p2_babs > df["close"] * 0.01) &
        (pb_abs < p2_babs * 0.5) & is_bull &
        (df["close"] > (p2_open + p2_close) / 2) &
        downtrend.shift(2)
    ).astype(int)
    df["pat_evening_star"] = (
        p2_bull & (p2_babs > df["close"] * 0.01) &
        (pb_abs < p2_babs * 0.5) & is_bear &
        (df["close"] < (p2_open + p2_close) / 2) &
        uptrend.shift(2)
    ).astype(int)
    df["pat_3white_soldiers"] = (
        is_bull & p_bull & p2_bull &
        (df["close"] > p_close) & (p_close > p2_close) &
        (body_abs > df["close"] * 0.005) &
        (pb_abs   > df["close"] * 0.005) &
        (p2_babs  > df["close"] * 0.005)
    ).astype(int)
    df["pat_3black_crows"] = (
        is_bear & p_bear & p2_bear &
        (df["close"] < p_close) & (p_close < p2_close) &
        (body_abs > df["close"] * 0.005) &
        (pb_abs   > df["close"] * 0.005) &
        (p2_babs  > df["close"] * 0.005)
    ).astype(int)

    # ── Structure bars ────────────────────────────────────────────────────────
    df["pat_inside_bar"]  = (
        (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    ).astype(int)
    df["pat_outside_bar"] = (
        (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    ).astype(int)

    # ── Summary counts ────────────────────────────────────────────────────────
    _bull = ["pat_hammer", "pat_inv_hammer", "pat_dragonfly_doji",
             "pat_bull_engulf", "pat_piercing", "pat_bull_harami",
             "pat_tweezer_bottom", "pat_morning_star",
             "pat_3white_soldiers", "pat_bull_marubozu"]
    _bear = ["pat_shooting_star", "pat_hanging_man", "pat_gravestone_doji",
             "pat_bear_engulf", "pat_dark_cloud", "pat_bear_harami",
             "pat_tweezer_top", "pat_evening_star",
             "pat_3black_crows", "pat_bear_marubozu"]

    df["bullish_pattern_count"] = df[[c for c in _bull if c in df]].sum(axis=1)
    df["bearish_pattern_count"] = df[[c for c in _bear if c in df]].sum(axis=1)
    df["pattern_signal"]        = df["bullish_pattern_count"] - df["bearish_pattern_count"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  8. PATTERN STRENGTH SCORES
# ══════════════════════════════════════════════════════════════════════════════

def add_pattern_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Continuous quality scores per pattern.
    XGBoost learns its own optimal threshold rather than using hard cutoffs.
    Requires add_candle_geometry and add_pattern_features first.

    Columns added
    ─────────────
    hammer_quality       lower wick ÷ range (higher = longer shadow = stronger)
    star_quality         upper wick ÷ range
    bull_engulf_strength day-2 body ÷ day-1 body − 1 (how much bigger)
    bear_engulf_strength same for bearish engulfing
    doji_indecision      1 − body_ratio/0.10  (closer to zero body = more uncertain)
    marubozu_conviction  body_ratio where marubozu fired (higher = more decisive)
    """
    df  = df.copy()
    rng = df["candle_range"].replace(0, np.nan)
    _z  = pd.Series(0, index=df.index)

    df["hammer_quality"] = np.where(
        (df.get("pat_hammer", _z) == 1) | (df.get("pat_hanging_man", _z) == 1),
        _safe_div(df["lower_wick"], rng), 0.0
    )
    df["star_quality"] = np.where(
        (df.get("pat_shooting_star", _z) == 1) | (df.get("pat_inv_hammer", _z) == 1),
        _safe_div(df["upper_wick"], rng), 0.0
    )
    df["bull_engulf_strength"] = np.where(
        df.get("pat_bull_engulf", _z) == 1,
        (_safe_div(df["body_abs"], df["body_abs"].shift(1)) - 1).clip(0, 5), 0.0
    )
    df["bear_engulf_strength"] = np.where(
        df.get("pat_bear_engulf", _z) == 1,
        (_safe_div(df["body_abs"], df["body_abs"].shift(1)) - 1).clip(0, 5), 0.0
    )
    df["doji_indecision"]     = np.clip(1 - df["body_ratio"] / 0.10, 0, 1)
    df["marubozu_conviction"] = np.where(
        (df.get("pat_bull_marubozu", _z) == 1) |
        (df.get("pat_bear_marubozu", _z) == 1),
        df["body_ratio"], 0.0
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  9. INTERACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit cross-products: pattern × regime context.
    Requires all indicator modules + add_pattern_features to have run.

    Why cross-products?
    XGBoost needs many splits to learn "hammer + oversold + high volume = strong".
    One interaction column encodes all three simultaneously, cutting required
    tree depth and making the signal SHAP-attributable to a single feature.

    Columns added
    ─────────────
    Pattern × Volume:
      pat_hammer_vol, pat_bull_engulf_vol, pat_morning_star_vol,
      pat_3white_soldiers_vol, pat_bull_marubozu_vol,
      pat_shooting_star_vol, pat_bear_engulf_vol, pat_evening_star_vol,
      pat_3black_crows_vol, pat_bear_marubozu_vol

    Pattern × ADX strength:
      pat_hammer_adx, pat_bull_engulf_adx,
      pat_shooting_star_adx, pat_bear_engulf_adx

    Pattern × BB extreme (mean-reversion context):
      bull_pat_at_bb_low   bullish count × (1 − bb_percent)
      bear_pat_at_bb_high  bearish count × bb_percent

    Pattern × RSI zone:
      hammer_oversold, bull_engulf_oversold, morning_star_oversold
      shooting_star_overbought, bear_engulf_overbought, evening_star_overbought

    Divergence + pattern confirmation:
      confirmed_bull_divergence, confirmed_bear_divergence

    Pattern in volatility squeeze:
      doji_in_squeeze, spinning_top_in_squeeze

    Pattern × Trend alignment:
      soldiers_trend_align, crows_trend_align

    Pattern + MACD cross:
      bull_pat_macd_cross, bear_pat_macd_cross
    """
    df = df.copy()
    _z = pd.Series(0.0, index=df.index)

    vol_r  = df.get("volume_ratio",    _z + 1).fillna(1).clip(0, 5)
    adx_n  = df.get("adx_proxy",       _z).fillna(0) / 100
    bb_p   = df.get("bb_percent",      _z + 0.5).fillna(0.5).clip(0, 1)
    rsi_os = df.get("rsi_oversold",    _z).fillna(0)
    rsi_ob = df.get("rsi_overbought",  _z).fillna(0)
    t_agr  = df.get("trend_agreement", _z).fillna(0)
    bb_sq  = df.get("bb_squeeze",      _z).fillna(0)
    bc     = df.get("bullish_pattern_count", _z)
    be     = df.get("bearish_pattern_count", _z)

    # ── Pattern × Volume ──────────────────────────────────────────────────────
    for pat in ["pat_hammer", "pat_bull_engulf", "pat_morning_star",
                "pat_3white_soldiers", "pat_bull_marubozu",
                "pat_shooting_star", "pat_bear_engulf", "pat_evening_star",
                "pat_3black_crows", "pat_bear_marubozu"]:
        if pat in df.columns:
            df[f"{pat}_vol"] = df[pat] * vol_r

    # ── Pattern × ADX ─────────────────────────────────────────────────────────
    for pat in ["pat_hammer", "pat_bull_engulf",
                "pat_shooting_star", "pat_bear_engulf"]:
        if pat in df.columns:
            df[f"{pat}_adx"] = df[pat] * adx_n

    # ── Pattern × BB extreme ──────────────────────────────────────────────────
    df["bull_pat_at_bb_low"]  = bc * (1 - bb_p)
    df["bear_pat_at_bb_high"] = be * bb_p

    # ── Pattern × RSI zone ────────────────────────────────────────────────────
    for pat, col in [
        ("pat_hammer",       "hammer_oversold"),
        ("pat_bull_engulf",  "bull_engulf_oversold"),
        ("pat_morning_star", "morning_star_oversold"),
    ]:
        df[col] = df.get(pat, _z) * rsi_os

    for pat, col in [
        ("pat_shooting_star", "shooting_star_overbought"),
        ("pat_bear_engulf",   "bear_engulf_overbought"),
        ("pat_evening_star",  "evening_star_overbought"),
    ]:
        df[col] = df.get(pat, _z) * rsi_ob

    # ── Divergence + pattern confirmation ─────────────────────────────────────
    if "bullish_divergence" in df.columns:
        df["confirmed_bull_divergence"] = (
            (df["bullish_divergence"] == 1) & (bc >= 1)
        ).astype(int)
        df["confirmed_bear_divergence"] = (
            (df["bearish_divergence"] == 1) & (be >= 1)
        ).astype(int)

    # ── Pattern in squeeze ────────────────────────────────────────────────────
    df["doji_in_squeeze"]        = df.get("pat_doji",         _z) * bb_sq
    df["spinning_top_in_squeeze"] = df.get("pat_spinning_top", _z) * bb_sq

    # ── Trend alignment ───────────────────────────────────────────────────────
    df["soldiers_trend_align"] = df.get("pat_3white_soldiers", _z) * t_agr.clip(0, 1)
    df["crows_trend_align"]    = df.get("pat_3black_crows",     _z) * (-t_agr).clip(0, 1)

    # ── MACD cross confirmation ───────────────────────────────────────────────
    if "macd_bullish_cross" in df.columns:
        df["bull_pat_macd_cross"] = bc * df["macd_bullish_cross"]
        df["bear_pat_macd_cross"] = be * df["macd_bearish_cross"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  10. SEQUENCE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Short-horizon temporal memory for XGBoost.
    XGBoost has no built-in sequence awareness — these features
    inject recent trajectory explicitly.

    Columns added
    ─────────────
    Rolling direction counts:
      bull_candles_3/5/10   green candle count in last N days
      bear_candles_3/5/10   red candle count
      dir_balance_3/5/10    bull − bear  (−N to +N)

    Momentum streak:
      dir_streak   +N = N consecutive green, −N = N consecutive red

    Pattern clusters:
      bull_pat_5d_cluster   bullish pattern count over 5 days
      bear_pat_5d_cluster   bearish pattern count over 5 days
      pat_signal_5d         bull_cluster − bear_cluster
      pat_signal_3d_mean    rolling 3d mean of pattern_signal

    Price proximity to extremes:
      pct_from_high_10/20   % below 10/20d high
      pct_from_low_10/20    % above 10/20d low
    """
    df  = df.copy()
    d   = df["candle_direction"]
    bc  = df.get("bullish_pattern_count", pd.Series(0, index=df.index))
    be  = df.get("bearish_pattern_count", pd.Series(0, index=df.index))
    ps  = df.get("pattern_signal",        pd.Series(0, index=df.index))

    for w in [3, 5, 10]:
        df[f"bull_candles_{w}"] = (d ==  1).rolling(w).sum()
        df[f"bear_candles_{w}"] = (d == -1).rolling(w).sum()
        df[f"dir_balance_{w}"]  = df[f"bull_candles_{w}"] - df[f"bear_candles_{w}"]

    chg = (d != d.shift(1)).astype(int)
    grp = chg.cumsum()
    df["dir_streak"] = d * (d.groupby(grp).cumcount() + 1)

    df["bull_pat_5d_cluster"] = bc.rolling(5).sum()
    df["bear_pat_5d_cluster"] = be.rolling(5).sum()
    df["pat_signal_5d"]       = df["bull_pat_5d_cluster"] - df["bear_pat_5d_cluster"]
    df["pat_signal_3d_mean"]  = ps.rolling(3).mean()

    close = df["close"]
    for w in [10, 20]:
        rh = df["high"].rolling(w).max()
        rl = df["low"].rolling(w).min()
        df[f"pct_from_high_{w}"] = _safe_div(rh - close, rh)    * 100
        df[f"pct_from_low_{w}"]  = _safe_div(close - rl, rl)    * 100

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  11. SUPPORT / RESISTANCE PROXIMITY
# ══════════════════════════════════════════════════════════════════════════════

def add_sr_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distance to rolling highs / lows.
    Reversal patterns firing at key levels have higher reliability.

    Columns added
    ─────────────
    dist_to_high_20/50/100   % below rolling high
    dist_to_low_20/50/100    % above rolling low
    range_pos_20/50/100      0 = at low, 1 = at high (position in range)
    """
    df    = df.copy()
    close = df["close"]

    for w in [20, 50, 100]:
        mp = max(1, w // 2)
        rh = df["high"].rolling(w, min_periods=mp).max()
        rl = df["low"].rolling(w,  min_periods=mp).min()
        df[f"dist_to_high_{w}"] = _safe_div(rh - close, close) * 100
        df[f"dist_to_low_{w}"]  = _safe_div(close - rl, close)  * 100
        df[f"range_pos_{w}"]    = _safe_div(close - rl, rh - rl)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  12. LAG FEATURES
# ══════════════════════════════════════════════════════════════════════════════

_LAG_TARGETS = [
    "close",
    "rsi_14", "rsi_normalized",
    "macd_histogram_pct", "macd_above_signal",
    "price_vs_sma20", "price_vs_sma50", "trend_agreement",
    "bb_percent", "atr_pct", "vol_regime",
    "volume_ratio", "obv_change_5d",
    "body_ratio", "wick_imbalance", "candle_direction",
    "pattern_signal", "bullish_pattern_count", "bearish_pattern_count",
    "dir_streak", "dir_balance_5",
]

_LAG_PERIODS = [1, 2, 3, 5, 10]


def add_lag_features(
    df: pd.DataFrame,
    features: list = None,
    periods: list = None,
) -> pd.DataFrame:
    """
    Lagged copies of key features — gives XGBoost explicit trajectory memory.

    Without lags, XGBoost cannot tell whether RSI=35 is recovering from 22
    (bullish) or falling from 65 (bearish). Lags make the path explicit.

    Parameters
    ----------
    features : columns to lag  (default: _LAG_TARGETS)
    periods  : lag periods in days  (default: [1, 2, 3, 5, 10])

    Columns added
    ─────────────
    {feature}_lag{n}  e.g. rsi_14_lag1 … rsi_14_lag10
    """
    df      = df.copy()
    targets = [f for f in (features or _LAG_TARGETS) if f in df.columns]
    lags    = periods or _LAG_PERIODS

    lag_cols = {
        f"{feat}_lag{n}": df[feat].shift(n)
        for feat in targets
        for n in lags
    }
    df = pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  13. RETURN FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price return features at multiple horizons.

    Raw close is non-stationary and leaks price-level info.
    Log returns are comparable across all stocks and time periods.

    Columns added
    ─────────────
    Log returns:
      ret_1d, ret_2d, ret_3d, ret_5d, ret_10d, ret_20d   (in %)

    Volatility-normalised returns:
      ret_1d_norm   ret_1d ÷ atr_pct   (how many ATRs moved today?)
      ret_5d_norm   ret_5d ÷ rv5

    Rolling return statistics:
      ret_mean_5d/10d/20d
      ret_std_5d/10d/20d
      ret_skew_5d            return distribution asymmetry

    Win/loss metrics:
      ret_positive_5d/10d    fraction of up days
      up_down_ratio_10d      avg up-day ÷ avg down-day magnitude

    Drawdown:
      max_drawdown_10d       peak-to-trough over last 10 days (%)

    Momentum change:
      ret_acceleration       today's return − yesterday's return
    """
    df    = df.copy()
    close = df["close"]

    for n in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{n}d"] = np.log(close / close.shift(n)) * 100

    if "atr_pct" in df.columns:
        df["ret_1d_norm"] = _safe_div(df["ret_1d"], df["atr_pct"])
    if "rv5" in df.columns:
        df["ret_5d_norm"] = _safe_div(df["ret_5d"], df["rv5"])

    ret1 = df["ret_1d"]
    for w in [5, 10, 20]:
        df[f"ret_mean_{w}d"] = ret1.rolling(w).mean()
        df[f"ret_std_{w}d"]  = ret1.rolling(w).std()

    df["ret_skew_5d"] = ret1.rolling(5).skew()

    for w in [5, 10]:
        df[f"ret_positive_{w}d"] = (ret1 > 0).rolling(w).mean()

    up_mean   = ret1.clip(lower=0).rolling(10).mean()
    down_mean = ret1.clip(upper=0).abs().rolling(10).mean().replace(0, np.nan)
    df["up_down_ratio_10d"] = _safe_div(up_mean, down_mean, fill=1.0)

    df["max_drawdown_10d"] = (
        close.rolling(10).max() - close
    ).div(close.rolling(10).max().replace(0, np.nan)) * 100

    df["ret_acceleration"] = df["ret_1d"] - df["ret_1d"].shift(1)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL-READY FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

RAW_COLUMNS_TO_DROP = [
    # Raw MAs — use price_vs_* and ratio versions instead
    "sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "ema_50",
    # Raw BB levels — use bb_percent, bb_bandwidth instead
    "bb_upper", "bb_middle", "bb_lower",
    # Raw MACD levels — use *_pct versions instead
    "macd_line", "macd_signal", "macd_histogram",
    # Raw ATR — use atr_pct instead
    "atr_14",
    # Raw volume MA — use volume_ratio instead
    "volume_ma20",
    # Raw candle measurements — use ratio versions instead
    "body_size", "body_abs", "upper_wick", "lower_wick",
    # Intermediates
    "body_abs_ma5", "money_flow",
    # Raw VWAP — use dist_to_vwap instead
    "vwap_20",
    # Raw close — use ret_* and price_vs_* instead
    "close",
]


def get_model_features(
    df: pd.DataFrame,
    extra_drop: list = None,
    keep_ohlcv: bool = False,
) -> pd.DataFrame:
    """
    Return a model-ready DataFrame with non-stationary columns removed.

    Parameters
    ----------
    df         : output of engineer.py build_features()
    extra_drop : additional columns to drop (e.g. 'target', 'ticker')
    keep_ohlcv : if True, keep open/high/low/volume (close already dropped)

    Returns
    -------
    pd.DataFrame — all numeric, no inf, XGBoost and SHAP ready.
    """
    to_drop = list(RAW_COLUMNS_TO_DROP)
    if extra_drop:
        to_drop += list(extra_drop)
    if not keep_ohlcv:
        to_drop += ["open", "high", "low", "volume"]

    to_drop = [c for c in to_drop if c in df.columns]
    out = df.drop(columns=to_drop).select_dtypes(include=[np.number])
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE GROUP REGISTRY  — for SHAP grouping / ablation
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "trend": [
        "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
        "price_vs_ema9", "price_vs_ema21", "price_vs_ema50",
        "sma20_sma50_ratio", "sma50_sma200_ratio", "ema9_ema21_ratio",
        "golden_cross", "death_cross", "ema9_ema21_cross",
        "sma20_rising", "sma50_rising", "ema21_rising",
        "ma5_slope", "ma10_slope", "ma20_slope", "ma50_slope",
        "lr_slope_5", "lr_slope_10", "lr_slope_20",
        "di_plus", "di_minus", "adx_proxy",
        "hh_streak", "ll_streak", "trend_agreement",
    ],
    "momentum": [
        "rsi_7", "rsi_14", "rsi_21",
        "rsi_overbought", "rsi_oversold", "rsi_neutral",
        "rsi_momentum", "rsi_momentum_3d", "rsi_normalized",
        "rsi_fast_slow_diff", "roc_3", "roc_5", "roc_10", "roc_20",
    ],
    "macd": [
        "macd_line_pct", "macd_signal_pct", "macd_histogram_pct",
        "macd_bullish_cross", "macd_bearish_cross",
        "macd_zero_cross_up", "macd_zero_cross_down",
        "macd_above_signal", "macd_above_zero",
        "histogram_momentum", "histogram_growing",
        "bullish_divergence", "bearish_divergence",
    ],
    "volatility": [
        "bb_bandwidth", "bb_percent",
        "bb_near_upper", "bb_near_lower", "bb_above_upper", "bb_below_lower",
        "bb_squeeze", "bb_bandwidth_change", "bb_expanding",
        "atr_pct", "atr_ratio", "rv5", "rv20", "vol_regime",
        "kc_position", "high_volatility", "low_volatility", "range_percentile",
    ],
    "volume": [
        "volume_ratio", "volume_spike", "volume_extreme_spike", "volume_low",
        "volume_trend", "obv_change_5d", "obv_change_20d", "obv_above_ma",
        "price_vol_confirm", "close_position",
        "bullish_vol_conf", "bearish_vol_conf", "vol_percentile",
        "money_flow_ma10", "dist_to_vwap",
    ],
    "geometry": [
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "wick_imbalance", "body_position", "body_midpoint_pos",
        "candle_direction", "close_open_pct",
        "body_vs_atr", "range_vs_atr", "body_size_trend",
    ],
    "patterns": [
        "pat_doji", "pat_gravestone_doji", "pat_dragonfly_doji",
        "pat_hammer", "pat_inv_hammer", "pat_shooting_star", "pat_hanging_man",
        "pat_spinning_top", "pat_bull_marubozu", "pat_bear_marubozu",
        "pat_bull_engulf", "pat_bear_engulf", "pat_piercing", "pat_dark_cloud",
        "pat_bull_harami", "pat_bear_harami", "pat_tweezer_top", "pat_tweezer_bottom",
        "pat_morning_star", "pat_evening_star",
        "pat_3white_soldiers", "pat_3black_crows",
        "pat_inside_bar", "pat_outside_bar",
        "bullish_pattern_count", "bearish_pattern_count", "pattern_signal",
    ],
    "strength": [
        "hammer_quality", "star_quality",
        "bull_engulf_strength", "bear_engulf_strength",
        "doji_indecision", "marubozu_conviction",
    ],
    "interactions": [
        "bull_pat_at_bb_low", "bear_pat_at_bb_high",
        "hammer_oversold", "bull_engulf_oversold", "morning_star_oversold",
        "shooting_star_overbought", "bear_engulf_overbought",
        "evening_star_overbought",
        "confirmed_bull_divergence", "confirmed_bear_divergence",
        "doji_in_squeeze", "spinning_top_in_squeeze",
        "soldiers_trend_align", "crows_trend_align",
        "bull_pat_macd_cross", "bear_pat_macd_cross",
    ],
    "sequence": [
        "dir_balance_3", "dir_balance_5", "dir_balance_10",
        "dir_streak",
        "bull_pat_5d_cluster", "bear_pat_5d_cluster",
        "pat_signal_5d", "pat_signal_3d_mean",
        "pct_from_high_10", "pct_from_high_20",
        "pct_from_low_10", "pct_from_low_20",
    ],
    "sr_proximity": [
        "dist_to_high_20", "dist_to_low_20", "range_pos_20",
        "dist_to_high_50", "dist_to_low_50", "range_pos_50",
        "dist_to_high_100", "dist_to_low_100", "range_pos_100",
    ],
    "lags": [
        f"{f}_lag{n}"
        for f in ["rsi_14", "pattern_signal", "trend_agreement",
                  "bb_percent", "volume_ratio", "candle_direction",
                  "ret_1d", "macd_histogram_pct"]
        for n in _LAG_PERIODS
    ],
    "returns": [
        "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
        "ret_1d_norm", "ret_5d_norm",
        "ret_mean_5d", "ret_mean_10d", "ret_mean_20d",
        "ret_std_5d",  "ret_std_10d",  "ret_std_20d",
        "ret_skew_5d",
        "ret_positive_5d", "ret_positive_10d",
        "up_down_ratio_10d", "max_drawdown_10d",
        "ret_acceleration",
    ],
}