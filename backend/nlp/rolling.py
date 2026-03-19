"""
StockSense AI — nlp/rolling.py
================================
Rolling and derived sentiment features.

This file owns:
  - Rolling statistics on daily sentiment scores
  - Sentiment trend, momentum, and regime features
  - Sentiment RSI (overbought/oversold sentiment)
  - Article attention trend and spike detection
  - Cross-signal features (sentiment × price momentum)

It does NOT own:
  - Raw sentiment scoring          → nlp/finbert.py
  - Daily aggregation              → nlp/sentiment.py
  - Merging sentiment with prices  → data/merger.py
  - Feature engineering master     → features/engineer.py

Called by:
  features/engineer.py → build_features()
  Must be called AFTER merger.py has joined sentiment columns onto the
  price DataFrame, and AFTER add_momentum_features() so that ret_1d,
  ret_5d, and rsi_14 are already present for cross-signal features.

Why rolling sentiment features?
─────────────────────────────────────────────────────────────
  Daily sentiment is noisy — one off-topic article can flip the
  score. Rolling means smooth this noise and reveal the underlying
  trend. More importantly, XGBoost processes one row at a time and
  has no inherent memory. Rolling features encode "what has sentiment
  been doing recently" explicitly in each row, giving the model the
  same temporal context it would have in a recurrent network.

Why sentiment RSI?
─────────────────────────────────────────────────────────────
  If sentiment has been strongly positive for 10 consecutive days,
  it is likely already priced in — the market has had time to react.
  Sentiment RSI detects this "overbought" condition and lets the model
  discount stale signals. Conversely, sentiment just turning positive
  after a sustained negative period is a fresh, not-yet-priced signal.
  The RSI formula is identical to indicators.py add_momentum_features()
  for consistency — same math, different input series.

Why cross-signal features (sentiment × price momentum)?
─────────────────────────────────────────────────────────────
  Sentiment and price momentum can confirm or diverge:
    Both positive  → strong conviction, trend likely to continue
    Sentiment positive, price negative → potential reversal setup
    Both negative  → confirmed downtrend
    Sentiment negative, price positive → potential reversal warning
  XGBoost would need 3 splits to learn this. One interaction column
  encodes it in a single feature — same reasoning as add_interaction_features()
  in indicators.py.

Why streak features?
─────────────────────────────────────────────────────────────
  A single positive day may be noise. Five consecutive positive days
  means sustained bullish news coverage — a qualitatively stronger signal.
  The streak encodes this duration explicitly as a numeric value XGBoost
  can threshold on (+3 = three positive days, −2 = two negative days).

Required input columns (from nlp/sentiment.py → NEUTRAL_DAILY_FEATURES):
  sentiment_mean, sentiment_max, sentiment_min,
  sentiment_std, article_count, positive_ratio,
  confidence_mean, sentiment_momentum

Optional input columns (from features/indicators.py):
  ret_1d, ret_5d, rsi_14
  If absent, cross-signal features are set to 0.0 and a warning is issued.
"""

import warnings

import numpy as np
import pandas as pd
from typing import Dict, List

from nlp.sentiment import SENTIMENT_FEATURE_COLS

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Rolling window sizes used throughout this module.
# Mirrors the pattern of explicit constants in indicators.py.
# 3d = workweek,  7d = one trading week,  14d = two trading weeks.
ROLLING_WINDOWS: List[int] = [3, 7, 14]

# Sentiment regime thresholds.
# |score| < NEUTRAL_BAND → neutral regime (0).
# Calibrated to Financial PhraseBank score distribution:
# the distribution has a large neutral mass between ±0.10.
NEUTRAL_BAND: float = 0.10

# RSI period for sentiment overbought/oversold detection.
# Matches rsi_14 period in indicators.py for cross-signal alignment.
SENTIMENT_RSI_PERIOD: int = 14

# Attention spike multiplier: today's article count > mean * this → spike.
ATTENTION_SPIKE_MULTIPLIER: float = 2.0

# Column registry — mirrors FEATURE_GROUPS in indicators.py.
# Used by: engineer.py get_feature_columns(), SHAP ablation studies.
ROLLING_SENTIMENT_FEATURE_GROUPS: Dict[str, List[str]] = {
    "rolling_means": [
        "sentiment_ma3",
        "sentiment_ma7",
        "sentiment_ma14",
    ],
    "rolling_volatility": [
        "sentiment_vol_3d",
        "sentiment_vol_7d",
    ],
    "trend": [
        "sentiment_trend_3_7",    # ma3 − ma7  (MACD-equivalent)
        "sentiment_trend_7_14",   # ma7 − ma14
        "sentiment_regime",       # +1 / 0 / −1
        "sentiment_improving",    # binary: ma3 > ma7
    ],
    "momentum": [
        "sentiment_momentum_3d",  # 3d rolling mean of daily momentum
        "sentiment_accel",        # momentum of momentum (second derivative)
        "sentiment_rsi",          # RSI applied to sentiment_mean series
        "sentiment_streak",       # consecutive positive / negative days
    ],
    "attention": [
        "article_count_ma7",      # rolling avg article count
        "article_count_spike",    # binary: today > 2× rolling avg
        "attention_trend",        # binary: rolling avg rising vs 3d ago
    ],
    "cross_signal": [
        "sent_price_confirm",     # sentiment_regime × sign(ret_1d)
        "sent_rsi_diverge",       # sentiment positive but RSI overbought
        "sent_momentum_align",    # sentiment trend aligned with ret_5d
    ],
}

# Flat list of ALL rolling sentiment columns — used in engineer.py
# get_feature_columns() and SHAP drop lists.
ALL_ROLLING_SENTIMENT_COLS: List[str] = [
    col
    for group in ROLLING_SENTIMENT_FEATURE_GROUPS.values()
    for col in group
]


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling standard deviation with min_periods=2.

    Why min_periods=2?
    ─────────────────────────────────────────────────────────────
    Standard rolling().std() returns NaN for windows with < 2 values.
    min_periods=2 allows std calculation as soon as 2 values exist —
    it reduces NaN rows at the start of the series without introducing
    statistical bias. 2 is the minimum mathematically valid sample for
    standard deviation (Bessel correction requires n ≥ 2).
    """
    return series.rolling(window, min_periods=2).std().fillna(0.0)


def _sentiment_rsi(series: pd.Series, period: int = SENTIMENT_RSI_PERIOD) -> pd.Series:
    """
    Apply RSI formula to a sentiment_mean series.

    Implementation matches add_momentum_features() in indicators.py:
    same gain/loss rolling mean ratio, same min_periods = period // 2,
    same NaN fill of 50.0 (neutral midpoint).

    Why the same formula as price RSI?
    ─────────────────────────────────────────────────────────────
    RSI measures momentum exhaustion regardless of what is being
    measured. Price RSI detects "price moved too far too fast".
    Sentiment RSI detects "sentiment has been extreme for too long".
    Both use the same avg_gain / avg_loss ratio normalised to 0–100.
    Values > 70 → sentiment overbought (likely priced in).
    Values < 30 → sentiment oversold  (potential underreaction).
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=period // 2).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period // 2).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)   # neutral RSI = 50


def _streak(series: pd.Series) -> pd.Series:
    """
    Calculate consecutive positive / negative day streak.

    Positive streak : +1, +2, +3 … (consecutive days above 0)
    Negative streak : −1, −2, −3 … (consecutive days below 0)
    Resets to ±1 on direction change; holds at 0 on neutral days.

    Why integer streaks rather than a binary flag?
    ─────────────────────────────────────────────────────────────
    A binary flag loses duration — it cannot distinguish +1 day vs
    +5 days of positive sentiment. An integer streak encodes duration
    directly as a numeric value that XGBoost can threshold on.
    +5 is intuitively "stronger" than +1 and the tree will learn this.
    """
    direction = np.sign(series.fillna(0.0))
    streak    = pd.Series(0.0, index=series.index, dtype=float)
    current   = 0.0

    for i, d in enumerate(direction):
        if d == 0.0:
            current = 0.0
        elif i == 0:
            current = float(d)
        elif np.sign(current) == float(d):
            current += float(d)
        else:
            current = float(d)
        streak.iloc[i] = current

    return streak


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE BUILDERS  (private sub-functions)
# ══════════════════════════════════════════════════════════════════════════════

def _add_rolling_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling means of sentiment_mean at 3, 7, and 14 day windows.

    Why three windows?
    ─────────────────────────────────────────────────────────────
    3d  = workweek — immediate noise smoothing
    7d  = one trading week — weekly narrative
    14d = two trading weeks — bi-weekly sentiment trend
    These correspond to natural news cycles: daily noise, weekly
    earnings/macro themes, and bi-weekly narrative arcs.

    min_periods=1 so that the first row is not NaN — the rolling
    mean of a single value is just that value, which is a valid
    estimate. This is consistent with indicators.py behaviour.
    """
    s = df["sentiment_mean"]
    for w in ROLLING_WINDOWS:
        df[f"sentiment_ma{w}"] = s.rolling(w, min_periods=1).mean()
    return df


def _add_rolling_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling standard deviation of sentiment_mean at 3 and 7 days.

    Why not 14d volatility?
    ─────────────────────────────────────────────────────────────
    At 14 days, std is already available from sentiment_std rolled
    forward — adding a 14d window would be collinear. The 3d and 7d
    windows capture short-horizon disagreement without redundancy.

    High sentiment_vol_3d = articles strongly disagree over three days.
    This flags genuine short-term uncertainty, not just today's noise
    (already captured by sentiment_std from sentiment.py).
    """
    s = df["sentiment_mean"]
    df["sentiment_vol_3d"] = _safe_rolling_std(s, 3)
    df["sentiment_vol_7d"] = _safe_rolling_std(s, 7)
    return df


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sentiment trend direction features.

    sentiment_trend_3_7 mirrors MACD logic from add_macd_features():
    short_MA − long_MA = momentum direction.
    Positive value = short-term sentiment more positive than medium-term
                   = sentiment trend is improving (bullish).
    Negative value = sentiment trend is worsening (bearish).

    sentiment_regime maps the 7-day MA to a ternary signal:
    +1 (positive), 0 (neutral), −1 (negative).
    This is used directly in cross_signal features.
    """
    ma3  = df["sentiment_ma3"]
    ma7  = df["sentiment_ma7"]
    ma14 = df["sentiment_ma14"]

    # MACD-equivalent: short minus long MA = direction of momentum
    df["sentiment_trend_3_7"]  = (ma3 - ma7).round(4)
    df["sentiment_trend_7_14"] = (ma7 - ma14).round(4)

    # Regime: which zone is the 7-day rolling mean in?
    # 7d used (not raw daily) to reduce false regime flips from noisy days.
    df["sentiment_regime"] = np.where(
        ma7 >  NEUTRAL_BAND,  1.0,
        np.where(
            ma7 < -NEUTRAL_BAND, -1.0,
            0.0,
        ),
    )

    # Binary improving flag: short-term > medium-term (1 = improving)
    df["sentiment_improving"] = (ma3 > ma7).astype(int)

    return df


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sentiment momentum and RSI features.

    sentiment_momentum_3d: 3-day rolling mean of raw sentiment_momentum.
    The raw daily momentum (in sentiment.py) is noisy. A 3-day smooth
    mirrors ret_mean_3d in add_return_features() — same noise reduction
    logic applied to sentiment rather than price.

    sentiment_accel: diff(1) of raw momentum = second derivative of
    sentiment_mean. Mirrors ret_acceleration in add_return_features().
    Is momentum itself speeding up or slowing down?
    """
    if "sentiment_momentum" in df.columns:
        df["sentiment_momentum_3d"] = (
            df["sentiment_momentum"]
            .rolling(3, min_periods=1)
            .mean()
            .round(4)
        )
        df["sentiment_accel"] = df["sentiment_momentum"].diff(1).fillna(0.0).round(4)
    else:
        df["sentiment_momentum_3d"] = 0.0
        df["sentiment_accel"]       = 0.0

    df["sentiment_rsi"]    = _sentiment_rsi(df["sentiment_mean"])
    df["sentiment_streak"] = _streak(df["sentiment_mean"])

    return df


def _add_attention_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Article count / market attention features.

    Why separate from sentiment direction?
    ─────────────────────────────────────────────────────────────
    High article count carries information independent of direction:
      High count + positive → strong bullish signal (market is talking)
      High count + negative → strong bearish signal (market is alarmed)
      High count + neutral  → uncertainty / information overload
    article_count_spike flags an event day — earnings, announcement,
    sector news — where price is more likely to move significantly
    regardless of sentiment direction.

    article_count_spike multiplier of 2.0 mirrors volume_spike logic
    in add_volume_features(): today > 2× rolling mean = spike.
    """
    if "article_count" not in df.columns:
        df["article_count_ma7"]   = 0.0
        df["article_count_spike"] = 0
        df["attention_trend"]     = 0
        return df

    count  = df["article_count"]
    ma7    = count.rolling(7, min_periods=1).mean()
    ma7_3d = ma7.shift(3)   # 3 days ago for trend comparison

    df["article_count_ma7"]   = ma7.round(2)
    df["article_count_spike"] = (
        count > ma7 * ATTENTION_SPIKE_MULTIPLIER
    ).astype(int)
    df["attention_trend"] = (ma7 > ma7_3d).fillna(False).astype(int)

    return df


def _add_cross_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-signal features: sentiment × price / technical indicator.

    Why cross-signals?
    ─────────────────────────────────────────────────────────────
    Sentiment and price momentum can confirm or diverge.
    XGBoost would need multiple splits to learn: "if sentiment is
    positive AND price is also positive THEN strong up signal".
    Encoding this as a single column makes it SHAP-attributable
    and reduces required tree depth — same rationale as
    add_interaction_features() in indicators.py.

    Graceful degradation:
    ─────────────────────────────────────────────────────────────
    These features require ret_1d, ret_5d, and rsi_14 from
    features/indicators.py. If those columns are absent (e.g. in
    unit tests with sentiment-only DataFrames), the cross-signal
    features are set to 0.0 and a warning is issued rather than
    raising an exception — mirrors the behaviour of add_volume_features()
    when volume is absent.
    """
    regime = df.get("sentiment_regime", pd.Series(0.0, index=df.index))
    missing_price_cols = [c for c in ["ret_1d", "ret_5d", "rsi_14"]
                          if c not in df.columns]
    if missing_price_cols:
        warnings.warn(
            f"[rolling.py] Cross-signal features require price columns "
            f"from indicators.py: {missing_price_cols}. "
            f"Set to 0.0. Call add_momentum_features() and "
            f"add_return_features() before add_rolling_sentiment_features().",
            UserWarning,
            stacklevel=3,
        )

    # ── sent_price_confirm ────────────────────────────────────────────────────
    # sentiment_regime × sign(ret_1d):
    #   +1 = both sentiment and price direction agree (positive)
    #   -1 = both agree (negative)
    #    0 = divergence (sentiment and price point in opposite directions)
    if "ret_1d" in df.columns:
        price_dir = np.sign(df["ret_1d"].fillna(0.0))
        df["sent_price_confirm"] = (regime * price_dir).clip(-1.0, 1.0)
    else:
        df["sent_price_confirm"] = 0.0

    # ── sent_rsi_diverge ──────────────────────────────────────────────────────
    # Flags the case where sentiment is positive (regime > 0) BUT the stock
    # is technically overbought (rsi_14 > 70). This means the positive
    # sentiment is likely already fully reflected in price — discount signal.
    if "rsi_14" in df.columns:
        df["sent_rsi_diverge"] = (
            (regime > 0) & (df["rsi_14"].fillna(50.0) > 70.0)
        ).astype(int)
    else:
        df["sent_rsi_diverge"] = 0

    # ── sent_momentum_align ───────────────────────────────────────────────────
    # Sentiment trend (3_7 MA cross) and 5d price momentum agree in direction.
    # When both point the same way, the signal has dual confirmation.
    # Mirrors soldiers_trend_align in add_interaction_features().
    if "ret_5d" in df.columns:
        sent_trend     = df.get("sentiment_trend_3_7",
                                pd.Series(0.0, index=df.index))
        price_momentum = np.sign(df["ret_5d"].fillna(0.0))
        df["sent_momentum_align"] = (
            np.sign(sent_trend.fillna(0.0)) == price_momentum
        ).astype(int)
    else:
        df["sent_momentum_align"] = 0

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def add_rolling_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all rolling and derived sentiment features to a DataFrame.

    Called by features/engineer.py → build_features() after:
      1. merger.py has joined sentiment columns onto the price DataFrame
      2. add_momentum_features() has added ret_1d, ret_5d, rsi_14

    If sentiment columns are absent, returns df unchanged with a warning
    rather than raising — consistent with how add_volume_features() handles
    a missing volume column.

    Parameters
    ----------
    df : DataFrame with sentiment_mean (and ideally all SENTIMENT_FEATURE_COLS)
         already present. Typically the output of data/merger.py after the
         Phase 3 sentiment pipeline upgrade.

    Returns
    -------
    Same DataFrame with all ALL_ROLLING_SENTIMENT_COLS columns appended.
    Does NOT call df.copy() — caller is responsible for isolation if needed.
    Rows at the start of the series will have partial NaN from rolling windows;
    caller handles dropna() as usual after build_features() completes.

    Design: orchestration only, no math
    ─────────────────────────────────────────────────────────────
    This function calls private sub-functions in order. All math lives
    in the sub-functions. This mirrors the structure of build_features()
    in engineer.py — the public function is the conductor, the private
    functions are the instruments.
    """
    if "sentiment_mean" not in df.columns:
        warnings.warn(
            "[rolling.py] sentiment_mean not found in DataFrame. "
            "Rolling sentiment features skipped. "
            "Ensure merger.py has joined nlp/sentiment.py output before "
            "calling add_rolling_sentiment_features().",
            UserWarning,
            stacklevel=2,
        )
        return df

    # Step order matters:
    # 1. Rolling means must be computed before trend (trend reads ma3/ma7/ma14)
    # 2. Trend must be computed before cross-signal (cross reads sentiment_regime,
    #    sentiment_trend_3_7)
    # 3. Momentum must be computed before cross-signal (cross reads sentiment_rsi)
    df = _add_rolling_means(df)         # 1. ma3, ma7, ma14
    df = _add_rolling_volatility(df)    # 2. vol_3d, vol_7d
    df = _add_trend_features(df)        # 3. trend_3_7, regime, improving
    df = _add_momentum_features(df)     # 4. momentum_3d, accel, rsi, streak
    df = _add_attention_features(df)    # 5. count_ma7, spike, trend
    df = _add_cross_signal_features(df) # 6. confirm, diverge, align

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def list_rolling_sentiment_features(verbose: bool = True) -> List[str]:
    """
    List all rolling sentiment feature columns produced by this module.

    Mirrors list_sentiment_features() in nlp/sentiment.py and
    list_model_configs() in nlp/finbert.py — consistent discovery pattern
    across all StockSense AI modules.

    Parameters
    ----------
    verbose : If True, print a formatted table grouped by category.

    Returns
    -------
    List[str] of all column names this module adds.
    """
    if verbose:
        total = len(ALL_ROLLING_SENTIMENT_COLS)
        print(f"\nRolling Sentiment Features  ({total} total)")
        print("─" * 55)
        for group, cols in ROLLING_SENTIMENT_FEATURE_GROUPS.items():
            print(f"\n  {group}  ({len(cols)} features)")
            for col in cols:
                print(f"    {col}")
        print()
    return ALL_ROLLING_SENTIMENT_COLS


def get_rolling_feature_group(group: str) -> List[str]:
    """
    Return column names for a specific feature group.

    Parameters
    ----------
    group : Key from ROLLING_SENTIMENT_FEATURE_GROUPS.
            Options: rolling_means, rolling_volatility, trend,
                     momentum, attention, cross_signal.

    Returns
    -------
    List[str] of column names for that group, or empty list if not found.
    """
    return ROLLING_SENTIMENT_FEATURE_GROUPS.get(group, [])


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    import yfinance as yf
    from data.news import fetch_news_for_stock
    from nlp.sentiment import build_daily_sentiment

    # ── List features ────────────────────────────────────────────────────────
    list_rolling_sentiment_features()

    # ── Download price data ──────────────────────────────────────────────────
    print("Downloading AAPL (3mo)...")
    raw = yf.download("AAPL", period="3mo", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]

    # ── Fetch and score news ──────────────────────────────────────────────────
    print("Fetching AAPL news...")
    articles = fetch_news_for_stock("AAPL", days_back=60, max_articles=100)
    print(f"Fetched {len(articles)} articles")

    daily_sentiment = build_daily_sentiment(articles, raw, verbose=True)

    # ── Merge: left-join sentiment onto price (no lookahead shift here — test only)
    merged = raw.join(daily_sentiment, how="left")
    for col in daily_sentiment.columns:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0.0)

    # ── Simulate indicators.py columns being present ──────────────────────────
    # In production, engineer.py calls add_momentum_features() before this.
    # For the standalone test, create stub columns so cross-signal features work.
    merged["ret_1d"] = merged["close"].pct_change(1)
    merged["ret_5d"] = merged["close"].pct_change(5)
    merged["rsi_14"] = merged["close"].rolling(14).apply(
        lambda x: 100 - 100 / (1 + (
            pd.Series(x).diff().clip(lower=0).mean() /
            max((-pd.Series(x).diff().clip(upper=0)).mean(), 1e-9)
        )),
        raw=False,
    )

    # ── Add rolling sentiment features ────────────────────────────────────────
    print("\nAdding rolling sentiment features...")
    featured = add_rolling_sentiment_features(merged)

    # ── Display results ───────────────────────────────────────────────────────
    display_cols = [
        "sentiment_mean",
        "sentiment_ma3",
        "sentiment_ma7",
        "sentiment_trend_3_7",
        "sentiment_regime",
        "sentiment_rsi",
        "sentiment_streak",
        "article_count_spike",
        "sent_price_confirm",
    ]
    available = [c for c in display_cols if c in featured.columns]

    print(f"\n=== Rolling Sentiment Features — last 10 trading days ===")
    print(featured[available].tail(10).round(3).to_string())

    print(f"\nTotal rolling features added : {len(ALL_ROLLING_SENTIMENT_COLS)}")
    print(f"Total columns in output      : {len(featured.columns)}")
    print(f"NaN rows (expected from warm-up): "
          f"{featured[ALL_ROLLING_SENTIMENT_COLS].isnull().any(axis=1).sum()}")
