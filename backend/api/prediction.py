"""
StockSense AI — api/prediction.py
===================================
Prediction pipeline: ticker → full ML response.

This file owns:
  - The complete prediction pipeline (data → features → model → SHAP)
  - Response construction matching the frontend contract
  - Redis caching layer for prediction results
  - Graceful error handling with structured error responses
  - Multi-stock screener pipeline

It does NOT own:
  - FastAPI app creation    → api/main.py
  - Route definitions       → api/routes/predict.py
  - Model training          → models/trainer.py
  - Feature engineering     → features/engineer.py

WHY THIS FILE EXISTS — Separation of Concerns
─────────────────────────────────────────────────────────────
  The prediction pipeline has 15 sequential steps. Putting all
  15 steps in the route handler makes routes/predict.py unmaintainable.
  Isolating the pipeline here means:
    1. Routes stay clean (3-4 lines each)
    2. Pipeline can be called from scheduled jobs too (not just HTTP)
    3. Easy to test the pipeline without HTTP overhead
    4. Clear single responsibility

WHY async + ThreadPoolExecutor for blocking calls?
─────────────────────────────────────────────────────────────
  FastAPI is built on asyncio. Calling synchronous blocking code
  (yfinance, pandas, XGBoost) directly in async functions blocks
  the event loop — the entire server freezes during computation.

  Example of the WRONG way:
      @app.get("/predict")
      async def predict():
          df = yf.download("AAPL", ...)  # ← BLOCKS the event loop for 2 seconds
          ...                              #    No other request can be served

  Example of the RIGHT way (what we do):
      @app.get("/predict")
      async def predict():
          df = await loop.run_in_executor(executor, yf.download, ...)
          ...  # ← event loop is free to serve other requests while waiting

  ThreadPoolExecutor runs blocking code in a separate OS thread,
  freeing the event loop to handle other requests simultaneously.
  For a 500-stock screener with concurrent requests, this is essential.

  asyncio event loop architecture:
  ┌─────────────────────────────────┐
  │  Event Loop (single thread)     │
  │  ├─ Request A → waiting on I/O  │
  │  ├─ Request B → computing       │
  │  └─ Request C → waiting on I/O  │
  └─────────┬───────────────────────┘
            │ run_in_executor()
  ┌─────────▼───────────────────────┐
  │  ThreadPool (4 worker threads)  │
  │  ├─ Thread 1: yfinance fetch    │
  │  ├─ Thread 2: XGBoost predict   │
  │  ├─ Thread 3: SHAP explain      │
  │  └─ Thread 4: idle              │
  └─────────────────────────────────┘
"""

import asyncio
import json
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Thread pool for blocking I/O (yfinance, pandas, XGBoost) ─────────────────
# max_workers=4 means at most 4 blocking operations can run simultaneously.
# Why 4? Each worker holds one yfinance connection. More than 4 risks
# yfinance rate limiting. Fewer than 4 means the screener is too slow.
_executor = ThreadPoolExecutor(max_workers=4)


# ══════════════════════════════════════════════════════════════════════════════
#  TICKER VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def normalise_ticker(ticker: str) -> str:
    """
    Normalise ticker to yfinance format.
    Strips whitespace, uppercases, preserves exchange suffixes (.NS, .BO etc.)

    Examples:
      "  aapl "     → "AAPL"
      "reliance.ns" → "RELIANCE.NS"
      " msft"       → "MSFT"
    """
    return ticker.strip().upper()


async def validate_ticker(ticker: str) -> bool:
    """
    Validate ticker by attempting a minimal yfinance fetch.
    Returns True if valid, False if not found.

    Why a real fetch rather than a static list?
    ─────────────────────────────────────────────────────────────
    A static list of 10,000 tickers goes stale — new stocks list,
    others delist. A live check is always accurate and only takes
    one fast metadata request (not the full history download).
    """
    loop = asyncio.get_event_loop()

    def _check():
        try:
            info = yf.Ticker(ticker).fast_info
            return hasattr(info, "last_price") and info.last_price is not None
        except Exception:
            return False

    return await loop.run_in_executor(_executor, _check)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING (async wrappers around blocking calls)
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_live_ohlcv(
    ticker:        str,
    period:        str = "2y",
) -> pd.DataFrame:
    """
    Fetch OHLCV data asynchronously.
    Runs yfinance in a thread pool to avoid blocking the event loop.

    Why period="2y"?
    ─────────────────────────────────────────────────────────────
    SMA_200 in add_trend_features() needs 200 trading days to warm up.
    2 years gives ~500 trading days — enough for a clean feature set
    after dropping NaN rows from rolling windows.

    Parameters
    ----------
    ticker : Stock symbol (e.g. "AAPL").
    period : yfinance period string (e.g. "2y", "1y", "6mo").

    Returns
    -------
    pd.DataFrame : Cleaned OHLCV data with lowercase columns.

    Raises
    ------
    ValueError : If no data returned or insufficient rows.
    """
    loop = asyncio.get_event_loop()

    def _fetch():
        from data.cleaner import clean_stock_data

        raw = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(
                f"No price data found for '{ticker}'. "
                f"Check the ticker symbol is correct."
            )

        # yfinance returns MultiIndex columns when auto_adjust=True
        # Flatten them: ('Close', 'AAPL') → 'close'
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]

        clean = clean_stock_data(raw, ticker=ticker)

        if len(clean) < 250:
            raise ValueError(
                f"Insufficient data for '{ticker}': only {len(clean)} rows. "
                f"Need at least 250 trading days for reliable features."
            )

        return clean

    return await loop.run_in_executor(_executor, _fetch)


async def fetch_live_sentiment(
    ticker:    str,
    price_df:  pd.DataFrame,
    days_back: int = 14,
) -> pd.DataFrame:
    """
    Fetch and score recent news headlines asynchronously.
    Returns daily sentiment DataFrame aligned to price_df.

    Falls back to neutral if:
      - NewsAPI key is missing
      - No articles found for this ticker
      - FinBERT model is unavailable
      - Any unexpected error occurs

    Why fall back instead of fail?
    ─────────────────────────────────────────────────────────────
    Sentiment is a SUPPLEMENTARY signal. The ML model was trained
    with sentiment features, but if they're all-zero (neutral),
    the model still makes a valid prediction based on the 300+
    technical features. Failing the entire prediction because
    the NewsAPI key expired would be terrible UX.

    Parameters
    ----------
    ticker    : Stock symbol.
    price_df  : Price DataFrame (used to align sentiment dates).
    days_back : How many days of news to fetch.

    Returns
    -------
    pd.DataFrame : Daily sentiment features aligned to price_df.index.
    """
    loop = asyncio.get_event_loop()

    def _fetch_sentiment():
        try:
            from data.news import fetch_news_for_stock
            from nlp.sentiment import build_daily_sentiment

            articles = fetch_news_for_stock(ticker, days_back=days_back)

            if not articles:
                logger.info(f"No news articles for {ticker} — using neutral sentiment")
                return _neutral_sentiment(price_df.index)

            return build_daily_sentiment(
                articles=articles,
                price_df=price_df,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"Sentiment fetch failed for {ticker}: {e}")
            return _neutral_sentiment(price_df.index)

    return await loop.run_in_executor(_executor, _fetch_sentiment)


def _neutral_sentiment(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Return neutral sentiment DataFrame when news is unavailable.

    Uses the exact same neutral values defined in nlp/sentiment.py
    to ensure consistency — if we hardcode different defaults here,
    the model sees different "neutral" signals during inference
    than during training, which degrades accuracy.
    """
    from nlp.sentiment import NEUTRAL_DAILY_FEATURES
    return pd.DataFrame(NEUTRAL_DAILY_FEATURES, index=index)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (sync — called inside executor)
# ══════════════════════════════════════════════════════════════════════════════

def _build_live_features(
    price_df:     pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker:       str,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Build feature matrix from live price + sentiment data.
    Called inside ThreadPoolExecutor — no async here.

    Steps:
    ─────────────────────────────────────────────────────────────
    1. Merge price + sentiment (with 1-day shift to prevent leakage)
    2. Engineer features (14-step pipeline from features/engineer.py)
    3. Drop raw OHLCV columns (get_model_features)
    4. Align columns to match training order exactly
    5. Return single-row DataFrame (latest day only)

    Why align columns to training order?
    ─────────────────────────────────────────────────────────────
    XGBoost internally accesses features by position index, not name.
    If columns are in a different order than training, the model
    silently maps RSI values to the MACD slot, etc. — producing
    garbage predictions without any error message.

    Parameters
    ----------
    price_df     : Cleaned OHLCV DataFrame.
    sentiment_df : Daily sentiment features aligned to price dates.
    ticker       : Ticker symbol (for error messages).
    feature_cols : Ordered list of feature column names from training.

    Returns
    -------
    pd.DataFrame : Single-row DataFrame with features in training order.
    """
    from data.merger import merge_price_and_sentiment
    from features.engineer import build_features
    from features.indicators import get_model_features

    # Merge sentiment with 1-day shift to prevent look-ahead leakage
    # Day t's prediction uses day t-1's sentiment (news published yesterday)
    merged = merge_price_and_sentiment(
        price_df=price_df,
        sentiment_df=sentiment_df,
        shift_days=1,
    )

    # Run the 14-step feature engineering pipeline
    featured = build_features(merged).dropna()

    if len(featured) == 0:
        raise ValueError(
            f"Feature engineering produced empty DataFrame for {ticker}. "
            f"Not enough data to compute rolling indicators."
        )

    # Drop raw OHLCV and non-stationary columns
    X = get_model_features(featured, extra_drop=["target"]).fillna(0)

    # Align columns with trained feature names
    # Add missing columns as 0, drop extra columns, reorder
    missing = [c for c in feature_cols if c not in X.columns]
    for col in missing:
        X[col] = 0.0

    # Keep only training columns in exact training order
    X = X[feature_cols]

    # Return only the last row (today's features)
    return X.iloc[[-1]]


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP EXPLANATION (sync — called inside executor)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_explanation(
    pipeline:  object,
    X_single:  pd.DataFrame,
    proba:     float,
    top_n:     int = 5,
) -> Dict:
    """
    Generate SHAP-based explanation for one prediction.

    Returns a dict with:
      - explanation_text:  plain English paragraph
      - top_features:      list of {feature, shap_value, direction, ...}
      - feature_groups:    dict of group_name → total SHAP importance

    Falls back to a minimal explanation if SHAP computation fails.

    Why fall back instead of fail?
    ─────────────────────────────────────────────────────────────
    SHAP TreeExplainer can fail with certain XGBoost configurations
    or when memory is tight. The prediction itself is still valid —
    we should return it with a "sorry, no explanation" message
    rather than crashing the entire endpoint.
    """
    try:
        from models.explainer import explain_single_prediction

        explanation = explain_single_prediction(
            pipeline=pipeline,
            X_single=X_single,
            top_n=top_n,
            verbose=False,
        )
        return explanation

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        direction = "UP" if proba >= 0.5 else "DOWN"
        return {
            "prediction":       direction,
            "probability":      proba,
            "confidence_pct":   proba * 100,
            "top_features":     [],
            "feature_groups":   {},
            "explanation_text": (
                f"Our model predicts this stock will go {direction.lower()} "
                f"with {proba*100:.0f}% confidence. "
                f"Detailed explanation unavailable."
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MARKET DATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_market_data(
    price_df:  pd.DataFrame,
    X_latest:  pd.DataFrame,
) -> Dict:
    """
    Extract key market data metrics for the frontend's market snapshot.
    These are the "at-a-glance" numbers shown on the stock page header.

    What the frontend shows with this data:
    ─────────────────────────────────────────────────────────────
    ┌─────────────────────────────────────────────────┐
    │  AAPL  $198.32  ▲ +1.23%                        │
    │  RSI: 62.4 (neutral)  |  MACD: bullish          │
    │  Bollinger: 72% (upper)  |  Sentiment: +0.34    │
    │  Volume: 1.2× avg  |  Trend: aligned            │
    └─────────────────────────────────────────────────┘

    Parameters
    ----------
    price_df : Full OHLCV DataFrame (at least 2 rows).
    X_latest : Single-row feature DataFrame for today.

    Returns
    -------
    Dict with market snapshot values.
    """
    latest = price_df.iloc[-1]
    prev   = price_df.iloc[-2] if len(price_df) > 1 else latest

    price_change_pct = (
        (latest["close"] - prev["close"]) / prev["close"]
    ) * 100

    # Helper to safely extract a feature value
    def _get(col: str, default=None):
        if col in X_latest.columns:
            val = X_latest[col].iloc[-1]
            return None if pd.isna(val) else round(float(val), 4)
        return default

    return {
        "current_price":    round(float(latest["close"]), 2),
        "price_change_pct": round(float(price_change_pct), 2),
        "volume":           int(latest.get("volume", 0)),
        "volume_ratio":     _get("volume_ratio", 1.0),
        "rsi":              _get("rsi_14"),
        "macd_signal":      "bullish" if _get("macd_above_signal", 0) else "bearish",
        "bb_position":      _get("bb_percent"),
        "atr_pct":          _get("atr_pct"),
        "sentiment_score":  _get("sentiment_mean", 0.0),
        "sentiment_trend":  _get("sentiment_trend_3_7", 0.0),
        "trend_agreement":  _get("trend_agreement", 0.0),
        "pattern_signal":   _get("pattern_signal", 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  REDIS CACHING
#  See docs/redis_guide.py for full explanation of Redis concepts.
# ══════════════════════════════════════════════════════════════════════════════

def _get_cache_key(ticker: str, horizon: int, threshold: float) -> str:
    """
    Generate a deterministic Redis cache key.

    Format: "prediction:AAPL:1:0.500"

    Why include horizon + threshold in the key?
    ─────────────────────────────────────────────────────────────
    The same ticker with different thresholds produces different
    UP/DOWN classifications. Caching without these params would
    return stale results when the user changes settings.
    """
    return f"prediction:{ticker}:{horizon}:{threshold:.3f}"


async def _get_cached_prediction(
    redis_client,
    cache_key:    str,
) -> Optional[Dict]:
    """
    Attempt to load a cached prediction from Redis.
    Returns None if cache miss, key expired, or Redis unavailable.

    Why silent failure?
    ─────────────────────────────────────────────────────────────
    Redis is a PERFORMANCE OPTIMISATION, not a requirement.
    If Redis is down, the prediction still works — it's just slower
    (runs the full ML pipeline instead of returning cached result).
    Crashing because Redis is unreachable would be wrong.
    """
    if redis_client is None:
        return None
    try:
        cached = await redis_client.get(cache_key)
        if cached:
            logger.info(f"Redis HIT: {cache_key}")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis GET failed (non-fatal): {e}")
    return None


async def _cache_prediction(
    redis_client,
    cache_key:    str,
    result:       Dict,
    ttl_seconds:  int = 3600,
) -> None:
    """
    Store prediction in Redis with a TTL (time-to-live).

    After ttl_seconds, Redis automatically deletes the key.
    This ensures predictions don't go stale — after 1 hour,
    the next request will recompute with fresh market data.

    Why 3600 seconds (1 hour)?
    ─────────────────────────────────────────────────────────────
    Stock prices update every few seconds, but our ML features
    are computed from daily OHLCV bars. Within the same trading day,
    the features are identical. 1 hour strikes a balance between
    freshness and avoiding unnecessary recomputation.
    During market hours, you could reduce this to 900 (15 min).
    """
    if redis_client is None:
        return
    try:
        await redis_client.setex(
            cache_key,
            ttl_seconds,
            json.dumps(result, default=str),
        )
        logger.debug(f"Redis SET: {cache_key} (TTL={ttl_seconds}s)")
    except Exception as e:
        logger.warning(f"Redis SET failed (non-fatal): {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

async def generate_prediction(
    ticker:        str,
    pipeline:      object,
    feature_cols:  List[str],
    period:        str   = "2y",
    horizon:       int   = 1,
    threshold:     float = 0.5,
    top_n:         int   = 5,
    redis_client:  object = None,
    include_shap:  bool  = True,
) -> Dict:
    """
    Full prediction pipeline: ticker → complete API response.

    This is THE function called by the FastAPI route handler.
    Orchestrates all steps: validate → cache check → fetch →
    features → predict → explain → cache → return.

    Pipeline flow:
    ─────────────────────────────────────────────────────────────
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ 1. Cache │──▶│ 2. Fetch │──▶│3. Feature│──▶│4. Predict│
    │   check  │   │  OHLCV + │   │  engineer │   │  XGBoost │
    │          │   │sentiment │   │  pipeline │   │ predict_ │
    │ Redis GET│   │ yfinance │   │  315 cols │   │  proba() │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘
         │                                              │
         ▼                                              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  CACHE   │   │ 8. Cache │◀──│ 7. Build │◀──│ 5. SHAP  │
    │   HIT    │   │  result  │   │ response │   │  explain │
    │ (fast!)  │   │ Redis SET│   │   JSON   │   │ top_n=5  │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                       │
                                       ▼
                                  ┌──────────┐
                                  │ 6. Market│
                                  │   data   │
                                  │ snapshot │
                                  └──────────┘

    Parameters
    ----------
    ticker       : Stock symbol (already normalised to uppercase).
    pipeline     : Fitted sklearn Pipeline from app_state.
    feature_cols : Ordered list of feature column names from training.
    period       : yfinance period for data fetch (default "2y").
    horizon      : Prediction horizon in days (for cache key).
    threshold    : Classification threshold for UP/DOWN.
    top_n        : Number of top SHAP features to include.
    redis_client : Async Redis client (None = no caching).
    include_shap : Include SHAP explanation (slower but richer).

    Returns
    -------
    Dict matching the frontend API contract.

    Raises
    ------
    ValueError  : Invalid ticker or insufficient data.
    RuntimeError: Feature engineering or prediction failure.
    """
    start_time = time.time()

    # ── Step 1: Check Redis cache ─────────────────────────────────────────
    cache_key = _get_cache_key(ticker, horizon, threshold)
    cached    = await _get_cached_prediction(redis_client, cache_key)

    if cached:
        cached["_cache_hit"]  = True
        cached["_latency_ms"] = round((time.time() - start_time) * 1000, 1)
        return cached

    logger.info(f"Cache miss: {ticker} — running full pipeline")

    # ── Step 2: Fetch live price data (async — non-blocking) ──────────────
    price_df = await fetch_live_ohlcv(ticker, period=period)

    # ── Step 3: Fetch live sentiment (async — non-blocking) ───────────────
    sentiment_df = await fetch_live_sentiment(ticker, price_df, days_back=14)

    # ── Step 4: Feature engineering (blocking — runs in thread pool) ──────
    loop = asyncio.get_event_loop()

    try:
        X_latest = await loop.run_in_executor(
            _executor,
            lambda: _build_live_features(
                price_df, sentiment_df, ticker, feature_cols
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Feature engineering failed for {ticker}: {e}")

    # ── Step 5: Run the XGBoost model ─────────────────────────────────────
    try:
        proba_up   = float(pipeline.predict_proba(X_latest)[0][1])
        prediction = "UP" if proba_up >= threshold else "DOWN"
    except Exception as e:
        raise RuntimeError(f"Model prediction failed for {ticker}: {e}")

    # ── Step 6: SHAP explanation (blocking — runs in thread pool) ─────────
    if include_shap:
        explanation = await loop.run_in_executor(
            _executor,
            lambda: _generate_explanation(pipeline, X_latest, proba_up, top_n),
        )
    else:
        explanation = {
            "prediction":       prediction,
            "probability":      proba_up,
            "confidence_pct":   proba_up * 100,
            "top_features":     [],
            "feature_groups":   {},
            "explanation_text": (
                f"Prediction: {prediction} ({proba_up*100:.0f}% confidence)"
            ),
        }

    # ── Step 7: Market data extraction ────────────────────────────────────
    market_data = _extract_market_data(price_df, X_latest)

    # ── Step 8: Build complete response ───────────────────────────────────
    generated_at = datetime.now(timezone.utc).isoformat()
    latency_ms   = round((time.time() - start_time) * 1000, 1)

    # Get the prediction date (last row's date)
    last_date = price_df.index[-1]
    prediction_date = (
        str(last_date.date()) if hasattr(last_date, "date") else str(last_date)
    )

    response = {
        # ── Core prediction ───────────────────────────────────────────
        "ticker":           ticker,
        "prediction":       prediction,
        "probability":      round(proba_up, 4),
        "confidence_pct":   round(proba_up * 100, 1),
        "prediction_date":  prediction_date,
        "generated_at":     generated_at,
        "horizon_days":     horizon,
        "threshold_used":   threshold,

        # ── SHAP explanation ──────────────────────────────────────────
        "explanation": {
            "summary":        explanation.get("explanation_text", ""),
            "top_features":   explanation.get("top_features", []),
            "feature_groups": explanation.get("feature_groups", {}),
        },

        # ── Market snapshot (for the stock page header) ───────────────
        "market_data": market_data,

        # ── Data freshness metadata ───────────────────────────────────
        "data_freshness": {
            "price_data_as_of":     prediction_date,
            "features_computed_at": generated_at,
            "n_features_used":      len(feature_cols),
            "lookback_days":        len(price_df),
        },

        # ── Internal metadata (prefixed with _ = not shown to user) ──
        "_cache_hit":   False,
        "_latency_ms":  latency_ms,
    }

    # ── Step 9: Cache result in Redis ─────────────────────────────────────
    await _cache_prediction(redis_client, cache_key, response, ttl_seconds=3600)

    logger.info(
        f"Prediction complete: {ticker} → {prediction} "
        f"({proba_up*100:.1f}%) in {latency_ms:.0f}ms"
    )

    return response


# ══════════════════════════════════════════════════════════════════════════════
#  SCREENER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

async def run_screener(
    tickers:       List[str],
    pipeline:      object,
    feature_cols:  List[str],
    threshold:     float  = 0.5,
    redis_client:  object = None,
    top_n:         int    = 20,
) -> Dict:
    """
    Run predictions for multiple tickers and return ranked results.

    Used by GET /screener endpoint to scan 500+ stocks.
    Runs predictions concurrently with a semaphore to limit parallelism.

    Why a semaphore?
    ─────────────────────────────────────────────────────────────
    Without a semaphore, asyncio.gather on 500 tickers would fire
    500 simultaneous yfinance requests. yfinance would rate-limit
    or ban your IP. The semaphore limits to 10 concurrent requests,
    which yfinance handles comfortably.

    Parameters
    ----------
    tickers      : List of stock symbols to screen.
    pipeline     : Fitted sklearn Pipeline.
    feature_cols : Feature column names from training.
    threshold    : Classification threshold for UP/DOWN.
    redis_client : Async Redis client (None = no caching).
    top_n        : Return only top N UP signals by confidence.

    Returns
    -------
    Dict with ranked list of predictions and screener metadata.
    """
    semaphore = asyncio.Semaphore(10)

    async def _predict_one(ticker: str) -> Optional[Dict]:
        async with semaphore:
            try:
                result = await generate_prediction(
                    ticker=ticker,
                    pipeline=pipeline,
                    feature_cols=feature_cols,
                    period="1y",         # shorter period for speed
                    threshold=threshold,
                    redis_client=redis_client,
                    include_shap=False,  # skip SHAP for screener speed
                )
                return {
                    "ticker":        ticker,
                    "prediction":    result["prediction"],
                    "confidence":    result["confidence_pct"],
                    "probability":   result["probability"],
                    "rsi":           result["market_data"].get("rsi"),
                    "sentiment":     result["market_data"].get("sentiment_score"),
                    "price_change":  result["market_data"].get("price_change_pct"),
                    "current_price": result["market_data"].get("current_price"),
                }
            except Exception as e:
                logger.debug(f"Screener skipped {ticker}: {e}")
                return None

    start_time = time.time()

    # Run all predictions concurrently (limited by semaphore)
    tasks   = [_predict_one(t) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Filter out failures and rank by confidence
    valid      = [r for r in results if r is not None]
    up_signals = [r for r in valid if r["prediction"] == "UP"]
    up_signals.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "screener_results": up_signals[:top_n],
        "total_screened":   len(tickers),
        "signals_found":    len(up_signals),
        "up_signals_pct":   round(
            len(up_signals) / max(len(valid), 1) * 100, 1
        ),
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "latency_ms":       round((time.time() - start_time) * 1000, 1),
    }
