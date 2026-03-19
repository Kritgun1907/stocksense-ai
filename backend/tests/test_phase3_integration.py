"""
StockSense AI — tests/test_phase3_integration.py
==================================================
End-to-end integration test for Phase 3 sentiment pipeline.

Tests:
  1.  Sentiment columns present after merge
  2.  1-day shift correctly applied (no leakage)
  3.  No NaN in final feature matrix
  4.  Rolling sentiment features present and non-trivial
  5.  Feature matrix shape consistent with placeholder vs real sentiment
  6.  No lookahead bias in sentiment features (architectural check)
  7.  Exact sentiment feature counts (regression guard)
  8.  _aggregate_to_daily fills neutral values for all trading days
  9.  build_daily_sentiment gracefully handles zero articles
  10. Rolling feature group registry completeness
  11. Assembler placeholder parity with NEUTRAL_DAILY_FEATURES
  12. Cross-signal features degrade gracefully when price cols absent

Run with:
  cd backend
  python tests/test_phase3_integration.py
"""

import sys
import os
import inspect
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

# ── Ensure backend/ is on sys.path regardless of CWD ─────────────────────────
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def pass_fail(name: str, condition: bool, detail: str = "") -> bool:
    """Print a test result and return whether it passed."""
    status     = "✅ PASS" if condition else "❌ FAIL"
    detail_str = f"  ({detail})" if detail else ""
    print(f"  {status}  {name}{detail_str}")
    return condition


def section(title: str) -> None:
    print(f"\n{'═'*58}")
    print(f"  {title}")
    print(f"{'═'*58}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

def get_test_price_df(ticker: str = "AAPL", period: str = "3mo") -> pd.DataFrame:
    """
    Fetch a small OHLCV DataFrame for testing.
    Normalises column names and flattens any MultiIndex columns
    (yfinance wraps them in a MultiIndex when auto_adjust=True).
    """
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    return raw


def make_synthetic_price_df(periods: int = 30) -> pd.DataFrame:
    """
    Build a fully deterministic price DataFrame for unit tests
    that do not need live data.  Uses business-day dates starting
    2024-01-02 so the index is a valid DatetimeIndex.
    """
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    rng   = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, periods))
    return pd.DataFrame({
        "open":   close * 0.99,
        "high":   close * 1.01,
        "low":    close * 0.98,
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, periods).astype(float),
    }, index=dates)


def get_fake_articles(n: int = 20, ticker: str = "AAPL") -> list:
    """
    Generate fake article dicts for testing without hitting NewsAPI.
    Dates are spread across the last 30 calendar days so that
    _aggregate_to_daily assigns them to real trading days.

    Each dict includes both 'title' (used by FinBERT) and 'date'
    (used by _aggregate_to_daily to find the next trading day).
    The 'description' field is intentionally omitted to exercise
    the empty-description fallback in _clean_text().
    """
    from datetime import datetime, timedelta

    positives = [
        "Apple beats earnings expectations by wide margin",
        "iPhone demand remains strong in key markets",
        "Apple stock surges on analyst upgrade",
    ]
    negatives = [
        "Apple faces headwinds in China market",
        "iPhone sales disappoint analysts this quarter",
        "Apple supply chain disruption concerns mount",
    ]
    neutrals = [
        "Apple CEO Tim Cook to speak at conference",
        "Apple announces product launch event next month",
        "Apple files new patent application",
    ]

    all_headlines = positives + negatives + neutrals
    articles      = []
    today         = datetime.now()

    for i in range(n):
        date = today - timedelta(days=i % 28)   # stay within 30-day window
        articles.append({
            "ticker":    ticker,
            "title":     all_headlines[i % len(all_headlines)],
            "date":      date.strftime("%Y-%m-%d"),
            "publisher": "TestSource",
        })

    return articles


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — Sentiment columns present after merge
# ══════════════════════════════════════════════════════════════════════════════

def test_sentiment_columns_present() -> bool:
    """
    All 8 SENTIMENT_FEATURE_COLS must exist in the merged DataFrame
    after build_daily_sentiment → merge_price_and_sentiment.
    """
    section("Test 1: Sentiment columns present after merge")

    from nlp.sentiment import build_daily_sentiment, SENTIMENT_FEATURE_COLS
    from data.merger   import merge_price_and_sentiment

    price_df        = get_test_price_df()
    articles        = get_fake_articles(30)
    daily_sentiment = build_daily_sentiment(articles, price_df, verbose=False)
    merged          = merge_price_and_sentiment(price_df, daily_sentiment, shift_days=1)

    results = []
    for col in SENTIMENT_FEATURE_COLS:
        present = col in merged.columns
        results.append(pass_fail(f"Column '{col}' present", present))

    results.append(pass_fail(
        "All 8 columns present in one check",
        all(col in merged.columns for col in SENTIMENT_FEATURE_COLS),
        f"merged has {len(merged.columns)} columns total",
    ))

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — 1-day shift correctly applied (no lookahead)
# ══════════════════════════════════════════════════════════════════════════════

def test_one_day_shift_applied() -> bool:
    """
    With shift_days=1, row[1]['sentiment_mean'] must equal
    the *un-shifted* row[0] value — yesterday's news drives today's feature.

    Uses a fully synthetic price + sentiment DataFrame so the result is
    deterministic regardless of market data.
    """
    section("Test 2: 1-day shift correctly applied (no leakage)")

    from data.merger import merge_price_and_sentiment

    price_df = make_synthetic_price_df(10)

    # Sentiment with monotonically decreasing scores for easy tracing
    sent_scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    sentiment_df = pd.DataFrame({
        "sentiment_mean":     sent_scores,
        "sentiment_max":      [0.9] * 10,
        "sentiment_min":      [0.0] * 10,
        "sentiment_std":      [0.1] * 10,
        "sentiment_momentum": [0.0] * 10,
        "article_count":      [3.0] * 10,
        "positive_ratio":     [0.8] * 10,
        "confidence_mean":    [0.7] * 10,
    }, index=price_df.index)

    merged = merge_price_and_sentiment(price_df, sentiment_df, shift_days=1)

    # After a shift of 1:
    #   merged.iloc[1]['sentiment_mean'] should be sentiment_df.iloc[0] = 0.9
    #   merged.iloc[0]['sentiment_mean'] should be NaN (no previous day)
    #   (merger.py fills NaN with 0.0 in Step 7, so we check for 0.0)
    day0_after_merge  = float(merged["sentiment_mean"].iloc[0])
    day1_after_merge  = float(merged["sentiment_mean"].iloc[1])
    day0_original     = sent_scores[0]   # 0.9

    results = [
        pass_fail(
            "Day 2 sentiment_mean == Day 1 original score",
            abs(day1_after_merge - day0_original) < 0.001,
            f"got {day1_after_merge:.3f}, expected {day0_original:.3f}",
        ),
        pass_fail(
            "Day 1 sentiment_mean is 0.0 (neutral fill — no prior day)",
            day0_after_merge == 0.0,
            f"got {day0_after_merge}",
        ),
        pass_fail(
            "Merged row count unchanged",
            len(merged) == len(price_df),
            f"{len(price_df)} → {len(merged)}",
        ),
    ]

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — No NaN in final feature matrix
# ══════════════════════════════════════════════════════════════════════════════

def test_no_nan_after_pipeline() -> bool:
    """
    After the full pipeline (clean → sentiment → build_features → label →
    get_model_features), the final X matrix must have zero NaN and zero inf.

    Uses period='1y' so that sma_200 in add_trend_features() has enough
    rows to warm up (needs 200 trading days, 6mo only gives ~125).
    """
    section("Test 3: No NaN in final feature matrix")

    from data.cleaner    import clean_stock_data
    from nlp.sentiment   import build_daily_sentiment
    from data.merger     import merge_price_and_sentiment
    from features.engineer   import build_features
    from features.indicators import get_model_features
    from data.labeller   import create_labels

    price_df = get_test_price_df(period="1y")
    clean    = clean_stock_data(price_df, ticker="AAPL")
    articles = get_fake_articles(50)

    daily_sentiment = build_daily_sentiment(articles, clean, verbose=False)
    merged          = merge_price_and_sentiment(clean, daily_sentiment, shift_days=1)
    featured        = build_features(merged).dropna()
    labelled        = create_labels(featured, horizon=1, threshold=0.003, verbose=False)
    X               = get_model_features(labelled, extra_drop=["target"])

    nan_count = int(X.isnull().sum().sum())
    inf_count = int(np.isinf(X.select_dtypes(include=[np.number]).values).sum())

    results = [
        pass_fail("Zero NaN in feature matrix",  nan_count == 0,
                  f"{nan_count} NaN values found"),
        pass_fail("Zero inf in feature matrix",  inf_count == 0,
                  f"{inf_count} inf values found"),
        pass_fail("Feature matrix not empty",    len(X) > 0,
                  f"{len(X)} rows"),
        pass_fail("Feature matrix is all-numeric",
                  all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns),
                  f"{X.shape[1]} feature columns"),
    ]

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Rolling sentiment features present and non-trivial
# ══════════════════════════════════════════════════════════════════════════════

def test_rolling_features_present() -> bool:
    """
    Key rolling sentiment features must be present in the featured DataFrame
    and must not be all-zero — i.e. they must carry genuine information.

    Uses period='1y' so sma_200 in add_trend_features() has enough warmup rows.
    """
    section("Test 4: Rolling sentiment features present and non-trivial")

    from data.cleaner  import clean_stock_data
    from nlp.sentiment import build_daily_sentiment
    from data.merger   import merge_price_and_sentiment
    from features.engineer import build_features
    from nlp.rolling   import ALL_ROLLING_SENTIMENT_COLS

    price_df = get_test_price_df(period="1y")
    clean    = clean_stock_data(price_df, ticker="AAPL")
    articles = get_fake_articles(60)

    daily_sentiment = build_daily_sentiment(articles, clean, verbose=False)
    merged          = merge_price_and_sentiment(clean, daily_sentiment, shift_days=1)
    featured        = build_features(merged).dropna()

    results = []

    # Spot-check the most important columns across every feature group.
    # sent_rsi_diverge is a rare-event flag (regime>0 AND rsi>70 simultaneously)
    # so it can legitimately be all-zeros in a 1y sample — checked for
    # presence only, not non-zero activity.
    rare_event_cols = {"sent_rsi_diverge"}

    key_cols = [
        "sentiment_ma3",          # rolling_means
        "sentiment_ma7",
        "sentiment_ma14",
        "sentiment_vol_3d",       # rolling_volatility
        "sentiment_trend_3_7",    # trend
        "sentiment_regime",
        "sentiment_improving",
        "sentiment_momentum_3d",  # momentum
        "sentiment_rsi",
        "sentiment_streak",
        "article_count_ma7",      # attention
        "article_count_spike",
        "attention_trend",
        "sent_price_confirm",     # cross_signal
        "sent_rsi_diverge",
        "sent_momentum_align",
    ]

    for col in key_cols:
        present = col in featured.columns
        if col in rare_event_cols:
            # Rare-event columns: only check presence, not activity
            results.append(pass_fail(
                f"'{col}' present (rare-event, zero ok)",
                present,
                "missing" if not present else "ok",
            ))
        else:
            non_zero = present and (featured[col].abs().sum() > 0)
            results.append(pass_fail(
                f"'{col}' present & non-trivial",
                present and non_zero,
                "missing" if not present else "all zeros" if not non_zero else "ok",
            ))

    # All ALL_ROLLING_SENTIMENT_COLS must be present
    all_present = all(c in featured.columns for c in ALL_ROLLING_SENTIMENT_COLS)
    results.append(pass_fail(
        f"All {len(ALL_ROLLING_SENTIMENT_COLS)} rolling sentiment cols in featured",
        all_present,
    ))

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — Feature matrix shape consistent (placeholder vs real sentiment)
# ══════════════════════════════════════════════════════════════════════════════

def test_shape_consistency() -> bool:
    """
    The final model-ready X must have:
      - Identical feature column names regardless of whether sentiment
        came from placeholder zeros or real FinBERT output.
      - Approximately the same row count (within 10%).

    This is the key regression guard — adding a new column to one path
    but not the other breaks XGBoost at prediction time.

    Uses period='1y' so sma_200 in add_trend_features() has enough warmup rows.
    """
    section("Test 5: Shape consistency (placeholder vs real sentiment)")

    from data.cleaner    import clean_stock_data
    from nlp.sentiment   import build_daily_sentiment, SENTIMENT_FEATURE_COLS
    from data.merger     import merge_price_and_sentiment
    from features.engineer   import build_features
    from features.indicators import get_model_features
    from data.labeller   import create_labels

    price_df = get_test_price_df(period="1y")
    clean    = clean_stock_data(price_df, ticker="AAPL")

    # ── Pipeline A: placeholder sentiment (zeros) ─────────────────────────────
    clean_a = clean.copy()
    for col in SENTIMENT_FEATURE_COLS:
        clean_a[col] = 0.0
    featured_a = build_features(clean_a).dropna()
    labelled_a = create_labels(featured_a, horizon=1, threshold=0.003, verbose=False)
    X_a        = get_model_features(labelled_a, extra_drop=["target"])

    # ── Pipeline B: real sentiment (fake articles via FinBERT) ────────────────
    articles    = get_fake_articles(60)
    daily       = build_daily_sentiment(articles, clean, verbose=False)
    merged_b    = merge_price_and_sentiment(clean, daily, shift_days=1)
    featured_b  = build_features(merged_b).dropna()
    labelled_b  = create_labels(featured_b, horizon=1, threshold=0.003, verbose=False)
    X_b         = get_model_features(labelled_b, extra_drop=["target"])

    sym_diff = set(X_a.columns).symmetric_difference(set(X_b.columns))

    results = [
        pass_fail(
            "Same number of feature columns",
            X_a.shape[1] == X_b.shape[1],
            f"placeholder={X_a.shape[1]}, real={X_b.shape[1]}",
        ),
        pass_fail(
            "Identical feature column names",
            len(sym_diff) == 0,
            f"{len(sym_diff)} columns differ: {sorted(sym_diff)[:5]}",
        ),
        pass_fail(
            "Row counts within 10% of each other",
            abs(len(X_a) - len(X_b)) / max(len(X_a), 1) < 0.10,
            f"placeholder={len(X_a)}, real={len(X_b)}",
        ),
    ]

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — No lookahead bias (architectural / source inspection)
# ══════════════════════════════════════════════════════════════════════════════

def test_no_lookahead_in_sentiment() -> bool:
    """
    Architectural verification that no code path allows future data
    to contaminate sentiment features.

    Checks:
    - merger.py uses a positive shift (yesterday → today)
    - No negative shift applied to sentiment anywhere in merger.py
    - rolling.py contains no negative shifts on feature columns
    - assembler._add_real_sentiment calls merge with shift_days=1
    """
    section("Test 6: No lookahead bias in sentiment features")

    from data import merger
    from nlp  import rolling
    from data import assembler

    merger_src   = inspect.getsource(merger.merge_price_and_sentiment)
    rolling_src  = inspect.getsource(rolling)
    assembler_src= inspect.getsource(assembler._add_real_sentiment)

    results = [
        pass_fail(
            "merger.merge_price_and_sentiment accepts shift_days param",
            "shift_days" in merger_src,
        ),
        pass_fail(
            "Sentiment is shifted forward (positive shift)",
            "shift(shift_days)" in merger_src,
        ),
        pass_fail(
            "No negative shift applied to sentiment in merger.py",
            # There must be no shift(-N) before the shift_days line
            "shift(-" not in merger_src.split("shift_days")[0],
        ),
        pass_fail(
            "rolling.py contains no negative index shifts",
            "shift(-" not in rolling_src,
        ),
        pass_fail(
            "assembler._add_real_sentiment calls merger with shift_days=1",
            "shift_days=1" in assembler_src,
        ),
    ]

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 7 — Exact sentiment feature counts (regression guard)
# ══════════════════════════════════════════════════════════════════════════════

def test_sentiment_feature_count() -> bool:
    """
    Document and regression-test the exact number of sentiment features.
    If someone adds / removes columns, this test will catch the change.
    """
    section("Test 7: Sentiment feature counts (regression guard)")

    from nlp.sentiment import SENTIMENT_FEATURE_COLS, NEUTRAL_DAILY_FEATURES
    from nlp.rolling   import (
        ALL_ROLLING_SENTIMENT_COLS,
        ROLLING_SENTIMENT_FEATURE_GROUPS,
    )

    daily_count   = len(SENTIMENT_FEATURE_COLS)
    rolling_count = len(ALL_ROLLING_SENTIMENT_COLS)
    total         = daily_count + rolling_count

    # Verify NEUTRAL_DAILY_FEATURES and SENTIMENT_FEATURE_COLS are in sync
    keys_match = set(NEUTRAL_DAILY_FEATURES.keys()) == set(SENTIMENT_FEATURE_COLS)

    # Verify group registry is internally consistent
    group_total = sum(len(v) for v in ROLLING_SENTIMENT_FEATURE_GROUPS.values())

    results = [
        pass_fail(
            f"Daily sentiment features == 8",
            daily_count == 8,
            f"expected 8, got {daily_count}",
        ),
        pass_fail(
            "SENTIMENT_FEATURE_COLS matches NEUTRAL_DAILY_FEATURES keys",
            keys_match,
        ),
        pass_fail(
            f"Rolling sentiment features >= 15",
            rolling_count >= 15,
            f"expected ≥15, got {rolling_count}",
        ),
        pass_fail(
            "ROLLING_SENTIMENT_FEATURE_GROUPS sums to ALL_ROLLING_SENTIMENT_COLS",
            group_total == rolling_count,
            f"sum of groups={group_total}, ALL list={rolling_count}",
        ),
        pass_fail(
            f"Total sentiment features >= 23",
            total >= 23,
            f"expected ≥23, got {total}",
        ),
        pass_fail(
            "All 6 feature groups present in registry",
            len(ROLLING_SENTIMENT_FEATURE_GROUPS) == 6,
            f"got {len(ROLLING_SENTIMENT_FEATURE_GROUPS)} groups: "
            f"{list(ROLLING_SENTIMENT_FEATURE_GROUPS.keys())}",
        ),
    ]

    print(f"\n  Daily features   ({daily_count}): {SENTIMENT_FEATURE_COLS}")
    print(f"  Rolling features ({rolling_count}): {ALL_ROLLING_SENTIMENT_COLS}")

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 8 — _aggregate_to_daily fills all trading days with neutral values
# ══════════════════════════════════════════════════════════════════════════════

def test_aggregate_fills_all_trading_days() -> bool:
    """
    _aggregate_to_daily must reindex to all trading days in price_df.index
    and fill missing days with NEUTRAL_DAILY_FEATURES — not leave NaN gaps.

    This test does NOT call FinBERT — it uses pre-scored articles to test
    the aggregation logic in isolation.
    """
    section("Test 8: _aggregate_to_daily fills all trading days")

    # Private import is intentional here — we're testing internal plumbing
    from nlp.sentiment import (
        _aggregate_to_daily,
        NEUTRAL_DAILY_FEATURES,
        SENTIMENT_FEATURE_COLS,
    )

    price_df = make_synthetic_price_df(30)

    # Only 3 articles, all on day 0 — the other 29 days should get neutrals
    pre_scored = [
        {
            "title":    "Apple beats earnings",
            "date":     str(price_df.index[0].date()),
            "score":     0.8,
            "positive":  0.85,
            "negative":  0.05,
            "neutral":   0.10,
            "label":     "positive",
        },
        {
            "title":    "Apple raises guidance",
            "date":     str(price_df.index[0].date()),
            "score":     0.7,
            "positive":  0.75,
            "negative":  0.10,
            "neutral":   0.15,
            "label":     "positive",
        },
        {
            "title":    "iPhone demand strong",
            "date":     str(price_df.index[0].date()),
            "score":     0.6,
            "positive":  0.65,
            "negative":  0.15,
            "neutral":   0.20,
            "label":     "positive",
        },
    ]

    daily_df = _aggregate_to_daily(pre_scored, price_df, verbose=False)

    results = [
        pass_fail(
            "Output has exactly len(price_df) rows",
            len(daily_df) == len(price_df),
            f"expected {len(price_df)}, got {len(daily_df)}",
        ),
        pass_fail(
            "All SENTIMENT_FEATURE_COLS present",
            all(c in daily_df.columns for c in SENTIMENT_FEATURE_COLS),
        ),
        pass_fail(
            "No NaN in output",
            daily_df.isnull().sum().sum() == 0,
            f"{daily_df.isnull().sum().sum()} NaN values",
        ),
        pass_fail(
            "Day 0 has positive sentiment (articles scored there)",
            daily_df["sentiment_mean"].iloc[0] > 0,
            f"day 0 mean = {daily_df['sentiment_mean'].iloc[0]:.3f}",
        ),
        pass_fail(
            "Days without news get article_count == 0.0 (neutral fill)",
            (daily_df["article_count"].iloc[1:] == 0.0).all(),
            f"non-zero days after day0: "
            f"{(daily_df['article_count'].iloc[1:] != 0).sum()}",
        ),
        pass_fail(
            "Days without news get positive_ratio == 0.5 (neutral fill)",
            (daily_df["positive_ratio"].iloc[1:] == 0.5).all(),
            f"expected 0.5 for all empty days",
        ),
    ]

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 9 — build_daily_sentiment handles zero articles gracefully
# ══════════════════════════════════════════════════════════════════════════════

def test_empty_articles_graceful() -> bool:
    """
    build_daily_sentiment([]) must return a valid neutral DataFrame
    with the correct index and columns — not raise and not return empty.
    The returned DataFrame is safe to pass directly to merge_price_and_sentiment.
    """
    section("Test 9: build_daily_sentiment handles zero articles")

    from nlp.sentiment import build_daily_sentiment, SENTIMENT_FEATURE_COLS
    from data.merger   import merge_price_and_sentiment

    price_df = make_synthetic_price_df(20)

    # Call with zero articles
    neutral_df = build_daily_sentiment([], price_df, verbose=False)

    results = [
        pass_fail(
            "Returns a DataFrame (not None or exception)",
            isinstance(neutral_df, pd.DataFrame),
        ),
        pass_fail(
            "Row count == len(price_df)",
            len(neutral_df) == len(price_df),
            f"expected {len(price_df)}, got {len(neutral_df)}",
        ),
        pass_fail(
            "All SENTIMENT_FEATURE_COLS present",
            all(c in neutral_df.columns for c in SENTIMENT_FEATURE_COLS),
        ),
        pass_fail(
            "No NaN in neutral DataFrame",
            neutral_df.isnull().sum().sum() == 0,
        ),
        pass_fail(
            "sentiment_mean is all zeros (neutral)",
            (neutral_df["sentiment_mean"] == 0.0).all(),
        ),
        pass_fail(
            "positive_ratio is all 0.5 (neutral)",
            (neutral_df["positive_ratio"] == 0.5).all(),
        ),
    ]

    # Verify it merges without error
    try:
        merged = merge_price_and_sentiment(price_df, neutral_df, shift_days=1)
        results.append(pass_fail(
            "Empty-article path merges into price_df without error",
            len(merged) == len(price_df),
        ))
    except Exception as e:
        results.append(pass_fail("Empty-article merge raises no exception", False, str(e)))

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 10 — Rolling feature group registry completeness
# ══════════════════════════════════════════════════════════════════════════════

def test_rolling_group_registry() -> bool:
    """
    Verify the ROLLING_SENTIMENT_FEATURE_GROUPS registry is internally
    consistent and that get_rolling_feature_group() returns correct subsets.
    """
    section("Test 10: Rolling feature group registry completeness")

    from nlp.rolling import (
        ROLLING_SENTIMENT_FEATURE_GROUPS,
        ALL_ROLLING_SENTIMENT_COLS,
        get_rolling_feature_group,
    )

    expected_groups = {
        "rolling_means":      3,
        "rolling_volatility": 2,
        "trend":              4,
        "momentum":           4,
        "attention":          3,
        "cross_signal":       3,
    }

    results = []
    for group, expected_count in expected_groups.items():
        cols = get_rolling_feature_group(group)
        results.append(pass_fail(
            f"Group '{group}' has {expected_count} features",
            len(cols) == expected_count,
            f"got {len(cols)}: {cols}",
        ))

    # No duplicate columns across groups
    all_cols   = [c for g in ROLLING_SENTIMENT_FEATURE_GROUPS.values() for c in g]
    no_dups    = len(all_cols) == len(set(all_cols))
    results.append(pass_fail("No duplicate columns across groups", no_dups))

    # ALL_ROLLING_SENTIMENT_COLS is exactly the union of all group lists
    results.append(pass_fail(
        "ALL_ROLLING_SENTIMENT_COLS == union of all groups",
        set(ALL_ROLLING_SENTIMENT_COLS) == set(all_cols),
    ))

    # Unknown group returns empty list (not raises)
    unknown = get_rolling_feature_group("__nonexistent__")
    results.append(pass_fail(
        "Unknown group returns [] not raises",
        unknown == [],
        f"got {unknown}",
    ))

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 11 — assembler._SENTIMENT_PLACEHOLDER parity with NEUTRAL_DAILY_FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def test_assembler_placeholder_parity() -> bool:
    """
    assembler._SENTIMENT_PLACEHOLDER must have the exact same keys and
    neutral values as nlp.sentiment.NEUTRAL_DAILY_FEATURES.

    If these drift apart, placeholder mode silently produces different
    column sets from real-sentiment mode — causing shape mismatches at
    prediction time.
    """
    section("Test 11: assembler placeholder parity with NEUTRAL_DAILY_FEATURES")

    from nlp.sentiment import NEUTRAL_DAILY_FEATURES
    from data.assembler import _SENTIMENT_PLACEHOLDER

    results = [
        pass_fail(
            "_SENTIMENT_PLACEHOLDER has 8 keys",
            len(_SENTIMENT_PLACEHOLDER) == 8,
            f"got {len(_SENTIMENT_PLACEHOLDER)}",
        ),
        pass_fail(
            "Keys are identical",
            set(_SENTIMENT_PLACEHOLDER.keys()) == set(NEUTRAL_DAILY_FEATURES.keys()),
            f"diff: {set(_SENTIMENT_PLACEHOLDER.keys()) ^ set(NEUTRAL_DAILY_FEATURES.keys())}",
        ),
        pass_fail(
            "Values are identical",
            _SENTIMENT_PLACEHOLDER == NEUTRAL_DAILY_FEATURES,
            f"assembler: {_SENTIMENT_PLACEHOLDER}\n"
            f"    sentiment: {NEUTRAL_DAILY_FEATURES}",
        ),
    ]

    # _add_sentiment_placeholders must produce a DataFrame with all 8 cols
    from data.assembler import _add_sentiment_placeholders
    price_df    = make_synthetic_price_df(10)
    with_placeholders = _add_sentiment_placeholders(price_df)

    for col, expected_val in NEUTRAL_DAILY_FEATURES.items():
        results.append(pass_fail(
            f"Placeholder col '{col}' has value {expected_val}",
            col in with_placeholders.columns
            and (with_placeholders[col] == expected_val).all(),
        ))

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 12 — Cross-signal features degrade gracefully without price columns
# ══════════════════════════════════════════════════════════════════════════════

def test_cross_signal_graceful_degradation() -> bool:
    """
    add_rolling_sentiment_features() must produce cross-signal columns
    even when ret_1d / ret_5d / rsi_14 are absent — it must fill 0.0
    and issue a warning, not raise.

    This guards the case where rolling.py is called in isolation (e.g. in
    unit tests or SHAP ablation runs that omit price indicators).
    """
    section("Test 12: Cross-signal graceful degradation (no price cols)")

    from nlp.rolling import add_rolling_sentiment_features, ROLLING_SENTIMENT_FEATURE_GROUPS

    # DataFrame with ONLY sentiment columns — no price indicators
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    rng   = np.random.default_rng(7)
    df    = pd.DataFrame({
        "sentiment_mean":     rng.uniform(-0.5, 0.5, 30),
        "sentiment_max":      rng.uniform( 0.0, 1.0, 30),
        "sentiment_min":      rng.uniform(-1.0, 0.0, 30),
        "sentiment_std":      rng.uniform( 0.0, 0.3, 30),
        "sentiment_momentum": rng.uniform(-0.2, 0.2, 30),
        "article_count":      rng.integers(0, 10, 30).astype(float),
        "positive_ratio":     rng.uniform( 0.0, 1.0, 30),
        "confidence_mean":    rng.uniform( 0.5, 1.0, 30),
    }, index=dates)

    # Should not raise — should warn instead
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = add_rolling_sentiment_features(df.copy())

    cross_signal_cols = ROLLING_SENTIMENT_FEATURE_GROUPS["cross_signal"]
    price_col_warning_issued = any("price columns" in str(w.message) for w in caught)

    results = [
        pass_fail(
            "add_rolling_sentiment_features does not raise",
            isinstance(result, pd.DataFrame),
        ),
        pass_fail(
            "Warning issued about missing price columns",
            price_col_warning_issued,
            f"{len(caught)} warnings caught",
        ),
    ]

    for col in cross_signal_cols:
        results.append(pass_fail(
            f"Cross-signal '{col}' present (filled 0)",
            col in result.columns,
        ))

    # All cross-signal features should be 0 (no price data)
    zero_fill = all((result[c] == 0).all() for c in cross_signal_cols)
    results.append(pass_fail(
        "All cross-signal features == 0 when price cols absent",
        zero_fill,
    ))

    return all(results)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 58)
    print("  StockSense AI — Phase 3 Integration Tests")
    print("═" * 58)
    print("  Running 12 tests covering the full sentiment pipeline")
    print("  (news → FinBERT → daily → merge → rolling → model)")
    print("═" * 58)

    tests = [
        ("Sentiment columns present",              test_sentiment_columns_present),
        ("1-day shift correctly applied",          test_one_day_shift_applied),
        ("No NaN after full pipeline",             test_no_nan_after_pipeline),
        ("Rolling features present & non-trivial", test_rolling_features_present),
        ("Shape consistency: placeholder vs real", test_shape_consistency),
        ("No lookahead bias (architectural)",      test_no_lookahead_in_sentiment),
        ("Sentiment feature counts",               test_sentiment_feature_count),
        ("aggregate fills all trading days",       test_aggregate_fills_all_trading_days),
        ("Empty articles handled gracefully",      test_empty_articles_graceful),
        ("Rolling group registry complete",        test_rolling_group_registry),
        ("Assembler placeholder parity",           test_assembler_placeholder_parity),
        ("Cross-signal graceful degradation",      test_cross_signal_graceful_degradation),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            print(f"\n  ❌ EXCEPTION in '{name}': {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'═'*58}")
    print(f"  Results: {passed}/{passed + failed} tests passed")
    if failed == 0:
        print("  🎉 Phase 3 integration complete — ready for Phase 4")
    else:
        print(f"  ⚠️  {failed} test(s) failed — fix before proceeding")
    print(f"{'═'*58}\n")
