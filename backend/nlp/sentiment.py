"""
StockSense AI — nlp/sentiment.py
==================================
Sentiment scoring pipeline — bridge between raw articles and feature matrix.

This file owns:
  - Article cleaning and deduplication
  - Batch FinBERT scoring via finbert.py
  - Aggregation of N articles → one daily sentiment feature row
  - Sentiment momentum calculation
  - Daily sentiment DataFrame construction

It does NOT own:
  - News fetching              → data/news.py
  - FinBERT model loading      → nlp/finbert.py
  - Merging with price data    → data/merger.py
  - Feature engineering        → features/engineer.py

Why aggregate to multiple features rather than one score?
─────────────────────────────────────────────────────────────
  A single daily score loses information. The maximum score captures
  the "biggest story". The standard deviation captures disagreement
  between articles. The article count captures market attention.
  XGBoost learns which aspects are predictive — we should not
  pre-decide by collapsing to one number.

Why deduplicate by title before scoring?
─────────────────────────────────────────────────────────────
  Wire service articles (Reuters, Bloomberg) are syndicated to
  dozens of outlets. Without deduplication, one important headline
  appears 10× in article_count, artificially inflating attention
  signals and biasing the mean toward that one article's score.

Why include sentiment_momentum?
─────────────────────────────────────────────────────────────
  Absolute sentiment level tells you today's tone.
  Momentum tells you if the tone is improving or worsening.
  A stock going from negative to neutral news often recovers
  before the price does — momentum captures this leading signal.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from nlp.finbert import score_batch, get_cache_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Maximum characters from description to append to title.
# Longer descriptions get truncated to stay within FinBERT's 512 token limit.
# 200 chars ≈ 50 tokens — safe margin when combined with a typical title.
MAX_DESCRIPTION_CHARS = 200

# Neutral sentiment feature row used when no articles are available.
# These exact column names are referenced by merger.py and assembler.py.
NEUTRAL_DAILY_FEATURES: Dict[str, float] = {
    "sentiment_mean":     0.0,
    "sentiment_max":      0.0,
    "sentiment_min":      0.0,
    "sentiment_std":      0.0,
    "sentiment_momentum": 0.0,
    "article_count":      0.0,
    "positive_ratio":     0.5,
    "confidence_mean":    0.0,
}

# Column names exported to feature matrix.
# Must match NEUTRAL_DAILY_FEATURES keys exactly.
SENTIMENT_FEATURE_COLS = list(NEUTRAL_DAILY_FEATURES.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _clean_text(title: str, description: str = "") -> str:
    """
    Combine title and description into one clean scoring text.

    Why combine title + description?
    ─────────────────────────────────────────────────────────────
    Title alone: "Apple reports quarterly results"  → neutral
    Title + desc: "Apple reports quarterly results. Revenue beat
                   estimates by 8%, EPS up 15% year-over-year." → positive

    Description adds crucial context. We truncate it to 200 chars
    to stay safely within FinBERT's 512 token limit.

    Why these specific removal strings?
    ─────────────────────────────────────────────────────────────
    NewsAPI returns "[Removed]" for articles taken down after indexing.
    "N/A" and empty strings are API artifacts with no content.
    """
    # Clean title
    bad_values = {"[removed]", "n/a", "null", "none", ""}
    title = (title or "").strip()
    if title.lower() in bad_values:
        title = ""

    # Clean and truncate description
    desc = (description or "").strip()
    if desc.lower() in bad_values:
        desc = ""
    desc = desc[:MAX_DESCRIPTION_CHARS]

    # Combine
    if title and desc:
        return f"{title}. {desc}"
    return title or desc


def _deduplicate_articles(articles: List[Dict]) -> List[Dict]:
    """
    Remove duplicate articles by normalised title.

    Why normalised (lowercase, stripped)?
    ─────────────────────────────────────────────────────────────
    "Apple Beats Earnings" and "apple beats earnings" are the same
    article from different sources with different capitalisation.
    Normalising before comparison catches these duplicates.

    Why keep first occurrence?
    ─────────────────────────────────────────────────────────────
    The first occurrence is typically from the original source
    (e.g. Reuters). Duplicates are syndicated copies. Keeping first
    preserves the original source attribution.
    """
    seen   = set()
    unique = []
    for article in articles:
        title = (article.get("title") or "").strip().lower()
        if title and title not in seen:
            seen.add(title)
            unique.append(article)
    return unique


def _aggregate_scores(
    score_results: List[Dict],
    prev_mean: Optional[float] = None,
) -> Dict[str, float]:
    """
    Collapse N FinBERT score dicts into one day's sentiment features.

    Parameters
    ----------
    score_results : List of dicts from finbert.score_batch().
                    Each dict has: score, positive, negative, neutral, label.
    prev_mean     : Yesterday's sentiment_mean for momentum calculation.
                    None if no previous day available → momentum = 0.0.

    Returns
    -------
    Dict matching SENTIMENT_FEATURE_COLS keys.

    Why confidence_mean?
    ─────────────────────────────────────────────────────────────
    FinBERT confidence = max(P(positive), P(negative)).
    High confidence on a positive score is more reliable than
    low confidence. Feeding confidence_mean to XGBoost allows it
    to discount uncertain days automatically.

    Why sentiment_std?
    ─────────────────────────────────────────────────────────────
    High std means articles strongly disagree (some very positive,
    some very negative). This indicates genuine market uncertainty —
    often a signal of high volatility ahead regardless of direction.
    """
    if not score_results:
        features = dict(NEUTRAL_DAILY_FEATURES)
        if prev_mean is not None:
            features["sentiment_momentum"] = 0.0 - prev_mean
        return features

    scores      = [r["score"]    for r in score_results]
    confidences = [max(r["positive"], r["negative"]) for r in score_results]
    labels      = [r["label"]    for r in score_results]

    mean = float(np.mean(scores))

    return {
        "sentiment_mean":     round(mean, 4),
        "sentiment_max":      round(float(np.max(scores)),  4),
        "sentiment_min":      round(float(np.min(scores)),  4),
        "sentiment_std":      round(float(np.std(scores)),  4),
        "sentiment_momentum": round(mean - (prev_mean or 0.0), 4),
        "article_count":      float(len(scores)),
        "positive_ratio":     round(
            sum(1 for l in labels if l == "positive") / len(labels), 4
        ),
        "confidence_mean":    round(float(np.mean(confidences)), 4),
    }


def _aggregate_to_daily(
    scored_articles: List[Dict],
    price_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Aggregate pre-scored articles into a daily sentiment DataFrame.
    Called by both build_daily_sentiment (single stock) and
    build_sentiment_for_stocks (multi-stock, articles pre-scored).

    Separating aggregation from scoring allows:
    - build_daily_sentiment: scores + aggregates (single stock workflow)
    - build_sentiment_for_stocks: bulk scores first, then aggregates per stock
    """
    trading_days = price_df.index.normalize()

    if not scored_articles:
        return _neutral_sentiment_df(price_df.index)

    # Assign trading dates
    def get_next_trading_day(date_str: str) -> Optional[pd.Timestamp]:
        try:
            date = pd.Timestamp(date_str).normalize()
        except Exception:
            return None
        future = trading_days[trading_days >= date]
        return future[0] if len(future) > 0 else None

    for article in scored_articles:
        article["trading_date"] = get_next_trading_day(
            article.get("date", "") or article.get("trading_date", "")
        )

    scored_articles = [a for a in scored_articles if a["trading_date"] is not None]

    if not scored_articles:
        return _neutral_sentiment_df(price_df.index)

    # Group by trading day
    day_groups: Dict[pd.Timestamp, List[Dict]] = {}
    for article in scored_articles:
        day_groups.setdefault(article["trading_date"], []).append(article)

    # Aggregate each day with momentum
    daily_rows = {}
    prev_mean  = None

    for trading_day in sorted(day_groups.keys()):
        unique_articles = _deduplicate_articles(day_groups[trading_day])
        unique_scores = [
            {k: a[k] for k in ["score", "positive", "negative", "neutral", "label"]}
            for a in unique_articles
        ]
        features = _aggregate_scores(unique_scores, prev_mean=prev_mean)
        daily_rows[trading_day] = features
        prev_mean = features["sentiment_mean"]

    # Build and reindex DataFrame
    daily_df = pd.DataFrame.from_dict(daily_rows, orient="index")
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df.index.name = "date"
    daily_df = daily_df.sort_index().reindex(price_df.index)

    for col, neutral_val in NEUTRAL_DAILY_FEATURES.items():
        if col in daily_df.columns:
            daily_df[col] = daily_df[col].fillna(neutral_val)
        else:
            daily_df[col] = neutral_val

    if verbose:
        coverage = (daily_df["article_count"] > 0).mean() * 100
        print(f"  Coverage: {coverage:.1f}% | "
              f"Mean: {daily_df['sentiment_mean'].mean():+.3f} | "
              f"Avg articles: {daily_df['article_count'].mean():.1f}")

    return daily_df


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-STOCK PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def build_daily_sentiment(
    articles: List[Dict],
    price_df: pd.DataFrame,
    model_name: str = "finbert",
    batch_size: int = 32,
    show_progress: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: raw articles → daily sentiment feature DataFrame.

    Steps:
      1. Assign each article to its next trading day
         (weekend/holiday articles → next market open)
      2. Score all articles with FinBERT in one batched pass
      3. Group by trading day
      4. Deduplicate within each day
      5. Aggregate to sentiment features
      6. Add sentiment_momentum (requires sequential day processing)
      7. Reindex to all trading days, fill missing with neutral

    Parameters
    ----------
    articles     : Raw article dicts from data/news.py.
    price_df     : OHLCV DataFrame used to determine valid trading days.
                   Used for date alignment and reindexing.
    model_name   : FinBERT model config key.
    batch_size   : Texts per FinBERT forward pass.
    show_progress: Print FinBERT batch progress.
    verbose      : Print pipeline summary.

    Returns
    -------
    pd.DataFrame with DatetimeIndex (trading days only) and columns
    matching SENTIMENT_FEATURE_COLS.
    All trading days are present — missing days filled with NEUTRAL_DAILY_FEATURES.

    Why reindex to price_df.index?
    ─────────────────────────────────────────────────────────────
    merger.py merges sentiment with price data on date.
    If sentiment has gaps (weekends without news), the merge
    produces NaN. Reindexing fills all gaps with neutral (0.0)
    before the merge — cleaner than handling NaN in merger.py.
    """
    if not articles:
        if verbose:
            print("[sentiment] No articles provided — returning neutral DataFrame")
        return _neutral_sentiment_df(price_df.index)

    if verbose:
        print(f"[sentiment] Scoring {len(articles)} articles...")

    # Score
    texts      = [_clean_text(a.get("title",""), a.get("description","")) for a in articles]
    scores     = score_batch(texts, model_name=model_name,
                             batch_size=batch_size, show_progress=show_progress)
    scored     = [{**a, **s} for a, s in zip(articles, scores)]

    # Aggregate
    return _aggregate_to_daily(scored, price_df, verbose=verbose)


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH PROCESSING FOR MULTIPLE STOCKS
# ══════════════════════════════════════════════════════════════════════════════

# Add this to sentiment.py — replaces the sequential loop in build_sentiment_for_stocks

def score_all_stocks_articles(
    stock_articles: Dict[str, List[Dict]],
    model_name: str = "finbert",
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Score articles for ALL stocks in one single FinBERT batch pass.

    Why one pass across all stocks?
    ─────────────────────────────────────────────────────────────
    Calling score_batch once per stock pays the GPU kernel launch
    overhead 500 times. Concatenating all articles into one list
    and scoring together pays it once. For 500 stocks × 10 articles,
    this is ~10× faster on CPU and ~30× faster on GPU.

    Steps:
      1. Flatten all articles from all stocks into one list
      2. Track which stock each article belongs to (index mapping)
      3. Score everything in one score_batch call
      4. Re-split results back to per-stock lists

    Parameters
    ----------
    stock_articles : Dict mapping ticker → list of article dicts
    model_name     : FinBERT model config key
    batch_size     : Texts per forward pass
    verbose        : Print progress

    Returns
    -------
    Dict mapping ticker → list of scored article dicts
    (same structure as input but each dict has score/positive/negative/
    neutral/label keys added)
    """
    # ── Step 1: Flatten all articles with stock tracking ──────────────────
    flat_texts   = []
    flat_tickers = []   # parallel list: which ticker does flat_texts[i] belong to?
    flat_indices = []   # parallel list: which index within that ticker's list?

    for ticker, articles in stock_articles.items():
        for idx, article in enumerate(articles):
            text = _clean_text(
                article.get("title", ""),
                article.get("description", "")
            )
            flat_texts.append(text)
            flat_tickers.append(ticker)
            flat_indices.append(idx)

    if not flat_texts:
        return {ticker: [] for ticker in stock_articles}

    if verbose:
        print(f"[sentiment] Scoring {len(flat_texts)} articles across "
              f"{len(stock_articles)} stocks in one batch pass...")

    # ── Step 2: Score everything in one pass ──────────────────────────────
    all_scores = score_batch(
        flat_texts,
        model_name=model_name,
        batch_size=batch_size,
        show_progress=verbose,
    )

    # ── Step 3: Re-split back to per-stock lists ──────────────────────────
    # Build scored article dicts grouped by ticker
    scored: Dict[str, List[Dict]] = {ticker: [] for ticker in stock_articles}

    for i, (ticker, orig_idx, score_result) in enumerate(
        zip(flat_tickers, flat_indices, all_scores)
    ):
        original_article = stock_articles[ticker][orig_idx]
        scored[ticker].append({**original_article, **score_result})

    if verbose:
        from nlp.finbert import get_cache_stats
        stats = get_cache_stats()
        print(f"[sentiment] Batch complete. Cache hit rate: "
              f"{stats['hit_rate_pct']}%")

    return scored


def build_sentiment_for_stocks(
    stock_articles: Dict[str, List[Dict]],
    price_dfs: Dict[str, pd.DataFrame],
    model_name: str = "finbert",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Build daily sentiment DataFrames for multiple stocks.
    Scores all articles across all stocks in ONE FinBERT batch pass,
    then aggregates per stock. See score_all_stocks_articles for why.
    """
    # ── Score ALL articles across ALL stocks in one pass ──────────────────
    scored_articles = score_all_stocks_articles(
        stock_articles,
        model_name=model_name,
        verbose=verbose,
    )

    # ── Aggregate per stock ───────────────────────────────────────────────
    results = {}
    total   = len(scored_articles)

    for i, (ticker, articles) in enumerate(scored_articles.items()):
        if verbose:
            print(f"\n[{i+1}/{total}] Aggregating sentiment for {ticker}")

        if ticker not in price_dfs:
            if verbose:
                print(f"  [{ticker}] No price DataFrame — skipping")
            continue

        # Articles are already scored — skip the scoring step in build_daily_sentiment
        # by calling the aggregation pipeline directly
        daily_sentiment = _aggregate_to_daily(
            scored_articles=articles,
            price_df=price_dfs[ticker],
            verbose=verbose,
        )
        results[ticker] = daily_sentiment

    if verbose:
        print(f"\n[sentiment] Complete: {len(results)}/{total} stocks processed")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _neutral_sentiment_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Create a neutral sentiment DataFrame for all trading days.
    Used when no articles are available for a stock or period.
    Centralised here — same pattern as _neutral_result() in finbert.py.
    """
    df = pd.DataFrame(
        NEUTRAL_DAILY_FEATURES,
        index=index,
    )
    df.index.name = "date"
    return df


def get_sentiment_summary(daily_df: pd.DataFrame) -> Dict:
    """
    Produce a human-readable summary of a daily sentiment DataFrame.
    Used for logging, debugging, and the FastAPI health endpoint.

    Returns
    -------
    Dict with coverage, mean, volatility, and bullish/bearish day counts.
    """
    if daily_df.empty:
        return {"status": "empty"}

    covered = daily_df["article_count"] > 0
    scores  = daily_df.loc[covered, "sentiment_mean"]

    return {
        "total_days":       len(daily_df),
        "days_with_news":   int(covered.sum()),
        "coverage_pct":     round(covered.mean() * 100, 1),
        "mean_sentiment":   round(float(scores.mean()), 4) if len(scores) else 0.0,
        "std_sentiment":    round(float(scores.std()),  4) if len(scores) else 0.0,
        "bullish_days":     int((scores > 0.1).sum()),
        "bearish_days":     int((scores < -0.1).sum()),
        "neutral_days":     int((scores.between(-0.1, 0.1)).sum()),
        "avg_articles_day": round(float(daily_df["article_count"].mean()), 1),
    }


def list_sentiment_features(verbose: bool = True) -> List[str]:
    """
    List all sentiment feature columns produced by this module.
    Mirrors list_model_configs() / list_label_configs() pattern.
    """
    if verbose:
        print(f"\n{'Feature':<22}  {'Neutral value':>14}  Description")
        print("─" * 75)
        descriptions = {
            "sentiment_mean":     "Mean of P(pos)-P(neg) across all articles",
            "sentiment_max":      "Most positive article score of the day",
            "sentiment_min":      "Most negative article score of the day",
            "sentiment_std":      "Disagreement between articles (high=uncertain)",
            "sentiment_momentum": "Today's mean minus yesterday's mean",
            "article_count":      "Number of unique articles (attention signal)",
            "positive_ratio":     "Fraction of articles labelled positive",
            "confidence_mean":    "Mean FinBERT confidence across articles",
        }
        for col, neutral in NEUTRAL_DAILY_FEATURES.items():
            desc = descriptions.get(col, "")
            print(f"  {col:<20}  {str(neutral):>14}  {desc}")
    return SENTIMENT_FEATURE_COLS


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yfinance as yf
    import sys
    sys.path.append("..")

    from data.news import fetch_news_for_stock

    # ── List features
    list_sentiment_features()

    # ── Fetch price data for date alignment
    raw = yf.download("AAPL", period="3mo", auto_adjust=True, progress=False)
    raw.columns  = [c.lower() for c in raw.columns]
    price_df = raw[["open", "high", "low", "close", "volume"]]

    # ── Fetch news
    print("\nFetching AAPL news...")
    articles = fetch_news_for_stock("AAPL", days_back=30, max_articles=20)
    print(f"Fetched {len(articles)} articles")

    # ── Build daily sentiment
    daily = build_daily_sentiment(
        articles=articles,
        price_df=price_df,
        verbose=True,
    )

    # ── Show results
    print("\n=== Daily Sentiment (last 10 trading days) ===")
    display_cols = [
        "sentiment_mean", "sentiment_max",
        "article_count",  "positive_ratio",
        "sentiment_momentum"
    ]
    print(daily[display_cols].tail(10).round(3))

    # ── Summary
    print("\n=== Summary ===")
    summary = get_sentiment_summary(daily)
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
