"""
StockSense AI — api/routes/predict.py
=======================================
Route definitions ONLY. All pipeline logic lives in api/prediction.py.

Routes:
  GET /predict/              → single-stock prediction
  GET /predict/validate      → check if ticker is valid
  GET /predict/screener      → multi-stock screener (ranked by confidence)

Architecture note:
─────────────────────────────────────────────────────────────
  Route files should be THIN — 3-5 lines per endpoint.
  They handle: query parameters, dependency injection, error mapping.
  They do NOT: fetch data, engineer features, or run models.
  All heavy lifting is in api/prediction.py.

  Why this separation?
  - Routes can be tested with TestClient (no real ML needed)
  - prediction.py can be called from scheduled jobs, CLI, etc.
  - Easier to reason about: "where does /predict live?" → routes
                             "how does prediction work?" → prediction.py
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.deps import require_model, get_redis_client
from api.prediction import (
    generate_prediction,
    validate_ticker,
    normalise_ticker,
    run_screener,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class FeatureExplanation(BaseModel):
    """One feature's contribution to the prediction."""
    feature:         str
    shap_value:      float
    feature_value:   float
    direction:       str          # "bullish" or "bearish"
    strength:        str          # "strongly", "moderately", etc.
    explanation:     str          # plain English sentence


class FeatureGroupImportance(BaseModel):
    """SHAP importance summed by feature category."""
    group:      str
    importance: float


class MarketDataSnapshot(BaseModel):
    """Key market metrics shown in the stock page header."""
    current_price:    Optional[float] = None
    price_change_pct: Optional[float] = None
    volume:           Optional[int]   = None
    volume_ratio:     Optional[float] = None
    rsi:              Optional[float] = None
    macd_signal:      Optional[str]   = None
    bb_position:      Optional[float] = None
    atr_pct:          Optional[float] = None
    sentiment_score:  Optional[float] = None
    sentiment_trend:  Optional[float] = None
    trend_agreement:  Optional[float] = None
    pattern_signal:   Optional[float] = None


class DataFreshness(BaseModel):
    """Metadata about data freshness."""
    price_data_as_of:     Optional[str] = None
    features_computed_at: Optional[str] = None
    n_features_used:      Optional[int] = None
    lookback_days:        Optional[int] = None


class ExplanationBlock(BaseModel):
    """SHAP explanation section of the response."""
    summary:        str = ""
    top_features:   List[Dict] = Field(default_factory=list)
    feature_groups: Dict[str, float] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """Complete prediction response returned to the frontend."""

    # ── Core prediction ───────────────────────────────────────────────────
    ticker:           str
    prediction:       str         # "UP" or "DOWN"
    probability:      float
    confidence_pct:   float
    prediction_date:  str
    generated_at:     str
    horizon_days:     int = 1
    threshold_used:   float

    # ── Explanation (SHAP) ────────────────────────────────────────────────
    explanation:      ExplanationBlock = Field(default_factory=ExplanationBlock)

    # ── Market snapshot ───────────────────────────────────────────────────
    market_data:      Optional[MarketDataSnapshot] = None

    # ── Data freshness ────────────────────────────────────────────────────
    data_freshness:   Optional[DataFreshness] = None


class ScreenerItem(BaseModel):
    """One stock's result in the screener."""
    ticker:        str
    prediction:    str
    confidence:    float
    probability:   float
    rsi:           Optional[float] = None
    sentiment:     Optional[float] = None
    price_change:  Optional[float] = None
    current_price: Optional[float] = None


class ScreenerResponse(BaseModel):
    """Multi-stock screener response."""
    screener_results: List[ScreenerItem]
    total_screened:   int
    signals_found:    int
    up_signals_pct:   float
    generated_at:     str
    latency_ms:       float


class TickerValidation(BaseModel):
    """Result of ticker validation."""
    ticker: str
    valid:  bool


class ErrorDetail(BaseModel):
    """Structured error response."""
    error:   str
    detail:  str
    ticker:  Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/",
    response_model=PredictionResponse,
    summary="Predict stock direction",
    description="""
    Predicts whether a stock will go UP or DOWN tomorrow.
    Returns prediction with probability, SHAP explanation, and market data.

    The model generates a signal at today's CLOSE using today's features.
    The prediction applies to tomorrow's OPEN-to-CLOSE return.

    Features:
    - Live price data from yfinance
    - Live sentiment from NewsAPI + FinBERT
    - 315+ engineered features
    - SHAP explanation for top contributing features
    - Market data snapshot (RSI, MACD, BB, sentiment)
    - Redis caching (1 hour TTL) for repeat requests
    """,
)
async def predict_stock(
    ticker: str = Query(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g. AAPL, MSFT, GOOGL)",
        example="AAPL",
    ),
    period: str = Query(
        default="2y",
        description="Data period for feature engineering",
        pattern="^[0-9]+[ymd]$",
        example="2y",
    ),
    threshold: float = Query(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Classification threshold for UP signal (0.1–0.9)",
    ),
    top_n: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of top SHAP features to include",
    ),
    state: dict = Depends(require_model),
    redis_client=Depends(get_redis_client),
):
    """
    Main prediction endpoint — thin wrapper around generate_prediction().

    Everything interesting happens in api/prediction.py.
    This function only: normalises input → calls pipeline → maps errors.
    """
    ticker = normalise_ticker(ticker)

    try:
        result = await generate_prediction(
            ticker=ticker,
            pipeline=state["pipeline"],
            feature_cols=state["feature_cols"],
            period=period,
            threshold=threshold,
            top_n=top_n,
            redis_client=redis_client,
            include_shap=True,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error for {ticker}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed for '{ticker}': {str(e)}",
        )


@router.get(
    "/validate",
    response_model=TickerValidation,
    summary="Validate ticker symbol",
    description="Checks if a ticker symbol is valid and has price data.",
)
async def validate_ticker_endpoint(
    ticker: str = Query(..., min_length=1, max_length=10),
):
    """
    Lightweight ticker check — no model required.
    Frontend can call this to validate user input before prediction.
    """
    ticker = normalise_ticker(ticker)
    valid  = await validate_ticker(ticker)
    return TickerValidation(ticker=ticker, valid=valid)


@router.get(
    "/screener",
    response_model=ScreenerResponse,
    summary="Multi-stock screener",
    description="""
    Scans multiple tickers and returns UP signals ranked by confidence.
    Default scans a curated list of 20 liquid US stocks.
    """,
)
async def screener(
    threshold: float = Query(default=0.5, ge=0.1, le=0.9),
    top_n: int = Query(default=20, ge=1, le=100),
    state: dict = Depends(require_model),
    redis_client=Depends(get_redis_client),
):
    """
    Multi-stock screener endpoint.
    Runs concurrent predictions for a curated stock list.
    """
    # Curated list of liquid, widely-traded US stocks
    # Expand this as needed or accept a custom list via POST body
    default_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD",
        "DIS", "NFLX", "PYPL", "INTC", "AMD",
    ]

    try:
        result = await run_screener(
            tickers=default_tickers,
            pipeline=state["pipeline"],
            feature_cols=state["feature_cols"],
            threshold=threshold,
            redis_client=redis_client,
            top_n=top_n,
        )
        return result

    except Exception as e:
        logger.exception("Screener failed")
        raise HTTPException(
            status_code=500,
            detail=f"Screener failed: {str(e)}",
        )