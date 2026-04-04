"""
StockSense AI — api/routes/predict.py
=======================================
GET /predict?ticker=AAPL

This file owns:
  - The /predict endpoint
  - Data fetching → feature engineering → prediction → SHAP pipeline
  - Response schema for predictions

It does NOT own:
  - ML model training     → models/trainer.py
  - Feature engineering   → features/engineer.py
  - SHAP explanation      → models/explainer.py
  - The loaded pipeline   → main.py app_state (accessed via api/deps.py)
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# ── Dependency injection — avoids circular import with main.py ────────────────
# NEVER do: from api.main import app_state   ← causes circular import error
# ALWAYS do: Depends(require_model)          ← FastAPI injects at request time
from api.deps import require_model

warnings.filterwarnings("ignore")

# ── Create the router for this feature area ───────────────────────────────────
# This gets registered in main.py with prefix="/predict"
router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE SCHEMAS
#  Define the exact shape of JSON this endpoint returns.
#  These appear automatically in /docs.
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
    group:      str               # "trend", "momentum", "patterns", etc.
    importance: float             # sum of |SHAP values| for this group


class PredictionResponse(BaseModel):
    """Complete prediction response returned to the frontend."""
    
    # ── Identification ────────────────────────────────────────────────────
    ticker:           str
    prediction_date:  str         # date the features were computed for
    
    # ── Core prediction ───────────────────────────────────────────────────
    prediction:       str         # "UP" or "DOWN"
    probability:      float       # e.g. 0.73
    confidence_pct:   float       # e.g. 73.0 (same as probability × 100)
    
    # ── SHAP explanation ──────────────────────────────────────────────────
    top_features:     List[FeatureExplanation]
    feature_groups:   Dict[str, float]   # group_name → total SHAP
    explanation_text: str                # full plain-English paragraph
    
    # ── Model metadata ────────────────────────────────────────────────────
    n_features_used:  int
    threshold_used:   float


class ErrorDetail(BaseModel):
    """Structured error response."""
    error:   str
    detail:  str
    ticker:  Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
#  These functions do one job each. Keeping them small makes debugging easy.
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_and_prepare(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance and run the full feature engineering pipeline.
    
    Returns a feature DataFrame ready for model prediction.
    
    Why period="2y"?
    ────────────────────────────────────────────────────────────────
    SMA_200 in add_trend_features() needs 200 trading days to warm up.
    2 years gives ~500 trading days — enough for a clean feature set
    after dropping NaN rows from rolling windows.
    
    Raises ValueError if data is empty or insufficient.
    """
    # ── Step 1: Fetch from yfinance ────────────────────────────────────────
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
    
    # ── Step 2: Clean the data ─────────────────────────────────────────────
    from data.cleaner import clean_stock_data
    clean = clean_stock_data(raw, ticker=ticker)
    
    if len(clean) < 250:
        raise ValueError(
            f"Insufficient data for '{ticker}': only {len(clean)} rows. "
            f"Need at least 250 trading days."
        )
    
    # ── Step 3: Engineer features (14-step pipeline) ────────────────────────
    # This adds ~342 feature columns (trend, momentum, MACD, patterns, etc.)
    from features.engineer import build_features
    featured = build_features(clean).dropna()
    
    if len(featured) == 0:
        raise ValueError(
            f"Feature engineering produced no valid rows for '{ticker}'. "
            f"Data may have too many gaps."
        )
    
    return featured


def _get_latest_features(
    featured:     pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Extract the most recent row of features for prediction.
    
    Why the last row?
    ────────────────────────────────────────────────────────────────
    The last row represents today's feature values — the latest
    technical indicators, patterns, and momentum signals.
    The model predicts what happens TOMORROW based on TODAY's features.
    
    Returns a single-row DataFrame with only the model feature columns.
    """
    from features.indicators import get_model_features
    
    # get_model_features drops raw OHLCV and non-stationary columns
    # It also fills any remaining NaN with 0
    X = get_model_features(featured, extra_drop=["target"]).fillna(0)
    
    # Keep only columns the model was trained on
    # (the loaded pipeline expects exactly these columns in this order)
    available_cols = [c for c in feature_cols if c in X.columns]
    missing_cols   = [c for c in feature_cols if c not in X.columns]
    
    if missing_cols:
        # Fill missing columns with 0 rather than failing
        # This handles cases where sentiment columns are absent
        for col in missing_cols:
            X[col] = 0.0
    
    # Return only the last row, with columns in the exact training order
    X_latest = X[feature_cols].iloc[[-1]]
    
    return X_latest


# ══════════════════════════════════════════════════════════════════════════════
#  THE ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/",
    response_model=PredictionResponse,   # FastAPI validates the return value
    summary="Predict stock direction",
    description="""
    Predicts whether a stock will go UP or DOWN tomorrow.
    Returns the prediction with probability and SHAP-based explanation.
    
    The model generates a signal at today's CLOSE using today's features.
    The prediction applies to tomorrow's OPEN-to-CLOSE return.
    """,
)
def predict_stock(
    ticker: str = Query(
        ...,                              # ... = required, no default
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g. AAPL, MSFT, GOOGL)",
        example="AAPL",
    ),
    period: str = Query(
        default="2y",
        description="Data period for feature engineering",
        pattern="^[0-9]+[ymd]$",         # must match: 1y, 6mo, 90d etc.
        example="2y",
    ),
    threshold: float = Query(
        default=0.5,
        ge=0.1,                           # ge = greater than or equal
        le=0.9,                           # le = less than or equal
        description="Classification threshold for UP signal (0.1–0.9)",
    ),
    top_n: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of top SHAP features to include",
    ),
    # ── Dependency injection: injects app_state, raises 503 if model not loaded
    state: dict = Depends(require_model),
):
    """
    Main prediction endpoint.
    
    Called by the React frontend when a user views a stock page.
    The `state` parameter is injected by FastAPI via Depends(require_model).
    It is guaranteed to contain a loaded pipeline — require_model raises 503
    automatically if trainer.py has not been run.
    """
    # ── Normalise ticker ──────────────────────────────────────────────────
    # Always work with uppercase tickers — "aapl" → "AAPL"
    ticker = ticker.upper().strip()
    
    # ── Step 1: Fetch and engineer features ───────────────────────────────
    try:
        featured = _fetch_and_prepare(ticker, period=period)
    except ValueError as e:
        # Known error — bad ticker, no data, etc.
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Unexpected error — log and return 500
        raise HTTPException(
            status_code=500,
            detail=f"Data pipeline failed for '{ticker}': {str(e)}"
        )
    
    # ── Step 2: Extract latest features ──────────────────────────────────
    # ── Step 2: Extract pipeline and feature cols from injected state ────
    pipeline     = state["pipeline"]
    feature_cols = state["feature_cols"]
    
    try:
        X_latest = _get_latest_features(featured, feature_cols)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction failed: {str(e)}"
        )
    
    # ── Step 3: Get the prediction date ───────────────────────────────────
    # This is the date of the last row — today's data
    last_date = featured.index[-1]
    if hasattr(last_date, "date"):
        prediction_date = str(last_date.date())
    else:
        prediction_date = str(last_date)
    
    # ── Step 4: Run the model ─────────────────────────────────────────────
    try:
        # predict_proba returns [[P(DOWN), P(UP)]] for one sample
        # We take index [0][1] → probability of UP for the first (only) row
        proba_up   = float(pipeline.predict_proba(X_latest)[0][1])
        prediction = "UP" if proba_up >= threshold else "DOWN"
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(e)}"
        )
    
    # ── Step 5: Generate SHAP explanation ─────────────────────────────────
    try:
        from models.explainer import explain_single_prediction
        
        explanation = explain_single_prediction(
            pipeline=pipeline,
            X_single=X_latest,
            top_n=top_n,
            verbose=False,
        )
        
        # explanation dict from explainer.py contains:
        # {
        #   "prediction": "UP",
        #   "probability": 0.73,
        #   "top_features": [{"feature": ..., "shap_value": ..., ...}],
        #   "feature_groups": {"trend": 0.12, "momentum": 0.08, ...},
        #   "explanation_text": "Our AI model predicts..."
        # }
        
        top_features = [
            FeatureExplanation(**feat_dict)
            for feat_dict in explanation.get("top_features", [])
        ]
        feature_groups = explanation.get("feature_groups", {})
        explanation_text = explanation.get("explanation_text", "")
        
    except Exception as e:
        # SHAP failed — return prediction without explanation
        # This is better than failing the whole endpoint
        top_features     = []
        feature_groups   = {}
        explanation_text = (
            f"Our AI model predicts this stock will "
            f"{'go up' if prediction == 'UP' else 'go down'} "
            f"tomorrow with {proba_up*100:.0f}% confidence. "
            f"(Detailed explanation unavailable: {str(e)})"
        )
    
    # ── Step 6: Build and return response ────────────────────────────────
    # PredictionResponse is our Pydantic model — FastAPI validates it
    return PredictionResponse(
        ticker           = ticker,
        prediction_date  = prediction_date,
        prediction       = prediction,
        probability      = round(proba_up, 4),
        confidence_pct   = round(proba_up * 100, 1),
        top_features     = top_features,
        feature_groups   = feature_groups,
        explanation_text = explanation_text,
        n_features_used  = len(feature_cols),
        threshold_used   = threshold,
    )