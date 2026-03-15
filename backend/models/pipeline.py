"""
StockSense AI — Full ML Pipeline
=================================
Wires together:
  fetch.py      → raw OHLCV data
  cleaner.py    → removes bad rows / fills gaps
  timeutils.py  → returns, lags, target label, train/test split
  pipeline.py   → XGBoost model inside sklearn Pipeline

Usage:
    python -m backend.models.pipeline          # quick test on AAPL
    from backend.models.pipeline import run_pipeline
    results = run_pipeline("MSFT", horizon=5)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

# ── resolve imports whether run as a script or as a module ──
_HERE = Path(__file__).resolve().parent.parent  # backend/
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data.fetch import fetch_stock_data
from data.cleaner import clean_stock_data
from data.timeutils import (
    validate_timeseries,
    add_returns,
    add_lag_features,
    create_target_label,
    chronological_split,
)

# ─── Constants ────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
FEATURE_COLS: List[str] = []   # populated at runtime by build_features()


# ─── 1. Model Architecture ────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Build the sklearn Pipeline for StockSense AI.

    Steps:
      1. SimpleImputer  — fills any residual NaNs left after feature engineering
                          (rolling windows produce NaNs for the first N rows)
      2. StandardScaler — centres features to mean=0, std=1
                          XGBoost is tree-based so scaling doesn't change splits,
                          but it prevents numerical issues and speeds up convergence
      3. XGBClassifier  — gradient-boosted trees predict UP(1) / DOWN(0)

    Hyperparameter notes:
      n_estimators=300   — enough trees to learn complex patterns without overfitting
      learning_rate=0.05 — small steps → better generalisation
      max_depth=6        — controls tree complexity; 4–6 is typical for tabular data
      subsample=0.8      — row subsampling reduces overfitting
      colsample_bytree=0.8 — feature subsampling (like random forests)
      use_label_encoder removed in newer XGBoost → use eval_metric directly
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        ))
    ])


# ─── 2. Feature Engineering ───────────────────────────────────
def build_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 5, 10],
    horizon: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply the full feature engineering chain to a clean OHLCV DataFrame.

    Pipeline:
      add_returns()       → stationary price change features
      add_lag_features()  → give the model memory of past N days
      create_target_label() → binary UP/DOWN label `horizon` days ahead
      dropna()            → remove NaN rows created by rolling/shift ops

    Returns:
        (featured_df, feature_column_names)
    """
    df = add_returns(df)
    df = add_lag_features(df, lags=lags)
    df = create_target_label(df, horizon=horizon)

    # Drop NaN rows introduced by rolling windows / lagging
    df = df.dropna()

    # Feature columns = everything except the target and raw OHLCV identifiers
    exclude = {'open', 'high', 'low', 'close', 'volume', 'target', 'ticker'}
    feature_cols = [c for c in df.columns if c not in exclude]

    return df, feature_cols


# ─── 3. Training ──────────────────────────────────────────────
def train(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_ratio: float = 0.8,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split data and fit the pipeline on the training portion.

    Returns:
        (fitted_pipeline, train_df, test_df)
    """
    train_df, test_df = chronological_split(df, train_ratio=train_ratio)

    X_train = train_df[feature_cols]
    y_train = train_df['target']

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print(f"\n[TRAIN] Fitted on {len(X_train)} samples, "
          f"{len(feature_cols)} features")
    return pipeline, train_df, test_df


# ─── 4. Evaluation ────────────────────────────────────────────
def evaluate(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict:
    """
    Evaluate the fitted pipeline on held-out test data.

    Returns a dict with accuracy, ROC-AUC, and the full classification report.
    """
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=['DOWN', 'UP'])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n[EVAL] Accuracy : {acc:.4f}")
    print(f"[EVAL] ROC-AUC  : {auc:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        'accuracy': acc,
        'roc_auc': auc,
        'classification_report': report,
        'confusion_matrix': cm,
    }


# ─── 5. Feature Importance ────────────────────────────────────
def feature_importance(
    pipeline: Pipeline,
    feature_cols: List[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extract and display feature importances from the XGBoost model.
    """
    xgb_model = pipeline.named_steps['model']
    importances = xgb_model.feature_importances_

    fi_df = (
        pd.DataFrame({'feature': feature_cols, 'importance': importances})
        .sort_values('importance', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print(f"\nTop {top_n} Feature Importances:")
    print(fi_df.to_string(index=False))

    return fi_df


# ─── 6. Prediction (live) ─────────────────────────────────────
def predict_latest(
    pipeline: Pipeline,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict:
    """
    Run the pipeline on the most recent row of engineered features.
    Used for live UP/DOWN predictions on new data.

    Returns:
        {
          'date'       : last date in the dataset,
          'prediction' : 'UP' or 'DOWN',
          'confidence' : probability of the predicted class (0–1),
          'prob_up'    : raw probability of UP,
          'prob_down'  : raw probability of DOWN,
        }
    """
    latest = df[feature_cols].iloc[[-1]]
    pred = pipeline.predict(latest)[0]
    prob = pipeline.predict_proba(latest)[0]

    result = {
        'date': df.index[-1].date(),
        'prediction': 'UP' if pred == 1 else 'DOWN',
        'confidence': float(prob[pred]),
        'prob_up': float(prob[1]),
        'prob_down': float(prob[0]),
    }

    print(f"\n[PREDICT] {result['date']} → {result['prediction']} "
          f"(confidence: {result['confidence']:.1%})")
    return result


# ─── 7. Save / Load ───────────────────────────────────────────
def save_pipeline(pipeline: Pipeline, path: Optional[str] = None) -> str:
    """Save a fitted pipeline to disk as a .pkl file."""
    if path is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = str(MODELS_DIR / "pipeline.pkl")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    joblib.dump(pipeline, path)
    print(f"[SAVE] Pipeline saved → {path}")
    return path


def load_pipeline(path: Optional[str] = None) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    if path is None:
        path = str(MODELS_DIR / "pipeline.pkl")

    pipeline = joblib.load(path)
    print(f"[LOAD] Pipeline loaded ← {path}")
    return pipeline


# ─── 8. Full End-to-End Runner ────────────────────────────────
def run_pipeline(
    ticker: str = "AAPL",
    period: str = "2y",
    horizon: int = 1,
    lags: List[int] = [1, 2, 3, 5, 10],
    train_ratio: float = 0.8,
    save: bool = False,
) -> Dict:
    """
    End-to-end pipeline: fetch → clean → validate → features → train → evaluate → predict.

    Args:
        ticker        : Stock symbol (e.g. "AAPL")
        period        : yfinance period string (e.g. "2y", "5y")
        horizon       : Days ahead to predict (1=tomorrow, 5=next week)
        lags          : Past-day lag windows to include as features
        train_ratio   : Fraction of data used for training (rest = test)
        save          : Whether to persist the fitted pipeline to disk

    Returns:
        dict with 'pipeline', 'metrics', 'feature_importance', 'prediction', 'feature_cols'
    """
    print(f"\n{'='*55}")
    print(f"  StockSense AI — {ticker}  |  horizon={horizon}d")
    print(f"{'='*55}")

    # ── Step 1: Fetch ─────────────────────────────────────────
    print(f"\n[1/6] Fetching {ticker} data ({period})...")
    raw_df = fetch_stock_data(ticker, period=period, use_cache=True)
    if raw_df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # ── Step 2: Clean ─────────────────────────────────────────
    print(f"\n[2/6] Cleaning data...")
    clean_df = clean_stock_data(raw_df, ticker=ticker)

    # ── Step 3: Validate ──────────────────────────────────────
    print(f"\n[3/6] Validating time-series...")
    if not validate_timeseries(clean_df):
        print("[WARN] Validation issues found — continuing anyway")

    # ── Step 4: Feature Engineering ───────────────────────────
    print(f"\n[4/6] Engineering features (horizon={horizon}d, lags={lags})...")
    featured_df, feature_cols = build_features(clean_df, lags=lags, horizon=horizon)
    print(f"       {len(feature_cols)} features, {len(featured_df)} usable rows")

    # ── Step 5: Train ─────────────────────────────────────────
    print(f"\n[5/6] Training XGBoost pipeline...")
    fitted_pipeline, train_df, test_df = train(
        featured_df, feature_cols, train_ratio=train_ratio
    )

    # ── Step 6: Evaluate ──────────────────────────────────────
    print(f"\n[6/6] Evaluating on held-out test set...")
    metrics = evaluate(fitted_pipeline, test_df, feature_cols)

    # ── Extras: feature importance + live prediction ──────────
    fi_df = feature_importance(fitted_pipeline, feature_cols)
    prediction = predict_latest(fitted_pipeline, featured_df, feature_cols)

    # ── Optional: save pipeline ───────────────────────────────
    if save:
        save_pipeline(fitted_pipeline)

    print(f"\n{'='*55}")
    print(f"  Done.  Accuracy={metrics['accuracy']:.4f}  "
          f"AUC={metrics['roc_auc']:.4f}")
    print(f"{'='*55}\n")

    return {
        'pipeline': fitted_pipeline,
        'metrics': metrics,
        'feature_importance': fi_df,
        'prediction': prediction,
        'feature_cols': feature_cols,
    }


# ─── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    results = run_pipeline(
        ticker="AAPL",
        period="2y",
        horizon=1,
        save=False,
    )
