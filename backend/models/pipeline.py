"""
StockSense AI — models/pipeline.py
=====================================
Full ML pipeline: from raw price data to a fitted XGBoost model.

Wires together the complete StockSense AI stack:
  data/assembler.py     → multi-stock feature matrix (X, y)
  data/labeller.py      → binary target labels with config registry
  features/engineer.py  → 14-step feature engineering (build_features)
  features/indicators.py → get_model_features — drops raw/non-stationary cols
  models/timeseries_cv.py → leak-free cross-validation

This file owns:
  - XGBoost pipeline construction (build_sklearn_pipeline)
  - Per-stock training (run_pipeline — single-stock quick path)
  - Multi-stock training (run_multi_pipeline — production path)
  - Evaluation with full classification report
  - Feature importance extraction
  - Live single-row prediction
  - Save / load of fitted pipelines

It does NOT own:
  - Feature engineering logic  → features/engineer.py
  - Label creation             → data/labeller.py
  - Data fetching              → data/fetch.py / assembler.py
  - CV splitting               → models/timeseries_cv.py

Usage
─────
    # Single stock quick test
    python -m backend.models.pipeline

    # Import API
    from backend.models.pipeline import run_pipeline, run_multi_pipeline
    results      = run_pipeline("AAPL", period="2y", horizon=1)
    multi_results = run_multi_pipeline(["AAPL", "MSFT", "GOOGL"])
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import yfinance as yf

from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Resolve imports whether run as a script or as a module ─────────────────
_HERE = Path(__file__).resolve().parent.parent   # backend/
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── StockSense AI internal imports ─────────────────────────────────────────
from data.cleaner import clean_stock_data
from data.labeller import (
    create_labels,
    check_label_quality,
    get_label_config,
    LABEL_CONFIGS,
)
from data.assembler import (
    assemble_stock,
    assemble_multiple_stocks,
    chronological_split,
    check_feature_consistency,
    get_assembly_config,
    ASSEMBLY_CONFIGS,
    MIN_ROWS,
)
from features.engineer import build_features
from features.indicators import get_model_features, RAW_COLUMNS_TO_DROP
from models.timeseries_cv import (
    cross_validate_timeseries,
    visualise_splits,
    get_cv_config,
    DEFAULT_GAP_DAYS,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"

# Columns that must never reach XGBoost — raw prices are non-stationary,
# target is the label, ticker is a string identifier.
# Mirrors _ASSEMBLER_INTERNAL_COLS in assembler.py.
_PIPELINE_DROP_COLS: List[str] = [
    "target", "ticker",
    "sentiment_mean", "sentiment_max", "sentiment_min",
    "sentiment_std", "sentiment_momentum",
    "article_count", "positive_ratio", "confidence_mean",
]


# ══════════════════════════════════════════════════════════════════════════════
#  1. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_sklearn_pipeline(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
) -> SklearnPipeline:
    """
    Build the sklearn Pipeline for StockSense AI.

    Steps
    ─────
    1. SimpleImputer  — fills any residual NaNs from rolling window boundaries.
                        Strategy 'mean' is safe for financial features (returns,
                        RSI, etc.) where mean imputation is a reasonable neutral.
    2. StandardScaler — centres to mean=0, std=1.
                        XGBoost is tree-based so scaling doesn't affect split
                        thresholds, but it prevents numerical overflow on
                        extreme values (e.g. OBV which can reach millions).
    3. XGBClassifier  — gradient-boosted trees predict UP(1) / DOWN(0).

    Hyperparameter notes
    ─────────────────────
    n_estimators=300     — enough trees for complex patterns; tune upward if
                           underfitting on large multi-stock datasets.
    learning_rate=0.05   — small steps generalise better; pair with early
                           stopping in a future trainer.py.
    max_depth=6          — controls complexity; 4–6 is typical for tabular data.
    subsample=0.8        — row subsampling reduces overfitting.
    colsample_bytree=0.8 — feature subsampling; further regularises the model.
    scale_pos_weight     — pass check_label_quality()['scale_pos_weight'] here
                           when labels are imbalanced (common at 0.3% threshold).

    Parameters
    ----------
    n_estimators     : Number of boosting rounds.
    learning_rate    : Step size shrinkage.
    max_depth        : Maximum tree depth.
    subsample        : Row subsampling ratio per tree.
    colsample_bytree : Column subsampling ratio per tree.
    scale_pos_weight : Weight ratio for positive (UP) class.
                       Pass check_label_quality()['scale_pos_weight'].
    random_state     : Reproducibility seed.

    Returns
    -------
    Unfitted sklearn Pipeline ready for .fit(X, y).
    """
    return SklearnPipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   XGBClassifier(
            n_estimators     = n_estimators,
            learning_rate    = learning_rate,
            max_depth        = max_depth,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            scale_pos_weight = scale_pos_weight,
            random_state     = random_state,
            eval_metric      = "logloss",
            verbosity        = 0,
        )),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  2. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    pipeline: SklearnPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a fitted pipeline on held-out test data.

    Parameters
    ----------
    pipeline : Fitted SklearnPipeline from build_sklearn_pipeline().
    X_test   : Feature DataFrame — iloc-compatible, same columns as X_train.
    y_test   : Binary target Series.
    verbose  : Print the full classification report.

    Returns
    -------
    dict with keys:
      accuracy              : float
      roc_auc               : float
      classification_report : str
      confusion_matrix      : np.ndarray (2×2)
      majority_baseline     : float  — always-majority-class accuracy baseline
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc      = float(accuracy_score(y_test, y_pred))
    auc      = float(roc_auc_score(y_test, y_prob))
    report   = classification_report(y_test, y_pred, target_names=["DOWN", "UP"])
    cm       = confusion_matrix(y_test, y_pred)
    baseline = float(max(y_test.mean(), 1 - y_test.mean()))

    if verbose:
        print(f"\n[EVAL] Accuracy        : {acc:.4f}  "
              f"(baseline: {baseline:.4f}  "
              f"{'✅ beats baseline' if acc > baseline else '⚠ below baseline'})")
        print(f"[EVAL] ROC-AUC         : {auc:.4f}")
        print(f"\nClassification Report:\n{report}")
        print(f"Confusion Matrix:\n{cm}")

    return {
        "accuracy":              acc,
        "roc_auc":               auc,
        "classification_report": report,
        "confusion_matrix":      cm,
        "majority_baseline":     baseline,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def feature_importance(
    pipeline: SklearnPipeline,
    feature_cols: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract and display feature importances from the XGBoost model.

    Parameters
    ----------
    pipeline     : Fitted SklearnPipeline.
    feature_cols : Column names in the same order they were passed to fit().
    top_n        : Number of top features to show.

    Returns
    -------
    pd.DataFrame with columns ['feature', 'importance'] sorted descending.
    """
    xgb_model   = pipeline.named_steps["model"]
    importances = xgb_model.feature_importances_

    fi_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print(f"\nTop {top_n} Feature Importances:")
    print(fi_df.to_string(index=False))

    return fi_df


# ══════════════════════════════════════════════════════════════════════════════
#  4. LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_latest(
    pipeline: SklearnPipeline,
    X: pd.DataFrame,
) -> Dict:
    """
    Run the fitted pipeline on the most recent row of engineered features.
    Used for live UP/DOWN predictions on new data.

    Parameters
    ----------
    pipeline : Fitted SklearnPipeline.
    X        : Full feature DataFrame (iloc-indexable).
               The LAST row is used — this should be today's most recent data.

    Returns
    -------
    dict with keys:
      date       : last date in the dataset
      prediction : 'UP' or 'DOWN'
      confidence : probability of the predicted class (0–1)
      prob_up    : raw probability of UP (class=1)
      prob_down  : raw probability of DOWN (class=0)

    Note
    ─────
    The pipeline includes an imputer and scaler — pass raw model-feature
    columns directly. Do NOT pre-scale before calling this function.
    """
    latest = X.iloc[[-1]]
    pred   = int(pipeline.predict(latest)[0])
    prob   = pipeline.predict_proba(latest)[0]

    # Extract date from either plain DatetimeIndex or (date, ticker) MultiIndex
    last_idx = X.index[-1]
    if isinstance(last_idx, tuple):
        last_date = last_idx[0]
    else:
        last_date = last_idx

    result = {
        "date":       last_date.date() if hasattr(last_date, "date") else last_date,
        "prediction": "UP" if pred == 1 else "DOWN",
        "confidence": float(prob[pred]),
        "prob_up":    float(prob[1]),
        "prob_down":  float(prob[0]),
    }

    print(f"\n[PREDICT] {result['date']} → {result['prediction']} "
          f"(confidence: {result['confidence']:.1%}  "
          f"P(UP)={result['prob_up']:.3f}  P(DOWN)={result['prob_down']:.3f})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  5. SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_pipeline(
    pipeline: SklearnPipeline,
    feature_cols: List[str],
    path: Optional[str] = None,
) -> str:
    """
    Save a fitted pipeline + feature column list to disk.

    Both are persisted together so the loader can verify column alignment.

    Parameters
    ----------
    pipeline     : Fitted SklearnPipeline.
    feature_cols : Ordered list of column names used during training.
    path         : File path (without extension). Default: data/models/pipeline.

    Returns
    -------
    str path to the saved .pkl file.
    """
    if path is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = str(MODELS_DIR / "pipeline.pkl")
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    payload = {"pipeline": pipeline, "feature_cols": feature_cols}
    joblib.dump(payload, path)
    print(f"[SAVE] Pipeline saved → {path}  "
          f"({len(feature_cols)} features)")
    return path


def load_pipeline(
    path: Optional[str] = None,
) -> Tuple[SklearnPipeline, List[str]]:
    """
    Load a previously saved pipeline from disk.

    Returns
    -------
    (pipeline, feature_cols) — the fitted pipeline and the feature column list
    it was trained on.
    """
    if path is None:
        path = str(MODELS_DIR / "pipeline.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pipeline file not found: {path}. "
            f"Run run_pipeline() with save=True first."
        )

    payload      = joblib.load(path)
    pipeline     = payload["pipeline"]
    feature_cols = payload["feature_cols"]

    print(f"[LOAD] Pipeline loaded ← {path}  "
          f"({len(feature_cols)} features)")
    return pipeline, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
#  6. SINGLE-STOCK PIPELINE (quick path)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    ticker: str = "AAPL",
    period: str = "2y",
    horizon: int = 1,
    threshold: float = 0.003,
    label_config: Optional[str] = None,
    train_ratio: float = 0.80,
    gap_days: int = DEFAULT_GAP_DAYS,
    run_cv: bool = False,
    cv_n_splits: int = 5,
    save: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    End-to-end single-stock pipeline:
    fetch → clean → features → labels → train → evaluate → predict.

    This is the quick development path. For production multi-stock training,
    use run_multi_pipeline() which calls assembler.assemble_multiple_stocks().

    Parameters
    ----------
    ticker        : Stock symbol (e.g. "AAPL").
    period        : yfinance period (e.g. "2y", "5y"). Minimum "1y"
                    required for SMA_200 in add_trend_features().
    horizon       : Prediction horizon in days (overridden by label_config).
    threshold     : Minimum return to label directional (overridden by label_config).
    label_config  : Name from LABEL_CONFIGS registry. Overrides horizon/threshold.
                    Use get_label_config() to see available configs.
    train_ratio   : Fraction of data used for training (rest = test).
    gap_days      : Trading days to skip between train end and test start.
                    Prevents rolling window contamination at the boundary.
                    Default: 20. Use 50 for SMA_200 features.
    run_cv        : Also run time-series cross-validation.
    cv_n_splits   : Number of CV folds (only used if run_cv=True).
    save          : Persist the fitted pipeline to disk.
    verbose       : Print progress at each step.

    Returns
    -------
    dict with keys:
      pipeline          : fitted SklearnPipeline
      feature_cols      : List[str] of feature column names
      metrics           : eval dict (accuracy, roc_auc, classification_report, ...)
      feature_importance: pd.DataFrame top features
      prediction        : live prediction dict for the last row
      cv_results        : cross-validation results dict (if run_cv=True, else None)
      label_quality     : check_label_quality() output dict
    """
    # ── Apply label config override ───────────────────────────────────────────
    if label_config is not None:
        cfg       = get_label_config(label_config)
        horizon   = cfg["horizon"]
        threshold = cfg["threshold"]

    print(f"\n{'═'*58}")
    print(f"  StockSense AI — {ticker}  |  "
          f"horizon={horizon}d  |  threshold={threshold*100:.2f}%")
    print(f"{'═'*58}")

    # ── Step 1: Fetch ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[1/7] Fetching {ticker} ({period})...")

    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker='{ticker}'")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    if verbose:
        print(f"[2/7] Cleaning...")
    clean = clean_stock_data(raw, ticker=ticker)

    if len(clean) < MIN_ROWS:
        raise ValueError(
            f"Insufficient data after cleaning: {len(clean)} rows "
            f"(need {MIN_ROWS}). Use a longer period."
        )

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    # Uses the full 14-step engineer.py pipeline (~342 features):
    # trend → momentum → MACD → volatility → volume → candle geometry →
    # patterns → pattern strength → interactions → sequences → S/R →
    # lags → returns → rolling sentiment (no-op without sentiment cols).
    if verbose:
        print(f"[3/7] Engineering features (14 steps)...")
    featured = build_features(clean).dropna()

    if verbose:
        print(f"       {len(featured)} rows × {len(featured.columns)} columns")

    # ── Step 4: Labels ────────────────────────────────────────────────────────
    if verbose:
        print(f"[4/7] Creating labels (horizon={horizon}d, "
              f"threshold={threshold*100:.2f}%)...")
    labelled    = create_labels(featured, horizon=horizon, threshold=threshold,
                                verbose=verbose)
    lq          = check_label_quality(labelled, verbose=verbose)

    # ── Step 5: Model-ready X ─────────────────────────────────────────────────
    # get_model_features() drops raw OHLCV and non-stationary intermediate cols.
    # _PIPELINE_DROP_COLS additionally drops target and ticker.
    X = get_model_features(labelled, extra_drop=_PIPELINE_DROP_COLS).fillna(0)
    y = labelled["target"].astype(int)
    feature_cols = list(X.columns)

    if verbose:
        print(f"\n[5/7] Feature matrix: "
              f"{X.shape[0]} rows × {X.shape[1]} features")

    # ── Step 6: Chronological train/test split with gap ───────────────────────
    # Use unique dates (single stock has DatetimeIndex, not MultiIndex).
    # Replicate the assembler logic: split at train_ratio, then apply gap.
    dates     = pd.DatetimeIndex(X.index).normalize().unique().sort_values()
    n_dates   = len(dates)
    split_pos = int(n_dates * train_ratio)

    if split_pos <= gap_days:
        raise ValueError(
            f"Not enough dates ({n_dates}) for train_ratio={train_ratio} "
            f"with gap_days={gap_days}."
        )

    train_end  = dates[split_pos - gap_days - 1]
    test_start = dates[split_pos]

    row_dates  = pd.DatetimeIndex(X.index).normalize()
    X_train    = X[row_dates <= train_end]
    X_test     = X[row_dates >= test_start]
    y_train    = y[row_dates <= train_end]
    y_test     = y[row_dates >= test_start]

    if verbose:
        print(f"       Train : up to {train_end.date()} "
              f"({len(X_train):,} rows)")
        print(f"       Gap   : {gap_days} trading days")
        print(f"       Test  : from {test_start.date()} "
              f"({len(X_test):,} rows)")

    # ── Step 7: Build and fit pipeline ────────────────────────────────────────
    if verbose:
        print(f"\n[6/7] Training XGBoost pipeline...")

    scale_pw  = lq.get("scale_pos_weight", 1.0)
    skl_pipe  = build_sklearn_pipeline(scale_pos_weight=scale_pw)
    skl_pipe.fit(X_train, y_train)

    print(f"       Fitted on {len(X_train):,} samples, "
          f"{len(feature_cols)} features  "
          f"(scale_pos_weight={scale_pw:.3f})")

    # ── Step 8: Evaluate ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n[7/7] Evaluating on held-out test set...")
    metrics = evaluate(skl_pipe, X_test, y_test, verbose=verbose)

    # ── Feature importance ────────────────────────────────────────────────────
    fi_df      = feature_importance(skl_pipe, feature_cols, top_n=20)

    # ── Live prediction ───────────────────────────────────────────────────────
    prediction = predict_latest(skl_pipe, X)

    # ── Optional: cross-validation ────────────────────────────────────────────
    cv_results = None
    if run_cv:
        # Auto-cap folds: 5 folds needs ~300+ unique dates; use 3 for shorter sets.
        n_unique = len(pd.DatetimeIndex(X.index).normalize().unique())
        safe_n_splits = cv_n_splits if n_unique >= cv_n_splits * 60 else max(2, n_unique // 60)
        print(f"\n── Cross-Validation  (n_splits={safe_n_splits}) ──────────────")
        cv_results = cross_validate_timeseries(
            model    = build_sklearn_pipeline(scale_pos_weight=scale_pw),
            X        = X,
            y        = y,
            n_splits = safe_n_splits,
            gap_days = gap_days,
            verbose  = True,
        )

    # ── Optional: save ───────────────────────────────────────────────────────
    if save:
        save_pipeline(skl_pipe, feature_cols)

    print(f"\n{'═'*58}")
    print(f"  Done.  Accuracy={metrics['accuracy']:.4f}  "
          f"AUC={metrics['roc_auc']:.4f}  "
          f"Baseline={metrics['majority_baseline']:.4f}")
    print(f"{'═'*58}\n")

    return {
        "pipeline":           skl_pipe,
        "feature_cols":       feature_cols,
        "metrics":            metrics,
        "feature_importance": fi_df,
        "prediction":         prediction,
        "cv_results":         cv_results,
        "label_quality":      lq,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  7. MULTI-STOCK PIPELINE (production path)
# ══════════════════════════════════════════════════════════════════════════════

def run_multi_pipeline(
    tickers: List[str],
    period: str = "2y",
    assembly_config: str = "default",
    cv_config: str = "default",
    run_cv: bool = True,
    save: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Production multi-stock pipeline:
    assemble_multiple_stocks → train → evaluate → predict.

    Delegates data assembly, labelling, and feature engineering entirely to
    assembler.assemble_multiple_stocks() — which handles sentiment, MultiIndex,
    feature consistency, and per-stock error recovery.

    The resulting (X, y) with (date, ticker) MultiIndex is passed directly to
    the sklearn pipeline and timeseries_cv — both handle MultiIndex correctly.

    Parameters
    ----------
    tickers         : List of stock symbols.
    period          : yfinance period for each stock (default: "2y").
    assembly_config : Name from ASSEMBLY_CONFIGS registry.
                      Controls train_ratio, gap_days, horizon, threshold.
                      Use list_assembly_configs() to see all options.
    cv_config       : Name from CV_CONFIGS registry.
                      Controls n_splits and gap_days for cross-validation.
    run_cv          : Run time-series cross-validation after training.
    save            : Persist the fitted pipeline to disk.
    verbose         : Print per-stock and summary progress.

    Returns
    -------
    dict with keys:
      pipeline          : fitted SklearnPipeline
      feature_cols      : List[str]
      X_train / X_test  : split feature DataFrames
      y_train / y_test  : split target Series
      metrics           : eval dict
      feature_importance: top-N feature DataFrame
      cv_results        : CV results (if run_cv=True, else None)
      stock_metadata    : List of per-stock metadata dicts from assembler
    """
    # ── Resolve configs ───────────────────────────────────────────────────────
    asm_cfg = get_assembly_config(assembly_config)
    cv_cfg  = get_cv_config(cv_config)

    print(f"\n{'═'*58}")
    print(f"  StockSense AI — Multi-Stock Pipeline")
    print(f"  Stocks  : {len(tickers)}  |  "
          f"Assembly: {assembly_config}  |  CV: {cv_config}")
    print(f"{'═'*58}")

    # ── Step 1: Assemble all stocks ───────────────────────────────────────────
    # assembler.py handles: fetch → clean → sentiment → engineer → label →
    # get_model_features → MultiIndex.
    # Each stock that fails is skipped with a warning; the run continues.
    if verbose:
        print(f"\n[1/4] Assembling {len(tickers)} stocks...")

    X, y, stock_metadata = assemble_multiple_stocks(
        tickers     = tickers,
        period      = period,
        config_name = assembly_config,
        verbose     = verbose,
    )
    feature_cols = list(X.columns)

    # ── Step 2: Chronological split with gap ──────────────────────────────────
    if verbose:
        print(f"\n[2/4] Splitting (ratio={asm_cfg['train_ratio']:.0%}, "
              f"gap={asm_cfg['gap_days']}d)...")

    X_train, X_test, y_train, y_test = chronological_split(
        X, y, config_name=assembly_config
    )
    check_feature_consistency(X_train, X_test, verbose=verbose)

    lq = check_label_quality(
        pd.DataFrame({"target": y_train}), verbose=verbose
    )

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[3/4] Training XGBoost pipeline...")

    scale_pw  = lq.get("scale_pos_weight", 1.0)
    skl_pipe  = build_sklearn_pipeline(scale_pos_weight=scale_pw)
    skl_pipe.fit(X_train, y_train)

    print(f"       Fitted on {len(X_train):,} rows × "
          f"{len(feature_cols)} features  "
          f"(scale_pos_weight={scale_pw:.3f})")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n[4/4] Evaluating on test set...")

    metrics = evaluate(skl_pipe, X_test, y_test, verbose=verbose)
    fi_df   = feature_importance(skl_pipe, feature_cols, top_n=20)

    # ── Optional: cross-validation ────────────────────────────────────────────
    cv_results = None
    if run_cv:
        print(f"\n── Cross-Validation (config='{cv_config}') ──────────────")
        visualise_splits(X, **{k: v for k, v in cv_cfg.items()
                               if k != "description"})
        cv_results = cross_validate_timeseries(
            model    = build_sklearn_pipeline(scale_pos_weight=scale_pw),
            X        = X,
            y        = y,
            **{k: v for k, v in cv_cfg.items() if k != "description"},
            verbose  = True,
        )

    # ── Optional: save ────────────────────────────────────────────────────────
    if save:
        save_pipeline(skl_pipe, feature_cols)

    print(f"\n{'═'*58}")
    print(f"  Multi-Stock Done.  "
          f"Accuracy={metrics['accuracy']:.4f}  "
          f"AUC={metrics['roc_auc']:.4f}  "
          f"Baseline={metrics['majority_baseline']:.4f}")
    print(f"{'═'*58}\n")

    return {
        "pipeline":           skl_pipe,
        "feature_cols":       feature_cols,
        "X_train":            X_train,
        "X_test":             X_test,
        "y_train":            y_train,
        "y_test":             y_test,
        "metrics":            metrics,
        "feature_importance": fi_df,
        "cv_results":         cv_results,
        "stock_metadata":     stock_metadata,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick single-stock smoke test.
    # period="2y" is required (SMA_200 in add_trend_features needs 200+ rows).
    results = run_pipeline(
        ticker       = "AAPL",
        period       = "2y",
        label_config = "default",     # horizon=1, threshold=0.3%
        train_ratio  = 0.80,
        gap_days     = 20,
        run_cv       = True,
        cv_n_splits  = 5,
        save         = False,
    )

    print("\nPipeline result keys:", list(results.keys()))
    print(f"Feature count : {len(results['feature_cols'])}")
    print(f"Prediction    : {results['prediction']}")
