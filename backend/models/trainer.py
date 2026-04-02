"""
StockSense AI — models/trainer.py
===================================
XGBoost model training pipeline.

This file owns:
  - Building the sklearn Pipeline (imputer + scaler + XGBoost)
  - Three-way chronological split (train/val/test)
  - Training with early stopping
  - Saving and loading trained pipelines
  - Training summary reporting

It does NOT own:
  - Cross-validation logic     → models/timeseries_cv.py
  - Hyperparameter tuning      → models/tuner.py  (Chapter 4.6)
  - Evaluation metrics         → models/evaluator.py (Chapter 4.4)
  - SHAP explainability        → models/explainer.py (Chapter 4.5)

Why sklearn Pipeline wraps XGBoost?
─────────────────────────────────────────────────────────────
  Three reasons:
    1. Imputer + Scaler are fitted on training data only and
       automatically applied to any new data — prevents leakage.
    2. The entire pipeline (preprocessing + model) is saved as
       one object — no risk of forgetting to apply preprocessing
       during inference.
    3. FastAPI loads one pipeline object and calls .predict() —
       no separate preprocessing step to maintain.

Why three-way split instead of two-way + CV?
─────────────────────────────────────────────────────────────
  Cross-validation (timeseries_cv.py) measures generalisation.
  The three-way split here serves a different purpose:
    Train (70%):   model learns features
    Val   (15%):   early stopping monitors this — stops overfitting
    Test  (15%):   never touched during training, final honest evaluation
  Val is separate from test so early stopping cannot overfit to test.
  Using test for early stopping would make test score optimistic.

Why StandardScaler if XGBoost doesn't need it?
─────────────────────────────────────────────────────────────
  XGBoost trees are scale-invariant. But:
    1. SHAP values are easier to interpret on scaled features
    2. When we add an LSTM layer later, scaling is mandatory
    3. Consistent preprocessing across all model types
  The overhead is negligible.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, Tuple
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = "models/saved"

# Default XGBoost parameters — starting point before tuning.
# These are deliberately conservative to avoid overfitting on first run.
# Chapter 4.6 (Optuna) will find better values.
DEFAULT_XGB_PARAMS = {
    "n_estimators":        500,
    "learning_rate":       0.05,
    "max_depth":           5,
    "subsample":           0.8,
    "colsample_bytree":    0.7,
    "min_child_weight":    5,
    "gamma":               0.1,
    "reg_alpha":           0.1,    # L1 regularisation
    "reg_lambda":          1.0,    # L2 regularisation
    "eval_metric":         "logloss",
    "early_stopping_rounds": 50,
    "random_state":        42,
    "verbosity":           0,
    "n_jobs":             -1,      # use all CPU cores
}

# Three-way split ratios.
# Train:Val:Test = 70:15:15
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (implicit: whatever remains)

# Gap between splits to prevent rolling window overlap.
GAP_DAYS = 20


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate XGBoost scale_pos_weight from training labels.

    Why training labels only?
    ─────────────────────────────────────────────────────────────
    Using the full dataset would leak test label distribution into
    the training configuration. Only the training set class ratio
    should influence how the model weights samples.

    Formula: n_negative / n_positive
    (tells XGBoost to weight UP examples by this factor)
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    if n_pos == 0:
        return 1.0
    spw = n_neg / n_pos
    return round(float(spw), 4)


def _three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    gap_days:    int   = GAP_DAYS,
) -> Tuple:
    """
    Split into train/val/test chronologically with gaps.

    Works with both plain DatetimeIndex and (date, ticker) MultiIndex.
    Uses unique calendar dates for splitting — correct for multi-stock.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    (all as DataFrames/Series with original index preserved)
    """
    # Get date series regardless of index type
    if isinstance(X.index, pd.MultiIndex):
        dates = pd.DatetimeIndex(
            X.index.get_level_values("date")
        ).normalize()
    else:
        dates = pd.DatetimeIndex(X.index).normalize()

    unique_dates = dates.unique().sort_values()
    n_dates      = len(unique_dates)

    # Calculate split positions in unique_dates space
    train_end_pos = int(n_dates * train_ratio)
    val_end_pos   = int(n_dates * (train_ratio + val_ratio))

    # Apply gaps
    train_end_date = unique_dates[train_end_pos - gap_days - 1]
    val_start_date = unique_dates[train_end_pos]
    val_end_date   = unique_dates[val_end_pos - gap_days - 1]
    test_start_date = unique_dates[val_end_pos]

    # Create boolean masks
    train_mask = dates <= train_end_date
    val_mask   = (dates >= val_start_date) & (dates <= val_end_date)
    test_mask  = dates >= test_start_date

    X_train = X[train_mask]
    X_val   = X[val_mask]
    X_test  = X[test_mask]
    y_train = y[train_mask]
    y_val   = y[val_mask]
    y_test  = y[test_mask]

    return X_train, X_val, X_test, y_train, y_val, y_test


def _reset_for_xgboost(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Reset MultiIndex to plain integer index for XGBoost.

    Why?
    ─────────────────────────────────────────────────────────────
    XGBoost's DMatrix does not understand pandas MultiIndex.
    reset_index(drop=True) converts (date, ticker) → 0, 1, 2, ...
    The original index is preserved in X/y before this call
    for date-range reporting — this is only called at fit time.
    """
    return X.reset_index(drop=True), y.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    xgb_params: Optional[Dict] = None,
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """
    Build the sklearn Pipeline: Imputer → Scaler → XGBoost.

    Parameters
    ----------
    xgb_params        : XGBoost parameters dict. Defaults to DEFAULT_XGB_PARAMS.
    scale_pos_weight  : Class imbalance correction. Calculate with
                        _calculate_scale_pos_weight(y_train).

    Returns
    -------
    Unfitted sklearn Pipeline ready for .fit().

    Why this function rather than inline Pipeline construction?
    ─────────────────────────────────────────────────────────────
    Centralising pipeline construction means tuner.py, trainer.py,
    and any future experiment scripts all create the exact same
    architecture. A change here propagates everywhere automatically.
    """
    params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        params.update(xgb_params)

    params["scale_pos_weight"] = scale_pos_weight

    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   XGBClassifier(**params)),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(
    X: pd.DataFrame,
    y: pd.Series,
    xgb_params: Optional[Dict] = None,
    verbose: bool = True,
) -> Tuple[Pipeline, Dict]:
    """
    Full training pipeline: split → build → fit → evaluate.

    Parameters
    ----------
    X          : Feature DataFrame from assembler.py.
                 May have (date, ticker) MultiIndex.
    y          : Target Series with same index.
    xgb_params : Override DEFAULT_XGB_PARAMS. Used by tuner.py
                 to pass Optuna-optimised parameters.
    verbose    : Print training progress and results.

    Returns
    -------
    (pipeline, results) tuple where:
      pipeline : Fitted sklearn Pipeline ready for prediction.
      results  : Dict with split info, training metrics, model metadata.

    Why return both pipeline and results dict?
    ─────────────────────────────────────────────────────────────
    The pipeline is what gets saved and deployed.
    The results dict is what gets logged and compared across runs.
    Keeping them separate makes it easy to log results without
    needing to load the full pipeline object.
    """
    if verbose:
        print(f"\n{'═'*60}")
        print(f"StockSense AI — Model Training")
        print(f"{'═'*60}")
        print(f"Dataset: {len(X):,} rows × {len(X.columns)} features")

    # ── Step 1: Three-way chronological split ─────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = _three_way_split(X, y)

    if verbose:
        if isinstance(X_train.index, pd.MultiIndex):
            tr_dates = X_train.index.get_level_values("date")
            va_dates = X_val.index.get_level_values("date")
            te_dates = X_test.index.get_level_values("date")
        else:
            tr_dates = X_train.index
            va_dates = X_val.index
            te_dates = X_test.index

        print(f"\nChronological split:")
        print(f"  Train: {pd.Timestamp(tr_dates.min()).date()} → "
              f"{pd.Timestamp(tr_dates.max()).date()} "
              f"({len(X_train):,} rows)")
        print(f"  Val:   {pd.Timestamp(va_dates.min()).date()} → "
              f"{pd.Timestamp(va_dates.max()).date()} "
              f"({len(X_val):,} rows)")
        print(f"  Test:  {pd.Timestamp(te_dates.min()).date()} → "
              f"{pd.Timestamp(te_dates.max()).date()} "
              f"({len(X_test):,} rows)")

    # ── Step 2: Calculate scale_pos_weight from training labels ───────────
    spw = _calculate_scale_pos_weight(y_train)
    if verbose:
        print(f"\nClass balance (train):")
        print(f"  UP:   {(y_train==1).sum():,} ({(y_train==1).mean()*100:.1f}%)")
        print(f"  DOWN: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)")
        print(f"  scale_pos_weight: {spw:.4f}")

    # ── Step 3: Build pipeline ────────────────────────────────────────────
    pipeline = build_pipeline(xgb_params=xgb_params, scale_pos_weight=spw)

    # ── Step 4: Prepare validation set for early stopping ─────────────────
    # Must transform through preprocessor (imputer + scaler) only
    # Fitted on X_train — never on X_val
    preprocessor = Pipeline(pipeline.steps[:-1])
    preprocessor.fit(X_train.reset_index(drop=True))
    X_val_t = preprocessor.transform(X_val.reset_index(drop=True))

    # ── Step 5: Fit ───────────────────────────────────────────────────────
    if verbose:
        print(f"\nTraining XGBoost...")

    X_tr, y_tr = _reset_for_xgboost(X_train, y_train)
    _, y_va    = _reset_for_xgboost(X_val, y_val)

    pipeline.fit(
        X_tr, y_tr,
        model__eval_set=[(X_val_t, y_va.values)],
        model__verbose=False,
    )

    model     = pipeline.named_steps["model"]
    best_iter = model.best_iteration

    if verbose:
        print(f"  Early stopping: best iteration = {best_iter} / "
              f"{model.n_estimators}")
        print(f"  Best val logloss: {model.best_score:.4f}")

    # ── Step 6: Evaluate on all splits ───────────────────────────────────
    results = _evaluate_splits(
        pipeline, X_train, X_val, X_test,
        y_train, y_val, y_test, verbose
    )

    results["best_iteration"]    = best_iter
    results["best_val_logloss"]  = model.best_score
    results["scale_pos_weight"]  = spw
    results["n_features"]        = len(X.columns)
    results["n_train_rows"]      = len(X_train)
    results["trained_at"]        = datetime.now().isoformat()

    return pipeline, results


def _evaluate_splits(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_val:   pd.Series,
    y_test:  pd.Series,
    verbose: bool,
) -> Dict:
    """
    Evaluate the fitted pipeline on all three splits.
    Returns a results dict with accuracy, F1, AUC-ROC for each split.
    """
    splits = {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }

    results   = {}
    majority  = max((y_train==1).mean(), (y_train==0).mean())

    if verbose:
        print(f"\n{'─'*60}")
        print(f"{'Split':<8} {'Accuracy':>10} {'F1':>8} "
              f"{'Precision':>10} {'Recall':>9} {'AUC-ROC':>9}")
        print(f"{'─'*60}")
        print(f"{'Baseline':<8} {majority*100:>9.1f}%  "
              f"{'(majority class)':>35}")
        print(f"{'─'*60}")

    for split_name, (X_sp, y_sp) in splits.items():
        X_r, y_r = _reset_for_xgboost(X_sp, y_sp)
        y_pred   = pipeline.predict(X_r)
        y_proba  = pipeline.predict_proba(X_r)[:, 1]

        acc  = accuracy_score(y_r, y_pred)
        f1   = f1_score(y_r, y_pred, zero_division=0)
        prec = precision_score(y_r, y_pred, zero_division=0)
        rec  = recall_score(y_r, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_r, y_proba)
        except ValueError:
            auc = 0.5

        results[f"{split_name}_accuracy"]  = round(acc,  4)
        results[f"{split_name}_f1"]        = round(f1,   4)
        results[f"{split_name}_precision"] = round(prec, 4)
        results[f"{split_name}_recall"]    = round(rec,  4)
        results[f"{split_name}_auc_roc"]   = round(auc,  4)

        if verbose:
            beats = "✅" if acc > majority else "❌"
            print(f"{split_name:<8} {acc*100:>9.1f}%{beats} "
                  f"{f1:>7.4f}  {prec:>9.4f}  {rec:>8.4f}  {auc:>8.4f}")

    if verbose:
        print(f"{'─'*60}")
        gap = results["test_accuracy"] - majority
        print(f"\nTest accuracy above majority baseline: "
              f"{gap*100:+.1f}pp")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_pipeline(
    pipeline: Pipeline,
    results:  Dict,
    name:     str = "stocksense_model",
) -> str:
    """
    Save fitted pipeline and results to disk.

    Saves both the pipeline (for deployment) and the results dict
    (for experiment tracking). Includes timestamp in filename to
    preserve multiple training runs.

    Returns
    -------
    Path to saved pipeline file.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pipeline_path = f"{MODEL_DIR}/{name}_{timestamp}.pkl"
    results_path  = f"{MODEL_DIR}/{name}_{timestamp}_results.joblib"
    latest_path   = f"{MODEL_DIR}/{name}_latest.pkl"

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(results,  results_path)
    joblib.dump(pipeline, latest_path)   # always overwrite latest

    print(f"\nPipeline saved:")
    print(f"  Versioned: {pipeline_path}")
    print(f"  Latest:    {latest_path}")
    print(f"  Results:   {results_path}")

    return pipeline_path


def load_pipeline(name: str = "stocksense_model") -> Pipeline:
    """
    Load the most recently saved pipeline.
    Uses the '_latest.pkl' shortcut file.
    """
    latest_path = f"{MODEL_DIR}/{name}_latest.pkl"

    if not os.path.exists(latest_path):
        raise FileNotFoundError(
            f"No saved pipeline found at {latest_path}. "
            f"Run train() and save_pipeline() first."
        )

    pipeline = joblib.load(latest_path)
    model    = pipeline.named_steps["model"]
    print(f"Pipeline loaded: {model.best_iteration} trees, "
          f"val logloss {model.best_score:.4f}")
    return pipeline


def predict(
    pipeline: Pipeline,
    X:        pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions from a fitted pipeline.

    Parameters
    ----------
    pipeline : Fitted pipeline from train() or load_pipeline().
    X        : Feature DataFrame. Must have same columns as training data.

    Returns
    -------
    (predictions, probabilities) tuple
      predictions  : np.ndarray of 0/1 labels
      probabilities: np.ndarray of UP probability (class 1)
    """
    X_r     = X.reset_index(drop=True)
    preds   = pipeline.predict(X_r)
    probas  = pipeline.predict_proba(X_r)[:, 1]
    return preds, probas


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    import yfinance as yf
    from data.cleaner import clean_stock_data
    from features.engineer import build_features
    from features.indicators import get_model_features
    from data.labeller import create_labels

    # ── Fetch and prepare
    print("Fetching data...")
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
    all_X, all_y = [], []

    for ticker in tickers:
        raw = yf.download(ticker, period="2y",
                          auto_adjust=True, progress=False)
        if raw.empty:
            continue
        raw.columns = [c.lower() for c in raw.columns]
        clean    = clean_stock_data(raw, ticker=ticker)
        featured = build_features(clean).dropna()
        labelled = create_labels(featured, horizon=1,
                                 threshold=0.003, verbose=False)
        X = get_model_features(labelled, extra_drop=["target"]).fillna(0)
        y = labelled["target"]
        all_X.append(X)
        all_y.append(y)

    X_combined = pd.concat(all_X).sort_index()
    y_combined = pd.concat(all_y).reindex(X_combined.index)

    print(f"Combined: {len(X_combined):,} rows × "
          f"{len(X_combined.columns)} features")

    # ── Train
    pipeline, results = train(X_combined, y_combined, verbose=True)

    # ── Save
    save_pipeline(pipeline, results)

    # ── Quick prediction test
    sample   = X_combined.tail(5)
    preds, probas = predict(pipeline, sample)
    print(f"\nSample predictions:")
    for pred, prob in zip(preds, probas):
        direction = "UP" if pred == 1 else "DOWN"
        print(f"  {direction} ({prob*100:.1f}% confidence)")


# ══════════════════════════════════════════════════════════════════════════════
#  IMBALANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_imbalance(
    y_train: pd.Series,
    y_val:   pd.Series,
    y_test:  pd.Series,
    verbose: bool = True,
) -> Dict:
    """
    Analyse class imbalance across all three splits.

    Why check all three splits separately?
    ─────────────────────────────────────────────────────────────
    If train has 60% UP but test has 40% UP, the model trained on
    train is calibrated for a different distribution than test.
    This distribution shift is a major source of live performance
    degradation. Detecting it here helps explain evaluation results.

    Returns
    -------
    Dict with per-split class ratios, scale_pos_weight, and
    a distribution_shift flag if train vs test differ by > 10pp.
    """
    def split_stats(y: pd.Series, name: str) -> Dict:
        up   = (y == 1).mean()
        down = (y == 0).mean()
        spw  = down / up if up > 0 else 1.0
        return {
            f"{name}_up_pct":          round(up,   4),
            f"{name}_down_pct":        round(down, 4),
            f"{name}_scale_pos_weight": round(spw,  4),
            f"{name}_majority_baseline": round(max(up, down), 4),
        }

    stats = {}
    stats.update(split_stats(y_train, "train"))
    stats.update(split_stats(y_val,   "val"))
    stats.update(split_stats(y_test,  "test"))

    # Distribution shift detection
    shift = abs(stats["train_up_pct"] - stats["test_up_pct"])
    stats["distribution_shift"] = round(float(shift), 4)
    stats["shift_warning"]      = shift > 0.10

    if verbose:
        print(f"\n{'═'*55}")
        print(f"Class Imbalance Analysis")
        print(f"{'═'*55}")
        print(f"{'Split':<8} {'UP%':>7} {'DOWN%':>7} "
              f"{'SPW':>7} {'Baseline':>10}")
        print(f"{'─'*55}")
        for split in ["train", "val", "test"]:
            print(
                f"{split:<8} "
                f"{stats[f'{split}_up_pct']*100:>6.1f}%  "
                f"{stats[f'{split}_down_pct']*100:>6.1f}%  "
                f"{stats[f'{split}_scale_pos_weight']:>6.3f}  "
                f"{stats[f'{split}_majority_baseline']*100:>9.1f}%"
            )
        print(f"{'─'*55}")
        if stats["shift_warning"]:
            print(f"⚠️  Distribution shift detected: "
                  f"{stats['distribution_shift']*100:.1f}pp between train and test")
            print(f"   Model trained on different class ratio than test set.")
            print(f"   Consider expanding training data to cover more regimes.")
        else:
            print(f"✅ Distribution stable: "
                  f"{stats['distribution_shift']*100:.1f}pp shift (< 10pp threshold)")

    return stats


def find_optimal_threshold(
    y_true:  pd.Series,
    y_proba: np.ndarray,
    objective: str = "f1",
    verbose: bool = True,
) -> float:
    """
    Find the optimal classification threshold on validation data.

    Parameters
    ----------
    y_true    : True labels (0/1).
    y_proba   : Predicted probabilities for class 1 (UP).
    objective : What to optimise.
                'f1'        — maximise F1 score (balanced)
                'precision' — maximise precision (fewer false UP signals)
                'recall'    — maximise recall (catch more UP days)

    Returns
    -------
    float threshold in [0, 1].

    Why tune threshold on validation, not test?
    ─────────────────────────────────────────────────────────────
    Threshold tuning is a form of hyperparameter optimisation.
    Optimising on test data makes test results overly optimistic —
    the threshold is tuned to the specific test period, which won't
    generalise to future data. Validation set is the correct choice.

    Why does this matter for trading?
    ─────────────────────────────────────────────────────────────
    Default 0.5 threshold treats UP and DOWN predictions equally.
    In trading, a false UP signal (buy when you should sell) costs
    real money. A higher threshold for UP predictions means fewer
    but more reliable buy signals — often the right trade-off.
    """
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, y_proba
    )

    # precision_recall_curve returns one more precision/recall than thresholds
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    if objective == "f1":
        scores   = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = scores.argmax()
    elif objective == "precision":
        best_idx = precisions.argmax()
    elif objective == "recall":
        best_idx = recalls.argmax()
    else:
        raise ValueError(
            f"objective='{objective}' not recognised. "
            f"Choose: 'f1', 'precision', 'recall'"
        )

    best_threshold = float(thresholds[best_idx])

    if verbose:
        print(f"\nOptimal threshold (objective='{objective}'):")
        print(f"  Threshold:  {best_threshold:.4f}  "
              f"(default was 0.5000)")
        print(f"  Precision:  {precisions[best_idx]:.4f}")
        print(f"  Recall:     {recalls[best_idx]:.4f}")
        print(f"  F1:         {2*(precisions[best_idx]*recalls[best_idx])/(precisions[best_idx]+recalls[best_idx]+1e-10):.4f}")

    return best_threshold


def predict_with_threshold(
    pipeline:  Pipeline,
    X:         pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions using a custom threshold.
    Replaces the default 0.5 threshold from pipeline.predict().

    Parameters
    ----------
    pipeline  : Fitted pipeline.
    X         : Feature DataFrame.
    threshold : Classification threshold. Values above → predict UP (1).

    Returns
    -------
    (predictions, probabilities) tuple.
    """
    X_r    = X.reset_index(drop=True)
    probas = pipeline.predict_proba(X_r)[:, 1]
    preds  = (probas >= threshold).astype(int)
    return preds, probas


# ══════════════════════════════════════════════════════════════════════════════
#  UPDATED TRAIN FUNCTION (integrates imbalance analysis)
# ══════════════════════════════════════════════════════════════════════════════

def train_with_imbalance_analysis(
    X:          pd.DataFrame,
    y:          pd.Series,
    xgb_params: Optional[Dict] = None,
    threshold_objective: str   = "f1",
    verbose:    bool           = True,
) -> Tuple[Pipeline, float, Dict]:
    """
    Full training with class imbalance analysis and threshold optimisation.

    Extends train() with:
    - Pre-training imbalance analysis
    - Optimal threshold search on validation set
    - Distribution shift detection

    Returns
    -------
    (pipeline, optimal_threshold, results) tuple.
    Use predict_with_threshold(pipeline, X, optimal_threshold)
    for production predictions.
    """
    # Split first so we can analyse each split's balance
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = _three_way_split(X, y)

    # Analyse imbalance
    imbalance_stats = analyse_imbalance(y_train, y_val, y_test, verbose)

    # Train with scale_pos_weight from training set
    pipeline, results = train(X, y, xgb_params=xgb_params, verbose=verbose)
    results.update(imbalance_stats)

    # Find optimal threshold on validation set
    X_val_r   = X_val.reset_index(drop=True)
    _, probas = predict(pipeline, X_val_r)
    threshold = find_optimal_threshold(
        y_val.reset_index(drop=True),
        probas,
        objective=threshold_objective,
        verbose=verbose,
    )

    results["optimal_threshold"] = threshold

    return pipeline, threshold, results
