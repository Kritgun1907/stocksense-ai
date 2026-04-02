"""
StockSense AI — models/timeseries_cv.py
=========================================
Time-series cross-validation utilities.

This file owns:
  - TimeSeriesSplit with gap (prevents rolling window boundary leakage)
  - Walk-forward validation (purest simulation of real trading)
  - Cross-validation scoring with financial metrics
  - Fold visualisation

It does NOT own:
  - Model training          → models/trainer.py  (future)
  - Hyperparameter tuning   → models/tuner.py    (future)
  - Evaluation metrics      → models/evaluator.py (future)

Why TimeSeriesSplit over random split?
─────────────────────────────────────────────────────────────
  Random splits shuffle time order, causing three types of leakage:
    1. Lag features contain values from training rows
    2. Rolling window features use training-period prices
    3. Target labels reference adjacent rows in training set
  TimeSeriesSplit ensures test always comes AFTER train — eliminating
  all three leakage types simultaneously.

Why include a gap between train and test?
─────────────────────────────────────────────────────────────
  Rolling features (e.g. SMA_200) on the first test row use 200
  previous rows. If those rows are in training, there's overlap.
  A gap of MAX_WINDOW days (default 20) eliminates this overlap.
  Without the gap, cross-validation scores are ~1-2pp optimistic.

Why expanding window (not rolling window)?
─────────────────────────────────────────────────────────────
  Rolling window: train always uses the same amount of data.
  Expanding window: train grows with each fold.
  Expanding is better because: (1) more data = better model,
  (2) it mirrors real deployment where you retrain on all history.
  Use rolling only if you suspect severe concept drift in old data.

Integration contract
─────────────────────────────────────────────────────────────
  Consumes output of:
    data/assembler.py    → assemble_multiple_stocks() → (X, y)
                           Both have (date, ticker) MultiIndex.
                           X columns = get_model_features() output (~342 cols).

  Produced by:
    features/engineer.py → build_features()
    data/labeller.py     → create_labels()
    features/indicators.py → get_model_features()

  Expected X shape: (n_rows, ~342) with (date, ticker) MultiIndex or
                    plain DatetimeIndex for single-stock use.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Iterator, List, Optional, Tuple
from sklearn.base import BaseEstimator

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Default gap between train end and test start.
# 20 trading days = covers our longest rolling window (SMA_20, ATR_20).
# Our longest window is SMA_200 in indicators.py add_trend_features() —
# use CV config "conservative" (gap_days=50) for production runs.
DEFAULT_GAP_DAYS = 20

# Default number of cross-validation folds.
# 5 folds gives 5 independent test periods to average over.
# More folds = more reliable score estimate but slower.
DEFAULT_N_SPLITS = 5

# Minimum rows required in each test fold.
# Test folds smaller than this are unreliable for evaluation.
MIN_TEST_FOLD_SIZE = 50


# ══════════════════════════════════════════════════════════════════════════════
#  CORE SPLITTER
# ══════════════════════════════════════════════════════════════════════════════

class TimeSeriesSplitWithGap:
    """
    Time-series cross-validation with a configurable gap between
    train and test folds.

    Unlike sklearn's TimeSeriesSplit, this:
      1. Supports a gap parameter to prevent rolling window leakage
      2. Works with pandas DatetimeIndex AND (date, ticker) MultiIndex
         (splits by unique calendar dates, not by row count — correct
         for multi-stock DataFrames from assembler.py)
      3. Reports fold metadata (date ranges, row counts)
      4. Validates that test folds meet minimum size requirements

    Parameters
    ----------
    n_splits  : Number of cross-validation folds.
    gap_days  : Trading days to skip between train end and test start.
                Set to max(rolling_window_lengths) in your feature set.
                Default: 20 (covers SMA_20, ATR_20, BB_20).
                Use 50 if SMA_200 features are present (see get_cv_config).
    test_size : Fraction of total data to use as test in each fold.
                Default: None → equal-sized folds.

    Usage (single stock)
    --------------------
    cv = TimeSeriesSplitWithGap(n_splits=5, gap_days=20)
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    Usage (multi-stock from assembler.py)
    --------------------------------------
    X, y, metadata = assemble_multiple_stocks(tickers, config_name="default")
    cv = TimeSeriesSplitWithGap(n_splits=5, gap_days=20)
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    Or with metadata:
    -----------------
    for fold_info in cv.split_with_info(X):
        train_idx   = fold_info['train_idx']
        test_idx    = fold_info['test_idx']
        train_dates = fold_info['train_dates']
        ...
    """

    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        gap_days: int = DEFAULT_GAP_DAYS,
        test_size: Optional[float] = None,
    ):
        if n_splits < 2:
            raise ValueError(
                f"n_splits must be >= 2, got {n_splits}. "
                f"With only 1 fold you cannot cross-validate."
            )
        if gap_days < 0:
            raise ValueError(
                f"gap_days must be >= 0, got {gap_days}."
            )
        if test_size is not None and not (0.0 < test_size < 1.0):
            raise ValueError(
                f"test_size must be in (0, 1), got {test_size}."
            )

        self.n_splits  = n_splits
        self.gap_days  = gap_days
        self.test_size = test_size

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_dates(self, X: pd.DataFrame) -> pd.DatetimeIndex:
        """
        Extract the date axis from a DataFrame regardless of index type.

        Handles:
          - Plain DatetimeIndex (single-stock pipeline)
          - (date, ticker) MultiIndex (assembler.py multi-stock output)
        """
        if isinstance(X.index, pd.MultiIndex):
            raw = X.index.get_level_values("date")
        else:
            raw = X.index
        return pd.DatetimeIndex(raw).normalize()

    # ── sklearn-compatible split ───────────────────────────────────────────────

    def split(
        self, X: pd.DataFrame, y=None, groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_indices, test_indices) tuples.
        Indices are integer positions (iloc-compatible), not labels.

        Why integer positions rather than labels?
        ─────────────────────────────────────────────────────────────
        Labels (dates) in a multi-stock DataFrame are not unique —
        AAPL and GOOGL both have 2024-01-05. iloc positions are always
        unique. Consistent with sklearn's API convention.
        """
        for fold_info in self.split_with_info(X):
            yield fold_info["train_idx"], fold_info["test_idx"]

    def split_with_info(
        self, X: pd.DataFrame
    ) -> Iterator[Dict]:
        """
        Generate fold dictionaries with full metadata.
        Use this when you need date ranges for logging or visualisation.

        Yields
        ------
        dict with keys:
          fold        : int fold number (1-indexed)
          train_idx   : np.ndarray of integer positions for training
          test_idx    : np.ndarray of integer positions for testing
          train_dates : (start_date, end_date) tuple
          test_dates  : (start_date, end_date) tuple
          gap_dates   : (gap_start, gap_end) tuple
          n_train     : int number of training rows
          n_test      : int number of test rows
        """
        row_dates    = self._extract_dates(X)
        unique_dates = row_dates.unique().sort_values()
        n_dates      = len(unique_dates)

        # ── Calculate fold sizes ──────────────────────────────────────────────
        if self.test_size is not None:
            test_fold_dates = max(
                MIN_TEST_FOLD_SIZE,
                int(n_dates * self.test_size)
            )
        else:
            # Equal folds: divide total dates by (n_splits + 1)
            # The "+1" reserves the first chunk as minimum training set
            test_fold_dates = max(
                MIN_TEST_FOLD_SIZE,
                n_dates // (self.n_splits + 1)
            )

        min_train_dates = n_dates - (self.n_splits * test_fold_dates)

        if min_train_dates < 50:
            raise ValueError(
                f"Not enough data for {self.n_splits} folds. "
                f"Have {n_dates} unique dates, need at least "
                f"{50 + self.n_splits * test_fold_dates}. "
                f"Reduce n_splits or use more data."
            )

        for fold in range(self.n_splits):
            # ── Test fold boundaries in unique_dates space ────────────────────
            test_start_pos = min_train_dates + fold * test_fold_dates
            test_end_pos   = min(
                test_start_pos + test_fold_dates,
                n_dates
            )

            if test_start_pos >= n_dates:
                break

            test_start_date = unique_dates[test_start_pos]
            test_end_date   = unique_dates[test_end_pos - 1]

            # ── Train ends gap_days before test starts ────────────────────────
            train_end_pos = test_start_pos - self.gap_days
            if train_end_pos <= 0:
                warnings.warn(
                    f"Fold {fold+1}: gap_days={self.gap_days} is too large "
                    f"for this fold — skipping.",
                    UserWarning
                )
                continue

            train_end_date = unique_dates[train_end_pos - 1]
            gap_start_date = unique_dates[train_end_pos]
            gap_end_date   = unique_dates[test_start_pos - 1]

            # ── Convert date boundaries to integer row positions ──────────────
            train_mask = row_dates <= train_end_date
            test_mask  = (
                (row_dates >= test_start_date) &
                (row_dates <= test_end_date)
            )

            train_idx = np.where(train_mask)[0]
            test_idx  = np.where(test_mask)[0]

            if len(test_idx) < MIN_TEST_FOLD_SIZE:
                warnings.warn(
                    f"Fold {fold+1}: test set too small "
                    f"({len(test_idx)} rows < {MIN_TEST_FOLD_SIZE}). "
                    f"Skipping.",
                    UserWarning
                )
                continue

            yield {
                "fold":        fold + 1,
                "train_idx":   train_idx,
                "test_idx":    test_idx,
                "train_dates": (
                    row_dates[train_idx[0]],
                    row_dates[train_idx[-1]]
                ),
                "test_dates":  (
                    row_dates[test_idx[0]],
                    row_dates[test_idx[-1]]
                ),
                "gap_dates":   (gap_start_date, gap_end_date),
                "n_train":     len(train_idx),
                "n_test":      len(test_idx),
            }

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """sklearn API compatibility."""
        return self.n_splits


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION SCORER
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_timeseries(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
    gap_days: int = DEFAULT_GAP_DAYS,
    metrics: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Cross-validate a model using time-series-aware splitting.

    Designed to work with the output of assembler.assemble_multiple_stocks():
      X has (date, ticker) MultiIndex, ~342 columns from get_model_features().
      y is binary int {0, 1} from create_labels().

    Parameters
    ----------
    model    : Any sklearn-compatible estimator with fit/predict.
               XGBClassifier from pipeline.py is the primary use case.
    X        : Feature DataFrame (iloc-indexable).
               Single-stock DatetimeIndex or multi-stock (date, ticker) MultiIndex.
    y        : Binary target Series (0=DOWN, 1=UP).
    n_splits : Number of CV folds (default: 5).
    gap_days : Trading days gap between train end and test start (default: 20).
    metrics  : List of metric names to compute.
               Default: ["accuracy", "f1", "precision", "recall", "auc_roc"].
    verbose  : Print per-fold results.

    Returns
    -------
    dict with keys:
      fold_scores    : List of per-fold score dicts
      mean_scores    : Mean of each metric across folds
      std_scores     : Std deviation of each metric across folds
      cv             : The TimeSeriesSplitWithGap object used

    Available metrics
    -----------------
      accuracy  : Overall fraction correctly classified
      f1        : Harmonic mean of precision and recall (UP class)
      precision : True UP / (True UP + False UP) — how often predictions are right
      recall    : True UP / all actual UP — how many real UPs are caught
      auc_roc   : Area under ROC curve — ranking quality (threshold-independent)

    Why AUC-ROC matters for trading
    ---------------------------------
      A model predicting always UP gets 100% recall but 0% precision.
      AUC-ROC penalises this — it measures whether the model correctly
      ranks UP days above DOWN days, which is exactly what a trading
      signal needs to do to be useful.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    if metrics is None:
        metrics = ["accuracy", "f1", "precision", "recall", "auc_roc"]

    valid_metrics = {"accuracy", "f1", "precision", "recall", "auc_roc"}
    bad = set(metrics) - valid_metrics
    if bad:
        raise ValueError(
            f"Unknown metrics: {bad}. "
            f"Available: {sorted(valid_metrics)}"
        )

    cv          = TimeSeriesSplitWithGap(n_splits=n_splits, gap_days=gap_days)
    fold_scores: List[Dict] = []

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Time-Series Cross-Validation")
        print(f"  Folds : {n_splits}  |  Gap : {gap_days}d  |  "
              f"Metrics : {metrics}")
        print(f"{'═'*60}")

    for fold_info in cv.split_with_info(X):
        fold     = fold_info["fold"]
        X_train  = X.iloc[fold_info["train_idx"]]
        X_test   = X.iloc[fold_info["test_idx"]]
        y_train  = y.iloc[fold_info["train_idx"]]
        y_test   = y.iloc[fold_info["test_idx"]]

        # ── Fit and predict ───────────────────────────────────────────────────
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred.astype(float)
        )

        # ── Compute requested metrics ─────────────────────────────────────────
        scores: Dict = {"fold": fold}
        if "accuracy"  in metrics:
            scores["accuracy"]  = float(accuracy_score(y_test, y_pred))
        if "f1"        in metrics:
            scores["f1"]        = float(f1_score(y_test, y_pred, zero_division=0))
        if "precision" in metrics:
            scores["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        if "recall"    in metrics:
            scores["recall"]    = float(recall_score(y_test, y_pred, zero_division=0))
        if "auc_roc"   in metrics:
            try:
                scores["auc_roc"] = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                # Can happen if test fold has only one class
                scores["auc_roc"] = 0.5

        fold_scores.append(scores)

        if verbose:
            train_start, train_end = fold_info["train_dates"]
            test_start,  test_end  = fold_info["test_dates"]
            print(f"\nFold {fold}:")
            print(f"  Train : {train_start.date()} → {train_end.date()} "
                  f"({fold_info['n_train']:,} rows)")
            print(f"  Gap   : {fold_info['gap_dates'][0].date()} → "
                  f"{fold_info['gap_dates'][1].date()}")
            print(f"  Test  : {test_start.date()} → {test_end.date()} "
                  f"({fold_info['n_test']:,} rows)")
            metric_str = "  Scores: " + " | ".join(
                f"{k}={v:.4f}" for k, v in scores.items() if k != "fold"
            )
            print(metric_str)

    if not fold_scores:
        raise ValueError(
            "No valid folds were generated. "
            "Increase your dataset size or reduce n_splits / gap_days."
        )

    # ── Aggregate across folds ────────────────────────────────────────────────
    mean_scores: Dict[str, float] = {}
    std_scores:  Dict[str, float] = {}

    for metric in metrics:
        values = [s[metric] for s in fold_scores if metric in s]
        if values:
            mean_scores[metric] = float(np.mean(values))
            std_scores[metric]  = float(np.std(values))

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Mean scores across {len(fold_scores)} folds:")
        for metric, mean in mean_scores.items():
            std = std_scores.get(metric, 0.0)
            majority_note = ""
            if metric == "accuracy" and mean < 0.52:
                majority_note = "  ⚠ near majority-class baseline"
            print(f"  {metric:<12}: {mean:.4f} ± {std:.4f}{majority_note}")
        print(f"{'═'*60}\n")

    return {
        "fold_scores": fold_scores,
        "mean_scores": mean_scores,
        "std_scores":  std_scores,
        "cv":          cv,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_validate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    min_train_days: int = 252,
    test_window_days: int = 21,
    gap_days: int = DEFAULT_GAP_DAYS,
    retrain_freq: int = 1,
    verbose: bool = True,
) -> Dict:
    """
    Walk-forward validation: the purest simulation of live trading.

    At each step the model is trained on all historical data up to T,
    then evaluated on the next test_window_days rows, then the window
    advances by retrain_freq folds.

    Unlike cross_validate_timeseries (which uses fixed fold sizes),
    walk-forward always trains on maximum available history — mimicking
    how you would actually deploy: retrain weekly on all data to date.

    Parameters
    ----------
    model            : sklearn-compatible estimator.
    X                : Feature DataFrame.
    y                : Binary target.
    min_train_days   : Minimum unique trading days required before first fold.
                       Default: 252 (1 trading year). Set to lower for
                       shorter datasets during development.
    test_window_days : Number of unique trading days in each test window.
                       Default: 21 (≈1 month). Equivalent to monthly retraining.
    gap_days         : Gap between train end and test start (default: 20).
    retrain_freq     : Advance the window by this many test_window_days
                       before retraining. 1 = retrain every window (default).
                       2 = retrain every other window (faster).
    verbose          : Print per-step results.

    Returns
    -------
    dict with keys:
      step_scores    : List of per-step score dicts
      mean_scores    : Mean across all steps
      std_scores     : Std across all steps
      n_steps        : Number of evaluation steps completed
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    row_dates    = _extract_dates_from(X)
    unique_dates = row_dates.unique().sort_values()
    n_dates      = len(unique_dates)

    if n_dates < min_train_days + gap_days + test_window_days:
        raise ValueError(
            f"Not enough unique dates ({n_dates}) for walk-forward validation. "
            f"Need at least {min_train_days + gap_days + test_window_days} "
            f"({min_train_days} train + {gap_days} gap + {test_window_days} test)."
        )

    step_scores: List[Dict] = []
    step       = 0
    train_end_pos = min_train_days - 1

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Walk-Forward Validation")
        print(f"  Min train: {min_train_days}d | "
              f"Test window: {test_window_days}d | "
              f"Gap: {gap_days}d | Retrain every: {retrain_freq} window(s)")
        print(f"{'═'*60}")

    fitted_model = None

    while True:
        test_start_pos = train_end_pos + gap_days + 1
        test_end_pos   = min(test_start_pos + test_window_days, n_dates)

        if test_end_pos > n_dates or test_start_pos >= n_dates:
            break

        train_end_date  = unique_dates[train_end_pos]
        test_start_date = unique_dates[test_start_pos]
        test_end_date   = unique_dates[test_end_pos - 1]

        train_mask = row_dates <= train_end_date
        test_mask  = (
            (row_dates >= test_start_date) &
            (row_dates <= test_end_date)
        )

        X_train = X.iloc[np.where(train_mask)[0]]
        X_test  = X.iloc[np.where(test_mask)[0]]
        y_train = y.iloc[np.where(train_mask)[0]]
        y_test  = y.iloc[np.where(test_mask)[0]]

        if len(X_test) < 10:
            train_end_pos += test_window_days * retrain_freq
            continue

        # Retrain only every retrain_freq steps
        if step % retrain_freq == 0 or fitted_model is None:
            model.fit(X_train, y_train)
            fitted_model = model

        y_pred  = fitted_model.predict(X_test)
        y_proba = (
            fitted_model.predict_proba(X_test)[:, 1]
            if hasattr(fitted_model, "predict_proba")
            else y_pred.astype(float)
        )

        acc = float(accuracy_score(y_test, y_pred))
        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            auc = 0.5

        step_scores.append({
            "step":         step + 1,
            "train_rows":   int(train_mask.sum()),
            "test_rows":    len(X_test),
            "train_end":    train_end_date,
            "test_start":   test_start_date,
            "test_end":     test_end_date,
            "accuracy":     acc,
            "auc_roc":      auc,
        })

        if verbose:
            print(f"\nStep {step+1}: train→{train_end_date.date()} "
                  f"({int(train_mask.sum()):,} rows)  "
                  f"test {test_start_date.date()}→{test_end_date.date()} "
                  f"({len(X_test)} rows)  "
                  f"acc={acc:.4f}  auc={auc:.4f}")

        step += 1
        train_end_pos += test_window_days * retrain_freq

    if not step_scores:
        raise ValueError("Walk-forward produced no valid steps.")

    accs = [s["accuracy"] for s in step_scores]
    aucs = [s["auc_roc"]  for s in step_scores]

    mean_scores = {
        "accuracy": float(np.mean(accs)),
        "auc_roc":  float(np.mean(aucs)),
    }
    std_scores = {
        "accuracy": float(np.std(accs)),
        "auc_roc":  float(np.std(aucs)),
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Walk-forward summary ({len(step_scores)} steps):")
        print(f"  accuracy : {mean_scores['accuracy']:.4f} ± "
              f"{std_scores['accuracy']:.4f}")
        print(f"  auc_roc  : {mean_scores['auc_roc']:.4f}  ± "
              f"{std_scores['auc_roc']:.4f}")
        print(f"{'═'*60}\n")

    return {
        "step_scores": step_scores,
        "mean_scores": mean_scores,
        "std_scores":  std_scores,
        "n_steps":     len(step_scores),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _extract_dates_from(X: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Module-level helper: extract normalised date axis from
    either a plain DatetimeIndex or (date, ticker) MultiIndex.
    Used by both TimeSeriesSplitWithGap and walk_forward_validate.
    """
    if isinstance(X.index, pd.MultiIndex):
        raw = X.index.get_level_values("date")
    else:
        raw = X.index
    return pd.DatetimeIndex(raw).normalize()


def visualise_splits(
    X: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    gap_days: int = DEFAULT_GAP_DAYS,
) -> None:
    """
    Print ASCII visualisation of train/gap/test splits.
    Useful for verifying split configuration before training.

    Legend: █ = train   ░ = gap   ▓ = test
    """
    cv    = TimeSeriesSplitWithGap(n_splits=n_splits, gap_days=gap_days)
    width = 60

    print(f"\nTimeSeriesSplit Visualisation")
    print(f"  n_splits={n_splits}  |  gap_days={gap_days}")
    print(f"  Legend: █=train  ░=gap  ▓=test")
    print("─" * width)

    all_dates = _extract_dates_from(X).unique().sort_values()
    date_min  = all_dates[0]
    date_max  = all_dates[-1]
    total_days = (date_max - date_min).days
    if total_days == 0:
        print("(Only one unique date — cannot visualise)")
        return

    def _to_pos(date: pd.Timestamp) -> int:
        return int((date - date_min).days / total_days * (width - 1))

    for fold_info in cv.split_with_info(X):
        fold = fold_info["fold"]
        bar  = ["─"] * width

        train_start, train_end = fold_info["train_dates"]
        gap_start,   gap_end   = fold_info["gap_dates"]
        test_start,  test_end  = fold_info["test_dates"]

        for i in range(_to_pos(train_start), min(_to_pos(train_end) + 1, width)):
            bar[i] = "█"
        for i in range(_to_pos(gap_start), min(_to_pos(gap_end) + 1, width)):
            bar[i] = "░"
        for i in range(_to_pos(test_start), min(_to_pos(test_end) + 1, width)):
            bar[i] = "▓"

        print(f"Fold {fold}: {''.join(bar)}")
        print(f"        Train→{train_end.date()} | "
              f"Gap | Test {test_start.date()}→{test_end.date()}")

    print("─" * width)


def get_cv_config(name: str) -> Dict:
    """
    Retrieve a named cross-validation configuration.

    Mirrors get_label_config() / get_assembly_config() pattern from
    data/labeller.py and data/assembler.py — one consistent registry
    pattern across all StockSense AI modules.

    Parameters
    ----------
    name : Config name. Call list_cv_configs() to see all names.

    Returns
    -------
    dict with keys: n_splits, gap_days, description.

    Raises
    ------
    ValueError : If name not found.

    Example
    -------
    >>> cfg = get_cv_config('conservative')
    >>> cv = TimeSeriesSplitWithGap(**{k: v for k, v in cfg.items()
    ...                                if k != 'description'})
    """
    configs = {
        "default": {
            "n_splits": 5,
            "gap_days": 20,
            "description": "5-fold CV, 20-day gap — standard for daily data",
        },
        "fast": {
            "n_splits": 3,
            "gap_days": 10,
            "description": "3-fold CV, 10-day gap — faster for development",
        },
        "rigorous": {
            "n_splits": 8,
            "gap_days": 20,
            "description": "8-fold CV, 20-day gap — thorough evaluation",
        },
        "conservative": {
            "n_splits": 5,
            "gap_days": 50,
            "description": (
                "5-fold CV, 50-day gap — for models with long-window "
                "features (SMA_200 in add_trend_features)"
            ),
        },
    }
    if name not in configs:
        raise ValueError(
            f"CV config '{name}' not found. "
            f"Available: {list(configs.keys())}"
        )
    return dict(configs[name])


def list_cv_configs(verbose: bool = True) -> List[str]:
    """
    List all available CV configurations.

    Returns
    -------
    List of config names. Prints a summary table if verbose=True.
    """
    configs = {
        "default":      {"n_splits": 5, "gap_days": 20},
        "fast":         {"n_splits": 3, "gap_days": 10},
        "rigorous":     {"n_splits": 8, "gap_days": 20},
        "conservative": {"n_splits": 5, "gap_days": 50},
    }
    descriptions = {
        "default":      "Standard for daily data",
        "fast":         "Faster for development",
        "rigorous":     "Thorough evaluation",
        "conservative": "Long-window features (SMA_200)",
    }
    if verbose:
        print(f"\n{'Config':<16} {'Splits':>7} {'Gap':>6}  Description")
        print("─" * 60)
        for name, cfg in configs.items():
            print(
                f"  {name:<14} {cfg['n_splits']:>6} "
                f"{cfg['gap_days']:>5}d  {descriptions[name]}"
            )
    return list(configs.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import yfinance as yf
    from xgboost import XGBClassifier

    from data.cleaner import clean_stock_data
    from data.labeller import create_labels
    from features.engineer import build_features
    from features.indicators import get_model_features

    # ── List available configs ────────────────────────────────────────────────
    list_cv_configs()

    # ── Fetch and prepare AAPL (2 years — enough for 5 folds) ────────────────
    print("\nDownloading AAPL 2y...")
    raw = yf.download("AAPL", period="2y", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]

    clean    = clean_stock_data(raw, ticker="AAPL")
    featured = build_features(clean).dropna()
    labelled = create_labels(featured, horizon=1, threshold=0.003, verbose=False)
    X = get_model_features(labelled, extra_drop=["target"]).fillna(0)
    y = labelled["target"]

    n_rows = len(X)
    print(f"\nDataset: {n_rows} rows × {len(X.columns)} features")

    # ── Auto-select fold count based on available rows ────────────────────────
    # 5 folds needs ~300+ unique dates; use 3 folds for shorter datasets
    # (common after dropna on 2y period which yields ~230-260 labelled rows).
    n_folds   = 5 if n_rows >= 300 else 3
    gap       = 20
    min_train = max(100, n_rows // (n_folds + 1))
    print(f"  Auto-selected: n_splits={n_folds}, gap_days={gap}")

    # ── Visualise splits ──────────────────────────────────────────────────────
    visualise_splits(X, n_splits=n_folds, gap_days=gap)

    # ── Run standard cross-validation ────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )

    results = cross_validate_timeseries(
        model=model,
        X=X,
        y=y,
        n_splits=n_folds,
        gap_days=gap,
        verbose=True,
    )

    print("\nCross-validated scores:")
    for metric, mean in results["mean_scores"].items():
        std = results["std_scores"][metric]
        print(f"  {metric:<12}: {mean:.4f} ± {std:.4f}")

    # ── Run walk-forward validation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    wf_results = walk_forward_validate(
        model=XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
        ),
        X=X,
        y=y,
        min_train_days=min_train,
        test_window_days=21,
        gap_days=gap,
        verbose=True,
    )

    print(f"\nWalk-forward summary ({wf_results['n_steps']} steps):")
    for metric, mean in wf_results["mean_scores"].items():
        std = wf_results["std_scores"][metric]
        print(f"  {metric:<12}: {mean:.4f} ± {std:.4f}")
