"""
StockSense AI — models/tuner.py
=================================
Bayesian hyperparameter optimisation with Optuna.

This file owns:
  - Optuna study creation and optimisation
  - XGBoost search space definition
  - Objective function with time-series CV scoring
  - Tuning configuration presets
  - Best parameter extraction and reporting
  - Integration with trainer.py for final retraining

It does NOT own:
  - Model pipeline construction → models/trainer.py    (build_pipeline)
  - Cross-validation splitting  → models/timeseries_cv.py (TimeSeriesSplitWithGap)
  - Evaluation metrics          → models/evaluator.py
  - SHAP explainability         → models/explainer.py

Why Optuna (Bayesian) over GridSearch or RandomSearch?
─────────────────────────────────────────────────────────────
  GridSearch evaluates every combination. With 8 parameters ×
  5 values each = 390,625 trials. At ~30s per trial (3-fold CV
  with 500 estimators), that is 135 days of compute.

  RandomSearch samples randomly — better than grid but still
  wastes trials exploring regions that clearly don't work.

  Optuna uses a Tree-structured Parzen Estimator (TPE) which:
    1. Builds a probabilistic model of the objective function
    2. Samples next trial from regions that look promising
    3. Avoids regions that consistently underperform
  Result: finds better parameters in 50-100 trials (~25-50 min)
  vs GridSearch needing 100,000+ trials.

Why AUC-ROC as the optimisation metric?
─────────────────────────────────────────────────────────────
  Accuracy depends on the classification threshold (default 0.5).
  A model predicting always UP gets high accuracy when UP is majority.
  AUC-ROC is threshold-independent — it measures whether the model
  correctly ranks UP days above DOWN days. This is exactly what a
  trading signal needs: rank days by opportunity, then pick the top N.
  trainer.py's find_optimal_threshold() then selects the best cutoff.

Why TimeSeriesSplitWithGap inside the objective?
─────────────────────────────────────────────────────────────
  Standard CV (KFold) shuffles time order → future data leaks into
  training → scores are 3-5pp too optimistic → selected parameters
  overfit to noise. TimeSeriesSplitWithGap ensures every fold's test
  period strictly follows its training period, with a 20-day gap
  to prevent rolling feature overlap. Parameters selected this way
  generalise to unseen future data.

Why not use cross_validate_timeseries() directly?
─────────────────────────────────────────────────────────────
  cross_validate_timeseries() creates a fresh Pipeline via clone()
  for each fold. But Optuna needs to test each trial's specific
  hyperparameters. We build a fresh pipeline inside the objective
  using trainer.build_pipeline(xgb_params=...) and manually drive
  the CV loop for maximum control over what's measured.

Why Optuna pruning (MedianPruner)?
─────────────────────────────────────────────────────────────
  If fold 1 of a trial scores 0.48 AUC-ROC while the median
  of completed trials is 0.58, continuing folds 2-5 is a waste.
  MedianPruner kills underperforming trials early — typically
  cutting total runtime by 40-60% with no quality loss.

Integration contract
─────────────────────────────────────────────────────────────
  Consumes:
    trainer.build_pipeline(xgb_params, scale_pos_weight)   → unfitted Pipeline
    trainer._calculate_scale_pos_weight(y_train)           → float
    timeseries_cv.TimeSeriesSplitWithGap                   → CV splitter
    timeseries_cv.DEFAULT_GAP_DAYS, DEFAULT_N_SPLITS       → constants
    assembler.assemble_stock() / assemble_multiple_stocks() → (X, y)

  Produces:
    best_params dict → passed to trainer.train(xgb_params=best_params)
    study object     → Optuna study for inspection / visualisation
    tuning report    → printed summary of all trials and convergence
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Directory for saving tuning results (studies, best params).
TUNING_DIR = "models/tuning_results"

# Default optimisation metric.
# AUC-ROC is threshold-independent and measures ranking quality —
# the single most important property for a trading signal generator.
DEFAULT_METRIC = "auc_roc"

# Number of Optuna trials (Bayesian iterations).
# 80 trials balances exploration vs exploitation well.
# Optuna's TPE sampler transitions from exploration (random) to
# exploitation (focused) around trial 20-30, so 80 gives ~50 trials
# of focused Bayesian search after the warm-up phase.
DEFAULT_N_TRIALS = 80

# Cross-validation folds for tuning.
# 3 folds during tuning (fast), then 5 for final validation.
# Tuning doesn't need 5 — the goal is to compare trials relatively,
# not to get precise absolute scores. 3 folds cuts runtime by 40%.
TUNING_N_SPLITS = 3

# Gap days during tuning.
# Same as production default from timeseries_cv.py.
# Must match production settings or tuned params won't transfer.
TUNING_GAP_DAYS = 20

# Timeout in seconds (optional safety net).
# Prevents infinite runs if the user sets n_trials too high.
DEFAULT_TIMEOUT = 3600  # 1 hour

# Optuna verbosity.
# WARNING = only show pruned trials and errors, not every trial.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH SPACE
# ══════════════════════════════════════════════════════════════════════════════

# XGBoost hyperparameter search space for Optuna.
#
# Each entry defines (type, low, high, [step/log]):
#   - int ranges are inclusive on both ends
#   - float ranges use log=True where the parameter is scale-sensitive
#     (learning_rate, reg_alpha, reg_lambda) because the difference
#     between 0.01 and 0.02 matters more than between 0.2 and 0.21
#   - Subsample and colsample are uniform because all values in
#     [0.5, 1.0] are equally meaningful

SEARCH_SPACE = {
    # ── Tree structure ────────────────────────────────────────────────────
    "n_estimators": {
        "type": "int", "low": 100, "high": 1000, "step": 50,
        # Number of boosting rounds.
        # Low end: faster training, underfits. High end: overfits if
        # learning_rate is too high. Early stopping in trainer.py
        # clips this automatically, so erring high is safe.
    },
    "max_depth": {
        "type": "int", "low": 3, "high": 10,
        # Maximum tree depth. Deeper trees capture more interactions
        # but overfit faster. 3-6 is standard for tabular data.
        # Stock data has noise — shallow trees generalise better.
    },
    "min_child_weight": {
        "type": "int", "low": 1, "high": 20,
        # Minimum sum of instance weight (hessian) in a child.
        # Higher values = more conservative splits = less overfitting.
        # For 1000-row datasets use 5-10; for 10,000+ use 1-5.
    },

    # ── Learning dynamics ─────────────────────────────────────────────────
    "learning_rate": {
        "type": "float", "low": 0.005, "high": 0.3, "log": True,
        # Step size shrinkage. Smaller = slower but more accurate.
        # log scale because 0.01→0.02 is a 100% change but
        # 0.2→0.21 is only a 5% change.
    },
    "subsample": {
        "type": "float", "low": 0.5, "high": 1.0,
        # Fraction of rows used per tree. Lower = more regularisation.
        # 0.5 means each tree sees half the data — reduces overfitting
        # on the specific training rows.
    },
    "colsample_bytree": {
        "type": "float", "low": 0.3, "high": 1.0,
        # Fraction of features used per tree. With ~340 features,
        # 0.3 = each tree sees ~100 features. Prevents feature
        # co-adaptation — trees learn different feature subsets.
    },

    # ── Regularisation ────────────────────────────────────────────────────
    "gamma": {
        "type": "float", "low": 0.0, "high": 5.0,
        # Minimum loss reduction required to split a leaf further.
        # Acts as a complexity penalty — higher = fewer splits.
        # 0 = no penalty (default XGBoost). 1-5 for noisy data.
    },
    "reg_alpha": {
        "type": "float", "low": 1e-8, "high": 10.0, "log": True,
        # L1 regularisation on leaf weights.
        # Encourages sparsity — some leaves get weight exactly 0.
        # log scale because the effective range spans many orders.
    },
    "reg_lambda": {
        "type": "float", "low": 1e-8, "high": 10.0, "log": True,
        # L2 regularisation on leaf weights.
        # Prevents any single leaf from having extreme predictions.
        # XGBoost default is 1.0 — we search around that.
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  TUNING CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

# Named presets matching the pattern used in labeller.py (LABEL_CONFIGS),
# assembler.py, and timeseries_cv.py (CV_CONFIGS).
# Each preset balances speed vs thoroughness.

TUNING_CONFIGS = {
    "quick": {
        # 20 trials, 3 folds — smoke test in ~5 min.
        # Use to verify the tuning pipeline works before committing
        # to a full run. Not suitable for production parameters.
        "n_trials":   20,
        "n_splits":   3,
        "gap_days":   20,
        "timeout":    600,
        "metric":     "auc_roc",
    },
    "default": {
        # 80 trials, 3 folds — standard tuning run (~25-40 min).
        # Good balance of exploration and exploitation.
        # Produces parameters suitable for paper trading.
        "n_trials":   80,
        "n_splits":   3,
        "gap_days":   20,
        "timeout":    3600,
        "metric":     "auc_roc",
    },
    "thorough": {
        # 150 trials, 5 folds — deep search (~2-3 hours).
        # More folds = more reliable scores per trial.
        # More trials = better exploration of the search space.
        # Use for final production parameter selection.
        "n_trials":   150,
        "n_splits":   5,
        "gap_days":   20,
        "timeout":    10800,
        "metric":     "auc_roc",
    },
    "conservative": {
        # 100 trials, 5 folds, 50-day gap — safest against leakage.
        # 50-day gap covers even SMA_200 boundary contamination.
        # Scores will be slightly lower but most honest.
        "n_trials":   100,
        "n_splits":   5,
        "gap_days":   50,
        "timeout":    7200,
        "metric":     "auc_roc",
    },
}


def get_tuning_config(name: str = "default") -> Dict:
    """
    Look up a tuning configuration preset by name.

    Available presets: quick, default, thorough, conservative.

    Returns
    -------
    dict with n_trials, n_splits, gap_days, timeout, metric keys.

    Raises
    ------
    ValueError if name is not in TUNING_CONFIGS.
    """
    if name not in TUNING_CONFIGS:
        available = ", ".join(sorted(TUNING_CONFIGS.keys()))
        raise ValueError(
            f"Unknown tuning config '{name}'. "
            f"Available: {available}"
        )
    return dict(TUNING_CONFIGS[name])


# ══════════════════════════════════════════════════════════════════════════════
#  OBJECTIVE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def _sample_params(trial: optuna.Trial) -> Dict:
    """
    Sample hyperparameters from the search space for one Optuna trial.

    Uses trial.suggest_int / trial.suggest_float to draw from
    SEARCH_SPACE. Optuna's TPE sampler uses the history of all
    previous trials to bias sampling toward promising regions.

    Parameters
    ----------
    trial : Optuna trial object (manages sampling and pruning).

    Returns
    -------
    dict of XGBoost hyperparameters ready for build_pipeline().
    """
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
            )
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
    return params


def _create_objective(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = TUNING_N_SPLITS,
    gap_days: int = TUNING_GAP_DAYS,
    metric:   str = DEFAULT_METRIC,
):
    """
    Factory that creates the Optuna objective function.

    Why a factory (closure)?
    ─────────────────────────────────────────────────────────────
    Optuna's study.optimize() expects a callable(trial) → float.
    The objective needs access to X, y, and CV settings, but Optuna
    only passes the trial object. A closure captures X, y, etc.
    in the enclosing scope — clean, no globals, no monkey-patching.

    Parameters
    ----------
    X        : Feature DataFrame from assembler.py.
    y        : Binary target Series from labeller.py.
    n_splits : CV folds for scoring each trial.
    gap_days : Gap between train/test in each fold.
    metric   : Which metric to optimise ("auc_roc", "f1", "accuracy").

    Returns
    -------
    Callable objective function for study.optimize().
    """
    # Import here to avoid circular imports at module level.
    # trainer.py imports from timeseries_cv.py, and this file
    # imports from both — keeping imports local prevents cycles.
    from models.trainer import build_pipeline, _calculate_scale_pos_weight
    from models.timeseries_cv import TimeSeriesSplitWithGap

    # Pre-compute scale_pos_weight from the full training portion.
    # This is an approximation — each CV fold has a slightly different
    # train set. But the difference is <2% and it saves recomputing
    # per fold. The final retrain in trainer.train() recomputes exactly.
    global_spw = _calculate_scale_pos_weight(y)

    # Metric function mapping.
    # Each function takes (y_true, y_score_or_pred) → float.
    metric_fns = {
        "auc_roc":  lambda yt, yp: roc_auc_score(yt, yp),
        "f1":       lambda yt, yp: f1_score(yt, (yp >= 0.5).astype(int),
                                             zero_division=0),
        "accuracy": lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int)),
    }
    if metric not in metric_fns:
        raise ValueError(
            f"metric='{metric}' not supported. "
            f"Choose from: {list(metric_fns.keys())}"
        )
    score_fn = metric_fns[metric]

    # Build the CV splitter once — reused across all trials.
    cv = TimeSeriesSplitWithGap(n_splits=n_splits, gap_days=gap_days)

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective: sample params → build pipeline → CV score.

        Returns the mean CV score (higher = better, Optuna maximises).
        Reports intermediate per-fold scores to enable pruning.
        """
        # ── Step 1: Sample hyperparameters for this trial ─────────────────
        xgb_params = _sample_params(trial)

        # ── Step 2: Build pipeline with sampled params ────────────────────
        # build_pipeline() constructs: Imputer → Scaler → XGBClassifier.
        # We disable early_stopping_rounds here because we don't have a
        # separate validation set inside CV folds. Setting it to None
        # prevents XGBoost from requiring eval_set at fit time.
        xgb_params["early_stopping_rounds"] = None
        xgb_params["eval_metric"] = "logloss"
        xgb_params["verbosity"] = 0
        xgb_params["n_jobs"] = -1

        # ── Step 3: Time-series cross-validation ──────────────────────────
        fold_scores = []

        for fold_idx, fold_info in enumerate(cv.split_with_info(X)):
            # Extract train/test for this fold
            X_train = X.iloc[fold_info["train_idx"]]
            X_test  = X.iloc[fold_info["test_idx"]]
            y_train = y.iloc[fold_info["train_idx"]]
            y_test  = y.iloc[fold_info["test_idx"]]

            # Build a fresh pipeline for each fold.
            # Cannot reuse — fitting modifies the Imputer and Scaler state.
            fold_spw  = _calculate_scale_pos_weight(y_train)
            pipeline  = build_pipeline(
                xgb_params=xgb_params,
                scale_pos_weight=fold_spw,
            )

            # Fit on training fold.
            # reset_index(drop=True) because XGBoost's DMatrix
            # doesn't understand pandas MultiIndex.
            pipeline.fit(
                X_train.reset_index(drop=True),
                y_train.reset_index(drop=True),
            )

            # Score on test fold.
            y_proba = pipeline.predict_proba(
                X_test.reset_index(drop=True)
            )[:, 1]

            try:
                fold_score = score_fn(
                    y_test.reset_index(drop=True), y_proba
                )
            except ValueError:
                # Can happen if test fold has only one class
                fold_score = 0.5

            fold_scores.append(fold_score)

            # ── Report intermediate value for pruning ─────────────────────
            # Optuna's MedianPruner compares the running mean of this
            # trial's fold scores against the median of completed trials.
            # If this trial is clearly below median, it gets pruned —
            # saving the remaining fold computations.
            trial.report(np.mean(fold_scores), fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        # ── Step 4: Return mean score across all folds ────────────────────
        mean_score = float(np.mean(fold_scores))
        return mean_score

    return objective


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TUNING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def tune(
    X: pd.DataFrame,
    y: pd.Series,
    config_name: str = "default",
    n_trials:   Optional[int] = None,
    n_splits:   Optional[int] = None,
    gap_days:   Optional[int] = None,
    timeout:    Optional[int] = None,
    metric:     Optional[str] = None,
    study_name: Optional[str] = None,
    seed:       int = 42,
    verbose:    bool = True,
) -> Tuple[Dict, optuna.Study]:
    """
    Run Optuna Bayesian hyperparameter optimisation.

    This is the main entry point. It creates an Optuna study, builds
    the objective function (closure over X, y, CV settings), and runs
    the optimisation loop.

    Parameters
    ----------
    X           : Feature DataFrame from assembler.py.
                  (date, ticker) MultiIndex or plain DatetimeIndex.
    y           : Binary target Series (0=DOWN, 1=UP).
    config_name : Preset name from TUNING_CONFIGS.
                  "quick", "default", "thorough", "conservative".
    n_trials    : Override config's n_trials.
    n_splits    : Override config's n_splits.
    gap_days    : Override config's gap_days.
    timeout     : Override config's timeout (seconds).
    metric      : Override config's metric.
    study_name  : Human-readable study name (for logs and saved files).
    seed        : Random seed for reproducibility.
    verbose     : Print progress and final report.

    Returns
    -------
    (best_params, study) tuple where:
      best_params : dict of XGBoost hyperparameters — pass directly
                    to trainer.train(xgb_params=best_params).
      study       : Optuna Study object for inspection/visualisation.

    Example
    -------
    >>> best_params, study = tune(X, y, config_name="default")
    >>> pipeline, results = trainer.train(X, y, xgb_params=best_params)
    """
    # ── Step 1: Load config and apply overrides ───────────────────────────
    config = get_tuning_config(config_name)
    if n_trials is not None:
        config["n_trials"] = n_trials
    if n_splits is not None:
        config["n_splits"] = n_splits
    if gap_days is not None:
        config["gap_days"] = gap_days
    if timeout is not None:
        config["timeout"] = timeout
    if metric is not None:
        config["metric"] = metric

    if study_name is None:
        study_name = f"stocksense_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if verbose:
        print(f"\n{'═'*60}")
        print(f"StockSense AI — Optuna Hyperparameter Tuning")
        print(f"{'═'*60}")
        print(f"  Study    : {study_name}")
        print(f"  Config   : {config_name}")
        print(f"  Trials   : {config['n_trials']}")
        print(f"  CV folds : {config['n_splits']}  |  "
              f"Gap : {config['gap_days']}d")
        print(f"  Metric   : {config['metric']}  (maximise)")
        print(f"  Timeout  : {config['timeout']}s")
        print(f"  Dataset  : {len(X):,} rows × {len(X.columns)} features")
        print(f"{'─'*60}")

    # ── Step 2: Auto-cap CV folds for small datasets ──────────────────────
    # If the dataset doesn't have enough unique dates for the requested
    # number of folds, reduce folds rather than crashing.
    if isinstance(X.index, pd.MultiIndex):
        dates = pd.DatetimeIndex(
            X.index.get_level_values("date")
        ).normalize()
    else:
        dates = pd.DatetimeIndex(X.index).normalize()
    n_unique_dates = len(dates.unique())

    # Each fold needs ~60 unique dates minimum (50 test + gap)
    max_possible_folds = max(1, n_unique_dates // 60 - 1)
    if config["n_splits"] > max_possible_folds:
        original = config["n_splits"]
        config["n_splits"] = max_possible_folds
        if verbose:
            print(f"  ⚠ Auto-capped folds: {original} → "
                  f"{config['n_splits']} (only {n_unique_dates} "
                  f"unique dates available)")

    # ── Step 3: Create Optuna study ───────────────────────────────────────
    # TPESampler: Tree-structured Parzen Estimator — the core Bayesian
    # algorithm. It models P(params | score > threshold) and samples
    # from the "good" distribution, focusing on promising regions.
    #
    # MedianPruner: Kills trials whose intermediate scores fall below
    # the median of completed trials. n_startup_trials=10 means the
    # first 10 trials run to completion (no pruning) to establish
    # a reliable baseline for comparison.
    #
    # n_warmup_steps=1 means pruning can kick in after fold 1 of each
    # trial (don't wait for fold 2).
    sampler = TPESampler(seed=seed)
    pruner  = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=1,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",      # higher AUC-ROC / F1 / accuracy = better
        sampler=sampler,
        pruner=pruner,
    )

    # ── Step 4: Build objective and run ───────────────────────────────────
    objective = _create_objective(
        X, y,
        n_splits=config["n_splits"],
        gap_days=config["gap_days"],
        metric=config["metric"],
    )

    start_time = time.time()

    if verbose:
        print(f"\n  Starting {config['n_trials']} trials...\n")

    # Optuna's optimize() loop:
    # For each trial: sample params → call objective → record score.
    # TPE sampler uses completed trial history to improve sampling.
    # Pruned trials are recorded but marked incomplete.
    study.optimize(
        objective,
        n_trials=config["n_trials"],
        timeout=config["timeout"],
        show_progress_bar=verbose,
    )

    elapsed = time.time() - start_time

    # ── Step 5: Extract best parameters ───────────────────────────────────
    best_params = dict(study.best_trial.params)

    # Add fixed parameters that aren't tuned but needed by build_pipeline.
    # These match DEFAULT_XGB_PARAMS in trainer.py.
    best_params["eval_metric"]          = "logloss"
    best_params["early_stopping_rounds"] = 50
    best_params["random_state"]         = seed
    best_params["verbosity"]            = 0
    best_params["n_jobs"]               = -1

    # ── Step 6: Print report ──────────────────────────────────────────────
    if verbose:
        _print_tuning_report(study, best_params, elapsed, config)

    return best_params, study


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def _print_tuning_report(
    study:       optuna.Study,
    best_params: Dict,
    elapsed:     float,
    config:      Dict,
) -> None:
    """
    Print a formatted summary of the tuning run.

    Shows: convergence info, best trial details, parameter comparison
    vs defaults, and trial statistics (completed/pruned/failed).
    """
    from models.trainer import DEFAULT_XGB_PARAMS

    n_complete = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    n_pruned = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ])
    n_failed = len(study.trials) - n_complete - n_pruned

    print(f"\n{'═'*60}")
    print(f"Tuning Complete")
    print(f"{'═'*60}")
    print(f"  Time elapsed : {elapsed/60:.1f} min")
    print(f"  Total trials : {len(study.trials)}")
    print(f"  Completed    : {n_complete}")
    print(f"  Pruned       : {n_pruned}  "
          f"({n_pruned/max(len(study.trials),1)*100:.0f}% saved)")
    print(f"  Failed       : {n_failed}")
    print(f"  Best {config['metric']:<8}: {study.best_trial.value:.4f}")
    print(f"  Best trial # : {study.best_trial.number}")

    # Parameter comparison: default vs tuned
    print(f"\n{'─'*60}")
    print(f"{'Parameter':<22} {'Default':>12} {'Tuned':>12} {'Change':>10}")
    print(f"{'─'*60}")

    for param in SEARCH_SPACE:
        default_val = DEFAULT_XGB_PARAMS.get(param, "—")
        tuned_val   = best_params.get(param, "—")

        if isinstance(default_val, float) and isinstance(tuned_val, float):
            if default_val != 0:
                pct_change = (tuned_val - default_val) / abs(default_val) * 100
                change_str = f"{pct_change:+.0f}%"
            else:
                change_str = "—"
            print(f"  {param:<20} {default_val:>12.4f} "
                  f"{tuned_val:>12.4f} {change_str:>10}")
        elif isinstance(default_val, int) and isinstance(tuned_val, int):
            diff = tuned_val - default_val
            change_str = f"{diff:+d}"
            print(f"  {param:<20} {default_val:>12} "
                  f"{tuned_val:>12} {change_str:>10}")
        else:
            print(f"  {param:<20} {str(default_val):>12} "
                  f"{str(tuned_val):>12}")

    print(f"{'─'*60}")

    # Convergence: show score progression for top 5 trials
    top_trials = sorted(
        [t for t in study.trials
         if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True,
    )[:5]

    if top_trials:
        print(f"\nTop 5 trials:")
        for t in top_trials:
            print(f"  Trial {t.number:>3}: "
                  f"{config['metric']}={t.value:.4f}  "
                  f"depth={t.params.get('max_depth', '?')}, "
                  f"lr={t.params.get('learning_rate', 0):.4f}, "
                  f"n_est={t.params.get('n_estimators', '?')}")

    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD BEST PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

def save_best_params(
    best_params: Dict,
    study:       optuna.Study,
    name:        str = "best_params",
) -> str:
    """
    Save the best parameters and study summary to JSON.

    Why JSON not pickle?
    ─────────────────────────────────────────────────────────────
    JSON is human-readable — you can inspect best_params.json in
    any text editor or diff tool. Pickle is opaque and version-
    sensitive. The parameters are simple key:value pairs, so JSON
    is the natural format.

    Parameters
    ----------
    best_params : Dict from tune().
    study       : Optuna Study from tune().
    name        : Base filename (without extension).

    Returns
    -------
    Path to saved JSON file.
    """
    os.makedirs(TUNING_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = f"{TUNING_DIR}/{name}_{timestamp}.json"

    # Build a serialisable summary.
    # Optuna Study objects aren't JSON-serialisable, so we extract
    # only the parts we need for reproduction and comparison.
    summary = {
        "best_params":    best_params,
        "best_score":     study.best_trial.value,
        "best_trial":     study.best_trial.number,
        "n_trials":       len(study.trials),
        "n_completed":    len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]),
        "n_pruned":       len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]),
        "study_name":     study.study_name,
        "saved_at":       datetime.now().isoformat(),
    }

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Also save a "latest" symlink for easy access.
    latest_path = f"{TUNING_DIR}/{name}_latest.json"
    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Best params saved:")
    print(f"  Versioned: {filepath}")
    print(f"  Latest:    {latest_path}")

    return filepath


def load_best_params(name: str = "best_params") -> Dict:
    """
    Load the most recently saved best parameters.

    Returns
    -------
    Dict of XGBoost hyperparameters ready for
    trainer.train(xgb_params=params).

    Raises
    ------
    FileNotFoundError if no saved params exist.
    """
    latest_path = f"{TUNING_DIR}/{name}_latest.json"

    if not os.path.exists(latest_path):
        raise FileNotFoundError(
            f"No saved params found at {latest_path}. "
            f"Run tune() and save_best_params() first."
        )

    with open(latest_path, "r") as f:
        summary = json.load(f)

    params = summary["best_params"]
    print(f"Loaded best params (score={summary['best_score']:.4f}, "
          f"trial #{summary['best_trial']})")
    return params


# ══════════════════════════════════════════════════════════════════════════════
#  TUNE AND TRAIN (CONVENIENCE WRAPPER)
# ══════════════════════════════════════════════════════════════════════════════

def tune_and_train(
    X: pd.DataFrame,
    y: pd.Series,
    config_name: str = "default",
    save: bool = True,
    verbose: bool = True,
    **tune_kwargs,
) -> Tuple[Pipeline, Dict, Dict, optuna.Study]:
    """
    Complete pipeline: tune hyperparameters → retrain on full data.

    This is the high-level function that combines tuning + training
    in one call. The typical workflow is:

      1. tune() finds best_params via Bayesian search with CV
      2. trainer.train() retrains on the full train/val/test split
         using those best_params (with early stopping)
      3. Results are saved for experiment tracking

    Parameters
    ----------
    X           : Feature DataFrame.
    y           : Binary target Series.
    config_name : Tuning preset ("quick", "default", "thorough",
                  "conservative").
    save        : Save best params and trained pipeline to disk.
    verbose     : Print progress.
    **tune_kwargs : Additional arguments passed to tune().

    Returns
    -------
    (pipeline, train_results, best_params, study) tuple where:
      pipeline      : Fitted sklearn Pipeline (ready for prediction).
      train_results : Dict from trainer.train() — split info, metrics.
      best_params   : Dict of tuned XGBoost hyperparameters.
      study         : Optuna Study for inspection.
    """
    from models.trainer import train as trainer_train, save_pipeline

    # ── Step 1: Tune ──────────────────────────────────────────────────────
    best_params, study = tune(
        X, y,
        config_name=config_name,
        verbose=verbose,
        **tune_kwargs,
    )

    # ── Step 2: Retrain with best params ──────────────────────────────────
    if verbose:
        print(f"\n{'═'*60}")
        print(f"Retraining with Optuna-tuned parameters...")
        print(f"{'═'*60}")

    pipeline, train_results = trainer_train(
        X, y,
        xgb_params=best_params,
        verbose=verbose,
    )

    # ── Step 3: Save ──────────────────────────────────────────────────────
    if save:
        save_best_params(best_params, study)
        save_pipeline(pipeline, train_results, name="stocksense_tuned")

    return pipeline, train_results, best_params, study


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Quick smoke test: tune AAPL with the "quick" config (20 trials).

    Usage:
        cd backend
        python3 models/tuner.py

    This downloads 2 years of AAPL data, assembles features + labels,
    runs 20 Optuna trials, retrains with best params, and saves.
    Expected runtime: ~3-5 minutes.
    """
    import sys
    sys.path.append("..")

    import yfinance as yf
    from data.cleaner import clean_stock_data
    from features.engineer import build_features
    from features.indicators import get_model_features
    from data.labeller import create_labels

    # ── Fetch and prepare data ────────────────────────────────────────────
    print("Fetching AAPL data...")
    raw = yf.download("AAPL", period="2y", auto_adjust=True, progress=False)
    if raw.empty:
        print("ERROR: Failed to download data. Check network connection.")
        sys.exit(1)

    raw.columns = [c.lower() for c in raw.columns]
    clean    = clean_stock_data(raw, ticker="AAPL")
    featured = build_features(clean).dropna()
    labelled = create_labels(featured, horizon=1,
                             threshold=0.003, verbose=True)
    X = get_model_features(labelled, extra_drop=["target"]).fillna(0)
    y = labelled["target"]

    print(f"\nDataset: {len(X):,} rows × {len(X.columns)} features")
    print(f"Label balance: UP={int((y==1).sum())} "
          f"DOWN={int((y==0).sum())} "
          f"({(y==1).mean()*100:.1f}% / {(y==0).mean()*100:.1f}%)")

    # ── Tune and train ────────────────────────────────────────────────────
    pipeline, results, best_params, study = tune_and_train(
        X, y,
        config_name="quick",
        save=True,
        verbose=True,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"Smoke test complete!")
    print(f"{'═'*60}")
    print(f"  Best AUC-ROC (CV) : {study.best_trial.value:.4f}")
    print(f"  Test accuracy     : "
          f"{results.get('test_accuracy', 'N/A')}")
    print(f"  Test AUC-ROC      : "
          f"{results.get('test_auc_roc', 'N/A')}")
    print(f"\nBest params for trainer.train():")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")
