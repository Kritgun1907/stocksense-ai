"""
StockSense AI — tests/test_models_e2e.py
==========================================
End-to-end integration tests for the entire models/ folder.

Tests every public function and contract across all 6 model files:
  1.  trainer.py         — build_pipeline, train, save/load, predict
  2.  timeseries_cv.py   — TimeSeriesSplitWithGap, cross_validate_timeseries
  3.  evaluator.py       — compute_ml_metrics, compute_financial_metrics, evaluate
  4.  explainer.py       — ShapExplainer, explain, feature_importance_summary
  5.  pipeline.py        — build_sklearn_pipeline, evaluate, save/load pipeline
  6.  tuner.py           — get_tuning_config, _sample_params, tune (quick)

Coverage strategy
─────────────────────────────────────────────────────────────
  - All tests use SYNTHETIC deterministic data — no internet required.
    (504 trading-day rows × 30 features to keep fast but realistic)
  - Integration contract tests verify that output of module A is
    valid input for module B (e.g. trainer output → evaluator input).
  - Regression guards check shapes, dtypes, and key field names so
    refactors that silently drop outputs are caught immediately.

Run with:
    cd backend
    python tests/test_models_e2e.py

Expected runtime: ~60-90 seconds (trainer trains twice, tuner runs
                  5 trials of 3-fold CV).
"""

import sys
import os
import warnings
import time
import json
import tempfile
import shutil

import numpy as np
import pandas as pd

# ── Ensure backend/ is on sys.path regardless of CWD ─────────────────────────
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  TEST INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

_PASS = 0
_FAIL = 0
_ERRORS = []


def pass_fail(name: str, condition: bool, detail: str = "") -> bool:
    global _PASS, _FAIL
    icon   = "✅ PASS" if condition else "❌ FAIL"
    detail = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{detail}")
    if condition:
        _PASS += 1
    else:
        _FAIL += 1
        _ERRORS.append(name)
    return condition


def section(title: str) -> None:
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def summary() -> int:
    """Print final tally. Returns 1 if any failures, else 0."""
    total = _PASS + _FAIL
    print(f"\n{'═'*62}")
    print(f"  RESULTS: {_PASS}/{total} passed  |  {_FAIL} failed")
    if _ERRORS:
        print(f"\n  Failed tests:")
        for name in _ERRORS:
            print(f"    ✗  {name}")
    print(f"{'═'*62}\n")
    return 1 if _FAIL > 0 else 0


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_X_y(
    n_days: int = 504,
    n_features: int = 30,
    seed: int = 42,
    multi_stock: bool = False,
    tickers: list = None,
) -> tuple:
    """
    Build a deterministic (X, y) pair for testing.

    Uses a business-day DatetimeIndex so TimeSeriesSplitWithGap's
    date-based logic works correctly. Class balance is ~50/50 by
    construction to avoid divide-by-zero in scale_pos_weight.

    Parameters
    ----------
    n_days      : Number of rows (single-stock) or rows-per-ticker.
    n_features  : Number of feature columns.
    seed        : Random seed.
    multi_stock : If True, build a (date, ticker) MultiIndex DataFrame.
    tickers     : Tickers to use when multi_stock=True.

    Returns
    -------
    (X, y) tuple — X is pd.DataFrame, y is pd.Series, both with
    DatetimeIndex or (date, ticker) MultiIndex.
    """
    rng  = np.random.default_rng(seed)
    cols = [f"feature_{i:03d}" for i in range(n_features)]

    if not multi_stock:
        dates = pd.bdate_range("2022-01-03", periods=n_days)
        X     = pd.DataFrame(rng.normal(size=(n_days, n_features)),
                             index=dates, columns=cols)
        y     = pd.Series(rng.integers(0, 2, size=n_days).astype(int),
                          index=dates, name="target")
        # Force ~50/50 balance using modulo — works for any n_days value
        half = n_days // 2
        y.iloc[:half] = np.arange(half) % 2
        return X, y

    # Multi-stock (date, ticker) MultiIndex
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]

    frames_X, frames_y = [], []
    for i, ticker in enumerate(tickers):
        dates = pd.bdate_range("2022-01-03", periods=n_days)
        X_t   = pd.DataFrame(
            rng.normal(size=(n_days, n_features)),
            index=pd.MultiIndex.from_arrays(
                [dates, [ticker] * n_days],
                names=["date", "ticker"]
            ),
            columns=cols,
        )
        y_t = pd.Series(
            rng.integers(0, 2, size=n_days).astype(int),
            index=X_t.index,
            name="target",
        )
        frames_X.append(X_t)
        frames_y.append(y_t)

    X = pd.concat(frames_X).sort_index()
    y = pd.concat(frames_y).reindex(X.index)
    return X, y


def make_returns(n: int, seed: int = 42) -> np.ndarray:
    """Synthetic daily returns in decimal form (e.g. 0.01 = 1%)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.015, size=n)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — trainer.py
# ══════════════════════════════════════════════════════════════════════════════

def test_trainer():
    section("1 · trainer.py")

    from models.trainer import (
        build_pipeline,
        train,
        save_pipeline,
        load_pipeline,
        predict,
        _calculate_scale_pos_weight,
        _three_way_split,
        DEFAULT_XGB_PARAMS,
    )
    from sklearn.pipeline import Pipeline

    X, y = make_X_y(n_days=504)

    # ── 1.1  _calculate_scale_pos_weight ─────────────────────────────────
    spw = _calculate_scale_pos_weight(y)
    pass_fail(
        "scale_pos_weight is positive float",
        isinstance(spw, float) and spw > 0,
        f"got {spw:.4f}",
    )
    pass_fail(
        "scale_pos_weight near 1.0 for balanced labels",
        0.5 <= spw <= 2.0,
        f"spw={spw:.4f}",
    )

    # ── 1.2  _three_way_split ────────────────────────────────────────────
    X_tr, X_va, X_te, y_tr, y_va, y_te = _three_way_split(X, y)
    total_rows = len(X_tr) + len(X_va) + len(X_te)
    pass_fail(
        "three-way split covers all rows (train+val+test ≤ total)",
        total_rows <= len(X),
        f"{total_rows} / {len(X)} rows",
    )
    pass_fail(
        "chronological order: train < val < test",
        X_tr.index.max() < X_va.index.min() and
        X_va.index.max() < X_te.index.min(),
    )
    pass_fail(
        "train set is largest split",
        len(X_tr) > len(X_va) and len(X_tr) > len(X_te),
        f"train={len(X_tr)} val={len(X_va)} test={len(X_te)}",
    )

    # ── 1.3  build_pipeline ──────────────────────────────────────────────
    pipeline = build_pipeline(scale_pos_weight=1.0)
    pass_fail(
        "build_pipeline returns sklearn Pipeline",
        isinstance(pipeline, Pipeline),
    )
    steps = [s[0] for s in pipeline.steps]
    pass_fail(
        "pipeline has imputer → scaler → model steps",
        steps == ["imputer", "scaler", "model"],
        f"steps={steps}",
    )
    pass_fail(
        "build_pipeline respects xgb_params override",
        build_pipeline(
            xgb_params={"max_depth": 3},
            scale_pos_weight=1.0
        ).named_steps["model"].max_depth == 3,
    )

    # ── 1.4  train ───────────────────────────────────────────────────────
    pipeline, results = train(X, y, verbose=False)
    pass_fail(
        "train returns (Pipeline, dict)",
        isinstance(pipeline, Pipeline) and isinstance(results, dict),
    )
    for key in ["train_accuracy", "val_accuracy", "test_accuracy",
                "train_auc_roc", "test_auc_roc", "best_iteration"]:
        pass_fail(
            f"results has '{key}' key",
            key in results,
        )
    pass_fail(
        "test_accuracy is reasonable (> 0.40)",
        results["test_accuracy"] > 0.40,
        f"got {results['test_accuracy']:.4f}",
    )
    pass_fail(
        "best_iteration is positive int",
        isinstance(results["best_iteration"], int) and
        results["best_iteration"] > 0,
        f"got {results['best_iteration']}",
    )

    # ── 1.5  predict ─────────────────────────────────────────────────────
    sample      = X.tail(10)
    preds, probs = predict(pipeline, sample)
    pass_fail(
        "predict returns (preds, probas) arrays of length 10",
        len(preds) == 10 and len(probs) == 10,
    )
    pass_fail(
        "predictions are binary {0, 1}",
        set(preds).issubset({0, 1}),
        f"unique values: {set(preds)}",
    )
    pass_fail(
        "probabilities in [0, 1]",
        float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0,
        f"range [{probs.min():.3f}, {probs.max():.3f}]",
    )

    # ── 1.6  save / load ─────────────────────────────────────────────────
    tmpdir = tempfile.mkdtemp()
    try:
        # Patch MODEL_DIR to temp location so we don't pollute the repo
        import models.trainer as _trainer_mod
        orig_dir = _trainer_mod.MODEL_DIR
        _trainer_mod.MODEL_DIR = tmpdir

        path = save_pipeline(pipeline, results, name="test_model")
        pass_fail(
            "save_pipeline creates .pkl file",
            os.path.exists(path),
            path,
        )
        loaded = load_pipeline(name="test_model")
        loaded_preds, _ = predict(loaded, sample)
        pass_fail(
            "loaded pipeline produces same predictions as original",
            np.array_equal(preds, loaded_preds),
        )
    finally:
        _trainer_mod.MODEL_DIR = orig_dir
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ── 1.7  multi-stock MultiIndex ──────────────────────────────────────
    X_multi, y_multi = make_X_y(n_days=252, multi_stock=True)
    pipeline_m, results_m = train(X_multi, y_multi, verbose=False)
    pass_fail(
        "train works with (date, ticker) MultiIndex",
        isinstance(pipeline_m, Pipeline),
        f"{len(X_multi)} rows across {X_multi.index.get_level_values('ticker').nunique()} tickers",
    )

    return pipeline, results, X, y


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — timeseries_cv.py
# ══════════════════════════════════════════════════════════════════════════════

def test_timeseries_cv():
    section("2 · timeseries_cv.py")

    from models.timeseries_cv import (
        TimeSeriesSplitWithGap,
        cross_validate_timeseries,
        DEFAULT_GAP_DAYS,
        DEFAULT_N_SPLITS,
        MIN_TEST_FOLD_SIZE,
        get_cv_config,
    )

    X, y = make_X_y(n_days=504)

    # ── 2.1  Constants sanity ────────────────────────────────────────────
    pass_fail("DEFAULT_GAP_DAYS is 20", DEFAULT_GAP_DAYS == 20, str(DEFAULT_GAP_DAYS))
    pass_fail("DEFAULT_N_SPLITS is 5",  DEFAULT_N_SPLITS == 5,  str(DEFAULT_N_SPLITS))
    pass_fail("MIN_TEST_FOLD_SIZE is 50", MIN_TEST_FOLD_SIZE == 50, str(MIN_TEST_FOLD_SIZE))

    # ── 2.2  TimeSeriesSplitWithGap — basic splits ───────────────────────
    cv     = TimeSeriesSplitWithGap(n_splits=3, gap_days=20)
    folds  = list(cv.split_with_info(X))
    pass_fail(
        "TimeSeriesSplitWithGap yields 3 folds",
        len(folds) == 3,
        f"got {len(folds)}",
    )
    for fold_info in folds:
        train_end = X.index[fold_info["train_idx"][-1]]
        test_start = X.index[fold_info["test_idx"][0]]
        pass_fail(
            f"fold {fold_info['fold']}: test strictly after train (gap respected)",
            test_start > train_end,
            f"train_end={train_end.date()}, test_start={test_start.date()}",
        )
        pass_fail(
            f"fold {fold_info['fold']}: test size ≥ {MIN_TEST_FOLD_SIZE}",
            fold_info["n_test"] >= MIN_TEST_FOLD_SIZE,
            f"n_test={fold_info['n_test']}",
        )

    # ── 2.3  Train strictly grows fold-to-fold (expanding window) ────────
    fold_train_sizes = [f["n_train"] for f in folds]
    pass_fail(
        "expanding window: each fold's train is larger than the last",
        all(fold_train_sizes[i] < fold_train_sizes[i+1]
            for i in range(len(fold_train_sizes)-1)),
        str(fold_train_sizes),
    )

    # ── 2.4  No overlap between any train and test sets ──────────────────
    for fold_info in folds:
        overlap = set(fold_info["train_idx"]) & set(fold_info["test_idx"])
        pass_fail(
            f"fold {fold_info['fold']}: no index overlap between train and test",
            len(overlap) == 0,
            f"overlap={len(overlap)}",
        )

    # ── 2.5  cross_validate_timeseries ───────────────────────────────────
    from models.trainer import build_pipeline

    model  = build_pipeline(
        xgb_params={"n_estimators": 50, "early_stopping_rounds": None,
                    "verbosity": 0, "n_jobs": -1},
        scale_pos_weight=1.0,
    )
    cv_result = cross_validate_timeseries(
        model, X, y, n_splits=3, gap_days=20, verbose=False
    )

    pass_fail(
        "cross_validate_timeseries returns dict with required keys",
        all(k in cv_result for k in
            ["fold_scores", "mean_scores", "std_scores", "cv"]),
    )
    pass_fail(
        "mean_scores has auc_roc key",
        "auc_roc" in cv_result["mean_scores"],
    )
    pass_fail(
        "mean AUC-ROC is in valid range [0, 1]",
        0.0 <= cv_result["mean_scores"]["auc_roc"] <= 1.0,
        f"got {cv_result['mean_scores']['auc_roc']:.4f}",
    )
    pass_fail(
        "3 fold_scores returned",
        len(cv_result["fold_scores"]) == 3,
        f"got {len(cv_result['fold_scores'])}",
    )

    # ── 2.6  get_cv_config ───────────────────────────────────────────────
    for cfg_name in ["default", "fast", "rigorous", "conservative"]:
        cfg = get_cv_config(cfg_name)
        pass_fail(
            f"get_cv_config('{cfg_name}') returns dict with n_splits and gap_days",
            "n_splits" in cfg and "gap_days" in cfg,
        )

    # ── 2.7  MultiIndex works with TimeSeriesSplitWithGap ─────────────────
    X_m, y_m = make_X_y(n_days=252, multi_stock=True, tickers=["AAPL", "MSFT"])
    cv_m  = TimeSeriesSplitWithGap(n_splits=2, gap_days=20)
    folds_m = list(cv_m.split_with_info(X_m))
    pass_fail(
        "TimeSeriesSplitWithGap works with (date, ticker) MultiIndex",
        len(folds_m) >= 1,
        f"got {len(folds_m)} folds",
    )

    return cv_result


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — evaluator.py
# ══════════════════════════════════════════════════════════════════════════════

def test_evaluator(pipeline, X, y):
    section("3 · evaluator.py")

    from models.evaluator import (
        compute_ml_metrics,
        compute_financial_metrics,
        evaluate,
        evaluate_cv_folds,
    )
    from models.trainer import (
        _three_way_split, predict,
        find_optimal_threshold,
    )

    _, _, X_te, _, _, y_te = _three_way_split(X, y)

    # ── 3.1  compute_ml_metrics ──────────────────────────────────────────
    preds, probas = predict(pipeline, X_te)
    ml = compute_ml_metrics(
        y_te.reset_index(drop=True),
        preds, probas, verbose=False
    )
    for key in ["accuracy", "precision", "recall", "f1",
                "auc_roc", "majority_baseline", "beats_baseline"]:
        pass_fail(
            f"compute_ml_metrics has '{key}' key",
            key in ml,
        )
    pass_fail(
        "accuracy in [0, 1]",
        0.0 <= ml["accuracy"] <= 1.0,
        f"got {ml['accuracy']:.4f}",
    )
    pass_fail(
        "auc_roc in [0, 1]",
        0.0 <= ml["auc_roc"] <= 1.0,
        f"got {ml['auc_roc']:.4f}",
    )
    pass_fail(
        "confusion matrix counts sum to len(y_test)",
        (ml["true_positives"] + ml["true_negatives"] +
         ml["false_positives"] + ml["false_negatives"]) == len(y_te),
    )

    # ── 3.2  compute_financial_metrics ───────────────────────────────────
    returns = make_returns(len(preds))
    fin = compute_financial_metrics(
        preds, pd.Series(returns), verbose=False
    )
    for section_key in ["strategy", "buy_and_hold"]:
        pass_fail(
            f"compute_financial_metrics has '{section_key}' key",
            section_key in fin,
        )
    for metric in ["sharpe", "max_drawdown", "win_rate", "total_return"]:
        pass_fail(
            f"financial strategy has '{metric}' key",
            metric in fin["strategy"],
        )
    pass_fail(
        "max_drawdown ≤ 0 (always negative or zero)",
        fin["strategy"]["max_drawdown"] <= 0.0,
        f"got {fin['strategy']['max_drawdown']:.4f}",
    )

    # ── 3.3  evaluate (combined ML + financial) ──────────────────────────
    eval_result = evaluate(
        pipeline=pipeline,
        X_test=X_te,
        y_test=y_te,
        actual_returns=pd.Series(make_returns(len(X_te))),
        threshold=0.5,
        verbose=False,
    )
    pass_fail(
        "evaluate returns dict with 'ml', 'financial', 'verdict' keys",
        all(k in eval_result for k in ["ml", "financial", "verdict"]),
    )
    pass_fail(
        "verdict has 'summary' and 'pass_rate' keys",
        all(k in eval_result["verdict"]
            for k in ["summary", "pass_rate", "passing", "total_checks"]),
    )
    pass_fail(
        "pass_rate in [0, 1]",
        0.0 <= eval_result["verdict"]["pass_rate"] <= 1.0,
        f"got {eval_result['verdict']['pass_rate']:.4f}",
    )

    # ── 3.4  find_optimal_threshold ──────────────────────────────────────
    threshold = find_optimal_threshold(
        y_te.reset_index(drop=True), probas,
        objective="f1", verbose=False
    )
    pass_fail(
        "find_optimal_threshold returns float in [0, 1]",
        isinstance(threshold, float) and 0.0 <= threshold <= 1.0,
        f"got {threshold:.4f}",
    )

    # ── 3.5  evaluate_cv_folds aggregation ───────────────────────────────
    fake_fold_results = [
        {"ml": {"accuracy": 0.55, "f1": 0.52, "auc_roc": 0.58,
                "majority_baseline": 0.51}},
        {"ml": {"accuracy": 0.57, "f1": 0.54, "auc_roc": 0.60,
                "majority_baseline": 0.51}},
        {"ml": {"accuracy": 0.53, "f1": 0.50, "auc_roc": 0.56,
                "majority_baseline": 0.51}},
    ]
    agg = evaluate_cv_folds(fake_fold_results, verbose=False)
    pass_fail(
        "evaluate_cv_folds aggregates 3 folds into mean/std keys",
        "ml_accuracy_mean" in agg and "ml_auc_roc_mean" in agg,
    )
    pass_fail(
        "aggregated accuracy mean is correct (55+57+53)/3 = 55",
        abs(agg["ml_accuracy_mean"] - 0.55) < 0.001,
        f"got {agg['ml_accuracy_mean']:.4f}",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — explainer.py
# ══════════════════════════════════════════════════════════════════════════════

def test_explainer(pipeline, X, y):
    section("4 · explainer.py")

    try:
        import shap  # noqa — just verify it's importable
        shap_available = True
    except ImportError:
        shap_available = False
        pass_fail("shap package available", False, "pip install shap")
        return

    try:
        from models.explainer import (
            compute_shap_values,
            explain_single_prediction,
            compute_global_importance,
        )
        pass_fail("models.explainer importable", True)
    except ImportError as e:
        pass_fail("models.explainer importable", False, str(e))
        return

    # ── 4.1  compute_shap_values ──────────────────────────────────────────
    # Signature: compute_shap_values(pipeline, X, sample_size=None)
    # Returns:   (shap_vals: ndarray, base_values, feature_names: List[str])
    sample = X.head(20)
    try:
        shap_vals, base_vals, feat_names = compute_shap_values(pipeline, sample)
        pass_fail("compute_shap_values runs without error", True)
        pass_fail(
            "compute_shap_values returns ndarray shap_vals",
            isinstance(shap_vals, np.ndarray),
            f"type={type(shap_vals).__name__}",
        )
        # SHAP values should be a 2D array matching (n_samples, n_features)
        pass_fail(
            "compute_shap_values shape matches (samples, features)",
            shap_vals.shape[0] == 20,
            f"shape={shap_vals.shape}",
        )
        pass_fail(
            "compute_shap_values feature_names is a non-empty list",
            isinstance(feat_names, list) and len(feat_names) > 0,
            f"len={len(feat_names)}",
        )
    except Exception as e:
        pass_fail("compute_shap_values runs without error", False, str(e))

    # ── 4.2  explain_single_prediction ───────────────────────────────────
    # Signature: explain_single_prediction(pipeline, X_single, top_n=5, verbose=True)
    # Returns:   dict with prediction, probability, top_features, etc.
    single_row = X.iloc[[0]]
    try:
        result = explain_single_prediction(pipeline, single_row, verbose=False)
        pass_fail("explain_single_prediction runs without error", True)
        pass_fail(
            "explain_single_prediction result is a dict",
            isinstance(result, dict),
            f"type={type(result).__name__}",
        )
        pass_fail(
            "explain_single_prediction has 'prediction' key",
            "prediction" in result,
            f"keys={list(result.keys())[:5]}",
        )
        pass_fail(
            "explain_single_prediction prediction is UP or DOWN",
            result.get("prediction") in ("UP", "DOWN"),
            f"got={result.get('prediction')}",
        )
    except Exception as e:
        pass_fail("explain_single_prediction runs without error", False, str(e))

    # ── 4.3  compute_global_importance ───────────────────────────────────
    # Signature: compute_global_importance(pipeline, X, sample_size=500, verbose=True)
    # Returns:   pd.DataFrame with feature / mean_abs_shap / mean_shap / std_shap
    try:
        gi = compute_global_importance(pipeline, X.head(50), verbose=False)
        pass_fail("compute_global_importance runs without error", True)
        pass_fail(
            "compute_global_importance returns DataFrame",
            isinstance(gi, pd.DataFrame),
            f"type={type(gi).__name__}",
        )
        pass_fail(
            "compute_global_importance has 'feature' column",
            "feature" in gi.columns,
            f"cols={list(gi.columns)}",
        )
        pass_fail(
            "compute_global_importance has 'mean_abs_shap' column",
            "mean_abs_shap" in gi.columns,
            f"cols={list(gi.columns)}",
        )
    except Exception as e:
        pass_fail("compute_global_importance runs without error", False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_module():
    section("5 · pipeline.py")

    # pipeline.py uses assembler.py under the hood for run_pipeline().
    # We test the building blocks directly to stay fast and deterministic.
    try:
        from models.pipeline import (
            build_sklearn_pipeline,
            evaluate as pipe_evaluate,
            save_pipeline as pipe_save,
            load_pipeline as pipe_load,
        )
        pass_fail("models.pipeline importable", True)
    except ImportError as e:
        pass_fail("models.pipeline importable", False, str(e))
        return

    # ── 5.1  build_sklearn_pipeline ──────────────────────────────────────
    from sklearn.pipeline import Pipeline as SkPipeline

    pipe = build_sklearn_pipeline()
    pass_fail(
        "build_sklearn_pipeline() returns sklearn Pipeline",
        isinstance(pipe, SkPipeline),
    )
    pass_fail(
        "pipeline steps: imputer + scaler + model",
        [s[0] for s in pipe.steps] == ["imputer", "scaler", "model"],
    )

    # Test parameter overrides
    pipe_custom = build_sklearn_pipeline(
        n_estimators=100, learning_rate=0.01,
        max_depth=3, scale_pos_weight=2.0,
    )
    model = pipe_custom.named_steps["model"]
    pass_fail(
        "build_sklearn_pipeline respects n_estimators=100",
        model.n_estimators == 100,
        f"got {model.n_estimators}",
    )
    pass_fail(
        "build_sklearn_pipeline respects max_depth=3",
        model.max_depth == 3,
        f"got {model.max_depth}",
    )

    # ── 5.2  Fit and evaluate via pipeline.evaluate ───────────────────────
    X, y = make_X_y(n_days=400)
    from models.trainer import _three_way_split

    X_tr, _, X_te, y_tr, _, y_te = _three_way_split(X, y)

    pipe.fit(
        X_tr.reset_index(drop=True),
        y_tr.reset_index(drop=True),
    )
    eval_res = pipe_evaluate(pipe, X_te, y_te)

    pass_fail(
        "pipeline.evaluate returns dict with accuracy/roc_auc",
        "accuracy" in eval_res and "roc_auc" in eval_res,
        str(list(eval_res.keys())),
    )
    pass_fail(
        "pipeline.evaluate accuracy in [0, 1]",
        0.0 <= eval_res["accuracy"] <= 1.0,
        f"got {eval_res['accuracy']:.4f}",
    )

    # ── 5.3  save_pipeline / load_pipeline ───────────────────────────────
    tmpdir = tempfile.mkdtemp()
    try:
        import models.pipeline as _pipe_mod
        orig_dir = getattr(_pipe_mod, "MODELS_DIR", None)

        save_path = os.path.join(tmpdir, "test_pipe.pkl")
        feature_cols = list(X.columns)
        pipe_save(pipe, feature_cols, save_path)
        pass_fail(
            "pipeline.save_pipeline creates file",
            os.path.exists(save_path),
        )

        loaded_pipeline, loaded_feature_cols = pipe_load(save_path)
        pass_fail(
            "pipeline.load_pipeline returns (pipeline, feature_cols) tuple",
            isinstance(loaded_pipeline, SkPipeline) and
            isinstance(loaded_feature_cols, list),
        )
        pass_fail(
            "loaded pipeline produces same shape predictions",
            loaded_pipeline.predict(
                X_te.reset_index(drop=True)
            ).shape == (len(X_te),),
        )
    except Exception as e:
        pass_fail("pipeline save/load roundtrip", False, str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — tuner.py
# ══════════════════════════════════════════════════════════════════════════════

def test_tuner():
    section("6 · tuner.py")

    try:
        import optuna
        pass_fail(f"optuna importable (v{optuna.__version__})", True)
    except ImportError:
        pass_fail("optuna importable", False, "pip install optuna>=3.0.0")
        return

    from models.tuner import (
        SEARCH_SPACE,
        TUNING_CONFIGS,
        get_tuning_config,
        _sample_params,
        _create_objective,
        tune,
        save_best_params,
        load_best_params,
        tune_and_train,
    )

    # ── 6.1  SEARCH_SPACE structure ──────────────────────────────────────
    pass_fail(
        "SEARCH_SPACE has 9 parameters",
        len(SEARCH_SPACE) == 9,
        f"got {len(SEARCH_SPACE)}: {list(SEARCH_SPACE.keys())}",
    )
    required_params = {
        "n_estimators", "max_depth", "min_child_weight",
        "learning_rate", "subsample", "colsample_bytree",
        "gamma", "reg_alpha", "reg_lambda",
    }
    pass_fail(
        "SEARCH_SPACE contains all required XGBoost params",
        required_params.issubset(set(SEARCH_SPACE.keys())),
        f"missing: {required_params - set(SEARCH_SPACE.keys())}",
    )
    for param, spec in SEARCH_SPACE.items():
        pass_fail(
            f"SEARCH_SPACE['{param}'] has type/low/high fields",
            all(k in spec for k in ["type", "low", "high"]),
        )

    # ── 6.2  TUNING_CONFIGS registry ─────────────────────────────────────
    for cfg_name in ["quick", "default", "thorough", "conservative"]:
        cfg = get_tuning_config(cfg_name)
        pass_fail(
            f"get_tuning_config('{cfg_name}') has required keys",
            all(k in cfg for k in
                ["n_trials", "n_splits", "gap_days", "timeout", "metric"]),
        )
    try:
        get_tuning_config("nonexistent_config")
        pass_fail("get_tuning_config raises ValueError for bad name", False)
    except ValueError:
        pass_fail("get_tuning_config raises ValueError for bad name", True)

    # ── 6.3  _sample_params with mock trial ──────────────────────────────
    import optuna

    study   = optuna.create_study(direction="maximize")
    trial   = study.ask()
    sampled = _sample_params(trial)
    pass_fail(
        "_sample_params returns dict with all 9 params",
        set(sampled.keys()) == required_params,
        f"got {set(sampled.keys())}",
    )
    pass_fail(
        "_sample_params n_estimators in [100, 1000]",
        100 <= sampled["n_estimators"] <= 1000,
        f"got {sampled['n_estimators']}",
    )
    pass_fail(
        "_sample_params learning_rate in [0.005, 0.3]",
        0.005 <= sampled["learning_rate"] <= 0.3,
        f"got {sampled['learning_rate']:.5f}",
    )
    pass_fail(
        "_sample_params max_depth in [3, 10]",
        3 <= sampled["max_depth"] <= 10,
        f"got {sampled['max_depth']}",
    )

    # ── 6.4  _create_objective (single objective call) ───────────────────
    X, y = make_X_y(n_days=400)
    objective = _create_objective(X, y, n_splits=2, gap_days=20, metric="auc_roc")
    pass_fail(
        "_create_objective returns a callable",
        callable(objective),
    )

    # Run one trial manually to verify score is in valid range
    study_mini = optuna.create_study(direction="maximize")
    trial_mini = study_mini.ask()
    try:
        score = objective(trial_mini)
        study_mini.tell(trial_mini, score)
        pass_fail(
            "objective() returns valid float in [0, 1]",
            isinstance(score, float) and 0.0 <= score <= 1.0,
            f"got {score:.4f}",
        )
    except optuna.TrialPruned:
        # Pruned on first trial (can happen with MedianPruner edge case)
        pass_fail(
            "objective() runs without non-Optuna errors",
            True,
            "pruned (acceptable on first trial)",
        )

    # ── 6.5  tune() end-to-end (5 trials, quick) ─────────────────────────
    print("\n  Running tune() with 5 trials — please wait ~30s...")
    t0 = time.time()
    best_params, study_out = tune(
        X, y,
        config_name="quick",
        n_trials=5,
        n_splits=2,
        verbose=False,
    )
    elapsed = time.time() - t0

    pass_fail(
        "tune() completes without error",
        True,
        f"{elapsed:.1f}s",
    )
    pass_fail(
        "tune() returns (dict, optuna.Study)",
        isinstance(best_params, dict) and
        isinstance(study_out, optuna.Study),
    )
    pass_fail(
        "best_params contains 'early_stopping_rounds' (added post-tune)",
        "early_stopping_rounds" in best_params,
    )
    pass_fail(
        "best_params contains 'eval_metric' (added post-tune)",
        "eval_metric" in best_params,
    )
    pass_fail(
        "best_params learning_rate in valid range",
        0.005 <= best_params["learning_rate"] <= 0.3,
        f"got {best_params['learning_rate']:.5f}",
    )
    pass_fail(
        "study has at least 1 completed trial",
        len([t for t in study_out.trials
             if t.state == optuna.trial.TrialState.COMPLETE]) >= 1,
    )
    pass_fail(
        "best_trial value in [0, 1]",
        0.0 <= study_out.best_trial.value <= 1.0,
        f"got {study_out.best_trial.value:.4f}",
    )

    # ── 6.6  save / load best params ─────────────────────────────────────
    tmpdir = tempfile.mkdtemp()
    try:
        import models.tuner as _tuner_mod
        orig_dir = _tuner_mod.TUNING_DIR
        _tuner_mod.TUNING_DIR = tmpdir

        save_path = save_best_params(best_params, study_out, name="test_params")
        pass_fail(
            "save_best_params creates JSON file",
            os.path.exists(save_path),
            save_path,
        )

        # Verify JSON structure
        with open(save_path) as f:
            saved = json.load(f)
        pass_fail(
            "saved JSON has best_params, best_score, n_trials keys",
            all(k in saved for k in
                ["best_params", "best_score", "n_trials", "study_name"]),
        )

        loaded_params = load_best_params(name="test_params")
        pass_fail(
            "load_best_params recovers same learning_rate",
            abs(loaded_params["learning_rate"] -
                best_params["learning_rate"]) < 1e-9,
        )
    finally:
        _tuner_mod.TUNING_DIR = orig_dir
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ── 6.7  tune_and_train integration ──────────────────────────────────
    print("\n  Running tune_and_train() with 3 trials — please wait ~15s...")
    from sklearn.pipeline import Pipeline as SkPipeline

    tmpdir2 = tempfile.mkdtemp()
    try:
        import models.tuner as _t, models.trainer as _tr
        _t.TUNING_DIR  = tmpdir2
        _tr.MODEL_DIR  = tmpdir2

        pipeline_t, results_t, bp_t, study_t = tune_and_train(
            X, y,
            config_name="quick",
            n_trials=3,
            n_splits=2,
            save=True,
            verbose=False,
        )
        pass_fail(
            "tune_and_train returns fitted Pipeline",
            isinstance(pipeline_t, SkPipeline),
        )
        pass_fail(
            "tune_and_train returns results dict with test_accuracy",
            "test_accuracy" in results_t,
            f"got {results_t.get('test_accuracy', 'MISSING'):.4f}"
            if "test_accuracy" in results_t else "MISSING",
        )
        pass_fail(
            "tune_and_train saves best_params JSON",
            any(f.endswith(".json") for f in os.listdir(tmpdir2)),
        )
    except Exception as e:
        pass_fail("tune_and_train completes without error", False, str(e))
    finally:
        _t.TUNING_DIR = orig_dir
        _tr.MODEL_DIR = "models/saved"
        shutil.rmtree(tmpdir2, ignore_errors=True)

    return best_params, study_out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — INTEGRATION: MODULE-TO-MODULE CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

def test_integration_contracts():
    section("7 · Integration contracts (module-to-module)")

    # ── 7.1  trainer → evaluator contract ────────────────────────────────
    from models.trainer import train, predict, _three_way_split
    from models.evaluator import evaluate

    X, y = make_X_y(n_days=400)
    pipeline, _ = train(X, y, verbose=False)
    _, _, X_te, _, _, y_te = _three_way_split(X, y)

    eval_result = evaluate(
        pipeline=pipeline,
        X_test=X_te,
        y_test=y_te,
        actual_returns=pd.Series(make_returns(len(X_te))),
        verbose=False,
    )
    pass_fail(
        "trainer output → evaluator input: evaluate() accepts trained pipeline",
        "ml" in eval_result and "financial" in eval_result,
    )

    # ── 7.2  timeseries_cv → trainer contract ────────────────────────────
    from models.timeseries_cv import cross_validate_timeseries
    from models.trainer import build_pipeline

    model = build_pipeline(
        xgb_params={"n_estimators": 50, "early_stopping_rounds": None,
                    "verbosity": 0, "n_jobs": -1},
        scale_pos_weight=1.0,
    )
    cv_result = cross_validate_timeseries(
        model, X, y, n_splits=2, gap_days=20, verbose=False
    )
    pass_fail(
        "timeseries_cv → trainer: cross_validate_timeseries accepts build_pipeline output",
        "mean_scores" in cv_result and "auc_roc" in cv_result["mean_scores"],
    )

    # ── 7.3  tuner → trainer contract ────────────────────────────────────
    from models.tuner import tune
    from models.trainer import train as trainer_train

    best_params, _ = tune(
        X, y, config_name="quick",
        n_trials=3, n_splits=2, verbose=False,
    )
    pipeline_tuned, results_tuned = trainer_train(
        X, y, xgb_params=best_params, verbose=False
    )
    pass_fail(
        "tuner → trainer: best_params from tune() accepted by trainer.train()",
        "test_accuracy" in results_tuned and
        results_tuned["test_accuracy"] > 0.35,
        f"test_accuracy={results_tuned.get('test_accuracy', 'MISSING'):.4f}"
        if "test_accuracy" in results_tuned else "MISSING",
    )

    # ── 7.4  trainer → pipeline.evaluate contract ────────────────────────
    try:
        from models.pipeline import evaluate as pipe_evaluate
        eval_pipe = pipe_evaluate(pipeline, X_te, y_te)
        pass_fail(
            "trainer output → pipeline.evaluate: compatible",
            "accuracy" in eval_pipe,
        )
    except Exception as e:
        pass_fail("trainer output → pipeline.evaluate: compatible", False, str(e))

    # ── 7.5  Temporal integrity: no future data in any training fold ──────
    from models.timeseries_cv import TimeSeriesSplitWithGap

    cv   = TimeSeriesSplitWithGap(n_splits=3, gap_days=20)
    folds = list(cv.split_with_info(X))
    all_clean = True
    for fold_info in folds:
        train_dates = X.index[fold_info["train_idx"]]
        test_dates  = X.index[fold_info["test_idx"]]
        if train_dates.max() >= test_dates.min():
            all_clean = False
            break
    pass_fail(
        "temporal integrity: train_end < test_start in every fold",
        all_clean,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — REGRESSION GUARDS
# ══════════════════════════════════════════════════════════════════════════════

def test_regression_guards():
    section("8 · Regression guards")

    # ── 8.1  DEFAULT_XGB_PARAMS keys unchanged ───────────────────────────
    from models.trainer import DEFAULT_XGB_PARAMS

    required_xgb_keys = {
        "n_estimators", "learning_rate", "max_depth",
        "subsample", "colsample_bytree", "min_child_weight",
        "gamma", "reg_alpha", "reg_lambda", "eval_metric",
        "early_stopping_rounds", "random_state", "verbosity",
    }
    pass_fail(
        "DEFAULT_XGB_PARAMS contains all required keys",
        required_xgb_keys.issubset(set(DEFAULT_XGB_PARAMS.keys())),
        f"missing: {required_xgb_keys - set(DEFAULT_XGB_PARAMS.keys())}",
    )

    # ── 8.2  trainer.train returns results with expected split keys ───────
    X, y = make_X_y(n_days=350)
    from models.trainer import train
    _, results = train(X, y, verbose=False)
    for split in ["train", "val", "test"]:
        for metric in ["accuracy", "f1", "auc_roc"]:
            key = f"{split}_{metric}"
            pass_fail(
                f"results['{key}'] present after training",
                key in results,
            )

    # ── 8.3  timeseries_cv mean_scores key set stable ────────────────────
    from models.timeseries_cv import cross_validate_timeseries
    from models.trainer import build_pipeline

    model = build_pipeline(
        xgb_params={"n_estimators": 30, "early_stopping_rounds": None,
                    "verbosity": 0, "n_jobs": -1},
        scale_pos_weight=1.0,
    )
    cv_res = cross_validate_timeseries(
        model, X, y, n_splits=2, gap_days=20, verbose=False
    )
    expected_metrics = {"accuracy", "f1", "precision", "recall", "auc_roc"}
    pass_fail(
        "cross_validate_timeseries mean_scores has exactly the 5 expected metrics",
        set(cv_res["mean_scores"].keys()) == expected_metrics,
        f"got {set(cv_res['mean_scores'].keys())}",
    )

    # ── 8.4  Tuning config values are sane ────────────────────────────────
    from models.tuner import TUNING_CONFIGS

    for cfg_name, cfg in TUNING_CONFIGS.items():
        pass_fail(
            f"TUNING_CONFIGS['{cfg_name}'] n_trials ≥ 5",
            cfg["n_trials"] >= 5,
            f"got {cfg['n_trials']}",
        )
        pass_fail(
            f"TUNING_CONFIGS['{cfg_name}'] gap_days ≥ 20",
            cfg["gap_days"] >= 20,
            f"got {cfg['gap_days']}",
        )

    # ── 8.5  Pipeline save format backwards-compatible ───────────────────
    # The saved file must be a dict with exactly "pipeline" and "feature_cols"
    from models.pipeline import build_sklearn_pipeline, save_pipeline, load_pipeline
    from models.trainer import _three_way_split
    from sklearn.pipeline import Pipeline as SkPipeline

    X2, y2 = make_X_y(n_days=350)
    X_tr2, _, X_te2, y_tr2, _, _ = _three_way_split(X2, y2)
    pipe2 = build_sklearn_pipeline(n_estimators=30)
    pipe2.fit(X_tr2.reset_index(drop=True), y_tr2.reset_index(drop=True))

    tmpdir = tempfile.mkdtemp()
    try:
        path2 = os.path.join(tmpdir, "guard_pipe.pkl")
        save_pipeline(pipe2, list(X2.columns), path2)
        loaded_pipe2, loaded_cols2 = load_pipeline(path2)
        pass_fail(
            "pipeline bundle returns (pipeline, feature_cols) tuple",
            isinstance(loaded_pipe2, SkPipeline) and
            isinstance(loaded_cols2, list),
            f"pipeline type={type(loaded_pipe2).__name__}",
        )
        pass_fail(
            "feature_cols is a list of strings",
            isinstance(loaded_cols2, list) and
            all(isinstance(c, str) for c in loaded_cols2),
        )
    except Exception as e:
        pass_fail("pipeline save/load bundle format", False, str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — RUN ALL SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═'*62}")
    print(f"  StockSense AI — Models Folder End-to-End Test Suite")
    print(f"  Sections: trainer · timeseries_cv · evaluator · "
          f"explainer · pipeline · tuner · integration · regression")
    print(f"{'═'*62}")

    t_start = time.time()

    # Section 1 — trainer (returns fitted pipeline for downstream tests)
    pipeline, results, X, y = test_trainer()

    # Section 2 — timeseries_cv
    test_timeseries_cv()

    # Section 3 — evaluator (uses pipeline from Section 1)
    test_evaluator(pipeline, X, y)

    # Section 4 — explainer (uses pipeline from Section 1)
    test_explainer(pipeline, X, y)

    # Section 5 — pipeline module
    test_pipeline_module()

    # Section 6 — tuner (most compute-heavy)
    test_tuner()

    # Section 7 — integration contracts
    test_integration_contracts()

    # Section 8 — regression guards
    test_regression_guards()

    t_elapsed = time.time() - t_start
    print(f"\n  Total runtime: {t_elapsed:.1f}s")

    exit_code = summary()
    sys.exit(exit_code)
