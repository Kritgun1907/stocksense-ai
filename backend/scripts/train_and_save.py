"""
StockSense AI — scripts/train_and_save.py
==========================================
Train the ML pipeline on multiple stocks and save it for FastAPI.

This is the script you run ONCE (or periodically to retrain).
After this runs successfully, main.py can load the model at startup.

Usage:
    cd backend
    python scripts/train_and_save.py

    # With custom tickers:
    python scripts/train_and_save.py --tickers AAPL MSFT GOOGL NVDA

    # Force retrain even if model exists:
    python scripts/train_and_save.py --force

What this script does:
    1. Downloads 2y of price data for each ticker
    2. Cleans, engineers features, creates labels
    3. Trains XGBoost pipeline with early stopping
    4. Evaluates on held-out test set
    5. Saves pipeline + feature_cols + metadata to disk
    6. Prints a summary so you know if the model is any good
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Make sure backend/ is importable ─────────────────────────────────────────
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
#  Change these to control what gets trained.
# ══════════════════════════════════════════════════════════════════════════════

# Default tickers to train on — diverse sectors for a universal model
DEFAULT_TICKERS = [
    # Tech (large cap)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Finance
    "JPM", "BAC", "GS",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Consumer
    "WMT", "MCD", "COST",
    # Industrial
    "BA", "CAT", "HON",
]

# Where models get saved
MODEL_DIR      = _BACKEND / "models" / "saved"
METADATA_DIR   = MODEL_DIR / "metadata"
SCRIPTS_DIR    = _BACKEND / "scripts"

# Training configuration
PERIOD         = "2y"       # data period — needs 2y for SMA_200 warmup
LABEL_CONFIG   = "default"  # horizon=1 day, threshold=0.3%
GAP_DAYS       = 20         # gap between train and test splits


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def fetch_and_prepare_ticker(ticker: str) -> tuple:
    """
    Full data pipeline for one ticker.
    Returns (X, y, metadata_dict) or (None, None, error_dict) on failure.
    
    Why return None on failure instead of raising?
    ──────────────────────────────────────────────────────────────
    When training on 20 tickers, one bad ticker shouldn't stop
    the entire training run. We skip it and log the reason.
    The training continues with the remaining tickers.
    """
    print(f"  [{ticker}] Fetching {PERIOD} of data...")
    
    try:
        # ── Fetch ──────────────────────────────────────────────────────────
        raw = yf.download(ticker, period=PERIOD,
                         auto_adjust=True, progress=False)
        
        if raw.empty:
            return None, None, {"error": "No data from yfinance"}
        
        # Flatten MultiIndex columns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        
        print(f"  [{ticker}] Downloaded {len(raw)} rows")
        
        # ── Clean ──────────────────────────────────────────────────────────
        from data.cleaner import clean_stock_data
        clean = clean_stock_data(raw, ticker=ticker)
        
        if len(clean) < 250:
            return None, None, {
                "error": f"Too few rows after cleaning: {len(clean)}"
            }
        
        # ── Feature Engineering ────────────────────────────────────────────
        from features.engineer import build_features
        featured = build_features(clean).dropna()
        
        if len(featured) < 100:
            return None, None, {
                "error": f"Too few rows after feature engineering: {len(featured)}"
            }
        
        # ── Labelling ──────────────────────────────────────────────────────
        from data.labeller import create_labels
        labelled = create_labels(
            featured,
            horizon=1,
            threshold=0.003,
            verbose=False,
        )
        
        # ── Get Model Features ─────────────────────────────────────────────
        from features.indicators import get_model_features
        X = get_model_features(labelled, extra_drop=["target"]).fillna(0)
        y = labelled["target"].astype(int)
        
        print(f"  [{ticker}] Ready: {len(X)} rows, {len(X.columns)} features")
        
        return X, y, {
            "ticker":  ticker,
            "n_rows":  len(X),
            "up_pct":  round(float(y.mean()), 4),
        }
        
    except Exception as e:
        return None, None, {"error": str(e)}


def collect_all_data(tickers: list) -> tuple:
    """
    Run fetch_and_prepare_ticker for all tickers.
    Combines results into one big (X, y) for training.
    
    Returns (X_combined, y_combined, stock_metadata_list).
    """
    print(f"\n{'═'*55}")
    print(f"Step 1: Collecting data for {len(tickers)} tickers")
    print(f"{'═'*55}")
    
    all_X      = []
    all_y      = []
    successful = []
    failed     = []
    
    for ticker in tickers:
        X, y, meta = fetch_and_prepare_ticker(ticker)
        
        if X is not None:
            # Add ticker to index for MultiIndex (date, ticker)
            # This matches the assembler.py format
            dates = pd.DatetimeIndex(X.index).normalize()
            X.index = pd.MultiIndex.from_arrays(
                [dates, [ticker] * len(X)],
                names=["date", "ticker"]
            )
            y.index = X.index
            
            all_X.append(X)
            all_y.append(y)
            successful.append(meta)
            print(f"  ✅ {ticker}: {meta['n_rows']} rows, "
                  f"UP rate: {meta['up_pct']*100:.1f}%")
        else:
            failed.append({"ticker": ticker, **meta})
            print(f"  ❌ {ticker}: {meta['error']}")
    
    if not all_X:
        raise RuntimeError(
            "No tickers successfully prepared. "
            "Check your internet connection and ticker symbols."
        )
    
    # Combine all tickers into one DataFrame
    # sort_index() ensures chronological order across all tickers
    X_combined = pd.concat(all_X).sort_index()
    y_combined = pd.concat(all_y).reindex(X_combined.index)
    
    print(f"\n  Summary: {len(successful)}/{len(tickers)} tickers ready")
    print(f"  Combined: {len(X_combined):,} rows × "
          f"{len(X_combined.columns)} features")
    
    return X_combined, y_combined, successful, failed


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train the XGBoost pipeline using trainer.train().
    Returns (pipeline, results_dict).
    
    trainer.train() handles:
    - Three-way chronological split (70/15/15)
    - scale_pos_weight calculation from training labels
    - Early stopping on validation set
    - Evaluation on all three splits
    """
    print(f"\n{'═'*55}")
    print(f"Step 2: Training XGBoost pipeline")
    print(f"{'═'*55}")
    print(f"  Dataset: {len(X):,} rows × {len(X.columns)} features")
    
    from models.trainer import train
    
    pipeline, results = train(X, y, verbose=True)
    
    return pipeline, results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(pipeline, X: pd.DataFrame, y: pd.Series, results: dict) -> dict:
    """
    Run the evaluator.py checks on the trained model.
    Returns a dict of evaluation metrics for metadata.json.
    """
    print(f"\n{'═'*55}")
    print(f"Step 3: Evaluating model")
    print(f"{'═'*55}")
    
    from models.trainer import _three_way_split
    
    # Get test split
    _, _, X_test, _, _, y_test = _three_way_split(X, y)
    
    # Core metrics are already in results from trainer.train()
    test_acc      = results.get("test_accuracy", 0)
    test_auc      = results.get("test_auc_roc", 0)
    baseline      = results.get("majority_baseline",
                    float(max(y.mean(), 1 - y.mean())))
    beats_baseline = test_acc > baseline
    best_iter      = results.get("best_iteration", 0)
    spw            = results.get("scale_pos_weight", 1.0)
    
    # Print verdict
    print(f"\n  Test Accuracy:    {test_acc*100:.2f}%")
    print(f"  Majority Baseline:{baseline*100:.2f}%")
    print(f"  Beats Baseline:   {'✅ Yes' if beats_baseline else '❌ No'}")
    print(f"  Test AUC-ROC:     {test_auc:.4f}")
    print(f"  Best Iteration:   {best_iter}")
    
    if not beats_baseline:
        print(f"\n  ⚠️  WARNING: Model does not beat the majority baseline.")
        print(f"     This can happen with a small dataset or noisy labels.")
        print(f"     Consider adding more tickers or tuning hyperparameters.")
    
    return {
        "test_accuracy":     round(test_acc,  4),
        "test_auc_roc":      round(test_auc,  4),
        "majority_baseline": round(baseline,   4),
        "beats_baseline":    beats_baseline,
        "train_accuracy":    round(results.get("train_accuracy", 0), 4),
        "val_accuracy":      round(results.get("val_accuracy",   0), 4),
        "scale_pos_weight":  round(spw, 4),
        "best_iteration":    best_iter,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: SAVING
# ══════════════════════════════════════════════════════════════════════════════

def save_model(
    pipeline,
    feature_cols:   list,
    results:        dict,
    stock_metadata: list,
    tickers:        list,
    performance:    dict,
) -> str:
    """
    Save the pipeline and all associated metadata to disk.
    
    Files created:
    ─────────────────────────────────────────────────────────────────────
    models/saved/
      stocksense_YYYYMMDD_HHMMSS.pkl   ← versioned pipeline bundle
      stocksense_latest.pkl             ← always points to newest (for FastAPI)
      metadata/
        stocksense_YYYYMMDD_HHMMSS.json ← versioned metadata
        stocksense_latest.json           ← always points to newest
    
    The bundle format:
    ─────────────────────────────────────────────────────────────────────
    The .pkl file contains a dict:
    {
        "pipeline":     fitted sklearn Pipeline object,
        "feature_cols": ["rsi_14", "bb_percent", ...],  (342 items)
    }
    
    This is exactly what models/pipeline.py load_pipeline() expects.
    FastAPI loads it at startup with that function.
    """
    print(f"\n{'═'*55}")
    print(f"Step 4: Saving model")
    print(f"{'═'*55}")
    
    # ── Create directories ────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Generate timestamp-based filenames ───────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name  = f"stocksense_{timestamp}"
    
    versioned_pkl  = MODEL_DIR / f"{model_name}.pkl"
    latest_pkl     = MODEL_DIR / "stocksense_latest.pkl"
    versioned_json = METADATA_DIR / f"{model_name}.json"
    latest_json    = METADATA_DIR / "stocksense_latest.json"
    
    # ── Save the pipeline bundle ──────────────────────────────────────────
    # This is the format that models/pipeline.py:load_pipeline() expects.
    # It saves pipeline AND feature_cols as one unit so they can't get
    # out of sync (loading wrong feature list for wrong model).
    bundle = {
        "pipeline":     pipeline,
        "feature_cols": feature_cols,
    }
    
    print(f"  Saving pipeline bundle...")
    joblib.dump(bundle, versioned_pkl, compress=3)
    joblib.dump(bundle, latest_pkl,    compress=3)  # overwrite latest
    
    pkl_size_mb = versioned_pkl.stat().st_size / 1_048_576
    print(f"  ✅ Pipeline saved: {versioned_pkl.name} ({pkl_size_mb:.1f} MB)")
    print(f"  ✅ Latest updated: {latest_pkl.name}")
    
    # ── Save metadata JSON ────────────────────────────────────────────────
    # This is the human-readable record of this training run.
    # Used for debugging, model comparison, and audit trails.
    xgb_model = pipeline.named_steps["model"]
    
    metadata = {
        "model_name":    model_name,
        "saved_at":      datetime.now().isoformat(),
        "file":          str(versioned_pkl),
        
        "training": {
            "tickers":      tickers,
            "period":       PERIOD,
            "label_config": LABEL_CONFIG,
            "gap_days":     GAP_DAYS,
            "n_features":   len(feature_cols),
            "stocks_used":  stock_metadata,
        },
        
        "performance": performance,
        
        "xgb_params": {
            "n_estimators":    xgb_model.n_estimators,
            "max_depth":       xgb_model.max_depth,
            "learning_rate":   xgb_model.learning_rate,
            "best_iteration":  getattr(xgb_model, "best_iteration", None),
            "best_val_score":  getattr(xgb_model, "best_score", None),
        },
    }
    
    with open(versioned_json, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    with open(latest_json, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"  ✅ Metadata saved: {versioned_json.name}")
    
    return str(versioned_pkl)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_saved_model(pkl_path: str) -> bool:
    """
    Load the saved model back from disk and run a sanity check.
    
    Why verify after saving?
    ──────────────────────────────────────────────────────────────
    joblib.dump() can succeed but produce a corrupted file if
    disk runs out of space mid-write. This verification catches
    that immediately instead of discovering it when FastAPI
    tries to load the model in production.
    
    Returns True if verification passes, False otherwise.
    """
    print(f"\n{'═'*55}")
    print(f"Step 5: Verifying saved model")
    print(f"{'═'*55}")
    
    try:
        # ── Load back from disk ───────────────────────────────────────────
        print(f"  Loading from: {pkl_path}")
        bundle = joblib.load(pkl_path)
        
        pipeline_loaded  = bundle["pipeline"]
        feature_cols_loaded = bundle["feature_cols"]
        
        print(f"  ✅ Load successful")
        print(f"  ✅ Pipeline steps: "
              f"{[s[0] for s in pipeline_loaded.steps]}")
        print(f"  ✅ Feature count: {len(feature_cols_loaded)}")
        
        # ── Sanity prediction test ────────────────────────────────────────
        # Create a dummy row of zeros and run predict_proba
        # We don't care about the prediction value — just that it runs
        # without error and returns a valid probability
        import numpy as np
        dummy_X = pd.DataFrame(
            [np.zeros(len(feature_cols_loaded))],
            columns=feature_cols_loaded,
        )
        proba = pipeline_loaded.predict_proba(dummy_X)[0]
        
        # Probabilities must sum to 1.0 and be in [0, 1]
        proba_sum_ok  = abs(sum(proba) - 1.0) < 0.001
        proba_range_ok = all(0 <= p <= 1 for p in proba)
        
        print(f"  ✅ Dummy prediction: P(DOWN)={proba[0]:.3f}, "
              f"P(UP)={proba[1]:.3f}")
        
        if not proba_sum_ok:
            print(f"  ❌ Probabilities don't sum to 1: {sum(proba)}")
            return False
        
        if not proba_range_ok:
            print(f"  ❌ Probabilities out of range: {proba}")
            return False
        
        print(f"\n  ✅ All verification checks passed")
        print(f"  ✅ Model is ready for FastAPI")
        return True
        
    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main(tickers: list, force: bool = False):
    """
    Run the complete training and saving pipeline.
    """
    print(f"\n{'═'*55}")
    print(f"StockSense AI — Training Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers: {tickers}")
    print(f"{'═'*55}")
    
    # ── Check if model already exists ─────────────────────────────────────
    latest_pkl = MODEL_DIR / "stocksense_latest.pkl"
    if latest_pkl.exists() and not force:
        print(f"\n⚠️  A trained model already exists at: {latest_pkl}")
        print(f"   Use --force to retrain and overwrite it.")
        print(f"   Exiting without retraining.")
        return
    
    import time
    t_start = time.time()
    
    # ── Step 1: Collect data ──────────────────────────────────────────────
    X, y, successful, failed = collect_all_data(tickers)
    
    if len(successful) < 3:
        print(f"\n❌ Only {len(successful)} tickers succeeded. "
              f"Need at least 3 for a meaningful model.")
        return
    
    # ── Step 2: Train ─────────────────────────────────────────────────────
    pipeline, results = train_model(X, y)
    feature_cols = list(X.columns)
    
    # ── Step 3: Evaluate ──────────────────────────────────────────────────
    performance = evaluate_model(pipeline, X, y, results)
    
    # ── Step 4: Save ──────────────────────────────────────────────────────
    pkl_path = save_model(
        pipeline=pipeline,
        feature_cols=feature_cols,
        results=results,
        stock_metadata=successful,
        tickers=tickers,
        performance=performance,
    )
    
    # ── Step 5: Verify ────────────────────────────────────────────────────
    ok = verify_saved_model(pkl_path)
    
    # ── Final Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    
    print(f"\n{'═'*55}")
    print(f"Training Complete")
    print(f"{'═'*55}")
    print(f"  Time elapsed:    {elapsed/60:.1f} minutes")
    print(f"  Stocks trained:  {len(successful)}/{len(tickers)}")
    print(f"  Test accuracy:   {performance['test_accuracy']*100:.2f}%")
    print(f"  Beats baseline:  {'✅' if performance['beats_baseline'] else '❌'}")
    print(f"  Model ready:     {'✅' if ok else '❌ FAILED — check errors above'}")
    print(f"\n  Model saved to:  {pkl_path}")
    print(f"\n  Next step: run the FastAPI server")
    print(f"    uvicorn main:app --reload --port 8000")
    print(f"{'═'*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and save the StockSense AI pipeline"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to train on (default: 20 large-cap stocks)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if a saved model already exists",
    )
    
    args = parser.parse_args()
    main(tickers=args.tickers, force=args.force)