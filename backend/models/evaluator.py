"""
StockSense AI — models/evaluator.py
=====================================
Model evaluation: ML metrics + financial metrics + backtesting.

This file owns:
  - ML classification metrics (accuracy, F1, AUC-ROC etc.)
  - Financial performance metrics (Sharpe, drawdown, win rate)
  - Simple backtesting simulation
  - Benchmark comparison (vs buy-and-hold, vs random)
  - Evaluation report generation

It does NOT own:
  - Model training          → models/trainer.py
  - Cross-validation        → models/timeseries_cv.py
  - Hyperparameter tuning   → models/tuner.py (Chapter 4.6)
  - SHAP explainability     → models/explainer.py (Chapter 4.5)

Why both ML and financial metrics?
─────────────────────────────────────────────────────────────
  ML metrics (accuracy, F1) measure classification quality.
  Financial metrics (Sharpe, drawdown) measure trading value.
  A model can have 58% accuracy but lose money (predicts small moves).
  A model can have 53% accuracy but make money (predicts large moves).
  Both lenses are necessary to fully evaluate a trading model.

Why include a buy-and-hold benchmark?
─────────────────────────────────────────────────────────────
  The simplest possible strategy is to buy the stock and hold it.
  If your model cannot beat this benchmark, it has no trading value
  regardless of ML metrics. The benchmark establishes the minimum
  bar your model must clear to be worth deploying.

Why majority class baseline?
─────────────────────────────────────────────────────────────
  Predicting the majority class every day achieves non-zero accuracy
  without learning anything. Your model must beat this floor to
  demonstrate it has actually learned patterns, not just memorised
  the class distribution.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    precision_recall_curve, confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

TRADING_DAYS_PER_YEAR = 252

# Sharpe ratio interpretation thresholds.
# These are industry-standard benchmarks.
SHARPE_LABELS = [
    (2.0,  "Excellent (or overfitted — verify with OOS data)"),
    (1.0,  "Good — institutional quality"),
    (0.5,  "Acceptable"),
    (0.0,  "Poor — strategy loses on risk-adjusted basis"),
    (-999, "Negative — strategy destroys value"),
]

# Drawdown severity thresholds.
MAX_DRAWDOWN_LABELS = [
    (-0.05,  "Low risk  (<5%)"),
    (-0.15,  "Moderate  (5-15%)"),
    (-0.30,  "High risk (15-30%)"),
    (-999,   "Extreme   (>30%) — most investors would stop"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  ML METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_ml_metrics(
    y_true:     pd.Series,
    y_pred:     np.ndarray,
    y_proba:    Optional[np.ndarray] = None,
    threshold:  float = 0.5,
    verbose:    bool  = True,
) -> Dict:
    """
    Compute standard ML classification metrics.

    Parameters
    ----------
    y_true    : True binary labels (0/1).
    y_pred    : Predicted binary labels (0/1).
    y_proba   : Predicted probabilities for class 1 (UP).
                Required for AUC-ROC. If None, AUC-ROC is skipped.
    threshold : Classification threshold used (for reporting only).
    verbose   : Print formatted results.

    Returns
    -------
    Dict with all computed metrics.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Majority class baseline
    majority_baseline = max(y_true_arr.mean(), 1 - y_true_arr.mean())

    metrics = {
        "accuracy":          round(accuracy_score(y_true_arr, y_pred_arr), 4),
        "precision":         round(precision_score(y_true_arr, y_pred_arr,
                                                   zero_division=0), 4),
        "recall":            round(recall_score(y_true_arr, y_pred_arr,
                                                zero_division=0), 4),
        "f1":                round(f1_score(y_true_arr, y_pred_arr,
                                            zero_division=0), 4),
        "majority_baseline": round(majority_baseline, 4),
        "beats_baseline":    accuracy_score(y_true_arr, y_pred_arr) > majority_baseline,
        "threshold_used":    threshold,
    }

    if y_proba is not None:
        try:
            metrics["auc_roc"] = round(roc_auc_score(y_true_arr, y_proba), 4)
        except ValueError:
            metrics["auc_roc"] = 0.5

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()
    metrics["true_positives"]  = int(tp)
    metrics["true_negatives"]  = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    if verbose:
        _print_ml_metrics(metrics)

    return metrics


def _print_ml_metrics(metrics: Dict) -> None:
    """Format and print ML metrics."""
    beat = "✅" if metrics["beats_baseline"] else "❌"
    print(f"\n{'─'*50}")
    print(f"ML Metrics (threshold={metrics['threshold_used']:.3f})")
    print(f"{'─'*50}")
    print(f"  Accuracy:          {metrics['accuracy']*100:>7.2f}%  "
          f"{beat} (baseline: {metrics['majority_baseline']*100:.1f}%)")
    print(f"  Precision:         {metrics['precision']*100:>7.2f}%")
    print(f"  Recall:            {metrics['recall']*100:>7.2f}%")
    print(f"  F1 Score:          {metrics['f1']:>8.4f}")
    if "auc_roc" in metrics:
        print(f"  AUC-ROC:           {metrics['auc_roc']:>8.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True  UP:  {metrics['true_positives']:>5}  "
          f"False UP:  {metrics['false_positives']:>5}")
    print(f"    False DOWN:{metrics['false_negatives']:>5}  "
          f"True  DOWN:{metrics['true_negatives']:>5}")


# ══════════════════════════════════════════════════════════════════════════════
#  FINANCIAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_financial_metrics(
    y_pred:          np.ndarray,
    actual_returns:  np.ndarray,
    allow_short:     bool  = False,
    transaction_cost: float = 0.001,
    verbose:         bool  = True,
) -> Dict:
    """
    Compute financial performance metrics from predictions and actual returns.

    Parameters
    ----------
    y_pred           : Predicted labels (0=DOWN, 1=UP).
    actual_returns   : Actual next-day returns corresponding to each prediction.
                       Must be in decimal form (0.01 = 1% return).
    allow_short      : If True, predict DOWN → short the stock.
                       If False, predict DOWN → sit out (cash).
    transaction_cost : Round-trip cost per trade (0.001 = 0.1%).
                       Applied to all trades.
    verbose          : Print formatted results.

    Returns
    -------
    Dict with Sharpe, drawdown, win rate, and other financial metrics.

    Why default allow_short=False?
    ─────────────────────────────────────────────────────────────
    Short selling requires margin, has overnight risk, and may not
    be available for all stocks. The conservative default simulates
    a long-only strategy — the most common retail trading setup.

    Why include transaction costs?
    ─────────────────────────────────────────────────────────────
    Backtests without transaction costs are systematically optimistic.
    0.1% round-trip is conservative for retail (interactive brokers
    charges ~0.005%). Always subtract costs for honest evaluation.
    """
    y_pred         = np.array(y_pred)
    actual_returns = np.array(actual_returns)

    # Calculate strategy returns
    strategy_returns = _compute_strategy_returns(
        y_pred, actual_returns, allow_short, transaction_cost
    )

    # Calculate buy-and-hold benchmark
    bh_returns = actual_returns - transaction_cost  # one initial purchase

    metrics = {
        "strategy":    _financial_stats(strategy_returns, "strategy"),
        "buy_and_hold": _financial_stats(bh_returns, "buy_and_hold"),
    }

    # Add comparison metrics
    metrics["alpha"] = round(
        metrics["strategy"]["annualised_return"] -
        metrics["buy_and_hold"]["annualised_return"], 4
    )
    metrics["beats_buy_and_hold"] = (
        metrics["strategy"]["sharpe"] >
        metrics["buy_and_hold"]["sharpe"]
    )
    metrics["n_trades"] = int(y_pred.sum())
    metrics["trade_rate"] = round(float(y_pred.mean()), 4)

    if verbose:
        _print_financial_metrics(metrics)

    return metrics


def _compute_strategy_returns(
    y_pred:          np.ndarray,
    actual_returns:  np.ndarray,
    allow_short:     bool,
    transaction_cost: float,
) -> np.ndarray:
    """
    Compute daily strategy returns based on predictions.

    Long-only:  UP signal → get actual return - cost
                DOWN signal → 0 (in cash)

    Long-short: UP signal → get actual return - cost
                DOWN signal → get NEGATIVE actual return - cost
    """
    strategy = np.zeros(len(y_pred))

    for i, (pred, ret) in enumerate(zip(y_pred, actual_returns)):
        if pred == 1:
            strategy[i] = ret - transaction_cost
        elif allow_short:
            strategy[i] = -ret - transaction_cost
        # else: 0 (cash, no return, no cost)

    return strategy


def _financial_stats(returns: np.ndarray, name: str) -> Dict:
    """
    Calculate financial statistics for a returns series.
    Used for both strategy and benchmark calculations.
    """
    returns       = np.array(returns)
    non_zero      = returns[returns != 0] if name == "strategy" else returns
    trading_count = len(non_zero) if len(non_zero) > 0 else 1

    # Cumulative returns
    cum_returns = pd.Series((1 + returns).cumprod())

    # Sharpe Ratio
    mean_ret = np.mean(non_zero)
    std_ret  = np.std(non_zero)
    sharpe   = (mean_ret / (std_ret + 1e-10)) * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Maximum Drawdown
    peak      = cum_returns.cummax()
    drawdowns = (cum_returns - peak) / (peak + 1e-10)
    max_dd    = float(drawdowns.min())

    # Annualised return (compound)
    total_return     = float(cum_returns.iloc[-1] - 1)
    n_years          = len(returns) / TRADING_DAYS_PER_YEAR
    annualised_return = (1 + total_return) ** (1/n_years) - 1

    # Win rate and profit factor
    wins   = non_zero[non_zero > 0]
    losses = non_zero[non_zero < 0]
    win_rate     = len(wins) / trading_count
    profit_factor = (wins.sum() / (-losses.sum() + 1e-10)
                     if len(losses) > 0 else float('inf'))

    # Calmar ratio
    calmar = annualised_return / (-max_dd + 1e-10) if max_dd < 0 else 0.0

    return {
        "total_return":      round(total_return, 4),
        "annualised_return": round(annualised_return, 4),
        "sharpe":            round(float(sharpe), 4),
        "max_drawdown":      round(max_dd, 4),
        "win_rate":          round(win_rate, 4),
        "profit_factor":     round(min(profit_factor, 999.0), 4),
        "calmar":            round(float(calmar), 4),
        "n_days":            len(returns),
    }


def _sharpe_label(sharpe: float) -> str:
    for threshold, label in SHARPE_LABELS:
        if sharpe >= threshold:
            return label
    return SHARPE_LABELS[-1][1]


def _drawdown_label(max_dd: float) -> str:
    for threshold, label in MAX_DRAWDOWN_LABELS:
        if max_dd >= threshold:
            return label
    return MAX_DRAWDOWN_LABELS[-1][1]


def _print_financial_metrics(metrics: Dict) -> None:
    """Format and print financial metrics comparison."""
    strat = metrics["strategy"]
    bh    = metrics["buy_and_hold"]
    beat  = "✅" if metrics["beats_buy_and_hold"] else "❌"

    print(f"\n{'─'*60}")
    print(f"Financial Metrics")
    print(f"{'─'*60}")
    print(f"{'Metric':<22} {'Strategy':>12} {'Buy & Hold':>12}")
    print(f"{'─'*60}")
    print(f"{'Total Return':<22} {strat['total_return']*100:>11.2f}%"
          f" {bh['total_return']*100:>11.2f}%")
    print(f"{'Annual Return':<22} {strat['annualised_return']*100:>11.2f}%"
          f" {bh['annualised_return']*100:>11.2f}%")
    print(f"{'Sharpe Ratio':<22} {strat['sharpe']:>12.3f}"
          f" {bh['sharpe']:>12.3f}  {beat}")
    print(f"{'Max Drawdown':<22} {strat['max_drawdown']*100:>11.2f}%"
          f" {bh['max_drawdown']*100:>11.2f}%")
    print(f"{'Win Rate':<22} {strat['win_rate']*100:>11.1f}%"
          f" {bh['win_rate']*100:>11.1f}%")
    print(f"{'Profit Factor':<22} {strat['profit_factor']:>12.3f}"
          f" {bh['profit_factor']:>12.3f}")
    print(f"{'Calmar Ratio':<22} {strat['calmar']:>12.3f}"
          f" {bh['calmar']:>12.3f}")
    print(f"{'─'*60}")
    print(f"  Alpha vs Buy & Hold: {metrics['alpha']*100:+.2f}%/year")
    print(f"  Trades: {metrics['n_trades']} / {strat['n_days']} days "
          f"({metrics['trade_rate']*100:.1f}% activity)")
    print(f"\n  Strategy Sharpe: {_sharpe_label(strat['sharpe'])}")
    print(f"  Max Drawdown:    {_drawdown_label(strat['max_drawdown'])}")


# ══════════════════════════════════════════════════════════════════════════════
#  FULL EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    pipeline:        Pipeline,
    X_test:          pd.DataFrame,
    y_test:          pd.Series,
    actual_returns:  Optional[np.ndarray] = None,
    threshold:       float = 0.5,
    allow_short:     bool  = False,
    transaction_cost: float = 0.001,
    verbose:         bool  = True,
) -> Dict:
    """
    Complete model evaluation: ML metrics + financial metrics.

    Parameters
    ----------
    pipeline        : Fitted sklearn Pipeline from trainer.py.
    X_test          : Test feature DataFrame.
    y_test          : Test target Series.
    actual_returns  : Actual next-day returns for backtesting.
                      If None, financial metrics are skipped.
    threshold       : Classification threshold (from trainer.find_optimal_threshold).
    allow_short     : Short sell on DOWN predictions.
    transaction_cost: Round-trip cost per trade.
    verbose         : Print full evaluation report.

    Returns
    -------
    Dict combining ML metrics, financial metrics, and summary flags.
    """
    from models.trainer import predict_with_threshold

    if verbose:
        print(f"\n{'═'*60}")
        print(f"Model Evaluation Report")
        print(f"{'═'*60}")
        print(f"Test set: {len(X_test):,} rows | "
              f"Threshold: {threshold:.3f}")

    # ── Generate predictions
    y_pred, y_proba = predict_with_threshold(pipeline, X_test, threshold)

    # ── ML metrics
    ml = compute_ml_metrics(
        y_test.reset_index(drop=True),
        y_pred,
        y_proba,
        threshold=threshold,
        verbose=verbose,
    )

    results = {"ml": ml}

    # ── Financial metrics (if returns provided)
    if actual_returns is not None:
        fin = compute_financial_metrics(
            y_pred,
            np.array(actual_returns),
            allow_short=allow_short,
            transaction_cost=transaction_cost,
            verbose=verbose,
        )
        results["financial"] = fin

    # ── Summary verdict
    verdict = _generate_verdict(ml, results.get("financial"))
    results["verdict"] = verdict

    if verbose:
        print(f"\n{'═'*60}")
        print(f"VERDICT: {verdict['summary']}")
        for point in verdict["points"]:
            print(f"  {point}")
        print(f"{'═'*60}\n")

    return results


def _generate_verdict(
    ml_metrics: Dict,
    fin_metrics: Optional[Dict],
) -> Dict:
    """
    Generate a human-readable verdict on model quality.
    Used in the FastAPI response and the StockSense UI.
    """
    points  = []
    passing = 0
    total   = 0

    # ML checks
    total += 1
    if ml_metrics["beats_baseline"]:
        points.append(f"✅ Beats majority baseline "
                      f"({ml_metrics['accuracy']*100:.1f}% vs "
                      f"{ml_metrics['majority_baseline']*100:.1f}%)")
        passing += 1
    else:
        points.append(f"❌ Does not beat majority baseline — "
                      f"model has no real predictive power")

    total += 1
    if ml_metrics.get("auc_roc", 0.5) > 0.55:
        points.append(f"✅ AUC-ROC {ml_metrics['auc_roc']:.4f} > 0.55 "
                      f"— real signal detected")
        passing += 1
    else:
        points.append(f"❌ AUC-ROC {ml_metrics.get('auc_roc', 0.5):.4f} ≤ 0.55 "
                      f"— model struggles to rank predictions")

    total += 1
    if ml_metrics["f1"] > 0.50:
        points.append(f"✅ F1 {ml_metrics['f1']:.4f} > 0.50 "
                      f"— reasonable precision/recall balance")
        passing += 1
    else:
        points.append(f"⚠️  F1 {ml_metrics['f1']:.4f} ≤ 0.50 "
                      f"— consider threshold adjustment")

    # Financial checks
    if fin_metrics:
        strat = fin_metrics["strategy"]

        total += 1
        if fin_metrics["beats_buy_and_hold"]:
            points.append(f"✅ Beats buy-and-hold "
                          f"(Sharpe {strat['sharpe']:.2f} vs "
                          f"{fin_metrics['buy_and_hold']['sharpe']:.2f})")
            passing += 1
        else:
            points.append(f"❌ Does not beat buy-and-hold — "
                          f"simplest benchmark not cleared")

        total += 1
        if strat["sharpe"] > 0.5:
            points.append(f"✅ Sharpe {strat['sharpe']:.2f} > 0.5 "
                          f"— acceptable risk-adjusted return")
            passing += 1
        else:
            points.append(f"❌ Sharpe {strat['sharpe']:.2f} ≤ 0.5 "
                          f"— poor risk-adjusted return")

        total += 1
        if strat["max_drawdown"] > -0.20:
            points.append(f"✅ Max drawdown {strat['max_drawdown']*100:.1f}% "
                          f"— within acceptable range")
            passing += 1
        else:
            points.append(f"⚠️  Max drawdown {strat['max_drawdown']*100:.1f}% "
                          f"— high risk, consider position sizing")

    # Summary
    pct = passing / total if total > 0 else 0
    if pct >= 0.80:
        summary = "🟢 STRONG — Model ready for paper trading"
    elif pct >= 0.60:
        summary = "🟡 ACCEPTABLE — Consider tuning before deployment"
    elif pct >= 0.40:
        summary = "🟠 WEAK — Significant improvements needed"
    else:
        summary = "🔴 FAILING — Model does not demonstrate real predictive power"

    return {
        "summary":      summary,
        "points":       points,
        "passing":      passing,
        "total_checks": total,
        "pass_rate":    round(pct, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION FINANCIAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_cv_folds(
    fold_results: List[Dict],
    verbose:      bool = True,
) -> Dict:
    """
    Aggregate evaluation results across cross-validation folds.
    Called after cross_validate_timeseries() in timeseries_cv.py.

    Parameters
    ----------
    fold_results : List of per-fold result dicts from evaluate().
    verbose      : Print aggregated results.

    Returns
    -------
    Dict with mean ± std for all metrics across folds.
    """
    if not fold_results:
        return {}

    all_metrics = {}

    # Collect all numeric metrics
    for result in fold_results:
        ml  = result.get("ml", {})
        fin = result.get("financial", {})

        for key, val in ml.items():
            if isinstance(val, (int, float)):
                all_metrics.setdefault(f"ml_{key}", []).append(val)

        if fin:
            for key, val in fin.get("strategy", {}).items():
                if isinstance(val, (int, float)):
                    all_metrics.setdefault(f"fin_{key}", []).append(val)

    # Compute mean ± std
    aggregated = {}
    for metric, values in all_metrics.items():
        aggregated[f"{metric}_mean"] = round(np.mean(values), 4)
        aggregated[f"{metric}_std"]  = round(np.std(values),  4)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"Cross-Validation Evaluation Summary ({len(fold_results)} folds)")
        print(f"{'═'*60}")

        key_metrics = [
            "ml_accuracy", "ml_f1", "ml_auc_roc",
            "fin_sharpe", "fin_max_drawdown", "fin_win_rate"
        ]
        for metric in key_metrics:
            mean = aggregated.get(f"{metric}_mean")
            std  = aggregated.get(f"{metric}_std")
            if mean is not None:
                pct = "%" if "accuracy" in metric or "win_rate" in metric \
                      or "drawdown" in metric else ""
                scale = 100 if pct else 1
                print(f"  {metric:<22}: "
                      f"{mean*scale:>8.3f}{pct} ± {std*scale:.3f}{pct}")

    return aggregated


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
    from models.trainer import train, _three_way_split

    # ── Fetch and prepare
    raw = yf.download("AAPL", period="2y", auto_adjust=True, progress=False)
    raw.columns = [c.lower() for c in raw.columns]
    clean    = clean_stock_data(raw, ticker="AAPL")
    featured = build_features(clean).dropna()
    labelled = create_labels(featured, horizon=1,
                             threshold=0.003, verbose=False)
    X = get_model_features(labelled, extra_drop=["target"]).fillna(0)
    y = labelled["target"]

    # Get actual returns for backtesting
    actual_returns = featured.loc[labelled.index, "ret_1d"].fillna(0).values / 100

    # ── Train
    pipeline, results = train(X, y, verbose=False)

    # ── Split for evaluation
    _, _, X_test, _, _, y_test = _three_way_split(X, y)
    test_returns = actual_returns[-len(X_test):]

    # ── Full evaluation
    eval_results = evaluate(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        actual_returns=test_returns,
        threshold=0.5,
        allow_short=False,
        transaction_cost=0.001,
        verbose=True,
    )
    
    