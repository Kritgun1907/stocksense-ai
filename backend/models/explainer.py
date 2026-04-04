"""
StockSense AI — models/explainer.py
=====================================
SHAP-based model explainability.

This file owns:
  - SHAP value computation for individual predictions
  - Global feature importance aggregation
  - Feature group importance (trend vs momentum vs sentiment etc.)
  - Plain-English explanation generation for the StockSense UI
  - SHAP-based feature pruning recommendations

It does NOT own:
  - Model training          → models/trainer.py
  - Evaluation metrics      → models/evaluator.py
  - Cross-validation        → models/timeseries_cv.py

Why SHAP over built-in XGBoost feature importance?
─────────────────────────────────────────────────────────────
  XGBoost's feature_importances_ measures split frequency — how often
  a feature is used across all trees. This has known flaws:
    1. Biased toward high-cardinality continuous features
    2. Can't explain individual predictions
    3. No sign — can't tell if a feature pushes UP or DOWN
    4. Doesn't account for feature interaction effects

  SHAP (Shapley values) is mathematically guaranteed to be:
    1. Consistent: removing a feature can't increase its importance
    2. Local: explains each prediction individually
    3. Signed: positive=bullish, negative=bearish contribution
    4. Additive: values sum exactly to prediction - base value
    5. Fair: based on 60 years of cooperative game theory

Why feature group importance?
─────────────────────────────────────────────────────────────
  With 340 features, raw SHAP importance is overwhelming.
  Grouping by category (trend, momentum, sentiment, patterns)
  reveals which TYPE of signal drives predictions — much more
  actionable for model improvement and user communication.

Why plain-English generation?
─────────────────────────────────────────────────────────────
  StockSense AI's core promise is explaining predictions to
  non-traders. SHAP values are numbers. Plain English is what
  the user sees on the stock page. This module bridges the two.
"""

import warnings
import numpy as np
import pandas as pd
import shap
import joblib
from typing import Dict, List, Optional, Tuple
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Minimum absolute SHAP value to include in explanations.
# Features below this threshold are noise — not worth explaining.
MIN_SHAP_TO_EXPLAIN = 0.005

# Number of top features to show in UI explanations.
TOP_N_UI            = 5
TOP_N_ANALYSIS      = 20

# SHAP magnitude thresholds for plain-English strength descriptions.
SHAP_STRENGTH_LABELS = [
    (0.15,  "very strongly"),
    (0.08,  "strongly"),
    (0.04,  "moderately"),
    (0.01,  "slightly"),
    (0.0,   "negligibly"),
]

# Feature plain-English descriptions.
# Maps feature name → (bullish description, bearish description).
# Used to generate the "why" text on the stock page.
FEATURE_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    # Momentum
    "rsi_14":             ("RSI shows oversold conditions — bounce likely",
                           "RSI shows overbought conditions — pullback likely"),
    "rsi_oversold":       ("Stock is oversold — historically bullish reversal zone",
                           ""),
    "rsi_overbought":     ("",
                           "Stock is overbought — historically bearish reversal zone"),
    "rsi_momentum":       ("RSI momentum is accelerating upward",
                           "RSI momentum is decelerating"),
    # Trend
    "price_vs_sma20":     ("Price is above 20-day average — uptrend",
                           "Price is below 20-day average — downtrend"),
    "price_vs_sma50":     ("Price is above 50-day average — medium-term uptrend",
                           "Price is below 50-day average — medium-term downtrend"),
    "trend_agreement":    ("Multiple moving averages agree: uptrend",
                           "Multiple moving averages agree: downtrend"),
    "golden_cross":       ("Golden Cross: short-term MA crossed above long-term — bullish",
                           ""),
    "death_cross":        ("",
                           "Death Cross: short-term MA crossed below long-term — bearish"),
    # MACD
    "macd_above_signal":  ("MACD is above signal line — bullish momentum",
                           "MACD is below signal line — bearish momentum"),
    "macd_bullish_cross": ("MACD just crossed above signal — fresh buy signal",
                           ""),
    "macd_bearish_cross": ("",
                           "MACD just crossed below signal — fresh sell signal"),
    "histogram_momentum": ("MACD histogram is growing — momentum accelerating",
                           "MACD histogram is shrinking — momentum fading"),
    # Volatility
    "bb_percent":         ("Price is near lower Bollinger Band — oversold territory",
                           "Price is near upper Bollinger Band — overbought territory"),
    "bb_squeeze":         ("Bollinger squeeze detected — big move incoming",
                           ""),
    "atr_pct":            ("Low volatility environment — signals more reliable",
                           "High volatility environment — signals less reliable"),
    # Volume
    "volume_ratio":       ("Above-average volume confirms the move",
                           "Below-average volume — weak conviction"),
    "volume_spike":       ("Volume spike: institutional activity detected",
                           ""),
    "obv_change_5d":      ("On-balance volume rising — buying pressure",
                           "On-balance volume falling — selling pressure"),
    # Sentiment
    "sentiment_mean":     ("News sentiment is positive",
                           "News sentiment is negative"),
    "sentiment_ma7":      ("7-day news trend is improving",
                           "7-day news trend is worsening"),
    "sentiment_regime":   ("Sustained positive news environment",
                           "Sustained negative news environment"),
    "article_count":      ("High news coverage — market attention elevated",
                           ""),
    # Patterns
    "pat_hammer":         ("Hammer pattern: sellers exhausted, buyers taking control",
                           ""),
    "pat_bull_engulf":    ("Bullish engulfing: buyers overwhelmed sellers completely",
                           ""),
    "pat_morning_star":   ("Morning Star: classic 3-candle bullish reversal",
                           ""),
    "pat_shooting_star":  ("",
                           "Shooting Star: buyers rejected at highs, sellers in control"),
    "pat_bear_engulf":    ("",
                           "Bearish engulfing: sellers overwhelmed buyers completely"),
    "pattern_signal":     ("Multiple bullish candlestick patterns align",
                           "Multiple bearish candlestick patterns align"),
    # Returns
    "ret_1d":             ("Price already moving up — momentum present",
                           "Price moving down — negative momentum"),
    "ret_5d":             ("5-day trend is positive",
                           "5-day trend is negative"),
    "dir_streak":         ("Consecutive up days — trend establishing",
                           "Consecutive down days — trend establishing"),
}

# Generic descriptions for features without specific entries.
GENERIC_BULLISH  = "Technical indicator supports upward move"
GENERIC_BEARISH  = "Technical indicator signals downward pressure"


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_preprocessed_X(
    pipeline:  Pipeline,
    X:         pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Transform X through the pipeline's preprocessor steps (all except model).
    Returns (transformed_array, feature_names).

    Why extract feature names separately?
    ─────────────────────────────────────────────────────────────
    StandardScaler and SimpleImputer don't change column names or order.
    We can safely use the original column names after transformation.
    This would need to change if we added ColumnTransformer steps.
    """
    preprocessor   = Pipeline(pipeline.steps[:-1])
    X_r            = X.reset_index(drop=True)
    X_transformed  = preprocessor.transform(X_r)
    feature_names  = list(X.columns)
    return X_transformed, feature_names


def _build_explainer(
    pipeline: Pipeline,
    X_background: Optional[pd.DataFrame] = None,
) -> shap.TreeExplainer:
    """
    Build a TreeExplainer for the XGBoost model inside the pipeline.

    Why tree_path_dependent?
    ─────────────────────────────────────────────────────────────
    Two feature perturbation methods exist for TreeSHAP:
      'interventional': uses a background dataset to compute
                        marginal feature contributions. More accurate
                        but requires background data and is slower.
      'tree_path_dependent': uses the training data distribution
                             implicitly encoded in the tree paths.
                             Faster, no background data needed,
                             slightly less accurate for correlated features.
    For 340 features with many correlations (RSI variants, MA variants),
    tree_path_dependent is more stable and fast enough for real-time use.
    """
    model = pipeline.named_steps["model"]

    if X_background is not None:
        preprocessor  = Pipeline(pipeline.steps[:-1])
        X_bg_t        = preprocessor.transform(
            X_background.reset_index(drop=True)
        )
        explainer = shap.TreeExplainer(
            model,
            data=X_bg_t,
            feature_perturbation="interventional",
        )
    else:
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )

    return explainer


def _shap_strength(shap_val: float) -> str:
    """Map SHAP magnitude to a plain-English strength word."""
    abs_val = abs(shap_val)
    for threshold, label in SHAP_STRENGTH_LABELS:
        if abs_val >= threshold:
            return label
    return "negligibly"


def _feature_to_plain_english(
    feature_name: str,
    shap_val:     float,
    feat_val:     float,
) -> str:
    """
    Convert a (feature, SHAP value, feature value) triple
    into a plain-English explanation string.

    Used to generate the 'why' text on the StockSense stock page.
    Non-traders need to understand every signal shown.
    """
    is_bullish = shap_val > 0
    strength   = _shap_strength(shap_val)

    if feature_name in FEATURE_DESCRIPTIONS:
        bull_desc, bear_desc = FEATURE_DESCRIPTIONS[feature_name]
        description = bull_desc if is_bullish else bear_desc
        if not description:
            description = GENERIC_BULLISH if is_bullish else GENERIC_BEARISH
    else:
        description = GENERIC_BULLISH if is_bullish else GENERIC_BEARISH

    direction = "supporting" if is_bullish else "weighing against"
    return f"{description} — {strength} {direction} the prediction"


# ══════════════════════════════════════════════════════════════════════════════
#  CORE SHAP COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_values(
    pipeline:     Pipeline,
    X:            pd.DataFrame,
    sample_size:  Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute SHAP values for a DataFrame of predictions.

    Parameters
    ----------
    pipeline    : Fitted sklearn Pipeline.
    X           : Feature DataFrame (may have MultiIndex).
    sample_size : If set, compute SHAP on a random sample only.
                  Use for global analysis (speed).
                  Use None for individual prediction explanations.

    Returns
    -------
    (shap_values, base_values, feature_names) tuple where:
      shap_values  : np.ndarray shape (n_samples, n_features)
                     Positive = pushed toward UP, Negative = toward DOWN
      base_values  : np.ndarray or float — model's average prediction
      feature_names: List[str] in same order as shap_values columns
    """
    if sample_size is not None and sample_size < len(X):
        idx = np.random.choice(len(X), size=sample_size, replace=False)
        X   = X.iloc[idx]

    X_t, feature_names = _get_preprocessed_X(pipeline, X)
    X_df               = pd.DataFrame(X_t, columns=feature_names)

    explainer   = _build_explainer(pipeline)
    shap_values = explainer.shap_values(X_df, check_additivity=False)
    base_values = explainer.expected_value

    return shap_values, base_values, feature_names


# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL EXPLANATION (ONE PREDICTION)
# ══════════════════════════════════════════════════════════════════════════════

def explain_single_prediction(
    pipeline:     Pipeline,
    X_single:     pd.DataFrame,
    top_n:        int  = TOP_N_UI,
    verbose:      bool = True,
) -> Dict:
    """
    Generate a complete explanation for one stock prediction.

    This is the function called by the FastAPI endpoint
    GET /predict?ticker=AAPL → returns this dict to the frontend.

    Parameters
    ----------
    pipeline  : Fitted sklearn Pipeline.
    X_single  : Single-row DataFrame (one stock, one day).
    top_n     : Number of top features to include in explanation.
    verbose   : Print formatted explanation.

    Returns
    -------
    Dict with:
      prediction      : 'UP' or 'DOWN'
      probability     : float confidence 0-1
      base_value      : model's average prediction
      shap_sum        : sum of all SHAP values (= probability - base_value)
      top_features    : list of top contributing features with plain English
      feature_groups  : SHAP importance by feature group
      explanation_text: full plain-English explanation paragraph

    Why return explanation_text as a prebuilt string?
    ─────────────────────────────────────────────────────────────
    The frontend can display it directly without needing to
    reconstruct it from individual SHAP values. The React component
    just renders the string. Simpler frontend, all logic in backend.
    """
    shap_vals, base_val, feat_names = compute_shap_values(
        pipeline, X_single, sample_size=None
    )

    shap_row    = shap_vals[0]
    proba       = pipeline.predict_proba(
        X_single.reset_index(drop=True)
    )[0][1]
    prediction  = "UP" if proba >= 0.5 else "DOWN"

    # Sort features by absolute SHAP impact
    impacts = sorted(
        zip(feat_names, shap_row, X_single.iloc[0].values),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Build top features list with plain English
    top_features = []
    for feat, shap_val, feat_val in impacts[:top_n]:
        if abs(shap_val) < MIN_SHAP_TO_EXPLAIN:
            break
        top_features.append({
            "feature":      feat,
            "shap_value":   round(float(shap_val), 4),
            "feature_value": round(float(feat_val), 4),
            "direction":    "bullish" if shap_val > 0 else "bearish",
            "strength":     _shap_strength(shap_val),
            "explanation":  _feature_to_plain_english(feat, shap_val, feat_val),
        })

    # Feature group summary
    group_shap = _compute_group_shap(shap_row, feat_names)

    # Build explanation paragraph
    explanation_text = _build_explanation_paragraph(
        prediction, proba, top_features, group_shap
    )

    result = {
        "prediction":       prediction,
        "probability":      round(float(proba), 4),
        "confidence_pct":   round(float(proba * 100), 1),
        "base_value":       round(float(base_val), 4),
        "shap_sum":         round(float(shap_row.sum()), 4),
        "top_features":     top_features,
        "feature_groups":   group_shap,
        "explanation_text": explanation_text,
        "all_shap_values":  {
            feat: round(float(val), 6)
            for feat, val in zip(feat_names, shap_row)
            if abs(val) >= MIN_SHAP_TO_EXPLAIN
        },
    }

    if verbose:
        _print_single_explanation(result)

    return result


def _build_explanation_paragraph(
    prediction:   str,
    probability:  float,
    top_features: List[Dict],
    group_shap:   Dict,
) -> str:
    """
    Build the plain-English paragraph shown on the StockSense stock page.

    Design principle: explain like a financial advisor talking to
    someone who has never traded before. No jargon. Direct language.
    """
    direction   = "go up" if prediction == "UP" else "go down"
    confidence  = f"{probability*100:.0f}%"

    lines = [
        f"Our AI model predicts this stock will {direction} "
        f"tomorrow with {confidence} confidence."
    ]

    if top_features:
        bullish = [f for f in top_features if f["direction"] == "bullish"]
        bearish = [f for f in top_features if f["direction"] == "bearish"]

        if bullish:
            lines.append(
                f"\nThe main reasons supporting this prediction: "
                f"{bullish[0]['explanation'].split('—')[0].strip()}."
            )
            if len(bullish) > 1:
                lines.append(
                    f"Additionally: "
                    f"{bullish[1]['explanation'].split('—')[0].strip()}."
                )

        if bearish:
            lines.append(
                f"\nHowever, there are some cautionary signals: "
                f"{bearish[0]['explanation'].split('—')[0].strip()}."
            )

    # Add dominant signal group context (only if group data is available)
    if group_shap:
        top_group  = max(group_shap, key=lambda k: abs(group_shap[k]))
        group_name = top_group.replace("_", " ").title()
        lines.append(
            f"\nThe strongest signal category today is "
            f"{group_name} — this has the most influence on the prediction."
        )

    return " ".join(lines)


def _print_single_explanation(result: Dict) -> None:
    """Print a formatted explanation to console."""
    pred  = result["prediction"]
    prob  = result["confidence_pct"]
    emoji = "🟢" if pred == "UP" else "🔴"

    print(f"\n{'═'*55}")
    print(f"{emoji} Prediction: {pred} ({prob:.1f}% confidence)")
    print(f"   Base value: {result['base_value']:.4f} → "
          f"Final: {result['base_value'] + result['shap_sum']:.4f}")
    print(f"{'─'*55}")
    print(f"Top contributing features:")
    print(f"  {'Feature':<28} {'Value':>8} {'SHAP':>8}")
    print(f"  {'─'*50}")

    for feat in result["top_features"]:
        arrow = "↑" if feat["direction"] == "bullish" else "↓"
        print(f"  {arrow} {feat['feature']:<26} "
              f"{feat['feature_value']:>8.3f} "
              f"{feat['shap_value']:>+8.4f}")

    print(f"\n{'─'*55}")
    print(f"Feature group importance:")
    groups = sorted(result["feature_groups"].items(),
                    key=lambda x: abs(x[1]), reverse=True)
    for group, importance in groups[:5]:
        bar = "█" * int(abs(importance) * 50)
        sign = "+" if importance > 0 else "-"
        print(f"  {group:<16} {sign}{bar}")

    print(f"\n{'─'*55}")
    print(f"Plain English:\n{result['explanation_text']}")
    print(f"{'═'*55}")


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL ANALYSIS (ALL PREDICTIONS)
# ══════════════════════════════════════════════════════════════════════════════

def compute_global_importance(
    pipeline:    Pipeline,
    X:           pd.DataFrame,
    sample_size: int  = 500,
    verbose:     bool = True,
) -> pd.DataFrame:
    """
    Compute global feature importance from SHAP values.

    Mean absolute SHAP value per feature across all samples.
    More reliable than XGBoost's built-in feature_importances_.

    Parameters
    ----------
    pipeline    : Fitted sklearn Pipeline.
    X           : Full feature DataFrame.
    sample_size : Rows to sample for speed. 500 is enough for stable estimates.
    verbose     : Print top features.

    Returns
    -------
    pd.DataFrame with columns:
      feature         : feature name
      mean_abs_shap   : mean |SHAP| across samples (global importance)
      mean_shap       : mean SHAP (positive = generally bullish feature)
      std_shap        : std of SHAP (high = feature has variable impact)
    """
    shap_vals, _, feat_names = compute_shap_values(
        pipeline, X, sample_size=sample_size
    )

    importance_df = pd.DataFrame({
        "feature":       feat_names,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        "mean_shap":     shap_vals.mean(axis=0),
        "std_shap":      shap_vals.std(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"Global Feature Importance (top {TOP_N_ANALYSIS})")
        print(f"  Based on {min(sample_size, len(X))} samples")
        print(f"{'═'*60}")
        print(f"  {'#':<4} {'Feature':<30} {'Importance':>11} "
              f"{'Direction':>10}")
        print(f"  {'─'*60}")

        for i, row in importance_df.head(TOP_N_ANALYSIS).iterrows():
            direction = "bullish" if row["mean_shap"] > 0 else "bearish"
            bar       = "█" * int(row["mean_abs_shap"] * 200)
            print(f"  {i+1:<4} {row['feature']:<30} "
                  f"{row['mean_abs_shap']:>10.5f}  "
                  f"{direction:<10}  {bar[:15]}")

    return importance_df


def _compute_group_shap(
    shap_row:     np.ndarray,
    feat_names:   List[str],
) -> Dict[str, float]:
    """
    Sum SHAP values by feature group.
    Uses FEATURE_GROUPS from indicators.py.
    """
    try:
        from features.indicators import FEATURE_GROUPS
        from nlp.rolling import ROLLING_SENTIMENT_FEATURE_GROUPS

        all_groups = dict(FEATURE_GROUPS)
        for group, cols in ROLLING_SENTIMENT_FEATURE_GROUPS.items():
            all_groups[f"sentiment_{group}"] = cols
    except ImportError:
        return {}

    feat_to_shap = dict(zip(feat_names, shap_row))
    group_shap   = {}

    for group, cols in all_groups.items():
        group_vals = [
            feat_to_shap.get(col, 0.0) for col in cols
            if col in feat_to_shap
        ]
        if group_vals:
            group_shap[group] = round(float(np.sum(group_vals)), 4)

    return group_shap


def compute_group_importance(
    pipeline:    Pipeline,
    X:           pd.DataFrame,
    sample_size: int  = 500,
    verbose:     bool = True,
) -> pd.DataFrame:
    """
    Compute feature group importance from SHAP values.

    Reveals which CATEGORY of features drives predictions:
    trend vs momentum vs sentiment vs patterns vs volume etc.

    Returns
    -------
    pd.DataFrame with group-level SHAP importance.
    """
    shap_vals, _, feat_names = compute_shap_values(
        pipeline, X, sample_size=sample_size
    )

    try:
        from features.indicators import FEATURE_GROUPS
        from nlp.rolling import ROLLING_SENTIMENT_FEATURE_GROUPS
        all_groups = {**FEATURE_GROUPS,
                      **{f"sentiment_{k}": v
                         for k, v in ROLLING_SENTIMENT_FEATURE_GROUPS.items()}}
    except ImportError:
        return pd.DataFrame()

    rows = []
    for group, cols in all_groups.items():
        col_indices = [
            i for i, f in enumerate(feat_names) if f in cols
        ]
        if not col_indices:
            continue

        group_shap   = shap_vals[:, col_indices]
        mean_abs     = float(np.abs(group_shap).mean())
        mean_dir     = float(group_shap.mean())
        n_features   = len(col_indices)

        rows.append({
            "group":          group,
            "mean_abs_shap":  round(mean_abs, 6),
            "mean_direction": round(mean_dir, 6),
            "n_features":     n_features,
            "direction":      "bullish" if mean_dir > 0 else "bearish",
        })

    df = pd.DataFrame(rows).sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    if verbose:
        print(f"\n{'═'*55}")
        print(f"Feature Group Importance")
        print(f"{'═'*55}")
        print(f"  {'Group':<20} {'Importance':>11} {'Direction':>10} "
              f"{'Features':>9}")
        print(f"  {'─'*55}")
        for _, row in df.iterrows():
            bar = "█" * int(row["mean_abs_shap"] * 500)
            print(f"  {row['group']:<20} "
                  f"{row['mean_abs_shap']:>10.5f}  "
                  f"{row['direction']:<10}  "
                  f"{row['n_features']:>8}  {bar[:10]}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE PRUNING
# ══════════════════════════════════════════════════════════════════════════════

def recommend_feature_pruning(
    importance_df: pd.DataFrame,
    min_importance: float = 0.0001,
    verbose:       bool   = True,
) -> List[str]:
    """
    Recommend features to remove based on near-zero global SHAP importance.

    Features with mean_abs_shap < min_importance contribute almost nothing
    to predictions. Removing them:
      1. Speeds up training and inference
      2. Reduces overfitting risk
      3. Makes SHAP explanations cleaner (fewer noisy features)

    Parameters
    ----------
    importance_df  : Output of compute_global_importance().
    min_importance : SHAP threshold below which features are candidates.
    verbose        : Print recommendations.

    Returns
    -------
    List of feature names recommended for removal.
    """
    low_importance = importance_df[
        importance_df["mean_abs_shap"] < min_importance
    ]

    to_remove = low_importance["feature"].tolist()

    if verbose:
        print(f"\nFeature Pruning Recommendations")
        print(f"  Min importance threshold: {min_importance:.6f}")
        print(f"  Features to remove: {len(to_remove)} / "
              f"{len(importance_df)}")
        if to_remove:
            print(f"\n  Candidates (mean_abs_shap < {min_importance}):")
            for _, row in low_importance.head(20).iterrows():
                print(f"    {row['feature']:<40} "
                      f"{row['mean_abs_shap']:.7f}")
            if len(to_remove) > 20:
                print(f"    ... and {len(to_remove)-20} more")
        else:
            print(f"  ✅ No features below threshold — all features contribute")

    return to_remove


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_explainer_cache(
    pipeline:    Pipeline,
    X_sample:    pd.DataFrame,
    path:        str = "models/saved/shap_cache.joblib",
) -> None:
    """
    Pre-compute and cache SHAP explainer + background for fast inference.
    Called once after training, loaded in FastAPI for real-time explanations.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    X_t, feat_names = _get_preprocessed_X(pipeline, X_sample)
    explainer       = _build_explainer(pipeline)

    cache = {
        "explainer":     explainer,
        "feature_names": feat_names,
        "base_value":    float(explainer.expected_value),
    }

    joblib.dump(cache, path)
    print(f"SHAP cache saved: {path}")


def load_explainer_cache(
    path: str = "models/saved/shap_cache.joblib",
) -> Dict:
    """Load pre-computed SHAP explainer from cache."""
    if not __import__("os").path.exists(path):
        raise FileNotFoundError(
            f"SHAP cache not found at {path}. "
            f"Run save_explainer_cache() after training."
        )
    cache = joblib.load(path)
    print(f"SHAP cache loaded. Base value: {cache['base_value']:.4f}")
    return cache


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
    from models.trainer import train

    # ── Prepare data
    raw = yf.download("AAPL", period="2y", auto_adjust=True, progress=False)
    raw.columns = [c.lower() for c in raw.columns]
    clean    = clean_stock_data(raw, ticker="AAPL")
    featured = build_features(clean).dropna()
    labelled = create_labels(featured, horizon=1,
                             threshold=0.003, verbose=False)
    X = get_model_features(labelled, extra_drop=["target"]).fillna(0)
    y = labelled["target"]

    # ── Train
    pipeline, results = train(X, y, verbose=False)

    # ── Global importance
    importance_df = compute_global_importance(pipeline, X, sample_size=300)

    # ── Group importance
    group_df = compute_group_importance(pipeline, X, sample_size=300)

    # ── Single prediction explanation
    X_latest = X.tail(1)
    explain_single_prediction(pipeline, X_latest, top_n=5, verbose=True)

    # ── Feature pruning
    to_remove = recommend_feature_pruning(importance_df, min_importance=0.0001)

    # ── Save SHAP cache for FastAPI
    save_explainer_cache(pipeline, X.tail(100))
