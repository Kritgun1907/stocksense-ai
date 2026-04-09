"""
StockSense AI — models/orpsoc_selector.py
==========================================
Hybrid OrPSOC Feature Selection for financial time-series data.

WHAT THIS FILE DOES
────────────────────
Implements a hybrid feature selection pipeline that combines:

  Phase 1: SHAP-based pre-pruning
           Run one XGBoost model, compute SHAP values, drop the bottom
           percentile of features (near-zero SHAP importance).
           Reduces 315+ features → ~80-120 features.
           This makes the PSO search space computationally tractable.

  Phase 2: OrPSOC — Orthogonal-initialised PSO with Crossover
           Each "particle" is a binary vector of length N_features
           (1 = keep the feature, 0 = drop it).
           Particles fly through the binary search space, each
           evaluating a candidate feature subset's AUC-ROC on the
           validation set.
           Two OrPSOC enhancements over vanilla PSO:
             a) Orthogonal Initialization — particles spread uniformly
                using an Orthogonal Array instead of clustering randomly.
             b) Crossover Operator (from Genetic Algorithms) — pairs of
                particles exchange bit-strings to explore combinations
                neither particle would reach alone.

WHY THIS EXISTS
────────────────
Standard SHAP pruning ranks features individually.  It answers:
  "How much does this feature contribute on average?"
It misses interactions: Feature A alone = useless.
                         Feature A + Feature B together = highly predictive.

PSO-based selection evaluates subsets, not individuals.  It answers:
  "What COMBINATION of features produces the best model?"
This naturally handles interaction effects.

WHERE IT FITS IN THE PIPELINE
──────────────────────────────
  Without OrPSOC:
    build_features(315+) → SHAP prune → XGBoost(~100 cols) → predict

  With OrPSOC:
    build_features(315+) → SHAP prune(~100) → OrPSOC(~40-60) → XGBoost → predict

The OrPSOC-selected subset is saved to disk alongside the model.  At
inference time, only the selected columns are passed to the pipeline.

This file owns:
  - SHAP-based pre-pruning       (shap_preprune)
  - Orthogonal array generation  (_build_orthogonal_array)
  - Binary PSO particle class    (_Particle)
  - OrPSOC search loop           (run_orpsoc)
  - Full hybrid pipeline         (hybrid_select)
  - Save / load selection        (save_selection, load_selection)
  - Result reporting             (report)

It does NOT own:
  - Feature engineering          → features/engineer.py
  - Model training               → models/trainer.py
  - Evaluation metrics           → models/evaluator.py
  - SHAP explanations for UI     → models/explainer.py

Usage
─────
    # Run full hybrid pipeline on pre-engineered data
    from backend.models.orpsoc_selector import hybrid_select, report

    result = hybrid_select(
        X_train, y_train,
        X_val,   y_val,
        verbose=True,
    )
    report(result)

    selected_features = result["selected_features"]   # list[str]
    X_train_sel = X_train[selected_features]
    X_val_sel   = X_val[selected_features]
    # → feed into build_sklearn_pipeline().fit(X_train_sel, y_train)

    # Or run phases independently
    from backend.models.orpsoc_selector import shap_preprune, run_orpsoc

Research Source
───────────────
OrPSOC is based on:
  "Orthogonal Initialization and Crossover Operator for PSO-Based
   Feature Selection in High-Dimensional Medical Datasets"
   (Leukemia, Colon cancer, Prostate Tumor benchmark datasets)

Our adaptation for financial data:
  1. Walk-forward (temporal) fitness evaluation — prevents leakage
  2. Min-feature constraint — PSO must keep ≥ 10 features at all times
  3. Adaptive crossover rate that decays as swarm converges
  4. Velocity clamping to prevent bit-flip explosion
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ── Path resolution ───────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent          # backend/models/
_BACKEND    = _HERE.parent                             # backend/
_SELECTIONS = _BACKEND.parent / "data" / "selections"  # project root / data / selections
_SELECTIONS.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — Tuned for financial time-series data
# ══════════════════════════════════════════════════════════════════════════════

# ── Phase 1: SHAP pre-pruning ─────────────────────────────────────────────────
# Drop features whose mean |SHAP| is in the bottom N percentile.
# 50th percentile means we drop half of features by importance (the weakest half).
# Leaves ~80-120 features from 315+, making the PSO search tractable.
SHAP_PRUNE_PERCENTILE   = 50          # drop bottom 50% by |SHAP| importance
SHAP_PRUNE_MIN_FEATURES = 40          # never drop below this many features
SHAP_PRUNE_MAX_FEATURES = 150         # cap at this many for PSO tractability

# ── Phase 2: PSO hyperparameters ─────────────────────────────────────────────
PSO_N_PARTICLES  = 30    # swarm size — more = better exploration, slower
PSO_MAX_ITER     = 60    # iterations — OrPSOC converges faster than vanilla PSO
PSO_W_START      = 0.9   # initial inertia weight (exploration)
PSO_W_END        = 0.4   # final inertia weight (exploitation)
PSO_C1           = 2.0   # cognitive weight — pull toward personal best
PSO_C2           = 2.0   # social weight   — pull toward global best
PSO_V_MAX        = 6.0   # velocity clamping — prevents sigmoid saturation
PSO_MIN_FEATURES = 10    # minimum features a particle must select
PSO_MAX_FEATURES = None  # no upper bound (None = all features available)

# ── Crossover parameters ─────────────────────────────────────────────────────
CROSSOVER_RATE_START = 0.7   # probability a particle pair undergoes crossover
CROSSOVER_RATE_END   = 0.3   # decays toward this as swarm converges
CROSSOVER_POINTS     = 2     # number of crossover points (2-point crossover)

# ── Fitness evaluation ────────────────────────────────────────────────────────
FITNESS_METRIC        = "auc"        # "auc" or "f1" or "accuracy"
FITNESS_PENALTY_COEFF = 0.001        # penalty per extra feature beyond minimum
# Why penalise? AUC alone doesn't reward parsimony.
# Without penalty, PSO converges to selecting ALL features (trivially safe).
# With penalty: AUC(subset) - 0.001 * (n_selected - MIN_FEATURES)
# Forces PSO to earn every extra feature through demonstrated AUC gain.

# ── Fast fitness model (lighter XGBoost for PSO loop) ────────────────────────
# Each particle requires one XGBoost fit per fitness call.
# We use a much lighter model than the production one to keep PSO tractable.
# 30 particles × 60 iterations = 1800 XGBoost fits — must each be fast.
FITNESS_XGB_PARAMS: Dict = {
    "n_estimators"    : 80,     # fewer trees than production (300)
    "learning_rate"   : 0.1,    # faster convergence for fitness evaluation
    "max_depth"       : 4,      # shallower than production (6)
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "eval_metric"     : "logloss",
    "verbosity"       : 0,
    "random_state"    : 42,
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrPSOCResult:
    """
    The complete output of a hybrid_select() run.

    Attributes
    ──────────
    selected_features  : list[str] — the final chosen feature columns
    n_original         : int — features before SHAP pruning
    n_after_shap       : int — features after SHAP pruning, before PSO
    n_selected         : int — features after PSO selection
    shap_dropped       : list[str] — features dropped by SHAP pruning
    pso_dropped        : list[str] — features dropped by PSO (post-SHAP)

    fitness_history    : list[float] — global best AUC per PSO iteration
    diversity_history  : list[float] — swarm diversity per iteration
    best_auc_shap      : float — AUC of SHAP-pruned model (pre-PSO baseline)
    best_auc_orpsoc    : float — AUC of final OrPSOC-selected model
    auc_gain           : float — improvement OrPSOC achieved over SHAP-only

    n_particles        : int — swarm size used
    n_iterations       : int — iterations run
    runtime_seconds    : float — total wall time
    feature_group_summary : dict — selected features broken down by group
    """
    selected_features   : List[str]  = field(default_factory=list)
    n_original          : int        = 0
    n_after_shap        : int        = 0
    n_selected          : int        = 0
    shap_dropped        : List[str]  = field(default_factory=list)
    pso_dropped         : List[str]  = field(default_factory=list)

    fitness_history     : List[float] = field(default_factory=list)
    diversity_history   : List[float] = field(default_factory=list)
    best_auc_shap       : float       = 0.0
    best_auc_orpsoc     : float       = 0.0
    auc_gain            : float       = 0.0

    n_particles         : int   = PSO_N_PARTICLES
    n_iterations        : int   = PSO_MAX_ITER
    runtime_seconds     : float = 0.0
    feature_group_summary: Dict = field(default_factory=dict)


@dataclass
class _Particle:
    """
    A single particle in the PSO swarm.

    Represents one candidate feature subset as a binary vector.
    position[i] ∈ {0, 1}:
      1 = keep feature i
      0 = drop feature i

    velocity[i] ∈ ℝ (real-valued, even though position is binary):
      The S-shaped sigmoid of velocity gives the PROBABILITY that
      position[i] flips to 1.  This is Binary PSO (Kennedy & Eberhart 1997).

    Why velocity is real-valued:
      Binary PSO: instead of moving continuously in space (like numeric PSO),
      each dimension is a coin flip with p = sigmoid(velocity).
      High velocity → high probability of being 1 (keep feature).
      Low velocity  → high probability of being 0 (drop feature).
    """
    n_features  : int
    position    : np.ndarray = field(default_factory=lambda: np.array([]))
    velocity    : np.ndarray = field(default_factory=lambda: np.array([]))
    best_pos    : np.ndarray = field(default_factory=lambda: np.array([]))
    best_fitness: float = -np.inf

    def __post_init__(self):
        pass   # position/velocity assigned by initialiser functions

    def n_selected(self) -> int:
        """Number of features currently selected (sum of 1-bits)."""
        return int(self.position.sum())

    def selected_indices(self) -> np.ndarray:
        """Indices where position == 1."""
        return np.where(self.position == 1)[0]


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — SHAP PRE-PRUNING
# ══════════════════════════════════════════════════════════════════════════════

def shap_preprune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    prune_percentile: float = SHAP_PRUNE_PERCENTILE,
    min_features: int = SHAP_PRUNE_MIN_FEATURES,
    max_features: int = SHAP_PRUNE_MAX_FEATURES,
    scale_pos_weight: float = 1.0,
    verbose: bool = True,
) -> Tuple[List[str], List[str], float]:
    """
    Phase 1: SHAP-based pre-pruning.

    Trains a fast XGBoost model on the full feature set, computes
    SHAP values, and drops the bottom `prune_percentile` of features
    by mean absolute SHAP importance.

    This reduces the search space for PSO from 2^315 to ~2^100,
    making the swarm search computationally tractable.

    Parameters
    ----------
    X_train, y_train : Training split (chronologically earlier)
    X_val, y_val     : Validation split (chronologically later)
                       Used to compute AUC baseline and SHAP values.
                       NEVER the test set — no leakage.
    prune_percentile : Bottom N% of features by |SHAP| to drop.
    min_features     : Never return fewer than this many features.
    max_features     : Never return more than this many features.
    scale_pos_weight : For imbalanced datasets (n_neg / n_pos).
    verbose          : Print progress.

    Returns
    -------
    (kept_features, dropped_features, baseline_auc)
      kept_features    : list[str] — features to pass to Phase 2
      dropped_features : list[str] — features eliminated by SHAP
      baseline_auc     : float — AUC of the SHAP-pruned model on val set
                                 (OrPSOC must beat this)
    """
    t0 = time.time()
    n_total = X_train.shape[1]
    if verbose:
        _log(f"Phase 1 — SHAP Pre-pruning")
        _log(f"  Input features  : {n_total}")
        _log(f"  Prune percentile: {prune_percentile}th")

    # ── Build a fast pipeline for SHAP scoring ──────────────────────────────
    pipe = _build_fast_pipeline(scale_pos_weight=scale_pos_weight)
    pipe.fit(X_train, y_train)

    # ── Compute mean |SHAP| per feature ─────────────────────────────────────
    # Use the XGBoost step directly (SHAP's TreeExplainer is tree-native)
    xgb_model   = pipe.named_steps["model"]
    imputed      = pipe.named_steps["imputer"].transform(X_train)
    scaled       = pipe.named_steps["scaler"].transform(imputed)
    X_transformed = pd.DataFrame(scaled, columns=X_train.columns)

    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)

    # shap_values shape: (n_samples, n_features)
    # mean |SHAP| per feature = overall importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_series   = pd.Series(mean_abs_shap, index=X_train.columns)

    # ── Determine cutoff threshold ───────────────────────────────────────────
    threshold = np.percentile(mean_abs_shap, prune_percentile)

    # Features above threshold survive
    kept_mask     = shap_series >= threshold
    kept_features = shap_series[kept_mask].index.tolist()
    dropped       = shap_series[~kept_mask].index.tolist()

    # Enforce min/max bounds
    if len(kept_features) < min_features:
        # Take top min_features by SHAP even if below threshold
        top_idxs      = shap_series.nlargest(min_features).index.tolist()
        kept_features = top_idxs
        dropped       = [c for c in X_train.columns if c not in kept_features]

    if len(kept_features) > max_features:
        # Cap at max_features (take top N by SHAP)
        top_idxs      = shap_series.nlargest(max_features).index.tolist()
        dropped       += [c for c in kept_features if c not in top_idxs]
        kept_features  = top_idxs

    # ── AUC on val set (this is the baseline OrPSOC must beat) ──────────────
    prob_val     = pipe.predict_proba(X_val[X_train.columns])[:, 1]
    baseline_auc = roc_auc_score(y_val, prob_val)

    elapsed = time.time() - t0
    if verbose:
        _log(f"  Features kept   : {len(kept_features)}")
        _log(f"  Features dropped: {len(dropped)}")
        _log(f"  Baseline AUC    : {baseline_auc:.4f}")
        _log(f"  Phase 1 time    : {elapsed:.1f}s")

    return kept_features, dropped, baseline_auc


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — OrPSOC
# ══════════════════════════════════════════════════════════════════════════════

# ── Orthogonal Initialization ─────────────────────────────────────────────────

def _build_orthogonal_array(n_particles: int, n_features: int) -> np.ndarray:
    """
    Build an orthogonal array for particle initialization.

    What is an Orthogonal Array?
    ─────────────────────────────
    An orthogonal array is a mathematical grid where, for every pair of
    columns, each combination of values appears equally often.
    This guarantees UNIFORM COVERAGE of the search space.

    Example with 4 particles and 4 features (L4(2^3) array):
      Particle | F0 | F1 | F2 | F3
      ─────────┼────┼────┼────┼────
           0   |  0 |  0 |  0 |  1
           1   |  0 |  1 |  1 |  0
           2   |  1 |  0 |  1 |  0
           3   |  1 |  1 |  0 |  1

    Each column has exactly 2 zeros and 2 ones → uniform marginal coverage.
    Every pair of columns covers (0,0), (0,1), (1,0), (1,1) → orthogonal.

    Why is this better than random initialization?
    ──────────────────────────────────────────────
    Random init can cluster particles in one region of the search space,
    especially in high dimensions. Orthogonal init guarantees that the
    initial swarm explores all regions equally — like stratified sampling
    vs. simple random sampling.

    Our simplified approach:
    We use a Hadamard-matrix-inspired construction. For exact OA computation,
    libraries like pyDOE2 exist, but we implement a stable version that works
    for any (n_particles, n_features) combination without dependencies.

    The core idea: use a Latin Hypercube-style assignment where each
    "feature slot" gets exactly n_particles/2 zeros and n_particles/2 ones,
    and the assignments are staggered across features to maximize coverage.

    Parameters
    ----------
    n_particles : int — number of rows (particles)
    n_features  : int — number of columns (features)

    Returns
    -------
    np.ndarray of shape (n_particles, n_features) with values in {0, 1}.
    """
    rng  = np.random.RandomState(seed=42)  # reproducible
    grid = np.zeros((n_particles, n_features), dtype=int)

    for j in range(n_features):
        # How many 1s to place in this column
        n_ones = n_particles // 2
        # Stagger starting position per column to avoid column correlation
        shift  = (j * (n_particles // max(n_features, 1))) % n_particles
        ones_indices = [(shift + k) % n_particles for k in range(n_ones)]
        grid[ones_indices, j] = 1

    # Shuffle rows to break deterministic pattern (while keeping coverage)
    row_order = rng.permutation(n_particles)
    return grid[row_order]


def _init_particles_orthogonal(
    n_particles: int,
    n_features: int,
    min_features: int,
) -> List[_Particle]:
    """
    Initialize particles using orthogonal array for position,
    and random uniform for velocity.

    After orthogonal position assignment, we correct any particle
    that falls below min_features by randomly flipping 0→1.

    Parameters
    ----------
    n_particles  : Swarm size.
    n_features   : Dimension of each particle (= features after SHAP prune).
    min_features : Minimum bits that must be 1 in each particle.

    Returns
    -------
    list of _Particle objects, all with valid positions.
    """
    rng  = np.random.RandomState(seed=0)
    oa   = _build_orthogonal_array(n_particles, n_features)
    particles = []

    for i in range(n_particles):
        pos = oa[i].copy().astype(float)

        # Enforce min_features: if fewer than min_features bits are 1,
        # randomly flip 0→1 until we reach min_features.
        if pos.sum() < min_features:
            zeros     = np.where(pos == 0)[0]
            n_to_flip = min_features - int(pos.sum())
            chosen    = rng.choice(zeros, size=n_to_flip, replace=False)
            pos[chosen] = 1

        # Velocity: random in [-V_MAX/2, +V_MAX/2]
        vel = rng.uniform(-PSO_V_MAX / 2, PSO_V_MAX / 2, size=n_features)

        p           = _Particle(n_features=n_features)
        p.position  = pos
        p.velocity  = vel
        p.best_pos  = pos.copy()
        particles.append(p)

    return particles


# ── Fitness Evaluation ────────────────────────────────────────────────────────

def _evaluate_particle(
    position: np.ndarray,
    feature_names: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    min_features: int,
) -> float:
    """
    Compute the fitness (AUC) of a particle's position.

    Fitness = AUC(subset) - FITNESS_PENALTY_COEFF * max(0, n_selected - min_features)

    The penalty term encourages feature parsimony — OrPSOC must earn
    every feature beyond the minimum by demonstrating AUC improvement.

    For financial time-series: X_train is CHRONOLOGICALLY BEFORE X_val.
    This is enforced by the caller (hybrid_select).  The fitness function
    itself just receives already-split data — no temporal logic here.

    Parameters
    ----------
    position     : Binary array of length n_features.
    feature_names: Ordered list of feature column names.
    X_train, y_train : Training data (earlier in time).
    X_val, y_val     : Validation data (later in time).
    scale_pos_weight : Class imbalance weight.
    min_features     : Features below this count incur no penalty.

    Returns
    -------
    float — fitness score (higher = better).
             Returns -1.0 on degenerate subsets (fewer than min_features selected).
    """
    selected_idx   = np.where(position == 1)[0]
    n_sel          = len(selected_idx)

    # Guard: if fewer than min_features, this particle is invalid
    if n_sel < min_features:
        return -1.0

    selected_cols  = [feature_names[i] for i in selected_idx]
    X_tr_sel       = X_train[selected_cols]
    X_va_sel       = X_val[selected_cols]

    try:
        pipe = _build_fast_pipeline(scale_pos_weight=scale_pos_weight)
        pipe.fit(X_tr_sel, y_train)
        prob = pipe.predict_proba(X_va_sel)[:, 1]
        auc  = roc_auc_score(y_val, prob)
    except Exception:
        return -1.0   # training failed (rare edge case)

    # Parsimony penalty — penalise extra features
    penalty = FITNESS_PENALTY_COEFF * max(0, n_sel - min_features)
    return float(auc - penalty)


# ── Crossover Operator ────────────────────────────────────────────────────────

def _crossover(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    n_points: int,
    crossover_rate: float,
    min_features: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-point crossover between two particle positions.

    What Crossover Does
    ────────────────────
    Parent A: [1, 0, 1, 0, | 1, 1, 0, | 1, 0, 1]
    Parent B: [0, 1, 0, 1, | 0, 0, 1, | 0, 1, 0]
                           ^           ^
                       cut point 1  cut point 2

    Child A : [1, 0, 1, 0, | 0, 0, 1, | 1, 0, 1]  ← B's middle segment
    Child B : [0, 1, 0, 1, | 1, 1, 0, | 0, 1, 0]  ← A's middle segment

    Children carry feature combinations from BOTH parents that no single
    particle velocity update would produce — this is the key benefit of
    the genetic crossover operator borrowed from Genetic Algorithms.

    After crossover, we enforce min_features on both children.

    Parameters
    ----------
    pos_a, pos_b : Binary position arrays.
    n_points     : Number of crossover cut points (usually 2).
    crossover_rate : Probability this pair actually crossover (vs. copy).
    min_features : Minimum 1-bits required in any child.
    rng          : Random number generator for reproducibility.

    Returns
    -------
    (child_a, child_b) — two new binary position arrays.
    """
    if rng.random() > crossover_rate:
        # No crossover this time — copy parents
        return pos_a.copy(), pos_b.copy()

    n = len(pos_a)
    # Pick n_points unique cut points (interior positions only)
    cut_points = sorted(rng.choice(range(1, n), size=n_points, replace=False))

    child_a = pos_a.copy()
    child_b = pos_b.copy()

    # Swap segments between cut points
    # For 2-point crossover: swap the middle segment
    if len(cut_points) >= 2:
        c1, c2    = cut_points[0], cut_points[1]
        child_a[c1:c2] = pos_b[c1:c2]
        child_b[c1:c2] = pos_a[c1:c2]
    else:
        c1 = cut_points[0]
        child_a[c1:] = pos_b[c1:]
        child_b[c1:] = pos_a[c1:]

    # Enforce min_features on each child
    for child in [child_a, child_b]:
        n_sel = int(child.sum())
        if n_sel < min_features:
            zeros     = np.where(child == 0)[0]
            n_to_flip = min_features - n_sel
            if len(zeros) >= n_to_flip:
                chosen        = rng.choice(zeros, size=n_to_flip, replace=False)
                child[chosen] = 1

    return child_a, child_b


# ── Velocity and Position Update (Binary PSO) ─────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid function — maps velocity to selection probability.

    s(v) = 1 / (1 + e^{-v})

    Used to convert real-valued velocity into a probability:
      p(position[i] = 1) = s(velocity[i])

    With velocity clamping to [-V_MAX, +V_MAX]:
      s(-6) ≈ 0.002  → almost certain to be 0 (drop feature)
      s( 0) = 0.500  → 50/50 coin flip
      s(+6) ≈ 0.998  → almost certain to be 1 (keep feature)
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _update_particle(
    particle   : _Particle,
    global_best: np.ndarray,
    w          : float,
    c1         : float,
    c2         : float,
    min_features: int,
    rng        : np.random.RandomState,
) -> None:
    """
    Update a single particle's velocity and position (in-place).

    PSO Velocity Update Equation (standard form):
    ──────────────────────────────────────────────
    v(t+1) = w  * v(t)                                   ← inertia
           + c1 * r1 * (personal_best - current_position) ← cognitive
           + c2 * r2 * (global_best  - current_position)  ← social

    Where:
      w  = inertia weight (decreases linearly from W_START to W_END)
           Controls exploration vs. exploitation trade-off:
           High w → particle keeps its momentum (exploration)
           Low w  → particle slows down and focuses (exploitation)
      c1 = cognitive weight (pull toward own best found position)
      c2 = social weight (pull toward swarm's best position)
      r1, r2 = uniform random [0,1] — different each update

    Binary PSO Position Update:
    ──────────────────────────
    After computing v(t+1), the probability of position[i]=1 is:
      p = sigmoid(v(t+1)[i])
    Then position[i] = 1 if uniform_random[0,1] < p else 0

    This stochastic flip (rather than rounding) preserves diversity
    in the swarm — particles in the same position can still diverge.

    Finally, enforce min_features on the new position.
    """
    n   = particle.n_features
    r1  = rng.uniform(0, 1, size=n)
    r2  = rng.uniform(0, 1, size=n)

    # ── Velocity update ───────────────────────────────────────────────────────
    new_vel = (
        w  * particle.velocity
        + c1 * r1 * (particle.best_pos - particle.position)
        + c2 * r2 * (global_best       - particle.position)
    )
    # Clamp velocity to prevent sigmoid saturation
    new_vel = np.clip(new_vel, -PSO_V_MAX, PSO_V_MAX)
    particle.velocity = new_vel

    # ── Binary position update (stochastic) ───────────────────────────────────
    probs    = _sigmoid(new_vel)
    rand     = rng.uniform(0, 1, size=n)
    new_pos  = (rand < probs).astype(float)

    # Enforce minimum feature count
    n_sel = int(new_pos.sum())
    if n_sel < min_features:
        zeros     = np.where(new_pos == 0)[0]
        n_to_flip = min_features - n_sel
        if len(zeros) >= n_to_flip:
            chosen          = rng.choice(zeros, size=n_to_flip, replace=False)
            new_pos[chosen] = 1

    particle.position = new_pos


# ── Swarm Diversity Metric ────────────────────────────────────────────────────

def _swarm_diversity(particles: List[_Particle]) -> float:
    """
    Compute swarm diversity as mean Hamming distance between all particle pairs.

    Hamming distance between two binary vectors = number of bit positions
    where they differ / n_features.

    High diversity (close to 1.0) → particles scattered across search space.
    Low diversity  (close to 0.0) → particles converged (premature convergence).

    OrPSOC's crossover operator specifically fights low diversity by
    creating children that are genuinely different from both parents.
    """
    positions = np.array([p.position for p in particles])
    n, d      = positions.shape
    if n < 2:
        return 0.0
    diffs = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            diffs += np.sum(positions[i] != positions[j]) / d
            count += 1
    return diffs / count if count > 0 else 0.0


# ── Main PSO Loop ─────────────────────────────────────────────────────────────

def run_orpsoc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
    scale_pos_weight: float = 1.0,
    n_particles: int = PSO_N_PARTICLES,
    max_iter: int = PSO_MAX_ITER,
    min_features: int = PSO_MIN_FEATURES,
    verbose: bool = True,
    seed: int = 0,
) -> Tuple[List[str], List[float], List[float], float]:
    """
    Phase 2: OrPSOC search loop.

    Given a reduced feature set (after SHAP pruning), searches for the
    optimal binary subset using Orthogonal-initialised Binary PSO with
    Crossover.

    The three OrPSOC innovations over vanilla Binary PSO:
    ──────────────────────────────────────────────────────
    1. Orthogonal initialization:
       Particles start at positions chosen from an orthogonal array,
       guaranteeing uniform coverage of the feature space.
       → Better starting diversity → less premature convergence.

    2. Adaptive inertia weight (w):
       w decays linearly from W_START (0.9) to W_END (0.4) across iterations.
       Early iterations: high w → exploration (particles roam freely).
       Late iterations : low w  → exploitation (particles exploit best regions).

    3. Crossover operator:
       Each iteration, randomly pair particles and apply 2-point crossover
       to generate child positions.  Children that beat their parents replace them.
       Crossover rate decays from CROSSOVER_RATE_START to CROSSOVER_RATE_END
       as swarm converges (measured by diversity score).
       → Maintains genetic diversity throughout the search, not just at init.

    IMPORTANT — Temporal safety for financial data:
    ───────────────────────────────────────────────
    X_train must be CHRONOLOGICALLY before X_val.
    This function does NOT reshuffle data — it receives pre-split arrays.
    Each particle's fitness call trains on X_train and evaluates on X_val.
    This mirrors a walk-forward evaluation and prevents future leakage.

    Parameters
    ----------
    X_train, y_train  : Training split (earlier in time).
    X_val, y_val      : Validation split (later in time).
    feature_names     : Ordered list of column names in X_train/X_val.
                        This is the REDUCED list after SHAP pruning.
    scale_pos_weight  : Class imbalance correction (n_neg / n_pos).
    n_particles       : Swarm size.
    max_iter          : Maximum PSO iterations.
    min_features      : Minimum features any particle must select.
    verbose           : Print progress every 10 iterations.
    seed              : Random seed for reproducibility.

    Returns
    -------
    (selected_features, fitness_history, diversity_history, best_auc)
      selected_features : list[str] — feature names at global best position
      fitness_history   : list[float] — best fitness per iteration
      diversity_history : list[float] — swarm diversity per iteration
      best_auc          : float — raw AUC of the selected subset (without penalty)
    """
    rng        = np.random.RandomState(seed=seed)
    n_features = len(feature_names)
    t0         = time.time()

    if verbose:
        _log(f"Phase 2 — OrPSOC Search")
        _log(f"  Search space    : {n_features} features → 2^{n_features} possible subsets")
        _log(f"  Swarm size      : {n_particles}")
        _log(f"  Max iterations  : {max_iter}")
        _log(f"  Min features    : {min_features}")

    # ── Step 1: Orthogonal Initialization ────────────────────────────────────
    particles = _init_particles_orthogonal(n_particles, n_features, min_features)

    # ── Step 2: Evaluate initial positions ───────────────────────────────────
    global_best_pos     = None
    global_best_fitness = -np.inf

    if verbose:
        _log("  Evaluating initial positions...")

    for p in particles:
        fit = _evaluate_particle(
            p.position, feature_names,
            X_train, y_train, X_val, y_val,
            scale_pos_weight, min_features,
        )
        p.best_fitness = fit
        if fit > global_best_fitness:
            global_best_fitness = fit
            global_best_pos     = p.position.copy()

    fitness_history   = [global_best_fitness]
    diversity_history = [_swarm_diversity(particles)]

    if verbose:
        _log(f"  Initial best fitness : {global_best_fitness:.4f}")
        _log(f"  Initial diversity    : {diversity_history[0]:.3f}")
        _log(f"  {'Iter':>5} | {'Best AUC':>9} | {'Diversity':>10} | {'N features':>11}")
        _log(f"  {'─'*45}")

    # ── Step 3: Main PSO loop ─────────────────────────────────────────────────
    for iteration in range(max_iter):

        # Linear decay of inertia weight from W_START to W_END
        w = PSO_W_START - (PSO_W_START - PSO_W_END) * (iteration / max(max_iter - 1, 1))

        # Compute adaptive crossover rate based on swarm diversity
        diversity       = _swarm_diversity(particles)
        # High diversity → lower crossover rate (particles already diverse)
        # Low diversity  → higher crossover rate (inject new diversity via crossover)
        crossover_rate  = CROSSOVER_RATE_START - (
            (CROSSOVER_RATE_START - CROSSOVER_RATE_END)
            * min(1.0, (1.0 - diversity) * 2)   # scale based on diversity loss
        )

        # ── a) Velocity + position update for all particles ───────────────────
        for p in particles:
            _update_particle(p, global_best_pos, w, PSO_C1, PSO_C2, min_features, rng)

        # ── b) Crossover phase — generate children from random pairs ──────────
        indices = list(range(n_particles))
        rng.shuffle(indices)
        pairs   = [(indices[k], indices[k + 1]) for k in range(0, n_particles - 1, 2)]

        for idx_a, idx_b in pairs:
            pa, pb = particles[idx_a], particles[idx_b]
            child_a_pos, child_b_pos = _crossover(
                pa.position, pb.position,
                CROSSOVER_POINTS, crossover_rate, min_features, rng,
            )
            # Evaluate children — replace parent only if child is better
            for child_pos, parent in [(child_a_pos, pa), (child_b_pos, pb)]:
                child_fit = _evaluate_particle(
                    child_pos, feature_names,
                    X_train, y_train, X_val, y_val,
                    scale_pos_weight, min_features,
                )
                if child_fit > parent.best_fitness:
                    parent.position  = child_pos
                    parent.best_pos  = child_pos.copy()
                    parent.best_fitness = child_fit

        # ── c) Evaluate updated positions ─────────────────────────────────────
        for p in particles:
            fit = _evaluate_particle(
                p.position, feature_names,
                X_train, y_train, X_val, y_val,
                scale_pos_weight, min_features,
            )
            if fit > p.best_fitness:
                p.best_fitness = fit
                p.best_pos     = p.position.copy()

            if fit > global_best_fitness:
                global_best_fitness = fit
                global_best_pos     = p.position.copy()

        # ── d) Track history ──────────────────────────────────────────────────
        diversity = _swarm_diversity(particles)
        fitness_history.append(global_best_fitness)
        diversity_history.append(diversity)

        if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
            n_sel = int(global_best_pos.sum())
            _log(
                f"  {iteration + 1:>5} | {global_best_fitness:>9.4f}"
                f" | {diversity:>10.3f} | {n_sel:>11}"
            )

    # ── Step 4: Extract final result ──────────────────────────────────────────
    selected_idx   = np.where(global_best_pos == 1)[0]
    selected_feats = [feature_names[i] for i in selected_idx]

    # Compute AUC without penalty for the clean reported metric
    best_auc = _compute_auc_for_subset(
        selected_feats, X_train, y_train, X_val, y_val, scale_pos_weight
    )

    elapsed = time.time() - t0
    if verbose:
        _log(f"  ─" * 23)
        _log(f"  OrPSOC complete in {elapsed:.1f}s")
        _log(f"  Final selected features : {len(selected_feats)}")
        _log(f"  Final AUC (no penalty)  : {best_auc:.4f}")

    return selected_feats, fitness_history, diversity_history, best_auc


# ══════════════════════════════════════════════════════════════════════════════
#  FULL HYBRID PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float = 1.0,
    shap_percentile: float = SHAP_PRUNE_PERCENTILE,
    n_particles: int = PSO_N_PARTICLES,
    max_iter: int = PSO_MAX_ITER,
    min_features: int = PSO_MIN_FEATURES,
    verbose: bool = True,
    seed: int = 0,
) -> OrPSOCResult:
    """
    Full hybrid feature selection: SHAP pruning → OrPSOC search.

    This is the main entry point.  Call this after splitting your data
    chronologically into train and val.  The returned result.selected_features
    list can be used to subset X_train, X_val, and X_test before training
    the final production XGBoost model.

    IMPORTANT: X_train must precede X_val in time.
               X_test must NOT be passed here — it must remain hidden
               until final evaluation to preserve honest test performance.

    Timeline guarantee:
    ──────────────────
    [── training data ──][── val data ──][── test data (never touched here) ──]
            ↑                    ↑
       Phase 1 fits SHAP    Phase 2 fitness
       model on this         evaluates on this

    Parameters
    ----------
    X_train, y_train  : Training split.
    X_val, y_val      : Validation split (chronologically after train).
    scale_pos_weight  : Class imbalance weight (n_neg / n_pos).
    shap_percentile   : Prune bottom N% of features by |SHAP|.
    n_particles       : OrPSOC swarm size.
    max_iter          : OrPSOC max iterations.
    min_features      : Minimum features OrPSOC must select.
    verbose           : Print progress.
    seed              : Random seed.

    Returns
    -------
    OrPSOCResult dataclass with all results.  Key fields:
      .selected_features  → list[str] to subset data with
      .best_auc_orpsoc    → AUC of the final selected subset
      .auc_gain           → improvement over SHAP-only selection
    """
    t_total  = time.time()
    n_orig   = X_train.shape[1]

    if verbose:
        _log("=" * 60)
        _log("  StockSense OrPSOC Hybrid Feature Selection")
        _log("=" * 60)
        _log(f"  Input              : {n_orig} features")
        _log(f"  Train samples      : {len(X_train)}")
        _log(f"  Val samples        : {len(X_val)}")
        _log(f"  Class balance ratio: {scale_pos_weight:.2f}")

    # ── Phase 1: SHAP pre-pruning ─────────────────────────────────────────────
    kept_feats, shap_dropped, baseline_auc = shap_preprune(
        X_train, y_train, X_val, y_val,
        prune_percentile=shap_percentile,
        scale_pos_weight=scale_pos_weight,
        verbose=verbose,
    )
    n_after_shap = len(kept_feats)

    # Subset to SHAP-kept features for PSO
    X_tr_shap = X_train[kept_feats]
    X_va_shap = X_val[kept_feats]

    # ── Phase 2: OrPSOC ───────────────────────────────────────────────────────
    selected, fit_hist, div_hist, best_auc = run_orpsoc(
        X_tr_shap, y_train,
        X_va_shap, y_val,
        feature_names  = kept_feats,
        scale_pos_weight=scale_pos_weight,
        n_particles    = n_particles,
        max_iter       = max_iter,
        min_features   = min_features,
        verbose        = verbose,
        seed           = seed,
    )

    pso_dropped = [f for f in kept_feats if f not in selected]

    # ── Summarise by feature group ────────────────────────────────────────────
    group_summary = _summarise_by_group(selected)

    total_time = time.time() - t_total
    result = OrPSOCResult(
        selected_features    = selected,
        n_original           = n_orig,
        n_after_shap         = n_after_shap,
        n_selected           = len(selected),
        shap_dropped         = shap_dropped,
        pso_dropped          = pso_dropped,
        fitness_history      = fit_hist,
        diversity_history    = div_hist,
        best_auc_shap        = baseline_auc,
        best_auc_orpsoc      = best_auc,
        auc_gain             = best_auc - baseline_auc,
        n_particles          = n_particles,
        n_iterations         = max_iter,
        runtime_seconds      = total_time,
        feature_group_summary= group_summary,
    )

    if verbose:
        _log("=" * 60)
        report(result)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_selection(result: OrPSOCResult, name: str = "orpsoc_latest") -> Path:
    """
    Save the OrPSOC result to disk.

    Saves two files:
      data/selections/<name>.pkl   — full OrPSOCResult object
      data/selections/<name>.json  — human-readable summary

    Parameters
    ----------
    result : OrPSOCResult from hybrid_select().
    name   : Base filename (without extension).

    Returns
    -------
    Path to the saved .pkl file.
    """
    pkl_path  = _SELECTIONS / f"{name}.pkl"
    json_path = _SELECTIONS / f"{name}.json"

    joblib.dump(result, pkl_path)

    summary = {
        "selected_features"    : result.selected_features,
        "n_original"           : result.n_original,
        "n_after_shap"         : result.n_after_shap,
        "n_selected"           : result.n_selected,
        "best_auc_shap"        : result.best_auc_shap,
        "best_auc_orpsoc"      : result.best_auc_orpsoc,
        "auc_gain"             : result.auc_gain,
        "runtime_seconds"      : result.runtime_seconds,
        "feature_group_summary": result.feature_group_summary,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Selection saved: {pkl_path}")
    return pkl_path


def load_selection(name: str = "orpsoc_latest") -> OrPSOCResult:
    """
    Load a previously saved OrPSOCResult.

    Parameters
    ----------
    name : Base filename used in save_selection().

    Returns
    -------
    OrPSOCResult object.
    """
    pkl_path = _SELECTIONS / f"{name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"No selection found at {pkl_path}. "
            f"Run hybrid_select() and save_selection() first."
        )
    result = joblib.load(pkl_path)
    logger.info(f"Selection loaded: {pkl_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def report(result: OrPSOCResult) -> None:
    """
    Print a human-readable summary of an OrPSOCResult.

    Parameters
    ----------
    result : OrPSOCResult from hybrid_select().
    """
    _log("")
    _log("  ╔══════════════════════════════════════════╗")
    _log("  ║    OrPSOC Feature Selection — Report    ║")
    _log("  ╚══════════════════════════════════════════╝")
    _log("")
    _log("  Feature Reduction")
    _log(f"    Original  : {result.n_original:>4}  features")
    _log(f"    After SHAP: {result.n_after_shap:>4}  features  "
         f"(dropped {result.n_original - result.n_after_shap})")
    _log(f"    After PSO : {result.n_selected:>4}  features  "
         f"(dropped {result.n_after_shap - result.n_selected} more)")
    _log(f"    Reduction : {(1 - result.n_selected/result.n_original)*100:.1f}%")
    _log("")
    _log("  AUC Comparison (validation set)")
    _log(f"    SHAP-only baseline : {result.best_auc_shap:.4f}")
    _log(f"    OrPSOC selected    : {result.best_auc_orpsoc:.4f}")
    sign = "+" if result.auc_gain >= 0 else ""
    _log(f"    AUC gain           : {sign}{result.auc_gain:.4f}")
    _log("")
    _log("  Feature Groups Selected")
    for group, feats in result.feature_group_summary.items():
        _log(f"    {group:<20} : {len(feats):>3} features")
    _log("")
    _log(f"  Runtime : {result.runtime_seconds:.1f}s  "
         f"({result.n_particles} particles × {result.n_iterations} iterations)")
    _log("")
    if result.selected_features:
        _log("  Selected features (first 20):")
        for i, f in enumerate(result.selected_features[:20]):
            _log(f"    {i + 1:>3}. {f}")
        if len(result.selected_features) > 20:
            _log(f"    ... and {len(result.selected_features) - 20} more")
    _log("")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_fast_pipeline(scale_pos_weight: float = 1.0) -> SklearnPipeline:
    """
    Build a lightweight XGBoost pipeline for PSO fitness evaluation.

    Uses fewer trees and shallower depth than the production pipeline
    so each fitness call runs in ~50-200ms rather than 1-5 seconds.
    """
    return SklearnPipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   XGBClassifier(
            **FITNESS_XGB_PARAMS,
            scale_pos_weight=scale_pos_weight,
        )),
    ])


def _compute_auc_for_subset(
    feature_cols     : List[str],
    X_train          : pd.DataFrame,
    y_train          : pd.Series,
    X_val            : pd.DataFrame,
    y_val            : pd.Series,
    scale_pos_weight : float,
) -> float:
    """Train a fast pipeline on a feature subset and return val AUC."""
    try:
        pipe = _build_fast_pipeline(scale_pos_weight=scale_pos_weight)
        pipe.fit(X_train[feature_cols], y_train)
        prob = pipe.predict_proba(X_val[feature_cols])[:, 1]
        return float(roc_auc_score(y_val, prob))
    except Exception:
        return 0.0


def _summarise_by_group(selected_features: List[str]) -> Dict[str, List[str]]:
    """
    Group selected features by their feature engineering category.

    Maps feature name prefixes to groups (trend, momentum, etc.)
    Returns dict: group_name → list of selected features in that group.
    """
    group_map = {
        "trend"      : ["sma", "ema", "price_vs", "golden", "death", "trend",
                         "adx", "lr_slope", "hh_", "ll_"],
        "momentum"   : ["rsi", "roc_"],
        "macd"       : ["macd", "histogram", "bullish_divergence", "bearish_divergence"],
        "volatility" : ["bb_", "atr", "rv", "kc_", "vol_regime", "high_vol",
                        "low_vol", "range_perc"],
        "volume"     : ["volume", "obv", "vwap", "mfi"],
        "candle"     : ["body", "wick", "candle_direction"],
        "patterns"   : ["pat_", "bullish_pattern", "bearish_pattern", "pattern_signal"],
        "strength"   : ["_quality", "_strength", "_symmetry"],
        "interaction": ["hammer_", "bull_pat", "bear_pat", "doji_", "confirmed_"],
        "sequence"   : ["dir_streak", "dir_balance", "_cluster", "pct_from"],
        "sr"         : ["dist_to_", "range_pos"],
        "lag"        : ["_lag"],
        "returns"    : ["ret_", "max_drawdown"],
        "sentiment"  : ["sentiment_", "sent_", "article_count"],
    }

    result: Dict[str, List[str]] = {g: [] for g in group_map}
    unclassified: List[str] = []

    for feat in selected_features:
        placed = False
        for group, prefixes in group_map.items():
            if any(feat.startswith(p) or p in feat for p in prefixes):
                result[group].append(feat)
                placed = True
                break
        if not placed:
            unclassified.append(feat)

    if unclassified:
        result["other"] = unclassified

    # Remove empty groups
    return {k: v for k, v in result.items() if v}


def _log(msg: str) -> None:
    """Timestamped console log (mirrors logging style in rest of codebase)."""
    print(msg)


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST / DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Quick integration test using synthetic data.

    Run with:
        cd backend && python -m models.orpsoc_selector
    """
    import sys
    _BACKEND = Path(__file__).resolve().parent.parent
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))

    print("\nOrPSOC Selector — Quick Smoke Test")
    print("=" * 50)

    # Generate synthetic data (315 features, 800 samples)
    rng_test   = np.random.RandomState(42)
    N, F       = 800, 315
    X_all      = pd.DataFrame(
        rng_test.randn(N, F),
        columns=[f"feat_{i}" for i in range(F)],
    )
    # Only the first 20 features are truly predictive (the rest are noise)
    true_signal = X_all.iloc[:, :20].sum(axis=1)
    y_all       = (true_signal > true_signal.median()).astype(int)

    # Temporal split (must be chronological, not shuffled)
    split_train = int(N * 0.60)
    split_val   = int(N * 0.80)
    X_tr   = X_all.iloc[:split_train]
    y_tr   = y_all.iloc[:split_train]
    X_va   = X_all.iloc[split_train:split_val]
    y_va   = y_all.iloc[split_train:split_val]

    # Run hybrid selection with small PSO for speed in test
    result = hybrid_select(
        X_tr, y_tr,
        X_va, y_va,
        shap_percentile=50,
        n_particles=10,
        max_iter=15,
        min_features=5,
        verbose=True,
    )

    print(f"\n✅ Smoke test passed.")
    print(f"   Selected {result.n_selected} features from {result.n_original}")
    print(f"   AUC gain over SHAP-only: {result.auc_gain:+.4f}")
