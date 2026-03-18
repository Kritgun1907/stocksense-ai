"""
StockSense AI — nlp/finbert.py
================================
FinBERT sentiment model loader and inference engine.

This file owns:
  - Model loading and caching (singleton + inference cache)
  - Single and batch inference
  - Score conversion to ML-ready float
  - Model config registry for swapping models

It does NOT own:
  - News fetching              → data/news.py
  - Daily score aggregation    → nlp/sentiment.py  (Chapter 3.3)
  - Merging with prices        → data/merger.py

Why FinBERT over generic sentiment tools?
─────────────────────────────────────────────────────────────
  Financial language has domain-specific patterns that generic
  models misclassify:
    "Revenue declined less than feared"  → positive  (beat expectations)
    "Stock maintains steady performance" → positive  (stability valued)
    "Company guides below consensus"     → negative  (missed expectations)
  FinBERT was fine-tuned on Financial PhraseBank — 4,840 manually
  labelled financial sentences — giving it this domain awareness.
  Generic BERT or TextBlob score these incorrectly.

Why not GPT-4 / Claude for sentiment?
─────────────────────────────────────────────────────────────
  FinBERT: free, local, 440MB, 32 headlines/second, structured output.
  GPT-4:   $0.01/headline × 5000 headlines/day = $50/day, API round
           trips, unstructured text output needing parsing, data privacy
           concerns. For a classification task with a known domain,
           a fine-tuned specialist model beats a general LLM.

Why singleton model loading?
─────────────────────────────────────────────────────────────
  Loading FinBERT takes ~5 seconds and ~800MB RAM.
  Loading it fresh per call would make batch processing of 500 stocks
  impossibly slow. The singleton loads once at startup and is reused
  across all calls within the process lifetime.

Why inference caching?
─────────────────────────────────────────────────────────────
  Wire service headlines (Reuters, Bloomberg) are syndicated to dozens
  of outlets. In a 500-stock run, the same headline often appears for
  multiple tickers. Caching by headline hash eliminates redundant
  FinBERT forward passes — typically 20-30% cache hit rate in practice.

Why MIN_TEXT_LENGTH = 10?
─────────────────────────────────────────────────────────────
  Strings shorter than 10 characters are almost never meaningful
  financial headlines — they're metadata artifacts, formatting
  remnants, or API noise. Scoring them wastes compute and produces
  unreliable sentiment (FinBERT was not trained on fragments).
  10 characters is conservative — "Up 3%" (6 chars) gets excluded,
  "Sales up 3%" (11 chars) gets scored.

Why batch_size = 32?
─────────────────────────────────────────────────────────────
  32 is the largest batch that fits in CPU RAM without thrashing
  for BERT-base (110M params, 768 hidden dim). On GPU, increase
  to 64 or 128. Below 16, batch overhead exceeds parallelism gains.
  32 is the standard default for BERT inference in the literature.

Model details (default):
  HuggingFace ID : ProsusAI/finbert
  Architecture   : BERT-base (12 layers, 768 hidden, 12 heads)
  Parameters     : 110M
  Max input      : 512 tokens (~400 words)
  Output labels  : positive / negative / neutral
  Accuracy       : ~88% on Financial PhraseBank test set
"""

import hashlib
import warnings
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import os
from dotenv import load_dotenv

load_dotenv()   # loads .env into os.environ at import time
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL CONFIG REGISTRY
#  Mirrors LABEL_CONFIGS / ASSEMBLY_CONFIGS pattern from other modules.
#  Allows swapping sentiment models by name rather than hardcoding IDs.
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS: Dict[str, dict] = {
    "finbert": {
        "model_id":    "ProsusAI/finbert",
        "max_length":  512,
        "batch_size":  32,
        "labels":      ["positive", "negative", "neutral"],
        "description": "FinBERT — BERT fine-tuned on Financial PhraseBank. "
                       "Best for short financial headlines and news snippets.",
    },
    "finbert_tone": {
        "model_id":    "yiyanghkust/finbert-tone",
        "max_length":  512,
        "batch_size":  32,
        "labels":      ["positive", "negative", "neutral"],
        "description": "FinBERT-Tone — alternative FinBERT variant fine-tuned "
                       "on forward-looking financial statements.",
    },
    "distilbert_finance": {
        "model_id":    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        "max_length":  512,
        "batch_size":  64,   # smaller model → larger batches possible
        "labels":      ["positive", "negative", "neutral"],
        "description": "DistilRoBERTa finance — 40% smaller than FinBERT, "
                       "2× faster, ~2% lower accuracy. Good for high-volume runs.",
    },
    "finbert_custom": {
    "model_id":    os.path.join(_BASE_DIR, "finbert_finetuned"),  # relative to backend/
    "max_length":  512,
    "batch_size":  32,
    "labels":      ["positive", "negative", "neutral"],
    "description": "FinBERT fine-tuned on StockSense headlines. Test accuracy: 91.6%",
},
    
}

DEFAULT_MODEL = "finbert_custom"

# Minimum text length to bother scoring.
# Strings shorter than this are noise — not meaningful financial text.
# See module docstring for full reasoning.
MIN_TEXT_LENGTH = 10


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL SINGLETON STATE
#  One loaded model per process. Swapping model_name triggers a reload.
# ══════════════════════════════════════════════════════════════════════════════

_state: Dict = {
    "tokenizer":    None,
    "model":        None,
    "device":       None,
    "model_name":   None,   # which config is currently loaded
    "cache":        {},     # headline_hash → score dict
    "cache_hits":   0,
    "cache_misses": 0,
}


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_device() -> str:
    """
    Detect best available compute device.

    Priority: CUDA (Nvidia GPU) → MPS (Apple Silicon) → CPU.
    Why this order? CUDA is fastest for batch inference.
    MPS is available on M1/M2/M3 Macs and ~2× faster than CPU for BERT.
    CPU is the universal fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_label_index(model, label_name: str) -> int:
    """
    Find the output index for a label by name.

    Why not hardcode index 0=positive, 1=negative, 2=neutral?
    Different FinBERT variants use different label orderings.
    Hardcoding causes silent wrong predictions when switching models.
    This function is called once at load time and cached on the model object.

    Raises ValueError if label not found — fail loudly at load time,
    not silently during inference.
    """
    for idx, name in model.config.id2label.items():
        if name.lower() == label_name.lower():
            return idx
    raise ValueError(
        f"Label '{label_name}' not found in model config. "
        f"Available labels: {list(model.config.id2label.values())}. "
        f"Check that you are using a sentiment classification model."
    )


def _text_hash(text: str) -> str:
    """
    Generate a short hash key for inference caching.

    Why MD5 and not the raw string as key?
    Headlines can be 200+ characters. Using raw strings as dict keys
    wastes memory and slows dict lookups. MD5 hashes are 32 chars,
    consistent length, and collision probability is negligible for
    the ~5,000 unique headlines we process per day.
    """
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _neutral_result() -> Dict[str, float]:
    """
    Standard neutral result for invalid/empty/short texts.
    Centralised here so callers never construct this dict manually.
    """
    return {
        "score":    0.0,
        "positive": 0.0,
        "negative": 0.0,
        "neutral":  1.0,
        "label":    "neutral",
    }


def _probs_to_score(probs: torch.Tensor, model) -> torch.Tensor:
    """
    Convert a batch of probability tensors to scalar sentiment scores.

    Formula : score = P(positive) - P(negative)
    Range   : -1.0 (strongly negative) to +1.0 (strongly positive)

    Why exclude P(neutral)?
    ─────────────────────────────────────────────────────────────
    A headline that is 60% positive, 30% neutral, 10% negative
    should score higher than 45% positive, 45% neutral, 10% negative.
    P(positive) - P(negative) captures this directionality correctly.
    Including P(neutral) would dilute the directional signal — a text
    that is "very neutral" and one that is "slightly positive with
    some uncertainty" would look the same if neutral is factored in.

    Parameters
    ----------
    probs : torch.Tensor of shape (batch_size, num_labels)
    model : loaded model with _pos_idx and _neg_idx attributes

    Returns
    -------
    torch.Tensor of shape (batch_size,) with scores in [-1, +1]
    """
    return probs[:, model._pos_idx] - probs[:, model._neg_idx]


def _build_result(
    score: float,
    probs_row: torch.Tensor,
    model,
) -> Dict[str, float]:
    """
    Build a standardised result dict from a single probability row.
    Centralised to ensure consistent output format across all callers.
    """
    label = model.config.id2label[probs_row.argmax().item()]
    return {
        "score":    round(score, 4),
        "positive": round(probs_row[model._pos_idx].item(), 4),
        "negative": round(probs_row[model._neg_idx].item(), 4),
        "neutral":  round(probs_row[model._neu_idx].item(), 4),
        "label":    label,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_name: str = DEFAULT_MODEL,
    force_reload: bool = False,
) -> Tuple:
    """
    Load FinBERT tokenizer and model (singleton pattern).

    Downloads from HuggingFace on first call (~440MB for default).
    Subsequent calls with the same model_name return cached instance.
    Calling with a different model_name triggers a reload.

    Parameters
    ----------
    model_name   : Key from MODEL_CONFIGS. Default: 'finbert'.
    force_reload : Reload even if already loaded. Use if model seems corrupted.

    Returns
    -------
    (tokenizer, model, device) — all ready for inference.

    Raises
    ------
    ValueError : If model_name not found in MODEL_CONFIGS.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Model '{model_name}' not found in MODEL_CONFIGS. "
            f"Available: {list(MODEL_CONFIGS.keys())}. "
            f"Call list_model_configs() to inspect options."
        )

    # Return cached instance if same model already loaded
    already_loaded = (
        _state["tokenizer"] is not None and
        _state["model"] is not None and
        _state["model_name"] == model_name and
        not force_reload
    )
    if already_loaded:
        return _state["tokenizer"], _state["model"], _state["device"]

    cfg = MODEL_CONFIGS[model_name]
    model_id = cfg["model_id"]

    if _state["model_name"] is not None and _state["model_name"] != model_name:
        print(f"Switching model: {_state['model_name']} → {model_name}")
        # Clear inference cache — scores from old model are invalid
        _state["cache"].clear()
        _state["cache_hits"]   = 0
        _state["cache_misses"] = 0

    print(f"Loading {model_name} ({model_id})...")
    print(f"First load: downloads ~440MB. Subsequent loads: instant from cache.")

    device    = _get_device()
    hf_token  = os.getenv("HF_TOKEN")   # None if not set — HF handles that gracefully
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model     = AutoModelForSequenceClassification.from_pretrained(model_id, token=hf_token)
    model     = model.to(device)

    # Inference mode: disables dropout layers.
    # Why? Dropout randomly zeroes activations during training for regularisation.
    # During inference you want deterministic, reproducible outputs — no dropout.
    model.eval()

    # Cache label indices on model object — computed once, used millions of times
    model._pos_idx = _get_label_index(model, "positive")
    model._neg_idx = _get_label_index(model, "negative")
    model._neu_idx = _get_label_index(model, "neutral")

    # Store current config parameters on model for use in score_batch
    model._max_length  = cfg["max_length"]
    model._batch_size  = cfg["batch_size"]
    model._model_name  = model_name

    # Update singleton state
    _state["tokenizer"]  = tokenizer
    _state["model"]      = model
    _state["device"]     = device
    _state["model_name"] = model_name

    print(f"{model_name} loaded on {device.upper()}")
    print(f"Label map: {model.config.id2label}")

    return tokenizer, model, device


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def score_text(
    text: str,
    model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> Dict[str, float]:
    """
    Score a single text string for financial sentiment.

    Parameters
    ----------
    text       : Financial news headline or sentence.
    model_name : Model config key. Default: 'finbert'.
    use_cache  : Return cached result if this exact text was seen before.

    Returns
    -------
    dict with keys:
      score    : float in [-1, +1]  P(positive) - P(negative)
      positive : float in [0, 1]    P(positive)
      negative : float in [0, 1]    P(negative)
      neutral  : float in [0, 1]    P(neutral)
      label    : str                argmax label

    Texts shorter than MIN_TEXT_LENGTH return neutral (score=0.0).

    Example
    -------
    >>> score_text("Apple beats earnings expectations")
    {'score': 0.921, 'positive': 0.947, 'negative': 0.026,
     'neutral': 0.027, 'label': 'positive'}
    """
    # ── Validate and filter
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return _neutral_result()

    # ── Cache lookup
    if use_cache:
        key = _text_hash(text)
        if key in _state["cache"]:
            _state["cache_hits"] += 1
            return _state["cache"][key]
        _state["cache_misses"] += 1

    tokenizer, model, device = load_model(model_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=model._max_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs  = torch.softmax(outputs.logits, dim=-1)
    score  = _probs_to_score(probs, model).item()
    result = _build_result(score, probs[0], model)

    # ── Cache store
    if use_cache:
        _state["cache"][key] = result

    return result


def score_batch(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: Optional[int] = None,
    use_cache: bool = True,
    show_progress: bool = False,
) -> List[Dict[str, float]]:
    """
    Score a list of texts efficiently using batch inference.

    Why batch processing?
    ─────────────────────────────────────────────────────────────
    GPU and CPU matrix operations are fundamentally parallel.
    Processing 32 texts together in one forward pass takes almost
    the same wall-clock time as processing 1 text — the hardware
    runs all 32 in parallel. Sequential calls waste this parallelism.
    For 5,000 headlines, batching gives ~10× speedup on CPU.

    Parameters
    ----------
    texts        : List of financial news strings.
    model_name   : Model config key. Default: 'finbert'.
    batch_size   : Override default batch size from MODEL_CONFIGS.
    use_cache    : Skip already-scored texts (useful for repeated headlines).
    show_progress: Print per-batch progress for long runs.

    Returns
    -------
    List of score dicts in same order as input texts.
    Invalid/short texts return neutral (score=0.0).
    """
    if not texts:
        return []

    tokenizer, model, device = load_model(model_name)
    effective_batch = batch_size or model._batch_size

    results = [None] * len(texts)

    # ── Separate cached from uncached ────────────────────────────────────────
    # Why do this before batching?
    # Building the batch from only uncached texts means we don't waste
    # a forward pass on headlines we've already scored. On a 500-stock
    # run with many syndicated headlines, cache hit rate is typically 20-30%.
    uncached_indices = []
    uncached_texts   = []

    for i, text in enumerate(texts):
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            results[i] = _neutral_result()
            continue

        if use_cache:
            key = _text_hash(text)
            if key in _state["cache"]:
                results[i] = _state["cache"][key]
                _state["cache_hits"] += 1
                continue
            _state["cache_misses"] += 1

        uncached_indices.append(i)
        uncached_texts.append(text)

    if not uncached_texts:
        return results

    # ── Batch inference on uncached texts ────────────────────────────────────
    n_batches = (len(uncached_texts) + effective_batch - 1) // effective_batch

    for batch_idx in range(n_batches):
        start = batch_idx * effective_batch
        end   = min(start + effective_batch, len(uncached_texts))
        batch_texts   = uncached_texts[start:end]
        batch_indices = uncached_indices[start:end]

        if show_progress:
            print(f"  Scoring batch {batch_idx+1}/{n_batches} "
                  f"({start}–{end} of {len(uncached_texts)} uncached texts)")

        # Tokenise batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=model._max_length,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs  = torch.softmax(outputs.logits, dim=-1)
        scores = _probs_to_score(probs, model)

        # Store results and populate cache
        for j, (orig_idx, text) in enumerate(zip(batch_indices, batch_texts)):
            result = _build_result(scores[j].item(), probs[j], model)
            results[orig_idx] = result
            if use_cache:
                _state["cache"][_text_hash(text)] = result

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def warmup(model_name: str = DEFAULT_MODEL) -> None:
    """
    Pre-load the model and run one dummy inference to warm up JIT/caches.

    Why warmup?
    ─────────────────────────────────────────────────────────────
    The first real inference after model load is slower than subsequent
    ones because PyTorch JIT-compiles kernels on first use. Running a
    dummy forward pass at startup pays this cost before real requests arrive.
    Call this once in your FastAPI startup event (Phase 6).
    """
    load_model(model_name)
    _ = score_text("Warming up FinBERT inference pipeline.", model_name)
    print(f"Warmup complete for {model_name}.")


def get_cache_stats() -> Dict[str, int]:
    """
    Return inference cache hit/miss statistics.
    Useful for tuning batch processing and understanding headline redundancy.

    Returns
    -------
    dict with keys: hits, misses, cached_entries, hit_rate_pct
    """
    hits   = _state["cache_hits"]
    misses = _state["cache_misses"]
    total  = hits + misses
    return {
        "hits":           hits,
        "misses":         misses,
        "cached_entries": len(_state["cache"]),
        "hit_rate_pct":   round(hits / total * 100, 1) if total > 0 else 0.0,
    }


def clear_cache() -> None:
    """
    Clear the inference cache and reset statistics.
    Call between training runs if you want fresh scores.
    """
    _state["cache"].clear()
    _state["cache_hits"]   = 0
    _state["cache_misses"] = 0
    print("Inference cache cleared.")


def get_model_info() -> Dict:
    """
    Return information about the currently loaded model.
    Useful for logging, debugging, and API health checks.
    """
    if _state["model"] is None:
        return {"status": "not loaded", "model_name": None}

    model = _state["model"]
    cfg   = MODEL_CONFIGS[_state["model_name"]]

    return {
        "status":      "loaded",
        "model_name":  _state["model_name"],
        "model_id":    cfg["model_id"],
        "device":      _state["device"],
        "max_length":  model._max_length,
        "batch_size":  model._batch_size,
        "label_map":   model.config.id2label,
        "cache_stats": get_cache_stats(),
    }


def get_model_config(name: str) -> dict:
    """
    Retrieve a named model configuration from MODEL_CONFIGS.

    Raises ValueError if name not found.
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Model config '{name}' not found. "
            f"Available: {list(MODEL_CONFIGS.keys())}. "
            f"Call list_model_configs() to inspect."
        )
    return dict(MODEL_CONFIGS[name])


def list_model_configs(verbose: bool = True) -> list:
    """
    List all available model configurations.
    Returns list of config names. Prints summary table if verbose=True.
    """
    names = list(MODEL_CONFIGS.keys())
    if verbose:
        default_marker = "(default)"
        print(f"\n{'Model':<22} {'HuggingFace ID':<55}  Description")
        print("─" * 100)
        for name, cfg in MODEL_CONFIGS.items():
            marker = " ← default" if name == DEFAULT_MODEL else ""
            print(
                f"  {name:<20} {cfg['model_id']:<55}"
                f"  {cfg['description'][:50]}...{marker}"
            )
    return names


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── List available models
    list_model_configs()

    # ── Warmup
    warmup()

    # ── Test headlines covering tricky financial cases
    test_cases = [
        ("Apple beats Q4 earnings expectations by wide margin",   "positive"),
        ("iPhone sales decline sharply in China trade tensions",  "negative"),
        ("Revenue declined less than feared, shares rally",       "positive"),
        ("Company guides below analyst consensus next quarter",   "negative"),
        ("Management reaffirms full year guidance",               "neutral"),
        ("Stock maintains steady performance amid volatility",    "positive"),
        ("",                                                      "neutral"),
        ("Hi",                                                    "neutral"),
    ]

    print("\n=== Single text scoring ===")
    result = score_text(test_cases[0][0])
    print(f"Input:  {test_cases[0][0]}")
    print(f"Result: {result}")

    print("\n=== Batch scoring ===")
    texts   = [t for t, _ in test_cases]
    results = score_batch(texts, show_progress=True)

    print(f"\n{'Score':>7}  {'Label':<10}  {'Expected':<10}  Headline")
    print("─" * 80)
    for (headline, expected), res in zip(test_cases, results):
        match = "✅" if res["label"] == expected else "❌"
        display = headline if headline else "(empty)"
        print(f"{res['score']:>+7.3f}  {res['label']:<10}  "
              f"{expected:<10}{match}  {display[:45]}")

    print("\n=== Cache stats ===")
    print(get_cache_stats())

    print("\n=== Model info ===")
    print(get_model_info())

    print("\n=== Score same headlines again (should be all cache hits) ===")
    _ = score_batch(texts)
    print(get_cache_stats())
