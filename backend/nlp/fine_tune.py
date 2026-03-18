"""
StockSense AI — nlp/fine_tune.py
==================================
Fine-tuning pipeline for FinBERT on automatically fetched financial headlines.

This script owns:
  - Fetching financial headlines from NewsAPI (free tier) + yfinance
  - Auto-labelling headlines using the current FinBERT model as a weak supervisor
  - Class imbalance handling via weighted loss
  - Fine-tuning with early stopping + cosine LR scheduler
  - Weights & Biases experiment tracking
  - Saving fine-tuned model locally + pushing to HuggingFace Hub

It does NOT own:
  - Inference on production data        → nlp/finbert.py
  - Daily sentiment aggregation         → nlp/sentiment.py
  - News fetching for live scoring      → data/news.py

Pipeline overview:
─────────────────────────────────────────────────────────────
  1. Fetch 1000+ headlines from yfinance (free, no key needed)
     + NewsAPI free tier (100 req/day, ~20 headlines/req = ~2000/day)
  2. Deduplicate + clean headlines
  3. Auto-label with current FinBERT as weak supervisor
     (gives us a starting label; human review improves this further)
  4. Optionally merge with hand-labelled CSV if you have one
  5. Compute class weights to handle pos/neg/neutral imbalance
  6. Fine-tune FinBERT with:
       - Weighted CrossEntropyLoss  (handles imbalance)
       - Cosine LR schedule with warmup
       - Early stopping on eval loss
       - W&B logging for experiment tracking
  7. Evaluate on held-out test split
  8. Save locally to nlp/finbert_finetuned/
  9. Push to HuggingFace Hub (optional, controlled by PUSH_TO_HUB)

Why auto-labelling as weak supervision?
─────────────────────────────────────────────────────────────
  Hand-labelling 1000+ headlines takes hours. Auto-labelling with the
  existing FinBERT gives us a fast starting point. The fine-tuning then
  corrects systematic errors in those labels using the small set of
  hard-coded gold examples (GOLD_EXAMPLES below) as anchors.
  This is called "self-training" — a standard NLP technique.

Why yfinance + NewsAPI free tier?
─────────────────────────────────────────────────────────────
  yfinance.Ticker.news is completely free, no key required, and returns
  recent headlines for any ticker symbol. NewsAPI free tier gives 100
  requests/day × ~20 articles/request = ~2000 additional headlines/day.
  Combined, this easily exceeds 1000 unique headlines without any cost.

Why cosine LR schedule with warmup?
─────────────────────────────────────────────────────────────
  BERT-based models are sensitive to the learning rate. A linear warmup
  over the first 6% of steps prevents large gradient updates early in
  training from destroying pretrained weights. Cosine decay then smoothly
  reduces the LR to near-zero, consistently outperforming linear decay
  on small fine-tuning datasets in the literature.

Why early stopping?
─────────────────────────────────────────────────────────────
  With only 1000-2000 examples, overfitting can occur within 3-4 epochs.
  Early stopping monitors eval loss and halts training when it stops
  improving, saving the best checkpoint automatically.

Why class weights?
─────────────────────────────────────────────────────────────
  Financial news is heavily skewed: ~50% neutral, ~30% negative, ~20%
  positive in practice. Without correction, the model learns to predict
  "neutral" for everything and achieves deceptively high accuracy.
  Inverse-frequency class weights force the model to pay equal attention
  to all three classes.

Usage:
─────────────────────────────────────────────────────────────
  # Install dependencies first:
  pip install torch transformers datasets wandb yfinance newsapi-python python-dotenv

  # Set environment variables in .env:
  HF_TOKEN=hf_...
  NEWSAPI_KEY=...         (optional but recommended)
  WANDB_API_KEY=...       (optional — skips W&B if not set)
  HF_USERNAME=...         (required only if PUSH_TO_HUB=true)

  # Run:
  python3 nlp/fine_tune.py

  # After training, update DEFAULT_MODEL in finbert.py:
  DEFAULT_MODEL = "finbert_custom"
  and add to MODEL_CONFIGS (instructions printed at end of script).
"""

import os
import sys
import time
import hashlib
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dotenv import load_dotenv

# ── Suppress noisy HuggingFace warnings ──────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
#  All tuneable parameters live here. Do not hardcode values deeper in the
#  script — change them here so diffs are easy to read in git.
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # ── Model ────────────────────────────────────────────────────────────────
    "base_model_id":    "ProsusAI/finbert",
    "num_labels":       3,
    "label2id":         {"positive": 0, "negative": 1, "neutral": 2},
    "id2label":         {0: "positive", 1: "negative", 2: "neutral"},

    # ── Output paths ─────────────────────────────────────────────────────────
    "output_dir":       "nlp/finbert_finetuned",
    "checkpoint_dir":   "nlp/finbert_checkpoints",
    "log_dir":          "nlp/logs",

    # ── HuggingFace Hub ──────────────────────────────────────────────────────
    # Set PUSH_TO_HUB=True to push after training.
    # Repo will be created as {HF_USERNAME}/{HF_REPO_NAME}.
    "push_to_hub":      os.getenv("PUSH_TO_HUB", "false").lower() == "true",
    "hf_repo_name":     "stocksense-finbert",

    # ── Data ─────────────────────────────────────────────────────────────────
    "target_headlines": 1200,       # fetch until we have this many unique headlines
    "min_text_length":  15,         # characters — shorter texts are noise
    "max_text_length":  512,        # tokens — hard limit for BERT
    "test_split":       0.15,       # 15% held out for final evaluation
    "val_split":        0.10,       # 10% used for early stopping

    # ── Training ─────────────────────────────────────────────────────────────
    "num_epochs":       10,         # early stopping will halt before this if needed
    "batch_size":       16,         # safe for MPS/CPU; increase to 32 on GPU
    "learning_rate":    2e-5,       # standard sweet spot for BERT fine-tuning
    "weight_decay":     0.01,       # L2 regularisation
    "warmup_ratio":     0.06,       # fraction of total steps used for LR warmup
    "max_grad_norm":    1.0,        # gradient clipping — prevents exploding gradients
    "early_stop_patience": 3,       # stop after N epochs with no eval loss improvement

    # ── Logging ──────────────────────────────────────────────────────────────
    "wandb_project":    "stocksense-finbert",
    "log_every_n_steps": 20,

    # ── Tickers for yfinance headline fetching ────────────────────────────────
    # Covers large caps across sectors for diverse training data.
    "tickers": [
        # Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
        "INTC", "CRM", "ORCL", "ADBE", "NFLX", "PYPL", "UBER",
        # Finance
        "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP",
        # Healthcare
        "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "BMY",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG",
        # Consumer
        "WMT", "TGT", "COST", "MCD", "SBUX", "NKE", "PG", "KO", "PEP",
        # Industrial
        "BA", "CAT", "GE", "HON", "MMM", "UPS", "FDX",
        # Index ETFs (broad market news)
        "SPY", "QQQ", "DIA",
    ],

    # ── NewsAPI query terms ───────────────────────────────────────────────────
    "newsapi_queries": [
        "stock earnings results",
        "company revenue guidance",
        "quarterly profits analyst",
        "stock market financial results",
        "corporate earnings beat miss",
        "dividend acquisition merger",
        "stock downgrade upgrade analyst",
        "revenue declined below expectations",
        "earnings exceeded forecasts",
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
#  GOLD EXAMPLES
#  Hand-labelled anchors that the model MUST get right.
#  These are included in every training epoch with 3× oversampling to
#  ensure the fine-tuned model doesn't regress on these known hard cases.
#
#  Add more here as you find cases where the model fails in production.
# ══════════════════════════════════════════════════════════════════════════════

GOLD_EXAMPLES = [
    # ── Expectation-relative positives (FinBERT's known weakness) ────────────
    {"text": "Revenue declined less than feared, shares rally",          "label": 0},
    {"text": "Loss narrowed more than expected, stock jumps",            "label": 0},
    {"text": "Earnings fell but beat lowered analyst expectations",      "label": 0},
    {"text": "Sales dropped less than feared in challenging quarter",    "label": 0},
    {"text": "Profit decline smaller than consensus estimate",           "label": 0},
    {"text": "Results better than feared as margins held up",            "label": 0},
    {"text": "Stock surges after smaller than expected loss reported",   "label": 0},
    {"text": "Quarterly revenue topped reduced Wall Street estimates",   "label": 0},

    # ── Expectation-relative negatives ───────────────────────────────────────
    {"text": "Company guides below analyst consensus next quarter",      "label": 1},
    {"text": "Sales fell short of already lowered expectations",         "label": 1},
    {"text": "Beat on revenue but missed badly on profit margins",       "label": 1},
    {"text": "Raised guidance still below what analysts had hoped",      "label": 1},
    {"text": "Results in line but forward outlook disappoints market",   "label": 1},
    {"text": "Topped estimates but shares fall on weak guidance",        "label": 1},

    # ── Stability and maintenance (often misclassified) ──────────────────────
    {"text": "Management reaffirms full year guidance unchanged",        "label": 2},
    {"text": "Stock maintains steady performance amid volatility",       "label": 0},
    {"text": "Company holds dividend steady at current level",           "label": 2},
    {"text": "Board reiterates long term growth targets",                "label": 2},

    # ── Clear positives ──────────────────────────────────────────────────────
    {"text": "Apple beats Q4 earnings expectations by wide margin",      "label": 0},
    {"text": "Revenue exceeds analyst forecasts on strong demand",       "label": 0},
    {"text": "Dividend increased for 10th consecutive year",             "label": 0},
    {"text": "Company raises full year guidance above consensus",        "label": 0},
    {"text": "Record quarterly profit driven by strong iPhone sales",    "label": 0},
    {"text": "Stock hits all time high after blowout earnings report",   "label": 0},

    # ── Clear negatives ──────────────────────────────────────────────────────
    {"text": "iPhone sales decline sharply in China trade tensions",     "label": 1},
    {"text": "Operating margins decline sharply on rising costs",        "label": 1},
    {"text": "Profits fall well below market consensus estimates",       "label": 1},
    {"text": "Company slashes dividend amid cash flow concerns",         "label": 1},
    {"text": "Revenue misses estimates as consumer demand weakens",      "label": 1},
    {"text": "CEO resigns amid accounting irregularities investigation", "label": 1},
    {"text": "Layoffs announced as company restructures operations",     "label": 1},
]


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs(CFG["log_dir"], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(CFG["log_dir"], f"finetune_{datetime.now():%Y%m%d_%H%M%S}.log")
        ),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — HEADLINE FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_yfinance_headlines(tickers: List[str]) -> List[str]:
    """
    Fetch recent financial headlines from yfinance.

    Why yfinance?
    ─────────────────────────────────────────────────────────────
    yfinance.Ticker.news is completely free — no API key, no rate limit
    beyond polite usage. Each ticker returns ~10-20 recent headlines.
    With 60+ tickers we easily get 600-1200 headlines in one pass.

    Returns
    -------
    List of raw headline strings (not deduplicated yet).
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed. Run: pip install yfinance")
        return []

    headlines = []
    log.info(f"Fetching headlines from yfinance for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        try:
            news = yf.Ticker(ticker).news
            for item in news:
                # yfinance returns dicts with 'content' key containing headline info
                content = item.get("content", {})
                title = (
                    content.get("title")                    # newer yfinance versions
                    or item.get("title")                    # older versions
                    or ""
                )
                if title:
                    headlines.append(title.strip())

            # Polite delay — avoid hammering Yahoo Finance
            if i % 10 == 9:
                time.sleep(1.0)
                log.info(f"  yfinance: {i+1}/{len(tickers)} tickers fetched, "
                         f"{len(headlines)} headlines so far")

        except Exception as e:
            log.warning(f"  yfinance fetch failed for {ticker}: {e}")
            continue

    log.info(f"yfinance: fetched {len(headlines)} raw headlines")
    return headlines


def fetch_newsapi_headlines(queries: List[str], api_key: str) -> List[str]:
    """
    Fetch financial headlines from NewsAPI free tier.

    Why NewsAPI?
    ─────────────────────────────────────────────────────────────
    NewsAPI free tier allows 100 requests/day × ~20 articles/request.
    Using targeted financial queries ensures relevance.
    We stay well under the 100/day limit with our ~9 queries.

    Parameters
    ----------
    queries : List of search query strings.
    api_key : NewsAPI key from NEWSAPI_KEY env var.

    Returns
    -------
    List of raw headline strings.
    """
    try:
        from newsapi import NewsApiClient
    except ImportError:
        log.warning("newsapi-python not installed. Run: pip install newsapi-python")
        log.warning("Skipping NewsAPI fetch — yfinance headlines only.")
        return []

    headlines = []
    client = NewsApiClient(api_key=api_key)
    log.info(f"Fetching headlines from NewsAPI for {len(queries)} queries...")

    for query in queries:
        try:
            response = client.get_everything(
                q=query,
                language="en",
                sort_by="relevancy",
                page_size=100,      # max per request on free tier
            )
            for article in response.get("articles", []):
                title = article.get("title", "").strip()
                # NewsAPI sometimes returns "[Removed]" for deleted articles
                if title and title != "[Removed]" and len(title) > CFG["min_text_length"]:
                    headlines.append(title)

            time.sleep(0.5)     # stay within free tier rate limits

        except Exception as e:
            log.warning(f"  NewsAPI query failed '{query}': {e}")
            continue

    log.info(f"NewsAPI: fetched {len(headlines)} raw headlines")
    return headlines


def deduplicate_headlines(headlines: List[str]) -> List[str]:
    """
    Deduplicate headlines by MD5 hash of lowercased, stripped text.

    Why hash-based dedup?
    ─────────────────────────────────────────────────────────────
    Wire service headlines are syndicated verbatim to dozens of outlets.
    Simple set() on raw strings misses near-duplicates with minor
    punctuation differences. Lowercasing + stripping before hashing
    catches most of these. For a training set, exact dedup is sufficient
    — we don't need fuzzy dedup at this stage.
    """
    seen   = set()
    unique = []
    for h in headlines:
        key = hashlib.md5(h.strip().lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(h.strip())
    log.info(f"Deduplication: {len(headlines)} → {len(unique)} unique headlines")
    return unique


def filter_headlines(headlines: List[str]) -> List[str]:
    """
    Remove headlines that are too short, too long, or clearly non-financial.

    Filters applied:
      - Minimum length: CFG['min_text_length'] characters
      - Skip headlines that are clearly clickbait/non-financial noise
        (e.g. "Watch this video", "See more stories")
    """
    noise_patterns = [
        "watch this", "see more", "click here", "read more",
        "subscribe to", "sign up", "breaking news",
    ]
    filtered = []
    for h in headlines:
        if len(h) < CFG["min_text_length"]:
            continue
        lower = h.lower()
        if any(p in lower for p in noise_patterns):
            continue
        filtered.append(h)

    log.info(f"Filtering: {len(headlines)} → {len(filtered)} headlines after noise removal")
    return filtered


def fetch_all_headlines() -> List[str]:
    """
    Orchestrate full headline fetch pipeline.

    Order:
      1. yfinance (free, no key needed)
      2. NewsAPI (free tier, uses NEWSAPI_KEY from .env if set)
      3. Deduplicate + filter
      4. Trim to CFG['target_headlines'] if we have more than needed

    Returns
    -------
    List of clean, unique, filtered headline strings.
    """
    all_headlines = []

    # ── yfinance ─────────────────────────────────────────────────────────────
    yf_headlines = fetch_yfinance_headlines(CFG["tickers"])
    all_headlines.extend(yf_headlines)

    # ── NewsAPI (optional) ────────────────────────────────────────────────────
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        na_headlines = fetch_newsapi_headlines(CFG["newsapi_queries"], newsapi_key)
        all_headlines.extend(na_headlines)
    else:
        log.warning("NEWSAPI_KEY not set in .env — skipping NewsAPI fetch.")
        log.warning("Add NEWSAPI_KEY=your_key to .env for more training data.")
        log.warning("Free key at: https://newsapi.org/register")

    # ── Clean ─────────────────────────────────────────────────────────────────
    all_headlines = deduplicate_headlines(all_headlines)
    all_headlines = filter_headlines(all_headlines)

    # ── Trim if over target ───────────────────────────────────────────────────
    if len(all_headlines) > CFG["target_headlines"]:
        # Shuffle before trimming for random coverage across tickers
        import random
        random.shuffle(all_headlines)
        all_headlines = all_headlines[:CFG["target_headlines"]]
        log.info(f"Trimmed to {CFG['target_headlines']} headlines (target reached)")

    log.info(f"Final headline count: {len(all_headlines)}")
    if len(all_headlines) < 200:
        log.warning(
            f"Only {len(all_headlines)} headlines fetched — this may be too few "
            f"for meaningful fine-tuning. Consider adding NEWSAPI_KEY to .env."
        )

    return all_headlines


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — AUTO-LABELLING (WEAK SUPERVISION)
# ══════════════════════════════════════════════════════════════════════════════

def auto_label_headlines(headlines: List[str]) -> List[Dict]:
    """
    Use the existing FinBERT model as a weak supervisor to auto-label headlines.

    Why weak supervision instead of hand-labelling?
    ─────────────────────────────────────────────────────────────
    Hand-labelling 1000+ headlines takes ~4 hours of focused work.
    Auto-labelling with FinBERT takes ~2 minutes and gives us a
    reasonable starting point. The fine-tuning process then corrects
    systematic errors using the GOLD_EXAMPLES as anchors.

    Confidence threshold:
    ─────────────────────────────────────────────────────────────
    We only keep headlines where FinBERT's top label probability
    exceeds 0.65. Low-confidence labels introduce noise that can
    hurt fine-tuning more than having fewer, cleaner examples.

    Returns
    -------
    List of dicts: [{"text": str, "label": int, "confidence": float}, ...]
    """
    # Import here to avoid circular dependency if fine_tune is imported by finbert
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    hf_token   = os.getenv("HF_TOKEN")
    model_id   = CFG["base_model_id"]
    label2id   = CFG["label2id"]
    confidence_threshold = 0.65

    log.info(f"Auto-labelling {len(headlines)} headlines with FinBERT...")
    log.info(f"Confidence threshold: {confidence_threshold} (lower confidence discarded)")

    device    = "mps" if (hasattr(torch.backends, "mps") and
                          torch.backends.mps.is_available()) else \
                "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_id, token=hf_token
    ).to(device)
    model.eval()

    # Build label index map from model config
    id2label_model = model.config.id2label    # {0: 'positive', 1: 'negative', 2: 'neutral'}
    label2id_model = {v.lower(): k for k, v in id2label_model.items()}

    labelled    = []
    discarded   = 0
    batch_size  = 32

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs      = torch.softmax(outputs.logits, dim=-1)
        confidences, pred_indices = probs.max(dim=-1)

        for j, (headline, conf, pred_idx) in enumerate(
            zip(batch, confidences.tolist(), pred_indices.tolist())
        ):
            if conf < confidence_threshold:
                discarded += 1
                continue

            pred_label_name = id2label_model[pred_idx].lower()
            # Map to our canonical label space (0=pos, 1=neg, 2=neu)
            canonical_label = label2id[pred_label_name]

            labelled.append({
                "text":       headline,
                "label":      canonical_label,
                "confidence": round(conf, 4),
                "source":     "auto",
            })

        if (i // batch_size) % 10 == 9:
            log.info(f"  Auto-labelled {min(i + batch_size, len(headlines))}"
                     f"/{len(headlines)} headlines...")

    # Free model from memory before training loads it again
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    log.info(f"Auto-labelling complete: {len(labelled)} kept, "
             f"{discarded} discarded (confidence < {confidence_threshold})")

    return labelled


def build_dataset(headlines: List[str]) -> List[Dict]:
    """
    Build the full training dataset by combining:
      1. Auto-labelled fetched headlines
      2. Gold examples (3× oversampled to act as anchors)

    Why oversample gold examples?
    ─────────────────────────────────────────────────────────────
    Gold examples are high-quality, hand-verified labels. Oversampling
    them 3× ensures the model sees them proportionally more during
    training, acting as anchors that prevent regression on known-hard cases.
    3× is a reasonable multiplier — high enough to matter, low enough
    that gold examples don't dominate and cause overfitting on 30 sentences.

    Returns
    -------
    Shuffled list of {"text": str, "label": int} dicts.
    """
    import random

    auto_labelled = auto_label_headlines(headlines)

    # Oversample gold examples 3×
    gold_oversampled = []
    for example in GOLD_EXAMPLES:
        for _ in range(3):
            gold_oversampled.append({
                "text":   example["text"],
                "label":  example["label"],
                "source": "gold",
            })

    full_dataset = auto_labelled + gold_oversampled
    random.shuffle(full_dataset)

    # Log class distribution
    from collections import Counter
    id2label = CFG["id2label"]
    counts   = Counter(d["label"] for d in full_dataset)
    log.info("Dataset class distribution:")
    for label_id, count in sorted(counts.items()):
        pct = count / len(full_dataset) * 100
        log.info(f"  {id2label[label_id]:<10}: {count:>5} ({pct:.1f}%)")
    log.info(f"  Total         : {len(full_dataset)}")

    return full_dataset


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — DATASET SPLITS + PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class HeadlineDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping tokenised financial headlines.

    Why a custom Dataset instead of HuggingFace datasets.Dataset?
    ─────────────────────────────────────────────────────────────
    A custom Dataset gives us full control over tokenisation behaviour
    and makes it easy to integrate with our custom training loop.
    The HuggingFace Trainer API would work too, but a manual loop lets
    us implement custom weighted loss and gradient clipping more cleanly.
    """

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item  = self.examples[idx]
        enc   = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(item["label"], dtype=torch.long),
        }


def split_dataset(
    examples: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train / validation / test sets.

    Why stratified split?
    ─────────────────────────────────────────────────────────────
    Random splitting on an imbalanced dataset can leave one class
    with 0 examples in the val or test set. Stratified splitting
    ensures each split has the same class proportions as the full set.

    Returns
    -------
    (train_examples, val_examples, test_examples)
    """
    from collections import defaultdict
    import random

    # Group by label
    by_label = defaultdict(list)
    for ex in examples:
        by_label[ex["label"]].append(ex)

    train_ex, val_ex, test_ex = [], [], []

    for label_id, label_examples in by_label.items():
        random.shuffle(label_examples)
        n        = len(label_examples)
        n_test   = max(1, int(n * CFG["test_split"]))
        n_val    = max(1, int(n * CFG["val_split"]))
        n_train  = n - n_test - n_val

        train_ex.extend(label_examples[:n_train])
        val_ex.extend(label_examples[n_train:n_train + n_val])
        test_ex.extend(label_examples[n_train + n_val:])

    random.shuffle(train_ex)
    random.shuffle(val_ex)
    random.shuffle(test_ex)

    log.info(f"Dataset splits — train: {len(train_ex)}, "
             f"val: {len(val_ex)}, test: {len(test_ex)}")

    return train_ex, val_ex, test_ex


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — CLASS WEIGHTS (IMBALANCE HANDLING)
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(train_examples: List[Dict]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted CrossEntropyLoss.

    Formula: weight_c = total_samples / (num_classes × count_c)

    Why inverse frequency?
    ─────────────────────────────────────────────────────────────
    If 50% of headlines are neutral, 30% negative, 20% positive:
      weight_neutral  = 1.0 / 0.50 = 2.0
      weight_negative = 1.0 / 0.30 = 3.3
      weight_positive = 1.0 / 0.20 = 5.0
    The loss for misclassifying a rare positive is penalised 2.5×
    more than misclassifying a neutral — forcing the model to pay
    attention to minority classes.

    Returns
    -------
    torch.Tensor of shape (num_labels,) with per-class weights.
    """
    from collections import Counter
    counts   = Counter(ex["label"] for ex in train_examples)
    n_total  = len(train_examples)
    n_classes = CFG["num_labels"]

    weights = []
    id2label = CFG["id2label"]
    for label_id in range(n_classes):
        count  = counts.get(label_id, 1)    # avoid division by zero
        weight = n_total / (n_classes * count)
        weights.append(weight)
        log.info(f"  Class weight [{id2label[label_id]}]: {weight:.3f} "
                 f"(n={count})")

    return torch.tensor(weights, dtype=torch.float)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """
    Cosine LR schedule with linear warmup.

    Why cosine + warmup?
    ─────────────────────────────────────────────────────────────
    Warmup prevents large gradient updates in the first steps from
    destroying pretrained weights — a well-known BERT fine-tuning
    failure mode. Cosine decay then smoothly reduces LR to near-zero,
    consistently outperforming step decay and linear decay on small
    fine-tuning datasets.
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def evaluate(
    model,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Run evaluation on a dataloader.

    Returns
    -------
    (avg_loss, accuracy)
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)

            total_loss    += loss.item() * len(labels)
            preds          = outputs.logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train(
    model,
    tokenizer,
    train_examples: List[Dict],
    val_examples:   List[Dict],
    class_weights:  torch.Tensor,
    device:         str,
) -> None:
    """
    Main fine-tuning loop.

    Features:
      - Weighted CrossEntropyLoss for class imbalance
      - AdamW optimiser (standard for BERT)
      - Cosine LR schedule with linear warmup
      - Gradient clipping (max_grad_norm)
      - Early stopping on validation loss
      - W&B logging (if WANDB_API_KEY is set)
      - Checkpoint saving every epoch + best model tracking
    """
    from torch.optim import AdamW

    # ── W&B setup ─────────────────────────────────────────────────────────────
    use_wandb = bool(os.getenv("WANDB_API_KEY"))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=CFG["wandb_project"],
                name=f"finetune_{datetime.now():%Y%m%d_%H%M%S}",
                config=CFG,
            )
            log.info("W&B logging enabled.")
        except ImportError:
            log.warning("wandb not installed. Run: pip install wandb")
            use_wandb = False
    else:
        log.info("WANDB_API_KEY not set — skipping W&B logging.")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dataset = HeadlineDataset(train_examples, tokenizer, CFG["max_text_length"])
    val_dataset   = HeadlineDataset(val_examples,   tokenizer, CFG["max_text_length"])

    train_loader  = DataLoader(
        train_dataset, batch_size=CFG["batch_size"], shuffle=True, num_workers=0
    )
    val_loader    = DataLoader(
        val_dataset, batch_size=CFG["batch_size"], shuffle=False, num_workers=0
    )

    # ── Loss function with class weights ─────────────────────────────────────
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Why AdamW and not Adam?
    # AdamW decouples weight decay from the gradient update, which is
    # theoretically correct and empirically better for BERT fine-tuning.
    optimizer = AdamW(
        model.parameters(),
        lr=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    total_steps   = len(train_loader) * CFG["num_epochs"]
    warmup_steps  = int(total_steps * CFG["warmup_ratio"])
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log.info(f"Training config:")
    log.info(f"  Epochs         : {CFG['num_epochs']} (max, early stopping active)")
    log.info(f"  Train batches  : {len(train_loader)} per epoch")
    log.info(f"  Total steps    : {total_steps}")
    log.info(f"  Warmup steps   : {warmup_steps}")
    log.info(f"  Learning rate  : {CFG['learning_rate']}")
    log.info(f"  Device         : {device.upper()}")

    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_loss     = float("inf")
    patience_counter  = 0
    best_model_path   = None
    global_step       = 0

    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CFG["output_dir"],     exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, CFG["num_epochs"] + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_samples = 0

        log.info(f"\n{'─'*60}")
        log.info(f"EPOCH {epoch}/{CFG['num_epochs']}")
        log.info(f"{'─'*60}")

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])

            optimizer.step()
            scheduler.step()

            preds          = outputs.logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_samples += len(labels)
            epoch_loss    += loss.item() * len(labels)
            global_step   += 1

            # Per-step logging
            if global_step % CFG["log_every_n_steps"] == 0:
                step_acc = epoch_correct / epoch_samples
                step_loss = epoch_loss / epoch_samples
                current_lr = scheduler.get_last_lr()[0]
                log.info(
                    f"  Step {global_step:>5} | "
                    f"loss: {step_loss:.4f} | "
                    f"acc: {step_acc:.3f} | "
                    f"lr: {current_lr:.2e}"
                )
                if use_wandb:
                    wandb.log({
                        "train/loss": step_loss,
                        "train/accuracy": step_acc,
                        "train/learning_rate": current_lr,
                        "epoch": epoch,
                    }, step=global_step)

        # ── Epoch-end evaluation ──────────────────────────────────────────────
        train_loss = epoch_loss / epoch_samples
        train_acc  = epoch_correct / epoch_samples
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        log.info(f"\n  Epoch {epoch} summary:")
        log.info(f"    Train loss : {train_loss:.4f}  |  Train acc : {train_acc:.3f}")
        log.info(f"    Val loss   : {val_loss:.4f}  |  Val acc   : {val_acc:.3f}")

        if use_wandb:
            wandb.log({
                "epoch/train_loss":  train_loss,
                "epoch/train_acc":   train_acc,
                "epoch/val_loss":    val_loss,
                "epoch/val_acc":     val_acc,
            }, step=global_step)

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt_path = os.path.join(CFG["checkpoint_dir"], f"epoch_{epoch:02d}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        log.info(f"  Checkpoint saved: {ckpt_path}")

        # ── Early stopping check ──────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_model_path  = ckpt_path
            log.info(f"  ✅ New best val loss: {best_val_loss:.4f} — saving as best")
        else:
            patience_counter += 1
            log.info(
                f"  ⏳ No improvement. Patience: {patience_counter}"
                f"/{CFG['early_stop_patience']}"
            )
            if patience_counter >= CFG["early_stop_patience"]:
                log.info(
                    f"\n  🛑 Early stopping triggered at epoch {epoch}. "
                    f"Best epoch had val loss: {best_val_loss:.4f}"
                )
                break

    if use_wandb:
        wandb.finish()

    return best_model_path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def final_evaluation(
    model,
    tokenizer,
    test_examples: List[Dict],
    device: str,
) -> Dict:
    """
    Run final evaluation on the held-out test set.
    Reports per-class precision, recall, F1 alongside overall accuracy.

    Why per-class metrics?
    ─────────────────────────────────────────────────────────────
    Overall accuracy is misleading on imbalanced datasets — a model
    that always predicts "neutral" can achieve 50% accuracy. Per-class
    F1 reveals whether the model is actually learning all three classes.
    """
    from collections import defaultdict

    test_dataset = HeadlineDataset(test_examples, tokenizer, CFG["max_text_length"])
    test_loader  = DataLoader(
        test_dataset, batch_size=CFG["batch_size"], shuffle=False, num_workers=0
    )

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Per-class metrics
    id2label    = CFG["id2label"]
    n_classes   = CFG["num_labels"]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            tp[label] += 1
        else:
            fp[pred]  += 1
            fn[label] += 1

    overall_correct = sum(tp.values())
    overall_total   = len(all_labels)
    overall_acc     = overall_correct / overall_total

    log.info(f"\n{'═'*60}")
    log.info("FINAL TEST SET EVALUATION")
    log.info(f"{'═'*60}")
    log.info(f"  Overall accuracy: {overall_acc:.3f} ({overall_correct}/{overall_total})")
    log.info(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    log.info(f"  {'─'*44}")

    metrics = {"overall_accuracy": overall_acc, "per_class": {}}

    for label_id in range(n_classes):
        precision = tp[label_id] / max(1, tp[label_id] + fp[label_id])
        recall    = tp[label_id] / max(1, tp[label_id] + fn[label_id])
        f1        = 2 * precision * recall / max(1e-9, precision + recall)
        name      = id2label[label_id]

        log.info(f"  {name:<12} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")
        metrics["per_class"][name] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
        }

    # ── Gold examples check ───────────────────────────────────────────────────
    log.info(f"\n  Gold examples check (these MUST pass):")
    log.info(f"  {'Label':<12} {'Predicted':<12}  {'Text'}")
    log.info(f"  {'─'*70}")

    gold_pass = 0
    for ex in GOLD_EXAMPLES:
        inputs = tokenizer(
            ex["text"], return_tensors="pt",
            truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out   = model(**inputs)
        pred  = out.logits.argmax(dim=-1).item()
        match = "✅" if pred == ex["label"] else "❌"
        if pred == ex["label"]:
            gold_pass += 1
        log.info(
            f"  {match} {id2label[ex['label']]:<12} "
            f"{id2label[pred]:<12}  {ex['text'][:50]}"
        )

    gold_acc = gold_pass / len(GOLD_EXAMPLES)
    log.info(f"\n  Gold accuracy: {gold_pass}/{len(GOLD_EXAMPLES)} ({gold_acc:.1%})")
    metrics["gold_accuracy"] = gold_acc

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — SAVE + PUSH TO HUB
# ══════════════════════════════════════════════════════════════════════════════

def save_and_push(model, tokenizer) -> None:
    """
    Save the fine-tuned model locally and optionally push to HuggingFace Hub.

    Local save: CFG['output_dir']  (nlp/finbert_finetuned/)
    Hub push:   {HF_USERNAME}/{HF_REPO_NAME}  (if PUSH_TO_HUB=true)
    """
    # ── Local save ────────────────────────────────────────────────────────────
    os.makedirs(CFG["output_dir"], exist_ok=True)
    model.save_pretrained(CFG["output_dir"])
    tokenizer.save_pretrained(CFG["output_dir"])
    log.info(f"Fine-tuned model saved locally: {CFG['output_dir']}/")

    # ── Hub push ──────────────────────────────────────────────────────────────
    if CFG["push_to_hub"]:
        hf_username = os.getenv("HF_USERNAME")
        hf_token    = os.getenv("HF_TOKEN")

        if not hf_username:
            log.warning(
                "PUSH_TO_HUB=true but HF_USERNAME not set in .env. "
                "Add HF_USERNAME=your_username to .env and re-run."
            )
            return

        repo_id = f"{hf_username}/{CFG['hf_repo_name']}"
        log.info(f"Pushing model to HuggingFace Hub: {repo_id}...")
        try:
            model.push_to_hub(repo_id, token=hf_token)
            tokenizer.push_to_hub(repo_id, token=hf_token)
            log.info(f"✅ Model pushed to: https://huggingface.co/{repo_id}")
        except Exception as e:
            log.error(f"Hub push failed: {e}")
            log.error("Model is still saved locally — hub push is optional.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Orchestrate the full fine-tuning pipeline.

    Steps:
      1. Fetch + clean headlines
      2. Auto-label + merge with gold examples
      3. Split into train/val/test
      4. Compute class weights
      5. Load base FinBERT
      6. Fine-tune with early stopping + cosine LR + W&B logging
      7. Load best checkpoint, run final evaluation
      8. Save locally + push to Hub
      9. Print instructions for integrating into finbert.py
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    log.info("=" * 60)
    log.info("StockSense AI — FinBERT Fine-Tuning Pipeline")
    log.info("=" * 60)

    # ── Detect device ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info(f"Device: {device.upper()}")

    # ── Step 1: Fetch headlines ───────────────────────────────────────────────
    headlines = fetch_all_headlines()
    if len(headlines) < 50:
        log.error(
            "Fewer than 50 headlines fetched. Cannot fine-tune meaningfully. "
            "Check your internet connection and add NEWSAPI_KEY to .env."
        )
        sys.exit(1)

    # ── Step 2: Build dataset ─────────────────────────────────────────────────
    dataset = build_dataset(headlines)

    # ── Step 3: Split ─────────────────────────────────────────────────────────
    train_ex, val_ex, test_ex = split_dataset(dataset)

    # ── Step 4: Class weights ─────────────────────────────────────────────────
    log.info("Computing class weights for imbalance handling:")
    class_weights = compute_class_weights(train_ex)

    # ── Step 5: Load base model ───────────────────────────────────────────────
    hf_token   = os.getenv("HF_TOKEN")
    model_id   = CFG["base_model_id"]
    log.info(f"Loading base model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_id, token=hf_token
    ).to(device)

    # ── Step 6: Fine-tune ─────────────────────────────────────────────────────
    best_model_path = train(
        model, tokenizer, train_ex, val_ex, class_weights, device
    )

    # ── Step 7: Load best checkpoint + evaluate ───────────────────────────────
    log.info(f"\nLoading best checkpoint from: {best_model_path}")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_model_path
    ).to(device)

    metrics = final_evaluation(best_model, tokenizer, test_ex, device)

    # ── Step 8: Save + push ───────────────────────────────────────────────────
    save_and_push(best_model, tokenizer)

    # ── Step 9: Integration instructions ─────────────────────────────────────
    output_path = os.path.abspath(CFG["output_dir"])
    log.info(f"""
{'═'*60}
NEXT STEPS — integrate fine-tuned model into finbert.py
{'═'*60}

1. Add this entry to MODEL_CONFIGS in nlp/finbert.py:

    "finbert_custom": {{
        "model_id":    "{output_path}",
        "max_length":  512,
        "batch_size":  32,
        "labels":      ["positive", "negative", "neutral"],
        "description": "FinBERT fine-tuned on StockSense headlines. "
                       "Test accuracy: {metrics['overall_accuracy']:.1%}",
    }},

2. Change the default model:

    DEFAULT_MODEL = "finbert_custom"

3. Run your test suite to confirm improvement:

    python3 nlp/test_finbert.py

Gold accuracy: {metrics['gold_accuracy']:.1%}  ({len(GOLD_EXAMPLES)} examples)
{'═'*60}
""")


if __name__ == "__main__":
    main()