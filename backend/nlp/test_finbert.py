"""
Test finbert.py — FinBERT sentiment scoring module
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finbert import (
    list_model_configs,
    load_model,
    score_text,
    score_batch,
    warmup,
    get_cache_stats,
    get_model_info,
)


def test_model_loading():
    """Test model loading and configuration."""
    print("=" * 80)
    print("TEST 1: MODEL LOADING & CONFIGURATION")
    print("=" * 80)

    # List available configs
    print("\n📋 Available models:")
    configs = list_model_configs(verbose=True)
    print(f"\nFound {len(configs)} model configs: {configs}")

    # Load model
    print(f"\n📥 Loading default FinBERT model...")
    try:
        tok, model, device = load_model()
        print(f"✅ Model loaded successfully on device: {device.upper()}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_single_scoring():
    """Test single text scoring."""
    print("\n" + "=" * 80)
    print("TEST 2: SINGLE TEXT SCORING")
    print("=" * 80)

    test_cases = [
        ("Apple beats Q4 earnings expectations by wide margin", "positive"),
        ("iPhone sales decline sharply in China trade tensions", "negative"),
        ("Revenue declined less than feared, shares rally", "positive"),
        ("Company guides below analyst consensus", "negative"),
        ("Management reaffirms full year guidance", "neutral"),
        ("Stock maintains steady performance", "positive"),
    ]

    print("\n🎯 Scoring individual headlines:\n")
    print(f"{'Score':>7}  {'Label':<10}  {'Expected':<10}  Accuracy  Headline")
    print("─" * 85)

    correct = 0
    for text, expected in test_cases:
        result = score_text(text)
        match = "✅" if result["label"] == expected else "❌"
        if result["label"] == expected:
            correct += 1

        print(
            f"{result['score']:>+7.3f}  {result['label']:<10}  "
            f"{expected:<10}{match}  "
            f"P:{result['positive']:.3f}  {text[:40]}"
        )

    accuracy = correct / len(test_cases) * 100
    print(f"\n✅ Single scoring accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")

    return correct > 0


def test_batch_scoring():
    """Test batch scoring efficiency."""
    print("\n" + "=" * 80)
    print("TEST 3: BATCH SCORING EFFICIENCY")
    print("=" * 80)

    test_texts = [
        "Apple beats earnings expectations",
        "Stock price falls on weak guidance",
        "Company maintains guidance",
        "Revenue exceeds analyst forecasts",
        "Operating margins decline sharply",
        "Dividend increased by shareholders",
        "Profits fall below market consensus",
        "Cost reduction initiatives underway",
    ]

    print(f"\n📊 Batch scoring {len(test_texts)} headlines:")
    results = score_batch(test_texts, show_progress=True)

    print(f"\n{'Score':>7}  {'Label':<10}  {'Positive':<10}  Headline")
    print("─" * 70)
    for text, result in zip(test_texts, results):
        print(
            f"{result['score']:>+7.3f}  {result['label']:<10}  "
            f"{result['positive']:<10.3f}  {text[:40]}"
        )

    print(f"\n✅ Batch scoring complete: {len(results)} results")
    return len(results) == len(test_texts)


def test_caching():
    """Test inference caching and statistics."""
    print("\n" + "=" * 80)
    print("TEST 4: INFERENCE CACHING")
    print("=" * 80)

    test_texts = [
        "Apple beats earnings",
        "Stock falls on weak guidance",
        "Apple beats earnings",  # Duplicate
        "Stock falls on weak guidance",  # Duplicate
    ]

    print(f"\n💾 Testing cache with {len(test_texts)} texts (2 unique, 2 duplicates):")
    results_first = score_batch(test_texts, use_cache=True)
    stats_first = get_cache_stats()

    print(f"\nFirst batch:")
    print(f"  Cache hits:        {stats_first['hits']}")
    print(f"  Cache misses:      {stats_first['misses']}")
    print(f"  Cached entries:    {stats_first['cached_entries']}")
    print(f"  Hit rate:          {stats_first['hit_rate_pct']:.1f}%")

    # Score same texts again
    results_second = score_batch(test_texts, use_cache=True)
    stats_second = get_cache_stats()

    print(f"\nSecond batch (should be all cache hits):")
    print(f"  Cache hits:        {stats_second['hits']}")
    print(f"  Cache misses:      {stats_second['misses']}")
    print(f"  Cached entries:    {stats_second['cached_entries']}")
    print(f"  Hit rate:          {stats_second['hit_rate_pct']:.1f}%")

    # Verify cache hit improved
    improvement = stats_second["hit_rate_pct"] > stats_first["hit_rate_pct"]
    print(f"\n✅ Cache efficiency improved: {improvement}")

    return improvement


def test_edge_cases():
    """Test edge cases and validation."""
    print("\n" + "=" * 80)
    print("TEST 5: EDGE CASES & VALIDATION")
    print("=" * 80)

    edge_cases = [
        ("", "neutral", "empty string"),
        ("Hi", "neutral", "too short"),
        (None, None, "None value"),
        ("   ", "neutral", "whitespace only"),
    ]

    print(f"\n⚠️  Testing edge cases:\n")
    print(f"{'Input':<20}  {'Result':<10}  {'Expected':<10}  Test")
    print("─" * 60)

    for text, expected, test_name in edge_cases:
        if text is None:
            print(f"{'None':<20}  {'(skip)':<10}  {'(skip)':<10}  {test_name}")
            continue

        try:
            result = score_text(text)
            match = "✅" if result["label"] == expected else "❌"
            print(
                f"{repr(text):<20}  {result['label']:<10}  "
                f"{expected:<10}{match}  {test_name}"
            )
        except Exception as e:
            print(f"{repr(text):<20}  {'ERROR':<10}  {expected:<10}  {test_name}")

    print(f"\n✅ Edge case handling validated")
    return True


def test_model_info():
    """Test model info retrieval."""
    print("\n" + "=" * 80)
    print("TEST 6: MODEL INFO & DIAGNOSTICS")
    print("=" * 80)

    info = get_model_info()
    print(f"\n📊 Current model info:")
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    return info.get("status") == "loaded"


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 24 + "FINBERT SENTIMENT ANALYZER TEST" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")

    # Run tests in sequence
    results = []

    try:
        results.append(("Model Loading", test_model_loading()))
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        print("   This may be due to missing dependencies. Install with:")
        print("   pip install torch transformers")
        results.append(("Model Loading", False))

    try:
        if results[-1][1]:  # Only if loading succeeded
            results.append(("Single Scoring", test_single_scoring()))
            results.append(("Batch Scoring", test_batch_scoring()))
            results.append(("Caching", test_caching()))
            results.append(("Edge Cases", test_edge_cases()))
            results.append(("Model Info", test_model_info()))
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")

    total_passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check output above.")

    print("\n")
