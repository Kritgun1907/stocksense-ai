"""
Test assembler.py — multi-stock feature matrix assembly pipeline
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.data.assembler import (
    assemble_stock,
    assemble_multiple_stocks,
    chronological_split,
    check_feature_consistency,
    list_assembly_configs,
)


def test_single_stock():
    """Test assembling a single stock."""
    print("=" * 80)
    print("TEST 1: SINGLE STOCK ASSEMBLY")
    print("=" * 80)

    result = assemble_stock("AAPL", period="2y", horizon=1, threshold=0.003, verbose=True)

    if result is None:
        print("❌ Single stock assembly failed")
        return None

    X, y, metadata = result
    print(f"\n✅ Single stock assembly successful!")
    print(f"   Features: {X.shape[1]}")
    print(f"   Rows: {len(X)}")
    print(f"   Metadata: {metadata}")

    return X, y, metadata


def test_multiple_stocks():
    """Test assembling multiple stocks."""
    print("\n" + "=" * 80)
    print("TEST 2: MULTIPLE STOCKS ASSEMBLY")
    print("=" * 80)

    # List available configs first
    list_assembly_configs()

    TICKERS = ["AAPL", "GOOGL", "MSFT"]

    try:
        X, y, metadata = assemble_multiple_stocks(
            tickers=TICKERS,
            period="2y",
            config_name="default",
            verbose=True,
        )

        print(f"\n✅ Multi-stock assembly successful!")
        print(f"   Combined shape: {X.shape}")
        print(f"   Stocks assembled: {len(metadata)}")
        print(f"   Unique tickers in result: {X.index.get_level_values('ticker').nunique()}")

        return X, y, metadata

    except Exception as e:
        print(f"❌ Multi-stock assembly failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_split_and_validation(X, y):
    """Test chronological split and feature consistency."""
    print("\n" + "=" * 80)
    print("TEST 3: CHRONOLOGICAL SPLIT & VALIDATION")
    print("=" * 80)

    try:
        X_train, X_test, y_train, y_test = chronological_split(
            X, y, config_name="default"
        )

        print(f"\n✅ Chronological split successful!")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_test shape:  {X_test.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_test shape:  {y_test.shape}")

        # Check feature consistency
        print(f"\n   Checking feature consistency...")
        is_consistent = check_feature_consistency(X_train, X_test, verbose=True)

        if is_consistent:
            print(f"\n✅ Feature consistency check passed!")
        else:
            print(f"\n⚠️  Feature consistency issues detected")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"❌ Split or validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ASSEMBLER PIPELINE TEST" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")

    # Test 1: Single stock
    single_result = test_single_stock()

    # Test 2: Multiple stocks
    multi_result = test_multiple_stocks()

    # Test 3: Split and validation
    if multi_result:
        X, y, metadata = multi_result
        split_result = test_split_and_validation(X, y)

        if split_result:
            X_train, X_test, y_train, y_test = split_result
            print(f"\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"✅ All tests completed successfully!")
            print(f"\nReady for model training with:")
            print(f"  Training set:   {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
            print(f"  Test set:       {X_test.shape[0]:,} rows × {X_test.shape[1]} features")

    print("\n")
