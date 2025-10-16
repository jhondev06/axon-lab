#!/usr/bin/env python3
"""Test script to validate debug logs in AXON modules."""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

# Import functions with logs (only those without Unicode issues)
from src.dataset import load_raw_data, create_time_based_splits, clean_and_validate_data
from src.features import calculate_ema, calculate_rsi, generate_triple_barrier_labels
from src.metrics import sharpe_ratio, maximum_drawdown

def test_dataset_logs():
    """Test dataset functions with debug logs."""
    print("=== Testing Dataset Functions ===")

    # Load synthetic data
    df = pd.read_csv('data/raw/btc_synthetic_1m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded {len(df)} rows of synthetic data")

    # Test load_raw_data
    try:
        df_loaded = load_raw_data('data/raw/btc_synthetic_1m.csv')
        print("SUCCESS: load_raw_data completed")
    except Exception as e:
        print(f"ERROR: load_raw_data failed: {e}")

    # Test create_time_based_splits
    try:
        splits = create_time_based_splits(df_loaded)
        print("SUCCESS: create_time_based_splits completed")
    except Exception as e:
        print(f"ERROR: create_time_based_splits failed: {e}")

    # Test clean_and_validate_data
    try:
        df_clean = clean_and_validate_data(df_loaded)
        print("SUCCESS: clean_and_validate_data completed")
    except Exception as e:
        print(f"ERROR: clean_and_validate_data failed: {e}")

def test_features_logs():
    """Test features functions with debug logs."""
    print("\n=== Testing Features Functions ===")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
    prices = 50000 + np.random.randn(1000).cumsum()
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.random.randint(100, 1000, 1000)
    })

    # Test calculate_ema
    try:
        ema = calculate_ema(df['close'], span=20)
        print("SUCCESS: calculate_ema completed")
    except Exception as e:
        print(f"ERROR: calculate_ema failed: {e}")

    # Test calculate_rsi
    try:
        rsi = calculate_rsi(df['close'], window=14)
        print("SUCCESS: calculate_rsi completed")
    except Exception as e:
        print(f"ERROR: calculate_rsi failed: {e}")

    # Test generate_triple_barrier_labels
    try:
        labels = generate_triple_barrier_labels(df)
        print("SUCCESS: generate_triple_barrier_labels completed")
    except Exception as e:
        print(f"ERROR: generate_triple_barrier_labels failed: {e}")

def test_metrics_logs():
    """Test metrics functions with debug logs."""
    print("\n=== Testing Metrics Functions ===")

    # Create sample returns
    returns = np.random.randn(1000) * 0.01

    # Test sharpe_ratio
    try:
        sharpe = sharpe_ratio(returns)
        print(f"SUCCESS: sharpe_ratio completed: {sharpe:.4f}")
    except Exception as e:
        print(f"ERROR: sharpe_ratio failed: {e}")

    # Create sample equity curve
    equity = 10000 + np.random.randn(1000).cumsum()

    # Test maximum_drawdown
    try:
        dd_metrics = maximum_drawdown(equity)
        print(f"SUCCESS: maximum_drawdown completed: {dd_metrics['max_drawdown_pct']:.2%}")
    except Exception as e:
        print(f"ERROR: maximum_drawdown failed: {e}")

def main():
    """Run all debug log tests."""
    print("Starting AXON Debug Log Validation")
    print("=" * 50)

    test_dataset_logs()
    test_features_logs()
    test_metrics_logs()

    print("\n" + "=" * 50)
    print("Debug log validation completed")
    print("Check the logs above for DEBUG messages")

if __name__ == "__main__":
    main()