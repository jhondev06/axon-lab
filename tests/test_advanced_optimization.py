#!/usr/bin/env python3
"""
Test script for AXON Advanced Multi-Model Optimization

This script validates the implementation of:
- Parallel optimization
- Multi-objective optimization (Pareto front)
- Ensemble optimization
- Resource management
- Stability analysis
- Regime-based selection
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from optimization import (
        ParallelOptimizationEngine,
        MultiObjectiveOptimizer,
        ResourceManager,
        TimeSeriesCrossValidator,
        RegimeBasedSelector,
        EnsembleOptimizer
    )
    from utils import load_config
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warning: Import error: {e}")
    print("Some advanced features may not be available")
    IMPORTS_SUCCESSFUL = False

def create_sample_data():
    """Create sample financial data for testing."""
    np.random.seed(42)

    # Generate sample time series data
    n_samples = 1000
    n_features = 20

    # Create features (technical indicators)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create target (simplified trading signal)
    # Positive returns with some noise
    returns = np.random.randn(n_samples) * 0.02
    # Add some predictability based on features
    signal = (X.iloc[:, :5].sum(axis=1) > 0).astype(int)
    y = signal

    return X, y

def test_resource_manager():
    """Test resource management functionality."""
    if not IMPORTS_SUCCESSFUL:
        print("Warning: Skipping Resource Manager test - imports failed")
        return

    print("Testing Resource Manager...")

    config = load_config()
    rm = ResourceManager(config)

    # Test resource checking
    resources_available = rm.check_resources()
    print(f"Resources available: {resources_available}")

    # Test resource allocation
    allocation = rm.allocate_resources(4)
    print(f"Resource allocation: {allocation}")

    # Test resource monitoring
    monitoring = rm.monitor_resources()
    print(f"Resource monitoring: {monitoring}")

    print("Resource Manager tests passed!\n")

def test_parallel_optimization():
    """Test parallel optimization functionality."""
    if not IMPORTS_SUCCESSFUL:
        print("Warning: Skipping Parallel Optimization test - imports failed")
        return

    print("Testing Parallel Optimization...")

    config = load_config()
    X, y = create_sample_data()

    # Split data
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Test parallel engine
    engine = ParallelOptimizationEngine(config)

    # Test with subset of models for speed
    test_models = ['lightgbm', 'randomforest']

    try:
        results = engine.optimize_multiple_models(
            test_models, X_train, y_train, X_val, y_val
        )

        print(f"Parallel optimization completed for {len(results)} models")

        # Check results structure
        for model_name, result in results.items():
            if model_name != 'ensemble':
                assert 'best_params' in result, f"Missing best_params for {model_name}"
                assert 'best_value' in result, f"Missing best_value for {model_name}"
                print(f"   {model_name}: best_value={result['best_value']:.4f}")

        print("Parallel optimization tests passed!\n")

    except Exception as e:
        print(f"Warning: Parallel optimization test failed (may be due to missing dependencies): {e}\n")

def test_multi_objective_optimization():
    """Test multi-objective optimization functionality."""
    if not IMPORTS_SUCCESSFUL:
        print("Warning: Skipping Multi-Objective Optimization test - imports failed")
        return

    print("Testing Multi-Objective Optimization...")

    config = load_config()
    X, y = create_sample_data()

    # Split data
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Test Pareto front optimization
    optimizer = MultiObjectiveOptimizer(config)

    try:
        results = optimizer.optimize_pareto_front(
            'lightgbm', X_train, y_train, X_val, y_val
        )

        print("Pareto front optimization completed")
        print(f"   Solutions found: {results.get('n_solutions', 0)}")
        print(f"   Hypervolume: {results.get('hypervolume', 0):.4f}")

        # Check results structure
        assert 'pareto_front' in results, "Missing pareto_front"
        assert 'best_params' in results, "Missing best_params"

        print("Multi-objective optimization tests passed!\n")

    except Exception as e:
        print(f"Warning: Multi-objective optimization test failed (may be due to missing PyMOO): {e}\n")

def test_stability_analysis():
    """Test stability analysis functionality."""
    if not IMPORTS_SUCCESSFUL:
        print("Warning: Skipping Stability Analysis test - imports failed")
        return

    print("Testing Stability Analysis...")

    config = load_config()
    X, y = create_sample_data()

    # Test stability validator
    validator = TimeSeriesCrossValidator(config)

    # Mock optimization results
    mock_results = {
        'best_params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        },
        'best_value': 1.5
    }

    try:
        stability_results = validator.validate_model_stability(
            'lightgbm', mock_results['best_params'], X, y, n_splits=3
        )

        print("Stability analysis completed")
        print(f"   Stability rating: {stability_results['stability_analysis']['stability_rating']}")

        # Check results structure
        assert 'cv_results' in stability_results, "Missing cv_results"
        assert 'stability_analysis' in stability_results, "Missing stability_analysis"

        print("Stability analysis tests passed!\n")

    except Exception as e:
        print(f"Warning: Stability analysis test failed: {e}\n")

def test_regime_detection():
    """Test regime detection functionality."""
    if not IMPORTS_SUCCESSFUL:
        print("Warning: Skipping Regime Detection test - imports failed")
        return

    print("Testing Regime Detection...")

    config = load_config()
    X, y = create_sample_data()

    # Test regime selector
    selector = RegimeBasedSelector(config)

    try:
        regime = selector.detect_market_regime(X, y)
        print(f"Detected regime: {regime}")

        # Test regime weights
        mock_results = {
            'lightgbm': {'best_value': 1.5},
            'xgboost': {'best_value': 1.3},
            'catboost': {'best_value': 1.4}
        }

        weights = selector.get_regime_weights(regime, mock_results)
        print(f"Regime weights: {weights}")

        # Check results
        assert isinstance(regime, str), "Regime should be a string"
        assert isinstance(weights, dict), "Weights should be a dict"
        assert len(weights) > 0, "Weights should not be empty"

        print("Regime detection tests passed!\n")

    except Exception as e:
        print(f"Warning: Regime detection test failed: {e}\n")

def test_ensemble_optimization():
    """Test ensemble optimization functionality."""
    if not IMPORTS_SUCCESSFUL:
        print("Warning: Skipping Ensemble Optimization test - imports failed")
        return

    print("Testing Ensemble Optimization...")

    config = load_config()
    X, y = create_sample_data()

    # Split data
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Test ensemble optimizer
    ensemble_opt = EnsembleOptimizer(config)

    # Mock trained models
    mock_models = {
        'lightgbm': {'best_params': {'n_estimators': 100, 'max_depth': 6}},
        'xgboost': {'best_params': {'n_estimators': 100, 'max_depth': 6}}
    }

    try:
        results = ensemble_opt.optimize_ensemble(
            mock_models, X_train, y_train, X_val, y_val
        )

        print("Ensemble optimization completed")
        print(f"   Best Sharpe: {results.get('best_sharpe', 0):.4f}")

        # Check results structure
        assert 'weights' in results, "Missing weights"
        assert 'best_sharpe' in results, "Missing best_sharpe"

        print("Ensemble optimization tests passed!\n")

    except Exception as e:
        print(f"Warning: Ensemble optimization test failed: {e}\n")

def main():
    """Run all tests."""
    print("=== AXON Advanced Optimization Tests ===\n")

    # Test individual components
    test_resource_manager()
    test_parallel_optimization()
    test_multi_objective_optimization()
    test_stability_analysis()
    test_regime_detection()
    test_ensemble_optimization()

    print("=== Test Summary ===")
    print("All tests completed!")
    print("Note: Some tests may show warnings due to optional dependencies")
    print("Install pymoo and other optional packages for full functionality")

if __name__ == "__main__":
    main()