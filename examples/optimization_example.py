#!/usr/bin/env python3
"""
AXON Hyperparameter Optimization - Example Usage

Este exemplo demonstra como usar o sistema completo de otimização
de hiperparâmetros do AXON-V3.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.optimization import OptimizationEngine
from src.utils import load_config, ensure_dir
from src.models import train_model_with_optimized_params, save_model, ModelRegistry, train_model
from src.metrics import sharpe_ratio, maximum_drawdown, calculate_returns


def load_sample_data():
    """Load sample data for demonstration."""
    print("Loading sample data...")

    # Try to load processed data
    train_path = "data/processed/train_features.parquet"
    val_path = "data/processed/validation_features.parquet"

    if Path(train_path).exists() and Path(val_path).exists():
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        print("Loaded existing processed data")
    else:
        print("Processed data not found. Using synthetic data for demonstration...")

        # Generate synthetic data
        np.random.seed(42)
        n_samples = 10000
        n_features = 20

        # Generate features
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X = np.random.randn(n_samples, n_features)

        # Generate target (simple trend following signal)
        trend = np.sin(np.linspace(0, 4*np.pi, n_samples)) + 0.1 * np.random.randn(n_samples)
        y = (trend > 0).astype(int)

        # Create DataFrames
        train_df = pd.DataFrame(X[:8000], columns=feature_names)
        train_df['y'] = y[:8000]

        val_df = pd.DataFrame(X[8000:], columns=feature_names)
        val_df['y'] = y[8000:]

    return train_df, val_df


def example_basic_optimization():
    """Example 1: Basic optimization with default settings."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Optimization")
    print("="*60)

    # Load configuration
    config = load_config()

    # Modify config for this example
    config['optimization'] = {
        'enabled': True,
        'n_trials': 50,  # Reduced for faster demo
        'models': ['lightgbm'],
        'method': 'optuna',
        'multi_objective': False,
        'reporting': {
            'enabled': True,
            'include_visualizations': True
        }
    }

    # Load data
    train_df, val_df = load_sample_data()

    # Prepare features
    target_col = 'y'
    feature_cols = [col for col in train_df.columns if col != target_col]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    print(f"Data: {X_train.shape[0]} train, {X_val.shape[0]} validation samples")
    print(f"Target distribution: {y_train.value_counts().to_dict()}")

    # Initialize optimization engine
    engine = OptimizationEngine(config)

    # Optimize LightGBM
    print("\nStarting optimization...")
    results = engine.optimize_model('lightgbm', X_train, y_train, X_val, y_val)

    # Display results
    print("\nOptimization completed!")
    print(f"Best value: {results['best_value']:.4f}")
    print(f"Method: {results['method']}")
    print("\nBest parameters:")
    for param, value in results['best_params'].items():
        print(f"   {param}: {value}")

    return results


def example_multi_objective_optimization():
    """Example 2: Multi-objective optimization."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Objective Optimization")
    print("="*60)

    # Load configuration
    config = load_config()

    # Multi-objective config
    config['optimization'] = {
        'enabled': True,
        'n_trials': 30,  # Reduced for demo
        'models': ['lightgbm'],
        'method': 'optuna',
        'multi_objective': True,
        'objectives': [
            {
                'metric': 'sharpe_ratio',
                'weight': 0.7,
                'direction': 'maximize',
                'target': 1.0
            },
            {
                'metric': 'max_drawdown',
                'weight': 0.3,
                'direction': 'minimize',
                'target': 0.2
            }
        ],
        'reporting': {
            'enabled': True,
            'include_visualizations': True
        }
    }

    # Load data
    train_df, val_df = load_sample_data()

    # Prepare features
    feature_cols = [col for col in train_df.columns if col != 'y']
    X_train = train_df[feature_cols]
    y_train = train_df['y']
    X_val = val_df[feature_cols]
    y_val = val_df['y']

    # Initialize optimization engine
    engine = OptimizationEngine(config)

    # Run multi-objective optimization
    print("\nStarting multi-objective optimization...")
    results = engine.optimize_model('lightgbm', X_train, y_train, X_val, y_val)

    print("\nMulti-objective optimization completed!")
    print(f"Best composite score: {results['best_value']:.4f}")

    return results


def example_grid_search():
    """Example 3: Grid Search for RandomForest."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Grid Search Optimization")
    print("="*60)

    # Load configuration
    config = load_config()

    # Grid search config
    config['optimization'] = {
        'enabled': True,
        'models': ['randomforest'],
        'method': 'grid_search',
        'grid_search': {
            'cv_folds': 3,  # Reduced for demo
            'scoring': 'f1'
        },
        'reporting': {
            'enabled': True
        }
    }

    # Load data
    train_df, val_df = load_sample_data()

    # Prepare features
    feature_cols = [col for col in train_df.columns if col != 'y']
    X_train = train_df[feature_cols]
    y_train = train_df['y']
    X_val = val_df[feature_cols]
    y_val = val_df['y']

    # Initialize optimization engine
    engine = OptimizationEngine(config)

    # Run grid search
    print("\nStarting Grid Search optimization...")
    results = engine.optimize_model('randomforest', X_train, y_train, X_val, y_val)

    print("\nGrid Search completed!")
    print(f"Best CV score: {results['best_score']:.4f}")
    print("\nBest parameters:")
    for param, value in results['best_params'].items():
        print(f"   {param}: {value}")

    return results


def example_comparison_before_after():
    """Example 4: Compare performance before and after optimization."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Before/After Comparison")
    print("="*60)

    # Load data
    train_df, val_df = load_sample_data()
    feature_cols = [col for col in train_df.columns if col != 'y']
    X_train = train_df[feature_cols]
    y_train = train_df['y']
    X_val = val_df[feature_cols]
    y_val = val_df['y']

    # Load configuration
    config = load_config()

    # 1. Train baseline model (default parameters)
    print("\nTraining baseline model...")

    registry = ModelRegistry(config)
    baseline_model = registry.get_model('lightgbm')
    baseline_trained, baseline_metrics = train_model(
        baseline_model, X_train, y_train, X_val, y_val, 'lightgbm_baseline', config
    )

    # 2. Optimize and train optimized model
    print("\nOptimizing hyperparameters...")
    config['optimization'] = {
        'enabled': True,
        'n_trials': 20,  # Quick optimization for demo
        'models': ['lightgbm'],
        'method': 'optuna',
        'multi_objective': False
    }

    engine = OptimizationEngine(config)
    opt_results = engine.optimize_model('lightgbm', X_train, y_train, X_val, y_val)

    # Train with optimized parameters
    print("\nTraining optimized model...")
    opt_trained, opt_metrics = train_model_with_optimized_params(
        'lightgbm', opt_results['best_params'], X_train, y_train, X_val, y_val, config
    )

    # 3. Compare results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    comparison = pd.DataFrame({
        'Baseline': baseline_metrics,
        'Optimized': opt_metrics
    })

    print(comparison.round(4))

    # Calculate improvements
    improvements = {}
    for metric in baseline_metrics.keys():
        if metric in opt_metrics:
            baseline_val = baseline_metrics[metric]
            opt_val = opt_metrics[metric]
            if baseline_val != 0:
                improvement = ((opt_val - baseline_val) / abs(baseline_val)) * 100
                improvements[metric] = improvement

    print("\nImprovements:")
    for metric, improvement in improvements.items():
        print(".2f")

    return {
        'baseline': baseline_metrics,
        'optimized': opt_metrics,
        'improvements': improvements
    }


def main():
    """Run all examples."""
    print("AXON Hyperparameter Optimization - Examples")
    print("="*60)

    try:
        # Example 1: Basic optimization
        results1 = example_basic_optimization()

        # Example 2: Multi-objective
        results2 = example_multi_objective_optimization()

        # Example 3: Grid Search
        results3 = example_grid_search()

        # Example 4: Before/After comparison
        comparison = example_comparison_before_after()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nCheck the following directories for results:")
        print("   outputs/reports/ - Detailed reports")
        print("   outputs/figures/ - Visualizations")
        print("   outputs/optuna_studies/ - Optimization studies")
        print("   outputs/artifacts/ - Optimized models")

    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()