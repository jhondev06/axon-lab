#!/usr/bin/env python3
"""
Test script for XGBoost integration in AXON pipeline.
Tests basic functionality with synthetic data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import AXON modules
from src.models import XGBoostModel, ModelRegistry, train_model
from src.utils import load_config

def generate_synthetic_data(n_samples=1000, n_features=20, random_state=42):
    """Generate synthetic trading data for testing."""
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Add some correlation structure
    for i in range(5):
        X[:, i] = X[:, 0] + 0.5 * np.random.randn(n_samples)

    # Generate target (binary classification)
    # Create some predictive signal
    signal = X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.1 * np.random.randn(n_samples)
    y = (signal > np.median(signal)).astype(int)

    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y

    return df, feature_names

def test_xgboost_basic():
    """Test basic XGBoost functionality."""
    print("Testing XGBoost basic functionality...")

    # Generate synthetic data
    df, feature_names = generate_synthetic_data(n_samples=1000, n_features=10)
    print(f"Generated synthetic data: {df.shape}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['y'])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df['y'])

    X_train = train_df[feature_names]
    y_train = train_df['y']
    X_val = val_df[feature_names]
    y_val = val_df['y']
    X_test = test_df[feature_names]
    y_test = test_df['y']

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Test XGBoostModel directly
    print("\nTesting XGBoostModel class...")
    model = XGBoostModel(n_estimators=50, max_depth=3, learning_rate=0.1)

    # Train model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print("Model trained successfully")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    print(f"Predictions shape: {y_pred.shape}")
    print(f"Probabilities shape: {y_pred_proba.shape}")

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    print("Test Metrics:")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

    # Test predict_proba output validation
    assert y_pred_proba.shape[1] == 2, "predict_proba should return 2 columns"
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)), "Probabilities should be between 0 and 1"
    assert np.allclose(y_pred_proba.sum(axis=1), 1.0), "Probabilities should sum to 1"
    print("predict_proba output validation passed")

    return True

def test_model_registry():
    """Test XGBoost integration with ModelRegistry."""
    print("\nTesting ModelRegistry integration...")

    # Load config
    config = load_config()

    # Initialize registry
    registry = ModelRegistry(config)

    # Check available models
    available = registry.list_available_models()
    print(f"Available models: {available}")

    assert 'xgboost' in available, "XGBoost should be in available models"
    assert 'xgb' in available, "XGB alias should be in available models"
    print("XGBoost available in registry")

    # Get XGBoost model
    model = registry.get_model('xgboost')
    assert isinstance(model, XGBoostModel), "Should return XGBoostModel instance"
    print("XGBoost model retrieved from registry")

    return True

def test_train_model_function():
    """Test the train_model function with XGBoost."""
    print("\nTesting train_model function...")

    # Generate data
    df, feature_names = generate_synthetic_data(n_samples=500, n_features=8)
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['y'])

    X_train = train_df[feature_names]
    y_train = train_df['y']
    X_val = val_df[feature_names]
    y_val = val_df['y']

    # Create XGBoost model
    model = XGBoostModel(n_estimators=30, max_depth=3)

    # Train using train_model function
    trained_model, metrics = train_model(
        model, X_train, y_train, X_val, y_val,
        model_name='xgboost', config=load_config()
    )

    print("train_model function completed")
    print(f"Metrics: {metrics}")

    # Validate metrics
    required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"

    print("All required metrics present")

    return True

def test_gpu_support():
    """Test GPU support detection."""
    print("\nTesting GPU support...")

    try:
        import xgboost as xgb
        from src.models import XGB_GPU_AVAILABLE

        if XGB_GPU_AVAILABLE:
            print("XGBoost GPU support detected")
        else:
            print("XGBoost GPU support not available (using CPU)")

        # Test model creation with GPU params
        model = XGBoostModel()
        params = model.params

        if XGB_GPU_AVAILABLE:
            assert 'gpu_id' in params, "GPU parameters should be set"
            assert params.get('tree_method') == 'gpu_hist', "Should use GPU tree method"
        else:
            assert params.get('tree_method') == 'hist', "Should use CPU tree method"

        print("GPU support configuration correct")

    except ImportError:
        print("XGBoost not available")
        return False

    return True

def main():
    """Run all tests."""
    print("=== AXON XGBoost Integration Tests ===\n")

    tests = [
        test_xgboost_basic,
        test_model_registry,
        test_train_model_function,
        test_gpu_support
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"PASS: {test_func.__name__}\n")
            else:
                failed += 1
                print(f"FAIL: {test_func.__name__}\n")
        except Exception as e:
            failed += 1
            print(f"FAIL: {test_func.__name__} - Error: {e}\n")

    print(f"=== Test Results: {passed} passed, {failed} failed ===")

    if failed == 0:
        print("All tests passed! XGBoost integration is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)