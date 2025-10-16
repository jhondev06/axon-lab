#!/usr/bin/env python3
"""
Test script for CatBoost integration in AXON pipeline.
Tests basic functionality with synthetic data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import AXON modules
from src.models import CatBoostModel, ModelRegistry, train_model
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

def test_catboost_basic():
    """Test basic CatBoost functionality."""
    print("Testing CatBoost basic functionality...")

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

    # Test CatBoostModel directly
    print("\nTesting CatBoostModel class...")
    model = CatBoostModel(iterations=50, depth=3, learning_rate=0.1)

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

    # Test feature importance
    importance = model.get_feature_importance()
    assert isinstance(importance, dict), "Feature importance should be a dict"
    assert len(importance) > 0, "Should have feature importance"
    # Check that importance values are reasonable
    importance_values = list(importance.values())
    assert all(isinstance(v, (int, float, np.floating)) for v in importance_values), "Importance values should be numeric"
    assert all(v >= 0 for v in importance_values), "Importance values should be non-negative"
    print("Feature importance test passed")

    return True

def test_model_registry():
    """Test CatBoost integration with ModelRegistry."""
    print("\nTesting ModelRegistry integration...")

    # Load config
    config = load_config()

    # Initialize registry
    registry = ModelRegistry(config)

    # Check available models
    available = registry.list_available_models()
    print(f"Available models: {available}")

    assert 'catboost' in available, "CatBoost should be in available models"
    assert 'cb' in available, "CB alias should be in available models"
    print("CatBoost available in registry")

    # Get CatBoost model
    model = registry.get_model('catboost')
    assert isinstance(model, CatBoostModel), "Should return CatBoostModel instance"
    print("CatBoost model retrieved from registry")

    return True

def test_train_model_function():
    """Test the train_model function with CatBoost."""
    print("\nTesting train_model function...")

    # Generate data
    df, feature_names = generate_synthetic_data(n_samples=500, n_features=8)
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['y'])

    X_train = train_df[feature_names]
    y_train = train_df['y']
    X_val = val_df[feature_names]
    y_val = val_df['y']

    # Create CatBoost model
    model = CatBoostModel(iterations=30, depth=3)

    # Train using train_model function
    trained_model, metrics = train_model(
        model, X_train, y_train, X_val, y_val,
        model_name='catboost', config=load_config()
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
        import catboost as cb
        from src.models import CB_GPU_AVAILABLE

        if CB_GPU_AVAILABLE:
            print("CatBoost GPU support detected")
        else:
            print("CatBoost GPU support not available (using CPU)")

        # Test model creation with GPU params
        model = CatBoostModel()
        params = model.params

        if CB_GPU_AVAILABLE:
            assert params.get('task_type') == 'GPU', "Should use GPU task type"
            assert 'devices' in params, "GPU devices should be set"
        else:
            assert params.get('task_type') == 'CPU', "Should use CPU task type"

        print("GPU support configuration correct")

    except ImportError:
        print("CatBoost not available")
        return False

    return True

def test_symmetric_trees():
    """Test SymmetricTree grow policy."""
    print("\nTesting SymmetricTree grow policy...")

    # Generate data
    df, feature_names = generate_synthetic_data(n_samples=200, n_features=5)
    X_train = df[feature_names]
    y_train = df['y']

    # Test with SymmetricTree
    model = CatBoostModel(
        iterations=20,
        depth=4,
        grow_policy='SymmetricTree',
        verbose=False
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train), "Should have predictions for all samples"
    print("SymmetricTree grow policy test passed")

    return True

def main():
    """Run all tests."""
    print("=== AXON CatBoost Integration Tests ===\n")

    tests = [
        test_catboost_basic,
        test_model_registry,
        test_train_model_function,
        test_gpu_support,
        test_symmetric_trees
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
        print("All tests passed! CatBoost integration is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)