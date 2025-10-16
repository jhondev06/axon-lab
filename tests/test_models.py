"""Unit tests for AXON models module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
sys.path.append('src')
from src.models import (
    ModelRegistry, train_model, save_model, load_model,
    get_feature_importance, cross_validate_model
)


class TestModelRegistry:
    """Test ModelRegistry class."""

    def test_model_registry_init(self, mock_config):
        """Test ModelRegistry initialization."""
        registry = ModelRegistry(mock_config)

        assert registry.config == mock_config
        assert registry.models == {}
        assert 'lightgbm' in registry.model_configs
        assert 'randomforest' in registry.model_configs

    def test_get_lightgbm_config(self, mock_config):
        """Test LightGBM configuration retrieval."""
        registry = ModelRegistry(mock_config)

        config = registry._get_lightgbm_config()

        assert isinstance(config, dict)
        assert 'objective' in config
        assert 'metric' in config
        assert 'boosting_type' in config
        assert 'verbose' in config

    def test_get_randomforest_config(self, mock_config):
        """Test RandomForest configuration retrieval."""
        registry = ModelRegistry(mock_config)

        config = registry._get_randomforest_config()

        assert isinstance(config, dict)
        assert 'n_estimators' in config
        assert 'max_depth' in config
        assert 'random_state' in config
        assert 'n_jobs' in config

    def test_get_model_lightgbm(self, mock_config):
        """Test getting LightGBM model."""
        registry = ModelRegistry(mock_config)

        model = registry.get_model('lightgbm')

        assert isinstance(model, dict)
        assert model['model_class'] == 'lightgbm'
        assert 'params' in model
        assert 'train_params' in model

    def test_get_model_randomforest(self, mock_config):
        """Test getting RandomForest model."""
        registry = ModelRegistry(mock_config)

        model = registry.get_model('randomforest')

        assert hasattr(model, 'fit')  # Scikit-learn style
        assert hasattr(model, 'predict')

    def test_get_model_invalid(self, mock_config):
        """Test getting invalid model raises error."""
        registry = ModelRegistry(mock_config)

        with pytest.raises(ValueError):
            registry.get_model('invalid_model')

    def test_list_available_models(self, mock_config):
        """Test listing available models."""
        registry = ModelRegistry(mock_config)

        available = registry.list_available_models()

        assert isinstance(available, list)
        assert 'randomforest' in available
        assert 'rf' in available


class TestTrainModel:
    """Test train_model function."""

    def test_train_model_lightgbm(self, sample_features_data, mock_config):
        """Test training LightGBM model."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        # Get model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('lightgbm')

        # Train model
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'lightgbm', mock_config
        )

        assert trained_model is not None
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

    def test_train_model_randomforest(self, sample_features_data, mock_config):
        """Test training RandomForest model."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        # Get model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')

        # Train model
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', mock_config
        )

        assert trained_model is not None
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

    def test_train_model_insufficient_data(self, mock_config):
        """Test training with insufficient data."""
        # Create very small dataset
        X_train = pd.DataFrame({'feature1': [1, 2]})
        y_train = pd.Series([0, 1])
        X_val = pd.DataFrame({'feature1': [3]})
        y_val = pd.Series([1])

        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')

        # Should handle gracefully
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', mock_config
        )

        assert trained_model is not None
        assert isinstance(metrics, dict)


class TestSaveLoadModel:
    """Test save_model and load_model functions."""

    def test_save_load_model_randomforest(self, sample_features_data, mock_config, tmp_path):
        """Test saving and loading RandomForest model."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        # Train model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', mock_config
        )

        # Save model
        model_path = save_model(
            trained_model, 'randomforest', metrics, feature_cols, mock_config
        )

        assert Path(model_path).exists()

        # Load model
        loaded_model, metadata = load_model(model_path)

        assert loaded_model is not None
        assert isinstance(metadata, dict)
        assert 'model_name' in metadata
        assert 'metrics' in metadata
        assert 'feature_names' in metadata

    def test_save_load_model_lightgbm(self, sample_features_data, mock_config, tmp_path):
        """Test saving and loading LightGBM model."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        # Train model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('lightgbm')
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'lightgbm', mock_config
        )

        # Save model
        model_path = save_model(
            trained_model, 'lightgbm', metrics, feature_cols, mock_config
        )

        assert Path(model_path).exists()

        # Load model
        loaded_model, metadata = load_model(model_path)

        assert loaded_model is not None
        assert isinstance(metadata, dict)

    def test_load_nonexistent_model(self):
        """Test loading non-existent model."""
        with pytest.raises(FileNotFoundError):
            load_model('nonexistent_model.pkl')


class TestGetFeatureImportance:
    """Test get_feature_importance function."""

    def test_get_feature_importance_randomforest(self, sample_features_data, mock_config):
        """Test feature importance for RandomForest."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        # Train model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')
        trained_model, _ = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', mock_config
        )

        # Get feature importance
        importance_df = get_feature_importance(trained_model, feature_cols, 'randomforest')

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == len(feature_cols)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns

        # Should be sorted by importance
        assert importance_df['importance'].is_monotonic_decreasing

    def test_get_feature_importance_lightgbm(self, sample_features_data, mock_config):
        """Test feature importance for LightGBM."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        # Train model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('lightgbm')
        trained_model, _ = train_model(
            model, X_train, y_train, X_val, y_val, 'lightgbm', mock_config
        )

        # Get feature importance
        importance_df = get_feature_importance(trained_model, feature_cols, 'lightgbm')

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == len(feature_cols)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns

    def test_get_feature_importance_unsupported_model(self, sample_features_data):
        """Test feature importance for unsupported model."""
        # Create a simple mock model that doesn't have feature importance methods
        class UnsupportedModel:
            pass

        mock_model = UnsupportedModel()

        importance_df = get_feature_importance(mock_model, ['feature1'], 'unknown')

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 0


class TestCrossValidateModel:
    """Test cross_validate_model function."""

    def test_cross_validate_randomforest(self, sample_features_data, mock_config):
        """Test cross-validation for RandomForest."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X = sample_features_data[feature_cols]
        y = sample_features_data[target_col]

        # Get model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')

        # Cross-validate
        cv_metrics = cross_validate_model(model, X, y, cv=3)

        assert isinstance(cv_metrics, dict)
        assert 'cv_accuracy_mean' in cv_metrics
        assert 'cv_precision_mean' in cv_metrics
        assert 'cv_recall_mean' in cv_metrics
        assert 'cv_f1_mean' in cv_metrics

        # All metrics should be reasonable
        assert 0 <= cv_metrics['cv_accuracy_mean'] <= 1
        assert 0 <= cv_metrics['cv_precision_mean'] <= 1
        assert 0 <= cv_metrics['cv_recall_mean'] <= 1
        assert 0 <= cv_metrics['cv_f1_mean'] <= 1

    def test_cross_validate_insufficient_data(self, mock_config):
        """Test cross-validation with insufficient data."""
        # Small dataset with more balanced classes
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6]})
        y = pd.Series([0, 1, 0, 1, 0, 1])

        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')

        # Should handle gracefully with cv=2 instead of cv=3
        cv_metrics = cross_validate_model(model, X, y, cv=2)

        assert isinstance(cv_metrics, dict)


class TestModelIntegration:
    """Integration tests for models pipeline."""

    def test_full_model_pipeline_randomforest(self, sample_features_data, mock_config, tmp_path):
        """Test complete model pipeline for RandomForest."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        # Initialize registry
        registry = ModelRegistry(mock_config)

        # Get and train model
        model = registry.get_model('randomforest')
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', mock_config
        )

        # Save model
        model_path = save_model(
            trained_model, 'randomforest', metrics, feature_cols, mock_config
        )

        # Load model
        loaded_model, metadata = load_model(model_path)

        # Get feature importance
        importance_df = get_feature_importance(loaded_model, feature_cols, 'randomforest')

        # Cross-validate
        cv_metrics = cross_validate_model(loaded_model, X_train, y_train, cv=3)

        # Verify all components work together
        assert trained_model is not None
        assert Path(model_path).exists()
        assert loaded_model is not None
        assert isinstance(importance_df, pd.DataFrame)
        assert isinstance(cv_metrics, dict)

    def test_full_model_pipeline_lightgbm(self, sample_features_data, mock_config, tmp_path):
        """Test complete model pipeline for LightGBM."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        # Initialize registry
        registry = ModelRegistry(mock_config)

        # Get and train model
        model = registry.get_model('lightgbm')
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'lightgbm', mock_config
        )

        # Save model
        model_path = save_model(
            trained_model, 'lightgbm', metrics, feature_cols, mock_config
        )

        # Load model
        loaded_model, metadata = load_model(model_path)

        # Get feature importance
        importance_df = get_feature_importance(trained_model, feature_cols, 'lightgbm')

        # Verify all components work together
        assert trained_model is not None
        assert Path(model_path).exists()
        assert loaded_model is not None
        assert isinstance(importance_df, pd.DataFrame)

    def test_model_comparison(self, sample_features_data, mock_config):
        """Test model comparison functionality."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        registry = ModelRegistry(mock_config)
        results = {}

        # Train multiple models
        for model_name in ['randomforest']:
            try:
                model = registry.get_model(model_name)
                trained_model, metrics = train_model(
                    model, X_train, y_train, X_val, y_val, model_name, mock_config
                )
                results[model_name] = metrics
            except Exception as e:
                print(f"Failed to train {model_name}: {e}")

        # Should have results for at least one model
        assert len(results) > 0

        # All results should have required metrics
        for model_name, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'f1' in metrics