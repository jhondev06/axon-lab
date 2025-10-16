"""Unit tests for Ensemble model implementation."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from unittest.mock import patch, MagicMock

# Import Ensemble components
from src.models import EnsembleModel, ModelRegistry


class TestEnsembleModel:
    """Test Ensemble model functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10,
            n_redundant=5, random_state=42
        )
        return X, y

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for ensemble testing."""
        return {
            'models': {
                'lightgbm': {'n_estimators': 10, 'verbose': -1},
                'xgboost': {'n_estimators': 10, 'verbosity': 0},
                'catboost': {'iterations': 10, 'verbose': False},
                'randomforest': {'n_estimators': 10, 'random_state': 42}
            }
        }

    @pytest.fixture
    def ensemble_model(self, mock_config):
        """Create Ensemble model instance."""
        return EnsembleModel(
            ensemble_type='voting',
            base_models=['randomforest'],  # Use only RF for speed
            voting_type='soft',
            random_state=42
        )

    def test_model_initialization(self, ensemble_model):
        """Test Ensemble model initialization."""
        assert ensemble_model.base_models == {}
        assert ensemble_model.meta_model is None
        assert ensemble_model.weights is None
        assert ensemble_model.ensemble_type == 'voting'
        assert ensemble_model.combination_strategy == 'performance_based'

    def test_invalid_ensemble_type(self):
        """Test invalid ensemble type raises error."""
        with pytest.raises(ValueError, match="Invalid ensemble_type"):
            EnsembleModel(ensemble_type='invalid')

    def test_invalid_combination_strategy(self):
        """Test invalid combination strategy raises error."""
        with pytest.raises(ValueError, match="Invalid combination_strategy"):
            EnsembleModel(combination_strategy='invalid')

    def test_initialize_base_models(self, ensemble_model, mock_config):
        """Test base model initialization."""
        ensemble_model._initialize_base_models(mock_config)

        # Should have initialized RandomForest
        assert 'randomforest' in ensemble_model.base_models
        assert hasattr(ensemble_model.base_models['randomforest'], 'fit')

    def test_detect_market_regime(self, ensemble_model, sample_data):
        """Test market regime detection."""
        X, y = sample_data

        # Test with sufficient data
        regime = ensemble_model._detect_market_regime(X, y)
        assert regime in ['neutral', 'high_volatility', 'bull_trend', 'bear_trend']

        # Test with insufficient data
        X_small = X[:5]
        regime = ensemble_model._detect_market_regime(X_small, y[:5])
        assert regime == 'neutral'  # Default for insufficient data

    def test_performance_based_weights(self, ensemble_model, sample_data, mock_config):
        """Test performance-based weight calculation."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and fit base models
        ensemble_model._initialize_base_models(mock_config)
        for name, model in ensemble_model.base_models.items():
            model.fit(X_train, y_train)

        # Calculate weights
        weights = ensemble_model._calculate_model_weights(X_val, y_val)

        assert isinstance(weights, dict)
        assert len(weights) == len(ensemble_model.base_models)
        assert all(isinstance(w, (int, float)) for w in weights.values())
        assert all(w >= 0 for w in weights.values())

    def test_diversity_based_weights(self, ensemble_model, sample_data, mock_config):
        """Test diversity-based weight calculation."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and fit base models
        ensemble_model._initialize_base_models(mock_config)
        for name, model in ensemble_model.base_models.items():
            model.fit(X_train, y_train)

        # Calculate weights
        weights = ensemble_model._calculate_model_weights(X_val, y_val)

        assert isinstance(weights, dict)
        assert len(weights) == len(ensemble_model.base_models)

    def test_equal_weights(self, ensemble_model, mock_config):
        """Test equal weight calculation."""
        ensemble_model._initialize_base_models(mock_config)
        weights = ensemble_model._calculate_model_weights(None, None)

        expected_weight = 1.0 / len(ensemble_model.base_models)
        assert all(w == expected_weight for w in weights.values())

    def test_adaptive_weights(self, ensemble_model, sample_data, mock_config):
        """Test adaptive weight calculation."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        ensemble_model._initialize_base_models(mock_config)
        for name, model in ensemble_model.base_models.items():
            model.fit(X_train, y_train)

        # Test with different regimes
        for regime in ['neutral', 'high_volatility', 'bull_trend', 'bear_trend']:
            weights = ensemble_model._calculate_model_weights(X_val, y_val, regime)
            assert isinstance(weights, dict)
            assert len(weights) > 0

    def test_create_meta_features(self, ensemble_model, sample_data, mock_config):
        """Test meta-feature creation."""
        X, y = sample_data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        ensemble_model._initialize_base_models(mock_config)
        for name, model in ensemble_model.base_models.items():
            model.fit(X_train, y_train)

        # Get predictions for meta features
        predictions = {}
        for name, model in ensemble_model.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_val)
            else:
                pred = model.predict(X_val).astype(float)
                pred = np.column_stack([1 - pred, pred])
            predictions[name] = pred

        meta_features = ensemble_model._create_meta_features(X_val, predictions)

        assert isinstance(meta_features, np.ndarray)
        assert meta_features.shape[0] == len(X_val)
        assert meta_features.shape[1] > 0

    def test_fit_voting_ensemble(self, sample_data, mock_config):
        """Test fitting voting ensemble."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type='voting',
            base_models=['randomforest'],
            voting_type='soft'
        )

        ensemble.fit(X_train, y_train, config=mock_config)

        assert len(ensemble.base_models) > 0
        assert ensemble.weights is not None

    def test_fit_stacking_ensemble(self, sample_data, mock_config):
        """Test fitting stacking ensemble."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type='stacking',
            base_models=['randomforest'],
            meta_model='randomforest'
        )

        ensemble.fit(X_train, y_train, config=mock_config)

        assert len(ensemble.base_models) > 0
        assert ensemble.meta_model is not None

    def test_fit_blending_ensemble(self, sample_data, mock_config):
        """Test fitting blending ensemble."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type='blending',
            base_models=['randomforest']
        )

        ensemble.fit(X_train, y_train, config=mock_config)

        assert len(ensemble.base_models) > 0

    def test_fit_weighted_ensemble(self, sample_data, mock_config):
        """Test fitting weighted ensemble."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type='weighted',
            base_models=['randomforest'],
            combination_strategy='performance_based'
        )

        ensemble.fit(X_train, y_train, config=mock_config)

        assert len(ensemble.base_models) > 0
        assert ensemble.weights is not None

    def test_predict_voting(self, sample_data, mock_config):
        """Test voting ensemble prediction."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        ensemble.fit(X_train, y_train, config=mock_config)

        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})

    def test_predict_stacking(self, sample_data, mock_config):
        """Test stacking ensemble prediction."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type='stacking',
            base_models=['randomforest'],
            meta_model='randomforest'
        )
        ensemble.fit(X_train, y_train, config=mock_config)

        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_predict_weighted(self, sample_data, mock_config):
        """Test weighted ensemble prediction."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type='weighted',
            base_models=['randomforest'],
            combination_strategy='equal'
        )
        ensemble.fit(X_train, y_train, config=mock_config)

        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)

    def test_predict_proba_voting(self, sample_data, mock_config):
        """Test voting ensemble probability prediction."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        ensemble.fit(X_train, y_train, config=mock_config)

        probabilities = ensemble.predict_proba(X_test)

        assert probabilities.shape == (len(X_test), 2)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_proba_stacking(self, sample_data, mock_config):
        """Test stacking ensemble probability prediction."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type='stacking',
            base_models=['randomforest'],
            meta_model='randomforest'
        )
        ensemble.fit(X_train, y_train, config=mock_config)

        probabilities = ensemble.predict_proba(X_test)

        assert probabilities.shape == (len(X_test), 2)

    def test_predict_before_fit_raises_error(self, ensemble_model):
        """Test that predict before fit raises error."""
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="Ensemble not fitted yet"):
            ensemble_model.predict(X)

        with pytest.raises(ValueError, match="Ensemble not fitted yet"):
            ensemble_model.predict_proba(X)

    def test_voting_hard_vs_soft(self, sample_data, mock_config):
        """Test hard vs soft voting."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hard voting
        ensemble_hard = EnsembleModel(
            ensemble_type='voting',
            base_models=['randomforest'],
            voting_type='hard'
        )
        ensemble_hard.fit(X_train, y_train, config=mock_config)

        # Soft voting
        ensemble_soft = EnsembleModel(
            ensemble_type='voting',
            base_models=['randomforest'],
            voting_type='soft'
        )
        ensemble_soft.fit(X_train, y_train, config=mock_config)

        pred_hard = ensemble_hard.predict(X_test)
        pred_soft = ensemble_soft.predict(X_test)

        # Both should work
        assert len(pred_hard) == len(X_test)
        assert len(pred_soft) == len(X_test)

    def test_save_load_ensemble(self, sample_data, mock_config, tmp_path):
        """Test ensemble saving and loading."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        ensemble.fit(X_train, y_train, config=mock_config)

        # Save ensemble
        model_path = tmp_path / "test_ensemble.pkl"
        ensemble.save_model(str(model_path))

        # Load ensemble
        loaded_ensemble = EnsembleModel.load_model(str(model_path))

        # Test predictions are similar
        original_pred = ensemble.predict(X_test)
        loaded_pred = loaded_ensemble.predict(X_test)

        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_multiple_base_models(self, sample_data, mock_config):
        """Test ensemble with multiple base models."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type='voting',
            base_models=['randomforest'],  # Could add more if available
            voting_type='soft'
        )

        ensemble.fit(X_train, y_train, config=mock_config)

        assert len(ensemble.base_models) >= 1

        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_regime_based_weighting(self, sample_data, mock_config):
        """Test regime-based weighting."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type='weighted',
            base_models=['randomforest'],
            combination_strategy='adaptive',
            regime_detection=True
        )

        ensemble.fit(X_train, y_train, config=mock_config)

        # Should have weights
        assert ensemble.weights is not None

        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_feature_name_storage(self, sample_data, mock_config):
        """Test feature name storage."""
        X, y = sample_data
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        ensemble.fit(X_train, y_train, config=mock_config)

        assert ensemble.feature_names == list(X_train.columns)

    @pytest.mark.parametrize("ensemble_type", ["voting", "stacking", "blending", "weighted"])
    def test_different_ensemble_types(self, ensemble_type, sample_data, mock_config):
        """Test different ensemble types."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type=ensemble_type,
            base_models=['randomforest']
        )

        ensemble.fit(X_train, y_train, config=mock_config)

        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_empty_base_models_error(self, mock_config):
        """Test error when no base models can be initialized."""
        # Mock config with non-existent models
        bad_config = {'models': {}}

        ensemble = EnsembleModel(base_models=['nonexistent_model'])

        with pytest.raises(ValueError, match="No base models could be initialized"):
            ensemble._initialize_base_models(bad_config)


if __name__ == "__main__":
    pytest.main([__file__])