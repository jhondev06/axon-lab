"""Tests for hyperparameter validation in AXON models."""

import pytest
import numpy as np
from sklearn.datasets import make_classification

# Import AXON models
from src.models import (
    XGBoostModel, CatBoostModel, LSTMModel, EnsembleModel,
    XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, PYTORCH_AVAILABLE
)


class TestXGBoostHyperparameterValidation:
    """Test XGBoost hyperparameter validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        return X, y

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_valid_hyperparameters(self, sample_data):
        """Test valid hyperparameter ranges."""
        X, y = sample_data

        # Test various valid parameter combinations
        valid_params = [
            {'n_estimators': 10, 'max_depth': 3, 'learning_rate': 0.1},
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.01},
            {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8},
            {'n_estimators': 20, 'max_depth': 5, 'gamma': 0.1, 'reg_alpha': 0.1},
        ]

        for params in valid_params:
            model = XGBoostModel(**params)
            model.fit(X, y)
            predictions = model.predict(X)
            assert len(predictions) == len(X)

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_invalid_hyperparameters(self):
        """Test invalid hyperparameter handling."""
        # Test negative values that should be handled gracefully
        with pytest.raises((ValueError, TypeError)):
            XGBoostModel(n_estimators=-1)

        with pytest.raises((ValueError, TypeError)):
            XGBoostModel(max_depth=0)

        with pytest.raises((ValueError, TypeError)):
            XGBoostModel(learning_rate=-0.1)

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_gpu_parameters(self):
        """Test GPU-related parameter validation."""
        # Test GPU parameters when GPU is available
        if hasattr(XGBoostModel, 'XGB_GPU_AVAILABLE') and XGBoostModel.XGB_GPU_AVAILABLE:
            model = XGBoostModel()
            assert 'gpu_id' in model.params
            assert model.params.get('tree_method') == 'gpu_hist'
        else:
            model = XGBoostModel()
            assert model.params.get('tree_method') == 'hist'

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_objective_validation(self, sample_data):
        """Test objective function validation."""
        X, y = sample_data

        valid_objectives = ['binary:logistic', 'binary:logitraw', 'binary:hinge']

        for objective in valid_objectives:
            model = XGBoostModel(objective=objective, n_estimators=10)
            model.fit(X, y)
            assert model.model is not None

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_parameter_bounds(self):
        """Test parameter bounds and constraints."""
        # Test extreme but valid values
        model = XGBoostModel(
            n_estimators=1000,  # Large number
            max_depth=15,       # Deep tree
            learning_rate=0.9,  # High learning rate
            subsample=0.5,      # Low subsample
            colsample_bytree=0.3  # Low column sample
        )

        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model.fit(X, y)

        assert model.model is not None


class TestCatBoostHyperparameterValidation:
    """Test CatBoost hyperparameter validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        return X, y

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_valid_hyperparameters(self, sample_data):
        """Test valid hyperparameter ranges."""
        X, y = sample_data

        valid_params = [
            {'iterations': 10, 'depth': 3, 'learning_rate': 0.1},
            {'iterations': 100, 'depth': 6, 'learning_rate': 0.01},
            {'iterations': 50, 'depth': 4, 'l2_leaf_reg': 3.0},
            {'iterations': 20, 'depth': 5, 'random_strength': 1.0, 'bagging_temperature': 1.0},
        ]

        for params in valid_params:
            model = CatBoostModel(**params, verbose=False)
            model.fit(X, y)
            predictions = model.predict(X)
            assert len(predictions) == len(X)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_invalid_hyperparameters(self):
        """Test invalid hyperparameter handling."""
        with pytest.raises((ValueError, TypeError)):
            CatBoostModel(iterations=-1, verbose=False)

        with pytest.raises((ValueError, TypeError)):
            CatBoostModel(depth=0, verbose=False)

        with pytest.raises((ValueError, TypeError)):
            CatBoostModel(learning_rate=-0.1, verbose=False)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_bootstrap_types(self, sample_data):
        """Test different bootstrap types."""
        X, y = sample_data

        bootstrap_configs = [
            {'bootstrap_type': 'Bayesian'},
            {'bootstrap_type': 'Bernoulli', 'subsample': 0.8},
            {'bootstrap_type': 'MVS'},
            {'bootstrap_type': 'No'},
        ]

        for config in bootstrap_configs:
            model = CatBoostModel(iterations=10, verbose=False, **config)
            model.fit(X, y)
            assert model.model is not None

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_grow_policies(self, sample_data):
        """Test different grow policies."""
        X, y = sample_data

        grow_policies = ['SymmetricTree', 'Depthwise', 'Lossguide']

        for policy in grow_policies:
            model = CatBoostModel(
                iterations=10,
                depth=4,
                grow_policy=policy,
                verbose=False
            )
            model.fit(X, y)
            assert model.model is not None

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_score_functions(self, sample_data):
        """Test different score functions."""
        X, y = sample_data

        score_functions = ['Cosine', 'L2', 'Newton', 'NewtonCosine']

        for score_func in score_functions:
            model = CatBoostModel(
                iterations=10,
                score_function=score_func,
                verbose=False
            )
            model.fit(X, y)
            assert model.model is not None


class TestLSTMHyperparameterValidation:
    """Test LSTM hyperparameter validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        return X, y

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_valid_hyperparameters(self, sample_data):
        """Test valid LSTM hyperparameter ranges."""
        X, y = sample_data

        valid_params = [
            {'hidden_size': 16, 'num_layers': 1, 'sequence_length': 10},
            {'hidden_size': 64, 'num_layers': 2, 'sequence_length': 20},
            {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.2, 'bidirectional': True},
            {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.3, 'bidirectional': False},
        ]

        for params in valid_params:
            model = LSTMModel(epochs=2, batch_size=16, **params)
            model.fit(X, y)
            predictions = model.predict(X)
            assert len(predictions) > 0

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_invalid_hyperparameters(self):
        """Test invalid LSTM hyperparameter handling."""
        with pytest.raises((ValueError, TypeError)):
            LSTMModel(hidden_size=0)

        with pytest.raises((ValueError, TypeError)):
            LSTMModel(num_layers=0)

        with pytest.raises((ValueError, TypeError)):
            LSTMModel(sequence_length=0)

        with pytest.raises((ValueError, TypeError)):
            LSTMModel(dropout=-0.1)

        with pytest.raises((ValueError, TypeError)):
            LSTMModel(dropout=1.5)

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_sequence_length_validation(self, sample_data):
        """Test sequence length validation."""
        X, y = sample_data

        # Test with sequence length larger than data
        model = LSTMModel(sequence_length=300, epochs=1, batch_size=16)

        # Should handle gracefully (padding)
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) > 0

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_bidirectional_settings(self, sample_data):
        """Test bidirectional parameter."""
        X, y = sample_data

        # Test bidirectional True
        model_bi = LSTMModel(
            hidden_size=16,
            bidirectional=True,
            epochs=2,
            batch_size=16
        )
        model_bi.fit(X, y)

        # Test bidirectional False
        model_uni = LSTMModel(
            hidden_size=16,
            bidirectional=False,
            epochs=2,
            batch_size=16
        )
        model_uni.fit(X, y)

        # Both should work
        assert model_bi.model is not None
        assert model_uni.model is not None

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_optimizer_parameters(self):
        """Test optimizer-related parameters."""
        # Valid optimizer parameters
        model = LSTMModel(
            learning_rate=0.001,
            weight_decay=1e-4,
            epochs=1
        )

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model.fit(X, y)

        assert model.model is not None


class TestEnsembleHyperparameterValidation:
    """Test Ensemble hyperparameter validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        return X, y

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            'models': {
                'randomforest': {'n_estimators': 5, 'random_state': 42}
            }
        }

    def test_valid_ensemble_types(self, sample_data, mock_config):
        """Test valid ensemble types."""
        X, y = sample_data

        valid_types = ['voting', 'stacking', 'blending', 'weighted']

        for ensemble_type in valid_types:
            model = EnsembleModel(
                ensemble_type=ensemble_type,
                base_models=['randomforest']
            )
            model.fit(X, y, config=mock_config)
            assert len(model.base_models) > 0

    def test_invalid_ensemble_type(self):
        """Test invalid ensemble type."""
        with pytest.raises(ValueError, match="Invalid ensemble_type"):
            EnsembleModel(ensemble_type='invalid')

    def test_invalid_combination_strategy(self):
        """Test invalid combination strategy."""
        with pytest.raises(ValueError, match="Invalid combination_strategy"):
            EnsembleModel(combination_strategy='invalid')

    def test_voting_types(self, sample_data, mock_config):
        """Test voting types."""
        X, y = sample_data

        for voting_type in ['hard', 'soft']:
            model = EnsembleModel(
                ensemble_type='voting',
                voting_type=voting_type,
                base_models=['randomforest']
            )
            model.fit(X, y, config=mock_config)
            assert model.params['voting_type'] == voting_type

    def test_cv_folds_validation(self, sample_data, mock_config):
        """Test CV folds validation."""
        X, y = sample_data

        # Valid CV folds
        model = EnsembleModel(
            ensemble_type='stacking',
            cv_folds=5,
            base_models=['randomforest']
        )
        model.fit(X, y, config=mock_config)
        assert model.params['cv_folds'] == 5

        # Invalid CV folds
        with pytest.raises((ValueError, TypeError)):
            EnsembleModel(cv_folds=0)

    def test_holdout_size_validation(self, sample_data, mock_config):
        """Test holdout size validation."""
        X, y = sample_data

        # Valid holdout size
        model = EnsembleModel(
            ensemble_type='blending',
            holdout_size=0.2,
            base_models=['randomforest']
        )
        model.fit(X, y, config=mock_config)
        assert model.params['holdout_size'] == 0.2

        # Invalid holdout sizes
        with pytest.raises((ValueError, TypeError)):
            EnsembleModel(holdout_size=-0.1)

        with pytest.raises((ValueError, TypeError)):
            EnsembleModel(holdout_size=1.5)


class TestHyperparameterCombinations:
    """Test combinations of hyperparameters."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X, y = make_classification(n_samples=300, n_features=8, random_state=42)
        return X, y

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_parameter_combinations(self, sample_data):
        """Test XGBoost parameter combinations."""
        X, y = sample_data

        # Test regularization parameters together
        model = XGBoostModel(
            n_estimators=20,
            max_depth=4,
            learning_rate=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            min_child_weight=1,
            verbosity=0
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert model.model is not None

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_parameter_combinations(self, sample_data):
        """Test CatBoost parameter combinations."""
        X, y = sample_data

        # Test complex parameter combinations
        model = CatBoostModel(
            iterations=20,
            depth=4,
            learning_rate=0.1,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            bagging_temperature=1.0,
            border_count=128,
            grow_policy='SymmetricTree',
            verbose=False
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert model.model is not None

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_parameter_combinations(self, sample_data):
        """Test LSTM parameter combinations."""
        X, y = sample_data

        # Test complex LSTM configuration
        model = LSTMModel(
            hidden_size=32,
            num_layers=2,
            sequence_length=15,
            dropout=0.2,
            bidirectional=True,
            batch_size=16,
            learning_rate=0.001,
            weight_decay=1e-4,
            epochs=3
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) > 0
        assert model.model is not None

    def test_ensemble_parameter_combinations(self, sample_data, mock_config):
        """Test Ensemble parameter combinations."""
        X, y = sample_data

        # Test complex ensemble configuration
        model = EnsembleModel(
            ensemble_type='weighted',
            combination_strategy='adaptive',
            base_models=['randomforest'],
            voting_type='soft',
            cv_folds=3,
            holdout_size=0.2,
            regime_detection=True,
            regime_window=30
        )

        model.fit(X, y, config=mock_config)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert len(model.base_models) > 0


class TestHyperparameterEdgeCases:
    """Test edge cases in hyperparameter validation."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_extreme_values(self):
        """Test XGBoost with extreme parameter values."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        # Test with very small learning rate
        model = XGBoostModel(
            n_estimators=10,
            learning_rate=1e-6,
            verbosity=0
        )
        model.fit(X, y)
        assert model.model is not None

        # Test with very large regularization
        model = XGBoostModel(
            n_estimators=10,
            reg_alpha=1000,
            reg_lambda=1000,
            verbosity=0
        )
        model.fit(X, y)
        assert model.model is not None

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_extreme_values(self):
        """Test CatBoost with extreme parameter values."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        # Test with very small learning rate
        model = CatBoostModel(
            iterations=10,
            learning_rate=1e-6,
            verbose=False
        )
        model.fit(X, y)
        assert model.model is not None

        # Test with high regularization
        model = CatBoostModel(
            iterations=10,
            l2_leaf_reg=1000,
            verbose=False
        )
        model.fit(X, y)
        assert model.model is not None

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_extreme_values(self):
        """Test LSTM with extreme parameter values."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        # Test with very small learning rate
        model = LSTMModel(
            learning_rate=1e-6,
            epochs=2,
            batch_size=16
        )
        model.fit(X, y)
        assert model.model is not None

        # Test with high weight decay
        model = LSTMModel(
            weight_decay=1e-2,
            epochs=2,
            batch_size=16
        )
        model.fit(X, y)
        assert model.model is not None


if __name__ == "__main__":
    pytest.main([__file__])