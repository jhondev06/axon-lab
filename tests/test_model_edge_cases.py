"""Tests for edge cases in AXON models."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Import AXON models
from src.models import (
    XGBoostModel, CatBoostModel, LSTMModel, EnsembleModel,
    XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, PYTORCH_AVAILABLE
)


class TestDataEdgeCases:
    """Test models with edge case data."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_minimum_data(self):
        """Test XGBoost with minimum possible data."""
        # Very small dataset
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        model = XGBoostModel(n_estimators=1, verbosity=0)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_minimum_data(self):
        """Test CatBoost with minimum possible data."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        model = CatBoostModel(iterations=1, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_minimum_data(self):
        """Test LSTM with minimum data for sequences."""
        # Need enough data for sequence creation
        X = np.random.randn(25, 5)  # 25 samples, 5 features
        y = np.random.randint(0, 2, 25)

        model = LSTMModel(sequence_length=10, epochs=1, batch_size=8)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) > 0

    def test_ensemble_minimum_data(self, mock_config):
        """Test Ensemble with minimum data."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        model = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        model.fit(X, y, config=mock_config)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_high_dimensional_data(self):
        """Test XGBoost with high-dimensional sparse data."""
        # High dimensional but sparse data
        X, y = make_classification(
            n_samples=100, n_features=1000, n_informative=10,
            n_redundant=0, random_state=42
        )

        model = XGBoostModel(n_estimators=10, verbosity=0)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_high_dimensional_data(self):
        """Test CatBoost with high-dimensional data."""
        X, y = make_classification(
            n_samples=100, n_features=500, n_informative=10,
            random_state=42
        )

        model = CatBoostModel(iterations=10, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_high_dimensional_features(self):
        """Test LSTM with many features."""
        X = np.random.randn(100, 50)  # 100 samples, 50 features
        y = np.random.randint(0, 2, 100)

        model = LSTMModel(
            sequence_length=20,
            epochs=2,
            batch_size=16,
            hidden_size=32
        )
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) > 0

    def test_ensemble_high_dimensional_data(self, mock_config):
        """Test Ensemble with high-dimensional data."""
        X, y = make_classification(
            n_samples=100, n_features=200, n_informative=10,
            random_state=42
        )

        model = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        model.fit(X, y, config=mock_config)

        predictions = model.predict(X)
        assert len(predictions) == len(X)


class TestImbalancedData:
    """Test models with imbalanced datasets."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_extreme_imbalance(self):
        """Test XGBoost with extremely imbalanced data."""
        # Create very imbalanced dataset
        X = np.random.randn(1000, 10)
        y = np.zeros(1000)
        y[:10] = 1  # Only 1% positive class

        model = XGBoostModel(n_estimators=20, scale_pos_weight=99, verbosity=0)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

        # Should predict some positives
        assert sum(predictions) > 0

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_extreme_imbalance(self):
        """Test CatBoost with extremely imbalanced data."""
        X = np.random.randn(1000, 10)
        y = np.zeros(1000)
        y[:10] = 1

        # CatBoost handles imbalance automatically
        model = CatBoostModel(iterations=20, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_imbalanced_sequences(self):
        """Test LSTM with imbalanced sequential data."""
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 10)
        y = np.zeros(n_samples)
        y[:20] = 1  # 10% positive

        model = LSTMModel(sequence_length=15, epochs=3, batch_size=16)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) > 0

    def test_ensemble_imbalanced_data(self, mock_config):
        """Test Ensemble with imbalanced data."""
        X = np.random.randn(500, 10)
        y = np.zeros(500)
        y[:25] = 1  # 5% positive

        model = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        model.fit(X, y, config=mock_config)

        predictions = model.predict(X)
        assert len(predictions) == len(X)


class TestCorruptedData:
    """Test models with corrupted or noisy data."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_with_nan_values(self):
        """Test XGBoost handling of NaN values."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # Introduce NaN values
        X[10:20, 0] = np.nan
        X[30:40, 2] = np.nan

        model = XGBoostModel(n_estimators=10, verbosity=0)

        # XGBoost should handle NaN gracefully
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_with_nan_values(self):
        """Test CatBoost handling of NaN values."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        X[10:20, 0] = np.nan
        X[30:40, 2] = np.nan

        model = CatBoostModel(iterations=10, verbose=False)

        # CatBoost handles NaN natively
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_with_nan_values(self):
        """Test LSTM handling of NaN values."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        X[10:20, 0] = np.nan

        model = LSTMModel(sequence_length=10, epochs=2, batch_size=16)

        # Should handle NaN or raise informative error
        try:
            model.fit(X, y)
            predictions = model.predict(X)
            assert len(predictions) > 0
        except Exception as e:
            # Acceptable to fail with NaN in LSTM
            assert "NaN" in str(e) or "nan" in str(e).lower()

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_extreme_outliers(self):
        """Test XGBoost with extreme outliers."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # Add extreme outliers
        X[10, 0] = 1000
        X[20, 1] = -1000
        X[30, 2] = 1e6

        model = XGBoostModel(n_estimators=10, verbosity=0)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_extreme_outliers(self):
        """Test CatBoost with extreme outliers."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        X[10, 0] = 1000
        X[20, 1] = -1000

        model = CatBoostModel(iterations=10, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)


class TestSequenceEdgeCases:
    """Test LSTM-specific sequence edge cases."""

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_sequence_longer_than_data(self):
        """Test LSTM when sequence length exceeds data length."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        # Sequence length longer than data
        model = LSTMModel(sequence_length=100, epochs=1, batch_size=16)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) > 0

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_variable_sequence_lengths(self):
        """Test LSTM with different sequence lengths."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        for seq_len in [5, 10, 20, 50]:
            model = LSTMModel(sequence_length=seq_len, epochs=1, batch_size=16)
            model.fit(X, y)

            predictions = model.predict(X)
            assert len(predictions) > 0

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_single_feature(self):
        """Test LSTM with single feature."""
        X = np.random.randn(100, 1)  # Single feature
        y = np.random.randint(0, 2, 100)

        model = LSTMModel(sequence_length=10, epochs=2, batch_size=16)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) > 0


class TestEnsembleEdgeCases:
    """Test Ensemble-specific edge cases."""

    def test_ensemble_single_model(self, mock_config):
        """Test ensemble with single base model."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        model = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        model.fit(X, y, config=mock_config)

        assert len(model.base_models) >= 1

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_ensemble_empty_base_models(self, mock_config):
        """Test ensemble with no valid base models."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        # Use invalid model names
        model = EnsembleModel(ensemble_type='voting', base_models=['invalid_model'])

        with pytest.raises(ValueError, match="No base models could be initialized"):
            model.fit(X, y, config=mock_config)

    def test_ensemble_mixed_model_types(self, mock_config):
        """Test ensemble with mixed model types."""
        # This depends on what's available
        available_models = ['randomforest']  # Always available

        if XGBOOST_AVAILABLE:
            available_models.append('xgboost')
        if CATBOOST_AVAILABLE:
            available_models.append('catboost')

        if len(available_models) > 1:
            X = np.random.randn(100, 5)
            y = np.random.randint(0, 2, 100)

            model = EnsembleModel(
                ensemble_type='voting',
                base_models=available_models
            )
            model.fit(X, y, config=mock_config)

            assert len(model.base_models) > 1

            predictions = model.predict(X)
            assert len(predictions) == len(X)


class TestMemoryEdgeCases:
    """Test models with memory-intensive scenarios."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_large_dataset(self):
        """Test XGBoost with relatively large dataset."""
        X, y = make_classification(
            n_samples=1000, n_features=50, random_state=42
        )

        model = XGBoostModel(n_estimators=20, verbosity=0)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_large_dataset(self):
        """Test CatBoost with relatively large dataset."""
        X, y = make_classification(
            n_samples=1000, n_features=50, random_state=42
        )

        model = CatBoostModel(iterations=20, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_large_sequences(self):
        """Test LSTM with large sequences."""
        X = np.random.randn(200, 20)  # More features
        y = np.random.randint(0, 2, 200)

        model = LSTMModel(
            sequence_length=30,
            hidden_size=64,
            epochs=2,
            batch_size=32
        )
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) > 0


class TestPredictionEdgeCases:
    """Test prediction edge cases."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_predict_single_sample(self):
        """Test XGBoost prediction on single sample."""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)

        model = XGBoostModel(n_estimators=5, verbosity=0)
        model.fit(X, y)

        # Predict single sample
        single_sample = X[0:1]
        prediction = model.predict(single_sample)
        probability = model.predict_proba(single_sample)

        assert len(prediction) == 1
        assert probability.shape == (1, 2)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_predict_single_sample(self):
        """Test CatBoost prediction on single sample."""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)

        model = CatBoostModel(iterations=5, verbose=False)
        model.fit(X, y)

        single_sample = X[0:1]
        prediction = model.predict(single_sample)
        probability = model.predict_proba(single_sample)

        assert len(prediction) == 1
        assert probability.shape == (1, 2)

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_predict_single_sample(self):
        """Test LSTM prediction on minimal data."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        model = LSTMModel(sequence_length=10, epochs=1, batch_size=16)
        model.fit(X, y)

        # Predict with minimal data
        test_X = np.random.randn(15, 5)  # Just enough for one sequence
        predictions = model.predict(test_X)

        assert len(predictions) > 0

    def test_ensemble_predict_single_sample(self, mock_config):
        """Test Ensemble prediction on single sample."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        model = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        model.fit(X, y, config=mock_config)

        single_sample = X[0:1]
        prediction = model.predict(single_sample)
        probability = model.predict_proba(single_sample)

        assert len(prediction) == 1
        assert probability.shape == (1, 2)


class TestDataTypeEdgeCases:
    """Test models with different data types."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_integer_features(self):
        """Test XGBoost with integer features."""
        X = np.random.randint(0, 10, size=(50, 5))
        y = np.random.randint(0, 2, 50)

        model = XGBoostModel(n_estimators=5, verbosity=0)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_integer_features(self):
        """Test CatBoost with integer features."""
        X = np.random.randint(0, 10, size=(50, 5))
        y = np.random.randint(0, 2, 50)

        model = CatBoostModel(iterations=5, verbose=False)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_ensemble_integer_features(self, mock_config):
        """Test Ensemble with integer features."""
        X = np.random.randint(0, 10, size=(50, 5))
        y = np.random.randint(0, 2, 50)

        model = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        model.fit(X, y, config=mock_config)

        predictions = model.predict(X)
        assert len(predictions) == len(X)


if __name__ == "__main__":
    pytest.main([__file__])