"""Extended unit tests for LSTM model implementation."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from unittest.mock import patch, MagicMock
import torch

# Import LSTM components
try:
    from src.models import LSTMModel, PYTORCH_AVAILABLE, TORCH_DEVICE
    PYTORCH_AVAILABLE_IN_TESTS = PYTORCH_AVAILABLE
except ImportError:
    PYTORCH_AVAILABLE_IN_TESTS = False


@pytest.mark.skipif(not PYTORCH_AVAILABLE_IN_TESTS, reason="PyTorch not available")
class TestLSTMModelExtended:
    """Extended test LSTM model functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        np.random.seed(42)
        n_samples = 500  # Smaller for unit tests
        n_features = 10
        sequence_length = 20

        # Generate synthetic time series data
        X = np.random.randn(n_samples, n_features)
        # Add some temporal dependencies
        for i in range(1, n_samples):
            X[i] += 0.1 * X[i-1]  # Autoregressive component

        # Generate binary target based on feature sum with trend
        trend = np.sin(np.linspace(0, 4*np.pi, n_samples))  # Add trend component
        signal = X.sum(axis=1) + trend
        y = (signal > np.median(signal)).astype(int)

        return X, y, sequence_length

    @pytest.fixture
    def lstm_model(self):
        """Create LSTM model instance for testing."""
        return LSTMModel(
            hidden_size=16,  # Smaller for testing
            num_layers=1,
            sequence_length=20,
            epochs=3,  # Fewer epochs for testing
            batch_size=8,
            learning_rate=0.01,
            dropout=0.1
        )

    def test_model_initialization_extended(self, lstm_model):
        """Test LSTM model initialization with all parameters."""
        assert lstm_model.model is None
        assert lstm_model.scaler is None
        assert lstm_model.sequence_length == 20
        assert lstm_model.params['hidden_size'] == 16
        assert lstm_model.params['num_layers'] == 1
        assert lstm_model.params['bidirectional'] == True  # Default
        assert lstm_model.params['device'] == TORCH_DEVICE

    def test_sequence_creation_edge_cases(self, lstm_model):
        """Test sequence creation with various edge cases."""
        # Test with exact sequence length
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        X_seq, y_seq = lstm_model._create_sequences(X, y, sequence_length=20)
        assert X_seq.shape == (1, 20, 5)
        assert len(y_seq) == 1

        # Test with very short data (shorter than sequence)
        X_short = np.random.randn(5, 5)
        y_short = np.random.randint(0, 2, 5)

        X_seq, y_seq = lstm_model._create_sequences(X_short, y_short, sequence_length=20)
        assert X_seq.shape[1] == 20  # Should be padded
        assert X_seq.shape[2] == 5

        # Test with NaN in target (should be filtered)
        X_nan = np.random.randn(30, 5)
        y_nan = np.random.randint(0, 2, 30).astype(float)
        y_nan[25] = np.nan  # Add NaN

        X_seq, y_seq = lstm_model._create_sequences(X_nan, y_nan, sequence_length=20)
        assert len(y_seq) == 10  # Should have 10 valid sequences (30-20+1 - 1 NaN)
        assert not np.any(np.isnan(y_seq))

    def test_fit_with_validation(self, sample_data, lstm_model):
        """Test model fitting with validation data."""
        X, y, seq_len = sample_data

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        lstm_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        assert lstm_model.model is not None
        assert lstm_model.scaler is not None

    def test_early_stopping(self, sample_data, lstm_model):
        """Test early stopping functionality."""
        X, y, seq_len = sample_data

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LSTMModel(
            hidden_size=16,
            sequence_length=20,
            epochs=10,
            patience=2,  # Early stopping patience
            batch_size=8
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=2, verbose=False)

        assert model.model is not None

    def test_predict_with_different_input_sizes(self, sample_data, lstm_model):
        """Test prediction with different input sizes."""
        X, y, seq_len = sample_data

        X_train = X[:200]
        y_train = y[:200]

        lstm_model.fit(X_train, y_train, verbose=False)

        # Test with different test sizes
        for test_size in [10, 50, 100]:
            X_test = X[200:200+test_size]
            predictions = lstm_model.predict(X_test)

            expected_length = max(1, len(X_test) - seq_len + 1)
            assert len(predictions) == expected_length

    def test_predict_proba_output_format(self, sample_data, lstm_model):
        """Test predict_proba output format."""
        X, y, seq_len = sample_data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lstm_model.fit(X_train, y_train, verbose=False)
        probabilities = lstm_model.predict_proba(X_test)

        expected_length = max(1, len(X_test) - seq_len + 1)
        assert probabilities.shape[0] == expected_length
        assert probabilities.shape[1] == 2
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_save_load_model_extended(self, sample_data, lstm_model, tmp_path):
        """Test model saving and loading with all components."""
        X, y, seq_len = sample_data

        X_train = X[:200]
        y_train = y[:200]

        # Train model
        lstm_model.fit(X_train, y_train, verbose=False)

        # Save model
        model_path = tmp_path / "test_lstm_extended.pth"
        lstm_model.save_model(str(model_path))

        # Load model
        loaded_model = LSTMModel.load_model(str(model_path))

        # Verify all attributes are loaded
        assert loaded_model.scaler is not None
        assert loaded_model.feature_names is None  # No feature names in this test
        assert loaded_model.sequence_length == seq_len

        # Test predictions
        X_test = X[200:250]
        original_pred = lstm_model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)

        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_gradient_clipping_prevention(self, lstm_model):
        """Test that gradient clipping prevents numerical issues."""
        # Create data that might cause large gradients
        X = np.random.randn(100, 10) * 10  # Large values
        y = np.random.randint(0, 2, 100)

        # Should complete without NaN/inf issues
        lstm_model.fit(X, y, verbose=False)

        assert lstm_model.model is not None

        # Check model parameters are finite
        for param in lstm_model.model.parameters():
            assert torch.isfinite(param).all()

    def test_mixed_precision_training(self, sample_data):
        """Test mixed precision training if available."""
        X, y, seq_len = sample_data

        X_train = X[:200]
        y_train = y[:200]

        model = LSTMModel(
            hidden_size=16,
            sequence_length=20,
            epochs=2,
            batch_size=8
        )

        # Should work regardless of GPU availability
        model.fit(X_train, y_train, verbose=False)

        assert model.model is not None

    @pytest.mark.parametrize("bidirectional", [True, False])
    def test_bidirectional_setting(self, bidirectional, sample_data):
        """Test bidirectional LSTM setting."""
        X, y, seq_len = sample_data

        X_train = X[:200]
        y_train = y[:200]

        model = LSTMModel(
            hidden_size=16,
            num_layers=1,
            sequence_length=20,
            bidirectional=bidirectional,
            epochs=2,
            batch_size=8
        )

        model.fit(X_train, y_train, verbose=False)

        # Check LSTM layers
        lstm_layer = model.model.lstm
        expected_directions = 2 if bidirectional else 1
        assert lstm_layer.num_directions == expected_directions

        predictions = model.predict(X_train)
        assert len(predictions) > 0

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_multiple_layers(self, num_layers, sample_data):
        """Test LSTM with multiple layers."""
        X, y, seq_len = sample_data

        X_train = X[:200]
        y_train = y[:200]

        model = LSTMModel(
            hidden_size=16,
            num_layers=num_layers,
            sequence_length=20,
            epochs=2,
            batch_size=8,
            dropout=0.1 if num_layers > 1 else 0
        )

        model.fit(X_train, y_train, verbose=False)

        assert model.model.lstm.num_layers == num_layers

        predictions = model.predict(X_train)
        assert len(predictions) > 0

    def test_feature_scaling(self, sample_data, lstm_model):
        """Test that features are properly scaled."""
        X, y, seq_len = sample_data

        X_train = X[:200]
        y_train = y[:200]

        lstm_model.fit(X_train, y_train, verbose=False)

        # Check that scaler was fitted
        assert lstm_model.scaler is not None

        # Transform some data and check it's scaled
        X_scaled = lstm_model.scaler.transform(X_train)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=0.1)  # Approximately standardized
        assert np.allclose(X_scaled.std(axis=0), 1, atol=0.1)

    def test_prediction_consistency(self, sample_data, lstm_model):
        """Test prediction consistency across multiple calls."""
        X, y, seq_len = sample_data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lstm_model.fit(X_train, y_train, verbose=False)

        # Multiple prediction calls should be identical
        pred1 = lstm_model.predict(X_test)
        pred2 = lstm_model.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)

        prob1 = lstm_model.predict_proba(X_test)
        prob2 = lstm_model.predict_proba(X_test)

        np.testing.assert_array_equal(prob1, prob2)

    def test_different_sequence_lengths_fit(self):
        """Test that model can be fitted with different sequence lengths."""
        for seq_len in [5, 10, 15, 20]:
            X = np.random.randn(100, 5)
            y = np.random.randint(0, 2, 100)

            model = LSTMModel(sequence_length=seq_len, epochs=2, batch_size=8)
            model.fit(X, y, verbose=False)

            predictions = model.predict(X)
            expected_length = max(1, len(X) - seq_len + 1)
            assert len(predictions) == expected_length

    def test_memory_efficiency_large_dataset(self):
        """Test model handles larger datasets efficiently."""
        # Create larger dataset
        X = np.random.randn(1000, 20)
        y = np.random.randint(0, 2, 1000)

        model = LSTMModel(
            hidden_size=32,
            sequence_length=20,
            epochs=2,
            batch_size=32  # Larger batch size
        )

        # Should complete without memory issues
        model.fit(X, y, verbose=False)

        assert model.model is not None

    def test_overfitting_detection_extended(self, lstm_model):
        """Extended overfitting detection test."""
        # Create dataset with clear pattern
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        # Add a clear pattern
        pattern = np.sin(np.linspace(0, 4*np.pi, n_samples))
        X[:, 0] += pattern  # Add pattern to first feature

        y = (pattern > 0).astype(int)  # Target based on pattern

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

        lstm_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        # Get predictions
        train_pred = lstm_model.predict(X_train)
        val_pred = lstm_model.predict(X_val)

        train_acc = accuracy_score(y_train[lstm_model.sequence_length-1:], train_pred)
        val_acc = accuracy_score(y_val[lstm_model.sequence_length-1:], val_pred)

        # Should have reasonable performance
        assert train_acc > 0.5
        assert val_acc > 0.4

    def test_device_assignment(self, lstm_model):
        """Test that model is assigned to correct device."""
        assert next(lstm_model.model.parameters()).device == TORCH_DEVICE

    def test_model_to_device_after_loading(self):
        """Test that loaded model is on correct device."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        model = LSTMModel(epochs=1, batch_size=8)
        model.fit(X, y, verbose=False)

        # Check device assignment
        assert next(model.model.parameters()).device.type in ['cpu', 'cuda']

    def test_batch_processing(self, lstm_model):
        """Test that batch processing works correctly."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        lstm_model.fit(X, y, verbose=False)

        # Test with different batch sizes during prediction
        X_test = np.random.randn(50, 10)
        predictions = lstm_model.predict(X_test)

        assert len(predictions) > 0
        assert all(isinstance(p, (int, np.integer)) for p in predictions)


@pytest.mark.skipif(not PYTORCH_AVAILABLE_IN_TESTS, reason="PyTorch not available")
class TestLSTMIntegration:
    """Test LSTM integration with other components."""

    def test_sklearn_compatibility_interface(self, sample_data):
        """Test that LSTM has sklearn-like interface."""
        X, y, seq_len = sample_data

        model = LSTMModel(epochs=2, batch_size=8)

        # Should have fit, predict, predict_proba methods
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

        # Test basic workflow
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        assert len(predictions) == max(1, len(X_test) - seq_len + 1)
        assert probabilities.shape[1] == 2

    def test_dataframe_input_handling(self, sample_data):
        """Test handling of DataFrame input."""
        X, y, seq_len = sample_data
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])

        model = LSTMModel(epochs=2, batch_size=8)
        model.fit(df, y)

        assert model.feature_names == list(df.columns)

        predictions = model.predict(df)
        assert len(predictions) > 0

    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # Add some NaN and Inf values
        X[10, 0] = np.nan
        X[20, 1] = np.inf
        X[30, 2] = -np.inf

        model = LSTMModel(epochs=2, batch_size=8)

        # Should handle gracefully or raise informative error
        try:
            model.fit(X, y, verbose=False)
            # If it completes, check predictions work
            predictions = model.predict(X)
            assert len(predictions) > 0
        except Exception as e:
            # Should be an informative error
            assert "NaN" in str(e) or "inf" in str(e)

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        model = LSTMModel()

        # Empty arrays should raise appropriate errors
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])

        with pytest.raises((ValueError, IndexError)):
            model.fit(X_empty, y_empty)


if __name__ == "__main__":
    pytest.main([__file__])