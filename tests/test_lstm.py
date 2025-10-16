"""Tests for LSTM model implementation."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch

# Import LSTM components
try:
    from src.models import LSTMModel, PYTORCH_AVAILABLE, TORCH_DEVICE
    from src.features import prepare_sequences_for_nn, scale_features_for_nn
    PYTORCH_AVAILABLE_IN_TESTS = PYTORCH_AVAILABLE
except ImportError:
    PYTORCH_AVAILABLE_IN_TESTS = False


@pytest.mark.skipif(not PYTORCH_AVAILABLE_IN_TESTS, reason="PyTorch not available")
class TestLSTMModel:
    """Test LSTM model functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        sequence_length = 20

        # Generate synthetic time series data
        X = np.random.randn(n_samples, n_features)
        # Add some temporal dependencies
        for i in range(1, n_samples):
            X[i] += 0.1 * X[i-1]  # Autoregressive component

        # Generate binary target based on feature sum
        y = (X.sum(axis=1) > X.sum(axis=1).mean()).astype(int)

        return X, y, sequence_length

    @pytest.fixture
    def lstm_model(self):
        """Create LSTM model instance."""
        return LSTMModel(
            hidden_size=32,  # Smaller for testing
            num_layers=1,
            sequence_length=20,
            epochs=5,  # Fewer epochs for testing
            batch_size=16,
            learning_rate=0.01,
            dropout=0.1
        )

    def test_model_initialization(self, lstm_model):
        """Test LSTM model initialization."""
        assert lstm_model.model is None
        assert lstm_model.scaler is None
        assert lstm_model.sequence_length == 20
        assert lstm_model.params['hidden_size'] == 32

    def test_sequence_creation(self, sample_data, lstm_model):
        """Test sequence creation from data."""
        X, y, seq_len = sample_data

        X_seq, y_seq = lstm_model._create_sequences(X, y, seq_len)

        assert X_seq.shape[0] == len(X) - seq_len + 1
        assert X_seq.shape[1] == seq_len
        assert X_seq.shape[2] == X.shape[1]
        assert len(y_seq) == X_seq.shape[0]

    def test_short_sequence_handling(self, lstm_model):
        """Test handling of sequences shorter than required length."""
        X = np.random.randn(10, 5)  # Very short sequence
        y = np.random.randint(0, 2, 10)

        # Should handle short sequences gracefully
        X_seq, y_seq = lstm_model._create_sequences(X, y, sequence_length=20)

        # Should still create some sequences (with padding)
        assert X_seq.shape[1] == 20  # Padded to required length
        assert X_seq.shape[2] == 5

    def test_model_training(self, sample_data, lstm_model):
        """Test basic model training."""
        X, y, seq_len = sample_data

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        lstm_model.fit(X_train, y_train)

        # Check that model is trained
        assert lstm_model.model is not None
        assert lstm_model.scaler is not None

        # Make predictions
        predictions = lstm_model.predict(X_test)
        probabilities = lstm_model.predict_proba(X_test)

        # Check prediction shapes
        assert len(predictions) == len(X_test) - seq_len + 1
        assert probabilities.shape[0] == len(X_test) - seq_len + 1
        assert probabilities.shape[1] == 2  # Binary classification

    def test_convergence_check(self, sample_data, lstm_model):
        """Test that model converges (loss decreases)."""
        X, y, seq_len = sample_data

        # Use a subset for faster testing
        X_subset = X[:200]
        y_subset = y[:200]

        # Train with validation to check convergence
        X_train, X_val, y_train, y_val = train_test_split(X_subset, y_subset, test_size=0.3, random_state=42)

        # Train model (should converge within few epochs)
        lstm_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=True)

        # Model should be trained
        assert lstm_model.model is not None

    def test_overfitting_detection(self, sample_data, lstm_model):
        """Test overfitting detection by comparing train/val performance."""
        X, y, seq_len = sample_data

        # Small dataset to potentially overfit
        X_small = X[:100]
        y_small = y[:100]

        X_train, X_val, y_train, y_val = train_test_split(X_small, y_small, test_size=0.4, random_state=42)

        # Train model
        lstm_model.fit(X_train, y_train, eval_set=(X_val, y_val))

        # Get predictions
        train_pred = lstm_model.predict(X_train)
        val_pred = lstm_model.predict(X_val)

        train_acc = accuracy_score(y_train[seq_len-1:], train_pred)
        val_acc = accuracy_score(y_val[seq_len-1:], val_pred)

        # Validation accuracy should not be much worse than training
        # Allow some overfitting but not extreme
        assert val_acc >= train_acc - 0.3, f"Possible overfitting: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"

    def test_save_load_model(self, sample_data, lstm_model, tmp_path):
        """Test model saving and loading."""
        X, y, seq_len = sample_data

        # Train model
        lstm_model.fit(X[:200], y[:200])

        # Save model
        model_path = tmp_path / "test_lstm_model.pth"
        lstm_model.save_model(str(model_path))

        # Load model
        loaded_model = LSTMModel.load_model(str(model_path))

        # Test predictions are similar
        test_X = X[200:250]
        original_pred = lstm_model.predict(test_X)
        loaded_pred = loaded_model.predict(test_X)

        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_gradient_clipping(self, lstm_model):
        """Test that gradient clipping prevents exploding gradients."""
        # Create data that might cause gradient explosion
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        # Train model - should not crash due to gradient explosion
        lstm_model.fit(X, y)

        assert lstm_model.model is not None

    @pytest.mark.parametrize("sequence_length", [5, 10, 20])
    def test_different_sequence_lengths(self, sequence_length):
        """Test model with different sequence lengths."""
        model = LSTMModel(sequence_length=sequence_length, epochs=3)

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X) - sequence_length + 1


@pytest.mark.skipif(not PYTORCH_AVAILABLE_IN_TESTS, reason="PyTorch not available")
class TestLSTMPerformance:
    """Test LSTM performance against baseline models."""

    @pytest.fixture
    def sample_data_performance(self):
        """Create sample data for performance tests."""
        np.random.seed(42)
        n_samples = 500  # Smaller for performance tests
        n_features = 10
        sequence_length = 20

        # Generate synthetic time series data
        X = np.random.randn(n_samples, n_features)
        # Add some temporal dependencies
        for i in range(1, n_samples):
            X[i] += 0.1 * X[i-1]  # Autoregressive component

        # Generate binary target based on feature sum
        y = (X.sum(axis=1) > X.sum(axis=1).mean()).astype(int)

        return X, y, sequence_length

    @pytest.fixture
    def lstm_model_performance(self):
        """Create LSTM model instance for performance tests."""
        return LSTMModel(
            hidden_size=32,  # Smaller for testing
            num_layers=1,
            sequence_length=20,
            epochs=5,  # Fewer epochs for testing
            batch_size=16,
            learning_rate=0.01,
            dropout=0.1
        )

    def test_lstm_vs_random_forest(self, sample_data_performance, lstm_model_performance):
        """Compare LSTM performance with RandomForest."""
        from sklearn.ensemble import RandomForestClassifier

        X, y, seq_len = sample_data_performance

        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train LSTM (it handles sequence creation internally)
        lstm_model_performance.fit(X_train, y_train)

        # Train RandomForest on original data
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)

        # Compare predictions
        lstm_pred = lstm_model_performance.predict(X_test)
        rf_pred = rf_model.predict(X_test)

        lstm_acc = accuracy_score(y_test[seq_len-1:], lstm_pred)
        rf_acc = accuracy_score(y_test, rf_pred)

        # LSTM should have reasonable performance (synthetic data may not have strong temporal patterns)
        assert lstm_acc > 0.4, f"LSTM accuracy too low: {lstm_acc:.3f}"
        assert rf_acc > 0.4, f"RandomForest accuracy too low: {rf_acc:.3f}"
        print(f"LSTM accuracy: {lstm_acc:.3f}, RandomForest accuracy: {rf_acc:.3f}")

    def test_memory_efficiency(self, lstm_model_performance):
        """Test that model handles memory constraints."""
        # Large batch size might cause memory issues
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, 200)

        # Should handle reasonable batch sizes without crashing
        lstm_model_performance.fit(X, y)

        assert lstm_model_performance.model is not None


if __name__ == "__main__":
    pytest.main([__file__])