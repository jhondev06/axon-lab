"""Robustness tests for AXON models and pipeline."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import AXON components
from src.models import (
    XGBoostModel, CatBoostModel, LSTMModel, EnsembleModel,
    XGBOOST_AVAILABLE, CATBOOST_AVAILABLE, PYTORCH_AVAILABLE
)


class TestMissingDataRobustness:
    """Test robustness with missing data."""

    @pytest.fixture
    def data_with_missing(self):
        """Create dataset with missing values."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)

        # Introduce missing values
        X_missing = X.copy()
        np.random.seed(42)

        # 10% missing values randomly distributed
        missing_mask = np.random.random(X.shape) < 0.1
        X_missing[missing_mask] = np.nan

        return X_missing, y

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_missing_data_handling(self, data_with_missing):
        """Test XGBoost handles missing data."""
        X, y = data_with_missing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBoostModel(n_estimators=50, verbosity=0)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_missing_data_handling(self, data_with_missing):
        """Test CatBoost handles missing data."""
        X, y = data_with_missing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = CatBoostModel(iterations=50, verbose=False)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert not np.any(np.isnan(predictions))

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_missing_data_handling(self, data_with_missing):
        """Test LSTM handles missing data."""
        X, y = data_with_missing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LSTMModel(sequence_length=20, epochs=3, batch_size=16)

        # LSTM may not handle NaN as gracefully as tree models
        # This tests the robustness of error handling
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            assert len(predictions) > 0
        except Exception as e:
            # Acceptable to fail with informative error
            assert "NaN" in str(e) or "nan" in str(e).lower()

    def test_ensemble_missing_data_handling(self, data_with_missing):
        """Test Ensemble handles missing data."""
        X, y = data_with_missing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        model.fit(X_train, y_train, config={'models': {'randomforest': {}}})

        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)


class TestSequenceRobustness:
    """Test robustness with sequence data (LSTM-specific)."""

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_short_sequences(self):
        """Test LSTM with sequences shorter than required length."""
        # Create data shorter than sequence length
        X = np.random.randn(30, 5)  # Only 30 samples
        y = np.random.randint(0, 2, 30)

        model = LSTMModel(sequence_length=50, epochs=2, batch_size=8)

        # Should handle by padding or truncation
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) > 0

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_variable_sequence_lengths(self):
        """Test LSTM with different sequence lengths."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        for seq_len in [5, 15, 25]:
            model = LSTMModel(sequence_length=seq_len, epochs=2, batch_size=8)
            model.fit(X, y)

            predictions = model.predict(X)
            expected_min_length = max(1, len(X) - seq_len + 1)
            assert len(predictions) >= expected_min_length

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_sequence_padding(self):
        """Test LSTM sequence padding behavior."""
        # Very short data
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)

        model = LSTMModel(sequence_length=20, epochs=2, batch_size=8)
        model.fit(X, y)

        # Should create sequences by padding
        X_seq, y_seq = model._create_sequences(X, y, 20)

        assert X_seq.shape[1] == 20  # Padded to sequence length
        assert X_seq.shape[2] == 5   # Original features
        assert len(y_seq) > 0


class TestOverfittingDetection:
    """Test overfitting detection and prevention."""

    def test_overfitting_detection_random_forest(self):
        """Test overfitting detection with RandomForest."""
        # Create dataset prone to overfitting
        np.random.seed(42)
        X = np.random.randn(200, 10)
        # Add some noise that's hard to learn
        y = (X[:, 0] + 0.1 * np.random.randn(200) > 0).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        train_accuracy = accuracy_score(y_train, train_predictions)
        val_accuracy = accuracy_score(y_val, val_predictions)

        # Should not overfit severely
        overfitting_gap = train_accuracy - val_accuracy
        assert overfitting_gap < 0.3, f"Severe overfitting detected: gap = {overfitting_gap:.3f}"

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_overfitting_detection_xgboost(self):
        """Test overfitting detection with XGBoost."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] + 0.1 * np.random.randn(200) > 0).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

        model = XGBoostModel(n_estimators=100, verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        train_accuracy = accuracy_score(y_train, train_predictions)
        val_accuracy = accuracy_score(y_val, val_predictions)

        overfitting_gap = train_accuracy - val_accuracy
        assert overfitting_gap < 0.4, f"Severe overfitting: gap = {overfitting_gap:.3f}"

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_overfitting_detection_catboost(self):
        """Test overfitting detection with CatBoost."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] + 0.1 * np.random.randn(200) > 0).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

        model = CatBoostModel(iterations=100, verbose=False, early_stopping_rounds=10)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        train_accuracy = accuracy_score(y_train, train_predictions)
        val_accuracy = accuracy_score(y_val, val_predictions)

        overfitting_gap = train_accuracy - val_accuracy
        assert overfitting_gap < 0.4, f"Severe overfitting: gap = {overfitting_gap:.3f}"

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_overfitting_detection_lstm(self):
        """Test overfitting detection with LSTM."""
        np.random.seed(42)
        X = np.random.randn(150, 10)
        y = (X[:, 0] + 0.1 * np.random.randn(150) > 0).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

        model = LSTMModel(
            sequence_length=15,
            epochs=20,
            batch_size=16,
            patience=5  # Early stopping
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        # Get predictions aligned with sequence requirements
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Align predictions with targets
        train_target = y_train[-len(train_pred):]
        val_target = y_val[-len(val_pred):]

        if len(train_pred) > 0 and len(val_pred) > 0:
            train_accuracy = accuracy_score(train_target, train_pred)
            val_accuracy = accuracy_score(val_target, val_pred)

            overfitting_gap = train_accuracy - val_accuracy
            assert overfitting_gap < 0.5, f"LSTM overfitting: gap = {overfitting_gap:.3f}"


class TestRegimeDetectionRobustness:
    """Test regime detection robustness."""

    def test_regime_detection_stability(self):
        """Test regime detection stability."""
        # Create data with different market regimes
        np.random.seed(42)

        # Bull regime
        bull_data = np.random.randn(100, 5) + 0.1  # Upward trend
        bull_returns = np.cumsum(np.random.normal(0.001, 0.01, 100))

        # Bear regime
        bear_data = np.random.randn(100, 5) - 0.1  # Downward trend
        bear_returns = np.cumsum(np.random.normal(-0.001, 0.01, 100))

        # Volatile regime
        volatile_data = np.random.randn(100, 5)
        volatile_returns = np.cumsum(np.random.normal(0, 0.05, 100))  # High volatility

        # Combine data
        X = np.vstack([bull_data, bear_data, volatile_data])
        returns = np.concatenate([bull_returns, bear_returns, volatile_returns])

        ensemble = EnsembleModel(
            ensemble_type='weighted',
            combination_strategy='adaptive',
            regime_detection=True,
            regime_window=30
        )

        # Test regime detection doesn't crash
        regime = ensemble._detect_market_regime(X, returns)
        assert regime in ['neutral', 'high_volatility', 'bull_trend', 'bear_trend']

    def test_adaptive_weighting_robustness(self):
        """Test adaptive weighting robustness."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        ensemble = EnsembleModel(
            ensemble_type='weighted',
            combination_strategy='adaptive',
            base_models=['randomforest']
        )

        # Initialize base models
        ensemble._initialize_base_models({'models': {'randomforest': {}}})

        # Fit base models
        for name, model in ensemble.base_models.items():
            model.fit(X_train, y_train)

        # Test adaptive weighting with different regimes
        for regime in ['neutral', 'high_volatility', 'bull_trend', 'bear_trend']:
            weights = ensemble._calculate_model_weights(X_val, y_val, regime)
            assert isinstance(weights, dict)
            assert len(weights) > 0
            assert all(w >= 0 for w in weights.values())

            # Weights should sum to approximately 1
            total_weight = sum(weights.values())
            assert 0.99 <= total_weight <= 1.01

    def test_regime_detection_edge_cases(self):
        """Test regime detection with edge cases."""
        ensemble = EnsembleModel(regime_detection=True, regime_window=50)

        # Empty data
        regime = ensemble._detect_market_regime(np.array([]).reshape(0, 5))
        assert regime == 'neutral'

        # Data shorter than window
        short_data = np.random.randn(10, 5)
        regime = ensemble._detect_market_regime(short_data)
        assert regime == 'neutral'

        # Data with NaN
        nan_data = np.random.randn(100, 5)
        nan_data[50:60] = np.nan
        regime = ensemble._detect_market_regime(nan_data)
        assert regime in ['neutral', 'high_volatility', 'bull_trend', 'bear_trend']


class TestNoiseRobustness:
    """Test robustness to noisy data."""

    def test_noise_robustness_random_forest(self):
        """Test RandomForest robustness to noise."""
        # Create clean signal
        np.random.seed(42)
        X_clean = np.random.randn(500, 5)
        y = (X_clean[:, 0] > 0).astype(int)

        # Add increasing amounts of noise
        noise_levels = [0.1, 0.5, 1.0, 2.0]

        from sklearn.ensemble import RandomForestClassifier

        for noise_level in noise_levels:
            X_noisy = X_clean + noise_level * np.random.randn(*X_clean.shape)

            X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # Should maintain reasonable performance even with noise
            assert accuracy > 0.5, f"Poor performance with noise level {noise_level}: {accuracy:.3f}"

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_noise_robustness_xgboost(self):
        """Test XGBoost robustness to noise."""
        np.random.seed(42)
        X_clean = np.random.randn(500, 5)
        y = (X_clean[:, 0] > 0).astype(int)

        noise_levels = [0.1, 0.5, 1.0]

        for noise_level in noise_levels:
            X_noisy = X_clean + noise_level * np.random.randn(*X_clean.shape)

            X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)

            model = XGBoostModel(n_estimators=50, verbosity=0)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            assert accuracy > 0.5, f"XGBoost poor performance with noise {noise_level}: {accuracy:.3f}"

    def test_label_noise_robustness(self):
        """Test robustness to label noise."""
        X, y_clean = make_classification(n_samples=500, n_features=10, random_state=42)

        # Add label noise
        noise_ratios = [0.05, 0.1, 0.2]  # 5%, 10%, 20% noise

        from sklearn.ensemble import RandomForestClassifier

        for noise_ratio in noise_ratios:
            # Flip some labels
            y_noisy = y_clean.copy()
            flip_indices = np.random.choice(len(y_noisy), size=int(noise_ratio * len(y_noisy)), replace=False)
            y_noisy[flip_indices] = 1 - y_noisy[flip_indices]

            X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # Should still perform reasonably
            assert accuracy > 0.4, f"Poor performance with {noise_ratio*100}% label noise: {accuracy:.3f}"


class TestOutlierRobustness:
    """Test robustness to outliers."""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_outlier_robustness_xgboost(self):
        """Test XGBoost robustness to outliers."""
        X, y = make_classification(n_samples=500, n_features=5, random_state=42)

        # Add extreme outliers
        outlier_indices = np.random.choice(len(X), size=20, replace=False)
        X[outlier_indices] *= 100  # Make them extreme

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBoostModel(n_estimators=50, verbosity=0)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Should handle outliers reasonably
        assert accuracy > 0.6, f"Poor outlier handling: {accuracy:.3f}"

    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_outlier_robustness_catboost(self):
        """Test CatBoost robustness to outliers."""
        X, y = make_classification(n_samples=500, n_features=5, random_state=42)

        outlier_indices = np.random.choice(len(X), size=20, replace=False)
        X[outlier_indices] *= 100

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = CatBoostModel(iterations=50, verbose=False)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert accuracy > 0.6, f"CatBoost poor outlier handling: {accuracy:.3f}"

    def test_ensemble_outlier_robustness(self):
        """Test Ensemble robustness to outliers."""
        X, y = make_classification(n_samples=500, n_features=5, random_state=42)

        outlier_indices = np.random.choice(len(X), size=20, replace=False)
        X[outlier_indices] *= 100

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = EnsembleModel(ensemble_type='voting', base_models=['randomforest'])
        model.fit(X_train, y_train, config={'models': {'randomforest': {}}})

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert accuracy > 0.6, f"Ensemble poor outlier handling: {accuracy:.3f}"


class TestDistributionShiftRobustness:
    """Test robustness to distribution shifts."""

    def test_concept_drift_simulation(self):
        """Test robustness to concept drift."""
        # Train on one distribution, test on another
        np.random.seed(42)

        # Training data: feature 0 is informative
        X_train = np.random.randn(500, 5)
        X_train[:, 0] += np.random.randn(500) * 0.5  # Add signal
        y_train = (X_train[:, 0] > 0).astype(int)

        # Test data: feature 1 becomes informative instead
        X_test = np.random.randn(200, 5)
        X_test[:, 1] += np.random.randn(200) * 0.5  # Different signal
        y_test = (X_test[:, 1] > 0).astype(int)

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Performance will be poor due to distribution shift, but should not crash
        # Just verify it completes and gives reasonable output
        assert len(predictions) == len(X_test)
        assert accuracy > 0.3  # Better than random guessing


class TestNumericalStability:
    """Test numerical stability."""

    def test_extreme_feature_scales(self):
        """Test robustness to extreme feature scales."""
        X, y = make_classification(n_samples=300, n_features=5, random_state=42)

        # Make features have very different scales
        X[:, 0] *= 1e-6   # Very small
        X[:, 1] *= 1e6    # Very large
        X[:, 2] *= 1e-3   # Small
        X[:, 3] *= 1e3    # Large
        # X[:, 4] remains normal

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert accuracy > 0.5, f"Poor performance with scale differences: {accuracy:.3f}"

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_numerical_stability(self):
        """Test LSTM numerical stability."""
        # Create data that might cause numerical issues
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randint(0, 2, 100)

        # Add some extreme values
        X[10:20] *= 1000

        model = LSTMModel(sequence_length=15, epochs=3, batch_size=16)

        # Should complete without numerical issues
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) > 0
        assert not np.any(np.isnan(predictions))


if __name__ == "__main__":
    pytest.main([__file__])