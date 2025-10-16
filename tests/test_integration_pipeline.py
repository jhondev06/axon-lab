"""Integration tests for complete AXON pipeline."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import AXON modules
from src.models import ModelRegistry, train_model, save_model, load_model
from src.dataset import load_data, prepare_data
from src.features import create_features, prepare_features
from src.backtest import run_backtest
from src.metrics import calculate_metrics
from src.utils import load_config


@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    return {
        'data': {
            'sources': ['synthetic'],
            'symbols': ['BTC-USD'],
            'lookback_days': 7,  # Small for testing
            'interval': '1m',
            'quality': {
                'min_rows_test': 10
            }
        },
        'features': {
            'use': ['returns', 'ema_5', 'rsi_14']
        },
        'labels': {
            'method': 'triple_barrier',
            'horizon': '5m'
        },
        'models': {
            'train': ['randomforest']
        },
        'backtest': {
            'threshold': 0.5,
            'position_size': 1.0,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_positions': 1,
            'commission': 0.0,
            'initial_capital': 1000.0,
            'timeout': '1m'
        },
        'target': 'y'
    }


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for integration tests."""
    np.random.seed(42)

    n_rows = 1000
    base_price = 50000.0
    timestamps = pd.date_range('2023-01-01', periods=n_rows, freq='1min')

    # Generate realistic price movements
    returns = np.random.normal(0.0001, 0.01, n_rows)
    prices = base_price * np.exp(np.cumsum(returns))

    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_rows))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_rows))),
        'close': prices * (1 + np.random.normal(0, 0.002, n_rows)),
        'volume': np.random.randint(100, 10000, n_rows)
    }

    df = pd.DataFrame(data)

    # Ensure OHLCV integrity
    df['high'] = np.maximum(df[['open', 'high', 'close']].max(axis=1), df['high'])
    df['low'] = np.minimum(df[['open', 'low', 'close']].min(axis=1), df['low'])

    return df


class TestDataPipelineIntegration:
    """Test data loading and preparation pipeline."""

    def test_data_loading_pipeline(self, sample_ohlcv_data, integration_config):
        """Test complete data loading pipeline."""
        # Mock the data loading to use our sample data
        with patch('src.dataset.load_raw_data') as mock_load:
            mock_load.return_value = sample_ohlcv_data

            # Test data preparation
            prepared_data = prepare_data(integration_config)

            assert isinstance(prepared_data, pd.DataFrame)
            assert len(prepared_data) > 0
            assert 'timestamp' in prepared_data.columns

    def test_data_quality_checks(self, sample_ohlcv_data, integration_config):
        """Test data quality validation in pipeline."""
        with patch('src.dataset.load_raw_data') as mock_load:
            mock_load.return_value = sample_ohlcv_data

            prepared_data = prepare_data(integration_config)

            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                assert col in prepared_data.columns

            # Check for NaN values
            assert not prepared_data.isnull().any().any()


class TestFeaturesPipelineIntegration:
    """Test features creation and preparation pipeline."""

    def test_features_creation_pipeline(self, sample_ohlcv_data, integration_config):
        """Test complete features creation pipeline."""
        # Create features
        features_df = create_features(sample_ohlcv_data, integration_config)

        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        assert 'timestamp' in features_df.columns

        # Check that requested features are present
        expected_features = ['returns', 'ema_5', 'rsi_14']
        for feature in expected_features:
            assert any(feature in col for col in features_df.columns), f"Feature {feature} not found"

    def test_features_preparation_pipeline(self, sample_ohlcv_data, integration_config):
        """Test features preparation for modeling."""
        # Create and prepare features
        features_df = create_features(sample_ohlcv_data, integration_config)
        prepared_features = prepare_features(features_df, integration_config)

        assert isinstance(prepared_features, pd.DataFrame)
        assert len(prepared_features) > 0

        # Should have target column
        assert integration_config['target'] in prepared_features.columns

    def test_features_labels_integration(self, sample_ohlcv_data, integration_config):
        """Test features and labels creation together."""
        features_df = create_features(sample_ohlcv_data, integration_config)

        # Check that we have both features and labels
        feature_cols = [col for col in features_df.columns if col not in ['timestamp', 'y']]
        assert len(feature_cols) > 0
        assert 'y' in features_df.columns


class TestModelPipelineIntegration:
    """Test model training and evaluation pipeline."""

    def test_model_registry_integration(self, integration_config):
        """Test ModelRegistry integration."""
        registry = ModelRegistry(integration_config)

        # Check available models
        available = registry.list_available_models()
        assert isinstance(available, list)
        assert len(available) > 0

        # Test model retrieval
        for model_name in ['randomforest']:
            model = registry.get_model(model_name)
            assert model is not None

    def test_complete_model_training_pipeline(self, sample_ohlcv_data, integration_config):
        """Test complete model training pipeline."""
        # Prepare data
        features_df = create_features(sample_ohlcv_data, integration_config)
        prepared_features = prepare_features(features_df, integration_config)

        # Split data
        target_col = integration_config['target']
        feature_cols = [col for col in prepared_features.columns if col not in ['timestamp', target_col]]

        X = prepared_features[feature_cols]
        y = prepared_features[target_col]

        # Remove any NaN values
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) == 0:
            pytest.skip("No valid data after NaN removal")

        # Split train/validation
        split_idx = int(0.7 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train model
        registry = ModelRegistry(integration_config)
        model = registry.get_model('randomforest')

        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', integration_config
        )

        assert trained_model is not None
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics

    def test_model_save_load_pipeline(self, sample_ohlcv_data, integration_config, tmp_path):
        """Test model save/load in complete pipeline."""
        # Prepare data
        features_df = create_features(sample_ohlcv_data, integration_config)
        prepared_features = prepare_features(features_df, integration_config)

        target_col = integration_config['target']
        feature_cols = [col for col in prepared_features.columns if col not in ['timestamp', target_col]]

        X = prepared_features[feature_cols]
        y = prepared_features[target_col]

        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) == 0:
            pytest.skip("No valid data after NaN removal")

        split_idx = int(0.7 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train and save model
        registry = ModelRegistry(integration_config)
        model = registry.get_model('randomforest')

        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', integration_config
        )

        model_path = save_model(
            trained_model, 'randomforest', metrics, feature_cols, integration_config
        )

        assert Path(model_path).exists()

        # Load model
        loaded_model, metadata = load_model(model_path)

        assert loaded_model is not None
        assert isinstance(metadata, dict)
        assert 'metrics' in metadata


class TestBacktestPipelineIntegration:
    """Test backtesting pipeline integration."""

    def test_backtest_pipeline(self, sample_ohlcv_data, integration_config, tmp_path):
        """Test complete backtest pipeline."""
        # Prepare data and model
        features_df = create_features(sample_ohlcv_data, integration_config)
        prepared_features = prepare_features(features_df, integration_config)

        target_col = integration_config['target']
        feature_cols = [col for col in prepared_features.columns if col not in ['timestamp', target_col]]

        X = prepared_features[feature_cols]
        y = prepared_features[target_col]

        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        timestamps = prepared_features.loc[valid_idx, 'timestamp']

        if len(X) == 0:
            pytest.skip("No valid data after NaN removal")

        # Train model
        registry = ModelRegistry(integration_config)
        model = registry.get_model('randomforest')

        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        trained_model, _ = train_model(
            model, X_train, y_train, X_test, y_test, 'randomforest', integration_config
        )

        # Generate predictions for backtest
        predictions = trained_model.predict_proba(X)[:, 1]

        # Create backtest data
        backtest_data = prepared_features.loc[valid_idx].copy()
        backtest_data['prediction'] = predictions

        # Run backtest
        backtest_results = run_backtest(backtest_data, integration_config)

        assert isinstance(backtest_results, dict)
        assert 'equity_curve' in backtest_results
        assert 'trades' in backtest_results

    def test_backtest_metrics_integration(self, sample_ohlcv_data, integration_config):
        """Test backtest metrics calculation."""
        # Prepare data and model
        features_df = create_features(sample_ohlcv_data, integration_config)
        prepared_features = prepare_features(features_df, integration_config)

        target_col = integration_config['target']
        feature_cols = [col for col in prepared_features.columns if col not in ['timestamp', target_col]]

        X = prepared_features[feature_cols]
        y = prepared_features[target_col]

        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) == 0:
            pytest.skip("No valid data after NaN removal")

        # Train model
        registry = ModelRegistry(integration_config)
        model = registry.get_model('randomforest')

        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        trained_model, _ = train_model(
            model, X_train, y_train, X_test, y_test, 'randomforest', integration_config
        )

        predictions = trained_model.predict_proba(X)[:, 1]

        backtest_data = prepared_features.loc[valid_idx].copy()
        backtest_data['prediction'] = predictions

        backtest_results = run_backtest(backtest_data, integration_config)

        # Calculate metrics
        metrics = calculate_metrics(backtest_results, integration_config)

        assert isinstance(metrics, dict)
        assert 'sharpe_ratio' in metrics
        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics


class TestEndToEndPipeline:
    """Test complete end-to-end AXON pipeline."""

    @patch('src.dataset.load_raw_data')
    @patch('src.features.create_features')
    @patch('src.models.train_model')
    @patch('src.backtest.run_backtest')
    def test_full_pipeline_mock(self, mock_backtest, mock_train, mock_features, mock_data,
                               sample_ohlcv_data, integration_config):
        """Test full pipeline with mocked components."""
        # Setup mocks
        mock_data.return_value = sample_ohlcv_data
        mock_features.return_value = sample_ohlcv_data.assign(y=np.random.randint(0, 2, len(sample_ohlcv_data)))

        # Mock trained model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.random((len(sample_ohlcv_data), 2))
        mock_train.return_value = (mock_model, {'accuracy': 0.8})

        # Mock backtest results
        mock_backtest.return_value = {
            'equity_curve': pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, 100))),
            'trades': pd.DataFrame({
                'entry_time': pd.date_range('2023-01-01', periods=10),
                'exit_time': pd.date_range('2023-01-01', periods=10) + pd.Timedelta(minutes=5),
                'return_pct': np.random.normal(0, 0.02, 10)
            })
        }

        # This would be the main pipeline execution
        # For now, just verify the mocks are set up correctly
        assert mock_data.called
        assert mock_features.called
        assert mock_train.called
        assert mock_backtest.called

    def test_pipeline_error_handling(self, integration_config):
        """Test pipeline error handling."""
        # Test with invalid configuration
        invalid_config = integration_config.copy()
        invalid_config['models']['train'] = ['nonexistent_model']

        registry = ModelRegistry(invalid_config)

        # Should handle gracefully
        available = registry.list_available_models()
        assert isinstance(available, list)

        # Try to get invalid model
        with pytest.raises(ValueError):
            registry.get_model('nonexistent_model')


class TestCrossComponentIntegration:
    """Test integration between different AXON components."""

    def test_features_to_model_integration(self, sample_ohlcv_data, integration_config):
        """Test features output compatibility with models."""
        # Create features
        features_df = create_features(sample_ohlcv_data, integration_config)

        # Prepare for modeling
        prepared_features = prepare_features(features_df, integration_config)

        target_col = integration_config['target']
        feature_cols = [col for col in prepared_features.columns if col not in ['timestamp', target_col]]

        X = prepared_features[feature_cols]
        y = prepared_features[target_col]

        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) == 0:
            pytest.skip("No valid data")

        # Test with multiple model types
        registry = ModelRegistry(integration_config)

        for model_name in ['randomforest']:
            model = registry.get_model(model_name)
            trained_model, metrics = train_model(
                model, X, y, X, y, model_name, integration_config
            )

            assert trained_model is not None
            assert isinstance(metrics, dict)

    def test_model_to_backtest_integration(self, sample_ohlcv_data, integration_config):
        """Test model predictions compatibility with backtest."""
        # Prepare data
        features_df = create_features(sample_ohlcv_data, integration_config)
        prepared_features = prepare_features(features_df, integration_config)

        target_col = integration_config['target']
        feature_cols = [col for col in prepared_features.columns if col not in ['timestamp', target_col]]

        X = prepared_features[feature_cols]
        y = prepared_features[target_col]

        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        backtest_data = prepared_features[valid_idx]

        if len(backtest_data) == 0:
            pytest.skip("No valid data")

        # Generate mock predictions
        backtest_data = backtest_data.copy()
        backtest_data['prediction'] = np.random.random(len(backtest_data))

        # Run backtest
        results = run_backtest(backtest_data, integration_config)

        assert isinstance(results, dict)
        assert 'equity_curve' in results

    def test_config_consistency(self, integration_config):
        """Test configuration consistency across components."""
        # Test that config keys are recognized by different modules
        required_keys = ['data', 'features', 'labels', 'models', 'backtest']

        for key in required_keys:
            assert key in integration_config, f"Missing config key: {key}"

        # Test model registry with config
        registry = ModelRegistry(integration_config)
        assert registry.config == integration_config


class TestPipelineScalability:
    """Test pipeline scalability and performance."""

    def test_pipeline_with_different_data_sizes(self, integration_config):
        """Test pipeline with different data sizes."""
        for n_rows in [100, 500, 1000]:
            # Create data of different sizes
            np.random.seed(42)
            timestamps = pd.date_range('2023-01-01', periods=n_rows, freq='1min')
            prices = 50000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n_rows)))

            data = {
                'timestamp': timestamps,
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.randint(100, 1000, n_rows)
            }

            df = pd.DataFrame(data)

            # Test features creation scales
            features_df = create_features(df, integration_config)
            assert len(features_df) > 0

            # Test preparation scales
            prepared_features = prepare_features(features_df, integration_config)
            assert len(prepared_features) > 0

    def test_memory_usage_reasonable(self, sample_ohlcv_data, integration_config):
        """Test that pipeline doesn't use excessive memory."""
        # This is a basic test - in practice you'd use memory profiling
        features_df = create_features(sample_ohlcv_data, integration_config)
        prepared_features = prepare_features(features_df, integration_config)

        # Check that dataframes are reasonable size
        assert len(features_df) <= len(sample_ohlcv_data)
        assert len(prepared_features) <= len(features_df)


if __name__ == "__main__":
    pytest.main([__file__])