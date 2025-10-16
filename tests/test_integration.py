"""Integration tests for AXON system."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import main modules
import sys
sys.path.append('src')
from src.dataset import load_raw_data, create_time_based_splits, clean_and_validate_data
from src.features import create_feature_matrix, generate_labels, filter_features
from src.models import ModelRegistry, train_model
from src.backtest import BacktestEngine
from src.metrics import calculate_portfolio_metrics, calculate_trade_metrics


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    def test_dataset_to_features_pipeline(self, sample_ohlcv_data, mock_config):
        """Test dataset processing to features generation."""
        # Step 1: Clean and validate data
        cleaned_data = clean_and_validate_data(sample_ohlcv_data)

        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) > 0
        assert 'timestamp' in cleaned_data.columns

        # Step 2: Create time-based splits
        splits = create_time_based_splits(cleaned_data)

        assert 'train' in splits
        assert 'validation' in splits
        assert 'test' in splits
        assert len(splits['train']) > 0
        assert len(splits['test']) > 0

        # Step 3: Create features
        features_df = create_feature_matrix(splits['train'], mock_config)

        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        assert 'timestamp' in features_df.columns

        # Step 4: Generate labels
        labels = generate_labels(splits['train'], mock_config)

        assert isinstance(labels, pd.Series)
        assert len(labels) > 0

        # Step 5: Filter features
        filtered_df = filter_features(features_df, mock_config)

        assert isinstance(filtered_df, pd.DataFrame)
        assert len(filtered_df) > 0

    def test_features_to_model_pipeline(self, sample_features_data, mock_config):
        """Test features to model training pipeline."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        # Step 1: Initialize model registry
        registry = ModelRegistry(mock_config)

        # Step 2: Get model
        model = registry.get_model('randomforest')

        # Step 3: Train model
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', mock_config
        )

        assert trained_model is not None
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1' in metrics

    def test_model_to_backtest_pipeline(self, sample_features_data, mock_config):
        """Test model to backtest pipeline."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        # Step 1: Train model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')
        trained_model, _ = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', mock_config
        )

        # Step 2: Create test data for backtesting
        test_data = sample_features_data.copy()
        test_data = test_data.set_index('timestamp')

        # Step 3: Generate predictions
        predictions = trained_model.predict_proba(X_train)[:, 1]

        # Step 4: Run backtest
        engine = BacktestEngine(mock_config)
        results = engine.run_backtest(test_data, predictions)

        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'num_trades' in results
        assert 'win_rate' in results

    def test_backtest_to_metrics_pipeline(self, sample_features_data, mock_config):
        """Test backtest to metrics calculation pipeline."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]

        # Step 1: Train model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')
        trained_model, _ = train_model(
            model, X_train, y_train, X_train, y_train, 'randomforest', mock_config
        )

        # Step 2: Create test data
        test_data = sample_features_data.copy()
        test_data = test_data.set_index('timestamp')
        predictions = trained_model.predict_proba(X_train)[:, 1]

        # Step 3: Run backtest
        engine = BacktestEngine(mock_config)
        bt_results = engine.run_backtest(test_data, predictions)

        # Step 4: Calculate portfolio metrics
        if not engine.equity_curve.empty:
            portfolio_metrics = calculate_portfolio_metrics(engine.equity_curve['equity'])
            assert isinstance(portfolio_metrics, dict)
            assert 'sharpe_ratio' in portfolio_metrics

        # Step 5: Calculate trade metrics
        if engine.closed_positions:
            trades_df = pd.DataFrame([pos.to_dict() for pos in engine.closed_positions])
            if 'return_pct' in trades_df.columns:
                trade_metrics = calculate_trade_metrics(trades_df)
                assert isinstance(trade_metrics, dict)
                assert 'total_trades' in trade_metrics


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()

        # Dataset functions should handle empty data gracefully
        from src.dataset import create_time_based_splits
        with pytest.raises(Exception):  # Should raise appropriate error
            create_time_based_splits(empty_df)

    def test_single_row_data(self):
        """Test handling of single row data."""
        single_row = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01')],
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100],
            'volume': [1000]
        })

        # Should handle gracefully or raise appropriate error
        from src.dataset import create_time_based_splits
        with pytest.raises(Exception):  # Need minimum data for splits
            create_time_based_splits(single_row)

    def test_extreme_price_values(self):
        """Test handling of extreme price values."""
        extreme_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [1e-6, 1e6, 100, 100, 100, 100, 100, 100, 100, 100],  # Very small and large prices
            'high': [1e-6, 1e6, 101, 101, 101, 101, 101, 101, 101, 101],
            'low': [1e-6, 1e6, 99, 99, 99, 99, 99, 99, 99, 99],
            'close': [1e-6, 1e6, 100, 100, 100, 100, 100, 100, 100, 100],
            'volume': [1000] * 10
        })

        # Should handle extreme values without crashing
        from src.dataset import clean_and_validate_data
        result = clean_and_validate_data(extreme_data)

        assert isinstance(result, pd.DataFrame)

    def test_nan_inf_values(self):
        """Test handling of NaN and Inf values."""
        nan_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100, np.nan, 102, 103, np.inf, 105, 106, 107, 108, 109],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'low': [99, 98, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000] * 10
        })

        # Should handle NaN/Inf values
        from src.dataset import clean_and_validate_data
        result = clean_and_validate_data(nan_data)

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()  # No NaN values should remain

    def test_zero_volume_data(self):
        """Test handling of zero volume data."""
        zero_volume_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [0] * 10  # Zero volume
        })

        from src.dataset import clean_and_validate_data
        result = clean_and_validate_data(zero_volume_data)

        # Should remove rows with zero volume
        assert len(result) < len(zero_volume_data)

    def test_model_with_insufficient_features(self, mock_config):
        """Test model training with insufficient features."""
        # Very small dataset
        X_train = pd.DataFrame({'feature1': [1, 2, 3]})
        y_train = pd.Series([0, 1, 0])
        X_val = pd.DataFrame({'feature1': [4]})
        y_val = pd.Series([1])

        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')

        # Should handle gracefully
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', mock_config
        )

        assert trained_model is not None
        assert isinstance(metrics, dict)

    def test_backtest_with_no_signals(self):
        """Test backtest with no trading signals."""
        # Create data
        timestamps = pd.date_range('2023-01-01', periods=10, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        data.set_index('timestamp', inplace=True)

        # No signals (all probabilities below threshold)
        predictions = np.array([0.1] * 10)

        engine = BacktestEngine({'backtest': {'threshold': 0.5}})
        results = engine.run_backtest(data, predictions)

        assert results['num_trades'] == 0
        assert results['total_return'] == 0.0

    def test_metrics_with_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        # Create perfect prediction scenario
        trades_df = pd.DataFrame({
            'return_pct': [0.02, 0.03, 0.015, 0.025, 0.01]  # All positive
        })

        from src.metrics import calculate_trade_metrics
        metrics = calculate_trade_metrics(trades_df)

        assert metrics['hit_rate'] == 1.0
        assert metrics['total_trades'] == 5
        assert metrics['winning_trades'] == 5
        assert metrics['losing_trades'] == 0


class TestPerformanceValidation:
    """Test performance validation for critical functions."""

    def test_feature_creation_timing(self, sample_ohlcv_data, mock_config):
        """Validate feature creation completes within reasonable time."""
        import time

        start_time = time.time()
        result = create_feature_matrix(sample_ohlcv_data, mock_config)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within 5 seconds for reasonable dataset
        assert execution_time < 5.0
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_model_training_timing(self, sample_features_data, mock_config):
        """Validate model training completes within reasonable time."""
        import time

        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]
        X_val = X_train.copy()
        y_val = y_train.copy()

        start_time = time.time()
        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')
        trained_model, metrics = train_model(
            model, X_train, y_train, X_val, y_val, 'randomforest', mock_config
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within 10 seconds
        assert execution_time < 10.0
        assert trained_model is not None
        assert isinstance(metrics, dict)

    def test_backtest_timing(self, sample_features_data, mock_config):
        """Validate backtest completes within reasonable time."""
        import time

        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]

        # Train model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')
        trained_model, _ = train_model(
            model, X_train, y_train, X_train, y_train, 'randomforest', mock_config
        )

        # Prepare backtest data
        test_data = sample_features_data.copy()
        test_data = test_data.set_index('timestamp')
        predictions = trained_model.predict_proba(X_train)[:, 1]

        start_time = time.time()
        engine = BacktestEngine(mock_config)
        results = engine.run_backtest(test_data, predictions)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within 5 seconds
        assert execution_time < 5.0
        assert isinstance(results, dict)
        assert 'total_return' in results


class TestDataConsistency:
    """Test data consistency across pipeline."""

    def test_data_integrity_preservation(self, sample_ohlcv_data, mock_config):
        """Test that data integrity is preserved through pipeline."""
        original_length = len(sample_ohlcv_data)

        # Process through pipeline
        cleaned = clean_and_validate_data(sample_ohlcv_data)
        features = create_feature_matrix(cleaned, mock_config)
        filtered = filter_features(features, mock_config)

        # Check that essential columns are preserved
        assert 'timestamp' in filtered.columns
        assert 'open' in filtered.columns
        assert 'close' in filtered.columns

        # Check that timestamps are monotonic
        if len(filtered) > 1:
            assert filtered['timestamp'].is_monotonic_increasing

    def test_feature_label_alignment(self, sample_ohlcv_data, mock_config):
        """Test that features and labels are properly aligned."""
        # Create features and labels
        features = create_feature_matrix(sample_ohlcv_data, mock_config)
        labels = generate_labels(sample_ohlcv_data, mock_config)

        # They should have compatible lengths (allowing for NaN removal)
        assert len(features) >= len(labels.dropna())

        # Target column should be in features after generation
        target_col = mock_config.get('target', 'y')
        features_with_labels = features.copy()
        features_with_labels[target_col] = labels

        assert target_col in features_with_labels.columns

    def test_model_prediction_consistency(self, sample_features_data, mock_config):
        """Test that model predictions are consistent."""
        # Prepare data
        target_col = 'y'
        feature_cols = [col for col in sample_features_data.columns if col not in ['timestamp', target_col]]

        X_train = sample_features_data[feature_cols]
        y_train = sample_features_data[target_col]

        # Train model
        registry = ModelRegistry(mock_config)
        model = registry.get_model('randomforest')
        trained_model, _ = train_model(
            model, X_train, y_train, X_train, y_train, 'randomforest', mock_config
        )

        # Get predictions
        pred1 = trained_model.predict(X_train)
        pred2 = trained_model.predict(X_train)

        # Predictions should be identical for same input
        np.testing.assert_array_equal(pred1, pred2)

        # Predictions should be in valid range
        assert all(p in [0, 1] for p in pred1)