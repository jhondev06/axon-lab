"""Unit tests for AXON dataset module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
sys.path.append('src')
from src.dataset import (
    load_raw_data, create_time_based_splits, clean_and_validate_data,
    add_basic_features, generate_data_if_missing
)


class TestLoadRawData:
    """Test load_raw_data function."""

    def test_load_valid_csv(self, sample_ohlcv_data, temp_csv_file):
        """Test loading valid OHLCV CSV data."""
        csv_path = temp_csv_file(sample_ohlcv_data)

        result = load_raw_data(csv_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in result.columns for col in expected_cols)
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])

    def test_load_corrupted_data_raises_error(self, corrupted_ohlcv_data, temp_csv_file):
        """Test that corrupted OHLCV data raises assertion errors."""
        csv_path = temp_csv_file(corrupted_ohlcv_data)

        with pytest.raises(AssertionError):
            load_raw_data(csv_path)

    def test_load_extreme_price_data(self, extreme_price_data, temp_csv_file):
        """Test loading data with extreme price values."""
        csv_path = temp_csv_file(extreme_price_data)

        result = load_raw_data(csv_path)

        # Should still load successfully despite extreme values
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(extreme_price_data)

        # Check that extreme values are preserved
        assert result['close'].max() > 1e10

    def test_load_missing_file_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent_file.csv")


class TestCreateTimeBasedSplits:
    """Test create_time_based_splits function."""

    def test_default_splits(self, sample_ohlcv_data):
        """Test default time-based splits (60% train, 20% val, 20% test)."""
        result = create_time_based_splits(sample_ohlcv_data)

        assert isinstance(result, dict)
        assert 'train' in result
        assert 'validation' in result
        assert 'test' in result

        # Check approximate split sizes
        total_rows = len(sample_ohlcv_data)
        assert len(result['train']) == pytest.approx(total_rows * 0.6, abs=1)
        assert len(result['validation']) == pytest.approx(total_rows * 0.2, abs=1)
        assert len(result['test']) == pytest.approx(total_rows * 0.2, abs=1)

    def test_custom_split_ratios(self, sample_ohlcv_data):
        """Test custom split ratios."""
        result = create_time_based_splits(sample_ohlcv_data, train_ratio=0.7, test_ratio=0.15)

        total_rows = len(sample_ohlcv_data)
        assert len(result['train']) == pytest.approx(total_rows * 0.7, abs=1)
        assert len(result['validation']) == pytest.approx(total_rows * 0.15, abs=1)
        assert len(result['test']) == pytest.approx(total_rows * 0.15, abs=1)

    def test_temporal_leakage_prevention(self, sample_ohlcv_data):
        """Test that splits prevent temporal leakage."""
        result = create_time_based_splits(sample_ohlcv_data)

        train_max = result['train']['timestamp'].max()
        val_min = result['validation']['timestamp'].min()
        val_max = result['validation']['timestamp'].max()
        test_min = result['test']['timestamp'].min()

        # Train should end before validation starts
        assert train_max < val_min
        # Validation should end before test starts
        assert val_max < test_min

    def test_small_dataset_handling(self):
        """Test handling of very small datasets."""
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })

        result = create_time_based_splits(small_data)

        # Should still create splits even with small data
        assert isinstance(result, dict)
        assert len(result['train']) > 0
        assert len(result['validation']) >= 0
        assert len(result['test']) >= 0

    def test_unsorted_data_sorting(self):
        """Test that unsorted data is properly sorted."""
        # Create unsorted timestamps
        timestamps = pd.date_range('2023-01-01', periods=100, freq='1min')
        unsorted_timestamps = timestamps[::-1]  # Reverse order

        unsorted_data = pd.DataFrame({
            'timestamp': unsorted_timestamps,
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(100, 1000, 100)
        })

        result = create_time_based_splits(unsorted_data)

        # Should be sorted after processing
        assert result['train']['timestamp'].is_monotonic_increasing
        assert result['validation']['timestamp'].is_monotonic_increasing
        assert result['test']['timestamp'].is_monotonic_increasing


class TestCleanAndValidateData:
    """Test clean_and_validate_data function."""

    def test_clean_valid_data(self, sample_ohlcv_data):
        """Test cleaning of valid OHLCV data."""
        result = clean_and_validate_data(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_ohlcv_data)  # May remove some rows
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in result.columns for col in expected_cols)

    def test_remove_missing_values(self):
        """Test removal of rows with missing OHLCV values."""
        data_with_nans = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100, np.nan, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [101, 102, np.nan, 104, 105, 106, 107, 108, 109, 110],
            'low': [99, 98, 100, np.nan, 102, 103, 104, 105, 106, 107],
            'close': [100, 100, 102, 103, np.nan, 105, 106, 107, 108, 109],
            'volume': [1000, 1000, 1000, 1000, 1000, np.nan, 1000, 1000, 1000, 1000]
        })

        result = clean_and_validate_data(data_with_nans)

        # Should remove rows with NaN in critical columns
        assert len(result) < len(data_with_nans)
        assert not result[['open', 'high', 'low', 'close', 'volume']].isnull().any().any()

    def test_remove_zero_negative_prices(self):
        """Test removal of rows with zero or negative prices."""
        data_with_invalid_prices = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100, 0, -50, 103, 104, 105, 106, 107, 108, 109],
            'high': [101, 102, 100, 104, 105, 106, 107, 108, 109, 110],
            'low': [99, 98, 95, 102, 103, 104, 105, 106, 107, 108],
            'close': [100, 100, 98, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000] * 10
        })

        result = clean_and_validate_data(data_with_invalid_prices)

        # Should remove rows with zero/negative prices
        assert len(result) < len(data_with_invalid_prices)
        assert (result[['open', 'high', 'low', 'close']] > 0).all().all()

    def test_remove_extreme_price_changes(self):
        """Test removal of extreme price change outliers."""
        data_with_extremes = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'low': [99, 98, 97, 96, 95, 94, 93, 92, 91, 90],
            'close': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            'volume': [1000] * 10
        })

        # Make one extreme change (>50%)
        data_with_extremes.loc[5, 'close'] = 151  # 51% increase

        result = clean_and_validate_data(data_with_extremes)

        # Should remove the extreme change row
        assert len(result) < len(data_with_extremes)

    def test_forward_fill_nan_features(self):
        """Test forward filling of NaN values in derived features."""
        data_with_nan_features = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10,
            'returns': [np.nan, 0.01, 0.02, np.nan, 0.04, 0.05, np.nan, 0.07, 0.08, 0.09]
        })

        result = clean_and_validate_data(data_with_nan_features)

        # Should forward fill NaN values
        assert not result.isnull().any().any()


class TestAddBasicFeatures:
    """Test add_basic_features function."""

    def test_add_basic_features_structure(self, sample_ohlcv_data):
        """Test that basic features are added correctly."""
        result = add_basic_features(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)

        # Check that new features are present
        expected_features = ['returns', 'log_returns', 'price_range', 'body_size',
                           'volume_ma_5', 'volume_ratio', 'hour', 'day_of_week', 'is_weekend']
        for feature in expected_features:
            assert feature in result.columns

    def test_returns_calculation(self, sample_ohlcv_data):
        """Test returns calculation."""
        result = add_basic_features(sample_ohlcv_data)

        # Check returns calculation
        expected_returns = result['close'].pct_change()
        pd.testing.assert_series_equal(result['returns'], expected_returns, check_names=False)

    def test_log_returns_calculation(self, sample_ohlcv_data):
        """Test log returns calculation."""
        result = add_basic_features(sample_ohlcv_data)

        # Check log returns calculation
        expected_log_returns = np.log(result['close'] / result['close'].shift(1))
        pd.testing.assert_series_equal(result['log_returns'], expected_log_returns, check_names=False)

    def test_time_features(self, sample_ohlcv_data):
        """Test time-based features."""
        result = add_basic_features(sample_ohlcv_data)

        # Check hour extraction
        expected_hours = result['timestamp'].dt.hour
        pd.testing.assert_series_equal(result['hour'], expected_hours, check_names=False)

        # Check day of week
        expected_dow = result['timestamp'].dt.dayofweek
        pd.testing.assert_series_equal(result['day_of_week'], expected_dow, check_names=False)

        # Check weekend flag
        expected_weekend = (result['day_of_week'] >= 5).astype(int)
        pd.testing.assert_series_equal(result['is_weekend'], expected_weekend, check_names=False)


class TestGenerateDataIfMissing:
    """Test generate_data_if_missing function."""

    @patch('src.dataset.Path.exists')
    @patch('src.dataset.download_yahoo_finance_data')
    @patch('src.dataset.ensure_dir')
    def test_download_when_file_missing(self, mock_ensure_dir, mock_download, mock_exists):
        """Test downloading data when file is missing."""
        mock_exists.return_value = False
        mock_download.return_value = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })

        with patch('builtins.print'):  # Suppress prints
            result = generate_data_if_missing()

        mock_download.assert_called_once()
        mock_ensure_dir.assert_called()

    @patch('src.dataset.Path.exists')
    def test_return_existing_file_path(self, mock_exists):
        """Test returning path when file exists."""
        mock_exists.return_value = True

        with patch('builtins.print'):  # Suppress prints
            result = generate_data_if_missing()

        assert result == Path("data/raw/btc_real_1m.csv")


# Integration tests
class TestDatasetIntegration:
    """Integration tests for dataset pipeline."""

    def test_full_pipeline_workflow(self, sample_ohlcv_data, temp_csv_file, mock_config):
        """Test the full dataset processing pipeline."""
        # Save sample data to temporary file
        csv_path = temp_csv_file(sample_ohlcv_data)

        # Load and process data
        loaded_data = load_raw_data(csv_path)
        cleaned_data = clean_and_validate_data(loaded_data)
        featured_data = add_basic_features(cleaned_data)
        splits = create_time_based_splits(featured_data)

        # Verify pipeline results
        assert isinstance(splits, dict)
        assert all(key in splits for key in ['train', 'validation', 'test'])
        assert len(splits['train']) > 0
        assert len(splits['validation']) >= 0
        assert len(splits['test']) >= 0

        # Verify temporal ordering
        assert splits['train']['timestamp'].max() < splits['validation']['timestamp'].min()
        assert splits['validation']['timestamp'].max() < splits['test']['timestamp'].min()

    def test_extreme_values_handling(self, extreme_price_data, temp_csv_file):
        """Test handling of extreme values throughout pipeline."""
        csv_path = temp_csv_file(extreme_price_data)

        # Process through pipeline
        loaded_data = load_raw_data(csv_path)
        cleaned_data = clean_and_validate_data(loaded_data)
        featured_data = add_basic_features(cleaned_data)

        # Should handle extreme values without crashing
        assert isinstance(featured_data, pd.DataFrame)
        assert len(featured_data) > 0

        # Check that extreme values are still present (not filtered out)
        assert featured_data['close'].max() > 1e10