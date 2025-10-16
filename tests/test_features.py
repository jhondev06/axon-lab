"""Unit tests for AXON features module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import functions to test
import sys
sys.path.append('src')
from src.features import (
    calculate_ema, calculate_rsi, generate_triple_barrier_labels,
    create_feature_matrix, generate_labels, filter_features
)


class TestCalculateEMA:
    """Test calculate_ema function."""

    def test_calculate_ema_basic(self):
        """Test basic EMA calculation."""
        # Create simple price series
        prices = pd.Series([100, 101, 102, 103, 104, 105])

        result = calculate_ema(prices, span=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        assert not result.isna().any()  # EMA should fill all values

    def test_calculate_ema_values(self):
        """Test EMA calculation produces expected values."""
        prices = pd.Series([100.0, 102.0, 104.0])

        result = calculate_ema(prices, span=2)

        # Manual calculation for verification
        # EMA2 = (P0 * 2/3) + (P1 * 1/3) for second value
        # But pandas uses different smoothing factor
        assert result.iloc[0] == 100.0  # First value unchanged
        assert result.iloc[1] > 100.0   # Should be weighted average
        assert result.iloc[2] > result.iloc[1]  # Should follow trend

    def test_calculate_ema_different_spans(self):
        """Test EMA with different span values."""
        prices = pd.Series([100, 101, 102, 103, 104])

        ema_short = calculate_ema(prices, span=2)
        ema_long = calculate_ema(prices, span=5)

        # Shorter span should be more responsive to recent changes
        assert ema_short.iloc[-1] != ema_long.iloc[-1]

    def test_calculate_ema_with_nan(self):
        """Test EMA handles NaN values properly."""
        prices = pd.Series([100, np.nan, 102, 103])

        result = calculate_ema(prices, span=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        # Should handle NaN gracefully

    def test_calculate_ema_empty_series(self):
        """Test EMA with empty series."""
        prices = pd.Series([])

        result = calculate_ema(prices, span=3)

        assert len(result) == 0  # Should return empty series
        assert isinstance(result, pd.Series)


class TestCalculateRSI:
    """Test calculate_rsi function."""

    def test_calculate_rsi_basic(self):
        """Test basic RSI calculation."""
        # Create price series with clear up/down movements
        prices = pd.Series([100, 102, 98, 103, 97, 105, 95, 107, 93, 109])

        result = calculate_rsi(prices, window=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        assert result.min() >= 0
        assert result.max() <= 100

    def test_calculate_rsi_extremes(self):
        """Test RSI at extreme values (oversold/overbought)."""
        # Strongly upward trending prices
        uptrend_prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        rsi_up = calculate_rsi(uptrend_prices, window=3)

        # Should approach 100 for strong uptrend
        assert rsi_up.iloc[-1] > 80

        # Strongly downward trending prices
        downtrend_prices = pd.Series([100, 99, 98, 97, 96, 95, 94, 93, 92, 91])

        rsi_down = calculate_rsi(downtrend_prices, window=3)

        # Should approach 0 for strong downtrend
        assert rsi_down.iloc[-1] < 20

    def test_calculate_rsi_flat_prices(self):
        """Test RSI with flat prices (no movement)."""
        flat_prices = pd.Series([100] * 10)

        result = calculate_rsi(flat_prices, window=3)

        # RSI should be NaN for flat prices (no gains/losses)
        assert pd.isna(result.iloc[-1])

    def test_calculate_rsi_different_windows(self):
        """Test RSI with different window sizes."""
        prices = pd.Series([100, 102, 98, 103, 97, 105, 95, 107, 93, 109])

        rsi_short = calculate_rsi(prices, window=2)
        rsi_long = calculate_rsi(prices, window=5)

        # Different windows should give different results
        assert not rsi_short.equals(rsi_long)

    def test_calculate_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = pd.Series([100, 102])  # Less than window size

        result = calculate_rsi(prices, window=5)

        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)


class TestGenerateTripleBarrierLabels:
    """Test generate_triple_barrier_labels function."""

    def test_generate_labels_basic(self, sample_ohlcv_data):
        """Test basic triple barrier label generation."""
        result = generate_triple_barrier_labels(sample_ohlcv_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.dtype in [int, float]  # Can be int or float

        # Check valid label values (-1, 0, 1)
        unique_values = result.dropna().unique()
        for val in unique_values:
            assert val in [-1, 0, 1]

    def test_generate_labels_with_hits(self):
        """Test label generation with guaranteed barrier hits."""
        # Create simple data that will definitely hit barriers
        timestamps = pd.date_range('2023-01-01', periods=10, freq='1min')

        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': [100, 105, 95, 110, 90, 115, 85, 120, 80, 125],
            'high': [105, 110, 100, 115, 95, 120, 90, 125, 85, 130],  # High enough to hit profit target
            'low': [95, 90, 85, 95, 75, 100, 70, 105, 65, 110],     # Low enough to hit stop loss
            'close': [102, 98, 107, 93, 112, 88, 117, 83, 122, 78],
            'volume': [1000] * 10
        })

        result = generate_triple_barrier_labels(data, horizon_minutes=3, pt_pct=0.05, sl_pct=0.05)

        # Should have some non-zero labels
        assert (result != 0).any()

    def test_generate_labels_timeout_only(self):
        """Test label generation that results in only timeouts."""
        # Create flat prices that won't hit barriers
        timestamps = pd.date_range('2023-01-01', periods=50, freq='1min')

        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': [100] * 50,
            'high': [100.5] * 50,  # Small movements
            'low': [99.5] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })

        result = generate_triple_barrier_labels(data, horizon_minutes=10, pt_pct=0.1, sl_pct=0.1)

        # Should be all timeouts (0) due to small price movements
        assert (result == 0).all()

    def test_generate_labels_different_parameters(self, sample_ohlcv_data):
        """Test label generation with different parameters."""
        # Test with different profit targets and stop losses
        result_tight = generate_triple_barrier_labels(
            sample_ohlcv_data, horizon_minutes=5, pt_pct=0.01, sl_pct=0.01
        )
        result_wide = generate_triple_barrier_labels(
            sample_ohlcv_data, horizon_minutes=5, pt_pct=0.05, sl_pct=0.05
        )

        # Different parameters should give different results
        assert not result_tight.equals(result_wide)

    def test_generate_labels_edge_cases(self):
        """Test label generation with edge cases."""
        # Very small dataset
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000] * 5
        })

        result = generate_triple_barrier_labels(small_data, horizon_minutes=2)

        assert isinstance(result, pd.Series)
        assert len(result) == len(small_data)


class TestCreateFeatureMatrix:
    """Test create_feature_matrix function."""

    def test_create_feature_matrix_basic(self, sample_ohlcv_data, mock_config):
        """Test basic feature matrix creation."""
        result = create_feature_matrix(sample_ohlcv_data, mock_config)

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= len(sample_ohlcv_data)  # May have NaN rows removed
        assert 'timestamp' in result.columns
        assert 'close' in result.columns

    def test_create_feature_matrix_returns(self, sample_ohlcv_data, mock_config):
        """Test returns features are created."""
        result = create_feature_matrix(sample_ohlcv_data, mock_config)

        returns_features = [col for col in result.columns if 'ret' in col]
        assert len(returns_features) > 0

        # Check returns calculation
        if 'ret_1m' in result.columns:
            expected_returns = result['close'].pct_change()
            pd.testing.assert_series_equal(
                result['ret_1m'], expected_returns, check_names=False
            )

    def test_create_feature_matrix_ema(self, sample_ohlcv_data, mock_config):
        """Test EMA features are created."""
        result = create_feature_matrix(sample_ohlcv_data, mock_config)

        ema_features = [col for col in result.columns if 'ema' in col]
        assert len(ema_features) > 0

        # Check EMA values are reasonable
        if 'ema_5' in result.columns:
            assert result['ema_5'].notna().any()

    def test_create_feature_matrix_rsi(self, sample_ohlcv_data, mock_config):
        """Test RSI features are created."""
        result = create_feature_matrix(sample_ohlcv_data, mock_config)

        rsi_features = [col for col in result.columns if 'rsi' in col]
        assert len(rsi_features) > 0

        # Check RSI values are in valid range (excluding NaN)
        if 'rsi_14' in result.columns:
            rsi_values = result['rsi_14'].dropna()
            if len(rsi_values) > 0:
                assert (rsi_values >= 0).all()
                assert (rsi_values <= 100).all()

    def test_create_feature_matrix_with_config(self, sample_ohlcv_data):
        """Test feature matrix respects configuration."""
        config = {
            'features': {
                'use': ['ret_1m', 'ema_5']  # Only specific features
            }
        }

        result = create_feature_matrix(sample_ohlcv_data, config)

        # Should contain basic columns
        assert 'timestamp' in result.columns
        assert 'open' in result.columns
        assert 'close' in result.columns

        # Should contain at least some features
        feature_cols = [col for col in result.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        assert len(feature_cols) > 0


class TestGenerateLabels:
    """Test generate_labels function."""

    def test_generate_labels_triple_barrier(self, sample_ohlcv_data, mock_config):
        """Test triple barrier label generation."""
        result = generate_labels(sample_ohlcv_data, mock_config)

        assert isinstance(result, pd.Series)
        assert len(result) <= len(sample_ohlcv_data)  # May have NaN rows removed

        # Check valid label values (triple barrier returns binary after processing)
        if len(result.dropna()) > 0:
            unique_values = result.dropna().unique()
            for val in unique_values:
                assert val in [0, 1]  # Binary labels for triple barrier

    def test_generate_labels_sign_return(self, sample_ohlcv_data):
        """Test sign return label generation."""
        config = {
            'labels': {
                'method': 'sign_return_h',
                'horizon': '15m'
            }
        }

        result = generate_labels(sample_ohlcv_data, config)

        assert isinstance(result, pd.Series)
        assert len(result) <= len(sample_ohlcv_data)

        # Check valid label values (may include NaN for last horizon_minutes)
        if len(result.dropna()) > 0:
            unique_values = result.dropna().unique()
            for val in unique_values:
                assert val in [0, 1]  # Binary labels

    def test_generate_labels_invalid_method(self, sample_ohlcv_data):
        """Test invalid label method raises error."""
        config = {
            'labels': {
                'method': 'invalid_method'
            }
        }

        with pytest.raises(ValueError):
            generate_labels(sample_ohlcv_data, config)


class TestFilterFeatures:
    """Test filter_features function."""

    def test_filter_features_basic(self, sample_features_data, mock_config):
        """Test basic feature filtering."""
        result = filter_features(sample_features_data, mock_config)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_filter_features_with_config(self, sample_features_data):
        """Test feature filtering respects configuration."""
        config = {
            'features': {
                'use': ['ret_1m', 'ema_5']  # Only specific features
            }
        }

        result = filter_features(sample_features_data, config)

        # Should contain required columns
        assert 'timestamp' in result.columns
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns

        # Should contain at least some features (may not have ret_1m if not in original data)
        feature_cols = [col for col in result.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        assert len(feature_cols) > 0


# Integration tests
class TestFeaturesIntegration:
    """Integration tests for features pipeline."""

    def test_full_features_pipeline(self, sample_ohlcv_data, mock_config):
        """Test the complete features pipeline."""
        # Create features
        features_df = create_feature_matrix(sample_ohlcv_data, mock_config)

        # Generate labels
        labels = generate_labels(sample_ohlcv_data, mock_config)

        # Filter features
        filtered_df = filter_features(features_df, mock_config)

        assert isinstance(filtered_df, pd.DataFrame)
        assert isinstance(labels, pd.Series)
        assert len(filtered_df) > 0
        assert len(labels) > 0

    def test_features_with_extreme_values(self, extreme_price_data, mock_config):
        """Test features pipeline with extreme values."""
        # Should handle extreme values without crashing
        features_df = create_feature_matrix(extreme_price_data, mock_config)

        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0

        # Should still have valid feature values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0