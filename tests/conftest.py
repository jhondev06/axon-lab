"""Shared fixtures and configuration for AXON tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible tests

    # Create realistic BTC-like data
    n_rows = 1000
    base_price = 50000.0
    timestamps = pd.date_range('2023-01-01', periods=n_rows, freq='1min')

    # Generate realistic price movements
    returns = np.random.normal(0.0001, 0.01, n_rows)  # Small mean return, 1% volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    high_mult = 1 + np.abs(np.random.normal(0, 0.005, n_rows))
    low_mult = 1 - np.abs(np.random.normal(0, 0.005, n_rows))

    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices * (1 + np.random.normal(0, 0.002, n_rows)),
        'volume': np.random.randint(100, 10000, n_rows)
    }

    df = pd.DataFrame(data)

    # Ensure OHLCV integrity
    df['high'] = np.maximum(df[['open', 'high', 'close']].max(axis=1), df['high'])
    df['low'] = np.minimum(df[['open', 'low', 'close']].min(axis=1), df['low'])

    return df


@pytest.fixture
def corrupted_ohlcv_data():
    """Create corrupted OHLCV data for testing error handling."""
    np.random.seed(123)

    n_rows = 100
    timestamps = pd.date_range('2023-01-01', periods=n_rows, freq='1min')

    data = {
        'timestamp': timestamps,
        'open': np.random.uniform(100, 1000, n_rows),
        'high': np.random.uniform(50, 500, n_rows),  # Intentionally lower than open
        'low': np.random.uniform(200, 2000, n_rows),  # Intentionally higher than open
        'close': np.random.uniform(100, 1000, n_rows),
        'volume': np.random.uniform(-100, 100, n_rows)  # Negative volumes
    }

    return pd.DataFrame(data)


@pytest.fixture
def extreme_price_data():
    """Create data with extreme price values for testing."""
    np.random.seed(456)

    n_rows = 100
    timestamps = pd.date_range('2023-01-01', periods=n_rows, freq='1min')

    # Mix of normal and extreme prices
    normal_prices = np.random.uniform(100, 1000, n_rows//2)
    extreme_prices = np.random.uniform(1e10, 1e15, n_rows//2)
    base_prices = np.concatenate([normal_prices, extreme_prices])

    data = {
        'timestamp': timestamps,
        'open': base_prices,
        'high': base_prices * 1.01,  # Ensure high > open
        'low': base_prices * 0.99,   # Ensure low < open
        'close': base_prices * (1 + np.random.normal(0, 0.005, n_rows)),  # Small variation
        'volume': np.random.randint(100, 10000, n_rows)
    }

    df = pd.DataFrame(data)

    # Ensure OHLCV integrity by adjusting values
    # Make sure high is the maximum of open, high, close
    df['high'] = np.maximum.reduce([df['open'], df['high'], df['close']])
    # Make sure low is the minimum of open, low, close
    df['low'] = np.minimum.reduce([df['open'], df['low'], df['close']])

    return df


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    def _create_temp_csv(dataframe):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            dataframe.to_csv(f.name, index=False)
            return f.name

    yield _create_temp_csv

    # Cleanup
    for file in Path(tempfile.gettempdir()).glob("*.csv"):
        if file.name.startswith("tmp"):
            try:
                file.unlink()
            except:
                pass


@pytest.fixture
def mock_config():
    """Mock configuration dictionary for testing."""
    return {
        'data': {
            'sources': ['synthetic'],
            'symbols': ['BTC-USD'],
            'lookback_days': 30,
            'interval': '1m',
            'quality': {
                'min_rows_test': 100
            },
            'cache': {
                'dir': 'data/cache'
            }
        },
        'features': {
            'use': ['ret_1m', 'ema_5', 'rsi_14']
        },
        'labels': {
            'method': 'triple_barrier',
            'horizon': '15m'
        },
        'models': {
            'train': ['lightgbm', 'randomforest']
        },
        'backtest': {
            'threshold': 0.6,
            'position_size': 1.0,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_positions': 1,
            'commission': 0.0,
            'initial_capital': 10000.0,
            'timeout': '60m'
        },
        'target': 'y'
    }


@pytest.fixture
def sample_features_data(sample_ohlcv_data):
    """Create sample features data for testing."""
    df = sample_ohlcv_data.copy()

    # Add basic features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']

    # Add technical features
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['rsi_14'] = calculate_rsi(df['close'], 14)

    # Add target (simple sign return)
    df['y'] = (df['close'].shift(-15) / df['close'] - 1 > 0).astype(int)

    return df.dropna()


def calculate_rsi(series, window=14):
    """Calculate RSI for testing."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@pytest.fixture
def sample_predictions(sample_ohlcv_data):
    """Create sample predictions for backtesting."""
    np.random.seed(42)
    n_samples = len(sample_ohlcv_data)
    # Create realistic predictions around 0.5
    predictions = 0.5 + np.random.normal(0, 0.2, n_samples)
    predictions = np.clip(predictions, 0, 1)  # Ensure valid probabilities
    return predictions


@pytest.fixture
def sample_trades_data():
    """Create sample trades data for metrics testing."""
    np.random.seed(42)

    n_trades = 100
    timestamps = pd.date_range('2023-01-01', periods=n_trades, freq='1H')

    # Generate realistic trade returns
    returns = np.random.normal(0.001, 0.02, n_trades)  # 0.1% mean, 2% std

    data = {
        'entry_timestamp': timestamps,
        'exit_timestamp': timestamps + pd.Timedelta(minutes=15),
        'return_pct': returns,
        'side': np.random.choice(['long', 'short'], n_trades),
        'entry_price': np.random.uniform(40000, 60000, n_trades),
        'exit_price': np.random.uniform(40000, 60000, n_trades),
        'size': np.random.uniform(0.1, 1.0, n_trades),
        'pnl': returns * np.random.uniform(1000, 10000, n_trades)
    }

    return pd.DataFrame(data)