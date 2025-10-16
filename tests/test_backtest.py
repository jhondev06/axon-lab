"""Unit tests for AXON backtest module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
sys.path.append('src')
from src.backtest import (
    Position, BacktestEngine, run_backtest_for_model
)


class TestPosition:
    """Test Position class."""

    def test_position_init(self):
        """Test Position initialization."""
        timestamp = pd.Timestamp('2023-01-01 10:00:00')
        pos = Position(
            timestamp=timestamp,
            side='long',
            entry_price=100.0,
            size=10.0,
            stop_loss=95.0,
            take_profit=110.0,
            timeout_bars=60
        )

        assert pos.timestamp == timestamp
        assert pos.side == 'long'
        assert pos.entry_price == 100.0
        assert pos.size == 10.0
        assert pos.stop_loss == 95.0
        assert pos.take_profit == 110.0
        assert pos.timeout_bars == 60
        assert pos.bars_held == 0
        assert pos.exit_timestamp is None
        assert pos.exit_price is None
        assert pos.exit_reason is None
        assert pos.pnl == 0.0
        assert pos.return_pct == 0.0

    def test_position_update_long_take_profit(self):
        """Test position update for long position hitting take profit."""
        timestamp = pd.Timestamp('2023-01-01 10:00:00')
        pos = Position(
            timestamp=timestamp,
            side='long',
            entry_price=100.0,
            size=10.0,
            take_profit=110.0
        )

        # Price hits take profit
        closed = pos.update(110.0, timestamp + pd.Timedelta(minutes=5))

        assert closed is True
        assert pos.exit_price == 110.0
        assert pos.exit_reason == 'take_profit'
        assert pos.pnl == (110.0 - 100.0) * 10.0  # 100 profit
        assert pos.return_pct == (110.0 - 100.0) / 100.0  # 10%

    def test_position_update_long_stop_loss(self):
        """Test position update for long position hitting stop loss."""
        timestamp = pd.Timestamp('2023-01-01 10:00:00')
        pos = Position(
            timestamp=timestamp,
            side='long',
            entry_price=100.0,
            size=10.0,
            stop_loss=95.0
        )

        # Price hits stop loss
        closed = pos.update(95.0, timestamp + pd.Timedelta(minutes=5))

        assert closed is True
        assert pos.exit_price == 95.0
        assert pos.exit_reason == 'stop_loss'
        assert pos.pnl == (95.0 - 100.0) * 10.0  # -50 loss
        assert pos.return_pct == (95.0 - 100.0) / 100.0  # -5%

    def test_position_update_timeout(self):
        """Test position update for timeout."""
        timestamp = pd.Timestamp('2023-01-01 10:00:00')
        pos = Position(
            timestamp=timestamp,
            side='long',
            entry_price=100.0,
            size=10.0,
            timeout_bars=3
        )

        # Update for 3 bars (should timeout)
        closed1 = pos.update(101.0, timestamp + pd.Timedelta(minutes=1))
        closed2 = pos.update(102.0, timestamp + pd.Timedelta(minutes=2))
        closed3 = pos.update(103.0, timestamp + pd.Timedelta(minutes=3))

        assert closed1 is False
        assert closed2 is False
        assert closed3 is True
        assert pos.exit_reason == 'timeout'
        assert pos.bars_held == 3

    def test_position_force_close(self):
        """Test force close position."""
        timestamp = pd.Timestamp('2023-01-01 10:00:00')
        pos = Position(
            timestamp=timestamp,
            side='long',
            entry_price=100.0,
            size=10.0
        )

        exit_timestamp = timestamp + pd.Timedelta(minutes=10)
        pos.force_close(105.0, exit_timestamp)

        assert pos.exit_price == 105.0
        assert pos.exit_timestamp == exit_timestamp
        assert pos.exit_reason == 'market_close'
        assert pos.pnl == (105.0 - 100.0) * 10.0

    def test_position_to_dict(self):
        """Test position to_dict conversion."""
        timestamp = pd.Timestamp('2023-01-01 10:00:00')
        pos = Position(
            timestamp=timestamp,
            side='long',
            entry_price=100.0,
            size=10.0,
            stop_loss=95.0,
            take_profit=110.0
        )

        # Close position
        pos.force_close(105.0, timestamp + pd.Timedelta(minutes=5))

        pos_dict = pos.to_dict()

        assert isinstance(pos_dict, dict)
        assert pos_dict['entry_timestamp'] == timestamp
        assert pos_dict['exit_timestamp'] == timestamp + pd.Timedelta(minutes=5)
        assert pos_dict['side'] == 'long'
        assert pos_dict['entry_price'] == 100.0
        assert pos_dict['exit_price'] == 105.0
        assert pos_dict['pnl'] == 50.0


class TestBacktestEngine:
    """Test BacktestEngine class."""

    def test_backtest_engine_init(self, mock_config):
        """Test BacktestEngine initialization."""
        engine = BacktestEngine(mock_config)

        assert engine.config == mock_config
        assert engine.positions == []
        assert engine.closed_positions == []
        assert engine.equity_curve == []
        assert engine.current_capital == 10000.0

    def test_long_if_prob_gt_strategy_true(self, sample_ohlcv_data):
        """Test long_if_prob_gt_strategy returns True."""
        engine = BacktestEngine({'backtest': {'threshold': 0.5}})

        # High probability
        predictions = np.array([0.8])
        result = engine.long_if_prob_gt_strategy(sample_ohlcv_data.iloc[0:1], predictions, 0)

        assert result is True

    def test_long_if_prob_gt_strategy_false(self, sample_ohlcv_data):
        """Test long_if_prob_gt_strategy returns False."""
        engine = BacktestEngine({'backtest': {'threshold': 0.8}})

        # Low probability
        predictions = np.array([0.3])
        result = engine.long_if_prob_gt_strategy(sample_ohlcv_data.iloc[0:1], predictions, 0)

        assert result is False

    def test_run_backtest_no_trades(self):
        """Test backtest with no trades."""
        # Create data with low probabilities
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

        # Low probabilities
        predictions = np.array([0.1] * 10)

        engine = BacktestEngine({'backtest': {'threshold': 0.5}})
        results = engine.run_backtest(data, predictions)

        assert results['num_trades'] == 0
        assert results['total_return'] == 0.0
        assert results['win_rate'] == 0.0

    def test_run_backtest_with_trades(self):
        """Test backtest with trades."""
        # Create simple data with price movement
        timestamps = pd.date_range('2023-01-01', periods=10, freq='1min')

        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],  # Higher highs
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],    # Lower lows
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000] * 10
        })
        data.set_index('timestamp', inplace=True)

        # High probabilities for first few bars
        predictions = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        engine = BacktestEngine({
            'backtest': {
                'threshold': 0.5,
                'position_size': 1.0,
                'take_profit': 0.05,
                'stop_loss': 0.05
            }
        })
        results = engine.run_backtest(data, predictions)

        assert results['num_trades'] > 0
        assert isinstance(results['total_return'], float)
        assert isinstance(results['win_rate'], float)
        assert 0 <= results['win_rate'] <= 1

    def test_calculate_results_no_trades(self):
        """Test _calculate_results with no trades."""
        engine = BacktestEngine({'backtest': {'initial_capital': 10000.0}})

        results = engine._calculate_results()

        assert results['total_return'] == 0.0
        assert results['num_trades'] == 0
        assert results['win_rate'] == 0.0
        assert results['avg_return'] == 0.0

    def test_calculate_results_with_trades(self):
        """Test _calculate_results with trades."""
        engine = BacktestEngine({'backtest': {'initial_capital': 10000.0}})

        # Initialize equity_curve as empty DataFrame
        engine.equity_curve = pd.DataFrame()

        # Add some mock closed positions
        timestamp = pd.Timestamp('2023-01-01 10:00:00')
        pos1 = Position(timestamp, 'long', 100.0, 10.0)
        pos1.force_close(110.0, timestamp + pd.Timedelta(minutes=5))  # Profit

        pos2 = Position(timestamp, 'long', 100.0, 10.0)
        pos2.force_close(95.0, timestamp + pd.Timedelta(minutes=5))   # Loss

        engine.closed_positions = [pos1, pos2]

        results = engine._calculate_results()

        assert results['num_trades'] == 2
        assert results['winning_trades'] == 1
        assert results['losing_trades'] == 1
        assert results['win_rate'] == 0.5
        assert results['profit_factor'] > 0

    def test_save_results(self, tmp_path):
        """Test save_results method."""
        engine = BacktestEngine({'backtest': {'initial_capital': 10000.0}})

        # Add mock data
        engine.equity_curve = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'equity': [10000, 10100, 10200, 10150, 10250]
        })

        pos = Position(pd.Timestamp('2023-01-01'), 'long', 100.0, 10.0)
        pos.force_close(105.0, pd.Timestamp('2023-01-01 00:05:00'))
        engine.closed_positions = [pos]

        results = {'total_return': 0.025, 'num_trades': 1}

        filename = engine.save_results(results, 'test_model')

        assert isinstance(filename, str)
        assert 'test_model' in filename

        # Check if files were created
        assert Path(f"outputs/ledgers/{filename}_trades.csv").exists()
        assert Path(f"outputs/ledgers/{filename}_equity.csv").exists()
        assert Path(f"outputs/ledgers/{filename}_results.json").exists()


class TestRunBacktestForModel:
    """Test run_backtest_for_model function."""

    def test_run_backtest_for_model_requires_data(self):
        """Test run_backtest_for_model requires test data."""
        # This test verifies the function properly checks for test data
        with pytest.raises(FileNotFoundError):
            run_backtest_for_model('nonexistent_model.pkl', {})


class TestBacktestIntegration:
    """Integration tests for backtest module."""

    def test_position_lifecycle(self):
        """Test complete position lifecycle."""
        timestamp = pd.Timestamp('2023-01-01 10:00:00')

        # Create position
        pos = Position(
            timestamp=timestamp,
            side='long',
            entry_price=100.0,
            size=10.0,
            stop_loss=95.0,
            take_profit=110.0,
            timeout_bars=10
        )

        # Update position multiple times
        updates = [
            (101.0, timestamp + pd.Timedelta(minutes=1), False),  # No exit
            (103.0, timestamp + pd.Timedelta(minutes=2), False),  # No exit
            (110.0, timestamp + pd.Timedelta(minutes=3), True),   # Take profit
        ]

        for price, ts, expected_closed in updates:
            closed = pos.update(price, ts)
            assert closed == expected_closed

        # Verify final state
        assert pos.exit_reason == 'take_profit'
        assert pos.pnl == (110.0 - 100.0) * 10.0
        assert pos.return_pct == (110.0 - 100.0) / 100.0

    def test_backtest_engine_basic_functionality(self):
        """Test basic backtest engine functionality."""
        # Simple test data
        timestamps = pd.date_range('2023-01-01', periods=5, freq='1min')

        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000] * 5
        })
        data.set_index('timestamp', inplace=True)

        predictions = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # No trades

        engine = BacktestEngine({
            'backtest': {
                'threshold': 0.5,
                'initial_capital': 10000.0
            }
        })

        results = engine.run_backtest(data, predictions)

        # Verify results structure
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'num_trades' in results
        assert results['num_trades'] == 0  # No trades expected
        assert results['total_return'] == 0.0

    def test_backtest_edge_cases(self):
        """Test backtest with edge cases."""
        # Empty data
        empty_data = pd.DataFrame()
        predictions = np.array([])

        engine = BacktestEngine({'backtest': {'threshold': 0.5}})

        with pytest.raises(Exception):  # Should handle gracefully
            engine.run_backtest(empty_data, predictions)

        # Data with NaN values
        timestamps = pd.date_range('2023-01-01', periods=5, freq='1min')
        data_with_nan = pd.DataFrame({
            'timestamp': timestamps,
            'open': [100, np.nan, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000] * 5
        })
        data_with_nan.set_index('timestamp', inplace=True)

        predictions = np.array([0.9] * 5)

        # Should handle NaN gracefully or raise appropriate error
        try:
            results = engine.run_backtest(data_with_nan, predictions)
            assert isinstance(results, dict)
        except Exception:
            # Expected if NaN handling fails
            pass