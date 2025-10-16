"""Unit tests for AXON metrics module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import functions to test
import sys
sys.path.append('src')
from src.metrics import (
    calculate_returns, calculate_log_returns, sharpe_ratio, sortino_ratio,
    maximum_drawdown, profit_factor, hit_rate, calmar_ratio,
    value_at_risk, conditional_value_at_risk, calculate_trade_metrics,
    calculate_portfolio_metrics, save_metrics_report
)


class TestBasicCalculations:
    """Test basic calculation functions."""

    def test_calculate_returns_basic(self):
        """Test basic returns calculation."""
        prices = np.array([100, 101, 102, 103])

        result = calculate_returns(prices)

        # Check basic properties
        assert len(result) == 3
        assert result[0] == 0.01  # (101-100)/100 = 0.01
        assert result[1] > 0  # Should be positive
        assert result[2] > 0  # Should be positive
        assert result[1] < result[0]  # Decreasing returns due to higher base

    def test_calculate_returns_pandas(self):
        """Test returns calculation with pandas Series."""
        prices = pd.Series([100, 101, 102, 103])

        result = calculate_returns(prices)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_calculate_returns_empty(self):
        """Test returns calculation with empty array."""
        prices = np.array([])

        result = calculate_returns(prices)

        assert len(result) == 0

    def test_calculate_log_returns(self):
        """Test log returns calculation."""
        prices = np.array([100, 101, 102])

        result = calculate_log_returns(prices)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        assert result[0] > 0  # Log return should be positive for price increase

    def test_calculate_log_returns_pandas(self):
        """Test log returns calculation with pandas Series."""
        prices = pd.Series([100, 101, 102])

        result = calculate_log_returns(prices)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2


class TestRiskMetrics:
    """Test risk and performance metrics."""

    def test_sharpe_ratio_basic(self):
        """Test basic Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])

        result = sharpe_ratio(returns)

        assert isinstance(result, float)
        assert result > 0  # Should be positive for this return series

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = np.array([0.01, 0.01, 0.01])

        result = sharpe_ratio(returns)

        assert result == 0.0

    def test_sharpe_ratio_empty(self):
        """Test Sharpe ratio with empty array."""
        returns = np.array([])

        result = sharpe_ratio(returns)

        assert result == 0.0

    def test_sharpe_ratio_pandas(self):
        """Test Sharpe ratio with pandas Series."""
        returns = pd.Series([0.01, 0.02, -0.01])

        result = sharpe_ratio(returns)

        assert isinstance(result, float)

    def test_sortino_ratio_basic(self):
        """Test basic Sortino ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])

        result = sortino_ratio(returns)

        assert isinstance(result, float)

    def test_sortino_ratio_no_downside(self):
        """Test Sortino ratio with no downside returns."""
        returns = np.array([0.01, 0.02, 0.015])

        result = sortino_ratio(returns)

        assert result == float('inf')

    def test_maximum_drawdown_basic(self):
        """Test basic maximum drawdown calculation."""
        equity = np.array([100, 105, 102, 108, 95, 110])

        result = maximum_drawdown(equity)

        assert isinstance(result, dict)
        assert 'max_drawdown' in result
        assert 'max_drawdown_pct' in result
        assert 'drawdown_duration' in result
        assert result['max_drawdown'] < 0  # Drawdown should be negative
        assert result['max_drawdown_pct'] < 0

    def test_maximum_drawdown_pandas(self):
        """Test maximum drawdown with pandas Series."""
        equity = pd.Series([100, 105, 102, 108, 95, 110])

        result = maximum_drawdown(equity)

        assert isinstance(result, dict)
        assert 'max_drawdown' in result

    def test_maximum_drawdown_increasing(self):
        """Test maximum drawdown with only increasing equity."""
        equity = np.array([100, 101, 102, 103])

        result = maximum_drawdown(equity)

        assert result['max_drawdown'] == 0.0
        assert result['max_drawdown_pct'] == 0.0

    def test_profit_factor_basic(self):
        """Test basic profit factor calculation."""
        returns = np.array([0.02, -0.01, 0.015, -0.005, 0.01])

        result = profit_factor(returns)

        assert isinstance(result, float)
        assert result > 0

    def test_profit_factor_only_losses(self):
        """Test profit factor with only losses."""
        returns = np.array([-0.01, -0.02, -0.005])

        result = profit_factor(returns)

        assert result == 0.0

    def test_profit_factor_only_profits(self):
        """Test profit factor with only profits."""
        returns = np.array([0.01, 0.02, 0.015])

        result = profit_factor(returns)

        assert result == float('inf')

    def test_hit_rate_basic(self):
        """Test basic hit rate calculation."""
        returns = np.array([0.02, -0.01, 0.015, -0.005])

        result = hit_rate(returns)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_hit_rate_all_wins(self):
        """Test hit rate with all winning trades."""
        returns = np.array([0.02, 0.01, 0.015])

        result = hit_rate(returns)

        assert result == 1.0

    def test_hit_rate_all_losses(self):
        """Test hit rate with all losing trades."""
        returns = np.array([-0.02, -0.01, -0.015])

        result = hit_rate(returns)

        assert result == 0.0


class TestAdvancedMetrics:
    """Test advanced financial metrics."""

    def test_calmar_ratio_basic(self):
        """Test basic Calmar ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015])
        equity = np.array([100, 101, 103, 102, 103.5])

        result = calmar_ratio(returns, equity)

        assert isinstance(result, float)

    def test_value_at_risk_basic(self):
        """Test basic Value at Risk calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005, 0.01, -0.02])

        result = value_at_risk(returns, confidence_level=0.05)

        assert isinstance(result, float)
        assert result < 0  # VaR should be negative

    def test_value_at_risk_pandas(self):
        """Test Value at Risk with pandas Series."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015])

        result = value_at_risk(returns)

        assert isinstance(result, float)

    def test_conditional_value_at_risk(self):
        """Test Conditional Value at Risk calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005, 0.01, -0.02, -0.03])

        result = conditional_value_at_risk(returns, confidence_level=0.05)

        assert isinstance(result, float)
        assert result < 0  # CVaR should be negative


class TestTradeMetrics:
    """Test trade-level metrics."""

    def test_calculate_trade_metrics_basic(self):
        """Test basic trade metrics calculation."""
        trades_df = pd.DataFrame({
            'return_pct': [0.02, -0.01, 0.015, -0.005, 0.01]
        })

        result = calculate_trade_metrics(trades_df)

        assert isinstance(result, dict)
        assert 'total_trades' in result
        assert 'winning_trades' in result
        assert 'losing_trades' in result
        assert 'hit_rate' in result
        assert 'avg_win' in result
        assert 'avg_loss' in result
        assert result['total_trades'] == 5
        assert result['winning_trades'] == 3
        assert result['losing_trades'] == 2

    def test_calculate_trade_metrics_empty(self):
        """Test trade metrics with empty DataFrame."""
        trades_df = pd.DataFrame()

        result = calculate_trade_metrics(trades_df)

        assert isinstance(result, dict)
        assert result['total_trades'] == 0

    def test_calculate_trade_metrics_no_returns(self):
        """Test trade metrics without return_pct column."""
        trades_df = pd.DataFrame({'other_col': [1, 2, 3]})

        result = calculate_trade_metrics(trades_df)

        assert isinstance(result, dict)
        assert result['total_trades'] == 0

    def test_calculate_trade_metrics_all_wins(self):
        """Test trade metrics with all winning trades."""
        trades_df = pd.DataFrame({
            'return_pct': [0.02, 0.01, 0.015]
        })

        result = calculate_trade_metrics(trades_df)

        assert result['total_trades'] == 3
        assert result['winning_trades'] == 3
        assert result['losing_trades'] == 0
        assert result['hit_rate'] == 1.0

    def test_calculate_trade_metrics_all_losses(self):
        """Test trade metrics with all losing trades."""
        trades_df = pd.DataFrame({
            'return_pct': [-0.02, -0.01, -0.015]
        })

        result = calculate_trade_metrics(trades_df)

        assert result['total_trades'] == 3
        assert result['winning_trades'] == 0
        assert result['losing_trades'] == 3
        assert result['hit_rate'] == 0.0


class TestPortfolioMetrics:
    """Test portfolio-level metrics."""

    def test_calculate_portfolio_metrics_basic(self):
        """Test basic portfolio metrics calculation."""
        equity = pd.Series([100, 101, 103, 102, 104, 106])

        result = calculate_portfolio_metrics(equity)

        assert isinstance(result, dict)
        assert 'total_return' in result
        assert 'annualized_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result

    def test_calculate_portfolio_metrics_empty(self):
        """Test portfolio metrics with empty series."""
        equity = pd.Series([])

        result = calculate_portfolio_metrics(equity)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_calculate_portfolio_metrics_with_benchmark(self):
        """Test portfolio metrics with benchmark."""
        equity = pd.Series([100, 101, 103, 102, 104])
        benchmark = pd.Series([100, 100.5, 102, 101.5, 103])

        result = calculate_portfolio_metrics(equity, benchmark)

        assert isinstance(result, dict)
        assert 'beta' in result
        assert 'alpha' in result
        assert 'correlation' in result


class TestMetricsReporting:
    """Test metrics reporting functions."""

    def test_save_metrics_report(self, tmp_path):
        """Test saving metrics report."""
        metrics = {
            'total_return': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.03
        }

        filepath = save_metrics_report(metrics, 'test_metrics.json')

        assert Path(filepath).exists()

        # Verify content by reading JSON directly
        import json
        with open(filepath, 'r') as f:
            loaded = json.load(f)

        assert loaded['total_return'] == 0.05
        assert loaded['sharpe_ratio'] == 1.2
        assert loaded['max_drawdown'] == -0.03


class TestMetricsIntegration:
    """Integration tests for metrics module."""

    def test_complete_metrics_workflow(self):
        """Test complete metrics calculation workflow."""
        # Create sample data
        equity_curve = pd.Series([10000, 10200, 10100, 10300, 10200, 10400])
        trades_df = pd.DataFrame({
            'return_pct': [0.02, -0.01, 0.03, -0.005, 0.015]
        })

        # Calculate individual metrics
        returns = calculate_returns(equity_curve)
        sharpe = sharpe_ratio(returns)
        drawdown = maximum_drawdown(equity_curve)
        trade_metrics = calculate_trade_metrics(trades_df)
        portfolio_metrics = calculate_portfolio_metrics(equity_curve)

        # Verify all calculations work together
        assert isinstance(returns, np.ndarray)
        assert isinstance(sharpe, float)
        assert isinstance(drawdown, dict)
        assert isinstance(trade_metrics, dict)
        assert isinstance(portfolio_metrics, dict)

        # Check reasonable values
        assert len(returns) == len(equity_curve) - 1
        assert 'max_drawdown' in drawdown
        assert 'total_trades' in trade_metrics
        assert 'sharpe_ratio' in portfolio_metrics

    def test_metrics_with_extreme_values(self):
        """Test metrics calculation with extreme values."""
        # Create data with extreme values
        equity_curve = pd.Series([10000, 20000, 5000, 15000, 1000, 25000])

        returns = calculate_returns(equity_curve)
        sharpe = sharpe_ratio(returns)
        drawdown = maximum_drawdown(equity_curve)

        # Should handle extreme values without crashing
        assert isinstance(returns, np.ndarray)
        assert isinstance(sharpe, float)
        assert isinstance(drawdown, dict)

        # Drawdown should be significant due to extreme drop
        assert drawdown['max_drawdown_pct'] < -0.5  # At least 50% drawdown

    def test_metrics_edge_cases(self):
        """Test metrics with edge cases."""
        # Flat equity curve
        flat_equity = pd.Series([10000] * 10)
        flat_returns = calculate_returns(flat_equity)
        flat_sharpe = sharpe_ratio(flat_returns)

        assert len(flat_returns) == 9
        assert all(r == 0.0 for r in flat_returns)
        assert flat_sharpe == 0.0

        # Single value equity
        single_equity = pd.Series([10000])
        single_returns = calculate_returns(single_equity)

        assert len(single_returns) == 0

        # Very volatile returns
        volatile_returns = np.array([0.5, -0.4, 0.3, -0.6, 0.2])
        vol_sharpe = sharpe_ratio(volatile_returns)

        assert isinstance(vol_sharpe, float)
        # High volatility should result in lower Sharpe ratio
        assert vol_sharpe < 2.0