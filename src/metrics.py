"""AXON Metrics Module

Financial and ML performance metrics.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .utils import load_config, ensure_dir
from axon.core.logging import get_logger

# Initialize logger for debugging
logger = get_logger(__name__)

def calculate_returns(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Calculate simple returns from price series."""
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices) < 2:
        return np.array([])
    
    returns = np.diff(prices) / prices[:-1]
    return returns

def calculate_log_returns(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Calculate log returns from price series."""
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices) < 2:
        return np.array([])
    
    log_returns = np.diff(np.log(prices))
    return log_returns

def sharpe_ratio(returns: Union[pd.Series, np.ndarray],
                risk_free_rate: float = 0.0,
                periods_per_year: int = 252 * 24 * 60) -> float:
    """Calculate Sharpe ratio.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year (default for 1-minute data)

    Returns:
        Sharpe ratio
    """
    logger.info(f"DEBUG: Calculating Sharpe ratio with {len(returns)} returns, rf={risk_free_rate}, periods={periods_per_year}")

    if isinstance(returns, pd.Series):
        returns = returns.values

    if len(returns) == 0:
        logger.warning("DEBUG: No returns provided for Sharpe calculation")
        return 0.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    logger.info(f"DEBUG: After NaN removal: {len(returns)} returns")

    if len(returns) == 0 or np.std(returns) == 0:
        logger.warning("DEBUG: Insufficient data or zero variance for Sharpe calculation")
        return 0.0

    # Convert risk-free rate to per-period
    rf_per_period = risk_free_rate / periods_per_year

    # Calculate excess returns
    excess_returns = returns - rf_per_period

    # Calculate Sharpe ratio
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns)
    sharpe = mean_excess / std_returns * np.sqrt(periods_per_year)

    logger.info(f"DEBUG: Sharpe components - Mean excess: {mean_excess:.6f}, Std returns: {std_returns:.6f}, Sharpe: {sharpe:.4f}")

    return float(sharpe)

def sortino_ratio(returns: Union[pd.Series, np.ndarray], 
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = 252 * 24 * 60) -> float:
    """Calculate Sortino ratio (downside deviation)."""
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) == 0:
        return 0.0
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Convert risk-free rate to per-period
    rf_per_period = risk_free_rate / periods_per_year
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate downside deviation
    downside_returns = returns[returns < rf_per_period]
    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(periods_per_year)
    
    return float(sortino)

def maximum_drawdown(equity_curve: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """Calculate maximum drawdown and related metrics.

    Args:
        equity_curve: Equity curve values

    Returns:
        Dictionary with max_drawdown, max_drawdown_pct, drawdown_duration
    """
    logger.info(f"DEBUG: Calculating maximum drawdown for equity curve with {len(equity_curve)} points")

    if isinstance(equity_curve, pd.Series):
        equity_values = equity_curve.values
        timestamps = equity_curve.index if hasattr(equity_curve, 'index') else None
    else:
        equity_values = np.array(equity_curve)
        timestamps = None

    if len(equity_values) == 0:
        logger.warning("DEBUG: Empty equity curve provided")
        return {
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'drawdown_duration': 0,
            'peak_value': 0.0,
            'trough_value': 0.0
        }

    logger.info(f"DEBUG: Equity range: [{equity_values.min():.2f}, {equity_values.max():.2f}]")

    # Calculate running maximum (peak)
    running_max = np.maximum.accumulate(equity_values)

    # Calculate drawdown
    drawdown = equity_values - running_max
    drawdown_pct = drawdown / running_max

    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_drawdown = drawdown[max_dd_idx]
    max_drawdown_pct = drawdown_pct[max_dd_idx]

    # Find peak and trough values
    peak_value = running_max[max_dd_idx]
    trough_value = equity_values[max_dd_idx]

    # Calculate drawdown duration (simplified)
    # Find the peak before the maximum drawdown
    peak_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if equity_values[i] == running_max[i]:
            peak_idx = i
            break

    drawdown_duration = max_dd_idx - peak_idx

    logger.info(f"DEBUG: Max drawdown - Absolute: {max_drawdown:.2f}, Percentage: {max_drawdown_pct:.2%}")
    logger.info(f"DEBUG: Peak: {peak_value:.2f}, Trough: {trough_value:.2f}, Duration: {drawdown_duration} periods")

    return {
        'max_drawdown': float(max_drawdown),
        'max_drawdown_pct': float(max_drawdown_pct),
        'drawdown_duration': int(drawdown_duration),
        'peak_value': float(peak_value),
        'trough_value': float(trough_value)
    }

def profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate profit factor (gross profit / gross loss).
    
    Args:
        returns: Return series
    
    Returns:
        Profit factor
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) == 0:
        return 0.0
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate gross profit and gross loss
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    pf = gross_profit / gross_loss
    return float(pf)

def hit_rate(returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate hit rate (win rate).
    
    Args:
        returns: Return series
    
    Returns:
        Hit rate (percentage of winning trades)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) == 0:
        return 0.0
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate hit rate
    winning_trades = np.sum(returns > 0)
    total_trades = len(returns)
    
    hit_rate_pct = winning_trades / total_trades
    return float(hit_rate_pct)

def calmar_ratio(returns: Union[pd.Series, np.ndarray], 
                equity_curve: Union[pd.Series, np.ndarray],
                periods_per_year: int = 252 * 24 * 60) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)."""
    if len(returns) == 0 or len(equity_curve) == 0:
        return 0.0
    
    # Calculate annualized return
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0
    
    annualized_return = np.mean(returns) * periods_per_year
    
    # Calculate maximum drawdown
    dd_metrics = maximum_drawdown(equity_curve)
    max_dd_pct = abs(dd_metrics['max_drawdown_pct'])
    
    if max_dd_pct == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    calmar = annualized_return / max_dd_pct
    return float(calmar)

def value_at_risk(returns: Union[pd.Series, np.ndarray], 
                 confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk (VaR).
    
    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
    
    Returns:
        VaR value
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) == 0:
        return 0.0
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate VaR
    var = np.percentile(returns, confidence_level * 100)
    return float(var)

def conditional_value_at_risk(returns: Union[pd.Series, np.ndarray], 
                             confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)."""
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) == 0:
        return 0.0
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate VaR threshold
    var_threshold = np.percentile(returns, confidence_level * 100)
    
    # Calculate CVaR (mean of returns below VaR)
    tail_returns = returns[returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return var_threshold
    
    cvar = np.mean(tail_returns)
    return float(cvar)

def calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive trade-level metrics.
    
    Args:
        trades_df: DataFrame with trade data (must have 'return_pct' column)
    
    Returns:
        Dictionary of trade metrics
    """
    if trades_df.empty or 'return_pct' not in trades_df.columns:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'hit_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 0.0,
            'avg_trade': 0.0,
            'win_loss_ratio': 0.0
        }
    
    returns = trades_df['return_pct'].values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'hit_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 0.0,
            'avg_trade': 0.0,
            'win_loss_ratio': 0.0
        }
    
    # Basic counts
    total_trades = len(returns)
    winning_trades = np.sum(returns > 0)
    losing_trades = np.sum(returns < 0)
    
    # Win/loss analysis
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    largest_win = np.max(wins) if len(wins) > 0 else 0.0
    largest_loss = np.min(losses) if len(losses) > 0 else 0.0
    
    # Ratios
    hit_rate_val = hit_rate(returns)
    profit_factor_val = profit_factor(returns)
    avg_trade = np.mean(returns)
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0.0
    
    return {
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'hit_rate': float(hit_rate_val),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'largest_win': float(largest_win),
        'largest_loss': float(largest_loss),
        'profit_factor': float(profit_factor_val),
        'avg_trade': float(avg_trade),
        'win_loss_ratio': float(win_loss_ratio)
    }

def calculate_portfolio_metrics(equity_curve: pd.Series, 
                              benchmark_returns: Optional[pd.Series] = None,
                              risk_free_rate: float = 0.0) -> Dict[str, Any]:
    """Calculate comprehensive portfolio-level metrics.
    
    Args:
        equity_curve: Portfolio equity curve
        benchmark_returns: Benchmark return series (optional)
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Dictionary of portfolio metrics
    """
    if equity_curve.empty:
        return {}
    
    # Calculate returns
    returns = calculate_returns(equity_curve)
    
    if len(returns) == 0:
        return {}
    
    # Basic metrics
    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    annualized_return = np.mean(returns) * (252 * 24 * 60)  # 1-minute data
    volatility = np.std(returns) * np.sqrt(252 * 24 * 60)
    
    # Risk metrics
    sharpe = sharpe_ratio(returns, risk_free_rate)
    sortino = sortino_ratio(returns, risk_free_rate)
    dd_metrics = maximum_drawdown(equity_curve)
    calmar = calmar_ratio(returns, equity_curve)
    
    # VaR metrics
    var_95 = value_at_risk(returns, 0.05)
    cvar_95 = conditional_value_at_risk(returns, 0.05)
    
    metrics = {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'max_drawdown': dd_metrics['max_drawdown'],
        'max_drawdown_pct': dd_metrics['max_drawdown_pct'],
        'drawdown_duration': dd_metrics['drawdown_duration'],
        'var_95': float(var_95),
        'cvar_95': float(cvar_95),
        'skewness': float(pd.Series(returns).skew()),
        'kurtosis': float(pd.Series(returns).kurtosis()),
    }
    
    # Benchmark comparison (if provided)
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align returns
        aligned_returns = pd.Series(returns, index=equity_curve.index[1:])
        common_index = aligned_returns.index.intersection(benchmark_returns.index)
        
        if len(common_index) > 1:
            port_aligned = aligned_returns.loc[common_index]
            bench_aligned = benchmark_returns.loc[common_index]
            
            # Calculate beta and alpha
            covariance = np.cov(port_aligned, bench_aligned)[0, 1]
            benchmark_variance = np.var(bench_aligned)
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
                alpha = np.mean(port_aligned) - beta * np.mean(bench_aligned)
                
                metrics['beta'] = float(beta)
                metrics['alpha'] = float(alpha)
                metrics['correlation'] = float(np.corrcoef(port_aligned, bench_aligned)[0, 1])
    
    return metrics

def save_metrics_report(metrics: Dict[str, Any], filename: str) -> str:
    """Save metrics to JSON file."""
    ensure_dir("outputs/metrics")
    
    filepath = f"outputs/metrics/{filename}"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    metrics_clean = convert_numpy_types(metrics)
    
    with open(filepath, 'w') as f:
        json.dump(metrics_clean, f, indent=2, default=str)
    
    print(f"Metrics report saved: {filepath}")
    return filepath

def analyze_backtest_results(results_dir: str = "outputs/ledgers") -> Dict[str, Any]:
    """Analyze all backtest results and generate comprehensive metrics."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return {}
    
    # Find all result files
    result_files = list(results_path.glob("*_results.json"))
    trade_files = list(results_path.glob("*_trades.csv"))
    equity_files = list(results_path.glob("*_equity.csv"))
    
    if not result_files:
        print("No backtest results found")
        return {}
    
    print(f"Analyzing {len(result_files)} backtest results...")
    
    all_metrics = {}
    
    for result_file in result_files:
        try:
            # Extract model name from filename
            base_name = result_file.stem.replace('_results', '')
            model_name = base_name.split('_')[1] if len(base_name.split('_')) > 1 else 'unknown'
            
            # Load basic results
            with open(result_file, 'r') as f:
                basic_results = json.load(f)
            
            metrics = {
                'basic_metrics': basic_results,
                'trade_metrics': {},
                'portfolio_metrics': {}
            }
            
            # Load and analyze trades
            trade_file = results_path / f"{base_name}_trades.csv"
            if trade_file.exists():
                trades_df = pd.read_csv(trade_file)
                trade_metrics = calculate_trade_metrics(trades_df)
                metrics['trade_metrics'] = trade_metrics
            
            # Load and analyze equity curve
            equity_file = results_path / f"{base_name}_equity.csv"
            if equity_file.exists():
                equity_df = pd.read_csv(equity_file)
                if 'equity' in equity_df.columns:
                    equity_series = pd.Series(equity_df['equity'].values)
                    portfolio_metrics = calculate_portfolio_metrics(equity_series)
                    metrics['portfolio_metrics'] = portfolio_metrics
            
            all_metrics[model_name] = metrics
            
        except Exception as e:
            print(f"❌ Failed to analyze {result_file.name}: {e}")
            continue
    
    # Save comprehensive analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"comprehensive_analysis_{timestamp}.json"
    save_metrics_report(all_metrics, analysis_file)
    
    return all_metrics

def main():
    """Main metrics analysis pipeline."""
    print("=== AXON Metrics Module ===")
    
    # Load configuration
    config = load_config()
    
    try:
        # Analyze all backtest results
        analysis = analyze_backtest_results()
        
        if not analysis:
            print("❌ No backtest results found to analyze")
            print("Run backtesting first to generate results")
            return
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE METRICS ANALYSIS")
        print(f"{'='*60}")
        
        # Display summary for each model
        for model_name, metrics in analysis.items():
            print(f"\n📊 {model_name.upper()}:")
            
            # Basic metrics
            basic = metrics.get('basic_metrics', {})
            if basic:
                print(f"  Total Return: {basic.get('total_return', 0):.2%}")
                print(f"  Sharpe Ratio: {basic.get('sharpe_ratio', 0):.3f}")
                print(f"  Max Drawdown: {basic.get('max_drawdown', 0):.2%}")
            
            # Trade metrics
            trade = metrics.get('trade_metrics', {})
            if trade:
                print(f"  Total Trades: {trade.get('total_trades', 0)}")
                print(f"  Hit Rate: {trade.get('hit_rate', 0):.2%}")
                print(f"  Profit Factor: {trade.get('profit_factor', 0):.2f}")
            
            # Portfolio metrics
            portfolio = metrics.get('portfolio_metrics', {})
            if portfolio:
                print(f"  Volatility: {portfolio.get('volatility', 0):.2%}")
                print(f"  Sortino Ratio: {portfolio.get('sortino_ratio', 0):.3f}")
                print(f"  Calmar Ratio: {portfolio.get('calmar_ratio', 0):.3f}")
        
        # Find best performing model
        if len(analysis) > 1:
            best_model = None
            best_sharpe = float('-inf')
            
            for model_name, metrics in analysis.items():
                sharpe = metrics.get('basic_metrics', {}).get('sharpe_ratio', 0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_model = model_name
            
            if best_model:
                print(f"\n🏆 Best performing model: {best_model} (Sharpe: {best_sharpe:.3f})")
        
        print(f"\n✅ Metrics analysis completed successfully!")
        print(f"Detailed results saved in outputs/metrics/")
        
    except Exception as e:
        print(f"❌ Metrics analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()