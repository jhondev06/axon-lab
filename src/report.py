"""AXON Report Module

Markdown report generation and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from .utils import load_config, ensure_dir
from .metrics import (
    sharpe_ratio, sortino_ratio, maximum_drawdown, profit_factor,
    hit_rate, calmar_ratio, value_at_risk, conditional_value_at_risk
)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_backtest_results(ledgers_path: str = "outputs/ledgers") -> Dict[str, Dict[str, Any]]:
    """
    Load all backtest results from the ledgers directory.
    
    Args:
        ledgers_path: Path to the ledgers directory
    
    Returns:
        Dictionary with model names as keys and results as values
    """
    results = {}
    ledgers_dir = Path(ledgers_path)
    
    if not ledgers_dir.exists():
        print(f"[WARNING] Ledgers directory not found: {ledgers_path}")
        return results
    
    # Find all result files
    for result_file in ledgers_dir.glob("*_results.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract model name from filename
            model_name = result_file.stem.replace('_results', '')
            
            # Load corresponding equity curve
            equity_file = result_file.parent / f"{model_name}_equity.csv"
            if equity_file.exists():
                equity_df = pd.read_csv(equity_file)
                data['equity_curve'] = equity_df
            
            results[model_name] = data
            
        except Exception as e:
            print(f"[WARNING] Failed to load {result_file}: {e}")
    
    return results

def calculate_performance_metrics(equity_curve: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics from equity curve.
    
    Args:
        equity_curve: DataFrame with 'equity' column
    
    Returns:
        Dictionary of performance metrics
    """
    if equity_curve.empty or 'equity' not in equity_curve.columns:
        return {}
    
    equity = equity_curve['equity'].values
    returns = np.diff(equity) / equity[:-1]
    returns = returns[~np.isnan(returns)]  # Remove NaN values
    
    if len(returns) == 0:
        return {}
    
    # Calculate drawdown metrics
    dd_metrics = maximum_drawdown(equity)
    
    metrics = {
        'total_return': (equity[-1] / equity[0] - 1) * 100,
        'sharpe_ratio': sharpe_ratio(returns),
        'sortino_ratio': sortino_ratio(returns),
        'max_drawdown': abs(dd_metrics.get('max_drawdown_pct', 0)) * 100,
        'calmar_ratio': calmar_ratio(returns, equity),
        'var_95': value_at_risk(returns, confidence_level=0.05) * 100,
        'cvar_95': conditional_value_at_risk(returns, confidence_level=0.05) * 100,
        'volatility': np.std(returns) * np.sqrt(252) * 100,  # Annualized
        'skewness': float(pd.Series(returns).skew()) if len(returns) > 2 else 0.0,
        'kurtosis': float(pd.Series(returns).kurtosis()) if len(returns) > 3 else 0.0,
    }
    
    return metrics

def create_equity_curve_plot(results: Dict[str, Dict[str, Any]], 
                           save_path: str = "outputs/figures/equity_curves.png") -> str:
    """
    Create equity curve visualization for all models.
    
    Args:
        results: Dictionary of backtest results
        save_path: Path to save the plot
    
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for i, (model_name, data) in enumerate(results.items()):
        if 'equity_curve' in data and not data['equity_curve'].empty:
            equity_df = data['equity_curve']
            
            # Convert timestamp if it exists
            if 'timestamp' in equity_df.columns:
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                x_axis = equity_df['timestamp']
                plt.xlabel('Time')
            else:
                x_axis = range(len(equity_df))
                plt.xlabel('Bars')
            
            plt.plot(x_axis, equity_df['equity'], 
                    label=f"{model_name.replace('backtest_', '')}", 
                    color=colors[i], linewidth=2)
    
    plt.title('Equity Curves Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Ensure directory exists
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_performance_comparison_plot(metrics_df: pd.DataFrame, 
                                     save_path: str = "outputs/figures/performance_comparison.png") -> str:
    """
    Create performance metrics comparison visualization.
    
    Args:
        metrics_df: DataFrame with models as index and metrics as columns
        save_path: Path to save the plot
    
    Returns:
        Path to the saved plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
    
    # Key metrics to plot
    key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
    titles = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Volatility (%)']
    
    for i, (metric, title) in enumerate(zip(key_metrics, titles)):
        ax = axes[i // 2, i % 2]
        
        if metric in metrics_df.columns:
            data = metrics_df[metric].dropna()
            if not data.empty:
                bars = ax.bar(range(len(data)), data.values, 
                            color=plt.cm.Set2(np.linspace(0, 1, len(data))))
                ax.set_title(title, fontweight='bold')
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels([name.replace('backtest_', '') for name in data.index], 
                                 rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, data.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    
    # Ensure directory exists
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def generate_markdown_report(results: Dict[str, Dict[str, Any]], 
                           metrics_df: pd.DataFrame,
                           config: Dict[str, Any],
                           save_path: str = "outputs/reports/performance_report.md") -> str:
    """
    Generate comprehensive Markdown performance report.
    
    Args:
        results: Dictionary of backtest results
        metrics_df: DataFrame with performance metrics
        config: Configuration dictionary
        save_path: Path to save the report
    
    Returns:
        Path to the saved report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = [
        f"# {config.get('project', 'AXON')} Performance Report",
        "",
        f"*Generated: {timestamp}*",
        "",
        "## Executive Summary",
        "",
        f"This report analyzes the performance of {len(results)} trading strategies "
        f"across the backtesting period. The analysis includes risk-adjusted returns, "
        f"drawdown analysis, and comparative performance metrics.",
        "",
    ]
    
    # Best performing model
    if not metrics_df.empty and 'sharpe_ratio' in metrics_df.columns:
        best_model = metrics_df['sharpe_ratio'].idxmax()
        best_sharpe = metrics_df.loc[best_model, 'sharpe_ratio']
        report_lines.extend([
            f"**[BEST] Best Performing Strategy**: {best_model.replace('backtest_', '')} "
            f"(Sharpe Ratio: {best_sharpe:.3f})",
            "",
        ])
    
    # Performance Summary Table
    report_lines.extend([
        "## Performance Summary",
        "",
        "| Model | Total Return (%) | Sharpe Ratio | Max Drawdown (%) | Volatility (%) |",
        "|-------|------------------|--------------|------------------|----------------|",
    ])
    
    for model_name in metrics_df.index:
        clean_name = model_name.replace('backtest_', '')
        total_ret = metrics_df.loc[model_name, 'total_return'] if 'total_return' in metrics_df.columns else 0
        sharpe = metrics_df.loc[model_name, 'sharpe_ratio'] if 'sharpe_ratio' in metrics_df.columns else 0
        max_dd = metrics_df.loc[model_name, 'max_drawdown'] if 'max_drawdown' in metrics_df.columns else 0
        vol = metrics_df.loc[model_name, 'volatility'] if 'volatility' in metrics_df.columns else 0
        
        report_lines.append(
            f"| {clean_name} | {total_ret:.2f} | {sharpe:.3f} | {max_dd:.2f} | {vol:.2f} |"
        )
    
    report_lines.extend([
        "",
        "## Detailed Analysis",
        "",
    ])
    
    # Individual model analysis
    for model_name, data in results.items():
        clean_name = model_name.replace('backtest_', '')
        report_lines.extend([
            f"### {clean_name}",
            "",
        ])
        
        # Basic statistics
        if model_name in metrics_df.index:
            metrics = metrics_df.loc[model_name]
            report_lines.extend([
                "**Key Metrics:**",
                f"- Total Return: {metrics.get('total_return', 0):.2f}%",
                f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}",
                f"- Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}",
                f"- Maximum Drawdown: {metrics.get('max_drawdown', 0):.2f}%",
                f"- Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}",
                f"- Value at Risk (95%): {metrics.get('var_95', 0):.2f}%",
                f"- Conditional VaR (95%): {metrics.get('cvar_95', 0):.2f}%",
                f"- Volatility (Annualized): {metrics.get('volatility', 0):.2f}%",
                f"- Skewness: {metrics.get('skewness', 0):.3f}",
                f"- Kurtosis: {metrics.get('kurtosis', 0):.3f}",
                "",
            ])
        
        # Trade statistics
        if 'total_trades' in data:
            report_lines.extend([
                "**Trade Statistics:**",
                f"- Total Trades: {data.get('total_trades', 0)}",
                f"- Win Rate: {data.get('win_rate', 0):.2f}%",
                f"- Profit Factor: {data.get('profit_factor', 0):.2f}",
                f"- Average Trade: {data.get('avg_trade_return', 0):.2f}%",
                "",
            ])
    
    # Risk Analysis
    report_lines.extend([
        "## Risk Analysis",
        "",
        "### Drawdown Analysis",
        "",
        "Maximum drawdown represents the largest peak-to-trough decline in portfolio value. "
        "Lower values indicate better risk management.",
        "",
    ])
    
    if not metrics_df.empty and 'max_drawdown' in metrics_df.columns:
        dd_data = metrics_df['max_drawdown'].dropna()
        if not dd_data.empty:
            report_lines.extend([
                f"- **Lowest Drawdown**: {dd_data.min():.2f}% ({dd_data.idxmin().replace('backtest_', '')})",
                f"- **Highest Drawdown**: {dd_data.max():.2f}% ({dd_data.idxmax().replace('backtest_', '')})",
                f"- **Average Drawdown**: {dd_data.mean():.2f}%",
                "",
            ])
    
    # Visualizations
    report_lines.extend([
        "## Visualizations",
        "",
        "### Equity Curves",
        "",
        "![Equity Curves](../figures/equity_curves.png)",
        "",
        "### Performance Comparison",
        "",
        "![Performance Comparison](../figures/performance_comparison.png)",
        "",
    ])
    
    # Configuration
    report_lines.extend([
        "## Configuration",
        "",
        "**Backtesting Parameters:**",
        f"- Strategy: {config.get('backtest', {}).get('strategy', 'N/A')}",
        f"- Probability Threshold: {config.get('backtest', {}).get('threshold', 'N/A')}",
        f"- Position Size: {config.get('backtest', {}).get('position_size', 'N/A')}",
        f"- Take Profit: {config.get('backtest', {}).get('take_profit', 'N/A')}",
        f"- Stop Loss: {config.get('backtest', {}).get('stop_loss', 'N/A')}",
        f"- Commission: {config.get('backtest', {}).get('commission', 'N/A')}",
        "",
        "---",
        "",
        f"*Report generated by {config.get('project', 'AXON')} v{config.get('version', '1.0.0')}*",
    ])
    
    # Write report
    ensure_dir(Path(save_path).parent)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return save_path

def create_model_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a comparison table of all models with key metrics.
    
    Args:
        results: Dictionary of backtest results
    
    Returns:
        DataFrame with models as index and metrics as columns
    """
    comparison_data = []
    
    for model_name, data in results.items():
        row = {'model': model_name}
        
        # Extract basic metrics
        row.update({
            'total_trades': data.get('total_trades', 0),
            'win_rate': data.get('win_rate', 0),
            'profit_factor': data.get('profit_factor', 0),
            'total_return': data.get('total_return', 0),
            'avg_trade_return': data.get('avg_trade_return', 0),
        })
        
        # Calculate performance metrics if equity curve exists
        if 'equity_curve' in data and not data['equity_curve'].empty:
            perf_metrics = calculate_performance_metrics(data['equity_curve'])
            row.update(perf_metrics)
        
        comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df.set_index('model', inplace=True)
        return df
    else:
        return pd.DataFrame()

def main():
    """Main reporting pipeline."""
    config = load_config()
    print(f"[*] Generating comprehensive report for {config['project']}...")
    
    # Ensure output directories exist
    ensure_dir("outputs/reports")
    ensure_dir("outputs/figures")
    
    try:
        # Load backtest results
        print("[*] Loading backtest results...")
        results = load_backtest_results()
        
        if not results:
            print("[WARNING] No backtest results found. Run backtesting first.")
            return
        
        print(f"[*] Found {len(results)} backtest results")
        
        # Create model comparison table
        print("[*] Calculating performance metrics...")
        metrics_df = create_model_comparison_table(results)
        
        if metrics_df.empty:
            print("[WARNING] No valid metrics calculated")
            return
        
        # Generate visualizations
        print("[*] Creating equity curve visualization...")
        equity_plot_path = create_equity_curve_plot(results)
        print(f"[SUCCESS] Equity curves saved: {equity_plot_path}")
        
        print("[*] Creating performance comparison visualization...")
        comparison_plot_path = create_performance_comparison_plot(metrics_df)
        print(f"[SUCCESS] Performance comparison saved: {comparison_plot_path}")
        
        # Generate Markdown report
        print("[*] Generating Markdown report...")
        report_path = generate_markdown_report(results, metrics_df, config)
        print(f"[SUCCESS] Performance report saved: {report_path}")
        
        # Save metrics table
        metrics_path = "outputs/reports/performance_metrics.csv"
        metrics_df.to_csv(metrics_path)
        print(f"[SUCCESS] Metrics table saved: {metrics_path}")
        
        # Print summary
        print(f"\n[*] Report Generation Summary:")
        print(f"  Models analyzed: {len(results)}")
        print(f"  Visualizations: 2")
        print(f"  Reports generated: 1")
        
        if not metrics_df.empty and 'sharpe_ratio' in metrics_df.columns:
            best_model = metrics_df['sharpe_ratio'].idxmax()
            best_sharpe = metrics_df.loc[best_model, 'sharpe_ratio']
            print(f"  [BEST] Best model: {best_model.replace('backtest_', '')} (Sharpe: {best_sharpe:.3f})")
        
        print(f"\n[SUCCESS] Report generation completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        raise


if __name__ == "__main__":
    main()