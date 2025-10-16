"""AXON Error Lens Module

Performance analysis across volatility and trend regimes.
Identifies where strategies win/lose and generates actionable insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from ..utils import load_config, ensure_dir


def load_backtest_results(config: Dict) -> Dict[str, pd.DataFrame]:
    """Load all backtest results and equity curves."""
    results = {}
    ledgers_dir = Path("outputs/ledgers")
    
    if not ledgers_dir.exists():
        print("[WARNING] No backtest results found. Run backtesting first.")
        return results
    
    # Load equity curves for each model
    for equity_file in ledgers_dir.glob("*_equity.csv"):
        model_name = equity_file.stem.replace("_equity", "")
        try:
            equity_df = pd.read_csv(equity_file)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            results[model_name] = equity_df
            print(f"[*] Loaded equity curve for {model_name}: {len(equity_df)} points")
        except Exception as e:
            print(f"[WARNING] Failed to load {equity_file}: {e}")
    
    return results


def calculate_volatility_regimes(equity_df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Calculate volatility regimes using rolling standard deviation."""
    # Calculate returns
    equity_df = equity_df.copy()
    equity_df['returns'] = equity_df['equity'].pct_change()
    
    # Calculate rolling volatility
    equity_df['rolling_vol'] = equity_df['returns'].rolling(window=window).std()
    
    # Define volatility regimes based on quantiles
    vol_quantiles = equity_df['rolling_vol'].quantile([0.33, 0.67])
    
    def classify_vol_regime(vol):
        if pd.isna(vol):
            return 'unknown'
        elif vol <= vol_quantiles[0.33]:
            return 'low_vol'
        elif vol <= vol_quantiles[0.67]:
            return 'medium_vol'
        else:
            return 'high_vol'
    
    equity_df['vol_regime'] = equity_df['rolling_vol'].apply(classify_vol_regime)
    return equity_df


def calculate_trend_regimes(equity_df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Calculate trend regimes using rolling slope/momentum."""
    equity_df = equity_df.copy()
    
    # Calculate rolling slope (trend)
    def rolling_slope(series, window):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    equity_df['rolling_slope'] = rolling_slope(equity_df['equity'], window)
    
    # Define trend regimes based on slope quantiles
    slope_quantiles = equity_df['rolling_slope'].quantile([0.33, 0.67])
    
    def classify_trend_regime(slope):
        if pd.isna(slope):
            return 'unknown'
        elif slope <= slope_quantiles[0.33]:
            return 'downtrend'
        elif slope <= slope_quantiles[0.67]:
            return 'sideways'
        else:
            return 'uptrend'
    
    equity_df['trend_regime'] = equity_df['rolling_slope'].apply(classify_trend_regime)
    return equity_df


def analyze_regime_performance(equity_df: pd.DataFrame) -> Dict:
    """Analyze performance across different regimes."""
    analysis = {
        'volatility_analysis': {},
        'trend_analysis': {},
        'combined_analysis': {}
    }
    
    # Volatility regime analysis
    for regime in ['low_vol', 'medium_vol', 'high_vol']:
        regime_data = equity_df[equity_df['vol_regime'] == regime]
        if len(regime_data) > 0:
            regime_returns = regime_data['returns'].dropna()
            analysis['volatility_analysis'][regime] = {
                'count': len(regime_data),
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'win_rate': (regime_returns > 0).mean(),
                'total_return': (1 + regime_returns).prod() - 1
            }
    
    # Trend regime analysis
    for regime in ['downtrend', 'sideways', 'uptrend']:
        regime_data = equity_df[equity_df['trend_regime'] == regime]
        if len(regime_data) > 0:
            regime_returns = regime_data['returns'].dropna()
            analysis['trend_analysis'][regime] = {
                'count': len(regime_data),
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'win_rate': (regime_returns > 0).mean(),
                'total_return': (1 + regime_returns).prod() - 1
            }
    
    # Combined regime analysis
    for vol_regime in ['low_vol', 'medium_vol', 'high_vol']:
        for trend_regime in ['downtrend', 'sideways', 'uptrend']:
            combined_data = equity_df[
                (equity_df['vol_regime'] == vol_regime) & 
                (equity_df['trend_regime'] == trend_regime)
            ]
            if len(combined_data) > 0:
                regime_returns = combined_data['returns'].dropna()
                key = f"{vol_regime}_{trend_regime}"
                analysis['combined_analysis'][key] = {
                    'count': len(combined_data),
                    'mean_return': regime_returns.mean(),
                    'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'win_rate': (regime_returns > 0).mean()
                }
    
    return analysis


def identify_performance_patterns(all_analyses: Dict[str, Dict]) -> Dict:
    """Identify patterns across all models and regimes."""
    patterns = {
        'best_regimes': {},
        'worst_regimes': {},
        'consistent_performers': [],
        'unstable_performers': [],
        'regime_specialists': {}
    }
    
    # Find best and worst performing regimes for each model
    for model_name, analysis in all_analyses.items():
        model_patterns = {'vol': {}, 'trend': {}}
        
        # Volatility regime patterns
        vol_sharpes = {k: v['sharpe'] for k, v in analysis['volatility_analysis'].items()}
        if vol_sharpes:
            best_vol = max(vol_sharpes.items(), key=lambda x: x[1])
            worst_vol = min(vol_sharpes.items(), key=lambda x: x[1])
            model_patterns['vol'] = {'best': best_vol, 'worst': worst_vol}
        
        # Trend regime patterns
        trend_sharpes = {k: v['sharpe'] for k, v in analysis['trend_analysis'].items()}
        if trend_sharpes:
            best_trend = max(trend_sharpes.items(), key=lambda x: x[1])
            worst_trend = min(trend_sharpes.items(), key=lambda x: x[1])
            model_patterns['trend'] = {'best': best_trend, 'worst': worst_trend}
        
        patterns['best_regimes'][model_name] = model_patterns
        
        # Check consistency (low variance across regimes)
        all_sharpes = list(vol_sharpes.values()) + list(trend_sharpes.values())
        if len(all_sharpes) > 1:
            sharpe_std = np.std(all_sharpes)
            if sharpe_std < 0.1:  # Low variance threshold
                patterns['consistent_performers'].append(model_name)
            elif sharpe_std > 0.5:  # High variance threshold
                patterns['unstable_performers'].append(model_name)
    
    return patterns


def generate_actionable_insights(patterns: Dict, all_analyses: Dict[str, Dict]) -> List[str]:
    """Generate 3 actionable insights from the analysis."""
    insights = []
    
    # Insight 1: Best regime identification
    if patterns['consistent_performers']:
        best_model = patterns['consistent_performers'][0]
        insights.append(
            f"**Consistent Performance**: {best_model} shows stable performance across regimes. "
            f"Consider increasing position size or allocation to this strategy."
        )
    elif patterns['best_regimes']:
        # Find most common best regime
        best_vol_regimes = [v['vol']['best'][0] for v in patterns['best_regimes'].values() if 'vol' in v and 'best' in v['vol']]
        if best_vol_regimes:
            most_common_vol = max(set(best_vol_regimes), key=best_vol_regimes.count)
            insights.append(
                f"**Regime Optimization**: Most models perform best in {most_common_vol} conditions. "
                f"Consider regime-based position sizing or model switching."
            )
    
    # Insight 2: Weakness identification
    if patterns['unstable_performers']:
        unstable_model = patterns['unstable_performers'][0]
        insights.append(
            f"**Strategy Instability**: {unstable_model} shows high variance across regimes. "
            f"Review feature engineering or consider ensemble methods to improve stability."
        )
    elif patterns['worst_regimes']:
        # Find most common worst regime
        worst_regimes = []
        for model_data in patterns['best_regimes'].values():
            if 'vol' in model_data and 'worst' in model_data['vol']:
                worst_regimes.append(model_data['vol']['worst'][0])
        if worst_regimes:
            most_common_worst = max(set(worst_regimes), key=worst_regimes.count)
            insights.append(
                f"**Regime Weakness**: Multiple models struggle in {most_common_worst} conditions. "
                f"Consider developing specialized features or filters for this regime."
            )
    
    # Insight 3: Feature engineering suggestion
    high_vol_performance = []
    for model_name, analysis in all_analyses.items():
        if 'high_vol' in analysis['volatility_analysis']:
            high_vol_perf = analysis['volatility_analysis']['high_vol']['sharpe']
            high_vol_performance.append((model_name, high_vol_perf))
    
    if high_vol_performance:
        avg_high_vol_sharpe = np.mean([perf[1] for perf in high_vol_performance])
        if avg_high_vol_sharpe < 0:
            insights.append(
                f"**Volatility Adaptation**: All models underperform in high volatility (avg Sharpe: {avg_high_vol_sharpe:.3f}). "
                f"Consider adding volatility-adjusted features or dynamic stop-losses."
            )
        else:
            best_high_vol = max(high_vol_performance, key=lambda x: x[1])
            insights.append(
                f"**Volatility Opportunity**: {best_high_vol[0]} excels in high volatility (Sharpe: {best_high_vol[1]:.3f}). "
                f"Consider volatility-based model selection or increased allocation during volatile periods."
            )
    
    # Ensure we have exactly 3 insights
    while len(insights) < 3:
        insights.append(
            "**Data Collection**: Insufficient regime data for comprehensive analysis. "
            "Consider longer backtesting periods or additional market conditions."
        )
    
    return insights[:3]


def generate_markdown_report(all_analyses: Dict[str, Dict], patterns: Dict, insights: List[str]) -> str:
    """Generate comprehensive Markdown report."""
    report = []
    report.append("# AXON Error Lens Analysis")
    report.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("\nThis analysis examines strategy performance across volatility and trend regimes to identify")
    report.append("optimization opportunities and potential weaknesses.\n")
    
    # Key Insights
    report.append("## Key Actionable Insights")
    for i, insight in enumerate(insights, 1):
        report.append(f"\n{i}. {insight}")
    
    # Model Performance by Regime
    report.append("\n## Model Performance by Regime")
    
    for model_name, analysis in all_analyses.items():
        report.append(f"\n### {model_name}")
        
        # Volatility regimes
        report.append("\n#### Volatility Regimes")
        report.append("| Regime | Count | Mean Return | Sharpe | Win Rate | Total Return |")
        report.append("|--------|-------|-------------|--------|----------|--------------|")
        
        for regime, stats in analysis['volatility_analysis'].items():
            report.append(
                f"| {regime} | {stats['count']} | {stats['mean_return']:.6f} | "
                f"{stats['sharpe']:.3f} | {stats['win_rate']:.3f} | {stats['total_return']:.3f} |"
            )
        
        # Trend regimes
        report.append("\n#### Trend Regimes")
        report.append("| Regime | Count | Mean Return | Sharpe | Win Rate | Total Return |")
        report.append("|--------|-------|-------------|--------|----------|--------------|")
        
        for regime, stats in analysis['trend_analysis'].items():
            report.append(
                f"| {regime} | {stats['count']} | {stats['mean_return']:.6f} | "
                f"{stats['sharpe']:.3f} | {stats['win_rate']:.3f} | {stats['total_return']:.3f} |"
            )
    
    # Performance Patterns
    report.append("\n## Performance Patterns")
    
    if patterns['consistent_performers']:
        report.append(f"\n**Consistent Performers**: {', '.join(patterns['consistent_performers'])}")
    
    if patterns['unstable_performers']:
        report.append(f"\n**Unstable Performers**: {', '.join(patterns['unstable_performers'])}")
    
    # Best/Worst Regimes Summary
    report.append("\n### Best/Worst Regimes by Model")
    for model_name, model_patterns in patterns['best_regimes'].items():
        report.append(f"\n**{model_name}**:")
        if 'vol' in model_patterns and 'best' in model_patterns['vol']:
            best_vol, best_vol_sharpe = model_patterns['vol']['best']
            worst_vol, worst_vol_sharpe = model_patterns['vol']['worst']
            report.append(f"- Best volatility regime: {best_vol} (Sharpe: {best_vol_sharpe:.3f})")
            report.append(f"- Worst volatility regime: {worst_vol} (Sharpe: {worst_vol_sharpe:.3f})")
        
        if 'trend' in model_patterns and 'best' in model_patterns['trend']:
            best_trend, best_trend_sharpe = model_patterns['trend']['best']
            worst_trend, worst_trend_sharpe = model_patterns['trend']['worst']
            report.append(f"- Best trend regime: {best_trend} (Sharpe: {best_trend_sharpe:.3f})")
            report.append(f"- Worst trend regime: {worst_trend} (Sharpe: {worst_trend_sharpe:.3f})")
    
    report.append("\n---")
    report.append("\n*This analysis helps identify regime-specific performance patterns for strategy optimization.*")
    
    return "\n".join(report)


def main():
    """Main error lens analysis."""
    config = load_config()
    print(f"[*] Running error lens analysis for {config['project']}...")
    
    # Ensure output directory exists
    ensure_dir("outputs/reports")
    
    try:
        # Load backtest results
        print("[*] Loading backtest results...")
        backtest_results = load_backtest_results(config)
        
        if not backtest_results:
            print("[ERROR] No backtest results found. Please run backtesting first.")
            return
        
        # Analyze each model
        all_analyses = {}
        print("\n[*] Analyzing regime performance...")
        
        for model_name, equity_df in backtest_results.items():
            print(f"  Analyzing {model_name}...")
            
            # Calculate regimes
            equity_df = calculate_volatility_regimes(equity_df)
            equity_df = calculate_trend_regimes(equity_df)
            
            # Analyze performance
            analysis = analyze_regime_performance(equity_df)
            all_analyses[model_name] = analysis
        
        # Identify patterns
        print("[*] Identifying performance patterns...")
        patterns = identify_performance_patterns(all_analyses)
        
        # Generate insights
        print("[*] Generating actionable insights...")
        insights = generate_actionable_insights(patterns, all_analyses)
        
        # Generate report
        print("[*] Generating analysis report...")
        report_content = generate_markdown_report(all_analyses, patterns, insights)
        
        # Save report
        report_path = Path("outputs/reports/error_lens.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n[SUCCESS] Error lens analysis complete!")
        print(f"[*] Report saved to: {report_path}")
        print("\n[*] Key Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight.split('.')[0]}...")
        
        # Save analysis data for other modules
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'models_analyzed': list(all_analyses.keys()),
            'patterns': patterns,
            'insights': insights
        }
        
        analysis_path = Path("outputs/reports/error_lens_data.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        print(f"[*] Analysis data saved to: {analysis_path}")
        
    except Exception as e:
        print(f"[ERROR] Error lens analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()