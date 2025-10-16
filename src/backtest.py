"""AXON Backtest Module

Strategy backtesting and performance evaluation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .utils import load_config, ensure_dir
from .models import load_model
from axon.core.logging import get_logger

# Initialize logger for debugging
logger = get_logger(__name__)

class Position:
    """Represents a trading position."""
    
    def __init__(self, timestamp: pd.Timestamp, side: str, entry_price: float, 
                 size: float, stop_loss: Optional[float] = None, 
                 take_profit: Optional[float] = None, timeout_bars: Optional[int] = None):
        self.timestamp = timestamp
        self.side = side  # 'long' or 'short'
        self.entry_price = entry_price
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.timeout_bars = timeout_bars
        self.bars_held = 0
        self.exit_timestamp = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0
        self.return_pct = 0.0
    
    def update(self, current_price: float, current_timestamp: pd.Timestamp) -> bool:
        """Update position and check for exit conditions. Returns True if position should be closed."""
        self.bars_held += 1
        
        if self.side == 'long':
            # Check take profit
            if self.take_profit and current_price >= self.take_profit:
                self._close_position(current_price, current_timestamp, 'take_profit')
                return True
            
            # Check stop loss
            if self.stop_loss and current_price <= self.stop_loss:
                self._close_position(current_price, current_timestamp, 'stop_loss')
                return True
        
        elif self.side == 'short':
            # Check take profit
            if self.take_profit and current_price <= self.take_profit:
                self._close_position(current_price, current_timestamp, 'take_profit')
                return True
            
            # Check stop loss
            if self.stop_loss and current_price >= self.stop_loss:
                self._close_position(current_price, current_timestamp, 'stop_loss')
                return True
        
        # Check timeout
        if self.timeout_bars and self.bars_held >= self.timeout_bars:
            self._close_position(current_price, current_timestamp, 'timeout')
            return True
        
        return False
    
    def _close_position(self, exit_price: float, exit_timestamp: pd.Timestamp, reason: str):
        """Close the position and calculate PnL."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.exit_reason = reason
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.size
            self.return_pct = (exit_price - self.entry_price) / self.entry_price
        elif self.side == 'short':
            self.pnl = (self.entry_price - exit_price) * self.size
            self.return_pct = (self.entry_price - exit_price) / self.entry_price
    
    def force_close(self, exit_price: float, exit_timestamp: pd.Timestamp):
        """Force close position at market."""
        self._close_position(exit_price, exit_timestamp, 'market_close')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for logging."""
        return {
            'entry_timestamp': self.timestamp,
            'exit_timestamp': self.exit_timestamp,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timeout_bars': self.timeout_bars,
            'bars_held': self.bars_held,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'return_pct': self.return_pct,
        }

class BacktestEngine:
    """Deterministic backtesting engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy_config = config.get('backtest', {})
        
        # Strategy parameters aligned to axon.cfg.yml
        self.prob_threshold = float(self.strategy_config.get('threshold', 0.6))
        self.position_size = float(self.strategy_config.get('position_size', 1.0))
        self.stop_loss_pct = float(self.strategy_config.get('stop_loss', 0.02))  # 2%
        self.take_profit_pct = float(self.strategy_config.get('take_profit', 0.04))  # 4%
        self.max_positions = int(self.strategy_config.get('max_positions', 1))
        self.commission = float(self.strategy_config.get('commission', 0.0))

        # Timeout parsing from string (e.g., '4h') to bars; default 1m bars
        timeframe = config.get('data', {}).get('timeframe', '1m')
        timeout_cfg = str(self.strategy_config.get('timeout', '60m'))
        def to_minutes(s: str) -> int:
            s = s.strip().lower()
            if s.endswith('m'):
                return int(s[:-1])
            if s.endswith('h'):
                return int(s[:-1]) * 60
            if s.endswith('d'):
                return int(s[:-1]) * 60 * 24
            return int(float(s))
        bar_minutes = 1 if timeframe.endswith('m') else (60 if timeframe.endswith('h') else 1)
        self.timeout_bars = max(1, to_minutes(timeout_cfg) // bar_minutes)
        
        # State
        self.positions = []
        self.closed_positions = []
        self.equity_curve = []
        self.initial_capital = self.strategy_config.get('initial_capital', 10000.0)
        self.current_capital = self.initial_capital
        
        print(f"Backtest Engine initialized:")
        print(f"  Probability threshold: {self.prob_threshold}")
        print(f"  Position size: {self.position_size}")
        print(f"  Stop loss: {self.stop_loss_pct*100:.1f}%")
        print(f"  Take profit: {self.take_profit_pct*100:.1f}%")
        print(f"  Timeout: {self.timeout_bars} bars")
        print(f"  Max positions: {self.max_positions}")
        print(f"  Initial capital: ${self.initial_capital:,.2f}")
        if self.commission > 0:
            print(f"  Commission: {self.commission:.4f} per transaction on notional")
    
    def long_if_prob_gt_strategy(self, row: pd.Series, predictions: np.ndarray, idx: int) -> bool:
        """Long if probability > threshold strategy."""
        if idx >= len(predictions):
            return False
        
        prob = predictions[idx]
        
        # Check if we should enter a long position
        if prob > self.prob_threshold and len(self.positions) < self.max_positions:
            return True
        
        return False
    
    def run_backtest(self, data: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]:
        """Run the backtest on provided data and predictions."""
        logger.info(f"DEBUG: Starting backtest on {len(data):,} bars")
        logger.info(f"DEBUG: Predictions shape: {predictions.shape}, range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        logger.info(f"DEBUG: Data columns: {list(data.columns)}")
        logger.info(f"DEBUG: Data time range: {data.index[0]} to {data.index[-1]}")

        # Reset state
        self.positions = []
        self.closed_positions = []
        self.equity_curve = []
        self.current_capital = self.initial_capital

        # Track equity
        equity_records = []
        
        for idx, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            
            # Update existing positions
            positions_to_close = []
            for pos in self.positions:
                if pos.update(current_price, timestamp):
                    positions_to_close.append(pos)
            
            # Close positions that hit exit conditions
            for pos in positions_to_close:
                self.positions.remove(pos)
                self.closed_positions.append(pos)
                # Commission on exit
                exit_commission = current_price * pos.size * self.commission
                self.current_capital += (pos.pnl - exit_commission)
            
            # Check for new entry signals
            if self.long_if_prob_gt_strategy(row, predictions, idx):
                # Calculate position sizing
                position_value = self.current_capital * self.position_size
                shares = position_value / current_price
                
                # Calculate stop loss and take profit levels
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)
                
                # Create new position
                new_position = Position(
                    timestamp=timestamp,
                    side='long',
                    entry_price=current_price,
                    size=shares,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timeout_bars=self.timeout_bars
                )
                
                self.positions.append(new_position)
                # Commission on entry
                entry_commission = position_value * self.commission
                self.current_capital -= entry_commission
            
            # Calculate current equity (capital + unrealized PnL)
            unrealized_pnl = 0.0
            for pos in self.positions:
                if pos.side == 'long':
                    unrealized_pnl += (current_price - pos.entry_price) * pos.size
                elif pos.side == 'short':
                    unrealized_pnl += (pos.entry_price - current_price) * pos.size
            
            current_equity = self.current_capital + unrealized_pnl
            
            # Record equity
            equity_records.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'capital': self.current_capital,
                'unrealized_pnl': unrealized_pnl,
                'num_positions': len(self.positions),
                'price': current_price
            })
        
        # Force close any remaining positions at the end
        final_price = data.iloc[-1]['close']
        final_timestamp = data.index[-1]
        
        for pos in self.positions:
            pos.force_close(final_price, final_timestamp)
            self.closed_positions.append(pos)
            final_commission = final_price * pos.size * self.commission
            self.current_capital += (pos.pnl - final_commission)
        
        self.positions = []
        self.equity_curve = pd.DataFrame(equity_records)
        
        # Calculate final metrics
        results = self._calculate_results()

        logger.info(f"DEBUG: Backtest completed:")
        logger.info(f"DEBUG:   Total trades: {len(self.closed_positions)}")
        logger.info(f"DEBUG:   Final capital: ${self.current_capital:,.2f}")
        logger.info(f"DEBUG:   Total return: {results['total_return']:.2%}")
        logger.info(f"DEBUG:   Win rate: {results['win_rate']:.2%}")
        logger.info(f"DEBUG:   Sharpe ratio: {results['sharpe_ratio']:.4f}")
        logger.info(f"DEBUG:   Max drawdown: {results['max_drawdown']:.2%}")

        return results
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest results and metrics."""
        logger.info(f"DEBUG: Calculating backtest results from {len(self.closed_positions)} closed positions")

        if not self.closed_positions:
            logger.warning("DEBUG: No closed positions found")
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0,
            }

        # Basic metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        num_trades = len(self.closed_positions)

        logger.info(f"DEBUG: Basic metrics - Total return: {total_return:.2%}, Num trades: {num_trades}")

        # Trade analysis
        returns = [pos.return_pct for pos in self.closed_positions]
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]

        logger.info(f"DEBUG: Trade analysis - Winning: {len(winning_trades)}, Losing: {len(losing_trades)}")
        logger.info(f"DEBUG: Returns distribution - Min: {min(returns):.2%}, Max: {max(returns):.2%}, Mean: {np.mean(returns):.2%}")

        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0
        avg_return = np.mean(returns) if returns else 0.0

        # Profit factor
        total_wins = sum(winning_trades) if winning_trades else 0.0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0

        logger.info(f"DEBUG: Profit analysis - Total wins: {total_wins:.2%}, Total losses: {total_losses:.2%}, Profit factor: {profit_factor:.2f}")
        
        # Drawdown calculation
        if not self.equity_curve.empty:
            equity_series = self.equity_curve['equity']
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0.0
        
        # Sharpe ratio (corrected - use equity curve returns)
        if not self.equity_curve.empty and len(self.equity_curve) > 1:
            equity_returns = self.equity_curve['equity'].pct_change().dropna()
            if len(equity_returns) > 1 and equity_returns.std() > 0:
                # Calculate daily returns (resample for more stable Sharpe)
                equity_curve_copy = self.equity_curve.copy()
                equity_curve_copy['timestamp'] = pd.to_datetime(equity_curve_copy['timestamp'])
                equity_curve_copy.set_index('timestamp', inplace=True)
                
                # Resample to daily for Sharpe calculation (more stable)
                daily_equity = equity_curve_copy['equity'].resample('D').last().dropna()
                if len(daily_equity) > 1:
                    daily_returns = daily_equity.pct_change().dropna()
                    if len(daily_returns) > 1 and daily_returns.std() > 0:
                        # Annualized Sharpe (252 trading days per year)
                        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                    else:
                        sharpe_ratio = 0.0
                else:
                    # Fallback to minute returns if insufficient daily data
                    sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(252 * 24 * 60)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': np.mean(winning_trades) if winning_trades else 0.0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0.0,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
        }
    
    def save_results(self, results: Dict[str, Any], model_name: str) -> str:
        """Save backtest results to files."""
        ensure_dir("outputs/ledgers")
        ensure_dir("outputs/metrics")
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"backtest_{model_name}_{timestamp}"
        
        # Save trade ledger
        if self.closed_positions:
            trades_df = pd.DataFrame([pos.to_dict() for pos in self.closed_positions])
            trades_path = f"outputs/ledgers/{base_filename}_trades.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"Trade ledger saved: {trades_path}")
        
        # Save equity curve
        if not self.equity_curve.empty:
            equity_path = f"outputs/ledgers/{base_filename}_equity.csv"
            self.equity_curve.to_csv(equity_path, index=False)
            print(f"Equity curve saved: {equity_path}")
        
        # Save results summary
        results_path = f"outputs/ledgers/{base_filename}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results summary saved: {results_path}")

        # Also save compact metrics file under outputs/metrics
        bt_metrics = {
            'type': 'backtest',
            'timestamp': timestamp,
            'model': model_name,
            'metrics': results,
        }
        bt_path = f"outputs/metrics/BT_{model_name}_{timestamp}.json"
        with open(bt_path, 'w') as f:
            json.dump(bt_metrics, f, indent=2, default=str)
        print(f"Metrics saved: {bt_path}")
        
        return base_filename

def run_backtest_for_model(model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run backtest for a specific model."""
    print(f"\nRunning backtest for model: {model_path}")
    
    # Load model
    model, metadata = load_model(model_path)
    model_name = metadata.get('model_name', 'unknown')
    
    # Load test data
    test_path = "data/processed/test_features.parquet"
    if not Path(test_path).exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    test_df = pd.read_parquet(test_path)
    print(f"Test data loaded: {test_df.shape}")
    
    # Prepare features
    target_col = config.get('target', 'y')
    feature_cols = [col for col in test_df.columns if col not in ['timestamp', target_col]]
    
    X_test = test_df[feature_cols]
    
    # Generate predictions
    print("Generating predictions...")

    # Check feature compatibility
    expected_features = len(feature_cols)
    if hasattr(model, 'n_features_in_'):
        # Scikit-learn models
        model_features = model.n_features_in_
    elif hasattr(model, 'num_feature'):
        # LightGBM
        model_features = model.num_feature()
    else:
        model_features = expected_features  # Assume compatible

    if model_features != expected_features:
        raise ValueError(f"Feature mismatch: model trained with {model_features} features, but data has {expected_features} features")

    if hasattr(model, 'predict_proba'):
        # Scikit-learn style or EnsembleModel
        predictions = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'predict'):
        # LightGBM
        predictions = model.predict(X_test, num_iteration=getattr(model, 'best_iteration', None))
    else:
        raise ValueError(f"Unknown model type for prediction: {type(model)}")
    
    print(f"Predictions generated: {len(predictions)}")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Prepare price data for backtesting
    price_data = test_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    price_data.set_index('timestamp', inplace=True)
    
    # Run backtest
    engine = BacktestEngine(config)
    results = engine.run_backtest(price_data, predictions)
    
    # Add model info to results
    results['model_name'] = model_name
    results['model_path'] = model_path
    results['test_period'] = {
        'start': str(price_data.index[0]),
        'end': str(price_data.index[-1]),
        'num_bars': len(price_data)
    }
    
    # Save results
    filename = engine.save_results(results, model_name)
    results['output_files'] = filename
    
    return results

def main():
    """Main backtesting pipeline."""
    print("=== AXON Backtesting Module ===")
    
    # Load configuration
    config = load_config()
    
    try:
        # Find trained models
        artifacts_dir = Path("outputs/artifacts")
        if not artifacts_dir.exists():
            raise FileNotFoundError("No trained models found. Run model training first.")
        
        model_files = list(artifacts_dir.glob("*.pkl"))
        if not model_files:
            raise FileNotFoundError("No model files found in outputs/artifacts/")
        
        print(f"Found {len(model_files)} trained models")
        
        all_results = {}
        
        # Run backtest for each model
        for model_file in model_files:
            try:
                model_path = str(model_file)
                results = run_backtest_for_model(model_path, config)
                all_results[results['model_name']] = results
                
            except Exception as e:
                print(f"[ERROR] Failed to backtest {model_file.name}: {e}")
                continue
        
        # Model comparison
        if len(all_results) > 1:
            print(f"\n{'='*60}")
            print("BACKTEST COMPARISON")
            print(f"{'='*60}")
            
            comparison_data = {}
            for model_name, results in all_results.items():
                comparison_data[model_name] = {
                    'Total Return': f"{results['total_return']:.2%}",
                    'Num Trades': results['num_trades'],
                    'Win Rate': f"{results['win_rate']:.2%}",
                    'Sharpe Ratio': f"{results['sharpe_ratio']:.3f}",
                    'Max Drawdown': f"{results['max_drawdown']:.2%}",
                    'Profit Factor': f"{results['profit_factor']:.2f}"
                }
            
            comparison_df = pd.DataFrame(comparison_data).T
            print(comparison_df)
            
            # Save comparison
            comparison_path = "outputs/ledgers/backtest_comparison.csv"
            comparison_df.to_csv(comparison_path)
            print(f"\nBacktest comparison saved: {comparison_path}")
            
            # Find best model by Sharpe ratio
            best_model = max(all_results.keys(), key=lambda k: all_results[k]['sharpe_ratio'])
            best_sharpe = all_results[best_model]['sharpe_ratio']
            print(f"\n[*] Best performing model: {best_model} (Sharpe: {best_sharpe:.3f})")
        
        print(f"\n[SUCCESS] Backtesting completed successfully!")
        print(f"Results saved in outputs/ledgers/")
        
    except Exception as e:
        print(f"[ERROR] Backtesting failed: {e}")
        raise

if __name__ == "__main__":
    main()
