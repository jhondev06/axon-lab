"""AXON Ledger Manager

Manages trading ledger and transaction history.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from axon.core.types import Trade, Signal
from axon.utils.config import ConfigManager


class LedgerManager:
    """Manages trading ledger and transaction history."""
    
    def __init__(self, config: Optional[ConfigManager] = None, db_path: str = "data/ledger.db"):
        """Initialize ledger manager.
        
        Args:
            config: Configuration manager
            db_path: Path to SQLite database
        """
        self.config = config or ConfigManager()
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database."""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    trade_id TEXT,
                    commission REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    price REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create portfolio table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def record_trade(self, trade: Trade) -> int:
        """Record a trade in the ledger.
        
        Args:
            trade: Trade object
            
        Returns:
            Trade ID in database
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (symbol, side, quantity, price, timestamp, trade_id, commission)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol,
                trade.side,
                trade.quantity,
                trade.price,
                trade.timestamp.isoformat(),
                trade.trade_id,
                trade.commission
            ))
            
            trade_id = cursor.lastrowid
            conn.commit()
            
            # Update portfolio
            self._update_portfolio(trade)
            
            return trade_id
    
    def record_signal(self, signal: Signal) -> int:
        """Record a signal in the ledger.
        
        Args:
            signal: Signal object
            
        Returns:
            Signal ID in database
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO signals (symbol, signal_type, confidence, timestamp, price, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol,
                signal.signal_type.value,
                signal.confidence,
                signal.timestamp.isoformat(),
                signal.price,
                json.dumps(signal.metadata) if signal.metadata else None
            ))
            
            signal_id = cursor.lastrowid
            conn.commit()
            
            return signal_id
    
    def _update_portfolio(self, trade: Trade) -> None:
        """Update portfolio based on trade.
        
        Args:
            trade: Trade object
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current position
            cursor.execute("SELECT quantity, avg_price FROM portfolio WHERE symbol = ?", (trade.symbol,))
            result = cursor.fetchone()
            
            if result:
                current_qty, current_avg_price = result
            else:
                current_qty, current_avg_price = 0.0, 0.0
            
            # Calculate new position
            if trade.side == 'buy':
                new_qty = current_qty + trade.quantity
                if new_qty > 0:
                    new_avg_price = ((current_qty * current_avg_price) + (trade.quantity * trade.price)) / new_qty
                else:
                    new_avg_price = trade.price
            else:  # sell
                new_qty = current_qty - trade.quantity
                new_avg_price = current_avg_price  # Keep same avg price for sells
            
            # Update or insert portfolio record
            cursor.execute("""
                INSERT OR REPLACE INTO portfolio (symbol, quantity, avg_price, last_updated)
                VALUES (?, ?, ?, ?)
            """, (trade.symbol, new_qty, new_avg_price, datetime.now().isoformat()))
            
            conn.commit()
    
    def get_portfolio(self) -> Dict[str, Dict[str, float]]:
        """Get current portfolio positions.
        
        Returns:
            Dictionary of symbol -> position info
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT symbol, quantity, avg_price FROM portfolio WHERE quantity != 0")
            results = cursor.fetchall()
            
            portfolio = {}
            for symbol, quantity, avg_price in results:
                portfolio[symbol] = {
                    'quantity': quantity,
                    'avg_price': avg_price
                }
            
            return portfolio
    
    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history.
        
        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM trades WHERE symbol = ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM trades 
                    ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            trades = []
            
            for row in cursor.fetchall():
                trade_dict = dict(zip(columns, row))
                trades.append(trade_dict)
            
            return trades
    
    def get_signals(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get signal history.
        
        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of signals to return
            
        Returns:
            List of signal dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM signals WHERE symbol = ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM signals 
                    ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            signals = []
            
            for row in cursor.fetchall():
                signal_dict = dict(zip(columns, row))
                # Parse metadata JSON
                if signal_dict.get('metadata'):
                    signal_dict['metadata'] = json.loads(signal_dict['metadata'])
                signals.append(signal_dict)
            
            return signals
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]
            
            # Total volume
            cursor.execute("SELECT SUM(quantity * price) FROM trades")
            total_volume = cursor.fetchone()[0] or 0
            
            # Unique symbols
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM trades")
            unique_symbols = cursor.fetchone()[0]
            
            # Portfolio value (simplified - would need current prices for accurate calculation)
            portfolio = self.get_portfolio()
            portfolio_positions = len(portfolio)
            
            return {
                'total_trades': total_trades,
                'total_volume': total_volume,
                'unique_symbols': unique_symbols,
                'portfolio_positions': portfolio_positions,
                'portfolio': portfolio
            }
    
    def export_data(self, output_path: str) -> None:
        """Export ledger data to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        data = {
            'trades': self.get_trades(limit=10000),
            'signals': self.get_signals(limit=10000),
            'portfolio': self.get_portfolio(),
            'performance': self.get_performance_stats(),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Ledger data exported to {output_path}")
    
    def clear_data(self, confirm: bool = False) -> None:
        """Clear all ledger data.
        
        Args:
            confirm: Must be True to actually clear data
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear ledger data")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM signals")
            cursor.execute("DELETE FROM portfolio")
            
            conn.commit()
        
        print("Ledger data cleared")