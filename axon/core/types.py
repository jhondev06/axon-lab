"""AXON Core Types

Core data types and enums for the AXON system.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime


class PipelineAbort(Exception):
    """Exception raised when pipeline should abort."""
    pass


class StepResult(Enum):
    """Pipeline step result status."""
    SUCCESS = "success"
    FAILURE = "failure"
    SKIP = "skip"


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal data structure."""
    symbol: str
    signal_type: SignalType
    confidence: float
    timestamp: datetime
    price: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate signal after initialization."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


@dataclass
class Trade:
    """Trade execution data structure."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    trade_id: Optional[str] = None
    commission: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'trade_id': self.trade_id,
            'commission': self.commission
        }