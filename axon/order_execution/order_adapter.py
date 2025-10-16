"""AXON Order Execution - Symbolic link to src implementation"""

# Import from the actual implementation in src/
from src.order_execution.order_adapter import OrderAdapter

__all__ = ['OrderAdapter']