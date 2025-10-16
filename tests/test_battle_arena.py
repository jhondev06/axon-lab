#!/usr/bin/env python3
"""
Test script for Battle Arena components
"""

import os
import sys
sys.path.append('src')

from battle_arena.connectors.binance_connector import BinanceConnector
from battle_arena.core.paper_trader import PaperTrader
from battle_arena.core.order_engine import OrderEngine, ExecutionMode
from battle_arena.core.risk_manager import RiskManager
from battle_arena.utils.helpers import format_currency


def test_binance_connector():
    """Test Binance connector with testnet"""
    print("Testing Binance Connector...")

    # Use testnet API keys (empty for demo)
    api_key = os.environ.get('BINANCE_TEST_API_KEY', '')
    api_secret = os.environ.get('BINANCE_TEST_API_SECRET', '')

    if not api_key or not api_secret:
        print("Binance API keys not found, skipping live test")
        return False

    try:
        connector = BinanceConnector(api_key, api_secret, testnet=True)
        balances = connector.get_balance()
        print(f"Binance connector working. Balances: {len(balances)} assets")
        return True
    except Exception as e:
        print(f"Binance connector failed: {e}")
        return False


def test_paper_trader():
    """Test paper trader functionality"""
    print("Testing Paper Trader...")

    try:
        trader = PaperTrader(initial_capital=10000.0)

        # Test balance
        balances = trader.get_balance()
        print(f"Initial balance: {format_currency(balances[0].total, 'USDT')}")

        # Test order placement
        order = trader.place_order('BTCUSDT', 'buy', 'market', 0.001, 45000.0)
        print(f"Order placed: {order.order_id} - {order.side} {order.quantity} @ {order.price}")

        # Test performance stats
        stats = trader.get_performance_stats()
        print(f"Performance: ${stats['current_balance']:.2f} ({stats['total_return_pct']:.2f}%)")

        return True
    except Exception as e:
        print(f"Paper trader failed: {e}")
        return False


def test_risk_manager():
    """Test risk manager validation"""
    print("Testing Risk Manager...")

    try:
        risk_mgr = RiskManager()

        # Mock data
        balances = [type('Balance', (), {'asset': 'USDT', 'free': 10000.0, 'locked': 0.0, 'total': 10000.0})()]
        positions = []

        # Test order validation
        result = risk_mgr.validate_order('BTCUSDT', 'buy', 0.001, 45000.0, balances, positions)

        if result['approved']:
            print("Risk validation passed")
            return True
        else:
            print(f"Risk validation failed: {result['reason']}")
            return False

    except Exception as e:
        print(f"Risk manager failed: {e}")
        return False


def test_order_engine():
    """Test order engine"""
    print("Testing Order Engine...")

    try:
        # Create components
        trader = PaperTrader(initial_capital=10000.0)
        risk_mgr = RiskManager()
        engine = OrderEngine(mode=ExecutionMode.PAPER, paper_trader=trader)

        # Test order execution
        order = engine.place_order('BTCUSDT', 'buy', 'market', 0.001, 45000.0)
        print(f"Order executed: {order.order_id}")

        # Test balance
        balances = engine.get_balance()
        print(f"Balance after order: {format_currency(balances[0].total, 'USDT')}")

        return True
    except Exception as e:
        print(f"Order engine failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Battle Arena Test Suite")
    print("=" * 50)

    tests = [
        test_paper_trader,
        test_risk_manager,
        test_order_engine,
        test_binance_connector,  # Last because may fail without API keys
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! Battle Arena is ready.")
        return 0
    else:
        print("Some tests failed. Check configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())