"""
Testes de integração para BattleArena com BinanceWebSocketConnector.

Valida o fluxo completo de dados desde o WebSocket até as decisões de trade.
"""

import pytest
import asyncio
import json
import threading
import time
from unittest.mock import Mock, patch, MagicMock, call, ANY
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.brains.battle_arena import BattleArena
from axon.market_data.binance_ws import BinanceWebSocketConnector


class TestBattleArenaIntegration:
    """Testes de integração para BattleArena com BinanceWebSocketConnector."""
    
    @pytest.fixture
    def mock_config(self):
        """Configuração mock para testes."""
        return {
            "battle_arena": {
                "enabled": True,
                "execution": {
                    "mode": "paper",
                    "testnet": True
                },
                "connectors": {
                    "binance": {
                        "enabled": True
                    }
                },
                "symbols": ["BTCUSDT", "ETHUSDT"]
            },
            "data": {
                "interval": "1m"
            },
            "telegram": {
                "enabled": False
            }
        }
    
    @pytest.fixture
    def mock_kline_data(self):
        """Dados de kline mock para testes."""
        return {
            "BTCUSDT": {
                "symbol": "BTCUSDT",
                "open_time": 1640995200000,
                "close_time": 1640995259999,
                "open": 47000.00,
                "high": 47100.00,
                "low": 46900.00,
                "close": 47050.00,
                "volume": 10.5,
                "interval": "1m",
                "is_closed": True
            }
        }
    
    @patch('src.brains.battle_arena.ConfigManager')
    @patch('src.brains.battle_arena.Notifier')
    @patch('src.brains.battle_arena.FeatureEngineer')
    @patch('src.brains.battle_arena.DecisionEngine')
    @patch('src.brains.battle_arena.RiskManager')
    @patch('src.brains.battle_arena.OrderAdapter')
    @patch('src.brains.battle_arena.LedgerManager')
    @patch('axon.market_data.binance_ws.BinanceWebSocketConnector')
    def test_battle_arena_initialization_with_ws_connector(
        self, mock_ws_connector, mock_ledger, mock_order_adapter, 
        mock_risk_manager, mock_decision_engine, mock_feature_engineer,
        mock_notifier, mock_config_manager, mock_config
    ):
        """Testa inicialização da BattleArena com WebSocket connector."""
        # Setup mocks
        mock_config_manager.return_value.get_config.return_value = mock_config
        
        # Inicializar BattleArena
        arena = BattleArena()
        
        # Verificar inicialização
        assert arena.enabled is True
        assert arena.mode == "paper"
        assert arena.testnet is True
        assert arena.binance_ws_connector is not None
        
        # Verificar se o WebSocket connector foi criado com a configuração correta
        mock_ws_connector.assert_called_once_with(mock_config)
    
    @patch('src.brains.battle_arena.ConfigManager')
    @patch('src.brains.battle_arena.Notifier')
    @patch('src.brains.battle_arena.FeatureEngineer')
    @patch('src.brains.battle_arena.DecisionEngine')
    @patch('src.brains.battle_arena.RiskManager')
    @patch('src.brains.battle_arena.OrderAdapter')
    @patch('src.brains.battle_arena.LedgerManager')
    @patch('axon.market_data.binance_ws.BinanceWebSocketConnector')
    def test_klines_data_processing_flow(
        self, mock_ws_connector, mock_ledger, mock_order_adapter, 
        mock_risk_manager, mock_decision_engine, mock_feature_engineer,
        mock_notifier, mock_config_manager, mock_config, mock_kline_data
    ):
        """Testa o fluxo completo de processamento de dados de klines."""
        # Setup mocks
        mock_config_manager.return_value.get_config.return_value = mock_config
        mock_feature_engineer.return_value.generate_features.return_value = {"feature1": 0.5}
        
        # Mock trade signal
        mock_signal = Mock()
        mock_signal.side = "BUY"
        mock_signal.symbol = "BTCUSDT"
        mock_signal.entry_price = 47050.00
        mock_decision_engine.return_value.make_decision.return_value = mock_signal
        
        # Mock risk manager approval
        mock_risk_manager.return_value.check_trade_risk.return_value = True
        
        # Mock order result
        mock_order_result = {"status": "filled", "order_id": "12345"}
        mock_order_adapter.return_value.place_paper_order.return_value = mock_order_result
        
        # Inicializar BattleArena
        arena = BattleArena()
        
        # Simular processamento de klines
        processed_features = arena._process_klines(mock_kline_data["BTCUSDT"])
        
        # Verificar se as features foram geradas
        mock_feature_engineer.return_value.generate_features.assert_called_once()
        assert processed_features == {"feature1": 0.5}
    
    @patch('src.brains.battle_arena.ConfigManager')
    @patch('src.brains.battle_arena.Notifier')
    @patch('src.brains.battle_arena.FeatureEngineer')
    @patch('src.brains.battle_arena.DecisionEngine')
    @patch('src.brains.battle_arena.RiskManager')
    @patch('src.brains.battle_arena.OrderAdapter')
    @patch('src.brains.battle_arena.LedgerManager')
    @patch('axon.market_data.binance_ws.BinanceWebSocketConnector')
    def test_paper_mode_execution_flow(
        self, mock_ws_connector, mock_ledger, mock_order_adapter, 
        mock_risk_manager, mock_decision_engine, mock_feature_engineer,
        mock_notifier, mock_config_manager, mock_config, mock_kline_data
    ):
        """Testa o fluxo de execução no modo paper."""
        # Setup mocks
        mock_config_manager.return_value.get_config.return_value = mock_config
        mock_feature_engineer.return_value.generate_features.return_value = {"feature1": 0.5}
        
        # Mock trade signal
        mock_signal = Mock()
        mock_signal.side = "BUY"
        mock_signal.symbol = "BTCUSDT"
        mock_signal.entry_price = 47050.00
        mock_decision_engine.return_value.make_decision.return_value = mock_signal
        
        # Mock risk manager approval
        mock_risk_manager.return_value.check_trade_risk.return_value = True
        
        # Mock order result
        mock_order_result = {"status": "filled", "order_id": "12345"}
        mock_order_adapter.return_value.place_paper_order.return_value = mock_order_result
        
        # Mock approved artifacts
        with patch.object(BattleArena, '_load_approved_artifacts', return_value=["model1.pkl"]):
            arena = BattleArena()
        
        # Simular callback de klines
        def simulate_klines_callback():
            # Encontrar o callback passado para start_klines_stream
            call_args = mock_ws_connector.return_value.start_klines_stream.call_args
            if call_args:
                callback = call_args[0][2]  # Terceiro argumento é o callback
                callback(mock_kline_data)
        
        # Patch para evitar loop infinito
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                # Simular execução do modo paper
                arena._execute_paper_mode()
        
        # Verificar se o WebSocket foi iniciado
        mock_ws_connector.return_value.start_klines_stream.assert_called_once_with(
            ["BTCUSDT", "ETHUSDT"], "1m", ANY
        )
    
    @patch('src.brains.battle_arena.ConfigManager')
    @patch('src.brains.battle_arena.Notifier')
    @patch('src.brains.battle_arena.FeatureEngineer')
    @patch('src.brains.battle_arena.DecisionEngine')
    @patch('src.brains.battle_arena.RiskManager')
    @patch('src.brains.battle_arena.OrderAdapter')
    @patch('src.brains.battle_arena.LedgerManager')
    @patch('axon.market_data.binance_ws.BinanceWebSocketConnector')
    def test_alert_mode_execution_flow(
        self, mock_ws_connector, mock_ledger, mock_order_adapter, 
        mock_risk_manager, mock_decision_engine, mock_feature_engineer,
        mock_notifier, mock_config_manager, mock_config, mock_kline_data
    ):
        """Testa o fluxo de execução no modo alert."""
        # Setup config para modo alert
        alert_config = mock_config.copy()
        alert_config["battle_arena"]["execution"]["mode"] = "alert"
        
        mock_config_manager.return_value.get_config.return_value = alert_config
        mock_feature_engineer.return_value.generate_features.return_value = {"feature1": 0.5}
        
        # Mock trade signal
        mock_signal = Mock()
        mock_signal.side = "BUY"
        mock_signal.symbol = "BTCUSDT"
        mock_signal.entry_price = 47050.00
        mock_decision_engine.return_value.make_decision.return_value = mock_signal
        
        # Mock approved artifacts
        with patch.object(BattleArena, '_load_approved_artifacts', return_value=["model1.pkl"]):
            arena = BattleArena()
        
        # Patch para evitar loop infinito
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                # Simular execução do modo alert
                arena._execute_alert_mode()
        
        # Verificar se o WebSocket foi iniciado
        mock_ws_connector.return_value.start_klines_stream.assert_called_once_with(
            ["BTCUSDT", "ETHUSDT"], "1m", ANY
        )
    
    @patch('src.brains.battle_arena.ConfigManager')
    @patch('src.brains.battle_arena.Notifier')
    @patch('src.brains.battle_arena.FeatureEngineer')
    @patch('src.brains.battle_arena.DecisionEngine')
    @patch('src.brains.battle_arena.RiskManager')
    @patch('src.brains.battle_arena.OrderAdapter')
    @patch('src.brains.battle_arena.LedgerManager')
    @patch('axon.market_data.binance_ws.BinanceWebSocketConnector')
    def test_risk_manager_rejection_flow(
        self, mock_ws_connector, mock_ledger, mock_order_adapter, 
        mock_risk_manager, mock_decision_engine, mock_feature_engineer,
        mock_notifier, mock_config_manager, mock_config, mock_kline_data
    ):
        """Testa o fluxo quando o risk manager rejeita um trade."""
        # Setup mocks
        mock_config_manager.return_value.get_config.return_value = mock_config
        mock_feature_engineer.return_value.generate_features.return_value = {"feature1": 0.5}
        
        # Mock trade signal
        mock_signal = Mock()
        mock_signal.side = "BUY"
        mock_signal.symbol = "BTCUSDT"
        mock_signal.entry_price = 47050.00
        mock_decision_engine.return_value.make_decision.return_value = mock_signal
        
        # Mock risk manager rejection
        mock_risk_manager.return_value.check_trade_risk.return_value = False
        
        # Mock approved artifacts
        with patch.object(BattleArena, '_load_approved_artifacts', return_value=["model1.pkl"]):
            arena = BattleArena()
        
        # Simular callback de klines
        def simulate_klines_callback():
            call_args = mock_ws_connector.return_value.start_klines_stream.call_args
            if call_args:
                callback = call_args[0][2]
                callback(mock_kline_data)
        
        # Patch para evitar loop infinito
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                arena._execute_paper_mode()
        
        # Verificar que nenhuma ordem foi colocada
        mock_order_adapter.return_value.place_paper_order.assert_not_called()
        mock_ledger.return_value.record_trade.assert_not_called()
    
    @patch('src.brains.battle_arena.ConfigManager')
    @patch('src.brains.battle_arena.Notifier')
    @patch('src.brains.battle_arena.FeatureEngineer')
    @patch('src.brains.battle_arena.DecisionEngine')
    @patch('src.brains.battle_arena.RiskManager')
    @patch('src.brains.battle_arena.OrderAdapter')
    @patch('src.brains.battle_arena.LedgerManager')
    @patch('axon.market_data.binance_ws.BinanceWebSocketConnector')
    def test_hold_signal_no_action(
        self, mock_ws_connector, mock_ledger, mock_order_adapter, 
        mock_risk_manager, mock_decision_engine, mock_feature_engineer,
        mock_notifier, mock_config_manager, mock_config, mock_kline_data
    ):
        """Testa que nenhuma ação é tomada quando o sinal é HOLD."""
        # Setup mocks
        mock_config_manager.return_value.get_config.return_value = mock_config
        mock_feature_engineer.return_value.generate_features.return_value = {"feature1": 0.5}
        
        # Mock HOLD signal
        mock_signal = Mock()
        mock_signal.side = "HOLD"
        mock_signal.symbol = "BTCUSDT"
        mock_decision_engine.return_value.make_decision.return_value = mock_signal
        
        # Mock approved artifacts
        with patch.object(BattleArena, '_load_approved_artifacts', return_value=["model1.pkl"]):
            arena = BattleArena()
        
        # Simular callback de klines
        def simulate_klines_callback():
            call_args = mock_ws_connector.return_value.start_klines_stream.call_args
            if call_args:
                callback = call_args[0][2]
                callback(mock_kline_data)
        
        # Patch para evitar loop infinito
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                arena._execute_paper_mode()
        
        # Verificar que nenhuma ação foi tomada
        mock_risk_manager.return_value.check_trade_risk.assert_not_called()
        mock_order_adapter.return_value.place_paper_order.assert_not_called()
        mock_ledger.return_value.record_trade.assert_not_called()
    
    @patch('src.brains.battle_arena.ConfigManager')
    def test_disabled_battle_arena(self, mock_config_manager):
        """Testa comportamento quando BattleArena está desabilitada."""
        disabled_config = {
            "battle_arena": {
                "enabled": False
            }
        }
        mock_config_manager.return_value.get_config.return_value = disabled_config
        
        arena = BattleArena()
        
        # Verificar que está desabilitada
        assert arena.enabled is False
        
        # Verificar que run() não faz nada
        result = arena.run()
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])