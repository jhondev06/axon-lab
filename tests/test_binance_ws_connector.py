"""
Testes unitários para BinanceWebSocketConnector.

Testa conexão, parsing de dados, tratamento de erros e reconexão.
"""

import pytest
import asyncio
import json
import threading
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from axon.market_data.binance_ws import BinanceWebSocketConnector


class TestBinanceWebSocketConnector:
    """Testes unitários para BinanceWebSocketConnector."""
    
    @pytest.fixture
    def config(self):
        """Configuração de teste."""
        return {
            "battle_arena": {
                "execution": {
                    "testnet": True
                }
            }
        }
    
    @pytest.fixture
    def connector(self, config):
        """Instância do connector para testes."""
        return BinanceWebSocketConnector(config)
    
    def test_initialization(self, connector):
        """Testa inicialização do connector."""
        assert connector.use_testnet is True
        assert connector.ws_url == "wss://testnet.binance.vision/ws/"
        assert connector.is_connected is False
        assert connector.reconnect_attempts == 0
        assert connector.max_reconnect_attempts == 5
        assert connector.reconnect_delay == 5
        assert connector.active_streams == {}
        assert connector.callback_function is None
    
    def test_initialization_mainnet(self):
        """Testa inicialização para mainnet."""
        config = {
            "battle_arena": {
                "execution": {
                    "testnet": False
                }
            }
        }
        connector = BinanceWebSocketConnector(config)
        assert connector.use_testnet is False
        assert connector.ws_url == "wss://stream.binance.com:9443/ws/"
    
    def test_normalize_symbol(self, connector):
        """Testa normalização de símbolos."""
        assert connector._normalize_symbol("btc/usdt") == "BTCUSDT"
        assert connector._normalize_symbol("eth/usdt") == "ETHUSDT"
        assert connector._normalize_symbol("BTCUSDT") == "BTCUSDT"
        assert connector._normalize_symbol("btcusdt") == "BTCUSDT"
    
    def test_normalize_interval(self, connector):
        """Testa normalização de intervalos."""
        assert connector._normalize_interval("1m") == "1m"
        assert connector._normalize_interval("5M") == "5m"
        assert connector._normalize_interval("1H") == "1h"
        assert connector._normalize_interval("invalid") == "1m"  # default
    
    def test_create_stream_name(self, connector):
        """Testa criação de nomes de stream."""
        stream_name = connector._create_stream_name("BTC/USDT", "1m")
        assert stream_name == "btcusdt@kline_1m"
        
        stream_name = connector._create_stream_name("ETH/USDT", "5m")
        assert stream_name == "ethusdt@kline_5m"
    
    def test_parse_kline_data_valid(self, connector):
        """Testa parsing de dados válidos de kline."""
        raw_data = {
            "k": {
                "s": "BTCUSDT",
                "t": 1640995200000,
                "T": 1640995259999,
                "o": "47000.00",
                "h": "47100.00",
                "l": "46900.00",
                "c": "47050.00",
                "v": "10.5",
                "i": "1m",
                "x": True
            }
        }
        
        parsed = connector._parse_kline_data(raw_data)
        
        assert parsed["symbol"] == "BTCUSDT"
        assert parsed["open_time"] == 1640995200000
        assert parsed["close_time"] == 1640995259999
        assert parsed["open"] == 47000.00
        assert parsed["high"] == 47100.00
        assert parsed["low"] == 46900.00
        assert parsed["close"] == 47050.00
        assert parsed["volume"] == 10.5
        assert parsed["interval"] == "1m"
        assert parsed["is_closed"] is True
    
    def test_parse_kline_data_invalid(self, connector):
        """Testa parsing de dados inválidos."""
        # Dados malformados
        invalid_data = {"invalid": "data"}
        parsed = connector._parse_kline_data(invalid_data)
        assert parsed == {}
        
        # Dados com valores inválidos
        invalid_kline = {
            "k": {
                "s": "BTCUSDT",
                "o": "invalid_price"  # Preço inválido
            }
        }
        parsed = connector._parse_kline_data(invalid_kline)
        assert parsed == {}
    
    @pytest.mark.asyncio
    async def test_handle_message_kline(self, connector):
        """Testa tratamento de mensagens de kline."""
        callback_mock = Mock()
        connector.callback_function = callback_mock
        
        message = json.dumps({
            "k": {
                "s": "BTCUSDT",
                "t": 1640995200000,
                "T": 1640995259999,
                "o": "47000.00",
                "h": "47100.00",
                "l": "46900.00",
                "c": "47050.00",
                "v": "10.5",
                "i": "1m",
                "x": True
            }
        })
        
        await connector._handle_message(message)
        
        # Verifica se callback foi chamado
        callback_mock.assert_called_once()
        call_args = callback_mock.call_args[0][0]
        assert "BTCUSDT" in call_args
        assert call_args["BTCUSDT"]["open"] == 47000.00
    
    @pytest.mark.asyncio
    async def test_handle_message_ping_pong(self, connector):
        """Testa tratamento de ping/pong."""
        # Mock do websocket
        websocket_mock = AsyncMock()
        connector.websocket = websocket_mock
        
        ping_message = json.dumps({"ping": 1640995200000})
        
        await connector._handle_message(ping_message)
        
        # Verifica se pong foi enviado
        websocket_mock.send.assert_called_once()
        sent_message = json.loads(websocket_mock.send.call_args[0][0])
        assert "pong" in sent_message
        assert sent_message["pong"] == 1640995200000
    
    @pytest.mark.asyncio
    async def test_handle_message_invalid_json(self, connector):
        """Testa tratamento de JSON inválido."""
        invalid_message = "invalid json"
        
        # Não deve gerar exceção
        await connector._handle_message(invalid_message)
    
    @pytest.mark.asyncio
    async def test_handle_message_callback_error(self, connector):
        """Testa tratamento de erro no callback."""
        callback_mock = Mock(side_effect=Exception("Callback error"))
        connector.callback_function = callback_mock
        
        message = json.dumps({
            "k": {
                "s": "BTCUSDT",
                "t": 1640995200000,
                "T": 1640995259999,
                "o": "47000.00",
                "h": "47100.00",
                "l": "46900.00",
                "c": "47050.00",
                "v": "10.5",
                "i": "1m",
                "x": True
            }
        })
        
        # Não deve gerar exceção mesmo com erro no callback
        await connector._handle_message(message)
        callback_mock.assert_called_once()
    
    def test_start_klines_stream_no_symbols(self, connector):
        """Testa início de stream sem símbolos."""
        callback_mock = Mock()
        
        connector.start_klines_stream([], "1m", callback_mock)
        
        # Não deve iniciar stream
        assert connector.callback_function is None
        assert len(connector.active_streams) == 0
    
    @patch('threading.Thread')
    def test_start_klines_stream_valid(self, thread_mock, connector):
        """Testa início de stream com símbolos válidos."""
        callback_mock = Mock()
        thread_instance_mock = Mock()
        thread_mock.return_value = thread_instance_mock
        
        symbols = ["BTCUSDT", "ETHUSDT"]
        interval = "1m"
        
        connector.start_klines_stream(symbols, interval, callback_mock)
        
        # Verifica configuração
        assert connector.callback_function == callback_mock
        assert len(connector.active_streams) == 2
        assert "BTCUSDT" in connector.active_streams
        assert "ETHUSDT" in connector.active_streams
        
        # Verifica thread
        thread_mock.assert_called_once()
        thread_instance_mock.start.assert_called_once()
    
    @patch('asyncio.create_task')
    def test_stop_stream(self, create_task_mock, connector):
        """Testa parada do stream."""
        # Simula estado ativo
        connector.is_connected = True
        connector.active_streams = {"BTCUSDT": {"interval": "1m"}}
        connector.callback_function = Mock()
        
        # Mock websocket
        websocket_mock = Mock()
        websocket_mock.closed = False
        connector.websocket = websocket_mock
        
        connector.stop_stream()
        
        # Verifica limpeza
        assert connector.is_connected is False
        assert len(connector.active_streams) == 0
        assert connector.callback_function is None
    
    def test_get_connection_status(self, connector):
        """Testa obtenção do status da conexão."""
        # Estado inicial
        status = connector.get_connection_status()
        
        assert status["is_connected"] is False
        assert status["active_streams"] == []
        assert status["reconnect_attempts"] == 0
        assert status["use_testnet"] is True
        
        # Simula estado conectado
        connector.is_connected = True
        connector.active_streams = {"BTCUSDT": {"interval": "1m"}}
        connector.reconnect_attempts = 2
        
        status = connector.get_connection_status()
        
        assert status["is_connected"] is True
        assert status["active_streams"] == ["BTCUSDT"]
        assert status["reconnect_attempts"] == 2
    
    @pytest.mark.asyncio
    async def test_handle_reconnection_within_limit(self, connector):
        """Testa reconexão dentro do limite."""
        connector.reconnect_attempts = 2
        connector.max_reconnect_attempts = 5
        
        with patch.object(connector, '_connect_websocket') as connect_mock:
            with patch('asyncio.sleep') as sleep_mock:
                await connector._handle_reconnection(["btcusdt@kline_1m"])
                
                # Verifica incremento de tentativas
                assert connector.reconnect_attempts == 3
                
                # Verifica chamadas
                sleep_mock.assert_called_once_with(5)
                connect_mock.assert_called_once_with(["btcusdt@kline_1m"])
    
    @pytest.mark.asyncio
    async def test_handle_reconnection_max_attempts(self, connector):
        """Testa reconexão no limite máximo."""
        connector.reconnect_attempts = 5
        connector.max_reconnect_attempts = 5
        
        with patch.object(connector, '_connect_websocket') as connect_mock:
            await connector._handle_reconnection(["btcusdt@kline_1m"])
            
            # Não deve tentar reconectar
            connect_mock.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])