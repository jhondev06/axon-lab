"""
Testes para componentes da Battle Arena Fase 2.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Adicionar src ao path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from battle_arena.core.model_loader import ModelLoader, ModelInfo
from battle_arena.core.signal_generator import SignalGenerator, SignalType
from battle_arena.core.realtime_feature_engineer import RealtimeFeatureEngineer
from battle_arena.core.data_feed import DataFeed
from battle_arena.core.battle_controller import BattleController


class TestModelLoader:
    """Testes para ModelLoader."""

    @patch('battle_arena.core.model_loader.Path')
    def test_load_model_success(self, mock_path):
        """Testa carregamento bem-sucedido de modelo."""
        # Mock do arquivo de metadata
        mock_metadata = {
            'model_name': 'lightgbm',
            'timestamp': '20250926_094300',
            'feature_names': ['close', 'volume', 'returns'],
            'metrics': {'auc': 0.85}
        }

        # Mock do arquivo do modelo
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.3, 0.7]]

        with patch('builtins.open', create=True) as mock_open, \
             patch('json.load', return_value=mock_metadata), \
             patch('pickle.load', return_value=mock_model), \
             patch('os.path.exists', return_value=True):

            loader = ModelLoader()
            result = loader.load_model('lightgbm_20250926_094300')

            assert result is not None
            assert result.model_name == 'lightgbm'
            assert result.feature_names == ['close', 'volume', 'returns']
            assert result.metrics['auc'] == 0.85

    def test_validate_model_compatibility(self):
        """Testa validação de compatibilidade do modelo."""
        loader = ModelLoader()

        # Modelo válido
        valid_model = ModelInfo(
            model_name='lightgbm',
            timestamp='20250926_094300',
            model_type='lightgbm',
            model=Mock(),
            metadata={},
            feature_names=['close', 'volume', 'returns'],
            config={},
            metrics={'auc': 0.85}
        )

        result = loader.validate_model_compatibility(valid_model)
        assert result['compatible'] is True

        # Modelo sem features obrigatórias
        invalid_model = ModelInfo(
            model_name='lightgbm',
            timestamp='20250926_094300',
            model_type='lightgbm',
            model=Mock(),
            metadata={},
            feature_names=['invalid_feature'],
            config={},
            metrics={'auc': 0.5}
        )

        result = loader.validate_model_compatibility(invalid_model)
        assert result['compatible'] is False
        assert len(result['issues']) > 0


class TestSignalGenerator:
    """Testes para SignalGenerator."""

    def test_signal_generation_buy(self):
        """Testa geração de sinal BUY."""
        # Mock do modelo
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.2, 0.8]]  # Forte probabilidade de BUY

        # Mock do ModelInfo
        model_info = ModelInfo(
            model_name='lightgbm',
            timestamp='20250926_094300',
            model_type='lightgbm',
            model=mock_model,
            metadata={},
            feature_names=['close', 'volume', 'body_size'],  # Features que existem
            config={},
            metrics={'auc': 0.85}
        )

        generator = SignalGenerator(model_info, feature_window=5)  # Menor janela para teste

        # Adicionar dados históricos suficientes
        base_price = 50000.0
        for i in range(10):  # Mais dados que o necessário
            test_data = {
                'timestamp': datetime.now() + timedelta(minutes=i),
                'open': base_price + i * 10,
                'high': base_price + 100 + i * 10,
                'low': base_price - 100 + i * 10,
                'close': base_price + 50 + i * 10,
                'volume': 100.0 + i * 10
            }
            generator.add_market_data('BTCUSDT', test_data)

        # Gerar sinal
        signal = generator.generate_signal('BTCUSDT')

        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence > 0.5
        assert signal.probability > 0.5

    def test_signal_generation_hold(self):
        """Testa geração de sinal HOLD (baixa confiança)."""
        # Mock do modelo com baixa confiança
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.45, 0.55]]  # Confiança baixa

        model_info = ModelInfo(
            model_name='lightgbm',
            timestamp='20250926_094300',
            model_type='lightgbm',
            model=mock_model,
            metadata={},
            feature_names=['close', 'volume', 'body_size'],  # Features que existem
            config={},
            metrics={'auc': 0.85}
        )

        generator = SignalGenerator(model_info, min_confidence=0.6, feature_window=5)

        # Adicionar dados históricos suficientes
        base_price = 50000.0
        for i in range(10):  # Mais dados que o necessário
            test_data = {
                'timestamp': datetime.now() + timedelta(minutes=i),
                'open': base_price + i * 10,
                'high': base_price + 100 + i * 10,
                'low': base_price - 100 + i * 10,
                'close': base_price + 50 + i * 10,
                'volume': 100.0 + i * 10
            }
            generator.add_market_data('BTCUSDT', test_data)

        signal = generator.generate_signal('BTCUSDT')

        assert signal is not None
        assert signal.signal_type == SignalType.HOLD
        assert signal.position_size == 0.0


class TestRealtimeFeatureEngineer:
    """Testes para RealtimeFeatureEngineer."""

    def test_feature_calculation(self):
        """Testa cálculo de features em tempo real."""
        engineer = RealtimeFeatureEngineer()

        # Adicionar dados históricos suficientes para cálculos
        base_price = 50000.0
        for i in range(25):  # Dados suficientes para médias móveis
            test_data = {
                'timestamp': datetime.now() + timedelta(minutes=i),
                'open': base_price + i * 10,
                'high': base_price + 100 + i * 10,
                'low': base_price - 100 + i * 10,
                'close': base_price + 50 + i * 10,
                'volume': 100.0 + i * 10
            }
            engineer.add_market_data('BTCUSDT', test_data)

        # Calcular features
        features = engineer.calculate_features('BTCUSDT')

        assert features is not None
        assert 'close' in features
        assert 'volume' in features
        # Verificar algumas features que devem existir
        assert 'body_size' in features
        assert 'price_range' in features
        assert isinstance(features['close'], float)

    def test_feature_cache(self):
        """Testa funcionamento do cache de features."""
        engineer = RealtimeFeatureEngineer()

        # Adicionar dados suficientes
        base_price = 50000.0
        for i in range(25):
            test_data = {
                'timestamp': datetime.now() + timedelta(minutes=i),
                'open': base_price + i * 10,
                'high': base_price + 100 + i * 10,
                'low': base_price - 100 + i * 10,
                'close': base_price + 50 + i * 10,
                'volume': 100.0 + i * 10
            }
            engineer.add_market_data('BTCUSDT', test_data)

        # Primeira chamada - deve calcular
        features1 = engineer.calculate_features('BTCUSDT')
        stats1 = engineer.get_stats()

        # Segunda chamada - deve usar cache
        features2 = engineer.calculate_features('BTCUSDT')
        stats2 = engineer.get_stats()

        assert features1 == features2
        assert stats2['cache_hits'] > stats1['cache_hits']


class TestDataFeed:
    """Testes para DataFeed."""

    @patch('battle_arena.core.data_feed.BinanceWebSocketConnector')
    def test_data_feed_initialization(self, mock_ws_connector):
        """Testa inicialização do DataFeed."""
        symbols = ['BTCUSDT', 'ETHUSDT']

        feed = DataFeed(symbols=symbols, buffer_size=100)

        assert len(feed.data_buffers) == len(symbols)
        assert all(symbol in feed.data_buffers for symbol in symbols)

    def test_add_and_retrieve_data(self):
        """Testa adição e recuperação de dados."""
        feed = DataFeed(symbols=['BTCUSDT'], buffer_size=10)

        # Adicionar dados
        test_data = {
            'timestamp': datetime.now(),
            'open': 50000.0,
            'high': 51000.0,
            'low': 49000.0,
            'close': 50500.0,
            'volume': 100.0
        }

        # Simular adição via callback
        feed._add_to_buffer('BTCUSDT', test_data)

        # Recuperar dados
        latest = feed.get_latest_data('BTCUSDT', 1)

        assert len(latest) == 1
        assert latest[0]['close'] == 50500.0

    def test_buffer_limits(self):
        """Testa limites do buffer circular."""
        feed = DataFeed(symbols=['BTCUSDT'], buffer_size=3)

        # Adicionar mais dados que o buffer suporta
        for i in range(5):
            test_data = {
                'timestamp': datetime.now() + timedelta(minutes=i),
                'open': 50000.0 + i,
                'high': 51000.0 + i,
                'low': 49000.0 + i,
                'close': 50500.0 + i,
                'volume': 100.0 + i
            }
            feed._add_to_buffer('BTCUSDT', test_data)

        # Buffer deve ter no máximo 3 itens
        assert len(feed.data_buffers['BTCUSDT']) <= 3


class TestBattleController:
    """Testes para BattleController."""

    @patch('battle_arena.core.battle_controller.DataFeed')
    @patch('battle_arena.core.battle_controller.RealtimeFeatureEngineer')
    @patch('battle_arena.core.battle_controller.SignalGenerator')
    @patch('battle_arena.core.battle_controller.OrderEngine')
    @patch('battle_arena.core.battle_controller.PaperTrader')
    def test_initialization(self, mock_paper_trader, mock_order_engine,
                          mock_signal_gen, mock_feature_eng, mock_data_feed):
        """Testa inicialização do BattleController."""
        # Mock do ModelInfo
        model_info = ModelInfo(
            model_name='lightgbm',
            timestamp='20250926_094300',
            model_type='lightgbm',
            model=Mock(),
            metadata={},
            feature_names=['close', 'volume'],
            config={},
            metrics={'auc': 0.85}
        )

        config = {
            'execution': {'mode': 'paper'},
            'symbols': ['BTCUSDT'],
            'risk': {'max_position_size_pct': 0.1},
            'paper_trading': {'initial_capital': 10000.0}
        }

        controller = BattleController(config, model_info)

        # Tentar inicializar componentes
        result = controller.initialize_components()

        assert result is True
        assert controller.paper_trader is not None
        assert controller.order_engine is not None


if __name__ == '__main__':
    pytest.main([__file__])