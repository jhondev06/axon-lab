import pytest
import os
import joblib
import pandas as pd
from unittest.mock import MagicMock, patch

from src.brains.decision_engine import DecisionEngine
from src.battle_arena.core.signal_generator import Signal

# Mock para o modelo
class MockModel:
    def predict(self, features):
        # Simula uma previsão simples
        # Se a feature 'mock_prediction_buy' for True, retorna 0.6 (BUY)
        # Se a feature 'mock_prediction_sell' for True, retorna -0.6 (SELL)
        # Caso contrário, retorna 0.0 (HOLD)
        if 'mock_prediction_buy' in features.columns and features['mock_prediction_buy'].iloc[0]:
            return [0.6]
        if 'mock_prediction_sell' in features.columns and features['mock_prediction_sell'].iloc[0]:
            return [-0.6]
        return [0.0]

@pytest.fixture
def mock_config():
    return {
        "features": {
            "use": ["mock_feature_1", "mock_feature_2"]
        }
    }

@pytest.fixture
def mock_approved_artifacts(tmp_path):
    # Cria arquivos de modelo mock para diferentes símbolos
    model_paths = []
    symbols = ["BTCUSDT", "ETHUSDT"]
    for symbol in symbols:
        model_path = tmp_path / f"outputs/models/{symbol}_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(MockModel(), model_path)
        model_paths.append(str(model_path))
    return model_paths

def test_decision_engine_initialization_and_model_loading(mock_config, mock_approved_artifacts):
    with patch('joblib.load', side_effect=lambda x: MockModel()) as mock_joblib_load:
        engine = DecisionEngine(mock_config, mock_approved_artifacts)
        assert len(engine.models) == len(mock_approved_artifacts)
        assert "BTCUSDT" in engine.models
        assert "ETHUSDT" in engine.models
        mock_joblib_load.assert_called()

def test_make_decision_buy_signal(mock_config, mock_approved_artifacts):
    from src.battle_arena.core.signal_generator import SignalType
    
    with patch('joblib.load', side_effect=lambda x: MockModel()):
        engine = DecisionEngine(mock_config, mock_approved_artifacts)
        processed_features = {
            'symbol': 'BTCUSDT',
            'close': 100.0,
            'mock_prediction_buy': True
        }
        signal = engine.make_decision(processed_features)
        assert signal.signal_type == SignalType.BUY
        assert signal.symbol == "BTCUSDT"
        assert signal.confidence == 0.6

def test_make_decision_sell_signal(mock_config, mock_approved_artifacts):
    from src.battle_arena.core.signal_generator import SignalType
    
    with patch('joblib.load', side_effect=lambda x: MockModel()):
        engine = DecisionEngine(mock_config, mock_approved_artifacts)
        processed_features = {
            'symbol': 'ETHUSDT',
            'close': 2000.0,
            'mock_prediction_sell': True
        }
        signal = engine.make_decision(processed_features)
        assert signal.signal_type == SignalType.SELL
        assert signal.symbol == "ETHUSDT"
        assert signal.confidence == 0.6

def test_make_decision_hold_signal(mock_config, mock_approved_artifacts):
    from src.battle_arena.core.signal_generator import SignalType
    
    with patch('joblib.load', side_effect=lambda x: MockModel()):
        engine = DecisionEngine(mock_config, mock_approved_artifacts)
        processed_features = {
            'symbol': 'BTCUSDT',
            'close': 100.0,
            'mock_prediction_buy': False,
            'mock_prediction_sell': False
        }
        signal = engine.make_decision(processed_features)
        assert signal.signal_type == SignalType.HOLD
        assert signal.symbol == "BTCUSDT"
        assert signal.confidence == 0.0

def test_make_decision_no_model_for_symbol(mock_config, mock_approved_artifacts):
    from src.battle_arena.core.signal_generator import SignalType
    
    engine = DecisionEngine(mock_config, mock_approved_artifacts)
    processed_features = {
        'symbol': 'LTCUSDT',
        'close': 50.0
    }
    signal = engine.make_decision(processed_features)
    assert signal.signal_type == SignalType.HOLD
    assert signal.symbol == "LTCUSDT"
    assert signal.confidence == 0.0