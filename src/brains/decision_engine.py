import logging
from typing import Dict, Any
import joblib
import pandas as pd

from src.battle_arena.core.signal_generator import Signal

logger = logging.getLogger(__name__)

class DecisionEngine:
    def __init__(self, config: Dict[str, Any], approved_artifacts: list):
        self.config = config
        self.approved_artifacts = approved_artifacts
        self.models = {}  # Approved models will be loaded here, keyed by symbol
        self._load_approved_models()

    def _load_approved_models(self):
        for artifact_path in self.approved_artifacts:
            try:
                # Assuming artifact_path is something like 'outputs/models/BTCUSDT_model.joblib'
                # Handle both forward and backward slashes
                symbol = artifact_path.replace('\\', '/').split('/')[-1].split('_')[0]
                self.models[symbol] = joblib.load(artifact_path)
                logger.info(f"Modelo aprovado para {symbol} carregado de {artifact_path}")
            except Exception as e:
                logger.error(f"Erro ao carregar modelo de {artifact_path}: {e}")

    def load_approved_model(self, model_path: str):
        # This method might not be needed anymore if _load_approved_models handles everything
        # Keeping it for now, but it might be removed later.
        logger.warning("load_approved_model called, but models are loaded via _load_approved_models.")
        pass

    def make_decision(self, processed_features: Dict[str, Any]) -> Signal:
        from src.battle_arena.core.signal_generator import SignalType
        from datetime import datetime
        
        symbol = processed_features.get('symbol')
        if not symbol or symbol not in self.models:
            logger.warning(f"Nenhum modelo aprovado encontrado para o símbolo {symbol}. Retornando HOLD.")
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.HOLD, 
                confidence=0.0,
                probability=0.0,
                position_size=0.0,
                timestamp=datetime.now(),
                features={},
                model_info={}
            )

        model = self.models[symbol]
        # Ensure processed_features is in the correct format for the model
        # This might involve converting dict to DataFrame or specific feature vector
        # For now, assuming model.predict can handle the dict directly or needs a simple conversion
        try:
            # Assuming the model expects a single row DataFrame or a similar structure
            # You might need to adjust this based on how your models are trained
            prediction = model.predict(pd.DataFrame([processed_features]))[0]
            confidence = abs(prediction) # Example: confidence based on absolute prediction value
            probability = prediction

            if prediction > 0.5:  # Example threshold for a BUY signal
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.BUY, 
                    confidence=confidence,
                    probability=probability,
                    position_size=0.1,
                    timestamp=datetime.now(),
                    features=processed_features,
                    model_info={'prediction': prediction}
                )
            elif prediction < -0.5: # Example threshold for a SELL signal
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.SELL, 
                    confidence=confidence,
                    probability=probability,
                    position_size=0.1,
                    timestamp=datetime.now(),
                    features=processed_features,
                    model_info={'prediction': prediction}
                )
            else:
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.HOLD, 
                    confidence=confidence,
                    probability=probability,
                    position_size=0.0,
                    timestamp=datetime.now(),
                    features=processed_features,
                    model_info={'prediction': prediction}
                )
        except Exception as e:
            logger.error(f"Erro ao fazer previsão para {symbol}: {e}")
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.HOLD, 
                confidence=0.0,
                probability=0.0,
                position_size=0.0,
                timestamp=datetime.now(),
                features={},
                model_info={}
            )