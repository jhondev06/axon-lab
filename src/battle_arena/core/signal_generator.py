"""
Signal Generator
Gera sinais de trading em tempo real combinando modelos e features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .model_loader import ModelInfo


class SignalType(Enum):
    """Tipos de sinal."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Sinal de trading gerado."""
    symbol: str
    signal_type: SignalType
    confidence: float
    probability: float
    position_size: float
    timestamp: datetime
    features: Dict[str, float]
    model_info: Dict[str, Any]
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte sinal para dicionário."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'probability': self.probability,
            'position_size': self.position_size,
            'timestamp': self.timestamp.isoformat(),
            'features': self.features,
            'model_info': self.model_info,
            'metadata': self.metadata or {}
        }


class SignalGenerator:
    """
    Gera sinais de trading combinando modelos ML e features em tempo real.

    Suporte a thresholds configuráveis e position sizing baseado em confidence.
    """

    def __init__(self, model_info: ModelInfo,
                 buy_threshold: float = 0.6,
                 sell_threshold: float = 0.4,
                 min_confidence: float = 0.55,
                 max_position_size: float = 0.1,
                 feature_window: int = 50):
        """
        Inicializa o SignalGenerator.

        Args:
            model_info: Informações do modelo carregado
            buy_threshold: Threshold para sinal BUY (probabilidade > threshold)
            sell_threshold: Threshold para sinal SELL (probabilidade < threshold)
            min_confidence: Confiança mínima para gerar sinal
            max_position_size: Tamanho máximo da posição
            feature_window: Janela de dados para cálculo de features
        """
        self.model_info = model_info
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_confidence = min_confidence
        self.max_position_size = max_position_size
        self.feature_window = feature_window

        self.logger = logging.getLogger(self.__class__.__name__)

        # Buffer circular para dados históricos recentes
        self.data_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = feature_window * 2  # Buffer maior que a janela

        # Estatísticas de sinais
        self.signal_stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'last_signal': None
        }

        self.logger.info(f"SignalGenerator inicializado com modelo {model_info.model_name}")
        self.logger.info(f"Thresholds: BUY>{buy_threshold}, SELL<{sell_threshold}")

    def add_market_data(self, symbol: str, ohlcv_data: Dict[str, Any]) -> None:
        """
        Adiciona dados de mercado ao buffer.

        Args:
            symbol: Par de trading
            ohlcv_data: Dados OHLCV (open, high, low, close, volume, timestamp)
        """
        # Validar dados obrigatórios
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(field in ohlcv_data for field in required_fields):
            self.logger.warning(f"Dados OHLCV incompletos: {list(ohlcv_data.keys())}")
            return

        # Adicionar ao buffer
        data_point = {
            'symbol': symbol,
            **ohlcv_data
        }

        self.data_buffer.append(data_point)

        # Manter tamanho máximo do buffer
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)

        self.logger.debug(f"Adicionado dados para {symbol} @ {ohlcv_data['timestamp']}")

    def generate_signal(self, symbol: str) -> Optional[Signal]:
        """
        Gera sinal para um símbolo baseado nos dados atuais.

        Args:
            symbol: Par de trading

        Returns:
            Signal se gerado, None caso contrário
        """
        try:
            # Verificar se temos dados suficientes
            symbol_data = [d for d in self.data_buffer if d['symbol'] == symbol]
            if len(symbol_data) < self.feature_window:
                self.logger.debug(f"Dados insuficientes para {symbol}: {len(symbol_data)} < {self.feature_window}")
                return None

            # Pegar dados mais recentes
            recent_data = symbol_data[-self.feature_window:]
            df = pd.DataFrame(recent_data)

            # Calcular features
            features = self._calculate_features(df)
            if features is None:
                return None

            # Preparar dados para o modelo
            feature_vector = self._prepare_feature_vector(features)
            if feature_vector is None:
                return None

            # Fazer predição
            prediction = self._predict_signal(feature_vector)

            # Gerar sinal baseado na predição
            signal = self._create_signal(symbol, prediction, features, feature_vector)

            # Atualizar estatísticas
            if signal:
                self._update_signal_stats(signal)

            return signal

        except Exception as e:
            self.logger.error(f"Erro ao gerar sinal para {symbol}: {e}")
            return None

    def _calculate_features(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calcula features baseado nos dados OHLCV."""
        try:
            # Garantir que temos as colunas necessárias
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Colunas faltando no DataFrame: {df.columns.tolist()}")
                return None

            features = {}

            # Features básicas
            latest = df.iloc[-1]
            features['open'] = latest['open']
            features['high'] = latest['high']
            features['low'] = latest['low']
            features['close'] = latest['close']
            features['volume'] = latest['volume']

            # Retornos
            returns_series = df['close'].pct_change()
            features['returns'] = returns_series.iloc[-1] if not pd.isna(returns_series.iloc[-1]) else 0.0

            log_returns = np.log(df['close'] / df['close'].shift(1))
            features['log_returns'] = log_returns.iloc[-1] if not pd.isna(log_returns.iloc[-1]) else 0.0

            # Range do candle
            features['price_range'] = (latest['high'] - latest['low']) / latest['low']
            features['body_size'] = abs(latest['close'] - latest['open']) / latest['open']

            # Volume features
            features['volume_ma_5'] = df['volume'].rolling(5).mean().iloc[-1]
            features['volume_ratio'] = latest['volume'] / features['volume_ma_5'] if features['volume_ma_5'] > 0 else 1.0

            # Hora do dia (se timestamp disponível)
            if 'timestamp' in df.columns:
                try:
                    ts = pd.to_datetime(df['timestamp'].iloc[-1])
                    features['hour'] = ts.hour
                    features['day_of_week'] = ts.dayofweek
                    features['is_weekend'] = 1 if ts.dayofweek >= 5 else 0
                except:
                    features['hour'] = 12  # default
                    features['day_of_week'] = 0
                    features['is_weekend'] = 0

            # Retornos em diferentes períodos
            for period in [1, 5, 15, 60]:
                if len(df) >= period:
                    features[f'ret_{period}m'] = df['close'].pct_change(period).iloc[-1]
                else:
                    features[f'ret_{period}m'] = 0.0

            # EMAs
            for period in [5, 20, 50]:
                if len(df) >= period:
                    ema = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
                    features[f'ema_{period}'] = ema
                    features[f'price_ema{period}_ratio'] = latest['close'] / ema if ema > 0 else 1.0
                else:
                    features[f'ema_{period}'] = latest['close']
                    features[f'price_ema{period}_ratio'] = 1.0

            # EMA ratios
            if 'ema_5' in features and 'ema_20' in features:
                features['ema5_ema20_ratio'] = features['ema_5'] / features['ema_20'] if features['ema_20'] > 0 else 1.0

            # RSI
            for period in [7, 14, 21]:
                if len(df) >= period + 1:  # +1 para cálculo de delta
                    rsi = self._calculate_rsi(df['close'], period)
                    features[f'rsi_{period}'] = rsi
                else:
                    features[f'rsi_{period}'] = 50.0  # neutro

            # VWAP e features de imbalance (simplificado)
            for window in [5, 10, 20]:
                if len(df) >= window:
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    vwap = (typical_price * df['volume']).rolling(window).sum().iloc[-1] / df['volume'].rolling(window).sum().iloc[-1]
                    features[f'vwap_{window}'] = vwap
                    features[f'price_vwap_ratio_{window}'] = latest['close'] / vwap if vwap > 0 else 1.0

                    # Volume momentum
                    price_change = df['close'].pct_change()
                    volume_momentum = (price_change * df['volume']).rolling(window).sum().iloc[-1]
                    features[f'volume_momentum_{window}'] = volume_momentum

                    # Intraday strength
                    high_low_range = df['high'] - df['low']
                    close_position = (df['close'] - df['low']) / high_low_range.replace(0, np.nan)
                    features[f'intraday_strength_{window}'] = close_position.rolling(window).mean().iloc[-1]

                    # Volume-adjusted returns (usar returns já calculado)
                    if 'returns' in features:
                        vol_adj_returns = features['returns'] * np.log1p(df['volume'] / df['volume'].rolling(window).mean())
                        features[f'vol_adj_returns_{window}'] = vol_adj_returns.rolling(window).mean().iloc[-1]
                    else:
                        features[f'vol_adj_returns_{window}'] = 0.0

            # Bollinger Bands
            if len(df) >= 20:
                bb_window = 20
                bb_std = 2
                bb_ma = df['close'].rolling(bb_window).mean().iloc[-1]
                bb_std_val = df['close'].rolling(bb_window).std().iloc[-1]
                features['bb_upper'] = bb_ma + (bb_std_val * bb_std)
                features['bb_lower'] = bb_ma - (bb_std_val * bb_std)
                features['bb_position'] = (latest['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']) if (features['bb_upper'] - features['bb_lower']) > 0 else 0.5

            # Volatilidade
            for window in [5, 20]:
                if len(df) >= window:
                    returns_series = df['close'].pct_change()
                    vol = returns_series.rolling(window).std().iloc[-1]
                    features[f'volatility_{window}m'] = vol if not pd.isna(vol) else 0.0

            if 'volatility_5m' in features and 'volatility_20m' in features:
                features['volatility_ratio'] = features['volatility_5m'] / features['volatility_20m'] if features['volatility_20m'] > 0 else 1.0

            # Volume SMA
            if len(df) >= 20:
                features['volume_sma_20'] = df['volume'].rolling(20).mean().iloc[-1]
                features['volume_ratio_20'] = latest['volume'] / features['volume_sma_20'] if features['volume_sma_20'] > 0 else 1.0

            # Price action ratios
            features['high_low_ratio'] = latest['high'] / latest['low'] if latest['low'] > 0 else 1.0
            features['open_close_ratio'] = latest['open'] / latest['close'] if latest['close'] > 0 else 1.0

            # Limpar NaNs
            for key, value in features.items():
                if pd.isna(value) or np.isinf(value):
                    features[key] = 0.0

            return features

        except Exception as e:
            self.logger.error(f"Erro ao calcular features: {e}")
            return None

    def _calculate_rsi(self, series: pd.Series, window: int) -> float:
        """Calcula RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def _prepare_feature_vector(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Prepara vetor de features para o modelo."""
        try:
            # Filtrar apenas features que o modelo conhece
            model_features = []
            for feature_name in self.model_info.feature_names:
                if feature_name in features:
                    model_features.append(features[feature_name])
                else:
                    self.logger.warning(f"Feature faltando: {feature_name}")
                    model_features.append(0.0)  # default

            if len(model_features) != len(self.model_info.feature_names):
                self.logger.error(f"Número de features incompatível: {len(model_features)} vs {len(self.model_info.feature_names)}")
                return None

            return np.array(model_features).reshape(1, -1)

        except Exception as e:
            self.logger.error(f"Erro ao preparar feature vector: {e}")
            return None

    def _predict_signal(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Faz predição usando o modelo."""
        try:
            # Predição de probabilidade
            if hasattr(self.model_info.model, 'predict_proba'):
                probabilities = self.model_info.model.predict_proba(feature_vector)[0]
                # Assumindo classe positiva (1) como "comprar"
                prob_buy = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                prob_sell = 1 - prob_buy
            else:
                # Fallback para predict
                prediction = self.model_info.model.predict(feature_vector)[0]
                prob_buy = float(prediction)
                prob_sell = 1 - prob_buy

            return {
                'prob_buy': prob_buy,
                'prob_sell': prob_sell,
                'confidence': max(prob_buy, prob_sell)
            }

        except Exception as e:
            self.logger.error(f"Erro na predição: {e}")
            return {'prob_buy': 0.5, 'prob_sell': 0.5, 'confidence': 0.5}

    def _create_signal(self, symbol: str, prediction: Dict[str, float],
                      features: Dict[str, float], feature_vector: np.ndarray) -> Optional[Signal]:
        """Cria sinal baseado na predição."""
        try:
            prob_buy = prediction['prob_buy']
            prob_sell = prediction['prob_sell']
            confidence = prediction['confidence']

            # Determinar tipo de sinal
            if confidence < self.min_confidence:
                signal_type = SignalType.HOLD
            elif prob_buy > self.buy_threshold:
                signal_type = SignalType.BUY
            elif prob_sell > (1 - self.sell_threshold):
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD

            # Calcular position size baseado na confidence
            if signal_type in [SignalType.BUY, SignalType.SELL]:
                # Position size proporcional à confidence acima do threshold
                confidence_above_threshold = confidence - self.min_confidence
                max_additional_confidence = 1.0 - self.min_confidence
                position_size = self.max_position_size * (confidence_above_threshold / max_additional_confidence)
                position_size = min(position_size, self.max_position_size)
            else:
                position_size = 0.0

            # Metadata adicional
            metadata = {
                'prob_buy': prob_buy,
                'prob_sell': prob_sell,
                'thresholds': {
                    'buy': self.buy_threshold,
                    'sell': self.sell_threshold,
                    'min_confidence': self.min_confidence
                },
                'feature_count': len(feature_vector[0]),
                'model_version': self.model_info.timestamp
            }

            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                probability=prob_buy if signal_type == SignalType.BUY else prob_sell,
                position_size=position_size,
                timestamp=datetime.now(),
                features=features,
                model_info={
                    'name': self.model_info.model_name,
                    'type': self.model_info.model_type,
                    'metrics': self.model_info.metrics
                },
                metadata=metadata
            )

            return signal

        except Exception as e:
            self.logger.error(f"Erro ao criar sinal: {e}")
            return None

    def _update_signal_stats(self, signal: Signal) -> None:
        """Atualiza estatísticas de sinais."""
        self.signal_stats['total_signals'] += 1

        if signal.signal_type == SignalType.BUY:
            self.signal_stats['buy_signals'] += 1
        elif signal.signal_type == SignalType.SELL:
            self.signal_stats['sell_signals'] += 1
        else:
            self.signal_stats['hold_signals'] += 1

        # Atualizar média de confidence
        total_signals = self.signal_stats['total_signals']
        current_avg = self.signal_stats['avg_confidence']
        self.signal_stats['avg_confidence'] = (current_avg * (total_signals - 1) + signal.confidence) / total_signals

        self.signal_stats['last_signal'] = signal.timestamp.isoformat()

    def get_signal_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de sinais."""
        return self.signal_stats.copy()

    def reset_signal_stats(self) -> None:
        """Reseta estatísticas de sinais."""
        self.signal_stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'avg_confidence': 0.0,
            'last_signal': None
        }

    def clear_data_buffer(self) -> None:
        """Limpa o buffer de dados."""
        self.data_buffer.clear()
        self.logger.info("Buffer de dados limpo")