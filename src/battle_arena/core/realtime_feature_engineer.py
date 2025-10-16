"""
Realtime Feature Engineer
Calcula features em tempo real para a Battle Arena usando dados OHLCV live.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict


class RealtimeFeatureEngineer:
    """
    Engine de features em tempo real.

    Calcula features incrementalmente à medida que novos dados OHLCV chegam,
    mantendo cache para eficiência computacional.
    """

    def __init__(self, feature_config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o RealtimeFeatureEngineer.

        Args:
            feature_config: Configuração de features (similar ao AXON)
        """
        self.feature_config = feature_config or self._get_default_config()

        self.logger = logging.getLogger(self.__class__.__name__)

        # Cache de dados históricos por símbolo
        self.data_cache: Dict[str, pd.DataFrame] = {}

        # Cache de features calculadas
        self.feature_cache: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Último timestamp processado por símbolo
        self.last_timestamp: Dict[str, datetime] = {}

        # Estatísticas
        self.stats = {
            'features_calculated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }

        self.logger.info("RealtimeFeatureEngineer inicializado")

    def _get_default_config(self) -> Dict[str, Any]:
        """Retorna configuração padrão de features."""
        return {
            'returns': [1, 5, 15, 60],  # minutos
            'ema_periods': [5, 20, 50],
            'rsi_periods': [7, 14, 21],
            'imbalance_windows': [5, 10, 20],
            'bollinger_period': 20,
            'bollinger_std': 2,
            'volatility_windows': [5, 20],
            'volume_windows': [5, 20]
        }

    def add_market_data(self, symbol: str, ohlcv_data: Dict[str, Any]) -> None:
        """
        Adiciona dados de mercado e atualiza cache.

        Args:
            symbol: Par de trading
            ohlcv_data: Dados OHLCV (timestamp, open, high, low, close, volume)
        """
        try:
            # Verificar se já processamos este timestamp
            timestamp = ohlcv_data['timestamp']
            if symbol in self.last_timestamp and timestamp <= self.last_timestamp[symbol]:
                return  # Já processado

            # Adicionar ao cache de dados
            if symbol not in self.data_cache:
                # Criar DataFrame inicial
                self.data_cache[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Criar nova linha
            new_row = pd.DataFrame([{
                'timestamp': timestamp,
                'open': ohlcv_data['open'],
                'high': ohlcv_data['high'],
                'low': ohlcv_data['low'],
                'close': ohlcv_data['close'],
                'volume': ohlcv_data['volume']
            }])

            # Adicionar ao DataFrame
            if self.data_cache[symbol].empty:
                self.data_cache[symbol] = new_row
            else:
                self.data_cache[symbol] = pd.concat([self.data_cache[symbol], new_row], ignore_index=True)

            # Manter apenas últimas 200 linhas para eficiência
            if len(self.data_cache[symbol]) > 200:
                self.data_cache[symbol] = self.data_cache[symbol].tail(150).reset_index(drop=True)

            # Atualizar timestamp
            self.last_timestamp[symbol] = timestamp

            # Limpar cache de features (invalido após novos dados)
            if symbol in self.feature_cache:
                self.feature_cache[symbol].clear()

        except Exception as e:
            self.logger.error(f"Erro ao adicionar dados para {symbol}: {e}")
            self.stats['errors'] += 1

    def calculate_features(self, symbol: str) -> Dict[str, float]:
        """
        Calcula features para um símbolo.

        Args:
            symbol: Par de trading

        Returns:
            Dicionário com features calculadas
        """
        try:
            # Verificar cache
            if symbol in self.feature_cache and self.feature_cache[symbol]:
                self.stats['cache_hits'] += 1
                return self.feature_cache[symbol].copy()

            self.stats['cache_misses'] += 1

            # Verificar se temos dados suficientes
            if symbol not in self.data_cache or len(self.data_cache[symbol]) < 10:
                self.logger.debug(f"Dados insuficientes para {symbol}")
                return {}

            df = self.data_cache[symbol].copy()

            # Calcular features
            features = {}

            # Features básicas
            latest = df.iloc[-1]
            features['open'] = latest['open']
            features['high'] = latest['high']
            features['low'] = latest['low']
            features['close'] = latest['close']
            features['volume'] = latest['volume']

            # Calcular features técnicas
            features.update(self._calculate_price_features(df))
            features.update(self._calculate_volume_features(df))
            features.update(self._calculate_momentum_features(df))
            features.update(self._calculate_volatility_features(df))
            features.update(self._calculate_imbalance_features(df))

            # Limpar NaNs e infs
            features = self._clean_features(features)

            # Cache das features
            self.feature_cache[symbol] = features.copy()

            self.stats['features_calculated'] += 1

            return features

        except Exception as e:
            self.logger.error(f"Erro ao calcular features para {symbol}: {e}")
            self.stats['errors'] += 1
            return {}

    def _calculate_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula features baseadas em preço."""
        features = {}

        try:
            # Retornos
            for period in self.feature_config['returns']:
                if len(df) > period:
                    ret = df['close'].pct_change(period).iloc[-1]
                    features[f'ret_{period}m'] = ret

            # EMAs
            for period in self.feature_config['ema_periods']:
                if len(df) >= period:
                    ema = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
                    features[f'ema_{period}'] = ema

                    # Ratio preço/EMA
                    features[f'price_ema{period}_ratio'] = df['close'].iloc[-1] / ema if ema > 0 else 1.0

            # EMA ratios
            if 'ema_5' in features and 'ema_20' in features:
                features['ema5_ema20_ratio'] = features['ema_5'] / features['ema_20'] if features['ema_20'] > 0 else 1.0

            # RSI
            for period in self.feature_config['rsi_periods']:
                if len(df) >= period + 1:
                    rsi = self._calculate_rsi(df['close'], period)
                    features[f'rsi_{period}'] = rsi

            # Bollinger Bands
            bb_period = self.feature_config['bollinger_period']
            bb_std = self.feature_config['bollinger_std']
            if len(df) >= bb_period:
                bb_ma = df['close'].rolling(bb_period).mean().iloc[-1]
                bb_std_val = df['close'].rolling(bb_period).std().iloc[-1]

                features['bb_upper'] = bb_ma + (bb_std_val * bb_std)
                features['bb_lower'] = bb_ma - (bb_std_val * bb_std)
                features['bb_position'] = (df['close'].iloc[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']) if (features['bb_upper'] - features['bb_lower']) > 0 else 0.5

            # Price action
            features['high_low_ratio'] = df['high'].iloc[-1] / df['low'].iloc[-1] if df['low'].iloc[-1] > 0 else 1.0
            features['open_close_ratio'] = df['open'].iloc[-1] / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 1.0

            # Range do candle
            features['price_range'] = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['low'].iloc[-1] if df['low'].iloc[-1] > 0 else 0.0
            features['body_size'] = abs(df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1] if df['open'].iloc[-1] > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Erro ao calcular price features: {e}")

        return features

    def _calculate_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula features baseadas em volume."""
        features = {}

        try:
            # Volume MA
            for window in self.feature_config['volume_windows']:
                if len(df) >= window:
                    vol_ma = df['volume'].rolling(window).mean().iloc[-1]
                    features[f'volume_ma_{window}'] = vol_ma
                    features[f'volume_ratio_{window}'] = df['volume'].iloc[-1] / vol_ma if vol_ma > 0 else 1.0

            # Volume momentum
            for window in self.feature_config['volume_windows']:
                if len(df) >= window:
                    vol_change = df['volume'].pct_change(window).iloc[-1]
                    features[f'volume_momentum_{window}'] = vol_change

        except Exception as e:
            self.logger.error(f"Erro ao calcular volume features: {e}")

        return features

    def _calculate_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula features de momentum."""
        features = {}

        try:
            # Retorno logarítmico
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1)).iloc[-1]

            # Momentum horário
            if len(df) >= 60:  # Última hora
                hourly_ret = (df['close'].iloc[-1] / df['close'].iloc[-60]) - 1
                features['momentum_1h'] = hourly_ret

        except Exception as e:
            self.logger.error(f"Erro ao calcular momentum features: {e}")

        return features

    def _calculate_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula features de volatilidade."""
        features = {}

        try:
            # Volatilidade em diferentes janelas
            for window in self.feature_config['volatility_windows']:
                if len(df) >= window:
                    vol = df['close'].pct_change().rolling(window).std().iloc[-1]
                    features[f'volatility_{window}m'] = vol

            # Ratio de volatilidade
            if 'volatility_5m' in features and 'volatility_20m' in features:
                features['volatility_ratio'] = features['volatility_5m'] / features['volatility_20m'] if features['volatility_20m'] > 0 else 1.0

        except Exception as e:
            self.logger.error(f"Erro ao calcular volatility features: {e}")

        return features

    def _calculate_imbalance_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula features de order book imbalance (aproximado)."""
        features = {}

        try:
            for window in self.feature_config['imbalance_windows']:
                if len(df) >= window:
                    # VWAP aproximado
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    vwap = (typical_price * df['volume']).rolling(window).sum().iloc[-1] / df['volume'].rolling(window).sum().iloc[-1]
                    features[f'vwap_{window}'] = vwap

                    # Price vs VWAP
                    features[f'price_vwap_ratio_{window}'] = df['close'].iloc[-1] / vwap if vwap > 0 else 1.0

                    # Volume momentum
                    price_change = df['close'].pct_change()
                    volume_momentum = (price_change * df['volume']).rolling(window).sum().iloc[-1]
                    features[f'volume_momentum_{window}'] = volume_momentum

                    # Intraday strength
                    high_low_range = df['high'] - df['low']
                    close_position = (df['close'] - df['low']) / high_low_range.replace(0, np.nan)
                    features[f'intraday_strength_{window}'] = close_position.rolling(window).mean().iloc[-1]

                    # Volume-adjusted returns
                    vol_adj_returns = df['close'].pct_change() * np.log1p(df['volume'] / df['volume'].rolling(window).mean())
                    features[f'vol_adj_returns_{window}'] = vol_adj_returns.rolling(window).mean().iloc[-1]

        except Exception as e:
            self.logger.error(f"Erro ao calcular imbalance features: {e}")

        return features

    def _calculate_rsi(self, series: pd.Series, window: int) -> float:
        """Calcula RSI."""
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

        except Exception:
            return 50.0

    def _clean_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Limpa features removendo NaNs e valores infinitos."""
        cleaned = {}

        for key, value in features.items():
            if pd.isna(value) or np.isinf(value):
                cleaned[key] = 0.0
            else:
                cleaned[key] = float(value)

        return cleaned

    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes de features que podem ser calculadas."""
        # Simular cálculo com dados dummy para descobrir nomes
        dummy_data = {
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000.0
        }

        # Criar DataFrame dummy com dados suficientes
        dummy_df = pd.DataFrame([dummy_data] * 50)

        # Calcular features
        dummy_features = self._calculate_price_features(dummy_df)
        dummy_features.update(self._calculate_volume_features(dummy_df))
        dummy_features.update(self._calculate_momentum_features(dummy_df))
        dummy_features.update(self._calculate_volatility_features(dummy_df))
        dummy_features.update(self._calculate_imbalance_features(dummy_df))

        return list(dummy_features.keys())

    def get_cache_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o cache."""
        return {
            'symbols_cached': list(self.data_cache.keys()),
            'feature_cache_size': len(self.feature_cache),
            'stats': self.stats.copy()
        }

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Limpa cache de dados e features.

        Args:
            symbol: Símbolo específico ou None para todos
        """
        if symbol:
            if symbol in self.data_cache:
                del self.data_cache[symbol]
            if symbol in self.feature_cache:
                del self.feature_cache[symbol]
            if symbol in self.last_timestamp:
                del self.last_timestamp[symbol]
            self.logger.info(f"Cache limpo para {symbol}")
        else:
            self.data_cache.clear()
            self.feature_cache.clear()
            self.last_timestamp.clear()
            self.logger.info("Todo cache limpo")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance."""
        return self.stats.copy()