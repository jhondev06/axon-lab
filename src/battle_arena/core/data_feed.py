"""
Real-time Data Feed
Gerencia conexões WebSocket e fornece dados OHLCV em tempo real para a Battle Arena.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque
import pandas as pd

from ..connectors.binance_connector import BinanceConnector
from binance_ws import BinanceWebSocketConnector


class DataFeed:
    """
    Feed de dados em tempo real para a Battle Arena.

    Gerencia conexões WebSocket e mantém buffer circular de dados históricos recentes.
    """

    def __init__(self, symbols: List[str] = None, interval: str = '1m',
                 buffer_size: int = 1000, rest_connector: Optional[BinanceConnector] = None):
        """
        Inicializa o DataFeed.

        Args:
            symbols: Lista de símbolos para monitorar
            interval: Intervalo dos dados (1m, 5m, etc.)
            buffer_size: Tamanho máximo do buffer circular
            rest_connector: Conector REST para dados históricos iniciais
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.interval = interval
        self.buffer_size = buffer_size
        self.rest_connector = rest_connector

        self.logger = logging.getLogger(self.__class__.__name__)

        # Buffer circular para cada símbolo
        self.data_buffers: Dict[str, deque] = {}
        for symbol in self.symbols:
            self.data_buffers[symbol] = deque(maxlen=buffer_size)

        # Conexões WebSocket ativas
        self.ws_connectors: Dict[str, BinanceWebSocketConnector] = {}
        self.ws_threads: Dict[str, threading.Thread] = {}

        # Callbacks para novos dados
        self.data_callbacks: List[Callable] = []

        # Estado
        self.is_running = False
        self.last_update = {}

        # Estatísticas
        self.stats = {
            'messages_received': 0,
            'data_points_stored': 0,
            'buffer_overflows': 0,
            'connection_errors': 0
        }

        self.logger.info(f"DataFeed inicializado para {len(self.symbols)} símbolos: {self.symbols}")

    def add_data_callback(self, callback: Callable) -> None:
        """
        Adiciona callback para ser chamado quando novos dados chegam.

        Args:
            callback: Função que recebe (symbol, ohlcv_data)
        """
        self.data_callbacks.append(callback)

    def start(self) -> None:
        """Inicia o feed de dados."""
        if self.is_running:
            self.logger.warning("DataFeed já está rodando")
            return

        self.is_running = True
        self.logger.info("Iniciando DataFeed...")

        # Carregar dados históricos iniciais se disponível
        self._load_historical_data()

        # Iniciar conexões WebSocket
        for symbol in self.symbols:
            self._start_symbol_feed(symbol)

        self.logger.info("DataFeed iniciado com sucesso")

    def stop(self) -> None:
        """Para o feed de dados."""
        if not self.is_running:
            return

        self.is_running = False
        self.logger.info("Parando DataFeed...")

        # Parar todas as conexões WebSocket
        for symbol, connector in self.ws_connectors.items():
            try:
                connector.stop()
                self.logger.info(f"WebSocket parado para {symbol}")
            except Exception as e:
                self.logger.error(f"Erro ao parar WebSocket para {symbol}: {e}")

        # Aguardar threads terminarem
        for symbol, thread in self.ws_threads.items():
            if thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    self.logger.warning(f"Thread para {symbol} não terminou no timeout")

        self.ws_connectors.clear()
        self.ws_threads.clear()

        self.logger.info("DataFeed parado")

    def get_latest_data(self, symbol: str, n_points: int = 100) -> List[Dict[str, Any]]:
        """
        Obtém os n pontos de dados mais recentes para um símbolo.

        Args:
            symbol: Par de trading
            n_points: Número de pontos para retornar

        Returns:
            Lista de dados OHLCV (mais recentes primeiro)
        """
        if symbol not in self.data_buffers:
            self.logger.warning(f"Símbolo não encontrado: {symbol}")
            return []

        buffer = self.data_buffers[symbol]
        data = list(buffer)[-n_points:] if len(buffer) >= n_points else list(buffer)

        return data

    def get_data_as_dataframe(self, symbol: str, n_points: int = 100) -> pd.DataFrame:
        """
        Obtém dados como DataFrame pandas.

        Args:
            symbol: Par de trading
            n_points: Número de pontos para retornar

        Returns:
            DataFrame com colunas OHLCV
        """
        data = self.get_latest_data(symbol, n_points)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def get_buffer_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre os buffers.

        Returns:
            Dicionário com informações dos buffers
        """
        info = {}
        for symbol, buffer in self.data_buffers.items():
            info[symbol] = {
                'buffer_size': len(buffer),
                'max_size': buffer.maxlen,
                'last_timestamp': buffer[-1]['timestamp'] if buffer else None,
                'last_update': self.last_update.get(symbol)
            }

        return info

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do DataFeed."""
        return self.stats.copy()

    def _start_symbol_feed(self, symbol: str) -> None:
        """Inicia feed WebSocket para um símbolo."""
        try:
            # Criar conector WebSocket
            ws_connector = BinanceWebSocketConnector(
                symbol=symbol,
                interval=self.interval,
                raw_dir='data/battle_arena/ws_data'
            )

            # Substituir handler de mensagens para integrar com DataFeed
            original_handler = ws_connector.handle_socket_message
            ws_connector.handle_socket_message = lambda msg: self._handle_ws_message(symbol, msg, original_handler)

            self.ws_connectors[symbol] = ws_connector

            # Iniciar em thread separada
            thread = threading.Thread(
                target=self._run_ws_connector,
                args=(symbol, ws_connector),
                name=f"WS-{symbol}",
                daemon=True
            )

            self.ws_threads[symbol] = thread
            thread.start()

            self.logger.info(f"Feed WebSocket iniciado para {symbol}")

        except Exception as e:
            self.logger.error(f"Erro ao iniciar feed para {symbol}: {e}")
            self.stats['connection_errors'] += 1

    def _run_ws_connector(self, symbol: str, connector: BinanceWebSocketConnector) -> None:
        """Executa conector WebSocket em thread separada."""
        try:
            connector.start()
        except Exception as e:
            self.logger.error(f"Erro na thread WebSocket para {symbol}: {e}")
            self.stats['connection_errors'] += 1

    def _handle_ws_message(self, symbol: str, msg: Dict[str, Any],
                          original_handler: Callable) -> None:
        """Processa mensagens WebSocket e atualiza buffers."""
        try:
            self.stats['messages_received'] += 1

            # Chamar handler original para salvar em NDJSON
            original_handler(msg)

            # Processar para DataFeed
            if msg.get('e') == 'kline' and msg.get('k', {}).get('x', False):  # Kline fechado
                kline = msg['k']

                # Converter para formato OHLCV
                ohlcv_data = {
                    'timestamp': datetime.fromtimestamp(kline['T'] / 1000),  # Close time
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'symbol': symbol,
                    'interval': kline['i'],
                    'received_at': datetime.now()
                }

                # Adicionar ao buffer
                self._add_to_buffer(symbol, ohlcv_data)

                # Notificar callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(symbol, ohlcv_data)
                    except Exception as e:
                        self.logger.error(f"Erro em callback: {e}")

        except Exception as e:
            self.logger.error(f"Erro ao processar mensagem WS para {symbol}: {e}")

    def _add_to_buffer(self, symbol: str, data: Dict[str, Any]) -> None:
        """Adiciona dados ao buffer circular."""
        if symbol not in self.data_buffers:
            self.logger.warning(f"Buffer não encontrado para {symbol}")
            return

        buffer = self.data_buffers[symbol]

        # Verificar se é duplicado (mesmo timestamp)
        if buffer and buffer[-1]['timestamp'] == data['timestamp']:
            # Substituir último dado se for mais recente
            if data['received_at'] > buffer[-1]['received_at']:
                buffer[-1] = data
                self.logger.debug(f"Dado atualizado para {symbol} @ {data['timestamp']}")
            return

        # Adicionar novo dado
        buffer.append(data)
        self.last_update[symbol] = datetime.now()
        self.stats['data_points_stored'] += 1

        # Verificar overflow
        if len(buffer) == buffer.maxlen:
            self.stats['buffer_overflows'] += 1

        self.logger.debug(f"Dado adicionado para {symbol}: {data['close']} @ {data['timestamp']}")

    def _load_historical_data(self) -> None:
        """Carrega dados históricos iniciais se disponível."""
        if not self.rest_connector:
            self.logger.info("Nenhum conector REST fornecido, pulando dados históricos")
            return

        try:
            self.logger.info("Carregando dados históricos iniciais...")

            # Carregar últimas 24h de dados para cada símbolo
            for symbol in self.symbols:
                try:
                    # Usar REST API para pegar dados históricos
                    # Nota: BinanceConnector pode precisar ser adaptado
                    historical_data = self._fetch_historical_klines(symbol, '1d', 1)  # Último dia

                    if historical_data:
                        # Converter e adicionar ao buffer
                        for kline in historical_data:
                            ohlcv_data = {
                                'timestamp': datetime.fromtimestamp(kline[6] / 1000),  # Close time
                                'open': float(kline[1]),
                                'high': float(kline[2]),
                                'low': float(kline[3]),
                                'close': float(kline[4]),
                                'volume': float(kline[5]),
                                'symbol': symbol,
                                'interval': self.interval,
                                'received_at': datetime.now()
                            }
                            self._add_to_buffer(symbol, ohlcv_data)

                        self.logger.info(f"Dados históricos carregados para {symbol}: {len(historical_data)} pontos")

                except Exception as e:
                    self.logger.error(f"Erro ao carregar dados históricos para {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"Erro geral ao carregar dados históricos: {e}")

    def _fetch_historical_klines(self, symbol: str, interval: str, limit: int) -> List[List]:
        """Busca dados históricos de klines (placeholder - implementar baseado no conector)."""
        # Placeholder - implementar baseado no BinanceConnector disponível
        # Por enquanto retorna lista vazia
        return []

    def force_refresh_symbol(self, symbol: str) -> bool:
        """
        Força refresh da conexão WebSocket para um símbolo.

        Args:
            symbol: Par de trading

        Returns:
            True se refresh foi bem-sucedido
        """
        if symbol not in self.ws_connectors:
            self.logger.warning(f"Conector WS não encontrado para {symbol}")
            return False

        try:
            # Parar conector atual
            self.ws_connectors[symbol].stop()

            # Aguardar um pouco
            time.sleep(2)

            # Reiniciar
            self._start_symbol_feed(symbol)

            self.logger.info(f"Conexão WebSocket refreshed para {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao refresh conexão para {symbol}: {e}")
            return False

    def get_symbol_status(self, symbol: str) -> Dict[str, Any]:
        """
        Retorna status detalhado de um símbolo.

        Args:
            symbol: Par de trading

        Returns:
            Dicionário com status
        """
        status = {
            'symbol': symbol,
            'buffer_size': len(self.data_buffers.get(symbol, [])),
            'last_update': self.last_update.get(symbol),
            'ws_connected': symbol in self.ws_connectors and self.ws_connectors[symbol].is_running,
            'thread_alive': symbol in self.ws_threads and self.ws_threads[symbol].is_alive()
        }

        if symbol in self.data_buffers and self.data_buffers[symbol]:
            latest = self.data_buffers[symbol][-1]
            status['latest_price'] = latest['close']
            status['latest_timestamp'] = latest['timestamp']

        return status

    def cleanup_old_data(self, max_age_hours: int = 24) -> None:
        """
        Remove dados antigos dos buffers.

        Args:
            max_age_hours: Idade máxima dos dados em horas
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        for symbol, buffer in self.data_buffers.items():
            # Filtrar dados recentes
            original_size = len(buffer)
            recent_data = deque(
                [data for data in buffer if data['timestamp'] >= cutoff_time],
                maxlen=buffer.maxlen
            )

            self.data_buffers[symbol] = recent_data

            removed = original_size - len(recent_data)
            if removed > 0:
                self.logger.info(f"Removidos {removed} dados antigos para {symbol}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()