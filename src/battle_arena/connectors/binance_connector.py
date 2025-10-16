"""
Binance Connector

Conector para integração com Binance API (spot trading).
Suporte a testnet para paper trading e WebSocket para dados em tempo real.
"""

import time
import json
import logging
import asyncio
import websockets
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from threading import Thread
import traceback

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from .base_connector import (
    BaseConnector, Order, Position, Balance,
    ExchangeError, ConnectionError, AuthenticationError,
    InsufficientFundsError, RateLimitError
)


class BinanceConnector(BaseConnector):
    """
    Conector para Binance API.

    Suporte a spot trading com testnet para simulação.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Inicializa o conector Binance.

        Args:
            api_key: Chave da API Binance
            api_secret: Segredo da API Binance
            testnet: Usar testnet (True) ou produção (False)
        """
        super().__init__(api_key, api_secret, testnet)

        # Configurar cliente Binance
        try:
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )

            # Testar conexão
            self._validate_credentials()
            self.logger.info(f"Conectado à Binance {'Testnet' if self.testnet else 'Produção'}")

        except Exception as e:
            self._handle_error(e, "inicialização do cliente Binance")

    def get_balance(self) -> List[Balance]:
        """
        Obtém saldos da conta Binance.

        Returns:
            Lista de objetos Balance
        """
        try:
            account_info = self.client.get_account()
            balances = []

            for balance in account_info['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked

                if total > 0:  # Só incluir ativos com saldo
                    balances.append(Balance(
                        asset=balance['asset'],
                        free=free,
                        locked=locked,
                        total=total
                    ))

            self._log_operation("get_balance", {"balances_count": len(balances)})
            return balances

        except (BinanceAPIException, BinanceRequestException) as e:
            self._handle_binance_error(e, "get_balance")

    def place_order(self, symbol: str, side: str, order_type: str,
                   quantity: float, price: Optional[float] = None) -> Order:
        """
        Coloca uma ordem no Binance.

        Args:
            symbol: Par de trading (ex: 'BTCUSDT')
            side: 'BUY' ou 'SELL'
            order_type: 'MARKET' ou 'LIMIT'
            quantity: Quantidade
            price: Preço (obrigatório para LIMIT orders)

        Returns:
            Objeto Order
        """
        try:
            # Preparar parâmetros da ordem
            order_params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': quantity
            }

            if order_type.upper() == 'LIMIT':
                if price is None:
                    raise ValueError("Preço obrigatório para ordens LIMIT")
                order_params['price'] = price
                order_params['timeInForce'] = 'GTC'  # Good Till Cancelled

            # Rate limiting básico
            time.sleep(0.1)

            # Executar ordem
            response = self.client.create_order(**order_params)

            # Converter resposta para objeto Order
            order = Order(
                order_id=str(response['orderId']),
                symbol=symbol,
                side=side.lower(),
                order_type=order_type.lower(),
                quantity=quantity,
                price=price,
                status=response['status'].lower(),
                timestamp=datetime.fromtimestamp(response['transactTime'] / 1000)
            )

            self._log_operation("place_order", {
                "order_id": order.order_id,
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "price": price
            })

            return order

        except (BinanceAPIException, BinanceRequestException) as e:
            self._handle_binance_error(e, "place_order")

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancela uma ordem no Binance.

        Args:
            order_id: ID da ordem
            symbol: Par de trading

        Returns:
            True se cancelada com sucesso
        """
        try:
            # Rate limiting
            time.sleep(0.1)

            response = self.client.cancel_order(
                symbol=symbol,
                orderId=order_id
            )

            success = response['status'] == 'CANCELED'
            self._log_operation("cancel_order", {
                "order_id": order_id,
                "symbol": symbol,
                "success": success
            })

            return success

        except (BinanceAPIException, BinanceRequestException) as e:
            self._handle_binance_error(e, "cancel_order")

    def get_positions(self) -> List[Position]:
        """
        Obtém posições abertas (para spot, posições são baseadas em ordens abertas).

        Para spot trading, posições são derivadas de ordens abertas e saldos.
        """
        try:
            # Para spot trading, não há posições abertas como em futures
            # Retornar lista vazia por enquanto
            # Futuramente pode ser implementado baseado em ordens abertas
            positions = []

            self._log_operation("get_positions", {"positions_count": len(positions)})
            return positions

        except (BinanceAPIException, BinanceRequestException) as e:
            self._handle_binance_error(e, "get_positions")

    def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """
        Obtém status de uma ordem.

        Args:
            order_id: ID da ordem
            symbol: Par de trading

        Returns:
            Objeto Order ou None se não encontrada
        """
        try:
            # Rate limiting
            time.sleep(0.1)

            response = self.client.get_order(
                symbol=symbol,
                orderId=order_id
            )

            order = Order(
                order_id=str(response['orderId']),
                symbol=symbol,
                side=response['side'].lower(),
                order_type=response['type'].lower(),
                quantity=float(response['origQty']),
                price=float(response['price']) if response['price'] else None,
                status=response['status'].lower(),
                timestamp=datetime.fromtimestamp(response['time'] / 1000)
            )

            self._log_operation("get_order_status", {
                "order_id": order_id,
                "symbol": symbol,
                "status": order.status
            })

            return order

        except BinanceAPIException as e:
            if e.code == -2013:  # Order does not exist
                self.logger.warning(f"Ordem não encontrada: {order_id}")
                return None
            self._handle_binance_error(e, "get_order_status")
        except BinanceRequestException as e:
            self._handle_binance_error(e, "get_order_status")

    def _handle_binance_error(self, error: Exception, operation: str) -> None:
        """
        Trata erros específicos do Binance.

        Args:
            error: Exceção Binance
            operation: Operação que falhou
        """
        if isinstance(error, BinanceAPIException):
            error_code = error.code
            error_msg = error.message

            # Mapear códigos de erro para tipos específicos
            if error_code == -2010:
                raise InsufficientFundsError(f"Saldo insuficiente: {error_msg}")
            elif error_code in [-2008, -1003]:
                raise RateLimitError(f"Limite de taxa excedido: {error_msg}")
            elif error_code in [-2014, -2015]:
                raise AuthenticationError(f"Erro de autenticação: {error_msg}")
            else:
                raise ExchangeError(f"Erro Binance API ({error_code}): {error_msg}")

        elif isinstance(error, BinanceRequestException):
            raise ConnectionError(f"Erro de conexão Binance: {str(error)}")

        else:
            # Fallback para tratamento genérico
            self._handle_error(error, operation)

    def get_ticker_price(self, symbol: str) -> float:
        """
        Obtém preço atual de um símbolo.

        Args:
            symbol: Par de trading

        Returns:
            Preço atual
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except (BinanceAPIException, BinanceRequestException) as e:
            self._handle_binance_error(e, "get_ticker_price")

    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Obtém informações da exchange.

        Returns:
            Informações da exchange
        """
        try:
            return self.client.get_exchange_info()
        except (BinanceAPIException, BinanceRequestException) as e:
            self._handle_binance_error(e, "get_exchange_info")


class BinanceWebSocketConnector:
    """
    Conector WebSocket para streaming de dados da Binance em tempo real.
    
    Suporte para klines (candlesticks) 1m e 5m com reconexão automática.
    """
    
    def __init__(self, testnet: bool = True, reconnect_interval: int = 5):
        """
        Inicializa o conector WebSocket.
        
        Args:
            testnet: Usar testnet (True) ou produção (False)
            reconnect_interval: Intervalo de reconexão em segundos
        """
        self.testnet = testnet
        self.reconnect_interval = reconnect_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # URLs WebSocket
        if testnet:
            self.ws_base_url = "wss://testnet.binance.vision/ws"
        else:
            self.ws_base_url = "wss://stream.binance.com:9443/ws"
        
        # Estado da conexão
        self.websocket = None
        self.is_connected = False
        self.should_reconnect = True
        self.subscriptions = {}  # symbol -> callback
        self.ws_thread = None
        
        # Normalização de símbolos e intervalos
        self.symbol_map = {}  # Para normalizar símbolos
        self.interval_map = {
            '1m': '1m',
            '5m': '5m',
            '1min': '1m',
            '5min': '5m'
        }
        
        self.logger.info(f"BinanceWebSocketConnector inicializado ({'testnet' if testnet else 'produção'})")
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normaliza símbolo para formato Binance.
        
        Args:
            symbol: Símbolo original (ex: 'BTC/USDT', 'btcusdt')
            
        Returns:
            Símbolo normalizado (ex: 'BTCUSDT')
        """
        # Remover barras e converter para maiúsculo
        normalized = symbol.replace('/', '').replace('-', '').upper()
        
        # Cache para evitar processamento repetido
        self.symbol_map[symbol] = normalized
        
        return normalized
    
    def normalize_interval(self, interval: str) -> str:
        """
        Normaliza intervalo para formato Binance.
        
        Args:
            interval: Intervalo original
            
        Returns:
            Intervalo normalizado
        """
        return self.interval_map.get(interval.lower(), interval)
    
    def subscribe_klines(self, symbol: str, interval: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Inscreve-se para receber klines de um símbolo.
        
        Args:
            symbol: Par de trading
            interval: Intervalo ('1m' ou '5m')
            callback: Função callback para processar dados
        """
        normalized_symbol = self.normalize_symbol(symbol)
        normalized_interval = self.normalize_interval(interval)
        
        # Validar intervalo suportado
        if normalized_interval not in ['1m', '5m']:
            raise ValueError(f"Intervalo não suportado: {interval}. Use '1m' ou '5m'")
        
        stream_name = f"{normalized_symbol.lower()}@kline_{normalized_interval}"
        self.subscriptions[stream_name] = {
            'symbol': normalized_symbol,
            'interval': normalized_interval,
            'callback': callback
        }
        
        self.logger.info(f"Inscrito para klines: {stream_name}")
        
        # Iniciar conexão se não estiver conectado
        if not self.is_connected:
            self.start()
    
    def unsubscribe_klines(self, symbol: str, interval: str):
        """
        Remove inscrição de klines.
        
        Args:
            symbol: Par de trading
            interval: Intervalo
        """
        normalized_symbol = self.normalize_symbol(symbol)
        normalized_interval = self.normalize_interval(interval)
        stream_name = f"{normalized_symbol.lower()}@kline_{normalized_interval}"
        
        if stream_name in self.subscriptions:
            del self.subscriptions[stream_name]
            self.logger.info(f"Removida inscrição: {stream_name}")
    
    def start(self):
        """Inicia a conexão WebSocket em thread separada."""
        if self.ws_thread and self.ws_thread.is_alive():
            self.logger.warning("WebSocket já está rodando")
            return
        
        self.should_reconnect = True
        self.ws_thread = Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()
        
        self.logger.info("WebSocket thread iniciada")
    
    def stop(self):
        """Para a conexão WebSocket."""
        self.should_reconnect = False
        self.is_connected = False
        
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        
        self.logger.info("WebSocket parado")
    
    def _run_websocket(self):
        """Executa o loop WebSocket com reconexão automática."""
        while self.should_reconnect:
            try:
                # Criar novo loop de eventos para esta thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Executar conexão WebSocket
                loop.run_until_complete(self._websocket_handler())
                
            except Exception as e:
                self.logger.error(f"Erro no WebSocket: {e}")
                self.logger.debug(traceback.format_exc())
                self.is_connected = False
                
                if self.should_reconnect:
                    self.logger.info(f"Reconectando em {self.reconnect_interval} segundos...")
                    time.sleep(self.reconnect_interval)
            
            finally:
                try:
                    loop.close()
                except:
                    pass
    
    async def _websocket_handler(self):
        """Handler principal do WebSocket."""
        if not self.subscriptions:
            self.logger.warning("Nenhuma inscrição ativa")
            return
        
        # Construir URL com streams
        streams = list(self.subscriptions.keys())
        if len(streams) == 1:
            ws_url = f"{self.ws_base_url}/{streams[0]}"
        else:
            # Múltiplos streams
            streams_param = '/'.join(streams)
            ws_url = f"{self.ws_base_url}/{streams_param}"
        
        self.logger.info(f"Conectando ao WebSocket: {ws_url}")
        
        async with websockets.connect(ws_url) as websocket:
            self.websocket = websocket
            self.is_connected = True
            self.logger.info("WebSocket conectado")
            
            try:
                async for message in websocket:
                    if not self.should_reconnect:
                        break
                    
                    await self._process_message(message)
                    
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("Conexão WebSocket fechada")
                self.is_connected = False
            except Exception as e:
                self.logger.error(f"Erro ao processar mensagem: {e}")
                raise
    
    async def _process_message(self, message: str):
        """
        Processa mensagem recebida do WebSocket.
        
        Args:
            message: Mensagem JSON do WebSocket
        """
        try:
            data = json.loads(message)
            
            # Verificar se é dados de kline
            if 'stream' in data and 'data' in data:
                stream = data['stream']
                kline_data = data['data']
                
                if stream in self.subscriptions and 'k' in kline_data:
                    subscription = self.subscriptions[stream]
                    kline = kline_data['k']
                    
                    # Converter para formato padronizado
                    processed_kline = self._format_kline_data(kline, subscription['symbol'])
                    
                    # Chamar callback
                    try:
                        subscription['callback'](processed_kline)
                    except Exception as e:
                        self.logger.error(f"Erro no callback para {stream}: {e}")
            
            elif 'k' in data:
                # Formato de stream único
                kline = data['k']
                symbol = kline['s']
                interval = kline['i']
                stream_name = f"{symbol.lower()}@kline_{interval}"
                
                if stream_name in self.subscriptions:
                    subscription = self.subscriptions[stream_name]
                    processed_kline = self._format_kline_data(kline, symbol)
                    
                    try:
                        subscription['callback'](processed_kline)
                    except Exception as e:
                        self.logger.error(f"Erro no callback para {stream_name}: {e}")
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Erro ao decodificar JSON: {e}")
        except Exception as e:
            self.logger.error(f"Erro ao processar mensagem: {e}")
    
    def _format_kline_data(self, kline: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Formata dados de kline para formato padronizado.
        
        Args:
            kline: Dados brutos do kline
            symbol: Símbolo do par
            
        Returns:
            Dados formatados compatíveis com _process_klines
        """
        return {
            'symbol': symbol,
            'open_time': int(kline['t']),
            'close_time': int(kline['T']),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'quote_volume': float(kline['q']),
            'trades': int(kline['n']),
            'taker_buy_base_volume': float(kline['V']),
            'taker_buy_quote_volume': float(kline['Q']),
            'interval': kline['i'],
            'is_closed': kline['x'],  # True se o kline está fechado
            'timestamp': datetime.fromtimestamp(int(kline['t']) / 1000)
        }
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Retorna status da conexão WebSocket.
        
        Returns:
            Dicionário com informações de status
        """
        return {
            'connected': self.is_connected,
            'subscriptions': len(self.subscriptions),
            'streams': list(self.subscriptions.keys()),
            'should_reconnect': self.should_reconnect,
            'testnet': self.testnet
        }