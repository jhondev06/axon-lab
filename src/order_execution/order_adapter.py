import logging
from typing import Dict, Any
from datetime import datetime
import uuid # Adicionado

from binance.client import Client
from binance.enums import * # Importa todos os enums para tipos de ordem, etc.

from axon.core.types import Signal
from axon.utils.config import ConfigManager

logger = logging.getLogger(__name__)

class OrderAdapter:
    def __init__(self, config: ConfigManager, exchange: str, testnet: bool = True):
        self.config = config
        self.exchange = exchange
        self.testnet = testnet
        self.api_key = self.config.get_config().get("connectors", {}).get(exchange, {}).get("api_key")
        self.api_secret = self.config.get_config().get("connectors", {}).get(exchange, {}).get("api_secret")

        if not self.api_key or not self.api_secret:
            logger.error(f"API Key ou API Secret não configurados para {exchange}.")
            self.client = None
        else:
            self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
            logger.info(f"OrderAdapter inicializado para {exchange} (Testnet: {self.testnet}).")

    def place_paper_order(self, signal: Signal) -> Dict[str, Any]:
        """Simula a colocação de uma ordem para paper trading."""
        logger.info(f"Simulando ordem para paper trading: {signal}")
        # Lógica de simulação de ordem aqui
        # Retornar um dicionário simulando a resposta da exchange
        return {
            "orderId": "PAPER_ORDER_" + str(uuid.uuid4()), # Usar UUID para IDs únicos
            "symbol": signal.symbol,
            "side": signal.signal_type.value, # Usar o tipo de sinal (BUY/SELL)
            "type": "MARKET", # Assumindo ordens de mercado para paper trading
            "price": signal.entry_price, # Preço de entrada do sinal
            "origQty": signal.position_size, # Quantidade do sinal
            "executedQty": signal.position_size, # Assumindo preenchimento total
            "status": "FILLED",
            "transactTime": datetime.now().timestamp() * 1000
        }

    def place_live_order(self, signal: Signal, order_type: str = "MARKET") -> Dict[str, Any]:
        """Coloca uma ordem real na exchange."""
        if not self.client:
            logger.error("Cliente da Binance não inicializado. Não é possível colocar ordem real.")
            return {"status": "ERROR", "message": "Binance client not initialized"}

        logger.info(f"Colocando ordem real: {signal} com tipo {order_type}")
        try:
            params = {
                "symbol": signal.symbol,
                "side": signal.signal_type.value, # BUY ou SELL
                "quantity": signal.position_size
            }

            if order_type == "MARKET":
                params["type"] = ORDER_TYPE_MARKET
                order = self.client.create_order(**params)
            elif order_type == "LIMIT":
                params["type"] = ORDER_TYPE_LIMIT
                params["timeInForce"] = TIME_IN_FORCE_GTC # Good Till Cancel
                params["price"] = signal.entry_price # Usar o preço de entrada do sinal
                order = self.client.create_order(**params)
            else:
                logger.error(f"Tipo de ordem {order_type} não suportado.")
                return {"status": "ERROR", "message": f"Order type {order_type} not supported"}

            logger.info(f"Ordem real colocada com sucesso: {order}")
            return order
        except Exception as e:
            logger.error(f"Erro ao colocar ordem real: {e}")
            return {"status": "ERROR", "message": str(e)}

    def check_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Verifica o status de uma ordem na exchange."""
        if not self.client:
            logger.error("Cliente da Binance não inicializado. Não é possível verificar status da ordem.")
            return {"status": "ERROR", "message": "Binance client not initialized"}

        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            return order
        except Exception as e:
            logger.error(f"Erro ao verificar status da ordem {order_id}: {e}")
            return {"status": "ERROR", "message": str(e)}

    def get_account_balance(self, asset: str) -> float:
        """Obtém o saldo de um ativo na conta."""
        if not self.client:
            logger.error("Cliente da Binance não inicializado. Não é possível obter saldo.")
            return 0.0
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance["free"])
        except Exception as e:
            logger.error(f"Erro ao obter saldo do ativo {asset}: {e}")
            return 0.0