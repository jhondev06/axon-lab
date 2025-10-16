"""
Base Connector Interface

Interface abstrata para conectores de exchange na Battle Arena.
Define métodos padrão e tratamento unificado de erros.
"""

import abc
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Order:
    """Estrutura de dados para ordens."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', etc.
    quantity: float
    price: Optional[float] = None
    status: str = 'pending'
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """Estrutura de dados para posições."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percentage: float


@dataclass
class Balance:
    """Estrutura de dados para saldo."""
    asset: str
    free: float
    locked: float
    total: float


class ExchangeError(Exception):
    """Erro base para problemas de exchange."""
    pass


class ConnectionError(ExchangeError):
    """Erro de conexão com exchange."""
    pass


class AuthenticationError(ExchangeError):
    """Erro de autenticação."""
    pass


class InsufficientFundsError(ExchangeError):
    """Saldo insuficiente."""
    pass


class RateLimitError(ExchangeError):
    """Limite de taxa excedido."""
    pass


class BaseConnector(abc.ABC):
    """
    Interface abstrata para conectores de exchange.

    Define métodos padrão que todos os conectores devem implementar.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Inicializa o conector.

        Args:
            api_key: Chave da API
            api_secret: Segredo da API
            testnet: Usar testnet se disponível
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configurar logging se não estiver configurado
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    @abc.abstractmethod
    def get_balance(self) -> List[Balance]:
        """
        Obtém saldos da conta.

        Returns:
            Lista de objetos Balance
        """
        pass

    @abc.abstractmethod
    def place_order(self, symbol: str, side: str, order_type: str,
                   quantity: float, price: Optional[float] = None) -> Order:
        """
        Coloca uma nova ordem.

        Args:
            symbol: Par de trading (ex: 'BTCUSDT')
            side: 'buy' ou 'sell'
            order_type: Tipo da ordem ('market', 'limit', etc.)
            quantity: Quantidade
            price: Preço (obrigatório para limit orders)

        Returns:
            Objeto Order com detalhes da ordem
        """
        pass

    @abc.abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancela uma ordem.

        Args:
            order_id: ID da ordem
            symbol: Par de trading

        Returns:
            True se cancelada com sucesso
        """
        pass

    @abc.abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Obtém posições abertas.

        Returns:
            Lista de objetos Position
        """
        pass

    @abc.abstractmethod
    def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """
        Obtém status de uma ordem.

        Args:
            order_id: ID da ordem
            symbol: Par de trading

        Returns:
            Objeto Order ou None se não encontrada
        """
        pass

    def _handle_error(self, error: Exception, operation: str) -> None:
        """
        Trata erros de forma unificada.

        Args:
            error: Exceção ocorrida
            operation: Operação que falhou

        Raises:
            ExchangeError apropriado
        """
        error_msg = f"Erro em {operation}: {str(error)}"
        self.logger.error(error_msg)

        # Mapeia erros comuns para tipos específicos
        if "connection" in str(error).lower() or "timeout" in str(error).lower():
            raise ConnectionError(error_msg) from error
        elif "auth" in str(error).lower() or "key" in str(error).lower():
            raise AuthenticationError(error_msg) from error
        elif "insufficient" in str(error).lower() or "balance" in str(error).lower():
            raise InsufficientFundsError(error_msg) from error
        elif "rate" in str(error).lower() or "limit" in str(error).lower():
            raise RateLimitError(error_msg) from error
        else:
            raise ExchangeError(error_msg) from error

    def _validate_credentials(self) -> None:
        """Valida se as credenciais foram fornecidas."""
        if not self.api_key or not self.api_secret:
            raise AuthenticationError("API key e secret são obrigatórios")

    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Log estruturado de operações."""
        self.logger.info(f"{operation}: {details}")

    def is_connected(self) -> bool:
        """
        Verifica se a conexão com a exchange está ativa.

        Returns:
            True se conectado
        """
        try:
            self.get_balance()
            return True
        except Exception:
            return False