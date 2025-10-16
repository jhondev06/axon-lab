"""
Order Engine

Engine de execução de ordens para Battle Arena.
Coordena entre conectores reais e paper trading.
"""

import logging
import time
from typing import Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

from ..connectors.base_connector import BaseConnector, Order
from .paper_trader import PaperTrader


class ExecutionMode(Enum):
    """Modo de execução."""
    LIVE = "live"      # Trading real
    PAPER = "paper"    # Paper trading
    SIMULATION = "simulation"  # Simulação completa


class OrderEngine:
    """
    Engine para execução de ordens.

    Suporte a diferentes modos: live, paper, simulation.
    """

    def __init__(self, mode: ExecutionMode = ExecutionMode.PAPER,
                 connector: Optional[BaseConnector] = None,
                 paper_trader: Optional[PaperTrader] = None):
        """
        Inicializa o order engine.

        Args:
            mode: Modo de execução
            connector: Conector para trading real
            paper_trader: Instância do paper trader
        """
        self.mode = mode
        self.connector = connector
        self.paper_trader = paper_trader
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validar configuração
        if self.mode == ExecutionMode.LIVE and not self.connector:
            raise ValueError("Conector obrigatório para modo LIVE")

        if self.mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION] and not self.paper_trader:
            raise ValueError("Paper trader obrigatório para modos PAPER/SIMULATION")

        self.logger.info(f"Order Engine inicializado no modo: {self.mode.value}")

    def place_order(self, symbol: str, side: str, order_type: str,
                   quantity: float, price: Optional[float] = None,
                   **kwargs) -> Order:
        """
        Executa uma ordem baseada no modo configurado.

        Args:
            symbol: Par de trading
            side: 'buy' ou 'sell'
            order_type: 'market' ou 'limit'
            quantity: Quantidade
            price: Preço (para limit orders)
            **kwargs: Parâmetros adicionais

        Returns:
            Objeto Order
        """
        self.logger.info(f"Executando ordem: {side} {quantity} {symbol} @ {price or 'market'} (modo: {self.mode.value})")

        # Rate limiting básico
        time.sleep(0.1)

        if self.mode == ExecutionMode.LIVE:
            return self._execute_live_order(symbol, side, order_type, quantity, price, **kwargs)
        elif self.mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION]:
            return self._execute_paper_order(symbol, side, order_type, quantity, price, **kwargs)
        else:
            raise ValueError(f"Modo não suportado: {self.mode}")

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancela uma ordem.

        Args:
            order_id: ID da ordem
            symbol: Par de trading

        Returns:
            True se cancelada
        """
        self.logger.info(f"Cancelando ordem: {order_id} {symbol}")

        if self.mode == ExecutionMode.LIVE:
            return self.connector.cancel_order(order_id, symbol)
        elif self.mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION]:
            return self.paper_trader.cancel_order(order_id, symbol)
        else:
            raise ValueError(f"Modo não suportado: {self.mode}")

    def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """
        Obtém status de uma ordem.

        Args:
            order_id: ID da ordem
            symbol: Par de trading

        Returns:
            Ordem ou None
        """
        if self.mode == ExecutionMode.LIVE:
            return self.connector.get_order_status(order_id, symbol)
        elif self.mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION]:
            return self.paper_trader.get_order_status(order_id, symbol)
        else:
            raise ValueError(f"Modo não suportado: {self.mode}")

    def get_balance(self):
        """
        Obtém saldos.

        Returns:
            Lista de saldos
        """
        if self.mode == ExecutionMode.LIVE:
            return self.connector.get_balance()
        elif self.mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION]:
            return self.paper_trader.get_balance()
        else:
            raise ValueError(f"Modo não suportado: {self.mode}")

    def get_positions(self):
        """
        Obtém posições abertas.

        Returns:
            Lista de posições
        """
        if self.mode == ExecutionMode.LIVE:
            return self.connector.get_positions()
        elif self.mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION]:
            return self.paper_trader.get_positions()
        else:
            raise ValueError(f"Modo não suportado: {self.mode}")

    def _execute_live_order(self, symbol: str, side: str, order_type: str,
                           quantity: float, price: Optional[float],
                           **kwargs) -> Order:
        """Executa ordem no modo live."""
        if not self.connector:
            raise RuntimeError("Conector não configurado para modo LIVE")

        # Aplicar validações de risco aqui (futuramente)
        # ...

        return self.connector.place_order(symbol, side, order_type, quantity, price)

    def _execute_paper_order(self, symbol: str, side: str, order_type: str,
                            quantity: float, price: Optional[float],
                            **kwargs) -> Order:
        """Executa ordem no modo paper."""
        if not self.paper_trader:
            raise RuntimeError("Paper trader não configurado")

        # Aplicar validações de risco aqui (futuramente)
        # ...

        return self.paper_trader.place_order(symbol, side, order_type, quantity, price)

    def switch_mode(self, new_mode: ExecutionMode) -> None:
        """
        Altera o modo de execução.

        Args:
            new_mode: Novo modo
        """
        if new_mode == ExecutionMode.LIVE and not self.connector:
            raise ValueError("Conector obrigatório para modo LIVE")

        if new_mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION] and not self.paper_trader:
            raise ValueError("Paper trader obrigatório para modos PAPER/SIMULATION")

        self.mode = new_mode
        self.logger.info(f"Modo alterado para: {new_mode.value}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas de performance.

        Returns:
            Estatísticas (apenas para modos paper/simulation)
        """
        if self.mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION]:
            return self.paper_trader.get_performance_stats()
        else:
            return {
                'mode': self.mode.value,
                'note': 'Estatísticas não disponíveis para modo LIVE'
            }

    def is_connected(self) -> bool:
        """
        Verifica se está conectado.

        Returns:
            True se conectado
        """
        if self.mode == ExecutionMode.LIVE:
            return self.connector.is_connected() if self.connector else False
        else:
            return True  # Paper trading sempre "conectado"