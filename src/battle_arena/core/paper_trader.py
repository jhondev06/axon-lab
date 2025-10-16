"""
Paper Trader

Sistema de paper trading para simulação de trading.
Simula carteira virtual, execução de ordens e tracking de P&L.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

from ..connectors.base_connector import (
    BaseConnector, Order, Position, Balance
)


@dataclass
class PaperOrder(Order):
    """Ordem simulada para paper trading."""
    executed_price: Optional[float] = None
    executed_quantity: float = 0.0
    fees: float = 0.0


@dataclass
class PaperPosition(Position):
    """Posição simulada."""
    pass


@dataclass
class PaperBalance(Balance):
    """Saldo simulado."""
    pass


class PaperTrader:
    """
    Sistema de paper trading.

    Simula operações de trading sem risco financeiro.
    """

    def __init__(self, initial_capital: float = 10000.0,
                 fee_rate: float = 0.001,  # 0.1% fee
                 state_file: str = "data/battle_arena/paper_trader_state.json"):
        """
        Inicializa o paper trader.

        Args:
            initial_capital: Capital inicial em USD
            fee_rate: Taxa de fee por operação
            state_file: Arquivo para persistir estado
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.state_file = Path(state_file)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Estado da carteira
        self.balance = {
            'USDT': initial_capital,  # Stablecoin como base
        }

        # Ordens ativas
        self.active_orders: Dict[str, PaperOrder] = {}

        # Histórico de ordens executadas
        self.order_history: List[PaperOrder] = []

        # Posições abertas
        self.positions: Dict[str, PaperPosition] = {}

        # Estatísticas de performance
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat()
        }

        # Carregar estado se existir
        self._load_state()

        self.logger.info(f"Paper Trader inicializado com capital: ${initial_capital}")

    def connect_connector(self, connector: BaseConnector) -> None:
        """
        Conecta um conector real para obter preços.

        Args:
            connector: Instância do conector
        """
        self.connector = connector
        self.logger.info(f"Conector conectado: {connector.__class__.__name__}")

    def get_balance(self) -> List[PaperBalance]:
        """
        Obtém saldos simulados.

        Returns:
            Lista de saldos
        """
        balances = []
        for asset, amount in self.balance.items():
            if amount > 0:
                balances.append(PaperBalance(
                    asset=asset,
                    free=amount,
                    locked=0.0,  # Paper trading não tem locked funds
                    total=amount
                ))
        return balances

    def place_order(self, symbol: str, side: str, order_type: str,
                   quantity: float, price: Optional[float] = None) -> PaperOrder:
        """
        Coloca uma ordem simulada.

        Args:
            symbol: Par de trading
            side: 'buy' ou 'sell'
            order_type: 'market' ou 'limit'
            quantity: Quantidade
            price: Preço (para limit orders)

        Returns:
            Ordem simulada
        """
        # Validar ordem
        self._validate_order(symbol, side, order_type, quantity, price)

        # Obter preço de execução
        if order_type == 'market':
            execution_price = self._get_market_price(symbol)
        else:  # limit
            execution_price = price

        # Calcular fees
        fees = execution_price * quantity * self.fee_rate

        # Criar ordem
        order_id = f"paper_{len(self.order_history) + len(self.active_orders) + 1}"
        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status='filled',  # Ordens paper são executadas imediatamente
            executed_price=execution_price,
            executed_quantity=quantity,
            fees=fees
        )

        # Executar ordem
        self._execute_order(order)

        # Adicionar ao histórico
        self.order_history.append(order)

        # Atualizar estatísticas
        self._update_stats()

        # Salvar estado
        self._save_state()

        self.logger.info(f"Ordem executada: {order_id} {side} {quantity} {symbol} @ {execution_price}")
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancela uma ordem simulada.

        Args:
            order_id: ID da ordem
            symbol: Par de trading

        Returns:
            True se cancelada
        """
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            self.logger.info(f"Ordem cancelada: {order_id}")
            self._save_state()
            return True
        return False

    def get_positions(self) -> List[PaperPosition]:
        """
        Obtém posições abertas simuladas.

        Returns:
            Lista de posições
        """
        positions = []
        for symbol, position in self.positions.items():
            # Atualizar preço atual
            current_price = self._get_market_price(symbol)
            position.current_price = current_price

            # Recalcular P&L
            if position.quantity > 0:  # Long position
                position.pnl = (current_price - position.entry_price) * position.quantity
            else:  # Short position (não implementado ainda)
                position.pnl = 0.0

            position.pnl_percentage = (position.pnl / (position.entry_price * abs(position.quantity))) * 100
            positions.append(position)

        return positions

    def get_order_status(self, order_id: str, symbol: str) -> Optional[PaperOrder]:
        """
        Obtém status de uma ordem.

        Args:
            order_id: ID da ordem
            symbol: Par de trading

        Returns:
            Ordem ou None
        """
        # Verificar ordens ativas
        if order_id in self.active_orders:
            return self.active_orders[order_id]

        # Verificar histórico
        for order in self.order_history:
            if order.order_id == order_id:
                return order

        return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas de performance.

        Returns:
            Dicionário com estatísticas
        """
        # Calcular métricas atuais
        positions = self.get_positions()
        total_pnl = sum(pos.pnl for pos in positions) + self.stats['total_pnl']

        current_balance = sum(bal.total for bal in self.get_balance())
        total_return = ((current_balance - self.initial_capital) / self.initial_capital) * 100

        return {
            'initial_capital': self.initial_capital,
            'current_balance': current_balance,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'total_trades': self.stats['total_trades'],
            'winning_trades': self.stats['winning_trades'],
            'losing_trades': self.stats['losing_trades'],
            'win_rate': (self.stats['winning_trades'] / max(self.stats['total_trades'], 1)) * 100,
            'total_fees': self.stats['total_fees'],
            'open_positions': len(self.positions),
            'active_orders': len(self.active_orders)
        }

    def reset(self) -> None:
        """Reseta o estado do paper trader."""
        self.balance = {'USDT': self.initial_capital}
        self.active_orders.clear()
        self.order_history.clear()
        self.positions.clear()
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat()
        }
        self._save_state()
        self.logger.info("Paper trader resetado")

    def _validate_order(self, symbol: str, side: str, order_type: str,
                       quantity: float, price: Optional[float]) -> None:
        """Valida parâmetros da ordem."""
        if side not in ['buy', 'sell']:
            raise ValueError(f"Side deve ser 'buy' ou 'sell', não {side}")

        if order_type not in ['market', 'limit']:
            raise ValueError(f"Order type deve ser 'market' ou 'limit', não {order_type}")

        if quantity <= 0:
            raise ValueError("Quantidade deve ser positiva")

        if order_type == 'limit' and price is None:
            raise ValueError("Preço obrigatório para ordens limit")

        if order_type == 'limit' and price <= 0:
            raise ValueError("Preço deve ser positivo")

        # Verificar saldo suficiente (simplificado)
        if side == 'buy':
            base_asset = symbol.replace('USDT', '')  # Assumindo pares USDT
            required_amount = quantity * (price or self._get_market_price(symbol))
            if self.balance.get('USDT', 0) < required_amount:
                raise ValueError(f"Saldo insuficiente: {self.balance.get('USDT', 0)} < {required_amount}")

    def _execute_order(self, order: PaperOrder) -> None:
        """Executa uma ordem simulada."""
        base_asset = order.symbol.replace('USDT', '')  # Assumindo pares USDT

        if order.side == 'buy':
            # Debitar USDT
            cost = order.executed_price * order.executed_quantity + order.fees
            self.balance['USDT'] -= cost

            # Creditar asset
            self.balance[base_asset] = self.balance.get(base_asset, 0) + order.executed_quantity

            # Criar/atualizar posição
            if base_asset not in self.positions:
                self.positions[base_asset] = PaperPosition(
                    symbol=order.symbol,
                    quantity=order.executed_quantity,
                    entry_price=order.executed_price,
                    current_price=order.executed_price,
                    pnl=0.0,
                    pnl_percentage=0.0
                )
            else:
                # Atualizar posição média
                current_qty = self.positions[base_asset].quantity
                current_entry = self.positions[base_asset].entry_price
                new_qty = current_qty + order.executed_quantity
                new_entry = ((current_qty * current_entry) + (order.executed_quantity * order.executed_price)) / new_qty

                self.positions[base_asset].quantity = new_qty
                self.positions[base_asset].entry_price = new_entry

        elif order.side == 'sell':
            # Verificar se tem posição suficiente
            if self.balance.get(base_asset, 0) < order.executed_quantity:
                raise ValueError(f"Posição insuficiente para vender: {self.balance.get(base_asset, 0)} < {order.executed_quantity}")

            # Creditar USDT
            revenue = order.executed_price * order.executed_quantity - order.fees
            self.balance['USDT'] += revenue

            # Debitar asset
            self.balance[base_asset] -= order.executed_quantity

            # Atualizar/remover posição
            if base_asset in self.positions:
                self.positions[base_asset].quantity -= order.executed_quantity
                if self.positions[base_asset].quantity <= 0:
                    del self.positions[base_asset]

    def _get_market_price(self, symbol: str) -> float:
        """Obtém preço de mercado (usando conector se disponível)."""
        if hasattr(self, 'connector') and self.connector:
            try:
                if hasattr(self.connector, 'get_ticker_price'):
                    return self.connector.get_ticker_price(symbol)
            except Exception as e:
                self.logger.warning(f"Erro ao obter preço do conector: {e}")

        # Fallback: preços simulados (para desenvolvimento)
        # Em produção, sempre usar conector
        simulated_prices = {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 3000.0,
            'BNBUSDT': 300.0
        }
        return simulated_prices.get(symbol, 100.0)

    def _update_stats(self) -> None:
        """Atualiza estatísticas de performance."""
        self.stats['total_trades'] = len(self.order_history)
        self.stats['last_update'] = datetime.now().isoformat()

        # Calcular win/loss (simplificado - baseado em ordens sell)
        # Implementação completa precisaria de tracking de P&L por trade
        pass

    def _save_state(self) -> None:
        """Salva estado em arquivo."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                'balance': self.balance,
                'active_orders': {k: asdict(v) for k, v in self.active_orders.items()},
                'order_history': [asdict(order) for order in self.order_history[-100:]],  # Últimas 100
                'positions': {k: asdict(v) for k, v in self.positions.items()},
                'stats': self.stats,
                'config': {
                    'initial_capital': self.initial_capital,
                    'fee_rate': self.fee_rate
                }
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {e}")

    def _load_state(self) -> None:
        """Carrega estado do arquivo."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.balance = state.get('balance', self.balance)
            self.stats = state.get('stats', self.stats)

            # Reconstruir ordens ativas
            for order_id, order_data in state.get('active_orders', {}).items():
                self.active_orders[order_id] = PaperOrder(**order_data)

            # Reconstruir histórico
            for order_data in state.get('order_history', []):
                self.order_history.append(PaperOrder(**order_data))

            # Reconstruir posições
            for symbol, pos_data in state.get('positions', {}).items():
                self.positions[symbol] = PaperPosition(**pos_data)

            self.logger.info("Estado do paper trader carregado")

        except Exception as e:
            self.logger.error(f"Erro ao carregar estado: {e}")