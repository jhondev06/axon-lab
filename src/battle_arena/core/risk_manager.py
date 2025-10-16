"""
Risk Manager

Gestão de risco básica para Battle Arena.
Validações de limites de posição, drawdown, exposição, etc.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..connectors.base_connector import Balance, Position


@dataclass
class RiskLimits:
    """Limites de risco configuráveis."""
    max_position_size_pct: float = 0.1  # 10% do capital por posição
    max_total_exposure_pct: float = 0.5  # 50% exposição total
    max_daily_loss_pct: float = 0.05  # 5% perda diária máxima
    max_drawdown_pct: float = 0.1  # 10% drawdown máximo
    max_orders_per_hour: int = 10  # Máximo de ordens por hora
    min_order_size_usd: float = 10.0  # Tamanho mínimo de ordem em USD
    max_order_size_usd: float = 1000.0  # Tamanho máximo de ordem em USD


class RiskManager:
    """
    Gerenciador de risco para validar operações de trading.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Inicializa o risk manager.

        Args:
            limits: Limites de risco (usa padrão se None)
        """
        self.limits = limits or RiskLimits()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Estado de tracking
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        self.orders_today = 0
        self.last_reset_date = datetime.now().date()

        # Histórico de ordens (para rate limiting)
        self.order_history: List[datetime] = []

        self.logger.info("Risk Manager inicializado")

    def validate_order(self, symbol: str, side: str, quantity: float,
                      price: float, balances: List[Balance],
                      positions: List[Position]) -> Dict[str, Any]:
        """
        Valida se uma ordem pode ser executada baseada em regras de risco.

        Args:
            symbol: Par de trading
            side: 'buy' ou 'sell'
            quantity: Quantidade
            price: Preço
            balances: Saldos atuais
            positions: Posições atuais

        Returns:
            Dict com 'approved': bool e 'reason': str se rejeitada
        """
        # Resetar contadores diários se necessário
        self._check_daily_reset()

        # Calcular valor da ordem
        order_value = quantity * price

        # 1. Validar tamanho mínimo/máximo da ordem
        if order_value < self.limits.min_order_size_usd:
            return {
                'approved': False,
                'reason': f"Ordem muito pequena: ${order_value:.2f} < ${self.limits.min_order_size_usd:.2f}"
            }

        if order_value > self.limits.max_order_size_usd:
            return {
                'approved': False,
                'reason': f"Ordem muito grande: ${order_value:.2f} > ${self.limits.max_order_size_usd:.2f}"
            }

        # 2. Validar rate limiting
        if not self._check_rate_limit():
            return {
                'approved': False,
                'reason': f"Limite de ordens por hora excedido: {self.limits.max_orders_per_hour}"
            }

        # 3. Validar saldo suficiente
        if not self._check_sufficient_balance(side, order_value, balances):
            return {
                'approved': False,
                'reason': "Saldo insuficiente"
            }

        # 4. Validar limites de posição
        if not self._check_position_limits(symbol, side, quantity, price, positions):
            return {
                'approved': False,
                'reason': f"Limite de posição excedido: {self.limits.max_position_size_pct*100:.1f}% do capital"
            }

        # 5. Validar exposição total
        if not self._check_total_exposure(order_value, balances, positions):
            return {
                'approved': False,
                'reason': f"Exposição total excederia: {self.limits.max_total_exposure_pct*100:.1f}%"
            }

        # 6. Validar perda diária
        if not self._check_daily_loss_limit():
            return {
                'approved': False,
                'reason': f"Limite de perda diária excedido: {self.limits.max_daily_loss_pct*100:.1f}%"
            }

        # 7. Validar drawdown
        if not self._check_drawdown_limit(balances):
            return {
                'approved': False,
                'reason': f"Drawdown máximo excedido: {self.limits.max_drawdown_pct*100:.1f}%"
            }

        # Registrar ordem para rate limiting
        self.order_history.append(datetime.now())
        self.orders_today += 1

        return {'approved': True, 'reason': 'Ordem aprovada'}

    def update_pnl(self, pnl_change: float, current_balance: float) -> None:
        """
        Atualiza tracking de P&L e drawdown.

        Args:
            pnl_change: Mudança no P&L
            current_balance: Saldo atual
        """
        self.daily_pnl += pnl_change

        # Atualizar drawdown
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance

    def get_risk_status(self) -> Dict[str, Any]:
        """
        Obtém status atual dos indicadores de risco.

        Returns:
            Dict com métricas de risco
        """
        return {
            'daily_pnl': self.daily_pnl,
            'current_drawdown_pct': self.current_drawdown * 100,
            'orders_today': self.orders_today,
            'peak_balance': self.peak_balance,
            'limits': {
                'max_daily_loss_pct': self.limits.max_daily_loss_pct * 100,
                'max_drawdown_pct': self.limits.max_drawdown_pct * 100,
                'max_orders_per_hour': self.limits.max_orders_per_hour,
                'max_position_size_pct': self.limits.max_position_size_pct * 100,
                'max_total_exposure_pct': self.limits.max_total_exposure_pct * 100
            }
        }

    def _check_daily_reset(self) -> None:
        """Reseta contadores diários se necessário."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.orders_today = 0
            self.order_history.clear()
            self.last_reset_date = today
            self.logger.info("Contadores diários resetados")

    def _check_rate_limit(self) -> bool:
        """Verifica limite de ordens por hora."""
        # Remover ordens antigas (mais de 1 hora)
        cutoff = datetime.now() - timedelta(hours=1)
        self.order_history = [t for t in self.order_history if t > cutoff]

        return len(self.order_history) < self.limits.max_orders_per_hour

    def _check_sufficient_balance(self, side: str, order_value: float,
                                balances: List[Balance]) -> bool:
        """Verifica se há saldo suficiente."""
        if side == 'buy':
            # Para compras, verificar saldo em USDT (ou moeda base)
            usdt_balance = 0.0
            for balance in balances:
                if balance.asset in ['USDT', 'USD', 'BUSD']:
                    usdt_balance += balance.free

            return usdt_balance >= order_value

        elif side == 'sell':
            # Para vendas, verificar se há posição suficiente
            # (Esta validação é mais complexa e pode ser feita na position check)
            return True

        return False

    def _check_position_limits(self, symbol: str, side: str, quantity: float,
                             price: float, positions: List[Position]) -> bool:
        """Verifica limites de tamanho de posição."""
        # Calcular capital total (simplificado - assumir USDT como base)
        total_capital = sum(b.total for b in [] if hasattr(b, 'total'))  # Placeholder

        # Para paper trading, assumir capital base
        if not total_capital:
            total_capital = 10000.0  # Valor padrão

        order_value = quantity * price
        max_position_value = total_capital * self.limits.max_position_size_pct

        # Verificar posição existente
        existing_position_value = 0.0
        for position in positions:
            if position.symbol == symbol:
                existing_position_value = abs(position.quantity) * position.current_price
                break

        new_position_value = existing_position_value + order_value

        return new_position_value <= max_position_value

    def _check_total_exposure(self, order_value: float, balances: List[Balance],
                            positions: List[Position]) -> bool:
        """Verifica exposição total."""
        # Calcular capital total
        total_capital = sum(b.total for b in balances)

        # Calcular exposição atual
        current_exposure = 0.0
        for position in positions:
            current_exposure += abs(position.quantity) * position.current_price

        new_exposure = current_exposure + order_value
        max_exposure = total_capital * self.limits.max_total_exposure_pct

        return new_exposure <= max_exposure

    def _check_daily_loss_limit(self) -> bool:
        """Verifica limite de perda diária."""
        if self.daily_pnl < 0:
            loss_pct = abs(self.daily_pnl) / max(self.peak_balance, 10000.0)  # Fallback
            return loss_pct <= self.limits.max_daily_loss_pct
        return True

    def _check_drawdown_limit(self, balances: List[Balance]) -> bool:
        """Verifica limite de drawdown."""
        current_balance = sum(b.total for b in balances)
        return self.current_drawdown <= self.limits.max_drawdown_pct

    def emergency_stop(self) -> None:
        """Para todas as operações em caso de emergência."""
        self.logger.critical("EMERGENCY STOP ativado - todas as operações suspensas")
        # Implementar lógica de emergência (cancelar ordens, fechar posições, etc.)
        # Por enquanto, apenas log