import logging
from typing import Any, Dict
from datetime import datetime

from axon.core.types import Signal
from axon.utils.config import ConfigManager

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.risk_config = self.config.get_config().get("battle_arena", {}).get("compliance", {})
        self.circuit_breakers = self.risk_config.get("circuit_breakers", {})
        self.stop_trading_toggles = {
            "global": False, # Exemplo: um toggle global para parar todas as negociações
            "symbol_specific": {}
        }
        
        # Atributos para rastreamento de risco
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_profit = True
        self.last_reset_date = datetime.now().date()

        logger.info(f"RiskManager inicializado com configurações: {self.risk_config}")

    def _reset_daily_metrics(self):
        """Reseta as métricas diárias se a data mudou."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            logger.info(f"Resetando métricas diárias para {today}.")
            self.daily_pnl = 0.0
            self.last_reset_date = today

    def _check_daily_drawdown(self) -> bool:
        """Verifica se o drawdown diário excede o limite configurado."""
        self._reset_daily_metrics()
        max_daily_dd_percent = self.circuit_breakers.get("max_daily_drawdown_percent", 0.0)
        if max_daily_dd_percent > 0 and self.daily_pnl < -abs(max_daily_dd_percent): # Assumindo PnL negativo como drawdown
            logger.warning(f"Drawdown diário ({self.daily_pnl:.2f}%) excede o máximo permitido ({-abs(max_daily_dd_percent):.2f}%). Parando negociações.")
            self.stop_trading_toggles["global"] = True
            return True
        return False

    def _check_max_consecutive_losses(self) -> bool:
        """Verifica se o número de perdas consecutivas excede o limite configurado."""
        max_consecutive_losses = self.circuit_breakers.get("max_consecutive_losses", 0)
        if max_consecutive_losses > 0 and self.consecutive_losses >= max_consecutive_losses:
            logger.warning(f"Número de perdas consecutivas ({self.consecutive_losses}) excede o máximo permitido ({max_consecutive_losses}). Parando negociações.")
            self.stop_trading_toggles["global"] = True
            return True
        return False

    def update_trade_result(self, trade_result: Dict[str, Any]):
        """Atualiza as métricas de risco com base no resultado de um trade."""
        # Esta função será chamada pelo BattleArena após cada trade
        # trade_result deve conter informações sobre o lucro/prejuízo do trade
        pnl = trade_result.get("pnl", 0.0) # Assumindo que o resultado do trade tem um campo 'pnl'
        self.daily_pnl += pnl

        if pnl < 0: # Houve prejuízo
            if self.last_trade_profit:
                self.consecutive_losses = 1
            else:
                self.consecutive_losses += 1
            self.last_trade_profit = False
        else: # Houve lucro ou PnL zero
            self.consecutive_losses = 0
            self.last_trade_profit = True

        logger.info(f"Métricas de risco atualizadas: Daily PnL: {self.daily_pnl:.2f}, Perdas Consecutivas: {self.consecutive_losses}")

    def check_trade_risk(self, signal: Signal) -> bool:
        """Verifica se um sinal de trade está em conformidade com as regras de risco."""
        logger.debug(f"Verificando risco para o sinal: {signal}")

        # 1. Verificar toggles de stop trading
        if self.stop_trading_toggles["global"]:
            logger.warning("Negociação global desativada. Rejeitando trade.")
            return False
        if self.stop_trading_toggles["symbol_specific"].get(signal.symbol, False):
            logger.warning(f"Negociação para o símbolo {signal.symbol} desativada. Rejeitando trade.")
            return False

        # 2. Implementar lógica de Circuit Breakers
        # Exemplo: verificar slippage máximo
        max_slippage = self.circuit_breakers.get("max_slippage_percent", 0.0)
        # Assumindo que Signal tem slippage_percent. Se não tiver, precisará ser adicionado ou calculado.
        # if max_slippage > 0 and signal.slippage_percent > max_slippage:
        #     logger.warning(f"Slippage ({signal.slippage_percent:.2f}%) excede o máximo permitido ({max_slippage:.2f}%). Rejeitando trade.")
        #     return False

        # Exemplo: verificar número máximo de ordens abertas
        max_open_orders = self.circuit_breakers.get("max_open_orders", 0)
        # if max_open_orders > 0 and self.get_current_open_orders_count() >= max_open_orders: # Necessita de estado de ordens
        #     logger.warning(f"Número de ordens abertas excede o máximo permitido ({max_open_orders}). Rejeitando trade.")
        #     return False

        # 3. Verificar Daily Drawdown Cap
        if self._check_daily_drawdown():
            return False

        # 4. Verificar Max Consecutive Losses
        if self._check_max_consecutive_losses():
            return False

        # 5. Placeholder para Position Sizing (será implementado com mais detalhes depois)
        # self._apply_position_sizing(signal)

        return True