"""
Battle Controller
Controlador principal da Battle Arena que orquestra todas as operações de trading.
"""

import logging
import threading
import time
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from .data_feed import DataFeed
from .realtime_feature_engineer import RealtimeFeatureEngineer
from .signal_generator import SignalGenerator, Signal
from .model_loader import ModelLoader, ModelInfo
from ..core.order_engine import OrderEngine, ExecutionMode
from ..core.paper_trader import PaperTrader


class TradingState(Enum):
    """Estados do sistema de trading."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class BattleController:
    """
    Controlador principal da Battle Arena.

    Orquestra DataFeed, FeatureEngineer, SignalGenerator e OrderEngine
    para executar trading automatizado em tempo real.
    """

    def __init__(self, config: Dict[str, Any], model_info: ModelInfo):
        """
        Inicializa o BattleController.

        Args:
            config: Configuração da Battle Arena
            model_info: Informações do modelo carregado
        """
        self.config = config
        self.model_info = model_info

        self.logger = logging.getLogger(self.__class__.__name__)

        # Estado do sistema
        self.state = TradingState.INITIALIZING
        self.state_lock = threading.Lock()

        # Componentes principais
        self.data_feed: Optional[DataFeed] = None
        self.feature_engineer: Optional[RealtimeFeatureEngineer] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.order_engine: Optional[OrderEngine] = None
        self.paper_trader: Optional[PaperTrader] = None

        # Thread de trading
        self.trading_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Estado de posições ativas
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.pending_signals: List[Signal] = []

        # Estatísticas de performance
        self.performance_stats = {
            'start_time': None,
            'signals_generated': 0,
            'orders_executed': 0,
            'pnl_realized': 0.0,
            'pnl_unrealized': 0.0,
            'win_trades': 0,
            'loss_trades': 0,
            'max_drawdown': 0.0,
            'last_update': None
        }

        # Callbacks para eventos
        self.event_callbacks: Dict[str, List[Callable]] = {
            'signal_generated': [],
            'order_executed': [],
            'position_opened': [],
            'position_closed': [],
            'error': []
        }

        # Arquivo de estado
        self.state_file = Path("data/battle_arena/controller_state.json")

        self.logger.info("BattleController inicializado")

    def initialize_components(self) -> bool:
        """
        Inicializa todos os componentes necessários.

        Returns:
            True se inicialização bem-sucedida
        """
        try:
            self.logger.info("Inicializando componentes...")

            # 1. Paper Trader
            paper_config = self.config.get('paper_trading', {})
            self.paper_trader = PaperTrader(
                initial_capital=paper_config.get('initial_capital', 10000.0),
                fee_rate=paper_config.get('fee_rate', 0.001),
                state_file=paper_config.get('state_file', 'data/battle_arena/paper_trader_state.json')
            )

            # 2. Order Engine
            execution_mode = ExecutionMode.PAPER  # Sempre paper trading por segurança
            self.order_engine = OrderEngine(
                mode=execution_mode,
                paper_trader=self.paper_trader
            )

            # 3. Feature Engineer
            self.feature_engineer = RealtimeFeatureEngineer()

            # 4. Signal Generator
            signal_config = self.config.get('signals', {})
            self.signal_generator = SignalGenerator(
                model_info=self.model_info,
                buy_threshold=signal_config.get('buy_threshold', 0.6),
                sell_threshold=signal_config.get('sell_threshold', 0.4),
                min_confidence=signal_config.get('min_confidence', 0.55),
                max_position_size=signal_config.get('max_position_size', 0.1)
            )

            # 5. Data Feed
            symbols = self.config.get('symbols', ['BTCUSDT'])
            self.data_feed = DataFeed(
                symbols=symbols,
                interval=self.config.get('data_interval', '1m'),
                buffer_size=self.config.get('buffer_size', 1000)
            )

            # Conectar callbacks
            self._setup_callbacks()

            self.logger.info("Componentes inicializados com sucesso")
            return True

        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}")
            self.state = TradingState.ERROR
            return False

    def _setup_callbacks(self) -> None:
        """Configura callbacks entre componentes."""
        # DataFeed -> FeatureEngineer
        self.data_feed.add_data_callback(self._on_market_data)

        # FeatureEngineer -> SignalGenerator
        # SignalGenerator já tem buffer interno

    def _on_market_data(self, symbol: str, ohlcv_data: Dict[str, Any]) -> None:
        """Callback chamado quando novos dados de mercado chegam."""
        try:
            # Atualizar Feature Engineer
            self.feature_engineer.add_market_data(symbol, ohlcv_data)

            # Adicionar dados ao Signal Generator
            self.signal_generator.add_market_data(symbol, ohlcv_data)

            # Tentar gerar sinal
            signal = self.signal_generator.generate_signal(symbol)

            if signal:
                self._handle_signal(signal)

        except Exception as e:
            self.logger.error(f"Erro no callback de dados de mercado: {e}")
            self._trigger_event('error', {'error': str(e), 'symbol': symbol})

    def _handle_signal(self, signal: Signal) -> None:
        """Processa sinal gerado."""
        try:
            self.performance_stats['signals_generated'] += 1

            # Verificar se podemos executar o sinal
            if not self._can_execute_signal(signal):
                self.logger.debug(f"Sinal rejeitado: {signal.symbol} {signal.signal_type.value}")
                return

            # Adicionar à lista de pendentes
            self.pending_signals.append(signal)

            # Trigger event
            self._trigger_event('signal_generated', signal.to_dict())

            # Executar sinal se possível
            self._execute_pending_signals()

        except Exception as e:
            self.logger.error(f"Erro ao processar sinal: {e}")

    def _can_execute_signal(self, signal: Signal) -> bool:
        """Verifica se um sinal pode ser executado."""
        try:
            # Verificar estado do sistema
            if self.state != TradingState.RUNNING:
                return False

            # Verificar limites de risco
            risk_config = self.config.get('risk', {})

            # Verificar exposição máxima
            current_exposure = self._calculate_current_exposure()
            max_exposure = risk_config.get('max_total_exposure_pct', 0.5)
            if current_exposure >= max_exposure:
                return False

            # Verificar ordens por hora
            orders_per_hour = self._count_orders_last_hour()
            max_orders_hour = risk_config.get('max_orders_per_hour', 10)
            if orders_per_hour >= max_orders_hour:
                return False

            # Verificar se já temos posição neste símbolo
            if signal.symbol in self.active_positions:
                current_pos = self.active_positions[signal.symbol]

                # Se sinal é BUY e já temos posição, verificar se podemos aumentar
                if signal.signal_type.name == 'BUY' and current_pos['quantity'] > 0:
                    return False  # Não permitir posições long adicionais

                # Se sinal é SELL e não temos posição, ignorar
                if signal.signal_type.name == 'SELL' and current_pos['quantity'] <= 0:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Erro ao verificar sinal: {e}")
            return False

    def _execute_pending_signals(self) -> None:
        """Executa sinais pendentes."""
        try:
            # Processar sinais pendentes (limitado para evitar sobrecarga)
            max_to_process = 5
            processed = 0

            for signal in self.pending_signals[:max_to_process]:
                if self._execute_signal(signal):
                    processed += 1
                else:
                    break  # Parar se falhar

            # Remover sinais processados
            self.pending_signals = self.pending_signals[processed:]

        except Exception as e:
            self.logger.error(f"Erro ao executar sinais pendentes: {e}")

    def _execute_signal(self, signal: Signal) -> bool:
        """Executa um sinal específico."""
        try:
            # Calcular quantidade baseada no position size
            quantity = self._calculate_order_quantity(signal)

            if quantity <= 0:
                self.logger.warning(f"Quantidade inválida para sinal: {quantity}")
                return False

            # Determinar tipo de ordem
            order_type = 'market'  # Sempre market para sinais em tempo real

            # Executar ordem
            if signal.signal_type.name == 'BUY':
                order = self.order_engine.place_order(
                    signal.symbol, 'buy', order_type, quantity
                )
            elif signal.signal_type.name == 'SELL':
                order = self.order_engine.place_order(
                    signal.symbol, 'sell', order_type, quantity
                )
            else:
                return False  # HOLD não executa ordem

            # Atualizar estado
            self._update_position_state(signal.symbol, order)

            # Estatísticas
            self.performance_stats['orders_executed'] += 1
            self.performance_stats['last_update'] = datetime.now()

            # Trigger events
            self._trigger_event('order_executed', {
                'order': order.__dict__ if hasattr(order, '__dict__') else str(order),
                'signal': signal.to_dict()
            })

            self.logger.info(f"Ordem executada: {signal.symbol} {signal.signal_type.value} {quantity}")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao executar sinal: {e}")
            return False

    def _calculate_order_quantity(self, signal: Signal) -> float:
        """Calcula quantidade da ordem baseada no sinal."""
        try:
            # Obter preço atual
            current_price = self._get_current_price(signal.symbol)
            if not current_price:
                return 0.0

            # Obter saldo disponível
            balance = self.order_engine.get_balance()
            available_usdt = 0.0
            for bal in balance:
                if bal.asset == 'USDT':
                    available_usdt = bal.free
                    break

            # Calcular valor da posição
            position_value = available_usdt * signal.position_size

            # Quantidade = valor / preço
            quantity = position_value / current_price

            # Aplicar limites
            symbol_config = self.config.get('symbol_config', {}).get(signal.symbol, {})
            min_qty = symbol_config.get('min_qty', 0.001)
            max_qty = symbol_config.get('max_qty', 100.0)

            quantity = max(min_qty, min(quantity, max_qty))

            return quantity

        except Exception as e:
            self.logger.error(f"Erro ao calcular quantidade: {e}")
            return 0.0

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual de um símbolo."""
        try:
            latest_data = self.data_feed.get_latest_data(symbol, 1)
            if latest_data:
                return latest_data[0]['close']
            return None
        except Exception as e:
            self.logger.error(f"Erro ao obter preço atual: {e}")
            return None

    def _update_position_state(self, symbol: str, order: Any) -> None:
        """Atualiza estado das posições."""
        try:
            # Obter posições atuais
            positions = self.order_engine.get_positions()

            # Atualizar dicionário de posições ativas
            for pos in positions:
                if pos.symbol == symbol:
                    self.active_positions[symbol] = {
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'pnl': pos.pnl,
                        'pnl_percentage': pos.pnl_percentage,
                        'last_update': datetime.now()
                    }

                    # Trigger event se posição foi aberta/fechada
                    if abs(pos.quantity) > 0 and symbol not in self.active_positions:
                        self._trigger_event('position_opened', {'symbol': symbol, 'position': self.active_positions[symbol]})
                    elif abs(pos.quantity) == 0 and symbol in self.active_positions:
                        self._trigger_event('position_closed', {'symbol': symbol, 'position': self.active_positions[symbol]})
                        del self.active_positions[symbol]

                    break

        except Exception as e:
            self.logger.error(f"Erro ao atualizar estado da posição: {e}")

    def _calculate_current_exposure(self) -> float:
        """Calcula exposição atual como percentual do capital."""
        try:
            total_value = 0.0
            total_capital = self.paper_trader.initial_capital

            for symbol, position in self.active_positions.items():
                current_price = self._get_current_price(symbol) or position['entry_price']
                position_value = abs(position['quantity']) * current_price
                total_value += position_value

            return total_value / total_capital if total_capital > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Erro ao calcular exposição: {e}")
            return 0.0

    def _count_orders_last_hour(self) -> int:
        """Conta ordens executadas na última hora."""
        try:
            # Simplificado - em implementação real, manter histórico de ordens
            return 0  # Placeholder
        except Exception:
            return 0

    def start_trading(self) -> bool:
        """
        Inicia o loop de trading.

        Returns:
            True se iniciado com sucesso
        """
        with self.state_lock:
            if self.state == TradingState.RUNNING:
                self.logger.warning("Trading já está rodando")
                return True

            if not self.initialize_components():
                return False

            # Iniciar DataFeed
            self.data_feed.start()

            # Iniciar thread de trading
            self.stop_event.clear()
            self.trading_thread = threading.Thread(
                target=self._trading_loop,
                name="BattleController-Trading",
                daemon=True
            )
            self.trading_thread.start()

            self.state = TradingState.RUNNING
            self.performance_stats['start_time'] = datetime.now()

            self.logger.info("Trading iniciado")
            return True

    def stop_trading(self) -> None:
        """Para o trading."""
        with self.state_lock:
            if self.state in [TradingState.STOPPED, TradingState.ERROR]:
                return

            self.logger.info("Parando trading...")

            # Sinalizar parada
            self.stop_event.set()
            self.state = TradingState.STOPPED

            # Parar componentes
            if self.data_feed:
                self.data_feed.stop()

            # Aguardar thread
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10.0)

            # Salvar estado final
            self._save_state()

            self.logger.info("Trading parado")

    def pause_trading(self) -> None:
        """Pausa o trading."""
        with self.state_lock:
            if self.state == TradingState.RUNNING:
                self.state = TradingState.PAUSED
                self.logger.info("Trading pausado")

    def resume_trading(self) -> None:
        """Retoma o trading."""
        with self.state_lock:
            if self.state == TradingState.PAUSED:
                self.state = TradingState.RUNNING
                self.logger.info("Trading retomado")

    def _trading_loop(self) -> None:
        """Loop principal de trading."""
        self.logger.info("Loop de trading iniciado")

        while not self.stop_event.is_set():
            try:
                # Verificar estado
                if self.state != TradingState.RUNNING:
                    time.sleep(1)
                    continue

                # Atualizar estatísticas de performance
                self._update_performance_stats()

                # Verificar posições abertas e stop losses
                self._check_positions()

                # Pequena pausa para não sobrecarregar CPU
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Erro no loop de trading: {e}")
                self._trigger_event('error', {'error': str(e)})
                time.sleep(5)  # Pausa maior em caso de erro

        self.logger.info("Loop de trading finalizado")

    def _update_performance_stats(self) -> None:
        """Atualiza estatísticas de performance."""
        try:
            if self.paper_trader:
                stats = self.paper_trader.get_performance_stats()
                self.performance_stats.update({
                    'pnl_realized': stats.get('total_pnl', 0.0),
                    'current_balance': stats.get('current_balance', 0.0),
                    'total_return_pct': stats.get('total_return_pct', 0.0),
                    'open_positions': stats.get('open_positions', 0),
                    'last_update': datetime.now()
                })
        except Exception as e:
            self.logger.error(f"Erro ao atualizar estatísticas: {e}")

    def _check_positions(self) -> None:
        """Verifica posições abertas e aplica stop losses se necessário."""
        try:
            risk_config = self.config.get('risk', {})
            stop_loss_pct = risk_config.get('stop_loss_pct', 0.05)

            for symbol, position in list(self.active_positions.items()):
                pnl_pct = position['pnl_percentage']

                # Verificar stop loss
                if pnl_pct <= -stop_loss_pct:
                    self.logger.info(f"Stop loss ativado para {symbol}: {pnl_pct:.2%}")

                    # Fechar posição
                    self._close_position(symbol)

        except Exception as e:
            self.logger.error(f"Erro ao verificar posições: {e}")

    def _close_position(self, symbol: str) -> None:
        """Fecha posição em um símbolo."""
        try:
            position = self.active_positions.get(symbol)
            if not position or position['quantity'] == 0:
                return

            # Determinar lado oposto
            side = 'sell' if position['quantity'] > 0 else 'buy'
            quantity = abs(position['quantity'])

            # Executar ordem de fechamento
            order = self.order_engine.place_order(symbol, side, 'market', quantity)

            self.logger.info(f"Posição fechada: {symbol} {side} {quantity}")

            # Atualizar estado
            self._update_position_state(symbol, order)

        except Exception as e:
            self.logger.error(f"Erro ao fechar posição {symbol}: {e}")

    def _trigger_event(self, event_type: str, data: Any) -> None:
        """Dispara evento para callbacks registrados."""
        try:
            if event_type in self.event_callbacks:
                for callback in self.event_callbacks[event_type]:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Erro em callback {event_type}: {e}")
        except Exception as e:
            self.logger.error(f"Erro ao disparar evento {event_type}: {e}")

    def add_event_callback(self, event_type: str, callback: Callable) -> None:
        """
        Adiciona callback para evento específico.

        Args:
            event_type: Tipo de evento ('signal_generated', 'order_executed', etc.)
            callback: Função callback
        """
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []

        self.event_callbacks[event_type].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema."""
        return {
            'state': self.state.value,
            'performance': self.performance_stats.copy(),
            'active_positions': self.active_positions.copy(),
            'pending_signals': len(self.pending_signals),
            'components': {
                'data_feed': self.data_feed.get_stats() if self.data_feed else None,
                'feature_engineer': self.feature_engineer.get_stats() if self.feature_engineer else None,
                'signal_generator': self.signal_generator.get_signal_stats() if self.signal_generator else None
            }
        }

    def _save_state(self) -> None:
        """Salva estado do controlador."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                'state': self.state.value,
                'performance_stats': self.performance_stats,
                'active_positions': self.active_positions,
                'config': self.config,
                'saved_at': datetime.now().isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Erro ao salvar estado: {e}")

    def _load_state(self) -> None:
        """Carrega estado salvo."""
        try:
            if not self.state_file.exists():
                return

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.performance_stats.update(state.get('performance_stats', {}))
            self.active_positions.update(state.get('active_positions', {}))

            self.logger.info("Estado carregado")

        except Exception as e:
            self.logger.error(f"Erro ao carregar estado: {e}")

    def manual_order(self, symbol: str, side: str, quantity: float) -> bool:
        """
        Executa ordem manual.

        Args:
            symbol: Par de trading
            side: 'buy' ou 'sell'
            quantity: Quantidade

        Returns:
            True se executada
        """
        try:
            if self.state != TradingState.RUNNING:
                self.logger.warning("Sistema não está rodando")
                return False

            order = self.order_engine.place_order(symbol, side, 'market', quantity)
            self._update_position_state(symbol, order)

            self.logger.info(f"Ordem manual executada: {symbol} {side} {quantity}")
            return True

        except Exception as e:
            self.logger.error(f"Erro na ordem manual: {e}")
            return False

    def emergency_stop(self) -> None:
        """Para todas as operações imediatamente."""
        self.logger.warning("EMERGENCY STOP ativado!")

        # Fechar todas as posições
        for symbol in list(self.active_positions.keys()):
            self._close_position(symbol)

        # Parar trading
        self.stop_trading()

    def __enter__(self):
        """Context manager entry."""
        self.start_trading()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_trading()