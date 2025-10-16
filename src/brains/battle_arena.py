import logging
import os
import yaml
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.battle_arena.core.signal_generator import Signal
from axon.utils.config import ConfigManager
from axon.brains.notifier import Notifier
from axon.market_data.binance_ws import BinanceWebSocketConnector
from axon.features import FeatureEngineer
from axon.brains.decision_engine import DecisionEngine
from axon.risk_management.risk_manager import RiskManager
from axon.order_execution.order_adapter import OrderAdapter
from axon.ledger.ledger_manager import LedgerManager
import json

logger = logging.getLogger(__name__)

class BattleArena:
    def __init__(self, config_path: str = "axon.cfg.yml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.arena_config = self.config.get("battle_arena", {})
        self.enabled = self.arena_config.get("enabled", False)
        if not self.enabled:
            logger.info("Battle Arena está desabilitada na configuração.")
            return

        self.mode = self.arena_config.get("execution", {}).get("mode", "paper")
        self.testnet = self.arena_config.get("execution", {}).get("testnet", True)
        self.exchange = "binance" # Hardcoded for now, will be dynamic later

        self.notifier = Notifier(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.approved_artifacts = self._load_approved_artifacts()
        self.decision_engine = DecisionEngine(self.config, self.approved_artifacts)
        self.risk_manager = RiskManager(self.config)
        self.order_adapter = OrderAdapter(self.config, self.exchange, self.testnet)
        self.ledger_manager = LedgerManager(self.config)

        self.binance_ws_connector = None
        if self.arena_config.get("connectors", {}).get("binance", {}).get("enabled", False):
            self.binance_ws_connector = BinanceWebSocketConnector(self.config)

        logger.info(f"Battle Arena inicializada no modo: {self.mode}, Testnet: {self.testnet}")
        self.approved_artifacts = self._load_approved_artifacts()

    def _load_approved_artifacts(self):
        decision_file_path = "c:\\Users\\JHON-PC\\Desktop\\AXON-V3\\outputs\\metrics\\DECISION.json"
        try:
            with open(decision_file_path, 'r') as f:
                decision_data = json.load(f)

            if decision_data.get("pass"):
                logger.info("DECISION.json indica que 'pass' é verdadeiro. Carregando artefatos aprovados.")
                return [artifact['path'] for artifact in decision_data.get("artifacts", [])]
            else:
                logger.warning("DECISION.json indica que 'pass' é falso. Usando mecanismo de fallback.")
                return [] # Retornando uma lista vazia como fallback
        except FileNotFoundError:
            logger.warning(f"DECISION.json não encontrado em {decision_file_path}. Usando mecanismo de fallback.")
            return [] # Fallback se o arquivo não for encontrado
        except json.JSONDecodeError:
            logger.error(f"Erro ao decodificar JSON de {decision_file_path}. Usando mecanismo de fallback.")
            return [] # Fallback se o JSON for inválido

    def _process_klines(self, klines_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapta klines para features consistentes."""
        df = pd.DataFrame([klines_data])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df.astype(float)
        features = self.feature_engineer.generate_features(df)
        return features
 
    def _execute_alert_mode(self):
        """Executa o Battle Arena no modo 'alert-only'."""
        logger.info("Battle Arena operando no modo: ALERT-ONLY")
        if not self.approved_artifacts:
            logger.warning("Nenhum artefato aprovado para monitoramento no modo ALERT-ONLY.")
            return

        def on_klines_data(klines):
            for symbol, data in klines.items():
                logger.info(f"Recebendo dados de klines para {symbol}: {data}")
                processed_features = self._process_klines(data)
                trade_signal = self.decision_engine.make_decision(symbol, processed_features)
                
                if trade_signal.side != "HOLD":
                    alert_message = f"ALERTA DE TRADE ({self.mode.upper()}): {trade_signal.side} {trade_signal.symbol} @ {trade_signal.entry_price}"
                    self.notifier.send_telegram_message(alert_message)
                    logger.info(alert_message)

        if self.binance_ws_connector:
            symbols_to_stream = self.arena_config.get("symbols", [])
            self.binance_ws_connector.start_klines_stream(symbols_to_stream, self.config.get("data", {}).get("interval", "1m"), on_klines_data)
            logger.info(f"Iniciando stream de klines para {symbols_to_stream} no modo ALERT-ONLY.")
            # Manter o processo rodando
            while True:
                pass # Ou alguma lógica de loop para manter o conector ativo

    def _execute_paper_mode(self):
        """Executa o Battle Arena no modo 'paper' (simulação de trading)."""
        logger.info("Battle Arena operando no modo: PAPER")
        if not self.approved_artifacts:
            logger.warning("Nenhum artefato aprovado para trading em papel.")
            return

        def on_klines_data(klines):
            for symbol, data in klines.items():
                logger.info(f"Recebendo dados de klines para {symbol}: {data}")
                processed_features = self._process_klines(data)
                trade_signal = self.decision_engine.make_decision(symbol, processed_features)

                if trade_signal.side != "HOLD":
                    if self.risk_manager.check_trade_risk(trade_signal):
                        order_result = self.order_adapter.place_paper_order(trade_signal)
                        self.ledger_manager.record_trade(order_result)
                        alert_message = f"TRADE EM PAPEL ({self.mode.upper()}): {trade_signal.side} {trade_signal.symbol} @ {trade_signal.entry_price}. Resultado: {order_result}"
                        self.notifier.send_telegram_message(alert_message)
                        logger.info(alert_message)
                    else:
                        logger.warning(f"Trade para {trade_signal.symbol} rejeitado pelo Risk Manager no modo PAPER.")

        if self.binance_ws_connector:
            symbols_to_stream = self.arena_config.get("symbols", [])
            self.binance_ws_connector.start_klines_stream(symbols_to_stream, self.config.get("data", {}).get("interval", "1m"), on_klines_data)
            logger.info(f"Iniciando stream de klines para {symbols_to_stream} no modo PAPER.")
            while True:
                pass

    def _execute_shadow_live_mode(self, is_live: bool = False):
        """Executa o Battle Arena no modo 'shadow' ou 'live'."""
        mode_name = "LIVE" if is_live else "SHADOW"
        logger.info(f"Battle Arena operando no modo: {mode_name}")
        if not self.approved_artifacts:
            logger.warning(f"Nenhum artefato aprovado para trading no modo {mode_name}.")
            return

        def on_klines_data(klines):
            for symbol, data in klines.items():
                logger.info(f"Recebendo dados de klines para {symbol}: {data}")
                processed_features = self._process_klines(data)
                trade_signal = self.decision_engine.make_decision(symbol, processed_features)

                if trade_signal.side != "HOLD":
                    if self.risk_manager.check_trade_risk(trade_signal):
                        if is_live:
                            order_result = self.order_adapter.place_live_order(trade_signal)
                            self.ledger_manager.record_trade(order_result)
                            alert_message = f"TRADE REAL ({mode_name.upper()}): {trade_signal.side} {trade_signal.symbol} @ {trade_signal.entry_price}. Resultado: {order_result}"
                            self.notifier.send_telegram_message(alert_message)
                            logger.info(alert_message)
                        else: # Shadow mode
                            # No shadow mode, as ordens são simuladas mas não executadas no mercado real
                            alert_message = f"TRADE SOMBRA ({mode_name.upper()}): {trade_signal.side} {trade_signal.symbol} @ {trade_signal.entry_price}. (Simulado, não executado)"
                            self.notifier.send_telegram_message(alert_message)
                            logger.info(alert_message)
                    else:
                        logger.warning(f"Trade para {trade_signal.symbol} rejeitado pelo Risk Manager no modo {mode_name}.")

        if self.binance_ws_connector:
            symbols_to_stream = self.arena_config.get("symbols", [])
            self.binance_ws_connector.start_klines_stream(symbols_to_stream, self.config.get("data", {}).get("interval", "1m"), on_klines_data)
            logger.info(f"Iniciando stream de klines para {symbols_to_stream} no modo {mode_name}.")
            while True:
                pass

    def run(self):
        """Orquestra a execução do Battle Arena com base no modo configurado."""
        if not self.enabled:
            logger.info("Battle Arena está desabilitada. Encerrando.")
            return

        if self.mode == "alert":
            self._execute_alert_mode()
        elif self.mode == "paper":
            self._execute_paper_mode()
        elif self.mode == "shadow":
            self._execute_shadow_live_mode(is_live=False)
        elif self.mode == "live":
            self._execute_shadow_live_mode(is_live=True)
        else:
            logger.error(f"Modo de Battle Arena desconhecido: {self.mode}. Use 'alert', 'paper', 'shadow' ou 'live'.")

if __name__ == "__main__":
    # Exemplo de uso (para testes locais)
    # Certifique-se de ter um axon.cfg.yml configurado e variáveis de ambiente para Binance
    # arena = BattleArena()
    # arena.run()
    pass