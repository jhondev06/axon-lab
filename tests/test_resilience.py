"""
AXON Battle Arena - Resilience Tests

Testes unitários para o módulo de resiliência:
- StateManager (persistência atômica)
- HeartbeatMonitor (health checks)
- ReconnectionHandler (exponential backoff + jitter)
- TelegramCommandBot (comandos remotos)
"""

import pytest
import json
import os
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.battle_arena.core.resilience import (
    SystemState,
    StateManager,
    HeartbeatMonitor,
    ReconnectionHandler,
    ResilienceManager,
    get_system_status
)


class TestSystemState:
    """Testes para SystemState dataclass."""
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        state = SystemState()
        state_dict = state.to_dict()
        
        assert 'version' in state_dict
        assert 'is_running' in state_dict
        assert 'websocket_connected' in state_dict
    
    def test_from_dict(self):
        """Testa criação a partir de dicionário."""
        data = {
            'version': 1,
            'is_running': True,
            'websocket_connected': True,
            'reconnection_attempts': 5
        }
        
        state = SystemState.from_dict(data)
        
        assert state.is_running == True
        assert state.websocket_connected == True
        assert state.reconnection_attempts == 5
    
    def test_from_dict_unknown_fields_ignored(self):
        """Testa que campos desconhecidos são ignorados."""
        data = {
            'version': 1,
            'is_running': True,
            'unknown_field': 'should_be_ignored'
        }
        
        state = SystemState.from_dict(data)
        assert not hasattr(state, 'unknown_field')


class TestStateManager:
    """Testes para StateManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Cria diretório temporário para testes."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def state_manager(self, temp_dir):
        """Cria StateManager com diretório temporário."""
        state_file = os.path.join(temp_dir, 'state.json')
        backup_dir = os.path.join(temp_dir, 'backups')
        return StateManager(state_file=state_file, backup_dir=backup_dir)
    
    def test_initial_state(self, state_manager):
        """Testa estado inicial."""
        state = state_manager.state
        
        assert state.version == 1
        assert state.is_running == False
        assert state.is_emergency_stopped == False
    
    def test_update_state(self, state_manager):
        """Testa atualização de estado."""
        state_manager.update(is_running=True, websocket_connected=True)
        
        state = state_manager.state
        assert state.is_running == True
        assert state.websocket_connected == True
    
    def test_state_persistence(self, state_manager, temp_dir):
        """Testa que estado é persistido e recuperado."""
        state_manager.update(
            is_running=True,
            total_trades=42,
            reconnection_attempts=3
        )
        
        # Criar novo manager apontando para mesmo arquivo
        state_file = os.path.join(temp_dir, 'state.json')
        backup_dir = os.path.join(temp_dir, 'backups')
        new_manager = StateManager(state_file=state_file, backup_dir=backup_dir)
        
        new_state = new_manager.state
        assert new_state.is_running == True
        assert new_state.total_trades == 42
        assert new_state.reconnection_attempts == 3
    
    def test_atomic_write_creates_backup(self, state_manager, temp_dir):
        """Testa que backup é criado antes de sobrescrever."""
        # Primeiro update
        state_manager.update(is_running=True)
        
        # Segundo update deve criar backup
        state_manager.update(total_trades=10)
        
        backup_dir = Path(temp_dir) / 'backups'
        backups = list(backup_dir.glob('state_backup_*.json'))
        assert len(backups) >= 1
    
    def test_emergency_stop(self, state_manager):
        """Testa flag de emergency stop."""
        state_manager.set_emergency_stop(True)
        
        state = state_manager.state
        assert state.is_emergency_stopped == True
        assert state.is_running == False
    
    def test_corrupted_state_recovery(self, temp_dir):
        """Testa recuperação de estado corrompido."""
        state_file = os.path.join(temp_dir, 'state.json')
        backup_dir = os.path.join(temp_dir, 'backups')
        
        # Criar arquivo corrompido
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, 'w') as f:
            f.write('{ invalid json }}}')
        
        # Deve inicializar sem erro
        manager = StateManager(state_file=state_file, backup_dir=backup_dir)
        assert manager.state is not None


class TestReconnectionHandler:
    """Testes para ReconnectionHandler."""
    
    def test_exponential_backoff(self):
        """Testa que delay aumenta exponencialmente."""
        handler = ReconnectionHandler(base_delay=1.0, max_delay=60.0)
        
        delays = []
        for _ in range(5):
            if handler.should_retry():
                delays.append(handler.get_delay())
        
        # Delays devem aumentar (com alguma margem para jitter)
        for i in range(1, len(delays)):
            assert delays[i] > delays[i-1] * 0.5  # Permite jitter
    
    def test_jitter_varies_delay(self):
        """Testa que jitter adiciona variação."""
        handler = ReconnectionHandler(base_delay=10.0, jitter_range=0.3)
        
        # Mesmo attempt deve ter diferentes delays devido ao jitter
        delays = []
        for _ in range(10):
            h = ReconnectionHandler(base_delay=10.0, jitter_range=0.3)
            delays.append(h.get_delay())
        
        # Deve haver variação
        assert len(set(delays)) > 1
        
        # Delays devem estar no range esperado (10 ± 30%)
        for d in delays:
            assert 7.0 <= d <= 13.0
    
    def test_max_delay_cap(self):
        """Testa que delay não excede máximo."""
        handler = ReconnectionHandler(base_delay=1.0, max_delay=5.0, max_attempts=20)
        
        # Consumir tentativas
        for _ in range(10):
            if handler.should_retry():
                delay = handler.get_delay()
                assert delay <= 5.0 * 1.3  # Max + jitter
    
    def test_reset_clears_attempts(self):
        """Testa que reset zera contagem."""
        handler = ReconnectionHandler()
        
        handler.get_delay()
        handler.get_delay()
        assert handler.attempts == 2
        
        handler.reset()
        assert handler.attempts == 0
    
    def test_should_retry_respects_max_attempts(self):
        """Testa limite de tentativas."""
        handler = ReconnectionHandler(max_attempts=3)
        
        for _ in range(3):
            assert handler.should_retry()
            handler.get_delay()
        
        assert not handler.should_retry()


class TestHeartbeatMonitor:
    """Testes para HeartbeatMonitor."""
    
    @pytest.fixture
    def state_manager(self):
        """Mock StateManager."""
        mock = Mock(spec=StateManager)
        mock.state = SystemState()
        return mock
    
    def test_start_stop(self, state_manager):
        """Testa start e stop do monitor."""
        monitor = HeartbeatMonitor(state_manager, check_interval=1)
        
        monitor.start()
        assert monitor._running == True
        
        monitor.stop()
        assert monitor._running == False
    
    def test_registers_callbacks(self, state_manager):
        """Testa registro de callbacks."""
        monitor = HeartbeatMonitor(state_manager)
        
        callback1 = Mock()
        callback2 = Mock()
        
        monitor.on_websocket_dead(callback1)
        monitor.on_recovery_needed(callback2)
        
        assert callback1 in monitor._on_ws_dead_callbacks
        assert callback2 in monitor._on_recovery_callbacks


class TestResilienceManager:
    """Testes para ResilienceManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Cria diretório temporário."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)
    
    @pytest.fixture
    def resilience_manager(self, temp_dir):
        """Cria ResilienceManager com diretório temporário."""
        config = {
            'state_file': os.path.join(temp_dir, 'state.json'),
            'heartbeat_interval': 1,
            'ws_timeout': 5
        }
        return ResilienceManager(config)
    
    def test_emergency_stop(self, resilience_manager):
        """Testa emergency stop."""
        callback = Mock()
        resilience_manager.on_emergency_stop(callback)
        
        resilience_manager.emergency_stop("Test reason")
        
        state = resilience_manager.state_manager.state
        assert state.is_emergency_stopped == True
        callback.assert_called_once_with("Test reason")
    
    def test_clear_emergency_stop(self, resilience_manager):
        """Testa limpeza de emergency stop."""
        resilience_manager.emergency_stop("Test")
        resilience_manager.clear_emergency_stop()
        
        state = resilience_manager.state_manager.state
        assert state.is_emergency_stopped == False
    
    def test_record_ws_message(self, resilience_manager):
        """Testa registro de mensagem WebSocket."""
        resilience_manager.record_ws_message()
        
        state = resilience_manager.state_manager.state
        assert state.websocket_connected == True
        assert state.websocket_last_message is not None
    
    def test_get_status(self, resilience_manager):
        """Testa obtenção de status."""
        status = resilience_manager.get_status()
        
        assert 'is_running' in status
        assert 'is_emergency_stopped' in status
        assert 'websocket_connected' in status
        assert 'reconnection_attempts' in status


class TestGetSystemStatus:
    """Testes para função get_system_status."""
    
    def test_returns_error_for_missing_file(self):
        """Testa retorno de erro para arquivo inexistente."""
        status = get_system_status('/non/existent/file.json')
        assert 'error' in status
    
    def test_returns_state_for_valid_file(self, tmp_path):
        """Testa retorno de estado para arquivo válido."""
        state_file = tmp_path / 'state.json'
        state_file.write_text(json.dumps({
            'version': 1,
            'is_running': True
        }))
        
        status = get_system_status(str(state_file))
        assert status['version'] == 1
        assert status['is_running'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
