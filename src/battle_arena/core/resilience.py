"""
AXON Battle Arena - Resilience Module

Sistema centralizado de resiliência para trading live:
- StateManager: Persistência de estado com atomic writes
- HeartbeatMonitor: Health check com alertas
- ResilienceManager: Orquestrador de resiliência
"""

import json
import os
import time
import shutil
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict, field
import random

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Estado completo do sistema para persistência."""
    version: int = 1
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    is_running: bool = False
    is_emergency_stopped: bool = False
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Connection state
    websocket_connected: bool = False
    websocket_last_message: Optional[str] = None
    reconnection_attempts: int = 0
    
    # Trading state
    active_positions: Dict[str, Any] = field(default_factory=dict)
    pending_orders: Dict[str, Any] = field(default_factory=dict)
    last_signal: Optional[Dict[str, Any]] = None
    
    # Stats
    total_trades: int = 0
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        # Handle version migration
        version = data.get('version', 1)
        if version < cls.__dataclass_fields__['version'].default:
            data = cls._migrate_state(data, version)
        
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)
    
    @staticmethod
    def _migrate_state(data: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """Migrate state from older versions."""
        # Add migrations here as needed
        logger.info(f"Migrating state from version {from_version}")
        data['version'] = 1
        return data


class StateManager:
    """
    Gerenciador de estado com persistência atômica.
    
    Features:
    - Atomic writes (write to temp, then rename)
    - Automatic backups before overwrite
    - Corruption recovery
    - Version migration
    """
    
    def __init__(self, 
                 state_file: str = "data/battle_arena/system_state.json",
                 backup_dir: str = "data/battle_arena/backups",
                 max_backups: int = 10):
        self.state_file = Path(state_file)
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self._state: Optional[SystemState] = None
        self._lock = threading.Lock()
        
        # Ensure directories exist
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create initial state
        self._load_state()
    
    @property
    def state(self) -> SystemState:
        """Get current state (thread-safe)."""
        with self._lock:
            if self._state is None:
                self._state = SystemState()
            return self._state
    
    def update(self, **kwargs) -> None:
        """Update state fields and persist."""
        with self._lock:
            if self._state is None:
                self._state = SystemState()
            
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
                else:
                    logger.warning(f"Unknown state field: {key}")
            
            self._state.last_updated = datetime.now().isoformat()
            self._save_state()
    
    def mark_heartbeat(self) -> None:
        """Mark heartbeat timestamp."""
        self.update(last_heartbeat=datetime.now().isoformat())
    
    def set_emergency_stop(self, stopped: bool = True) -> None:
        """Set emergency stop flag."""
        self.update(is_emergency_stopped=stopped, is_running=not stopped)
        logger.warning(f"Emergency stop {'ACTIVATED' if stopped else 'DEACTIVATED'}")
    
    def _save_state(self) -> None:
        """Save state with atomic write."""
        try:
            # Create backup first
            if self.state_file.exists():
                self._create_backup()
            
            # Write to temp file
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self._state.to_dict(), f, indent=2, default=str)
            
            # Atomic rename
            temp_file.replace(self.state_file)
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise
    
    def _load_state(self) -> None:
        """Load state from file with corruption recovery."""
        if not self.state_file.exists():
            logger.info("No existing state file, starting fresh")
            self._state = SystemState()
            return
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            self._state = SystemState.from_dict(data)
            logger.info(f"State loaded: version={self._state.version}, "
                       f"last_updated={self._state.last_updated}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted state file, attempting recovery: {e}")
            self._recover_from_backup()
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            self._state = SystemState()
    
    def _create_backup(self) -> None:
        """Create backup of current state file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"state_backup_{timestamp}.json"
            shutil.copy2(self.state_file, backup_file)
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files."""
        try:
            backups = sorted(self.backup_dir.glob("state_backup_*.json"))
            while len(backups) > self.max_backups:
                oldest = backups.pop(0)
                oldest.unlink()
                logger.debug(f"Removed old backup: {oldest}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup backups: {e}")
    
    def _recover_from_backup(self) -> None:
        """Attempt to recover state from backup."""
        try:
            backups = sorted(self.backup_dir.glob("state_backup_*.json"), reverse=True)
            
            for backup in backups:
                try:
                    with open(backup, 'r') as f:
                        data = json.load(f)
                    self._state = SystemState.from_dict(data)
                    logger.info(f"State recovered from backup: {backup}")
                    return
                    
                except Exception:
                    continue
            
            logger.warning("No valid backup found, starting fresh")
            self._state = SystemState()
            
        except Exception as e:
            logger.error(f"Backup recovery failed: {e}")
            self._state = SystemState()


class HeartbeatMonitor:
    """
    Monitor de heartbeat para detectar problemas.
    
    Features:
    - Periodic health checks
    - WebSocket liveness detection
    - Telegram alerts on issues
    - Auto-recovery triggers
    """
    
    def __init__(self,
                 state_manager: StateManager,
                 check_interval: int = 30,
                 ws_timeout: int = 300,
                 notifier: Optional[Any] = None):
        self.state_manager = state_manager
        self.check_interval = check_interval
        self.ws_timeout = ws_timeout  # 5 minutes
        self.notifier = notifier
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_ws_dead_callbacks: List[Callable] = []
        self._on_recovery_callbacks: List[Callable] = []
    
    def start(self) -> None:
        """Start heartbeat monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Heartbeat monitor started")
    
    def stop(self) -> None:
        """Stop heartbeat monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Heartbeat monitor stopped")
    
    def on_websocket_dead(self, callback: Callable) -> None:
        """Register callback for dead WebSocket detection."""
        self._on_ws_dead_callbacks.append(callback)
    
    def on_recovery_needed(self, callback: Callable) -> None:
        """Register callback for recovery trigger."""
        self._on_recovery_callbacks.append(callback)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        consecutive_failures = 0
        
        while self._running:
            try:
                self._check_health()
                consecutive_failures = 0
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Health check failed ({consecutive_failures}): {e}")
                
                if consecutive_failures >= 3:
                    self._trigger_recovery()
            
            time.sleep(self.check_interval)
    
    def _check_health(self) -> None:
        """Perform health check."""
        state = self.state_manager.state
        
        # Update heartbeat
        self.state_manager.mark_heartbeat()
        
        # Check emergency stop
        if state.is_emergency_stopped:
            logger.info("System in emergency stop mode")
            return
        
        # Check WebSocket liveness
        if state.websocket_last_message:
            last_msg_time = datetime.fromisoformat(state.websocket_last_message)
            seconds_since = (datetime.now() - last_msg_time).total_seconds()
            
            if seconds_since > self.ws_timeout:
                logger.warning(f"WebSocket dead: no message for {seconds_since:.0f}s")
                self._handle_ws_dead(seconds_since)
    
    def _handle_ws_dead(self, seconds_since: float) -> None:
        """Handle dead WebSocket."""
        # Send alert
        if self.notifier:
            try:
                self.notifier.send(
                    f"⚠️ AXON Alert: WebSocket sem mensagens há {seconds_since/60:.1f} min",
                    "errors"
                )
            except Exception:
                pass
        
        # Trigger callbacks
        for callback in self._on_ws_dead_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"WS dead callback failed: {e}")
    
    def _trigger_recovery(self) -> None:
        """Trigger recovery procedures."""
        logger.warning("Triggering recovery procedures")
        
        if self.notifier:
            try:
                self.notifier.send(
                    "🔄 AXON: Iniciando procedimentos de recuperação",
                    "errors"
                )
            except Exception:
                pass
        
        for callback in self._on_recovery_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")


class ReconnectionHandler:
    """
    Handler de reconexão com exponential backoff + jitter.
    
    Previne thundering herd problem quando múltiplas conexões
    tentam reconectar ao mesmo tempo.
    """
    
    def __init__(self,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 max_attempts: int = 10,
                 jitter_range: float = 0.3):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.jitter_range = jitter_range
        
        self._attempts = 0
        self._last_attempt: Optional[datetime] = None
    
    def reset(self) -> None:
        """Reset attempt counter on successful connection."""
        self._attempts = 0
        self._last_attempt = None
    
    def should_retry(self) -> bool:
        """Check if should attempt reconnection."""
        return self._attempts < self.max_attempts
    
    def get_delay(self) -> float:
        """
        Get next delay with exponential backoff + jitter.
        
        Jitter prevents thundering herd when multiple clients
        reconnect simultaneously after an outage.
        """
        # Exponential backoff
        delay = min(self.base_delay * (2 ** self._attempts), self.max_delay)
        
        # Add jitter (±30% by default)
        jitter = delay * self.jitter_range * (2 * random.random() - 1)
        final_delay = max(0.1, delay + jitter)
        
        self._attempts += 1
        self._last_attempt = datetime.now()
        
        logger.info(f"Reconnection attempt {self._attempts}/{self.max_attempts}, "
                   f"delay: {final_delay:.2f}s")
        
        return final_delay
    
    @property
    def attempts(self) -> int:
        return self._attempts


class ResilienceManager:
    """
    Orquestrador central de resiliência.
    
    Coordena StateManager, HeartbeatMonitor e reconexões.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        state_file = self.config.get('state_file', 'data/battle_arena/system_state.json')
        self.state_manager = StateManager(state_file=state_file)
        
        self.heartbeat = HeartbeatMonitor(
            state_manager=self.state_manager,
            check_interval=self.config.get('heartbeat_interval', 30),
            ws_timeout=self.config.get('ws_timeout', 300)
        )
        
        self.reconnection = ReconnectionHandler(
            base_delay=self.config.get('reconnect_base_delay', 1.0),
            max_delay=self.config.get('reconnect_max_delay', 60.0),
            max_attempts=self.config.get('reconnect_max_attempts', 10)
        )
        
        self._emergency_stop_callbacks: List[Callable] = []
        
        logger.info("ResilienceManager initialized")
    
    def start(self) -> None:
        """Start resilience monitoring."""
        # Check for previous emergency stop
        if self.state_manager.state.is_emergency_stopped:
            logger.warning("Previous emergency stop detected - NOT auto-starting")
            return
        
        self.state_manager.update(is_running=True)
        self.heartbeat.start()
        logger.info("ResilienceManager started")
    
    def stop(self) -> None:
        """Stop resilience monitoring."""
        self.heartbeat.stop()
        self.state_manager.update(is_running=False)
        logger.info("ResilienceManager stopped")
    
    def emergency_stop(self, reason: str = "Manual stop") -> None:
        """Trigger emergency stop."""
        logger.critical(f"EMERGENCY STOP: {reason}")
        
        self.state_manager.set_emergency_stop(True)
        
        # Execute callbacks
        for callback in self._emergency_stop_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Emergency stop callback failed: {e}")
        
        self.heartbeat.stop()
    
    def clear_emergency_stop(self) -> None:
        """Clear emergency stop flag."""
        self.state_manager.set_emergency_stop(False)
        logger.info("Emergency stop cleared")
    
    def on_emergency_stop(self, callback: Callable[[str], None]) -> None:
        """Register emergency stop callback."""
        self._emergency_stop_callbacks.append(callback)
    
    def record_ws_message(self) -> None:
        """Record WebSocket message timestamp."""
        self.state_manager.update(
            websocket_connected=True,
            websocket_last_message=datetime.now().isoformat(),
            reconnection_attempts=0
        )
        self.reconnection.reset()
    
    def record_ws_disconnect(self) -> None:
        """Record WebSocket disconnection."""
        self.state_manager.update(websocket_connected=False)
    
    def get_reconnect_delay(self) -> Optional[float]:
        """Get delay for next reconnection attempt."""
        if not self.reconnection.should_retry():
            return None
        
        delay = self.reconnection.get_delay()
        self.state_manager.update(reconnection_attempts=self.reconnection.attempts)
        return delay
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        state = self.state_manager.state
        return {
            'is_running': state.is_running,
            'is_emergency_stopped': state.is_emergency_stopped,
            'websocket_connected': state.websocket_connected,
            'last_heartbeat': state.last_heartbeat,
            'reconnection_attempts': state.reconnection_attempts,
            'session_start': state.session_start,
            'active_positions': len(state.active_positions),
            'pending_orders': len(state.pending_orders)
        }


# Convenience function for quick status check
def get_system_status(state_file: str = "data/battle_arena/system_state.json") -> Dict[str, Any]:
    """Quick status check without full initialization."""
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {'error': 'State file not found or corrupted'}
