"""AXON Configuration Manager"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Configuration manager for AXON system."""
    
    def __init__(self, config_path: str = "axon.cfg.yml"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._config = None
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            return self._config
        except FileNotFoundError:
            # Create default config if not exists
            self._config = self._create_default_config()
            self.save_config()
            return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self._config is None:
            self.load_config()
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        if self._config is None:
            self.load_config()
        
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self) -> None:
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'binance': {
                'api_key': '',
                'api_secret': '',
                'testnet': True,
                'base_url': 'https://testnet.binance.vision'
            },
            'trading': {
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'timeframes': ['1m', '5m'],
                'max_position_size': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            },
            'risk_management': {
                'max_daily_loss': 0.05,
                'max_drawdown': 0.10,
                'position_sizing': 'fixed'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/axon.log'
            }
        }