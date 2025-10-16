"""
Battle Arena Configuration Loader
Carrega e valida configurações da Battle Arena.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_battle_config(config_path: str = "src/battle_arena/config/battle_config.yml") -> Dict[str, Any]:
    """
    Carrega configuração da Battle Arena.

    Args:
        config_path: Caminho para arquivo de configuração

    Returns:
        Dicionário com configurações
    """
    config_file = Path(config_path)

    if not config_file.exists():
        # Retorna configuração padrão se arquivo não existir
        return _get_default_config()

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Resolver variáveis de ambiente
        config = _resolve_env_vars(config)

        # Validar configuração
        _validate_config(config)

        return config

    except Exception as e:
        print(f"Erro ao carregar configuração Battle Arena: {e}")
        return _get_default_config()


def _resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve variáveis de ambiente na configuração."""
    if not isinstance(config, dict):
        return config

    resolved = {}
    for key, value in config.items():
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            # Extrair nome da variável
            env_var = value[2:-1]
            resolved[key] = os.environ.get(env_var, '')
        elif isinstance(value, dict):
            resolved[key] = _resolve_env_vars(value)
        else:
            resolved[key] = value

    return resolved


def _validate_config(config: Dict[str, Any]) -> None:
    """Valida configuração básica."""
    required_sections = ['execution', 'exchanges', 'risk', 'paper_trading']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Seção obrigatória faltando: {section}")

    # Validar modo de execução
    execution_mode = config['execution']['mode']
    if execution_mode not in ['paper', 'live', 'simulation']:
        raise ValueError(f"Modo de execução inválido: {execution_mode}")


def _get_default_config() -> Dict[str, Any]:
    """Retorna configuração padrão."""
    return {
        'execution': {
            'mode': 'paper',
            'testnet': True
        },
        'exchanges': {
            'binance': {
                'enabled': True,
                'api_key': '',
                'api_secret': '',
                'testnet': True,
                'rate_limit': 10
            }
        },
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'data_interval': '1m',
        'risk': {
            'max_position_size_pct': 0.1,
            'max_total_exposure_pct': 0.5,
            'max_daily_loss_pct': 0.05,
            'max_drawdown_pct': 0.1,
            'max_orders_per_hour': 10,
            'min_order_size_usd': 10.0,
            'max_order_size_usd': 1000.0
        },
        'paper_trading': {
            'initial_capital': 10000.0,
            'fee_rate': 0.001,
            'state_file': 'data/battle_arena/paper_trader_state.json'
        },
        'signals': {
            'buy_threshold': 0.6,
            'sell_threshold': 0.4,
            'min_confidence': 0.55,
            'max_position_size': 0.1
        },
        'buffer_size': 1000,
        'logging': {
            'level': 'INFO',
            'file': 'logs/battle_arena.log'
        }
    }