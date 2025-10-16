"""
Battle Arena Helpers

Utilitários e funções auxiliares para Battle Arena.
"""

import re
from typing import Optional, Tuple, Dict, Any
from datetime import datetime


def format_price(price: float, decimals: int = 2) -> str:
    """
    Formata preço com número específico de casas decimais.

    Args:
        price: Preço a formatar
        decimals: Número de casas decimais

    Returns:
        Preço formatado como string
    """
    return f"{price:.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formata valor como percentual.

    Args:
        value: Valor (0.05 = 5%)
        decimals: Casas decimais

    Returns:
        Percentual formatado
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calcula mudança percentual.

    Args:
        old_value: Valor antigo
        new_value: Valor novo

    Returns:
        Mudança percentual (0.05 = 5%)
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else (1.0 if new_value > 0 else -1.0)
    return (new_value - old_value) / abs(old_value)


def validate_symbol_format(symbol: str) -> bool:
    """
    Valida formato de símbolo de trading.

    Args:
        symbol: Símbolo (ex: 'BTCUSDT')

    Returns:
        True se válido
    """
    # Padrão: BASEQUOTE onde BASE são letras, QUOTE é moeda conhecida
    pattern = r'^[A-Z]{2,10}(USDT|USD|BUSD|BTC|ETH|BNB)$'
    return bool(re.match(pattern, symbol))


def split_symbol(symbol: str) -> Tuple[str, str]:
    """
    Divide símbolo em base e quote.

    Args:
        symbol: Símbolo (ex: 'BTCUSDT')

    Returns:
        Tupla (base, quote)
    """
    # Encontrar onde começa a quote currency
    quotes = ['USDT', 'USD', 'BUSD', 'BTC', 'ETH', 'BNB']
    for quote in quotes:
        if symbol.endswith(quote):
            base = symbol[:-len(quote)]
            return base, quote

    # Fallback: assumir últimos 4 caracteres como quote
    return symbol[:-4], symbol[-4:]


def round_to_precision(value: float, precision: float) -> float:
    """
    Arredonda valor para precisão específica.

    Args:
        value: Valor a arredondar
        precision: Precisão (ex: 0.01 para 2 casas decimais)

    Returns:
        Valor arredondado
    """
    return round(value / precision) * precision


def get_timestamp() -> str:
    """
    Obtém timestamp formatado atual.

    Returns:
        Timestamp ISO formatado
    """
    return datetime.now().isoformat()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divisão segura (evita divisão por zero).

    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor padrão se denominator == 0

    Returns:
        Resultado da divisão ou default
    """
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Limita valor entre mínimo e máximo.

    Args:
        value: Valor a limitar
        min_value: Valor mínimo
        max_value: Valor máximo

    Returns:
        Valor limitado
    """
    return max(min_value, min(max_value, value))


def calculate_position_size(capital: float, risk_pct: float,
                          stop_loss_pct: float, entry_price: float) -> float:
    """
    Calcula tamanho de posição baseado em risco.

    Args:
        capital: Capital disponível
        risk_pct: Percentual de risco (0.01 = 1%)
        stop_loss_pct: Stop loss percentual (0.02 = 2%)
        entry_price: Preço de entrada

    Returns:
        Tamanho da posição em unidades
    """
    risk_amount = capital * risk_pct
    stop_loss_amount = entry_price * stop_loss_pct
    position_size = risk_amount / stop_loss_amount
    return position_size


def format_currency(amount: float, currency: str = 'USD') -> str:
    """
    Formata valor monetário.

    Args:
        amount: Valor
        currency: Moeda

    Returns:
        Valor formatado
    """
    return f"{currency} {amount:,.2f}"


def parse_order_type(order_type: str) -> str:
    """
    Normaliza tipo de ordem.

    Args:
        order_type: Tipo de ordem (market, limit, etc.)

    Returns:
        Tipo normalizado
    """
    order_type = order_type.lower().strip()
    valid_types = ['market', 'limit', 'stop', 'stop_limit']

    if order_type in valid_types:
        return order_type

    # Mapeamentos comuns
    mappings = {
        'mkt': 'market',
        'lmt': 'limit',
        'stp': 'stop'
    }

    return mappings.get(order_type, 'market')


def validate_quantity(quantity: float, min_qty: float = 0.000001,
                     max_qty: float = 1000000.0) -> bool:
    """
    Valida quantidade de ordem.

    Args:
        quantity: Quantidade
        min_qty: Quantidade mínima
        max_qty: Quantidade máxima

    Returns:
        True se válida
    """
    return min_qty <= quantity <= max_qty


def calculate_fee(amount: float, fee_rate: float) -> float:
    """
    Calcula taxa de operação.

    Args:
        amount: Valor da operação
        fee_rate: Taxa (0.001 = 0.1%)

    Returns:
        Valor da taxa
    """
    return amount * fee_rate


def get_exchange_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Obtém informações sobre símbolo (placeholder para futuras implementações).

    Args:
        symbol: Símbolo

    Returns:
        Dict com informações
    """
    # Placeholder - em implementação real, consultaria exchange
    base, quote = split_symbol(symbol)

    return {
        'symbol': symbol,
        'base_asset': base,
        'quote_asset': quote,
        'min_qty': 0.000001,
        'max_qty': 1000000.0,
        'step_size': 0.000001,
        'price_precision': 2,
        'quantity_precision': 6
    }


def log_with_timestamp(message: str, level: str = 'INFO') -> str:
    """
    Adiciona timestamp a mensagem de log.

    Args:
        message: Mensagem
        level: Nível do log

    Returns:
        Mensagem com timestamp
    """
    timestamp = get_timestamp()
    return f"[{timestamp}] {level}: {message}"