"""
AXON Binance WebSocket Connector

Advanced WebSocket client for real-time Binance data with:
- Automatic reconnection
- Symbol mapping (BTC-USD → BTCUSDT)
- Data validation and deduplication
- Historical data accumulation
- Error handling and logging
"""

import logging
import json
import os
import time
import threading
from datetime import datetime, timedelta
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
import pandas as pd

class BinanceWebSocketConnector:
    """Advanced Binance WebSocket connector with robust error handling and resilience."""

    def __init__(self, symbol='BTCUSDT', interval='1m', raw_dir='data/raw/rt',
                 resilience_manager=None):
        self.symbol = self._map_symbol(symbol)
        self.interval = interval
        self.raw_dir = raw_dir
        self.ndjson_path = os.path.join(raw_dir, f'{self.symbol}_{interval}.ndjson')
        self.twm = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds
        self.last_message_time = None
        self.message_count = 0
        
        # Resilience integration
        self.resilience_manager = resilience_manager
        self._jitter_range = 0.3  # ±30% jitter on reconnect delay
        
        # Connection event callbacks
        self._on_connect_callbacks = []
        self._on_disconnect_callbacks = []

        # Ensure directory exists
        os.makedirs(raw_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(f'BinanceWS.{self.symbol}')
    
    def on_connect(self, callback):
        """Register callback for successful connection."""
        self._on_connect_callbacks.append(callback)
    
    def on_disconnect(self, callback):
        """Register callback for disconnection."""
        self._on_disconnect_callbacks.append(callback)

    def _map_symbol(self, symbol):
        """Map various symbol formats to Binance format."""
        # Remove separators and normalize
        clean_symbol = symbol.replace('-', '').replace('/', '').replace('_', '')

        # Handle common mappings
        mappings = {
            'BTCUSD': 'BTCUSDT',
            'BTCUS': 'BTCUSDT',
            'ETHUSD': 'ETHUSDT',
            'ETHUS': 'ETHUSDT',
            'BNBUSD': 'BNBUSDT',
            'ADAUSD': 'ADAUSDT',
            'SOLUSD': 'SOLUSDT',
            'DOTUSD': 'DOTUSDT',
            'DOGEUSD': 'DOGEUSDT',
            'AVAXUSD': 'AVAXUSDT'
        }

        return mappings.get(clean_symbol.upper(), clean_symbol.upper() + 'T' if not clean_symbol.endswith('T') else clean_symbol.upper())

    def handle_socket_message(self, msg):
        """Handle incoming WebSocket messages with validation."""
        try:
            self.last_message_time = datetime.now()
            self.message_count += 1

            if msg.get('e') == 'error':
                self.logger.error(f'WebSocket error: {msg}')
                return

            if msg.get('e') != 'kline':
                return  # Only process kline events

            # Extract kline data
            symbol = msg.get('s')
            kline = msg.get('k', {})

            # Validate kline data
            if not kline.get('x', False):  # Only closed klines
                return

            # Prepare data for storage
            kline_data = {
                'event_time': msg.get('E'),
                'symbol': symbol,
                'interval': kline.get('i'),
                'open_time': kline.get('t'),
                'close_time': kline.get('T'),
                'first_trade_id': kline.get('f'),
                'last_trade_id': kline.get('L'),
                'open': float(kline.get('o', 0)),
                'high': float(kline.get('h', 0)),
                'low': float(kline.get('l', 0)),
                'close': float(kline.get('c', 0)),
                'volume': float(kline.get('v', 0)),
                'quote_volume': float(kline.get('q', 0)),
                'trade_count': kline.get('n', 0),
                'taker_buy_volume': float(kline.get('V', 0)),
                'taker_buy_quote_volume': float(kline.get('Q', 0)),
                'is_closed': kline.get('x', False),
                'received_at': datetime.now().isoformat()
            }

            # Validate data quality
            if self._validate_kline_data(kline_data):
                # Save to NDJSON
                with open(self.ndjson_path, 'a') as f:
                    json.dump(kline_data, f)
                    f.write('\n')

                if self.message_count % 100 == 0:
                    self.logger.info(f'Saved {self.message_count} klines to {self.ndjson_path}')
            else:
                self.logger.warning(f'Invalid kline data: {kline_data}')

        except Exception as e:
            self.logger.error(f'Error processing message: {e}')

    def _validate_kline_data(self, data):
        """Validate kline data quality."""
        try:
            # Check required fields
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                if field not in data or data[field] <= 0:
                    return False

            # Validate OHLC relationships
            if not (data['high'] >= data['low'] >= 0):
                return False
            if not (data['high'] >= data['open'] >= data['low']):
                return False
            if not (data['high'] >= data['close'] >= data['low']):
                return False

            return True
        except:
            return False

    def start(self):
        """Start the WebSocket connection with auto-reconnection."""
        self.is_running = True
        self.logger.info(f'Starting Binance WebSocket for {self.symbol} ({self.interval})')

        while self.is_running and self.reconnect_attempts < self.max_reconnect_attempts:
            # Check for emergency stop
            if self.resilience_manager and self.resilience_manager.state_manager.state.is_emergency_stopped:
                self.logger.warning("Emergency stop active - not connecting")
                time.sleep(5)
                continue
            
            try:
                self.twm = ThreadedWebsocketManager()
                self.twm.start()

                # Start kline socket
                self.twm.start_kline_socket(
                    callback=self.handle_socket_message,
                    symbol=self.symbol,
                    interval=self.interval
                )

                self.logger.info(f'WebSocket connected for {self.symbol}')
                self.reconnect_attempts = 0  # Reset on successful connection
                
                # Notify resilience manager
                if self.resilience_manager:
                    self.resilience_manager.record_ws_message()
                
                # Execute connect callbacks
                for callback in self._on_connect_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        self.logger.error(f'Connect callback failed: {e}')

                # Monitor connection health
                self._monitor_connection()

            except Exception as e:
                self.logger.error(f'WebSocket connection failed: {e}')
                self._handle_reconnection()

        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f'Max reconnection attempts reached for {self.symbol}')

    def _monitor_connection(self):
        """Monitor connection health and trigger reconnection if needed."""
        while self.is_running:
            # Check for emergency stop
            if self.resilience_manager and self.resilience_manager.state_manager.state.is_emergency_stopped:
                self.logger.warning("Emergency stop - stopping monitoring")
                self.stop()
                break
            
            time.sleep(30)  # Check every 30 seconds

            if self.last_message_time:
                time_since_last_msg = (datetime.now() - self.last_message_time).total_seconds()
                if time_since_last_msg > 300:  # 5 minutes without messages
                    self.logger.warning(f'No messages received for {time_since_last_msg:.0f}s, reconnecting...')
                    self._handle_reconnection()
                    break

    def _handle_reconnection(self):
        """Handle reconnection logic with exponential backoff + jitter."""
        import random
        
        # Execute disconnect callbacks
        for callback in self._on_disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f'Disconnect callback failed: {e}')
        
        # Notify resilience manager
        if self.resilience_manager:
            self.resilience_manager.record_ws_disconnect()
        
        if self.twm:
            try:
                self.twm.stop()
            except:
                pass

        self.reconnect_attempts += 1
        if self.reconnect_attempts < self.max_reconnect_attempts:
            # Exponential backoff with jitter
            base_delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
            max_delay = 60  # Cap at 60 seconds
            base_delay = min(base_delay, max_delay)
            
            # Add jitter (±30%) to prevent thundering herd
            jitter = base_delay * self._jitter_range * (2 * random.random() - 1)
            delay = max(0.5, base_delay + jitter)
            
            self.logger.info(f'Reconnecting in {delay:.1f}s (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})')
            time.sleep(delay)


    def stop(self):
        """Stop the WebSocket connection."""
        self.is_running = False
        if self.twm:
            try:
                self.twm.stop()
                self.logger.info(f'WebSocket stopped for {self.symbol}')
            except Exception as e:
                self.logger.error(f'Error stopping WebSocket: {e}')

def start_binance_ws(symbol='BTC-USD', interval='1m'):
    """Start Binance WebSocket connector (legacy function for compatibility)."""
    connector = BinanceWebSocketConnector(symbol=symbol, interval=interval)
    connector.start()

def load_binance_ws_data(config):
    """
    Load historical data from Binance WebSocket NDJSON files.
    Enhanced version with better data processing and validation.
    """
    # Map symbol
    cfg_symbol = config['data']['symbols'][0] if isinstance(config['data'].get('symbols'), list) and config['data']['symbols'] else 'BTC-USD'
    connector = BinanceWebSocketConnector(symbol=cfg_symbol)
    symbol = connector.symbol

    interval = config['data']['interval']
    lookback_days = config['data']['lookback_days']
    raw_dir = 'data/raw/rt'
    ndjson_path = os.path.join(raw_dir, f'{symbol}_{interval}.ndjson')

    if not os.path.exists(ndjson_path):
        logging.warning(f'Arquivo NDJSON não encontrado: {ndjson_path}')
        return pd.DataFrame()

    start_time = datetime.now() - timedelta(days=lookback_days)
    data = []
    processed_count = 0
    valid_count = 0

    with open(ndjson_path, 'r') as f:
        for line in f:
            processed_count += 1
            try:
                msg = json.loads(line.strip())

                # Extract kline data
                if 'k' in msg:
                    k = msg['k']
                    open_time = datetime.fromtimestamp(k['t'] / 1000)

                    if open_time >= start_time:
                        kline_data = {
                            'timestamp': open_time,
                            'open': float(k['o']),
                            'high': float(k['h']),
                            'low': float(k['l']),
                            'close': float(k['c']),
                            'volume': float(k['v'])
                        }

                        # Additional validation
                        if (kline_data['high'] >= kline_data['low'] and
                            kline_data['open'] > 0 and
                            kline_data['close'] > 0 and
                            kline_data['volume'] >= 0):
                            data.append(kline_data)
                            valid_count += 1

            except json.JSONDecodeError:
                continue
            except KeyError:
                continue

    if not data:
        logging.warning(f'Nenhum dado válido encontrado em {ndjson_path}')
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Deduplicate by timestamp keeping the last occurrence
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)

    logging.info(f'Dados Binance WS carregados: {len(df)} registros válidos de {processed_count} processados')
    return df

if __name__ == '__main__':
    # Example usage
    connector = BinanceWebSocketConnector(symbol='BTC-USD', interval='1m')
    try:
        connector.start()
    except KeyboardInterrupt:
        connector.stop()