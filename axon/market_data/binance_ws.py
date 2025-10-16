"""
Binance WebSocket Connector for real-time market data streaming.
Integrates with BattleArena to provide live klines data.
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, Any, Callable, List, Optional
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)

class BinanceWebSocketConnector:
    """
    WebSocket connector for Binance to stream real-time klines data.
    Supports multiple symbols and intervals with automatic reconnection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = "wss://stream.binance.com:9443/ws/"
        self.testnet_url = "wss://testnet.binance.vision/ws/"
        
        # Use testnet if configured
        self.use_testnet = config.get("battle_arena", {}).get("execution", {}).get("testnet", True)
        self.ws_url = self.testnet_url if self.use_testnet else self.base_url
        
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        
        # Active streams
        self.active_streams = {}
        self.callback_function = None
        
        logger.info(f"BinanceWebSocketConnector initialized with {'testnet' if self.use_testnet else 'mainnet'}")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for Binance API."""
        return symbol.upper().replace("/", "")
    
    def _normalize_interval(self, interval: str) -> str:
        """Normalize interval format for Binance API."""
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        return interval_map.get(interval.lower(), "1m")
    
    def _create_stream_name(self, symbol: str, interval: str) -> str:
        """Create stream name for Binance WebSocket."""
        normalized_symbol = self._normalize_symbol(symbol).lower()
        normalized_interval = self._normalize_interval(interval)
        return f"{normalized_symbol}@kline_{normalized_interval}"
    
    def _parse_kline_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Binance kline data to standardized format."""
        try:
            kline = data.get("k", {})
            
            parsed_data = {
                "symbol": kline.get("s"),
                "open_time": int(kline.get("t")),
                "close_time": int(kline.get("T")),
                "open": float(kline.get("o")),
                "high": float(kline.get("h")),
                "low": float(kline.get("l")),
                "close": float(kline.get("c")),
                "volume": float(kline.get("v")),
                "interval": kline.get("i"),
                "is_closed": kline.get("x", False)  # True when kline is closed
            }
            
            return parsed_data
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing kline data: {e}")
            return {}
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            # Handle kline data
            if "k" in data:
                parsed_kline = self._parse_kline_data(data)
                if parsed_kline and self.callback_function:
                    # Group by symbol for callback
                    symbol = parsed_kline["symbol"]
                    klines_data = {symbol: parsed_kline}
                    
                    # Call the callback function
                    try:
                        self.callback_function(klines_data)
                    except Exception as e:
                        logger.error(f"Error in callback function: {e}")
            
            # Handle ping/pong
            elif "ping" in data:
                pong_message = {"pong": data["ping"]}
                await self.websocket.send(json.dumps(pong_message))
                logger.debug("Responded to ping with pong")
                
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _connect_websocket(self, stream_names: List[str]):
        """Establish WebSocket connection."""
        try:
            # Create combined stream URL
            if len(stream_names) == 1:
                url = f"{self.ws_url}{stream_names[0]}"
            else:
                streams = "/".join(stream_names)
                url = f"{self.ws_url.replace('/ws/', '/stream?streams=')}{streams}"
            
            logger.info(f"Connecting to WebSocket: {url}")
            
            self.websocket = await websockets.connect(url)
            self.is_connected = True
            self.reconnect_attempts = 0
            
            logger.info("WebSocket connection established")
            
            # Listen for messages
            async for message in self.websocket:
                await self._handle_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
            await self._handle_reconnection(stream_names)
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_connected = False
            await self._handle_reconnection(stream_names)
    
    async def _handle_reconnection(self, stream_names: List[str]):
        """Handle WebSocket reconnection logic."""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            
            await asyncio.sleep(self.reconnect_delay)
            await self._connect_websocket(stream_names)
        else:
            logger.error("Max reconnection attempts reached. Stopping WebSocket connection.")
    
    def start_klines_stream(self, symbols: List[str], interval: str, callback: Callable):
        """
        Start streaming klines data for specified symbols and interval.
        
        Args:
            symbols: List of trading symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
            interval: Time interval (e.g., '1m', '5m')
            callback: Function to call with received data
        """
        if not symbols:
            logger.warning("No symbols provided for streaming")
            return
        
        self.callback_function = callback
        
        # Create stream names
        stream_names = []
        for symbol in symbols:
            stream_name = self._create_stream_name(symbol, interval)
            stream_names.append(stream_name)
            self.active_streams[symbol] = {
                "interval": interval,
                "stream_name": stream_name
            }
        
        logger.info(f"Starting klines stream for symbols: {symbols}, interval: {interval}")
        
        # Start WebSocket connection in a separate thread
        def run_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._connect_websocket(stream_names))
            except Exception as e:
                logger.error(f"Error running WebSocket: {e}")
            finally:
                loop.close()
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        # Give some time for connection to establish
        time.sleep(2)
        
        if self.is_connected:
            logger.info("Klines stream started successfully")
        else:
            logger.warning("Failed to establish WebSocket connection")
    
    def stop_stream(self):
        """Stop the WebSocket stream."""
        if self.websocket and not self.websocket.closed:
            asyncio.create_task(self.websocket.close())
        
        self.is_connected = False
        self.active_streams.clear()
        self.callback_function = None
        
        logger.info("WebSocket stream stopped")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            "is_connected": self.is_connected,
            "active_streams": list(self.active_streams.keys()),
            "reconnect_attempts": self.reconnect_attempts,
            "use_testnet": self.use_testnet
        }