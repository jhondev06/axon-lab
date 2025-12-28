"""
AXON Battle Arena - Telegram Bot Command Handler

Handler para comandos de controle via Telegram:
- /stop - Emergency stop (kill switch)
- /start - Resume trading
- /status - Status atual do bot
- /positions - Lista posições abertas
- /balance - Saldo atual
"""

import asyncio
import logging
import threading
from typing import Optional, Dict, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional import - graceful degradation if telegram not installed
try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed - Telegram commands disabled")


class TelegramCommandBot:
    """
    Bot Telegram para controle remoto do Battle Arena.
    
    Permite:
    - Kill switch de emergência via /stop
    - Status check via /status
    - Resume trading via /start
    """
    
    def __init__(self,
                 bot_token: str,
                 allowed_chat_ids: list,
                 resilience_manager: Optional[Any] = None,
                 paper_trader: Optional[Any] = None):
        """
        Initialize Telegram command bot.
        
        Args:
            bot_token: Telegram bot token
            allowed_chat_ids: List of authorized chat IDs
            resilience_manager: ResilienceManager instance
            paper_trader: PaperTrader instance
        """
        if not TELEGRAM_AVAILABLE:
            raise RuntimeError("python-telegram-bot not installed")
        
        self.bot_token = bot_token
        self.allowed_chat_ids = [int(cid) for cid in allowed_chat_ids]
        self.resilience_manager = resilience_manager
        self.paper_trader = paper_trader
        
        self._app: Optional[Application] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # Custom callbacks
        self._on_stop_callbacks: list = []
        self._on_start_callbacks: list = []
    
    def on_stop(self, callback: Callable) -> None:
        """Register callback for /stop command."""
        self._on_stop_callbacks.append(callback)
    
    def on_start(self, callback: Callable) -> None:
        """Register callback for /start command."""
        self._on_start_callbacks.append(callback)
    
    def start(self) -> None:
        """Start the bot in a background thread."""
        if self._running:
            logger.warning("Telegram bot already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_bot, daemon=True)
        self._thread.start()
        logger.info("Telegram command bot started")
    
    def stop(self) -> None:
        """Stop the bot."""
        self._running = False
        if self._app:
            # Thread-safe stop
            asyncio.run_coroutine_threadsafe(
                self._app.stop(),
                self._app.loop
            )
        logger.info("Telegram command bot stopped")
    
    def _run_bot(self) -> None:
        """Run bot in event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._setup_and_run())
        except Exception as e:
            logger.error(f"Telegram bot error: {e}")
        finally:
            loop.close()
    
    async def _setup_and_run(self) -> None:
        """Setup handlers and start polling."""
        self._app = Application.builder().token(self.bot_token).build()
        
        # Register command handlers
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("balance", self._cmd_balance))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        
        # Start polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        
        logger.info("Telegram bot polling started")
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
        
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
    
    def _is_authorized(self, update: Update) -> bool:
        """Check if user is authorized."""
        chat_id = update.effective_chat.id
        authorized = chat_id in self.allowed_chat_ids
        
        if not authorized:
            logger.warning(f"Unauthorized command attempt from chat_id: {chat_id}")
        
        return authorized
    
    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stop command - EMERGENCY STOP."""
        if not self._is_authorized(update):
            await update.message.reply_text("⛔ Não autorizado")
            return
        
        logger.critical("EMERGENCY STOP command received via Telegram")
        
        # Trigger resilience manager emergency stop
        if self.resilience_manager:
            self.resilience_manager.emergency_stop("Telegram /stop command")
        
        # Execute callbacks
        for callback in self._on_stop_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Stop callback error: {e}")
        
        await update.message.reply_text(
            "🛑 EMERGENCY STOP ATIVADO\n\n"
            "• Todas operações pausadas\n"
            "• WebSocket desconectado\n"
            "• Posições mantidas\n\n"
            "Use /start para retomar"
        )
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command - Resume trading."""
        if not self._is_authorized(update):
            await update.message.reply_text("⛔ Não autorizado")
            return
        
        logger.info("Resume command received via Telegram")
        
        # Clear emergency stop
        if self.resilience_manager:
            self.resilience_manager.clear_emergency_stop()
        
        # Execute callbacks
        for callback in self._on_start_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Start callback error: {e}")
        
        await update.message.reply_text(
            "▶️ Trading RESUMIDO\n\n"
            "• Emergency stop desativado\n"
            "• Sistema pronto para operar"
        )
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not self._is_authorized(update):
            await update.message.reply_text("⛔ Não autorizado")
            return
        
        status = self._get_status()
        
        # Format status message
        emoji = "🟢" if status.get('is_running') else "🔴"
        ws_emoji = "🔗" if status.get('websocket_connected') else "❌"
        stop_emoji = "🛑" if status.get('is_emergency_stopped') else "✅"
        
        msg = (
            f"{emoji} **AXON Status**\n\n"
            f"{stop_emoji} Emergency Stop: {'ATIVO' if status.get('is_emergency_stopped') else 'OK'}\n"
            f"{ws_emoji} WebSocket: {'Conectado' if status.get('websocket_connected') else 'Desconectado'}\n"
            f"⏰ Último heartbeat: {status.get('last_heartbeat', 'N/A')}\n"
            f"🔄 Reconexões: {status.get('reconnection_attempts', 0)}\n"
            f"📊 Posições: {status.get('active_positions', 0)}\n"
            f"📝 Ordens: {status.get('pending_orders', 0)}\n"
        )
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /positions command."""
        if not self._is_authorized(update):
            await update.message.reply_text("⛔ Não autorizado")
            return
        
        if not self.paper_trader:
            await update.message.reply_text("📊 Paper trader não disponível")
            return
        
        try:
            positions = self.paper_trader.get_positions()
            
            if not positions:
                await update.message.reply_text("📊 Nenhuma posição aberta")
                return
            
            msg = "📊 **Posições Abertas**\n\n"
            
            for pos in positions:
                pnl_emoji = "🟢" if pos.pnl >= 0 else "🔴"
                msg += (
                    f"**{pos.symbol}**\n"
                    f"  Qty: {pos.quantity:.6f}\n"
                    f"  Entry: ${pos.entry_price:.2f}\n"
                    f"  Current: ${pos.current_price:.2f}\n"
                    f"  {pnl_emoji} PnL: ${pos.pnl:.2f} ({pos.pnl_percentage:.2f}%)\n\n"
                )
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Erro: {e}")
    
    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /balance command."""
        if not self._is_authorized(update):
            await update.message.reply_text("⛔ Não autorizado")
            return
        
        if not self.paper_trader:
            await update.message.reply_text("💰 Paper trader não disponível")
            return
        
        try:
            balances = self.paper_trader.get_balance()
            stats = self.paper_trader.get_performance_stats()
            
            msg = "💰 **Saldo Atual**\n\n"
            
            for bal in balances:
                msg += f"  {bal.asset}: {bal.total:.4f}\n"
            
            msg += f"\n📈 **Performance**\n"
            msg += f"  Capital Inicial: ${stats['initial_capital']:.2f}\n"
            msg += f"  Saldo Atual: ${stats['current_balance']:.2f}\n"
            msg += f"  Retorno: {stats['total_return_pct']:.2f}%\n"
            msg += f"  Win Rate: {stats['win_rate']:.1f}%\n"
            msg += f"  Total Trades: {stats['total_trades']}\n"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Erro: {e}")
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not self._is_authorized(update):
            await update.message.reply_text("⛔ Não autorizado")
            return
        
        msg = (
            "🤖 **AXON Trading Bot Commands**\n\n"
            "/stop - 🛑 Emergency stop (kill switch)\n"
            "/start - ▶️ Resume trading\n"
            "/status - 📊 System status\n"
            "/positions - 📈 Open positions\n"
            "/balance - 💰 Account balance\n"
            "/help - ❓ This help\n"
        )
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    def _get_status(self) -> Dict[str, Any]:
        """Get system status."""
        if self.resilience_manager:
            return self.resilience_manager.get_status()
        
        return {
            'is_running': True,
            'is_emergency_stopped': False,
            'websocket_connected': False,
            'last_heartbeat': datetime.now().isoformat(),
            'reconnection_attempts': 0,
            'active_positions': 0,
            'pending_orders': 0
        }


def create_telegram_bot(config: Dict[str, Any],
                        resilience_manager: Optional[Any] = None,
                        paper_trader: Optional[Any] = None) -> Optional[TelegramCommandBot]:
    """
    Factory function to create Telegram bot from config.
    
    Args:
        config: AXON configuration dict
        resilience_manager: ResilienceManager instance
        paper_trader: PaperTrader instance
    
    Returns:
        TelegramCommandBot or None if not configured
    """
    if not TELEGRAM_AVAILABLE:
        logger.warning("Telegram bot skipped - library not installed")
        return None
    
    import os
    
    tg_config = config.get('telegram', {})
    
    if not tg_config.get('enabled', False):
        logger.info("Telegram bot disabled in config")
        return None
    
    # Get credentials from environment
    token_env = tg_config.get('bot_token_env', 'TELEGRAM_BOT_TOKEN')
    chat_id_env = tg_config.get('chat_id_env', 'TELEGRAM_CHAT_ID')
    
    bot_token = os.getenv(token_env)
    chat_id = os.getenv(chat_id_env)
    
    if not bot_token or not chat_id:
        logger.warning(f"Telegram credentials not found in env: {token_env}, {chat_id_env}")
        return None
    
    try:
        bot = TelegramCommandBot(
            bot_token=bot_token,
            allowed_chat_ids=[chat_id],
            resilience_manager=resilience_manager,
            paper_trader=paper_trader
        )
        return bot
        
    except Exception as e:
        logger.error(f"Failed to create Telegram bot: {e}")
        return None
