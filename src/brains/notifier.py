import os
import asyncio
from typing import Optional

from ..utils import load_config


class TelegramNotifier:
    def __init__(self):
        config = load_config()
        tg_cfg = config.get('telegram', {}) or {}
        self.enabled = bool(tg_cfg.get('enabled', False))
        self._bot_token_env = tg_cfg.get('bot_token_env', 'TELEGRAM_BOT_TOKEN')
        self._chat_id_env = tg_cfg.get('chat_id_env', 'TELEGRAM_CHAT_ID')
        self._flags = (tg_cfg.get('notifications') or {})
        self._token = os.getenv(self._bot_token_env)
        self._chat_id = os.getenv(self._chat_id_env)
        # Lazy import to avoid hard dependency if disabled
        self._bot = None

    def _ensure_client(self):
        if self._bot is None:
            try:
                import telegram
                self._bot = telegram.Bot(token=self._token)
            except Exception:
                self._bot = None

    def can_send(self, flag: str) -> bool:
        if not self.enabled:
            return False
        if not self._token or not self._chat_id:
            return False
        return bool(self._flags.get(flag, True))

    async def send_async(self, message: str, flag: Optional[str] = None) -> bool:
        if flag and not self.can_send(flag):
            return False
        if not self.enabled:
            return False
        if not self._token or not self._chat_id:
            return False
        try:
            self._ensure_client()
            if not self._bot:
                return False
            await self._bot.send_message(chat_id=self._chat_id, text=message)
            return True
        except Exception:
            # Silencioso – nunca quebra o pipeline
            return False

    def send(self, message: str, flag: Optional[str] = None) -> bool:
        if flag and not self.can_send(flag):
            return False
        if not self.enabled:
            return False
        if not self._token or not self._chat_id:
            return False
        try:
            self._ensure_client()
            if not self._bot:
                return False
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._bot.send_message(chat_id=self._chat_id, text=message))
            loop.close()
            return True
        except Exception:
            # Silencioso – nunca quebra o pipeline
            return False