"""
Paz Bot - 創作アイデア専門エージェント

役割:
- LangGraphワークフローによる創作支援
- インスピレーション・ブレインストーミング
- 革新的解決・アイデア発想
- 効率的直接応答（中間投稿なし）
"""

import logging
from typing import Optional, Dict, Any

import discord

from ..base_bot import BaseBot
from ...core.bot_config import BotConfig


class PazBot(BaseBot):
    """Paz Bot - 創作アイデア専門エージェント"""
    
    def __init__(
        self,
        config: BotConfig,
        workflow,
        memory
    ):
        super().__init__(config, workflow, memory)
        self.logger = logging.getLogger(f"{__name__}.PazBot")
        
        # Paz固有の設定
        self.is_primary_receiver = False  # Spectraのみが受信
    
    async def on_bot_ready(self) -> None:
        """Paz Bot準備完了"""
        self.logger.info("🎨 Paz Bot ready - Creative inspiration specialist on standby")
    
    async def on_message_received(self, message: discord.Message) -> None:
        """メッセージ受信処理（Pazは直接受信しない）"""
        # Pazは単独受信せず、LangGraphワークフローで制御される
        # デバッグ用ログ
        if not message.author.bot:
            self.logger.debug(f"Paz received message (ignored): {message.content[:30]}...")