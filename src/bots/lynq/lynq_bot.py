"""
LynQ Bot - 論理分析専門エージェント

役割:
- シグナル待機による専門分析
- 問題分解・矛盾指摘・データ分析
- 論理的思考・検証・比較分析
- 効率的直接応答（中間投稿なし）
"""

import logging
from typing import Optional, Dict, Any

import discord

from ..base_bot import BaseBot
from ...core.bot_config import BotConfig


class LynQBot(BaseBot):
    """LynQ Bot - 論理分析専門エージェント"""
    
    def __init__(
        self,
        config: BotConfig,
        workflow,
        memory
    ):
        super().__init__(config, workflow, memory)
        self.logger = logging.getLogger(f"{__name__}.LynQBot")
        
        # LynQ固有の設定
        self.is_primary_receiver = False  # Spectraのみが受信
    
    async def on_bot_ready(self) -> None:
        """LynQ Bot準備完了"""
        self.logger.info("🔍 LynQ Bot ready - Logical analysis specialist on standby")
    
    async def on_message_received(self, message: discord.Message) -> None:
        """メッセージ受信処理（LynQは直接受信しない）"""
        # LynQは単独受信せず、LangGraphワークフローで制御される
        # デバッグ用ログ
        if not message.author.bot:
            self.logger.debug(f"LynQ received message (ignored): {message.content[:30]}...")
    
