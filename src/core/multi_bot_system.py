"""
統合プロセス型マルチボットシステム

3つの独立したDiscord Botクライアントを単一プロセスで効率管理:
- Spectra Bot: 基底エージェント兼一般対話ボット（単独受信）
- LynQ Bot: 論理分析専門エージェント（シグナル待機）
- Paz Bot: 創作アイデア専門エージェント（シグナル待機）
"""

import asyncio
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

import discord
from discord.ext import commands

from .signal_manager import BotSignalManager, BotSignal, SignalType
from .conversation_memory import ConversationMemory
from ..bots.spectra.spectra_bot import SpectraBot
from ..bots.lynq.lynq_bot import LynQBot
from ..bots.paz.paz_bot import PazBot


class BotType(Enum):
    SPECTRA = "spectra"
    LYNQ = "lynq"
    PAZ = "paz"


@dataclass
class BotConfig:
    """ボット設定データクラス"""
    token: str
    name: str
    description: str
    command_prefix: str = "!"


class MultiDiscordBotSystem:
    """統合プロセス型マルチボットシステム"""
    
    def __init__(
        self,
        spectra_config: BotConfig,
        lynq_config: BotConfig,
        paz_config: BotConfig,
        guild_id: int
    ):
        self.guild_id = guild_id
        self.signal_manager = BotSignalManager()
        self.memory = ConversationMemory()
        
        # ボットインスタンス初期化
        self.spectra_bot = SpectraBot(
            config=spectra_config,
            signal_manager=self.signal_manager,
            memory=self.memory
        )
        self.lynq_bot = LynQBot(
            config=lynq_config,
            signal_manager=self.signal_manager,
            memory=self.memory
        )
        self.paz_bot = PazBot(
            config=paz_config,
            signal_manager=self.signal_manager,
            memory=self.memory
        )
        
        # ボット管理辞書
        self.bots: Dict[BotType, commands.Bot] = {
            BotType.SPECTRA: self.spectra_bot.client,
            BotType.LYNQ: self.lynq_bot.client,
            BotType.PAZ: self.paz_bot.client
        }
        
        self.logger = logging.getLogger(__name__)
        self._running = False
    
    async def start_all_bots(self) -> None:
        """全ボットを統合起動"""
        try:
            self.logger.info("統合マルチボットシステムを起動中...")
            self._running = True
            
            # 全ボットを並行起動
            tasks = []
            for bot_type, bot in self.bots.items():
                config = self._get_bot_config(bot_type)
                task = asyncio.create_task(
                    bot.start(config.token),
                    name=f"bot_{bot_type.value}"
                )
                tasks.append(task)
            
            # シグナル管理タスクも並行実行
            signal_task = asyncio.create_task(
                self.signal_manager.start_signal_processing(),
                name="signal_manager"
            )
            tasks.append(signal_task)
            
            # 全タスク完了まで待機
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"ボットシステム起動エラー: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self) -> None:
        """全ボットを安全にシャットダウン"""
        self.logger.info("マルチボットシステムをシャットダウン中...")
        self._running = False
        
        # 全ボットを並行シャットダウン
        shutdown_tasks = []
        for bot_type, bot in self.bots.items():
            if not bot.is_closed():
                task = asyncio.create_task(
                    bot.close(),
                    name=f"shutdown_{bot_type.value}"
                )
                shutdown_tasks.append(task)
        
        # シグナル管理停止
        await self.signal_manager.stop_signal_processing()
        
        # 全シャットダウン完了まで待機
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.logger.info("マルチボットシステム シャットダウン完了")
    
    def _get_bot_config(self, bot_type: BotType) -> BotConfig:
        """ボットタイプに対応する設定を取得"""
        if bot_type == BotType.SPECTRA:
            return self.spectra_bot.config
        elif bot_type == BotType.LYNQ:
            return self.lynq_bot.config
        elif bot_type == BotType.PAZ:
            return self.paz_bot.config
        else:
            raise ValueError(f"Unknown bot type: {bot_type}")
    
    @property
    def is_running(self) -> bool:
        """システム稼働状態を取得"""
        return self._running
    
    def get_bot_status(self) -> Dict[str, bool]:
        """各ボットの接続状態を取得"""
        status = {}
        for bot_type, bot in self.bots.items():
            status[bot_type.value] = not bot.is_closed()
        return status