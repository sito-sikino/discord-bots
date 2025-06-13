"""
ベースボットクラス

全ボットに共通する基本機能:
- Discord.pyクライアント管理
- シグナリング統合
- 会話履歴連携
- 基本イベントハンドリング
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import discord
from discord.ext import commands

from ..core.signal_manager import BotSignalManager, BotSignal, SignalType
from ..core.conversation_memory import ConversationMemory, ConversationMessage
from ..core.multi_bot_system import BotConfig


class BaseBot(ABC):
    """ベースボットクラス"""
    
    def __init__(
        self,
        config: BotConfig,
        signal_manager: BotSignalManager,
        memory: ConversationMemory
    ):
        self.config = config
        self.signal_manager = signal_manager
        self.memory = memory
        self.logger = logging.getLogger(f"{__name__}.{self.config.name}")
        
        # Discord.pyクライアント初期化
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        
        self.client = commands.Bot(
            command_prefix=self.config.command_prefix,
            intents=intents,
            description=self.config.description
        )
        
        # イベントハンドラー登録
        self._register_events()
        self._register_signal_handlers()
    
    def _register_events(self) -> None:
        """Discordイベントハンドラー登録"""
        
        @self.client.event
        async def on_ready():
            self.logger.info(
                f"{self.config.name} bot logged in as {self.client.user}"
            )
            await self.on_bot_ready()
        
        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return  # 自分のメッセージは無視
            
            await self.on_message_received(message)
        
        @self.client.event
        async def on_error(event, *args, **kwargs):
            self.logger.error(f"Discord event error: {event}", exc_info=True)
            await self.on_discord_error(event, args, kwargs)
    
    def _register_signal_handlers(self) -> None:
        """シグナルハンドラー登録"""
        # 各ボットで実装すべきシグナルハンドラーを登録
        self.signal_manager.register_signal_handler(
            self.config.name.lower(),
            SignalType.DELEGATE_TO_LYNQ,
            self.handle_delegation_signal
        )
        self.signal_manager.register_signal_handler(
            self.config.name.lower(), 
            SignalType.DELEGATE_TO_PAZ,
            self.handle_delegation_signal
        )
        self.signal_manager.register_signal_handler(
            self.config.name.lower(),
            SignalType.COLLABORATION_REQUEST,
            self.handle_collaboration_signal
        )
        self.signal_manager.register_signal_handler(
            self.config.name.lower(),
            SignalType.RESPONSE_READY,
            self.handle_response_signal
        )
    
    async def save_message_to_memory(self, message: discord.Message) -> None:
        """メッセージを履歴に保存"""
        try:
            conv_message = ConversationMessage(
                message_id=str(message.id),
                user_id=message.author.id,
                username=message.author.display_name,
                content=message.content,
                timestamp=message.created_at,
                bot_name=None if not message.author.bot else self.config.name,
                message_type="bot" if message.author.bot else "user"
            )
            
            await self.memory.add_message(message.channel.id, conv_message)
            
        except Exception as e:
            self.logger.error(f"Error saving message to memory: {e}")
    
    async def send_response(
        self, 
        channel: discord.TextChannel, 
        content: str,
        reference: Optional[discord.Message] = None
    ) -> Optional[discord.Message]:
        """応答メッセージ送信"""
        try:
            # 長いメッセージの分割対応
            if len(content) > 2000:
                # 2000文字制限対応
                chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
                sent_message = None
                
                for i, chunk in enumerate(chunks):
                    if i == 0 and reference:
                        sent_message = await channel.send(chunk, reference=reference)
                    else:
                        sent_message = await channel.send(chunk)
                
                return sent_message
            else:
                return await channel.send(content, reference=reference)
                
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            return None
    
    async def get_conversation_context(
        self, 
        channel_id: int, 
        context_length: int = 10
    ) -> str:
        """会話文脈を取得"""
        return await self.memory.get_recent_context(channel_id, context_length)
    
    # 抽象メソッド - 各ボットで実装
    @abstractmethod
    async def on_bot_ready(self) -> None:
        """ボット準備完了時の処理"""
        pass
    
    @abstractmethod
    async def on_message_received(self, message: discord.Message) -> None:
        """メッセージ受信時の処理"""
        pass
    
    @abstractmethod
    async def handle_delegation_signal(self, signal: BotSignal) -> None:
        """委譲シグナル処理"""
        pass
    
    @abstractmethod
    async def handle_collaboration_signal(self, signal: BotSignal) -> None:
        """協調シグナル処理"""
        pass
    
    @abstractmethod
    async def handle_response_signal(self, signal: BotSignal) -> None:
        """応答シグナル処理"""
        pass
    
    async def on_discord_error(
        self, 
        event: str, 
        args: tuple, 
        kwargs: dict
    ) -> None:
        """Discordエラー処理（デフォルト実装）"""
        self.logger.error(f"Discord error in event {event}: {args}, {kwargs}")
    
    @property
    def bot_name(self) -> str:
        """ボット名取得"""
        return self.config.name.lower()
    
    def is_connected(self) -> bool:
        """接続状態確認"""
        return not self.client.is_closed()