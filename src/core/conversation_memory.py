"""
ファイルベース会話履歴管理

軽量な会話履歴管理システム:
- データベース不要のJSONファイルベース
- チャンネルごとに直近20メッセージ保持
- 適切なロック機構で競合解決
- 3ボット間での履歴共有
"""

import asyncio
import json
import logging
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ConversationMessage:
    """会話メッセージデータ"""
    message_id: str
    user_id: int
    username: str
    content: str
    timestamp: datetime
    bot_name: Optional[str] = None  # ボットからのメッセージの場合
    message_type: str = "user"  # user, bot, system
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "username": self.username,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "bot_name": self.bot_name,
            "message_type": self.message_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        """辞書から復元"""
        return cls(
            message_id=data["message_id"],
            user_id=data["user_id"],
            username=data["username"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            bot_name=data.get("bot_name"),
            message_type=data.get("message_type", "user")
        )


@dataclass
class ChannelHistory:
    """チャンネル履歴データ"""
    channel_id: int
    messages: List[ConversationMessage]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "channel_id": self.channel_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelHistory":
        """辞書から復元"""
        return cls(
            channel_id=data["channel_id"],
            messages=[
                ConversationMessage.from_dict(msg_data) 
                for msg_data in data["messages"]
            ],
            last_updated=datetime.fromisoformat(data["last_updated"])
        )


class ConversationMemory:
    """ファイルベース会話履歴管理"""
    
    def __init__(
        self, 
        data_dir: str = "data/channels",
        max_messages_per_channel: int = 20
    ):
        self.data_dir = Path(data_dir)
        self.max_messages_per_channel = max_messages_per_channel
        self.logger = logging.getLogger(__name__)
        
        # ファイルロック管理
        self._file_locks: Dict[int, asyncio.Lock] = {}
        
        # データディレクトリ作成
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_lock(self, channel_id: int) -> asyncio.Lock:
        """チャンネル別ファイルロック取得"""
        if channel_id not in self._file_locks:
            self._file_locks[channel_id] = asyncio.Lock()
        return self._file_locks[channel_id]
    
    def _get_channel_file_path(self, channel_id: int) -> Path:
        """チャンネルファイルパス取得"""
        return self.data_dir / f"{channel_id}.json"
    
    async def add_message(
        self, 
        channel_id: int, 
        message: ConversationMessage
    ) -> None:
        """メッセージを履歴に追加"""
        async with self._get_file_lock(channel_id):
            try:
                # 既存履歴読み込み
                history = await self._load_channel_history(channel_id)
                
                # メッセージ追加
                history.messages.append(message)
                
                # 最大メッセージ数制限
                if len(history.messages) > self.max_messages_per_channel:
                    history.messages = history.messages[-self.max_messages_per_channel:]
                
                # 更新時刻設定
                history.last_updated = datetime.now()
                
                # ファイル保存
                await self._save_channel_history(history)
                
                self.logger.debug(
                    f"Message added to channel {channel_id}: {message.message_id}"
                )
                
            except Exception as e:
                self.logger.error(f"Error adding message to channel {channel_id}: {e}")
                raise
    
    async def get_channel_history(
        self, 
        channel_id: int, 
        limit: Optional[int] = None
    ) -> List[ConversationMessage]:
        """チャンネル履歴取得"""
        async with self._get_file_lock(channel_id):
            try:
                history = await self._load_channel_history(channel_id)
                messages = history.messages
                
                if limit:
                    messages = messages[-limit:]
                
                return messages
                
            except Exception as e:
                self.logger.error(f"Error getting channel history {channel_id}: {e}")
                return []
    
    async def get_recent_context(
        self, 
        channel_id: int, 
        context_length: int = 10
    ) -> str:
        """最近の会話文脈を文字列として取得"""
        try:
            messages = await self.get_channel_history(channel_id, limit=context_length)
            
            context_lines = []
            for msg in messages:
                if msg.message_type == "user":
                    context_lines.append(f"{msg.username}: {msg.content}")
                elif msg.message_type == "bot" and msg.bot_name:
                    context_lines.append(f"{msg.bot_name}: {msg.content}")
            
            return "\\n".join(context_lines)
            
        except Exception as e:
            self.logger.error(f"Error getting recent context for channel {channel_id}: {e}")
            return ""
    
    async def _load_channel_history(self, channel_id: int) -> ChannelHistory:
        """チャンネル履歴をファイルから読み込み"""
        file_path = self._get_channel_file_path(channel_id)
        
        if not file_path.exists():
            # 新規チャンネル
            return ChannelHistory(
                channel_id=channel_id,
                messages=[],
                last_updated=datetime.now()
            )
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                data = json.loads(await f.read())
                return ChannelHistory.from_dict(data)
                
        except Exception as e:
            self.logger.error(f"Error loading channel history {channel_id}: {e}")
            # 破損時は新規作成
            return ChannelHistory(
                channel_id=channel_id,
                messages=[],
                last_updated=datetime.now()
            )
    
    async def _save_channel_history(self, history: ChannelHistory) -> None:
        """チャンネル履歴をファイルに保存"""
        file_path = self._get_channel_file_path(history.channel_id)
        
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(history.to_dict(), ensure_ascii=False, indent=2))
                
        except Exception as e:
            self.logger.error(f"Error saving channel history {history.channel_id}: {e}")
            raise
    
    async def clear_channel_history(self, channel_id: int) -> None:
        """チャンネル履歴をクリア"""
        async with self._get_file_lock(channel_id):
            try:
                file_path = self._get_channel_file_path(channel_id)
                if file_path.exists():
                    file_path.unlink()
                
                self.logger.info(f"Channel history cleared: {channel_id}")
                
            except Exception as e:
                self.logger.error(f"Error clearing channel history {channel_id}: {e}")
                raise
    
    async def get_channel_stats(self) -> Dict[str, Any]:
        """チャンネル統計情報取得"""
        try:
            stats = {
                "total_channels": 0,
                "total_messages": 0,
                "channels": {}
            }
            
            for file_path in self.data_dir.glob("*.json"):
                try:
                    channel_id = int(file_path.stem)
                    history = await self._load_channel_history(channel_id)
                    
                    stats["total_channels"] += 1
                    stats["total_messages"] += len(history.messages)
                    stats["channels"][str(channel_id)] = {
                        "message_count": len(history.messages),
                        "last_updated": history.last_updated.isoformat()
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error processing channel file {file_path}: {e}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting channel stats: {e}")
            return {"error": str(e)}