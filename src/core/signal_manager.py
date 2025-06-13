"""
AsyncIOキューによる軽量シグナリングシステム

ボット間の効率的な内部通信を提供:
- 中間Discord投稿なしでサブエージェントに直接委譲
- Redis不要の軽量AsyncIOキュー実装
- レート制限対応のシーケンシャル処理
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class SignalType(Enum):
    """シグナル種別"""
    DELEGATE_TO_LYNQ = "delegate_to_lynq"
    DELEGATE_TO_PAZ = "delegate_to_paz"
    COLLABORATION_REQUEST = "collaboration_request"
    RESPONSE_READY = "response_ready"
    SYSTEM_STATUS = "system_status"


@dataclass
class BotSignal:
    """ボット間シグナルデータ"""
    signal_type: SignalType
    from_bot: str
    to_bot: str
    channel_id: int
    user_id: int
    message_content: str
    context: Dict[str, Any]
    timestamp: datetime
    signal_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "signal_type": self.signal_type.value,
            "from_bot": self.from_bot,
            "to_bot": self.to_bot,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "message_content": self.message_content,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "signal_id": self.signal_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotSignal":
        """辞書から復元"""
        return cls(
            signal_type=SignalType(data["signal_type"]),
            from_bot=data["from_bot"],
            to_bot=data["to_bot"],
            channel_id=data["channel_id"],
            user_id=data["user_id"],
            message_content=data["message_content"],
            context=data["context"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            signal_id=data["signal_id"]
        )


class BotSignalManager:
    """ボット間シグナリング管理"""
    
    def __init__(self, max_queue_size: int = 100):
        # ボット別シグナルキュー
        self.signal_queues: Dict[str, asyncio.Queue] = {
            "lynq": asyncio.Queue(maxsize=max_queue_size),
            "paz": asyncio.Queue(maxsize=max_queue_size),
            "spectra": asyncio.Queue(maxsize=max_queue_size)
        }
        
        # シグナル処理状態管理
        self.processing_tasks: Dict[str, Optional[asyncio.Task]] = {
            "lynq": None,
            "paz": None,
            "spectra": None
        }
        
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._signal_handlers: Dict[str, Dict[SignalType, callable]] = {
            "lynq": {},
            "paz": {},
            "spectra": {}
        }
    
    async def send_signal(self, signal: BotSignal) -> bool:
        """シグナルを送信"""
        try:
            target_bot = signal.to_bot
            if target_bot not in self.signal_queues:
                self.logger.error(f"Unknown target bot: {target_bot}")
                return False
            
            # キューに追加（非ブロッキング）
            try:
                self.signal_queues[target_bot].put_nowait(signal)
                self.logger.info(
                    f"Signal sent: {signal.signal_type.value} "
                    f"from {signal.from_bot} to {signal.to_bot}"
                )
                return True
            except asyncio.QueueFull:
                self.logger.warning(f"Signal queue full for bot: {target_bot}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending signal: {e}")
            return False
    
    async def wait_for_signal(
        self, 
        bot_name: str, 
        timeout: Optional[float] = None
    ) -> Optional[BotSignal]:
        """シグナル待機"""
        try:
            if bot_name not in self.signal_queues:
                self.logger.error(f"Unknown bot: {bot_name}")
                return None
            
            queue = self.signal_queues[bot_name]
            
            if timeout:
                try:
                    signal = await asyncio.wait_for(queue.get(), timeout=timeout)
                    self.logger.info(
                        f"Signal received by {bot_name}: {signal.signal_type.value}"
                    )
                    return signal
                except asyncio.TimeoutError:
                    self.logger.debug(f"Signal wait timeout for {bot_name}")
                    return None
            else:
                signal = await queue.get()
                self.logger.info(
                    f"Signal received by {bot_name}: {signal.signal_type.value}"
                )
                return signal
                
        except Exception as e:
            self.logger.error(f"Error waiting for signal: {e}")
            return None
    
    def register_signal_handler(
        self, 
        bot_name: str, 
        signal_type: SignalType, 
        handler: callable
    ) -> None:
        """シグナルハンドラーを登録"""
        if bot_name not in self._signal_handlers:
            self._signal_handlers[bot_name] = {}
        
        self._signal_handlers[bot_name][signal_type] = handler
        self.logger.info(
            f"Signal handler registered: {bot_name} -> {signal_type.value}"
        )
    
    async def start_signal_processing(self) -> None:
        """シグナル処理を開始"""
        self._running = True
        self.logger.info("シグナル処理を開始")
        
        # 各ボット用の処理タスクを起動
        for bot_name in self.signal_queues.keys():
            task = asyncio.create_task(
                self._process_signals_for_bot(bot_name),
                name=f"signal_processor_{bot_name}"
            )
            self.processing_tasks[bot_name] = task
        
        # 全処理タスク完了まで待機
        tasks = [task for task in self.processing_tasks.values() if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_signal_processing(self) -> None:
        """シグナル処理を停止"""
        self.logger.info("シグナル処理を停止中...")
        self._running = False
        
        # 全処理タスクをキャンセル
        cancel_tasks = []
        for bot_name, task in self.processing_tasks.items():
            if task and not task.done():
                task.cancel()
                cancel_tasks.append(task)
        
        # キャンセル完了まで待機
        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)
        
        self.logger.info("シグナル処理停止完了")
    
    async def _process_signals_for_bot(self, bot_name: str) -> None:
        """ボット固有のシグナル処理ループ"""
        try:
            while self._running:
                try:
                    # 短時間タイムアウトでポーリング
                    signal = await self.wait_for_signal(bot_name, timeout=1.0)
                    
                    if signal:
                        await self._handle_signal(bot_name, signal)
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Signal processing error for {bot_name}: {e}")
                    await asyncio.sleep(1)  # エラー時の短時間待機
                    
        except asyncio.CancelledError:
            self.logger.info(f"Signal processing cancelled for {bot_name}")
        except Exception as e:
            self.logger.error(f"Fatal error in signal processing for {bot_name}: {e}")
    
    async def _handle_signal(self, bot_name: str, signal: BotSignal) -> None:
        """シグナルハンドリング"""
        try:
            handlers = self._signal_handlers.get(bot_name, {})
            handler = handlers.get(signal.signal_type)
            
            if handler:
                await handler(signal)
            else:
                self.logger.warning(
                    f"No handler for signal {signal.signal_type.value} "
                    f"in bot {bot_name}"
                )
                
        except Exception as e:
            self.logger.error(f"Error handling signal: {e}")
    
    def get_queue_status(self) -> Dict[str, Dict[str, Any]]:
        """キュー状態取得"""
        status = {}
        for bot_name, queue in self.signal_queues.items():
            status[bot_name] = {
                "queue_size": queue.qsize(),
                "queue_maxsize": queue.maxsize,
                "processing_active": (
                    self.processing_tasks[bot_name] is not None 
                    and not self.processing_tasks[bot_name].done()
                )
            }
        return status