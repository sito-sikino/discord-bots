"""
Spectra Bot - 基底エージェント兼一般対話ボット

役割:
- 単独メッセージ受信（レート制限回避）
- LLMによる深層分析・意思決定
- 一般対話ボットとしての直接応答
- サブエージェントへの効率的委譲
- マルチエージェント協調の調整
"""

import logging
from typing import Optional, Dict, Any

import discord

from ..base_bot import BaseBot
from ...core.bot_config import BotConfig


class SpectraBot(BaseBot):
    """Spectra Bot - 基底エージェント兼一般対話ボット"""
    
    def __init__(
        self,
        config: BotConfig,
        workflow,
        memory
    ):
        super().__init__(config, workflow, memory)
        self.logger = logging.getLogger(f"{__name__}.SpectraBot")
        
        # Spectra固有の設定
        self.is_primary_receiver = True  # 単独受信フラグ
    
    async def on_bot_ready(self) -> None:
        """Spectra Bot準備完了"""
        self.logger.info("🌟 Spectra Bot ready - Primary message receiver active")
        
        # Guild取得（今後の拡張用）
        try:
            if hasattr(self.client, 'guilds') and self.client.guilds:
                guild = self.client.guilds[0]
                self.logger.info(f"Connected to guild: {guild.name}")
        except Exception as e:
            self.logger.warning(f"Guild info retrieval failed: {e}")
    
    async def on_message_received(self, message: discord.Message) -> None:
        """メッセージ受信処理（Spectraのみが受信）"""
        try:
            # ボットメッセージは無視
            if message.author.bot:
                return
            
            # 履歴保存
            await self.save_message_to_memory(message)
            
            self.logger.info(
                f"Message received from {message.author.display_name}: "
                f"{message.content[:50]}..."
            )
            
            # LangGraphワークフローで処理
            await self._process_with_workflow(message)
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            await self._send_error_response(message.channel)
    
    async def _process_with_workflow(self, message: discord.Message) -> None:
        """LangGraphワークフローでメッセージを処理"""
        try:
            # AgentStateを構築
            initial_state = await self.build_agent_state(message)
            
            # ワークフロー実行
            result = await self.workflow.process_message(initial_state)
            
            # エラーチェック
            if result.get("error"):
                self.logger.error(f"Workflow error: {result['error']}")
                await self._send_error_response(message.channel)
                return
            
            # 結果から応答を取得
            response_messages = result.get("messages", [])
            
            # 最新のAIメッセージを探す
            ai_message = None
            for msg in reversed(response_messages):
                if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                    ai_message = msg
                    break
            
            if ai_message:
                # 応答送信
                sent_message = await self.send_response(
                    message.channel,
                    ai_message.content,
                    reference=message
                )
                
                if sent_message:
                    # 応答を履歴に保存
                    await self.save_message_to_memory(sent_message)
                    self.logger.info(f"LangGraph workflow response sent")
            else:
                self.logger.warning("No assistant message found in workflow result")
                await self._send_error_response(message.channel)
                
        except Exception as e:
            self.logger.error(f"Error in workflow processing: {e}")
            await self._send_error_response(message.channel)
    
    async def _send_error_response(self, channel: discord.TextChannel) -> None:
        """エラー時の応答"""
        try:
            error_message = (
                "⚠️ 申し訳ありませんが、処理中にエラーが発生しました。\\n"
                "しばらく待ってから再度お試しください。"
            )
            await channel.send(error_message)
            
        except Exception as e:
            self.logger.error(f"Error sending error response: {e}")
    
