"""
Spectra Bot - 基底エージェント兼一般対話ボット

役割:
- 単独メッセージ受信（レート制限回避）
- LLMによる深層分析・意思決定
- 一般対話ボットとしての直接応答
- サブエージェントへの効率的委譲
- マルチエージェント協調の調整
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

import discord

from ..base_bot import BaseBot
from ...core.signal_manager import BotSignal, SignalType
from ...core.multi_bot_system import BotConfig


class SpectraBot(BaseBot):
    """Spectra Bot - 基底エージェント兼一般対話ボット"""
    
    def __init__(
        self,
        config: BotConfig,
        signal_manager,
        memory
    ):
        super().__init__(config, signal_manager, memory)
        self.logger = logging.getLogger(f"{__name__}.SpectraBot")
        
        # Spectra固有の設定
        self.is_primary_receiver = True  # 単独受信フラグ
        self.llm_access_count = 0  # レート制限管理
        self.max_llm_calls = 2  # 最大2回のシーケンシャルアクセス
    
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
            
            # LLM分析による意思決定
            await self._analyze_and_respond(message)
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            await self._send_error_response(message.channel)
    
    async def _analyze_and_respond(self, message: discord.Message) -> None:
        """LLM分析による応答戦略決定"""
        try:
            # 会話文脈取得
            context = await self.get_conversation_context(message.channel.id)
            
            # Phase 1.2では簡単な応答を実装（LLM統合は後フェーズ）
            decision = await self._simple_decision_logic(message.content, context)
            
            if decision["action"] == "respond_directly":
                await self._respond_directly(message, decision["response"])
            elif decision["action"] == "delegate_to_lynq":
                await self._delegate_to_lynq(message, decision["context"])
            elif decision["action"] == "delegate_to_paz":
                await self._delegate_to_paz(message, decision["context"])
            elif decision["action"] == "collaborate":
                await self._initiate_collaboration(message, decision["context"])
            else:
                # デフォルト：Spectraが直接応答
                await self._respond_directly(message, {
                    "content": "こんにちは！Spectraです。どのようなことでお手伝いできますか？"
                })
                
        except Exception as e:
            self.logger.error(f"Error in analysis and response: {e}")
            await self._send_error_response(message.channel)
    
    async def _simple_decision_logic(
        self, 
        content: str, 
        context: str
    ) -> Dict[str, Any]:
        """Phase 1.2用の簡単な意思決定ロジック（LLM統合前）"""
        content_lower = content.lower()
        
        # 論理分析キーワード検出
        logical_keywords = [
            "分析", "問題", "論理", "矛盾", "解決", "データ", "検証", "比較"
        ]
        if any(keyword in content_lower for keyword in logical_keywords):
            return {
                "action": "delegate_to_lynq",
                "context": {"analysis_type": "logical", "priority": "high"}
            }
        
        # 創作キーワード検出  
        creative_keywords = [
            "アイデア", "創作", "ブレインストーミング", "発想", "革新", "インスピレーション"
        ]
        if any(keyword in content_lower for keyword in creative_keywords):
            return {
                "action": "delegate_to_paz",
                "context": {"creativity_type": "brainstorming", "priority": "high"}
            }
        
        # 複合要求検出
        if len(content) > 100 and ("+" in content or "と" in content):
            return {
                "action": "collaborate",
                "context": {"collaboration_type": "multi_aspect", "priority": "medium"}
            }
        
        # デフォルト：Spectra直接応答
        return {
            "action": "respond_directly",
            "response": {
                "content": f"こんにちは！あなたのメッセージ「{content[:50]}...」を受け取りました。\\n\\n"
                          f"Spectraとして、一般的な対話や情報提供をお手伝いできます。\\n"
                          f"論理的な分析が必要でしたらLynQが、創作的なアイデアが必要でしたらPazが協力します！"
            }
        }
    
    async def _respond_directly(
        self, 
        message: discord.Message, 
        response_data: Dict[str, Any]
    ) -> None:
        """Spectra直接応答（Pattern 1）"""
        try:
            response_content = response_data["content"]
            
            # 応答送信
            sent_message = await self.send_response(
                message.channel, 
                response_content, 
                reference=message
            )
            
            if sent_message:
                # 応答を履歴に保存
                await self.save_message_to_memory(sent_message)
                self.logger.info("Spectra direct response sent")
            
        except Exception as e:
            self.logger.error(f"Error in direct response: {e}")
            await self._send_error_response(message.channel)
    
    async def _delegate_to_lynq(
        self, 
        message: discord.Message, 
        context: Dict[str, Any]
    ) -> None:
        """LynQへの委譲（Pattern 2）"""
        try:
            signal = BotSignal(
                signal_type=SignalType.DELEGATE_TO_LYNQ,
                from_bot="spectra",
                to_bot="lynq", 
                channel_id=message.channel.id,
                user_id=message.author.id,
                message_content=message.content,
                context=context,
                timestamp=datetime.now(),
                signal_id=str(uuid.uuid4())
            )
            
            success = await self.signal_manager.send_signal(signal)
            if success:
                self.logger.info("Delegation signal sent to LynQ")
            else:
                # 委譲失敗時はSpectraが代替応答
                await self._respond_directly(message, {
                    "content": "申し訳ありませんが、論理分析の専門家LynQが現在利用できません。\\n"
                              "Spectraとして基本的な分析をお手伝いします。"
                })
                
        except Exception as e:
            self.logger.error(f"Error delegating to LynQ: {e}")
            await self._send_error_response(message.channel)
    
    async def _delegate_to_paz(
        self, 
        message: discord.Message, 
        context: Dict[str, Any]
    ) -> None:
        """Pazへの委譲（Pattern 2）"""
        try:
            signal = BotSignal(
                signal_type=SignalType.DELEGATE_TO_PAZ,
                from_bot="spectra",
                to_bot="paz",
                channel_id=message.channel.id,
                user_id=message.author.id,
                message_content=message.content,
                context=context,
                timestamp=datetime.now(),
                signal_id=str(uuid.uuid4())
            )
            
            success = await self.signal_manager.send_signal(signal)
            if success:
                self.logger.info("Delegation signal sent to Paz")
            else:
                # 委譲失敗時はSpectraが代替応答
                await self._respond_directly(message, {
                    "content": "申し訳ありませんが、創作の専門家Pazが現在利用できません。\\n"
                              "Spectraとして基本的なアイデア提供をお手伝いします。"
                })
                
        except Exception as e:
            self.logger.error(f"Error delegating to Paz: {e}")
            await self._send_error_response(message.channel)
    
    async def _initiate_collaboration(
        self, 
        message: discord.Message, 
        context: Dict[str, Any]
    ) -> None:
        """マルチエージェント協調開始（Pattern 3）"""
        try:
            # 協調開始の通知
            coordination_message = (
                "🤝 複合的なご要求ですね！チーム一丸となって対応します。\\n\\n"
                "📊 LynQ（論理分析）→ 🎨 Paz（創作アイデア）→ 🌟 Spectra（統合）の順で進めます。"
            )
            
            await self.send_response(message.channel, coordination_message, reference=message)
            
            # 順次協調シグナル送信（Phase 1.2では簡略化）
            self.logger.info("Collaboration initiated - Multi-agent workflow started")
            
        except Exception as e:
            self.logger.error(f"Error initiating collaboration: {e}")
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
    
    # シグナルハンドラー実装
    async def handle_delegation_signal(self, signal: BotSignal) -> None:
        """委譲シグナル処理（Spectraは通常受信しない）"""
        self.logger.info(f"Received delegation signal: {signal.signal_type}")
    
    async def handle_collaboration_signal(self, signal: BotSignal) -> None:
        """協調シグナル処理"""
        self.logger.info(f"Received collaboration signal: {signal.signal_type}")
        # Phase 2以降で実装
    
    async def handle_response_signal(self, signal: BotSignal) -> None:
        """応答シグナル処理"""
        self.logger.info(f"Received response signal: {signal.signal_type}")
        # Phase 2以降で実装