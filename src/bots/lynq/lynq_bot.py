"""
LynQ Bot - 論理分析専門エージェント

役割:
- シグナル待機による専門分析
- 問題分解・矛盾指摘・データ分析
- 論理的思考・検証・比較分析
- 効率的直接応答（中間投稿なし）
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

import discord

from ..base_bot import BaseBot
from ...core.signal_manager import BotSignal, SignalType
from ...core.multi_bot_system import BotConfig


class LynQBot(BaseBot):
    """LynQ Bot - 論理分析専門エージェント"""
    
    def __init__(
        self,
        config: BotConfig,
        signal_manager,
        memory
    ):
        super().__init__(config, signal_manager, memory)
        self.logger = logging.getLogger(f"{__name__}.LynQBot")
        
        # LynQ固有の設定
        self.is_primary_receiver = False  # シグナル待機ボット
        self.analysis_types = [
            "logical_analysis",
            "problem_decomposition", 
            "contradiction_detection",
            "data_verification",
            "comparative_analysis"
        ]
    
    async def on_bot_ready(self) -> None:
        """LynQ Bot準備完了"""
        self.logger.info("🔍 LynQ Bot ready - Logical analysis specialist on standby")
    
    async def on_message_received(self, message: discord.Message) -> None:
        """メッセージ受信処理（LynQは直接受信しない）"""
        # LynQは単独受信せず、シグナル待機のみ
        # デバッグ用ログ
        if not message.author.bot:
            self.logger.debug(f"LynQ received message (ignored): {message.content[:30]}...")
    
    async def handle_delegation_signal(self, signal: BotSignal) -> None:
        """委譲シグナル処理 - LynQの専門分析"""
        try:
            if signal.signal_type != SignalType.DELEGATE_TO_LYNQ:
                return
            
            self.logger.info(f"LynQ delegation received: {signal.signal_id}")
            
            # チャンネル取得
            channel = self.client.get_channel(signal.channel_id)
            if not channel:
                self.logger.error(f"Channel not found: {signal.channel_id}")
                return
            
            # 会話文脈取得
            context = await self.get_conversation_context(signal.channel_id)
            
            # 論理分析実行
            analysis_result = await self._perform_logical_analysis(
                signal.message_content,
                context,
                signal.context
            )
            
            # 直接応答（Pattern 2: 中間投稿なし）
            await self._send_analysis_response(channel, analysis_result, signal)
            
        except Exception as e:
            self.logger.error(f"Error handling delegation signal: {e}")
            await self._send_error_response(signal.channel_id)
    
    async def _perform_logical_analysis(
        self,
        message_content: str,
        conversation_context: str,
        signal_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """論理分析実行（Phase 1.2では基本実装）"""
        try:
            analysis_type = signal_context.get("analysis_type", "general")
            
            # Phase 1.2での基本的な論理分析
            analysis = {
                "original_query": message_content,
                "analysis_type": analysis_type,
                "logical_structure": await self._analyze_logical_structure(message_content),
                "key_concepts": await self._extract_key_concepts(message_content),
                "potential_issues": await self._identify_logical_issues(message_content),
                "recommendations": await self._generate_recommendations(message_content),
                "confidence_level": "medium"  # Phase 1.2では固定
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in logical analysis: {e}")
            return {
                "error": "Analysis failed",
                "fallback_response": "申し訳ありませんが、分析中にエラーが発生しました。"
            }
    
    async def _analyze_logical_structure(self, content: str) -> Dict[str, Any]:
        """論理構造分析"""
        structure = {
            "premise_count": content.count("。") + content.count("？") + content.count("！"),
            "question_type": "what" if "何" in content else "how" if "どう" in content else "why" if "なぜ" in content else "general",
            "complexity_level": "high" if len(content) > 100 else "medium" if len(content) > 50 else "low"
        }
        
        return structure
    
    async def _extract_key_concepts(self, content: str) -> list:
        """重要概念抽出"""
        # Phase 1.2では簡単なキーワード抽出
        important_words = []
        
        # 一般的な重要語パターン
        content_words = content.split()
        for word in content_words:
            if len(word) > 2 and word not in ["です", "ます", "から", "ので", "ため"]:
                important_words.append(word)
        
        return important_words[:5]  # 上位5つまで
    
    async def _identify_logical_issues(self, content: str) -> list:
        """論理的問題点の特定"""
        issues = []
        
        # 基本的な論理チェック
        if "すべて" in content and "ない" in content:
            issues.append("絶対的表現に注意が必要です")
        
        if content.count("？") > 2:
            issues.append("複数の質問が含まれており、焦点を絞ることをお勧めします")
        
        if len(content) > 200:
            issues.append("長い文章のため、要点を整理することをお勧めします")
        
        return issues
    
    async def _generate_recommendations(self, content: str) -> list:
        """推奨事項生成"""
        recommendations = []
        
        # コンテンツベースの推奨
        if "問題" in content:
            recommendations.append("問題を具体的な要素に分解して段階的に解決することをお勧めします")
        
        if "比較" in content or "違い" in content:
            recommendations.append("比較項目を明確にして表形式で整理することをお勧めします")
        
        if "データ" in content:
            recommendations.append("データの出典と信頼性を確認することをお勧めします")
        
        # デフォルト推奨
        if not recommendations:
            recommendations.append("段階的なアプローチで問題に取り組むことをお勧めします")
        
        return recommendations
    
    async def _send_analysis_response(
        self,
        channel: discord.TextChannel,
        analysis: Dict[str, Any],
        signal: BotSignal
    ) -> None:
        """分析結果応答送信"""
        try:
            if "error" in analysis:
                await channel.send(analysis["fallback_response"])
                return
            
            # 分析結果をフォーマット
            response_content = self._format_analysis_response(analysis)
            
            # 応答送信
            sent_message = await self.send_response(channel, response_content)
            
            if sent_message:
                # 応答を履歴に保存
                await self.save_message_to_memory(sent_message)
                self.logger.info(f"LynQ analysis response sent for signal {signal.signal_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending analysis response: {e}")
            await self._send_error_response(signal.channel_id)
    
    def _format_analysis_response(self, analysis: Dict[str, Any]) -> str:
        """分析結果のフォーマット"""
        response = f"🔍 **LynQ 論理分析結果**\\n\\n"
        
        # 論理構造
        structure = analysis["logical_structure"]
        response += f"**📊 論理構造:**\\n"
        response += f"- 複雑度: {structure['complexity_level']}\\n"
        response += f"- 質問タイプ: {structure['question_type']}\\n\\n"
        
        # 重要概念
        if analysis["key_concepts"]:
            response += f"**🔑 重要概念:**\\n"
            for concept in analysis["key_concepts"]:
                response += f"- {concept}\\n"
            response += "\\n"
        
        # 論理的問題点
        if analysis["potential_issues"]:
            response += f"**⚠️ 留意点:**\\n"
            for issue in analysis["potential_issues"]:
                response += f"- {issue}\\n"
            response += "\\n"
        
        # 推奨事項
        if analysis["recommendations"]:
            response += f"**💡 推奨アプローチ:**\\n"
            for rec in analysis["recommendations"]:
                response += f"- {rec}\\n"
            response += "\\n"
        
        response += f"*信頼度: {analysis['confidence_level']} | LynQ論理分析*"
        
        return response
    
    async def _send_error_response(self, channel_id: int) -> None:
        """エラー応答送信"""
        try:
            channel = self.client.get_channel(channel_id)
            if channel:
                error_message = (
                    "🔍 **LynQ エラー**\\n\\n"
                    "申し訳ありませんが、論理分析中にエラーが発生しました。\\n"
                    "しばらく待ってから再度お試しください。"
                )
                await channel.send(error_message)
        except Exception as e:
            self.logger.error(f"Error sending error response: {e}")
    
    # 他のシグナルハンドラー
    async def handle_collaboration_signal(self, signal: BotSignal) -> None:
        """協調シグナル処理"""
        self.logger.info(f"LynQ received collaboration signal: {signal.signal_type}")
        # Phase 2以降で実装
    
    async def handle_response_signal(self, signal: BotSignal) -> None:
        """応答シグナル処理"""
        self.logger.info(f"LynQ received response signal: {signal.signal_type}")
        # Phase 2以降で実装