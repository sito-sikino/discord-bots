"""
Paz Bot - 創作アイデア専門エージェント

役割:
- シグナル待機による創作支援
- インスピレーション・ブレインストーミング
- 革新的解決・アイデア発想
- 効率的直接応答（中間投稿なし）
"""

import logging
import random
from datetime import datetime
from typing import Optional, Dict, Any, List

import discord

from ..base_bot import BaseBot
from ...core.signal_manager import BotSignal, SignalType
from ...core.multi_bot_system import BotConfig


class PazBot(BaseBot):
    """Paz Bot - 創作アイデア専門エージェント"""
    
    def __init__(
        self,
        config: BotConfig,
        signal_manager,
        memory
    ):
        super().__init__(config, signal_manager, memory)
        self.logger = logging.getLogger(f"{__name__}.PazBot")
        
        # Paz固有の設定
        self.is_primary_receiver = False  # シグナル待機ボット
        self.creativity_techniques = [
            "brainstorming",
            "lateral_thinking",
            "metaphorical_thinking", 
            "scenario_building",
            "inspiration_synthesis"
        ]
        
        # 創作支援のためのシード要素
        self.inspiration_seeds = {
            "colors": ["深紅", "翠緑", "瑠璃", "金橙", "紫苑", "銀白"],
            "textures": ["滑らか", "ざらざら", "ふわふわ", "ひんやり", "温かい", "軽やか"],
            "emotions": ["喜び", "驚き", "好奇心", "安らぎ", "わくわく", "感動"],
            "concepts": ["変化", "調和", "可能性", "発見", "創造", "つながり"]
        }
    
    async def on_bot_ready(self) -> None:
        """Paz Bot準備完了"""
        self.logger.info("🎨 Paz Bot ready - Creative inspiration specialist on standby")
    
    async def on_message_received(self, message: discord.Message) -> None:
        """メッセージ受信処理（Pazは直接受信しない）"""
        # Pazは単独受信せず、シグナル待機のみ
        # デバッグ用ログ
        if not message.author.bot:
            self.logger.debug(f"Paz received message (ignored): {message.content[:30]}...")
    
    async def handle_delegation_signal(self, signal: BotSignal) -> None:
        """委譲シグナル処理 - Pazの創作支援"""
        try:
            if signal.signal_type != SignalType.DELEGATE_TO_PAZ:
                return
            
            self.logger.info(f"Paz delegation received: {signal.signal_id}")
            
            # チャンネル取得
            channel = self.client.get_channel(signal.channel_id)
            if not channel:
                self.logger.error(f"Channel not found: {signal.channel_id}")
                return
            
            # 会話文脈取得
            context = await self.get_conversation_context(signal.channel_id)
            
            # 創作支援実行
            creative_result = await self._perform_creative_assistance(
                signal.message_content,
                context,
                signal.context
            )
            
            # 直接応答（Pattern 2: 中間投稿なし）
            await self._send_creative_response(channel, creative_result, signal)
            
        except Exception as e:
            self.logger.error(f"Error handling delegation signal: {e}")
            await self._send_error_response(signal.channel_id)
    
    async def _perform_creative_assistance(
        self,
        message_content: str,
        conversation_context: str,
        signal_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """創作支援実行（Phase 1.2では基本実装）"""
        try:
            creativity_type = signal_context.get("creativity_type", "general")
            
            # Phase 1.2での基本的な創作支援
            creative_result = {
                "original_query": message_content,
                "creativity_type": creativity_type,
                "inspiration_themes": await self._generate_inspiration_themes(message_content),
                "idea_variations": await self._generate_idea_variations(message_content),
                "creative_techniques": await self._suggest_creative_techniques(message_content),
                "next_steps": await self._suggest_next_steps(message_content),
                "inspiration_level": "high"  # Phase 1.2では固定
            }
            
            return creative_result
            
        except Exception as e:
            self.logger.error(f"Error in creative assistance: {e}")
            return {
                "error": "Creative assistance failed",
                "fallback_response": "申し訳ありませんが、創作支援中にエラーが発生しました。"
            }
    
    async def _generate_inspiration_themes(self, content: str) -> List[str]:
        """インスピレーションテーマ生成"""
        themes = []
        
        # コンテンツキーワードベースのテーマ
        if "新しい" in content or "革新" in content:
            themes.extend(["未来への架け橋", "可能性の扉", "変革の風"])
        
        if "問題" in content or "課題" in content:
            themes.extend(["解決の糸口", "新たな視点", "突破口の発見"])
        
        if "アイデア" in content:
            themes.extend(["創造の種", "ひらめきの瞬間", "想像力の翼"])
        
        # ランダムなインスピレーション要素を追加
        random_elements = []
        for category in self.inspiration_seeds.values():
            random_elements.append(random.choice(category))
        
        themes.append(f"「{random_elements[0]}」と「{random_elements[1]}」の出会い")
        
        return themes[:4]  # 上位4つまで
    
    async def _generate_idea_variations(self, content: str) -> List[Dict[str, str]]:
        """アイデアバリエーション生成"""
        variations = []
        
        # 基本的なアイデア展開パターン
        base_concepts = content.split()[:3]  # 最初の3つの概念
        
        variation_approaches = [
            {"type": "逆転発想", "description": "既存の考えを180度変えてみる"},
            {"type": "組み合わせ", "description": "異なる要素を融合させる"},
            {"type": "拡張視点", "description": "スケールを大きく変えて考える"},
            {"type": "感情軸", "description": "感情や体験を中心に据える"}
        ]
        
        for approach in variation_approaches:
            variations.append({
                "approach": approach["type"],
                "description": approach["description"],
                "example_direction": f"{approach['type']}で考えると、新しい可能性が見えてきます"
            })
        
        return variations
    
    async def _suggest_creative_techniques(self, content: str) -> List[Dict[str, str]]:
        """創作技法提案"""
        techniques = []
        
        # コンテンツに応じた技法選択
        if "ブレインストーミング" in content or "アイデア" in content:
            techniques.append({
                "technique": "マインドマップ",
                "description": "中心概念から放射状にアイデアを展開",
                "usage": "キーワードを中心に、関連するアイデアを自由に書き出してみましょう"
            })
        
        techniques.extend([
            {
                "technique": "6つの帽子思考法",
                "description": "6つの異なる視点から物事を考察",
                "usage": "事実・感情・批判・楽観・創造・統制の視点で順番に考えてみましょう"
            },
            {
                "technique": "SCAMPER法",
                "description": "7つの質問でアイデアを発展",
                "usage": "代用・結合・適応・修正・他用途・除去・逆転の観点で見直してみましょう"
            }
        ])
        
        return techniques[:3]  # 上位3つまで
    
    async def _suggest_next_steps(self, content: str) -> List[str]:
        """次のステップ提案"""
        steps = []
        
        # コンテンツベースの推奨ステップ
        if "アイデア" in content:
            steps.append("まず、思い浮かんだアイデアを全て書き出してみましょう")
            steps.append("次に、実現可能性を考慮して優先順位をつけてみましょう")
        
        if "問題" in content:
            steps.append("問題を違う角度から見つめ直してみましょう")
            steps.append("解決策のプロトタイプや小さな実験を試してみましょう")
        
        # 一般的な創作ステップ
        steps.extend([
            "他の人の意見やフィードバックを求めてみましょう",
            "アイデアを具体的な形や行動に落とし込んでみましょう",
            "失敗を恐れず、まずは小さく始めてみましょう"
        ])
        
        return steps[:4]  # 上位4つまで
    
    async def _send_creative_response(
        self,
        channel: discord.TextChannel,
        creative_result: Dict[str, Any],
        signal: BotSignal
    ) -> None:
        """創作支援結果応答送信"""
        try:
            if "error" in creative_result:
                await channel.send(creative_result["fallback_response"])
                return
            
            # 創作結果をフォーマット
            response_content = self._format_creative_response(creative_result)
            
            # 応答送信
            sent_message = await self.send_response(channel, response_content)
            
            if sent_message:
                # 応答を履歴に保存
                await self.save_message_to_memory(sent_message)
                self.logger.info(f"Paz creative response sent for signal {signal.signal_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending creative response: {e}")
            await self._send_error_response(signal.channel_id)
    
    def _format_creative_response(self, creative_result: Dict[str, Any]) -> str:
        """創作結果のフォーマット"""
        response = f"🎨 **Paz 創作インスピレーション**\\n\\n"
        
        # インスピレーションテーマ
        if creative_result["inspiration_themes"]:
            response += f"**✨ インスピレーションテーマ:**\\n"
            for theme in creative_result["inspiration_themes"]:
                response += f"- {theme}\\n"
            response += "\\n"
        
        # アイデアバリエーション
        if creative_result["idea_variations"]:
            response += f"**🔄 アイデア展開の方向性:**\\n"
            for variation in creative_result["idea_variations"]:
                response += f"**{variation['approach']}**: {variation['description']}\\n"
            response += "\\n"
        
        # 創作技法
        if creative_result["creative_techniques"]:
            response += f"**🛠️ おすすめ創作技法:**\\n"
            for technique in creative_result["creative_techniques"]:
                response += f"**{technique['technique']}**: {technique['description']}\\n"
                response += f"  💡 {technique['usage']}\\n\\n"
        
        # 次のステップ
        if creative_result["next_steps"]:
            response += f"**🚀 次のアクションプラン:**\\n"
            for i, step in enumerate(creative_result["next_steps"], 1):
                response += f"{i}. {step}\\n"
            response += "\\n"
        
        response += f"*創作レベル: {creative_result['inspiration_level']} | Paz創作支援*"
        
        return response
    
    async def _send_error_response(self, channel_id: int) -> None:
        """エラー応答送信"""
        try:
            channel = self.client.get_channel(channel_id)
            if channel:
                error_message = (
                    "🎨 **Paz エラー**\\n\\n"
                    "申し訳ありませんが、創作支援中にエラーが発生しました。\\n"
                    "しばらく待ってから再度お試しください。"
                )
                await channel.send(error_message)
        except Exception as e:
            self.logger.error(f"Error sending error response: {e}")
    
    # 他のシグナルハンドラー
    async def handle_collaboration_signal(self, signal: BotSignal) -> None:
        """協調シグナル処理"""
        self.logger.info(f"Paz received collaboration signal: {signal.signal_type}")
        # Phase 2以降で実装
    
    async def handle_response_signal(self, signal: BotSignal) -> None:
        """応答シグナル処理"""
        self.logger.info(f"Paz received response signal: {signal.signal_type}")
        # Phase 2以降で実装