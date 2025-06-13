"""
LangGraphベースのマルチエージェントワークフロー

統合プロセス型エージェントシステムのワークフロー管理
- StateGraphによるワークフロー定義
- 条件付きエッジによるルーティング
- 効率的な3パターン動作フロー
"""

import logging
from typing import Dict, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage

from .state import AgentState, AnalysisResult
from ..core.gemini_api_manager import GeminiAPIManager, RateLimitConfig


class MultiAgentWorkflow:
    """LangGraphベースのマルチエージェントワークフロー"""
    
    def __init__(self, gemini_api_key: str):
        self.logger = logging.getLogger(__name__)
        
        # レート制限設定（Phase 2要件に基づく）
        rate_config = RateLimitConfig(
            max_calls_per_minute=30,  # 保守的設定
            max_calls_per_hour=500,   # 保守的設定  
            max_concurrent_calls=1,   # シーケンシャルアクセス
            retry_delay_base=1.0,
            max_retries=3,
            timeout_seconds=30
        )
        
        # Gemini API管理クラス初期化
        self.api_manager = GeminiAPIManager(
            api_key=gemini_api_key,
            model_name="gemini-2.0-flash",
            rate_limit_config=rate_config
        )
        
        # ワークフローの構築
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """ワークフローグラフの構築"""
        # StateGraphの初期化
        workflow = StateGraph(AgentState)
        
        # ノードの追加
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("spectra_respond", self._spectra_respond_node)
        workflow.add_node("lynq_respond", self._lynq_respond_node)
        workflow.add_node("paz_respond", self._paz_respond_node)
        workflow.add_node("multi_coordinate", self._multi_coordinate_node)
        
        # エッジの追加
        workflow.add_edge(START, "analyze")
        
        # 条件付きエッジ（ルーティング）
        workflow.add_conditional_edges(
            "analyze",
            self._route_decision,
            {
                "spectra": "spectra_respond",
                "lynq": "lynq_respond",
                "paz": "paz_respond",
                "multi": "multi_coordinate",
                "end": END
            }
        )
        
        # 各エージェントノードからENDへ
        workflow.add_edge("spectra_respond", END)
        workflow.add_edge("lynq_respond", END)
        workflow.add_edge("paz_respond", END)
        
        # マルチ協調の場合のフロー
        workflow.add_conditional_edges(
            "multi_coordinate",
            self._multi_route,
            {
                "lynq_first": "lynq_respond",
                "paz_first": "paz_respond",
                "end": END
            }
        )
        
        return workflow
    
    async def _analyze_node(self, state: AgentState) -> Dict:
        """Spectra分析ノード - 深層分析と意思決定"""
        try:
            # 最新のユーザーメッセージを取得
            user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_message = msg
                    break
            
            if not user_message:
                return {"error": "No user message found"}
            
            # 分析プロンプトの構築
            analysis_prompt = self._build_analysis_prompt(
                user_message.content,
                state.get("conversation_context", "")
            )
            
            # LLMによる分析（1回目のLLM呼び出し）- レート制限対応
            response = await self.api_manager.invoke_with_rate_limit([
                SystemMessage(content=analysis_prompt),
                HumanMessage(content=user_message.content)
            ])
            
            # 分析結果の解析
            analysis = self._parse_analysis_response(response.content)
            
            # ルーティング決定
            routing = self._determine_routing(analysis)
            
            return {
                "analysis": analysis,
                "routing": routing,
                "current_agent": "spectra"
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze node: {e}")
            return {"error": str(e), "routing": "spectra"}
    
    async def _spectra_respond_node(self, state: AgentState) -> Dict:
        """Spectra応答ノード - 一般対話・ファシリテーション"""
        try:
            user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_message = msg
                    break
            
            # Spectra用プロンプト - Phase 2改善版
            spectra_prompt = """🌟 あなたはSpectra - 統合的ファシリテーター・基底エージェントです。

## 🎯 あなたの役割・個性
- **穏やかで包容力のあるファシリテーター**として会話を進行
- **全体を見渡す俯瞰的視点**で状況を整理・統合
- **中立的な立場**から建設的な方向に導く
- **システム全体の責任者・調整者**として機能

## 💫 専門領域
- **一般対話・雑談**: 自然で親しみやすい会話
- **情報提供・整理**: 複雑な情報をわかりやすく整理
- **会話ファシリテーション**: 参加者の発言を分析し全体バランスを調整
- **統合的視点**: 複合的な要求への統合的な応答
- **メタ的な進行**: 会話の流れを客観視し、適切に方向付け

## 🎨 応答スタイル
- **温かみのある丁寧な口調**
- **相手の気持ちに寄り添う共感的な姿勢**
- **建設的で前向きな視点の提供**
- **適度な親しみやすさを保ちながらも、専門性を感じさせる応答**

## 🌟 今回の応答方針
相手のメッセージを深く理解し、最も価値ある応答を提供してください。
必要に応じて質問を投げかけ、対話を深めていってください。

---
**あなたならではの温かく、俯瞰的で統合的な視点から応答してください。**"""
            
            # LLMによる応答生成（2回目のLLM呼び出し）- レート制限対応
            response = await self.api_manager.invoke_with_rate_limit([
                SystemMessage(content=spectra_prompt),
                HumanMessage(content=user_message.content)
            ])
            
            # LangChain標準AIMessageを作成
            new_message = AIMessage(content=response.content)
            
            return {"messages": [new_message]}
            
        except Exception as e:
            self.logger.error(f"Error in spectra respond node: {e}")
            return {"error": str(e)}
    
    async def _lynq_respond_node(self, state: AgentState) -> Dict:
        """LynQ応答ノード - 論理分析専門"""
        try:
            user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_message = msg
                    break
            
            # LynQ用プロンプト - Phase 2改善版
            lynq_prompt = """📊 あなたはLynQ - 論理分析の専門家・思考探究者です。

## 🎯 あなたの役割・個性
- **鋭い洞察力を持つ論理分析の専門家**
- **論理的で客観的なアプローチ**を重視
- **深く掘り下げる質問**を投げかけ、思考を促進
- **事実と論理を重んじる分析者**として機能

## 🔬 専門分析機能
- **問題分解・構造化**: 複雑な問題を論理的に分解・整理
- **矛盾点・課題指摘**: 議論の矛盾点や不明確な部分を的確に指摘
- **客観的検証**: データ・事実に基づく客観的分析を提供
- **深い思考促進**: 本質に迫る質問で深い思考を促進
- **論理的推論**: 論理的な推論と分析を展開

## 🔍 分析アプローチ
- **事実ベース**: 感情論ではなく、事実とデータに基づく分析
- **構造的思考**: 問題を要素に分解し、関係性を明確化
- **批判的検証**: 前提条件や論理の飛躍を厳密に検証
- **本質追求**: 表面的ではなく、本質的な問題点を探求

## 🎨 応答スタイル
- **明確で論理的な説明**
- **具体的な根拠を示した分析**
- **建設的な批判と改善提案**
- **思考を促進する質問の投げかけ**

## 📊 今回の分析方針
提示された内容を多角的に分析し、論理的な視点から価値ある洞察を提供してください。
必要に応じて前提条件を確認し、より深い理解に導いてください。

---
**あなたの鋭い論理分析力で、本質的な洞察を提供してください。**"""
            
            # LLMによる応答生成（2回目のLLM呼び出し）- レート制限対応
            response = await self.api_manager.invoke_with_rate_limit([
                SystemMessage(content=lynq_prompt),
                HumanMessage(content=user_message.content)
            ])
            
            # LangChain標準AIMessageを作成
            new_message = AIMessage(content=response.content)
            
            return {"messages": [new_message]}
            
        except Exception as e:
            self.logger.error(f"Error in lynq respond node: {e}")
            return {"error": str(e)}
    
    async def _paz_respond_node(self, state: AgentState) -> Dict:
        """Paz応答ノード - 創作アイデア専門"""
        try:
            user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_message = msg
                    break
            
            # Paz用プロンプト - Phase 2改善版  
            paz_prompt = """🎨 あなたはPaz - 創造的発想家・インスピレーション創発者です。

## 🎯 あなたの役割・個性
- **自由で創造的な発想家**として新たな可能性を探求
- **既存の枠を超えた提案**で固定観念を打破
- **インスピレーションを重視**し、直感的な洞察を提供
- **可能性を広げる創造者**として革新的な視点を展開

## ✨ 専門創作機能
- **アイデア創発・発想**: 新しいアイデアや視点を無限に生み出す
- **インスピレーション提供**: 創作活動や思考に火を灯すきっかけを与える
- **固定観念打破**: 既存の枠組みや前提を疑い、新たな発想を促す
- **創造的解決**: 従来にない革新的で創造的な解決策を提案
- **発散的思考**: 多角的・発散的思考で可能性の地平を拡張

## 🌈 創造アプローチ
- **自由な発想**: 制約にとらわれない自由で柔軟な思考
- **直感重視**: 論理だけでなく、直感やひらめきを大切に
- **遊び心**: 楽しさや遊び心を取り入れた発想法
- **実験精神**: 新しいことにチャレンジする実験的な姿勢

## 🎨 応答スタイル
- **エネルギッシュで前向きな口調**
- **想像力をかき立てる表現**
- **具体的で実践的なアイデア提案**
- **ワクワクする可能性の提示**

## 🌟 今回の創作方針
提示された内容から創造的なインスピレーションを見出し、新たな視点やアイデアを提供してください。
既存の枠組みにとらわれず、革新的で実現可能な提案を心がけてください。

---
**あなたの創造力で、新たな可能性の扉を開いてください！**"""
            
            # LLMによる応答生成（2回目のLLM呼び出し）- レート制限対応
            response = await self.api_manager.invoke_with_rate_limit([
                SystemMessage(content=paz_prompt),
                HumanMessage(content=user_message.content)
            ])
            
            # LangChain標準AIMessageを作成
            new_message = AIMessage(content=response.content)
            
            return {"messages": [new_message]}
            
        except Exception as e:
            self.logger.error(f"Error in paz respond node: {e}")
            return {"error": str(e)}
    
    async def _multi_coordinate_node(self, state: AgentState) -> Dict:
        """マルチエージェント協調ノード"""
        # Phase 3で実装予定
        # 現在は簡略版として、調整メッセージのみ追加
        coordination_message = AIMessage(
            content="🤝 複合的なご要求ですね！チーム一丸となって対応します。\n\n"
                   "📊 LynQ（論理分析）→ 🎨 Paz（創作アイデア）→ 🌟 Spectra（統合）の順で進めます。"
        )
        
        return {
            "messages": [coordination_message],
            "collaboration_mode": True,
            "collaboration_agents": ["lynq", "paz", "spectra"]
        }
    
    def _route_decision(self, state: AgentState) -> str:
        """分析結果に基づくルーティング決定"""
        routing = state.get("routing", "spectra")
        
        # エラーの場合はSpectraが対応
        if state.get("error"):
            return "spectra"
        
        return routing
    
    def _multi_route(self, state: AgentState) -> str:
        """マルチエージェント協調のルーティング"""
        # Phase 3で詳細実装
        # 現在は簡略版として終了
        return "end"
    
    def _build_analysis_prompt(self, user_message: str, context: str) -> str:
        """分析プロンプトの構築 - Phase 2改善版"""
        return f"""🌟 あなたはSpectra（基底エージェント）として、最高レベルの深層分析・メタ判断を実行してください。

## 🧠 深層理解 - LLM知的分析力最大活用
以下の多層的分析を実行:

### 1. 意図分析（Intent Analysis）
- **表面的要求**: 文字通りの要望
- **深層的意図**: 隠れた真の目的・動機
- **文脈的背景**: 会話の流れからの推察
- **感情的側面**: 感情状態・ニーズの分析

### 2. 複雑度・専門性判定
- **問題の性質**: 単純/複合/多層的
- **必要な専門性**: 論理分析/創造発想/統合調整
- **応答の期待値**: 情報提供/深い分析/創造的発想

### 3. 最適エージェント判断（メタ認知）
各エージェントの価値提供能力を評価:

**🌟 Spectra (私)**: 
- 一般対話・雑談・情報提供
- 会話ファシリテーション・進行
- 複合要求の統合・調整
- 全体俯瞰・メタ視点

**📊 LynQ**: 
- 論理分析・問題分解・構造化
- 矛盾指摘・客観的検証
- データ分析・事実ベース思考
- 深い思考促進・質問投げかけ

**🎨 Paz**: 
- 創作・アイデア創発・インスピレーション
- 発散的思考・固定観念打破
- 新しい視点・革新的解決策
- ブレインストーミング・可能性拡張

**🤝 Multi**: 
- 論理+創造の複合要求
- 多角的視点が必要な複雑案件
- Sequential協調による価値最大化

### 4. 戦略的判断
- **単独最適**: どのエージェントが最高価値を提供できるか
- **協調必要性**: 複数視点が本質的に必要か
- **効率性**: レート制限内での最適な応答戦略

## 📋 会話コンテキスト
{context}

## 🎯 出力フォーマット
以下のJSON形式で精密な分析結果を返してください:

```json
{{
    "deep_intent": {{
        "surface_request": "表面的要求",
        "underlying_purpose": "深層的意図", 
        "emotional_state": "感情状態",
        "context_background": "文脈的背景"
    }},
    "complexity_analysis": {{
        "problem_type": "単純/複合/多層的",
        "required_expertise": "必要な専門性",
        "cognitive_load": "low/medium/high"
    }},
    "optimal_strategy": {{
        "recommended_agent": "spectra/lynq/paz/multi",
        "confidence": 0.95,
        "reasoning": "判断の詳細根拠",
        "expected_value": "期待される価値提供"
    }},
    "meta_judgment": {{
        "why_this_agent": "なぜこのエージェントが最適か",
        "alternative_consideration": "他の選択肢の検討",
        "strategic_impact": "システム全体への戦略的影響"
    }}
}}
```

🧠 **最高レベルの知的判断力を発揮し、ユーザーに最大価値をもたらす戦略的決定を行ってください。**"""
    
    def _parse_analysis_response(self, response: str) -> Dict:
        """分析結果の解析 - Phase 2改善版（JSON解析対応）"""
        import json
        import re
        
        try:
            # JSONブロックを抽出（```json ... ``` の形式）
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                parsed_data = json.loads(json_str)
                
                # 構造化データから必要な情報を抽出
                analysis = {
                    "deep_intent": parsed_data.get("deep_intent", {}),
                    "complexity_analysis": parsed_data.get("complexity_analysis", {}),
                    "optimal_strategy": parsed_data.get("optimal_strategy", {}),
                    "meta_judgment": parsed_data.get("meta_judgment", {}),
                    
                    # 後方互換性のための従来フィールド
                    "suggested_agent": parsed_data.get("optimal_strategy", {}).get("recommended_agent", "spectra"),
                    "confidence": parsed_data.get("optimal_strategy", {}).get("confidence", 0.8),
                    "reasoning": parsed_data.get("optimal_strategy", {}).get("reasoning", "分析実行")
                }
                
                self.logger.info(f"JSON分析成功: {analysis['suggested_agent']} (信頼度: {analysis['confidence']})")
                return analysis
                
            else:
                self.logger.warning("JSON形式が見つからない、フォールバック分析に切り替え")
                
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"JSON解析エラー、フォールバック分析に切り替え: {e}")
        
        # フォールバック: キーワードベース分析（Phase 1互換）
        return self._fallback_analysis(response)
    
    def _fallback_analysis(self, response: str) -> Dict:
        """フォールバック分析 - キーワードベース"""
        analysis = {
            "deep_intent": {"surface_request": "分析不能", "underlying_purpose": "不明"},
            "complexity_analysis": {"problem_type": "単純", "cognitive_load": "medium"},
            "optimal_strategy": {"recommended_agent": "spectra", "confidence": 0.6, "reasoning": "フォールバック判定"},
            "meta_judgment": {"why_this_agent": "JSON解析失敗のためSpectraが対応"},
            
            # 後方互換性
            "suggested_agent": "spectra",
            "confidence": 0.6,
            "reasoning": "フォールバック判定"
        }
        
        # キーワードベースの簡易判定
        response_lower = response.lower()
        if any(word in response_lower for word in ["論理", "分析", "問題", "矛盾", "検証", "データ"]):
            analysis["suggested_agent"] = "lynq"
            analysis["optimal_strategy"]["recommended_agent"] = "lynq"
            analysis["reasoning"] = "論理分析キーワード検出"
        elif any(word in response_lower for word in ["アイデア", "創作", "発想", "インスピレーション", "創造", "革新"]):
            analysis["suggested_agent"] = "paz"
            analysis["optimal_strategy"]["recommended_agent"] = "paz"
            analysis["reasoning"] = "創造的キーワード検出"
        elif any(word in response_lower for word in ["複合", "複数", "協調", "多角"]):
            analysis["suggested_agent"] = "multi"
            analysis["optimal_strategy"]["recommended_agent"] = "multi"
            analysis["reasoning"] = "複合要求キーワード検出"
        
        return analysis
    
    def _determine_routing(self, analysis: Dict) -> str:
        """分析結果からルーティングを決定"""
        suggested_agent = analysis.get("suggested_agent", "spectra")
        
        # エージェント名をルーティング名に変換
        routing_map = {
            "spectra": "spectra",
            "lynq": "lynq",
            "paz": "paz",
            "multi": "multi"
        }
        
        return routing_map.get(suggested_agent, "spectra")
    
    async def process_message(self, initial_state: AgentState) -> Dict:
        """メッセージ処理の実行"""
        try:
            # ワークフローの実行
            result = await self.app.ainvoke(initial_state)
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {"error": str(e)}
    
    def get_api_statistics(self) -> Dict:
        """API使用統計取得"""
        return self.api_manager.get_api_stats()