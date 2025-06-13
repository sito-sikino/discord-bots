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


class MultiAgentWorkflow:
    """LangGraphベースのマルチエージェントワークフロー"""
    
    def __init__(self, gemini_api_key: str):
        self.logger = logging.getLogger(__name__)
        
        # 環境変数設定
        import os
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # LangSmith統合（オプション）
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            self.logger.info("LangSmith tracing enabled")
        
        # Gemini LLMの初期化（推奨方法）
        self.llm = init_chat_model(
            "gemini-2.0-flash", 
            model_provider="google_genai",
            temperature=0.7,
            max_retries=2
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
            
            # LLMによる分析（1回目のLLM呼び出し）
            response = await self.llm.ainvoke([
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
            
            # Spectra用プロンプト
            spectra_prompt = """あなたはSpectraです。穏やかで包容力のあるファシリテーターとして、
全体を見渡す俯瞰的な視点から会話を進行してください。
一般的な対話、情報提供、会話の整理を行います。"""
            
            # LLMによる応答生成（2回目のLLM呼び出し）
            response = await self.llm.ainvoke([
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
            
            # LynQ用プロンプト
            lynq_prompt = """あなたはLynQです。鋭い洞察力を持つ論理分析の専門家として、
複雑な問題を論理的に分解・整理し、矛盾点や不明確な部分を指摘してください。
事実とデータに基づく客観的な分析を提供します。"""
            
            # LLMによる応答生成（2回目のLLM呼び出し）
            response = await self.llm.ainvoke([
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
            
            # Paz用プロンプト
            paz_prompt = """あなたはPazです。自由で創造的な発想家として、
新しいアイデアや視点を提供し、固定観念を打破する発想を促してください。
インスピレーションを重視し、可能性を広げる創造的な解決策を提案します。"""
            
            # LLMによる応答生成（2回目のLLM呼び出し）
            response = await self.llm.ainvoke([
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
        """分析プロンプトの構築"""
        return f"""あなたは基底エージェントとして、ユーザーメッセージを深層分析し、
最適な応答エージェントを判断してください。

分析項目:
1. 意図（intent）: ユーザーの主な目的
2. 複雑度（complexity）: low/medium/high
3. 感情（sentiment）: positive/neutral/negative
4. 推奨エージェント:
   - spectra: 一般対話、ファシリテーション、統合的視点
   - lynq: 論理分析、問題解決、客観的思考
   - paz: 創作、アイデア、発想、インスピレーション
   - multi: 複数の視点が必要な複合要求
5. 信頼度（confidence）: 0.0-1.0
6. 理由（reasoning）: 判断の根拠

会話コンテキスト:
{context}

JSON形式で応答してください。"""
    
    def _parse_analysis_response(self, response: str) -> Dict:
        """分析結果の解析"""
        # Phase 2では簡略版実装
        # 本来はJSONパースを行う
        analysis = {
            "intent": "general_conversation",
            "complexity": "medium",
            "sentiment": "neutral",
            "suggested_agent": "spectra",
            "confidence": 0.8,
            "reasoning": "一般的な対話として処理"
        }
        
        # キーワードベースの簡易判定（暫定）
        response_lower = response.lower()
        if any(word in response_lower for word in ["論理", "分析", "問題", "矛盾"]):
            analysis["suggested_agent"] = "lynq"
            analysis["reasoning"] = "論理分析が必要と判断"
        elif any(word in response_lower for word in ["アイデア", "創作", "発想", "インスピレーション"]):
            analysis["suggested_agent"] = "paz"
            analysis["reasoning"] = "創造的な視点が必要と判断"
        elif "複合" in response_lower or "複数" in response_lower:
            analysis["suggested_agent"] = "multi"
            analysis["reasoning"] = "複数の視点が必要と判断"
        
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