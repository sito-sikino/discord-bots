"""
LynQ Analyzer Agent
分析者・論理分析・構造化思考・概念明確化・矛盾検出・論理構造可視化
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class LynQAgent:
    """LynQ Analyzer - Logic analysis and structured thinking specialist"""
    
    def __init__(self, model_manager, memory_manager):
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.agent_name = "lynq"
        self.logger = self._setup_logging()
        
        # LynQ-specific configuration
        self.personality = {
            "role": "論理分析専門家",
            "strengths": ["論理分析", "構造化思考", "矛盾検出"],
            "temperature": 0.3,  # More focused and logical
            "system_prompt": self._get_system_prompt()
        }
        
        self.logger.info("LynQ Analyzer initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup LynQ-specific logger"""
        logger = logging.getLogger("LynQAgent")
        
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler("bot.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s | LYNQ    | %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def _get_system_prompt(self) -> str:
        """Get LynQ's specialized system prompt"""
        return """あなたはLynQ Analyzerです。

【役割】
- 論理分析の専門家として複雑な問題を構造化
- 構造化思考で情報を体系的に整理
- 概念を明確化し、曖昧さを排除
- 矛盾や論理的な問題を検出・指摘
- 論理構造を可視化して理解を促進

【特徴】
- 厳密で論理的な思考プロセス
- 情報を階層化・カテゴリ化する能力
- 因果関係や相関関係の分析
- 前提条件と結論の明確な区別
- データや事実に基づく客観的判断

【応答スタイル】
- 論点を明確に構造化
- ステップバイステップの論理展開
- 根拠と結論を明確に分離
- 図表や箇条書きを活用した整理
- 反対意見や代替案も考慮

【分析フレームワーク】
1. 問題の定義と範囲の明確化
2. 関連要素の抽出と分類
3. 因果関係・相関関係の分析
4. 論理的矛盾の検出
5. 結論と根拠の整理

常に客観性と論理性を重視し、感情に左右されない分析を提供してください。"""
    
    async def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input as LynQ Analyzer"""
        try:
            self.logger.info(f"processing analysis task: {user_input[:50]}...")
            
            # Search for relevant memories
            relevant_memories = await self.memory_manager.search_memory(user_input)
            memory_context = "; ".join(relevant_memories) if relevant_memories else ""
            
            # Build analytical system prompt
            system_content = self.personality["system_prompt"]
            if memory_context:
                system_content += f"\n\n【分析データ】\n{memory_context}"
            
            if context and context.get("analysis_requirements"):
                system_content += f"\n\n【分析要件】\n{context['analysis_requirements']}"
            
            # Generate response using LynQ's analytical approach
            user_msg = HumanMessage(content=user_input)
            system_msg = SystemMessage(content=system_content)
            
            ai_response = await self.model_manager.generate_response(
                [system_msg, user_msg],
                agent_name=self.agent_name,
                memory_context=memory_context
            )
            
            # Perform logical structure analysis
            logical_analysis = self._analyze_logical_structure(user_input, ai_response.content)
            
            self.logger.info(f"logical analysis completed: {logical_analysis['complexity']}")
            
            return {
                "agent": "LynQ",
                "response": ai_response.content,
                "reasoning": "論理分析として処理",
                "analysis": logical_analysis,
                "memory_used": len(relevant_memories),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"analysis error: {e}")
            return {
                "agent": "LynQ",
                "response": "申し訳ありません。論理分析処理中にエラーが発生しました。",
                "reasoning": "エラー処理",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_logical_structure(self, input_text: str, response_text: str) -> Dict[str, Any]:
        """Analyze logical structure and complexity"""
        input_lower = input_text.lower()
        response_lower = response_text.lower()
        
        # Logical indicators
        analysis_keywords = ["分析", "検討", "比較", "評価", "検証"]
        causal_keywords = ["原因", "結果", "理由", "なぜ", "ため", "による"]
        structure_keywords = ["構造", "関係", "パターン", "体系", "分類"]
        logic_keywords = ["論理", "矛盾", "一貫", "根拠", "証拠"]
        
        # Count logical elements
        analysis_count = sum(1 for keyword in analysis_keywords if keyword in input_lower)
        causal_count = sum(1 for keyword in causal_keywords if keyword in input_lower)
        structure_count = sum(1 for keyword in structure_keywords if keyword in input_lower)
        logic_count = sum(1 for keyword in logic_keywords if keyword in input_lower)
        
        # Analyze response structure
        has_enumeration = any(marker in response_text for marker in ["1.", "2.", "①", "②", "・"])
        has_categorization = any(marker in response_text for marker in ["【", "■", "▼", "◆"])
        has_conclusion = any(marker in response_lower for marker in ["結論", "まとめ", "したがって", "よって"])
        has_reasoning = any(marker in response_lower for marker in ["理由", "根拠", "なぜなら", "というのは"])
        
        # Determine complexity
        total_logical_elements = analysis_count + causal_count + structure_count + logic_count
        if total_logical_elements >= 3:
            complexity = "high"
        elif total_logical_elements >= 1:
            complexity = "medium"
        else:
            complexity = "low"
        
        # Calculate analytical depth
        analytical_depth = 0
        if has_enumeration:
            analytical_depth += 1
        if has_categorization:
            analytical_depth += 1
        if has_conclusion:
            analytical_depth += 1
        if has_reasoning:
            analytical_depth += 1
        
        return {
            "complexity": complexity,
            "logical_elements": {
                "analysis": analysis_count,
                "causal": causal_count,
                "structure": structure_count,
                "logic": logic_count,
                "total": total_logical_elements
            },
            "response_structure": {
                "has_enumeration": has_enumeration,
                "has_categorization": has_categorization,
                "has_conclusion": has_conclusion,
                "has_reasoning": has_reasoning,
                "analytical_depth": analytical_depth
            }
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get LynQ agent information"""
        return {
            "name": "LynQ Analyzer",
            "role": self.personality["role"],
            "strengths": self.personality["strengths"],
            "temperature": self.personality["temperature"],
            "specialization": "論理分析・構造化思考",
            "capabilities": [
                "論理構造分析",
                "矛盾検出",
                "概念明確化",
                "因果関係分析",
                "体系的整理"
            ]
        }