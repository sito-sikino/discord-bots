"""
Spectra Communicator Agent
進行役・コミュニケーションハブ・戦略的対話進行・議論構造化・集合知引き出し
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class SpectraAgent:
    """Spectra Communicator - Discussion facilitator and strategic communication"""
    
    def __init__(self, model_manager, memory_manager):
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.agent_name = "spectra"
        self.logger = self._setup_logging()
        
        # Spectra-specific configuration
        self.personality = {
            "role": "コミュニケーション促進者",
            "strengths": ["対話進行", "議論構造化", "集合知引き出し"],
            "temperature": 0.8,  # Slightly more creative for communication
            "system_prompt": self._get_system_prompt()
        }
        
        self.logger.info("Spectra Communicator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup Spectra-specific logger"""
        logger = logging.getLogger("SpectraAgent")
        
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler("bot.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s | SPECTRA | %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def _get_system_prompt(self) -> str:
        """Get Spectra's specialized system prompt"""
        return """あなたはSpectra Communicatorです。

【役割】
- 進行役として対話を円滑に導く
- コミュニケーションハブとして情報を整理・統合
- 戦略的な対話進行で本質的な議論を促進
- 議論を構造化し、論点を明確にする
- 参加者の集合知を効果的に引き出す

【特徴】
- 親しみやすく、包容力のあるコミュニケーション
- 相手の意図を正確に理解し、適切に応答
- 複雑な話題をわかりやすく整理
- 建設的な議論を促進する質問技術
- 多様な視点を尊重し、統合する能力

【応答スタイル】
- 簡潔で分かりやすい説明
- 相手の立場に配慮した丁寧な表現
- 議論を深める効果的な質問
- 要点を構造化した整理
- 次のアクションを明確に示す

常に対話相手との信頼関係を重視し、建設的なコミュニケーションを心がけてください。"""
    
    async def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input as Spectra Communicator"""
        try:
            self.logger.info(f"processing communication task: {user_input[:50]}...")
            
            # Search for relevant memories
            relevant_memories = await self.memory_manager.search_memory(user_input)
            memory_context = "; ".join(relevant_memories) if relevant_memories else ""
            
            # Build context-aware system prompt
            system_content = self.personality["system_prompt"]
            if memory_context:
                system_content += f"\n\n【参考記憶】\n{memory_context}"
            
            if context and context.get("conversation_history"):
                system_content += f"\n\n【会話履歴】\n{context['conversation_history']}"
            
            # Generate response using Spectra's specialized approach
            user_msg = HumanMessage(content=user_input)
            system_msg = SystemMessage(content=system_content)
            
            ai_response = await self.model_manager.generate_response(
                [system_msg, user_msg],
                agent_name=self.agent_name,
                memory_context=memory_context
            )
            
            # Analyze communication effectiveness
            communication_analysis = self._analyze_communication(user_input, ai_response.content)
            
            self.logger.info(f"communication facilitated: {communication_analysis['effectiveness']}")
            
            return {
                "agent": "Spectra",
                "response": ai_response.content,
                "reasoning": "コミュニケーション促進として処理",
                "analysis": communication_analysis,
                "memory_used": len(relevant_memories),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"processing error: {e}")
            return {
                "agent": "Spectra",
                "response": "申し訳ありません。コミュニケーション処理中にエラーが発生しました。",
                "reasoning": "エラー処理",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_communication(self, input_text: str, response_text: str) -> Dict[str, Any]:
        """Analyze communication effectiveness"""
        input_lower = input_text.lower()
        response_lower = response_text.lower()
        
        # Communication indicators
        question_indicators = ["？", "?", "どう", "なぜ", "なに", "いつ", "どこ", "だれ"]
        explanation_indicators = ["説明", "教え", "わかりやすく", "詳しく"]
        facilitation_indicators = ["まとめ", "整理", "構造", "進行"]
        
        # Analyze input type
        is_question = any(indicator in input_lower for indicator in question_indicators)
        needs_explanation = any(indicator in input_lower for indicator in explanation_indicators)
        needs_facilitation = any(indicator in input_lower for indicator in facilitation_indicators)
        
        # Analyze response quality
        response_length = len(response_text)
        has_structure = any(marker in response_text for marker in ["1.", "・", "①", "【", "■"])
        has_questions = any(marker in response_text for marker in ["？", "?"])
        
        # Determine effectiveness
        effectiveness = "high"
        if is_question and response_length < 50:
            effectiveness = "low"
        elif needs_explanation and not has_structure:
            effectiveness = "medium"
        elif needs_facilitation and not has_questions:
            effectiveness = "medium"
        
        return {
            "effectiveness": effectiveness,
            "input_type": {
                "is_question": is_question,
                "needs_explanation": needs_explanation,
                "needs_facilitation": needs_facilitation
            },
            "response_quality": {
                "length": response_length,
                "has_structure": has_structure,
                "has_questions": has_questions
            }
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get Spectra agent information"""
        return {
            "name": "Spectra Communicator",
            "role": self.personality["role"],
            "strengths": self.personality["strengths"],
            "temperature": self.personality["temperature"],
            "specialization": "対話促進・コミュニケーション統合",
            "capabilities": [
                "戦略的対話進行",
                "議論構造化",
                "集合知引き出し",
                "要点整理",
                "建設的質問"
            ]
        }