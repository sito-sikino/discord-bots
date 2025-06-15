"""
Paz Creator Agent
創造者・創造的発想・可能性探索・発散思考・メタファー構築・突破的アイデア
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class PazAgent:
    """Paz Creator - Creative thinking and possibility exploration specialist"""
    
    def __init__(self, model_manager, memory_manager):
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.agent_name = "paz"
        self.logger = self._setup_logging()
        
        # Paz-specific configuration
        self.personality = {
            "role": "創造的発想専門家",
            "strengths": ["創造的発想", "可能性探索", "突破的アイデア"],
            "temperature": 0.9,  # Most creative and exploratory
            "system_prompt": self._get_system_prompt()
        }
        
        self.logger.info("Paz Creator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup Paz-specific logger"""
        logger = logging.getLogger("PazAgent")
        
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler("bot.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s | PAZ     | %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def _get_system_prompt(self) -> str:
        """Get Paz's specialized system prompt"""
        return """あなたはPaz Creatorです。

【役割】
- 創造的発想の専門家として革新的なアイデアを生成
- 可能性探索で従来の枠を超えた視点を提供
- 発散思考で多様な選択肢と機会を発見
- メタファーや比喩を使った直感的な理解促進
- 突破的アイデアで既存の限界を打破

【特徴】
- 自由で柔軟な発想力
- 異なる分野からのアナロジー活用
- 直感と論理のバランス取れた思考
- 制約を機会として捉える視点
- 美しさや調和を重視した表現

【応答スタイル】
- インスピレーションに満ちた表現
- 比喩やメタファーを効果的に活用
- 複数の可能性や選択肢を提示
- 感情に訴える魅力的な描写
- 希望や可能性を感じさせる内容

【創造的プロセス】
1. 既存の枠組みを疑い、新しい角度から観察
2. 異分野の知識や経験からヒントを得る
3. 直感と論理を組み合わせたアイデア生成
4. 実現可能性より可能性の広がりを重視
5. 美的感覚や調和の観点から最適化

【発想技法】
- ブレインストーミング
- アナロジー思考
- 逆転の発想
- シナリオプランニング
- デザイン思考

常に好奇心を持ち、制約を可能性に変える創造的なソリューションを提供してください。"""
    
    async def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input as Paz Creator"""
        try:
            self.logger.info(f"processing creative task: {user_input[:50]}...")
            
            # Search for relevant memories
            relevant_memories = await self.memory_manager.search_memory(user_input)
            memory_context = "; ".join(relevant_memories) if relevant_memories else ""
            
            # Build creative system prompt
            system_content = self.personality["system_prompt"]
            if memory_context:
                system_content += f"\n\n【創造的素材】\n{memory_context}"
            
            if context and context.get("creative_constraints"):
                system_content += f"\n\n【創造的制約】\n{context['creative_constraints']}"
            
            # Generate response using Paz's creative approach
            user_msg = HumanMessage(content=user_input)
            system_msg = SystemMessage(content=system_content)
            
            ai_response = await self.model_manager.generate_response(
                [system_msg, user_msg],
                agent_name=self.agent_name,
                memory_context=memory_context
            )
            
            # Analyze creative elements
            creativity_analysis = self._analyze_creativity(user_input, ai_response.content)
            
            self.logger.info(f"creative exploration completed: {creativity_analysis['innovation_level']}")
            
            return {
                "agent": "Paz",
                "response": ai_response.content,
                "reasoning": "創造的発想として処理",
                "analysis": creativity_analysis,
                "memory_used": len(relevant_memories),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"creative processing error: {e}")
            return {
                "agent": "Paz",
                "response": "申し訳ありません。創造的処理中にエラーが発生しました。",
                "reasoning": "エラー処理",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_creativity(self, input_text: str, response_text: str) -> Dict[str, Any]:
        """Analyze creative elements and innovation"""
        input_lower = input_text.lower()
        response_lower = response_text.lower()
        
        # Creative indicators
        ideation_keywords = ["アイデア", "発想", "ひらめき", "創造", "革新"]
        exploration_keywords = ["可能性", "機会", "選択肢", "代替", "新しい"]
        metaphor_keywords = ["例え", "比喩", "ような", "みたい", "まるで"]
        innovation_keywords = ["突破", "画期的", "斬新", "独創", "オリジナル"]
        
        # Count creative elements
        ideation_count = sum(1 for keyword in ideation_keywords if keyword in input_lower)
        exploration_count = sum(1 for keyword in exploration_keywords if keyword in input_lower)
        metaphor_count = sum(1 for keyword in metaphor_keywords if keyword in response_lower)
        innovation_count = sum(1 for keyword in innovation_keywords if keyword in response_lower)
        
        # Analyze response creativity
        has_multiple_ideas = response_text.count("・") >= 3 or response_text.count("1.") >= 2
        has_metaphors = metaphor_count > 0
        has_questions = response_text.count("？") + response_text.count("?") >= 2
        has_inspiration = any(word in response_lower for word in ["想像", "イメージ", "夢", "希望"])
        
        # Determine innovation level
        creative_elements = ideation_count + exploration_count + innovation_count
        if creative_elements >= 3 and (has_multiple_ideas or has_metaphors):
            innovation_level = "breakthrough"
        elif creative_elements >= 2 or has_multiple_ideas:
            innovation_level = "innovative"
        elif creative_elements >= 1:
            innovation_level = "creative"
        else:
            innovation_level = "conventional"
        
        # Calculate creative diversity
        creative_diversity = 0
        if has_multiple_ideas:
            creative_diversity += 2
        if has_metaphors:
            creative_diversity += 2
        if has_questions:
            creative_diversity += 1
        if has_inspiration:
            creative_diversity += 1
        
        return {
            "innovation_level": innovation_level,
            "creative_elements": {
                "ideation": ideation_count,
                "exploration": exploration_count,
                "metaphor": metaphor_count,
                "innovation": innovation_count,
                "total": creative_elements
            },
            "response_creativity": {
                "has_multiple_ideas": has_multiple_ideas,
                "has_metaphors": has_metaphors,
                "has_questions": has_questions,
                "has_inspiration": has_inspiration,
                "creative_diversity": creative_diversity
            }
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get Paz agent information"""
        return {
            "name": "Paz Creator",
            "role": self.personality["role"],
            "strengths": self.personality["strengths"],
            "temperature": self.personality["temperature"],
            "specialization": "創造的発想・可能性探索",
            "capabilities": [
                "創造的アイデア生成",
                "可能性探索",
                "メタファー構築",
                "発散思考",
                "革新的ソリューション"
            ]
        }