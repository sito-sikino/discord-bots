"""
Multi-agent orchestrator using LangGraph Supervisor pattern
Coordinates Spectra, LynQ, and Paz agents with shared resources
"""
import logging
from typing import TypedDict, Annotated, List, Any, Literal, Optional
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import our core modules
from core.config import config
from core.memory import MemoryManager
from core.models import ModelManager
from core.obsidian import ObsidianManager
from core.mcp import MCPManager
from core.semantic_router import SemanticRouter


# ===== State Definition =====

class MultiAgentState(TypedDict):
    """Multi-agent system state"""
    messages: Annotated[List[Any], add_messages]
    current_input: str
    active_agent: str
    task_type: Literal["communication", "analysis", "creativity"]
    agent_reasoning: str
    final_response: str


# ===== Agent Implementations =====

class SpectraAgent:
    """Spectra - Communication facilitator and dialogue promotion specialist"""
    
    def __init__(self, model_manager: ModelManager, memory_manager: MemoryManager):
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.agent_name = "spectra"
        self.logger = logging.getLogger("SpectraAgent")
    
    async def process(self, state: MultiAgentState) -> dict:
        """Process communication and facilitation tasks"""
        user_input = state.get("current_input", "")
        messages = state.get("messages", [])
        
        # Search memory for context
        relevant_memories = await self.memory_manager.search_memory(user_input)
        memory_context = "; ".join(relevant_memories) if relevant_memories else ""
        
        # Spectra: Communication & Dialogue Facilitator
        system_content = (
            "あなたはSpectraです。コミュニケーション・説明・対話促進の専門家として、"
            "相手の話を聞き、理解し、分かりやすく説明し、議論を整理します。"
            "建設的で対話を促進する、分かりやすい回答をしてください。"
        )
        if memory_context:
            system_content += f"\n\n記憶: {memory_context}"
        
        # Generate response
        user_msg = HumanMessage(content=user_input)
        ai_response = await self.model_manager.generate_response(
            messages + [user_msg],
            agent_name=self.agent_name,
            memory_context=memory_context
        )
        
        self.logger.info(f"Spectra processed: {user_input[:30]}...")
        
        return {
            "messages": [user_msg, ai_response],
            "agent_reasoning": f"Spectra: 対話促進・説明として処理",
            "final_response": ai_response.content
        }


class LynQAgent:
    """LynQ - Logic analysis and structured thinking specialist"""
    
    def __init__(self, model_manager: ModelManager, memory_manager: MemoryManager):
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.agent_name = "lynq"
        self.logger = logging.getLogger("LynQAgent")
    
    async def process(self, state: MultiAgentState) -> dict:
        """Process analytical and logical tasks"""
        user_input = state.get("current_input", "")
        messages = state.get("messages", [])
        
        # Search memory for context
        relevant_memories = await self.memory_manager.search_memory(user_input)
        memory_context = "; ".join(relevant_memories) if relevant_memories else ""
        
        # LynQ: Logic Analysis & Structured Thinking
        system_content = (
            "あなたはLynQです。論理分析・構造化思考・概念明確化の専門家として、"
            "数学的計算、データ分析、プログラミング、科学的検証を担当します。"
            "論理的で構造化された、分析的な回答をしてください。"
        )
        if memory_context:
            system_content += f"\n\n記憶: {memory_context}"
        
        # Generate response
        user_msg = HumanMessage(content=user_input)
        ai_response = await self.model_manager.generate_response(
            messages + [user_msg],
            agent_name=self.agent_name,
            memory_context=memory_context
        )
        
        self.logger.info(f"LynQ analyzed: {user_input[:30]}...")
        
        return {
            "messages": [user_msg, ai_response],
            "agent_reasoning": f"LynQ: 論理分析・構造化思考として処理",
            "final_response": ai_response.content
        }


class PazAgent:
    """Paz - Creative thinking and possibility exploration specialist"""
    
    def __init__(self, model_manager: ModelManager, memory_manager: MemoryManager):
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.agent_name = "paz"
        self.logger = logging.getLogger("PazAgent")
    
    async def process(self, state: MultiAgentState) -> dict:
        """Process creative and exploratory tasks"""
        user_input = state.get("current_input", "")
        messages = state.get("messages", [])
        
        # Search memory for context
        relevant_memories = await self.memory_manager.search_memory(user_input)
        memory_context = "; ".join(relevant_memories) if relevant_memories else ""
        
        # Paz: Creative Thinking & Possibility Exploration
        system_content = (
            "あなたはPazです。創造的発想・可能性探索・発散思考の専門家として、"
            "革新的アイデア生成、デザイン思考、エンターテインメント企画を担当します。"
            "創造的で可能性を広げる、インスピレーションに富んだ回答をしてください。"
        )
        if memory_context:
            system_content += f"\n\n記憶: {memory_context}"
        
        # Generate response
        user_msg = HumanMessage(content=user_input)
        ai_response = await self.model_manager.generate_response(
            messages + [user_msg],
            agent_name=self.agent_name,
            memory_context=memory_context
        )
        
        self.logger.info(f"Paz created: {user_input[:30]}...")
        
        return {
            "messages": [user_msg, ai_response],
            "agent_reasoning": f"Paz: 創造的発想・可能性探索として処理",
            "final_response": ai_response.content
        }


# ===== Orchestrator =====

class MultiAgentOrchestrator:
    """LangGraph-based multi-agent orchestrator"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Initialize shared resources
        self.memory_manager = MemoryManager(config.thread_id)
        self.model_manager = ModelManager()
        self.obsidian_manager = ObsidianManager()
        self.mcp_manager = MCPManager()
        self.semantic_router = SemanticRouter(self.model_manager.get_embeddings())
        
        # Initialize agents
        self.spectra = SpectraAgent(self.model_manager, self.memory_manager)
        self.lynq = LynQAgent(self.model_manager, self.memory_manager)
        self.paz = PazAgent(self.model_manager, self.memory_manager)
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
        
        # Load Obsidian notes on startup
        self._load_obsidian_notes()
        
        # Initialize semantic routing vectors
        self._init_semantic_routing()
        
        self.logger.info("MultiAgentOrchestrator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup orchestrator logger"""
        logger = logging.getLogger("Orchestrator")
        
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler("bot.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s | ORCH    | %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def _load_obsidian_notes(self):
        """Load Obsidian notes at startup"""
        try:
            # Try to run async task
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create task in running loop
                asyncio.create_task(self._async_load_obsidian_notes())
            else:
                asyncio.run(self._async_load_obsidian_notes())
        except Exception as e:
            self.logger.error(f"Obsidian loading setup error: {e}")
    
    def _init_semantic_routing(self):
        """Initialize semantic routing vectors"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._async_init_semantic_routing())
            else:
                asyncio.run(self._async_init_semantic_routing())
        except Exception as e:
            self.logger.error(f"Semantic routing setup error: {e}")
    
    async def _async_init_semantic_routing(self):
        """Async semantic routing initialization"""
        try:
            success = await self.semantic_router.initialize_vectors()
            if success:
                info = self.semantic_router.get_vector_info()
                self.logger.info(f"semantic routing initialized: {info['count']} agents, {info['dimensions']} dims")
            else:
                self.logger.warning("semantic routing initialization failed, using fallback")
        except Exception as e:
            self.logger.error(f"semantic routing initialization error: {e}")
    
    async def _async_load_obsidian_notes(self):
        """Async Obsidian notes loading"""
        try:
            count = await self.obsidian_manager.load_notes_to_memory(self.memory_manager)
            self.logger.info(f"loaded {count} Obsidian notes")
        except Exception as e:
            self.logger.error(f"Obsidian loading error: {e}")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        builder = StateGraph(state_schema=MultiAgentState)
        
        # Add nodes
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("spectra", self._spectra_node)
        builder.add_node("lynq", self._lynq_node)
        builder.add_node("paz", self._paz_node)
        
        # Add edges
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self._route_to_agent,
            {
                "spectra": "spectra",
                "lynq": "lynq",
                "paz": "paz",
                "END": END
            }
        )
        builder.add_edge("spectra", END)
        builder.add_edge("lynq", END)
        builder.add_edge("paz", END)
        
        return builder.compile(
            checkpointer=self.memory_manager.get_memory_saver(),
            store=self.memory_manager.get_postgres_store()
        )
    
    async def _supervisor_node(self, state: MultiAgentState) -> dict:
        """Supervisor node for agent selection"""
        user_input = state.get("current_input", "")
        
        if not user_input:
            return {"active_agent": "paz"}  # Default to Paz
        
        # Intelligent agent routing based on input analysis
        selected_agent = await self._select_agent(user_input)
        
        self.logger.info(f"supervisor routed '{user_input[:30]}...' to {selected_agent}")
        
        return {
            "active_agent": selected_agent,
            "task_type": self._determine_task_type(selected_agent)
        }
    
    async def _select_agent(self, user_input: str) -> str:
        """Pure LLM-driven routing - optimized for clarity and efficiency"""
        
        llm_agent = self._enhanced_llm_routing(user_input)
        self.logger.info(f"LLM routing: '{user_input[:30]}...' → {llm_agent}")
        return llm_agent
    
    async def _semantic_routing(self, user_input: str) -> str:
        """Semantic vector similarity routing"""
        try:
            # Compute similarities
            similarities = await self.semantic_router.compute_similarity(user_input)
            if not similarities:
                return "uncertain"
            
            # Select best agent with confidence threshold (高品質セマンティック)
            best_agent, confidence = self.semantic_router.select_best_agent(similarities, threshold=0.75)
            
            # Check if confident enough (厳格なしきい値でセマンティック品質担保)
            if confidence > 0.90:
                self.logger.info(f"semantic routing: '{user_input[:30]}...' → {best_agent} (conf: {confidence:.3f})")
                return best_agent
            else:
                self.logger.info(f"semantic routing uncertain: max confidence {confidence:.3f} < 0.90, falling back to LLM")
                return "uncertain"
                
        except Exception as e:
            self.logger.error(f"semantic routing error: {e}")
            return "uncertain"
    
    def _keyword_based_routing(self, user_input: str) -> str:
        """Fast keyword-based routing"""
        input_lower = user_input.lower()
        
        # Strong indicators for each agent
        strong_communication = [
            "説明", "教え", "話", "まとめ", "整理", "伝え", 
            "どういうこと", "わかりやすく", "簡単に"
        ]
        
        strong_analysis = [
            "分析", "論理", "なぜ", "どうして", "原因", "理由",
            "比較", "検証", "構造", "関係", "正しい", "間違い",
            "数学", "計算", "統計", "証明", "解いて", "×", "*", "+", "-", "=", "?"
        ]
        
        strong_creativity = [
            "アイデア", "創造", "発想", "新しい", "面白い", "革新",
            "可能性", "想像", "ブレインストーミング", "ひらめき",
            "提案", "募集", "デザイン", "芸術", "インスピレーション"
        ]
        
        # Count strong indicators
        comm_count = sum(1 for kw in strong_communication if kw in input_lower)
        analysis_count = sum(1 for kw in strong_analysis if kw in input_lower)
        creativity_count = sum(1 for kw in strong_creativity if kw in input_lower)
        
        # Clear winner detection
        if analysis_count > 0 and analysis_count >= comm_count and analysis_count >= creativity_count:
            return "lynq"
        elif creativity_count > 0 and creativity_count >= comm_count and creativity_count >= analysis_count:
            return "paz"
        elif comm_count > 0:
            return "spectra"
        
        # No clear keywords found
        return "uncertain"
    
    def _enhanced_llm_routing(self, user_input: str) -> str:
        """Ultra-efficient LLM routing with clear role definitions"""
        try:
            routing_prompt = f"""入力: "{user_input}"

エージェント選択:
🔴 LynQ: 論理分析・構造化思考・概念明確化 (数学/統計/プログラミング/データ分析/技術問題/科学的検証)
🟡 Paz: 創造的発想・可能性探索・発散思考 (アイデア生成/デザイン/アート/エンタメ/革新/ブレスト)
🔵 Spectra: コミュニケーション・説明・対話促進 (説明/相談/調整/進行/議論整理/一般対応)

判定基準:
- 計算/分析/論理/技術/データ/統計/証明/検証/構造 → LynQ
- 創造/発想/アイデア/デザイン/アート/革新/可能性/ブレスト → Paz
- 説明/相談/対話/調整/進行/整理/一般質問 → Spectra

具体例:
"売上データを分析して" → LynQ (データ分析)
"A/Bテスト設計したい" → LynQ (統計的検証)
"戦略を論理的に構築" → LynQ (構造化思考)
"新しいアイデア募集" → Paz (創造的発想)
"ユーザー体験を向上" → Paz (可能性探索)
"ブランディング企画" → Paz (発散思考)
"状況をわかりやすく説明" → Spectra (説明・対話)
"チームとの調整方法" → Spectra (コミュニケーション)

回答: エージェント名のみ"""

            # Use default model for routing decision
            from langchain_core.messages import HumanMessage
            
            llm = self.model_manager.get_llm("default")
            response = llm.invoke([HumanMessage(content=routing_prompt)])
            
            # Extract agent name from response
            agent_name = response.content.strip().lower()
            
            # Clean up response (remove any extra text)
            for valid_agent in ["spectra", "lynq", "paz"]:
                if valid_agent in agent_name:
                    return valid_agent
            
            # If no valid agent found, return safe default
            self.logger.warning(f"Enhanced LLM routing failed to parse: {response.content}")
            return "spectra"  # Safe fallback
                
        except Exception as e:
            self.logger.error(f"Enhanced LLM routing error: {e}")
            return "spectra"  # Safe fallback
    
    def _is_ambiguous_query(self, user_input: str) -> bool:
        """Check if query needs multiple inference rounds"""
        ambiguous_patterns = [
            # 境界線キーワード
            "正しい", "やり方", "方法", "アプローチ", "解決", 
            # 短い・曖昧な表現
            "？？", "どう", "なに", "これ", "それ"
        ]
        
        input_lower = user_input.lower()
        return (
            len(user_input) < 10 or  # 短いクエリ
            any(pattern in input_lower for pattern in ambiguous_patterns) or
            user_input.count('?') > 1  # 複数の疑問符
        )
    
    def _multi_inference_routing(self, user_input: str) -> str:
        """Multiple inference routing for ambiguous cases"""
        try:
            # API制限を避けるため、現状は単一推論にフォールバック
            # 将来の拡張ポイント：3回推論して多数決
            return self._enhanced_llm_routing(user_input)
        except Exception as e:
            self.logger.error(f"Multi-inference routing error: {e}")
            return "spectra"
    
    def _llm_based_routing(self, user_input: str) -> str:
        """LLM-based context understanding for agent selection"""
        try:
            # Create routing prompt
            routing_prompt = f"""ユーザーの要求を分析し、最も適したAIエージェントを選んでください。

【入力】: "{user_input}"

【エージェント特性】:
🔵 spectra: 対話促進・説明・理解支援専門
   - 相手の立場で考え、分かりやすく説明する
   - 議論を整理し、コミュニケーションを円滑にする
   - 一般的な質問や相談に丁寧に対応

🔴 lynq: 論理分析・問題解決専門  
   - 複雑な問題を体系的に分解・分析する
   - データや事実に基づいて論理的に思考する
   - 数学、科学、技術的問題の解決

🟡 paz: 創造・発想・可能性探索専門
   - 既存の枠を超えた自由で斬新なアイデア
   - 想像力を刺激し、新しい視点を提供
   - エンターテインメントや芸術的表現

【判定指針】:
- 計算・分析・論理的思考が必要 → lynq
- 新しいアイデア・創造性・可能性探索 → paz
- 説明・対話・理解支援・一般相談 → spectra

エージェント名のみ回答: (spectra/lynq/paz)"""

            # Use default model for quick routing decision
            import asyncio
            from langchain_core.messages import HumanMessage
            
            # Simple synchronous call for routing
            llm = self.model_manager.get_llm("default")
            response = llm.invoke([HumanMessage(content=routing_prompt)])
            
            # Extract agent name from response
            agent_name = response.content.strip().lower()
            
            if agent_name in ["spectra", "lynq", "paz"]:
                self.logger.info(f"LLM routing: '{user_input[:20]}...' → {agent_name}")
                return agent_name
            else:
                self.logger.warning(f"LLM routing failed, using default: {response.content}")
                return "spectra"  # Safe fallback
                
        except Exception as e:
            self.logger.error(f"LLM routing error: {e}")
            return "spectra"  # Safe fallback
    
    def _determine_task_type(self, agent: str) -> str:
        """Determine task type based on selected agent"""
        task_map = {
            "spectra": "communication",
            "lynq": "analysis", 
            "paz": "creativity"
        }
        return task_map.get(agent, "communication")
    
    def _route_to_agent(self, state: MultiAgentState) -> str:
        """Route to selected agent"""
        active_agent = state.get("active_agent", "spectra")
        
        # Handle special cases
        user_input = state.get("current_input", "")
        
        # Check for MCP commands
        if self.mcp_manager.is_github_command(user_input):
            return "lynq"  # LynQ handles technical/search tasks
        
        # Check for memory commands
        if user_input.startswith('!'):
            return "spectra"  # Spectra handles memory management
            
        if user_input.lower() == 'memories':
            return "spectra"  # Spectra handles memory display
        
        return active_agent
    
    async def _spectra_node(self, state: MultiAgentState) -> dict:
        """Spectra agent node"""
        return await self.spectra.process(state)
    
    async def _lynq_node(self, state: MultiAgentState) -> dict:
        """LynQ agent node"""
        return await self.lynq.process(state)
    
    async def _paz_node(self, state: MultiAgentState) -> dict:
        """Paz agent node"""
        return await self.paz.process(state)
    
    async def chat(self, user_input: str) -> str:
        """Main chat interface"""
        config_dict = {"configurable": {"thread_id": self.memory_manager.thread_id}}
        
        try:
            # Handle special commands first
            if user_input.startswith('!'):
                memory_content = user_input[1:].strip()
                await self.memory_manager.save_memory(memory_content)
                return f"保存: {memory_content}"
            
            if user_input.lower() == 'memories':
                await self.memory_manager.show_memories()
                return "メモリ一覧を表示しました"
            
            # Handle MCP commands
            if self.mcp_manager.is_github_command(user_input):
                return await self.mcp_manager.handle_github_command(user_input)
            
            # Process through multi-agent workflow
            result = await self.graph.ainvoke(
                {"current_input": user_input},
                config=config_dict
            )
            
            # Return final response
            final_response = result.get("final_response", "応答を生成できませんでした")
            agent_reasoning = result.get("agent_reasoning", "")
            
            if config.enable_multi_agent and agent_reasoning:
                self.logger.info(f"agent reasoning: {agent_reasoning}")
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"chat error: {e}")
            return "申し訳ありません。処理中にエラーが発生しました。"
    
    def run(self):
        """Synchronous execution interface (CLI compatibility)"""
        print("🤖 Multi-Agent AI System (Spectra/LynQ/Paz)")
        print("長期メモリ: PostgreSQL永続化 + Obsidian自動統合")
        print("[Ctrl+C で終了, !テキスト で保存, memories で一覧]\n")
        self.logger.info(f"START | multi-agent system | thread={self.memory_manager.thread_id}")
        
        try:
            import readline
        except ImportError:
            pass
        
        while True:
            try:
                user_input = input("> ")
                
                if not user_input.strip():
                    continue
                
                # Exit commands
                if user_input.lower() in ['exit', 'quit', '終了', 'やめる']:
                    self.logger.info("EXIT | command")
                    break
                
                # Process chat
                response = asyncio.run(self.chat(user_input))
                print(f"{response}\n")
                
            except KeyboardInterrupt:
                print("\n")
                self.logger.info("EXIT | Ctrl+C")
                break
            except EOFError:
                self.logger.info("EXIT | EOF")
                break
            except Exception as e:
                self.logger.error(f"ERROR | {e}")
                print(f"ERROR: {e}\n")
    
    def get_system_stats(self) -> dict:
        """Get system statistics"""
        return {
            "orchestrator": "LangGraph Supervisor",
            "agents": ["Spectra", "LynQ", "Paz"],
            "memory_thread": self.memory_manager.thread_id,
            "multi_agent_enabled": config.enable_multi_agent,
            "models": self.model_manager.get_agent_info(),
            "obsidian": self.obsidian_manager.get_integration_stats(),
            "mcp": self.mcp_manager.get_mcp_stats(),
            "timestamp": datetime.now().isoformat()
        }