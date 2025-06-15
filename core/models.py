"""
LLM and embedding models management for multi-agent system
Handles Gemini 2.0 Flash and text-embedding-004 models
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages

from .config import config


class ModelManager:
    """Centralized model management for all agents"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Validate API key
        config.validate_required_keys()
        
        # Initialize shared embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=config.google_api_key,
            task_type=config.embedding_task_type
        )
        
        # Initialize agent-specific LLM instances
        self._llm_instances = {}
        self._init_agent_llms()
        
        self.logger.info("ModelManager initialized with all agent LLMs")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup dedicated model logger"""
        logger = logging.getLogger("ModelManager")
        
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler("bot.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s | MODELS  | %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def _init_agent_llms(self):
        """Initialize LLM instances for each agent"""
        agents = {
            "spectra": "Spectra Communicator",
            "lynq": "LynQ Analyzer", 
            "paz": "Paz Creator",
            "default": "Default Agent"
        }
        
        for agent_key, agent_name in agents.items():
            agent_config = config.get_agent_config(agent_key)
            
            llm = ChatGoogleGenerativeAI(
                model=agent_config["model_name"],
                google_api_key=agent_config["google_api_key"],
                temperature=agent_config["temperature"],
                max_tokens=agent_config["max_tokens"],
                max_retries=3
            )
            
            self._llm_instances[agent_key] = llm
            
            self.logger.info(
                f"initialized {agent_name} LLM "
                f"(temp={agent_config['temperature']})"
            )
    
    def get_llm(self, agent_name: str = "default") -> ChatGoogleGenerativeAI:
        """Get LLM instance for specific agent"""
        agent_key = agent_name.lower()
        
        if agent_key not in self._llm_instances:
            self.logger.warning(f"Agent '{agent_name}' not found, using default")
            agent_key = "default"
        
        return self._llm_instances[agent_key]
    
    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Get shared embedding model instance"""
        return self.embeddings
    
    async def generate_response(
        self, 
        messages: list, 
        agent_name: str = "default",
        memory_context: str = ""
    ) -> AIMessage:
        """Generate AI response with memory context"""
        try:
            llm = self.get_llm(agent_name)
            
            # Add system message with memory context
            system_content = "簡潔で分かりやすく回答してください。"
            if memory_context:
                system_content += f"\n\n記憶: {memory_context}"
            
            system_msg = SystemMessage(content=system_content)
            
            # Prepare messages with system context
            all_messages = [system_msg] + messages
            
            # Trim messages to fit context window
            trimmed_messages = trim_messages(
                all_messages,
                max_tokens=2500,
                strategy="last",
                token_counter=len
            )
            
            self.logger.info(
                f"generating response for {agent_name} "
                f"(messages: {len(all_messages)} → {len(trimmed_messages)})"
            )
            
            # Generate response
            response = llm.invoke(trimmed_messages)
            ai_message = AIMessage(content=response.content)
            
            # Log response
            if hasattr(ai_message, 'content') and ai_message.content:
                content_preview = ai_message.content[:50] + "..." if len(ai_message.content) > 50 else ai_message.content
                self.logger.info(f"{agent_name} response: \"{content_preview}\"")
            
            return ai_message
            
        except Exception as e:
            self.logger.error(f"response generation error for {agent_name}: {e}")
            return AIMessage(content="申し訳ありません。AI応答でエラーが発生しました。")
    
    def get_agent_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured agents"""
        agent_info = {}
        
        for agent_key, llm in self._llm_instances.items():
            if agent_key == "default":
                continue
                
            agent_config = config.get_agent_config(agent_key)
            agent_info[agent_key] = {
                "name": agent_key.title(),
                "temperature": agent_config["temperature"],
                "model": agent_config["model_name"],
                "max_tokens": agent_config["max_tokens"]
            }
        
        return agent_info
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model usage statistics"""
        return {
            "llm_instances": len(self._llm_instances),
            "embedding_model": config.embedding_model,
            "embedding_dims": config.embedding_dims,
            "agents_configured": list(self._llm_instances.keys()),
            "initialization_time": datetime.now().isoformat()
        }