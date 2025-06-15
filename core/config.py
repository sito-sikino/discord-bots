"""
Configuration management for multi-agent LangGraph system
"""
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config(BaseModel):
    """Unified configuration for all agents and shared resources"""
    
    # LLM Settings
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Memory Configuration
    short_term_memory_size: int = 10
    memory_importance_threshold: float = 0.3
    long_term_memory_enabled: bool = True
    
    # PostgreSQL Configuration
    postgres_uri: str = Field(
        default_factory=lambda: os.getenv(
            "POSTGRES_URI", 
            "postgresql://postgres:postgres@localhost:5432/langraph_memory"
        )
    )
    
    # Embedding Model Configuration
    embedding_model: str = "models/text-embedding-004"
    embedding_dims: int = 768
    embedding_task_type: str = "retrieval_document"
    
    # Obsidian Integration
    github_token: str = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    obsidian_owner: str = Field(default_factory=lambda: os.getenv("OBSIDIAN_OWNER", "sito-sikino"))
    obsidian_repo: str = Field(default_factory=lambda: os.getenv("OBSIDIAN_REPO", "Obsidian"))
    obsidian_branch: str = Field(default_factory=lambda: os.getenv("OBSIDIAN_BRANCH", "main"))
    force_reload_obsidian: bool = Field(
        default_factory=lambda: os.getenv("FORCE_RELOAD_OBSIDIAN", "false").lower() == "true"
    )
    
    # Multi-Agent Configuration
    enable_multi_agent: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_MULTI_AGENT", "false").lower() == "true"
    )
    thread_id: str = "main_conversation"
    
    # Agent-specific Settings
    spectra_temperature: float = 0.8  # Communicator - slightly more creative
    lynq_temperature: float = 0.3     # Analyzer - more focused
    paz_temperature: float = 0.9      # Creator - most creative
    
    # GitHub MCP Integration
    github_enabled: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def validate_required_keys(self) -> bool:
        """Validate that required API keys are present"""
        if not self.google_api_key:
            raise ValueError("Google APIキーが設定されていません (GOOGLE_API_KEY)")
        return True
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get agent-specific configuration"""
        base_config = {
            "google_api_key": self.google_api_key,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens
        }
        
        temperature_map = {
            "spectra": self.spectra_temperature,
            "lynq": self.lynq_temperature,
            "paz": self.paz_temperature
        }
        
        base_config["temperature"] = temperature_map.get(agent_name.lower(), self.temperature)
        return base_config


# Singleton instance
config = Config()