"""
LangGraphエージェントシステムの状態定義

マルチエージェントワークフローの共有状態を管理
"""

from typing import TypedDict, List, Dict, Optional, Literal, Annotated, Sequence
from typing_extensions import TypedDict
from datetime import datetime
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """マルチエージェントワークフローの共有状態"""
    
    # Discord関連情報
    channel_id: int
    user_id: int
    user_name: str
    message_id: str
    
    # 会話履歴（LangChain標準メッセージ）
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # 現在の処理状態
    current_agent: Optional[str]
    decision: Optional[str]
    
    # エージェント分析結果
    analysis: Optional[Dict]
    
    # ルーティング判定
    routing: Optional[Literal["spectra", "lynq", "paz", "multi", "end"]]
    
    # 協調モード情報
    collaboration_mode: Optional[bool]
    collaboration_agents: Optional[List[str]]
    
    # 文脈情報
    conversation_context: Optional[str]
    obsidian_context: Optional[List[Dict]]
    
    # エラー情報
    error: Optional[str]
    
    # メタデータ
    metadata: Optional[Dict]


class AnalysisResult(TypedDict):
    """Spectra分析結果"""
    intent: str
    entities: List[str]
    sentiment: str
    complexity: Literal["low", "medium", "high"]
    suggested_agent: str
    confidence: float
    reasoning: str


class AgentResponse(TypedDict):
    """エージェント応答構造"""
    agent_name: str
    content: str
    response_type: Literal["direct", "collaborative", "error"]
    metadata: Optional[Dict]