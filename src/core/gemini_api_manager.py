"""
Gemini API Manager

レート制限対応・エラーハンドリング・最適化されたAPI管理
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


@dataclass
class APICallRecord:
    """API呼び出し記録"""
    timestamp: datetime
    duration: float
    success: bool
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class RateLimitConfig:
    """レート制限設定"""
    max_calls_per_minute: int = 60
    max_calls_per_hour: int = 1000
    max_concurrent_calls: int = 1  # シーケンシャルアクセス
    retry_delay_base: float = 1.0
    max_retries: int = 3
    timeout_seconds: int = 30


class GeminiAPIManager:
    """Gemini API管理クラス - レート制限・エラーハンドリング対応"""
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gemini-2.0-flash",
        rate_limit_config: Optional[RateLimitConfig] = None
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.config = rate_limit_config or RateLimitConfig()
        self.logger = logging.getLogger(__name__)
        
        # API呼び出し履歴
        self.call_history: List[APICallRecord] = []
        
        # 同期制御
        self._api_lock = asyncio.Lock()
        self._active_calls = 0
        
        # 環境変数設定
        import os
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # LangSmith統合（オプション）
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            self.logger.info("LangSmith tracing enabled")
        
        # LLM初期化
        self.llm = init_chat_model(
            model_name,
            model_provider="google_genai",
            temperature=0.7,
            max_retries=self.config.max_retries
        )
        
        self.logger.info(f"GeminiAPIManager initialized: {model_name}")
    
    async def invoke_with_rate_limit(
        self, 
        messages: List[BaseMessage],
        context: Optional[str] = None
    ) -> AIMessage:
        """レート制限対応API呼び出し"""
        async with self._api_lock:
            # レート制限チェック
            await self._check_rate_limits()
            
            # 同時実行制限
            if self._active_calls >= self.config.max_concurrent_calls:
                self.logger.warning("Concurrent call limit reached, waiting...")
                await asyncio.sleep(self.config.retry_delay_base)
            
            self._active_calls += 1
            start_time = time.time()
            
            try:
                # リトライロジック付きAPI呼び出し
                result = await self._invoke_with_retry(messages)
                
                # 成功記録
                duration = time.time() - start_time
                self._record_api_call(duration, True)
                
                self.logger.debug(f"API call successful: {duration:.2f}s")
                return result
                
            except Exception as e:
                # エラー記録
                duration = time.time() - start_time
                self._record_api_call(duration, False, str(e))
                
                self.logger.error(f"API call failed: {e}")
                raise
                
            finally:
                self._active_calls -= 1
    
    async def _invoke_with_retry(self, messages: List[BaseMessage]) -> AIMessage:
        """リトライロジック付きAPI呼び出し"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # タイムアウト付きAPI呼び出し
                result = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=self.config.timeout_seconds
                )
                
                if isinstance(result, AIMessage):
                    return result
                else:
                    # BaseMessage から AIMessage に変換
                    return AIMessage(content=result.content)
                    
            except asyncio.TimeoutError:
                last_exception = Exception(f"API call timeout ({self.config.timeout_seconds}s)")
                self.logger.warning(f"API call timeout on attempt {attempt + 1}")
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
                
                # 指数バックオフ
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay_base * (2 ** attempt)
                    self.logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
        
        # 全リトライ失敗
        raise last_exception or Exception("API call failed after all retries")
    
    async def _check_rate_limits(self) -> None:
        """レート制限チェック"""
        now = datetime.now()
        
        # 古い記録を削除（1時間以上前）
        cutoff_time = now - timedelta(hours=1)
        self.call_history = [
            record for record in self.call_history 
            if record.timestamp > cutoff_time
        ]
        
        # 1分間の制限チェック
        minute_ago = now - timedelta(minutes=1)
        minute_calls = len([
            record for record in self.call_history 
            if record.timestamp > minute_ago and record.success
        ])
        
        if minute_calls >= self.config.max_calls_per_minute:
            wait_time = 60 - (now - minute_ago).total_seconds()
            self.logger.warning(f"Rate limit: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        # 1時間の制限チェック
        hour_calls = len([record for record in self.call_history if record.success])
        if hour_calls >= self.config.max_calls_per_hour:
            self.logger.error("Hourly rate limit exceeded")
            raise Exception("Hourly rate limit exceeded")
    
    def _record_api_call(
        self, 
        duration: float, 
        success: bool, 
        error: Optional[str] = None
    ) -> None:
        """API呼び出し記録"""
        record = APICallRecord(
            timestamp=datetime.now(),
            duration=duration,
            success=success,
            error=error
        )
        self.call_history.append(record)
        
        # 履歴サイズ制限（最新1000件）
        if len(self.call_history) > 1000:
            self.call_history = self.call_history[-1000:]
    
    def get_api_stats(self) -> Dict[str, Any]:
        """API使用統計取得"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        total_calls = len(self.call_history)
        successful_calls = len([r for r in self.call_history if r.success])
        failed_calls = total_calls - successful_calls
        
        minute_calls = len([
            r for r in self.call_history 
            if r.timestamp > minute_ago
        ])
        hour_calls = len([
            r for r in self.call_history 
            if r.timestamp > hour_ago
        ])
        
        avg_duration = 0.0
        if successful_calls > 0:
            avg_duration = sum(
                r.duration for r in self.call_history if r.success
            ) / successful_calls
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0.0,
            "calls_last_minute": minute_calls,
            "calls_last_hour": hour_calls,
            "average_duration": avg_duration,
            "active_calls": self._active_calls,
            "rate_limit_remaining_minute": max(0, self.config.max_calls_per_minute - minute_calls),
            "rate_limit_remaining_hour": max(0, self.config.max_calls_per_hour - hour_calls)
        }