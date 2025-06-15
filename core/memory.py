"""
Memory management system for multi-agent LangGraph
Handles both short-term (MemorySaver) and long-term (PostgresStore) memory
"""
import logging
from typing import List, Any, Optional, Dict, Tuple
from datetime import datetime
import asyncio

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.postgres import PostgresStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import psycopg

from .config import config


class MemoryManager:
    """Unified memory management for all agents"""
    
    def __init__(self, thread_id: str = "main_conversation"):
        self.thread_id = thread_id
        self.logger = self._setup_logging()
        
        # Short-term memory (session-based conversation history)
        self.memory = MemorySaver()
        
        # Initialize embeddings for semantic search
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=config.google_api_key,
            task_type=config.embedding_task_type
        )
        
        # Initialize long-term memory (PostgresStore)
        self.store = self._init_postgres_store()
        
        self.logger.info(f"MemoryManager initialized for thread={self.thread_id}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup dedicated memory logger"""
        logger = logging.getLogger("MemoryManager")
        
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler("bot.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s | MEMORY  | %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def _init_postgres_store(self) -> PostgresStore:
        """Initialize PostgresStore with vector indexing"""
        try:
            # Direct connection approach (stable)
            conn = psycopg.connect(config.postgres_uri, autocommit=True)
            
            # Create basic store first
            basic_store = PostgresStore(conn)
            basic_store.setup()
            
            # Create vector-indexed store
            store = PostgresStore(
                conn,
                index={
                    "dims": config.embedding_dims,
                    "embed": self.embeddings,
                    "fields": ["content"]
                }
            )
            
            # Setup vector tables
            store.setup()
            
            self.logger.info("PostgresStore connection successful (semantic search enabled)")
            return store
            
        except Exception as e:
            self.logger.error(f"PostgresStore connection failed: {e}")
            raise RuntimeError(
                f"PostgreSQL connection failed: {e}\n\n"
                "Please start Docker:\ndocker compose up -d"
            )
    
    async def save_memory(self, content: str) -> None:
        """Save content to long-term memory"""
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        memory_data = {
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.store.aput(
            (self.thread_id, "memories"),
            memory_id,
            memory_data
        )
        
        self.logger.info(f"saved: {content[:30]}...")
    
    async def search_memory(self, query: str, limit: int = 2) -> List[str]:
        """Search long-term memory using enhanced keyword search"""
        try:
            memories = []
            
            # Extract key terms from query
            processed_query = self._extract_key_terms(query)
            
            # 1. Search user memories
            namespace = (self.thread_id, "memories")
            try:
                items = await self.store.asearch(namespace, query=query, limit=limit)
                memory_count = 0
                for item in items:
                    if hasattr(item, 'value') and 'content' in item.value:
                        memories.append(item.value['content'])
                        memory_count += 1
            except Exception as memory_error:
                self.logger.warning(f"memory search error: {memory_error}")
                memory_count = 0
            
            # 2. Search Obsidian notes
            keyword_items = await self._enhanced_keyword_search_obsidian(processed_query)
            
            obsidian_count = 0
            for item in keyword_items[:3]:  # Max 3 Obsidian results
                if hasattr(item, 'value') and 'content' in item.value:
                    memories.append(item.value['content'])
                    obsidian_count += 1
                    path = item.value.get('path', 'unknown')
                    self.logger.info(f"obsidian hit: {path}")
            
            self.logger.info(
                f"found {len(memories)} matches "
                f"(memory: {memory_count}, obsidian: {obsidian_count})"
            )
            
            return memories
            
        except Exception as e:
            self.logger.error(f"search error: {e}")
            return []
    
    def _extract_key_terms(self, query: str) -> str:
        """Extract important keywords from query"""
        import re
        
        stop_words = [
            'について', '教えて', 'とは', 'って', 'だ', 'である', 'です', 
            'の', 'を', 'が', 'に', 'は', 'で', 'と', 'から', 'まで', 
            'より', 'など', 'という', 'といった', 'になんて', 'なんて', 
            '書いてある', '内容', 'ファイル'
        ]
        
        query_clean = query.strip()
        
        # Extract quoted content (『』)
        quoted_matches = re.findall(r'『([^』]+)』', query_clean)
        if quoted_matches:
            extracted = quoted_matches[0]
            self.logger.info(f"found quoted term: '{extracted}' from '{query_clean}'")
            return extracted
        
        # Remove .md extension
        query_clean = re.sub(r'\\.md\\b', '', query_clean)
        
        # Extract filename patterns
        filename_match = re.search(r'([^/.]+)(?:\\.md)?(?:に|の|について|って|とは)', query_clean)
        if filename_match:
            extracted = filename_match.group(1).strip()
            self.logger.info(f"found filename pattern: '{extracted}' from '{query_clean}'")
            return extracted
        
        # Remove stop words
        for stop_word in stop_words:
            query_clean = query_clean.replace(stop_word, ' ')
        
        extracted = ' '.join(query_clean.split())
        
        # Use original query if too much was removed
        if len(extracted) < len(query) * 0.3:
            extracted = query
        
        self.logger.info(f"'{query}' -> '{extracted}'")
        return extracted
    
    async def _enhanced_keyword_search_obsidian(self, query: str) -> List[Any]:
        """Enhanced Obsidian notes keyword search with scoring"""
        try:
            obsidian_namespace = (self.thread_id, "obsidian_notes")
            
            # Get all Obsidian notes
            all_items = await self.store.asearch(obsidian_namespace, query="", limit=200)
            
            keyword_matches = []
            query_lower = query.lower()
            query_parts = [q.strip() for q in query_lower.split() if len(q.strip()) > 1]
            
            self.logger.info(f"searching for: '{query_lower}' in {len(all_items)} notes")
            
            for item in all_items:
                if hasattr(item, 'value') and 'content' in item.value:
                    content = item.value.get('content', '').lower()
                    full_content = item.value.get('full_content', '').lower()
                    path = item.value.get('path', '').lower()
                    
                    # Scoring system
                    score = 0
                    match_details = []
                    
                    # 1. Exact matches (high score)
                    if query_lower in path:
                        score += 100
                        match_details.append("path_exact")
                    if query_lower in content:
                        score += 80
                        match_details.append("content_exact")
                    if query_lower in full_content:
                        score += 60
                        match_details.append("full_content_exact")
                    
                    # 2. Partial keyword matches (medium score)
                    for part in query_parts:
                        if part in path:
                            score += 40
                            match_details.append(f"path_part:{part}")
                        if part in content:
                            score += 30
                            match_details.append(f"content_part:{part}")
                        if part in full_content:
                            score += 20
                            match_details.append(f"full_part:{part}")
                    
                    # 3. Normalized search (handle 『』 brackets)
                    normalized_query = query_lower.replace('『', '').replace('』', '')
                    if normalized_query != query_lower:
                        if (normalized_query in path or 
                            normalized_query in content or 
                            normalized_query in full_content):
                            score += 50
                            match_details.append("normalized")
                    
                    # Add to matches if score > 0
                    if score > 0:
                        class ScoredItem:
                            def __init__(self, original_item, score, match_details):
                                self.value = original_item.value
                                self.key = getattr(original_item, 'key', None)
                                self.search_score = score
                                self.match_details = match_details
                                # Copy other attributes
                                for attr in dir(original_item):
                                    if not attr.startswith('_') and attr not in ['value', 'key']:
                                        try:
                                            setattr(self, attr, getattr(original_item, attr))
                                        except:
                                            pass
                        
                        enhanced_item = ScoredItem(item, score, match_details)
                        keyword_matches.append(enhanced_item)
                        
                        # Debug info
                        note_name = item.value.get('path', 'unknown').split('/')[-1]
                        self.logger.info(
                            f"MATCH: {note_name} "
                            f"(score={score}, matches={','.join(match_details[:3])})"
                        )
            
            # Sort by score
            keyword_matches.sort(key=lambda x: getattr(x, 'search_score', 0), reverse=True)
            
            self.logger.info(f"searched {len(all_items)} notes, found {len(keyword_matches)} matches")
            
            return keyword_matches
            
        except Exception as e:
            self.logger.error(f"enhanced search error: {e}")
            return []
    
    async def show_memories(self) -> None:
        """Display long-term memory list"""
        try:
            namespace = (self.thread_id, "memories")
            
            items = await self.store.asearch(
                namespace,
                query="",
                limit=50
            )
            
            if not items:
                print("長期メモリは空です\n")
                return
            
            print(f"\n=== 長期メモリ ({len(items)}件) ===")
            for i, item in enumerate(items[:10], 1):  # Latest 10 items
                if hasattr(item, 'value'):
                    timestamp = item.value.get('timestamp', '')[:16]
                    content = item.value.get('content', '')[:50]
                    if len(item.value.get('content', '')) > 50:
                        content += "..."
                    print(f"{i}. {timestamp} | {content}")
            print("")
            
            self.logger.info(f"memories: {len(items)}")
            
        except Exception as e:
            print(f"メモリ表示エラー: {e}\n")
    
    def get_memory_saver(self) -> MemorySaver:
        """Get MemorySaver instance for LangGraph checkpointer"""
        return self.memory
    
    def get_postgres_store(self) -> PostgresStore:
        """Get PostgresStore instance for LangGraph store"""
        return self.store