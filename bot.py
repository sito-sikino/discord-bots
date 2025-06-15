"""LangGraph CLIチャットボット（PostgreSQL永続メモリ + セマンティック検索）
"""
from typing import TypedDict, Annotated, List, Any
import logging
from datetime import datetime

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.postgres import PostgresStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from github_mcp import ObsidianIntegration


# ===== 状態定義 =====

class ChatState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    current_input: str


# ===== AIボット =====

class Bot:
    
    def __init__(self, config):
        if not config.google_api_key:
            raise ValueError("Google APIキーが設定されていません")
        
        # ログ設定
        self._setup_logging()
        
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=3
        )
        
        # Google Embeddings（セマンティック検索用）
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=config.google_api_key,
            task_type="retrieval_document"  # ドキュメント検索最適化
        )
        
        # 短期メモリ（MemorySaver）- スレッド単位の会話履歴
        self.memory = MemorySaver()
        self.thread_id = "main_conversation"
        
        # Obsidian統合
        self.obsidian = ObsidianIntegration()
        
        # 長期メモリ（PostgresStore + セマンティック検索）
        import os
        
        # PostgreSQL接続
        db_uri = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/langraph_memory")
        
        try:
            # PostgresStore初期化（ベクトルインデックス対応）
            import psycopg
            
            # 直接コネクション方式（安定）
            conn = psycopg.connect(db_uri, autocommit=True)
            
            # まず基本テーブルを作成
            basic_store = PostgresStore(conn)
            basic_store.setup()
            
            # 次にベクトルインデックス付きストアを作成
            self.store = PostgresStore(
                conn,
                index={
                    "dims": 768,  # text-embedding-004の次元数
                    "embed": self.embeddings,
                    "fields": ["content"]  # contentフィールドを埋め込み対象
                }
            )
            
            # ベクトルインデックス付きテーブルをセットアップ
            self.store.setup()
            
            self.logger.info("PostgresStore接続成功（セマンティック検索対応）")
        except Exception as e:
            self.logger.error(f"PostgresStore接続失敗: {e}")
            raise RuntimeError(f"PostgreSQL接続に失敗しました: {e}\n\nDockerを起動してください:\ndocker compose up -d")
        
        self.graph = self._build_graph()
        
        # Obsidianノート自動読み込み
        self._load_obsidian_notes()
        
        self.logger.info(f"INIT    | thread={self.thread_id} (store=PostgresStore)")
    
    def _setup_logging(self):
        self.logger = logging.getLogger("Bot")
        
        # 重複ハンドラー防止
        if self.logger.handlers:
            return
            
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler("bot.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        # パイプ區切りフォーマット
        formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def _load_obsidian_notes(self):
        """Obsidianノートを長期メモリに自動読み込み"""
        import asyncio
        asyncio.run(self._async_load_obsidian_notes())
    
    async def _async_load_obsidian_notes(self):
        """Obsidianノート非同期読み込み"""
        try:
            # リポジトリの最終更新日時をチェック
            last_updated = self.obsidian.get_repository_last_updated()
            if not last_updated:
                self.logger.warning("OBSIDIAN | repository update check failed")
                return
            
            # 強制リロード設定確認
            import os
            force_reload = os.getenv("FORCE_RELOAD_OBSIDIAN", "false").lower() == "true"
            
            # namespaceを事前に定義
            namespace = (self.thread_id, "obsidian_meta")
            
            if not force_reload:
                # 既存の更新チェック
                existing_update = await self.store.aget(namespace, "last_updated")
                
                # existing_updateがItemオブジェクトの場合は.valueでアクセス
                if existing_update and hasattr(existing_update, 'value') and existing_update.value.get("value") == last_updated:
                    self.logger.info("OBSIDIAN | notes already up-to-date")
                    return
            else:
                self.logger.info("OBSIDIAN | force reload enabled - reloading all notes")
            
            # ノート一覧を取得
            notes = self.obsidian.get_obsidian_notes()
            if not notes:
                self.logger.warning("OBSIDIAN | no notes found")
                return
            
            # 各ノートを長期メモリに保存（重複チェック付き）
            loaded_count = 0
            import time
            
            for i, note in enumerate(notes):  # 全ノート処理
                # API制限対策（適度な間隔）
                if i > 0 and i % 10 == 0:
                    time.sleep(0.1)  # 10件ごとに100ms休憩
                    self.logger.info(f"OBSIDIAN | processed {i+1}/{len(notes)} notes")
                
                content, content_hash = self.obsidian.get_note_content_with_hash(note["path"])
                
                # デバッグログ追加
                if i < 3:  # 最初の3件のみ詳細ログ
                    self.logger.info(f"OBSIDIAN | note {i+1}: {note['name']}")
                    self.logger.info(f"OBSIDIAN |   path: {note['path']}")
                    self.logger.info(f"OBSIDIAN |   content_length: {len(content) if content else 0}")
                    self.logger.info(f"OBSIDIAN |   has_hash: {bool(content_hash)}")
                
                if content and content_hash:
                    # 強制リロード時は重複チェックをスキップ
                    if force_reload:
                        should_save = True
                        if i < 3:
                            self.logger.info(f"OBSIDIAN |   force_reload: skipping duplicate check")
                    else:
                        # 既存ハッシュチェック
                        note_namespace = (self.thread_id, "obsidian_notes")
                        existing_note = await self.store.aget(note_namespace, note["name"])
                        
                        # existing_noteがItemオブジェクトの場合は.valueでアクセス
                        existing_hash = None
                        if existing_note and hasattr(existing_note, 'value'):
                            existing_hash = existing_note.value.get("hash")
                        elif existing_note:
                            existing_hash = existing_note.get("hash")
                        
                        # デバッグ情報（最初の3件）
                        if i < 3:
                            self.logger.info(f"OBSIDIAN |   existing_note: {bool(existing_note)}")
                            self.logger.info(f"OBSIDIAN |   existing_hash: {existing_hash}")
                            self.logger.info(f"OBSIDIAN |   new_hash: {content_hash}")
                            self.logger.info(f"OBSIDIAN |   hash_different: {existing_hash != content_hash}")
                        
                        should_save = not existing_note or existing_hash != content_hash
                    
                    if should_save:
                        # 新しいまたは更新されたノート（フィルタリング追加）
                        content_length = len(content.strip()) if content else 0
                        
                        # 詳細デバッグログ（全ノート対象）
                        self.logger.info(f"OBSIDIAN | [{i+1}/{len(notes)}] {note['name']}")
                        self.logger.info(f"OBSIDIAN |   content_length: {content_length}")
                        self.logger.info(f"OBSIDIAN |   has_content: {bool(content)}")
                        self.logger.info(f"OBSIDIAN |   passes_length_filter: {content_length > 10}")
                        
                        if content and content_length > 10:  # 最小長チェック
                            try:
                                self.logger.info(f"OBSIDIAN |   preparing save data for: {note['name']}")
                                note_data = {
                                    "content": f"Obsidianノート「{note['name']}」: {content[:500]}{'...' if len(content) > 500 else ''}",
                                    "hash": content_hash,
                                    "path": note["path"],
                                    "full_content": content,
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                self.logger.info(f"OBSIDIAN |   calling store.aput for: {note['name']}")
                                note_namespace = (self.thread_id, "obsidian_notes")
                                await self.store.aput(note_namespace, note["name"], note_data)
                                loaded_count += 1
                                
                                self.logger.info(f"OBSIDIAN |   ✓ SAVED: {note['name']} (total: {loaded_count})")
                            except Exception as save_error:
                                self.logger.error(f"OBSIDIAN |   ✗ SAVE_ERROR: {note['name']} - {save_error}")
                                self.logger.error(f"OBSIDIAN |   continuing with next note...")
                        else:
                            self.logger.info(f"OBSIDIAN |   ✗ FILTERED: {note['name']} (content_length: {content_length})")
                    else:
                        self.logger.info(f"OBSIDIAN |   ✗ SKIPPED: {note['name']} (existing/duplicate)")
            
            # 更新日時を記録
            await self.store.aput(namespace, "last_updated", {"value": last_updated})
            
            self.logger.info(f"OBSIDIAN | loaded {loaded_count} notes (total: {len(notes)})")
            
            # 取得したノート名をログに出力（最初の10件）
            if loaded_count > 0:
                self.logger.info("OBSIDIAN | loaded notes sample:")
                sample_count = 0
                for i, note in enumerate(notes):
                    if sample_count >= 10:
                        break
                    content, _ = self.obsidian.get_note_content_with_hash(note["path"])
                    if content and len(content.strip()) > 10:
                        self.logger.info(f"OBSIDIAN |   - {note['name']}.md")
                        sample_count += 1
                
            
        except Exception as e:
            self.logger.error(f"OBSIDIAN | loading error: {e}")
    
    def _build_graph(self):
        builder = StateGraph(state_schema=ChatState)
        builder.add_node("chat", self._chat_node)
        builder.add_edge(START, "chat")
        builder.add_edge("chat", END)
        return builder.compile(checkpointer=self.memory, store=self.store)
    
    async def _chat_node(self, state: ChatState) -> dict:
        user_input = state.get("current_input", "")
        if not user_input:
            return {"messages": []}
        
        messages = state.get("messages", [])
        self.logger.info(f'INPUT   | "{user_input}" (hist={len(messages)})')
        
        try:
            # 長期メモリ検索（セマンティック検索）
            relevant_memories = await self.search_memory(user_input)
            memory_context = f"\n\n記憶: {'; '.join(relevant_memories)}" if relevant_memories else ""
            
            system_content = "簡潔で分かりやすく回答してください。" + memory_context
            system_msg = SystemMessage(content=system_content)
            user_msg = HumanMessage(content=user_input)
            
            all_messages = [system_msg] + messages + [user_msg]
            trimmed_messages = trim_messages(
                all_messages,
                max_tokens=2500,
                strategy="last",
                token_counter=len
            )
            
            self.logger.info(f"TRIM    | messages: {len(all_messages)} → {len(trimmed_messages)}")
            
            # Event loop問題回避：同期的な呼び出しに変更
            try:
                # 非同期から同期に変更
                response = self.llm.invoke(trimmed_messages)
                ai_message = AIMessage(content=response.content)
            except Exception as llm_error:
                self.logger.error(f"CHAT    | LLM error: {llm_error}")
                ai_message = AIMessage(content="申し訳ありません。AI応答でエラーが発生しました。")
            
            # 応答内容をログ出力
            if hasattr(ai_message, 'content') and ai_message.content:
                self.logger.info(f'OUTPUT  | "{ai_message.content[:50]}..."')
            else:
                self.logger.info('OUTPUT  | (empty response)')
            
            return {"messages": [user_msg, ai_message]}
            
        except Exception as e:
            self.logger.error(f"CHAT    | error: {e}")
            error_msg = AIMessage(content="申し訳ありません。回答の生成中にエラーが発生しました。")
            return {"messages": [HumanMessage(content=user_input), error_msg]}
    
    async def chat(self, user_input: str) -> str:
        config = {"configurable": {"thread_id": self.thread_id}}
        
        
        result = await self.graph.ainvoke(
            {"current_input": user_input},
            config=config
        )
        
        messages = result.get("messages", [])
        if messages and hasattr(messages[-1], 'content'):
            return messages[-1].content
        
        self.logger.warning("WARN    | No response generated")
        return "応答を生成できませんでした"
    
    async def save_memory(self, content: str):
        """長期メモリ保存（PostgresStore）"""
        from datetime import datetime
        
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        memory_data = {
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # PostgresStoreに保存
        await self.store.aput(
            (self.thread_id, "memories"),
            memory_id,
            memory_data
        )
        
        self.logger.info(f"MEMORY  | saved: {content[:30]}...")
    
    async def search_memory(self, query: str) -> List[str]:
        """強化キーワード検索（セマンティック検索代替）"""
        try:
            memories = []
            
            # クエリの前処理：重要なキーワードを抽出
            processed_query = self._extract_key_terms(query)
            
            # 1. 通常の長期メモリ検索
            namespace = (self.thread_id, "memories")
            try:
                items = await self.store.asearch(
                    namespace,
                    query=query,
                    limit=2
                )
                memory_count = 0
                for item in items:
                    if hasattr(item, 'value') and 'content' in item.value:
                        memories.append(item.value['content'])
                        memory_count += 1
            except Exception as memory_error:
                self.logger.warning(f"SEARCH  | memory search error: {memory_error}")
                memory_count = 0
            
            # 2. Obsidianノート検索（強化キーワード検索）
            keyword_items = await self._enhanced_keyword_search_obsidian(processed_query)
            
            # 結果をメモリに追加
            obsidian_count = 0
            for item in keyword_items[:3]:  # 最大3件まで
                if hasattr(item, 'value') and 'content' in item.value:
                    memories.append(item.value['content'])
                    obsidian_count += 1
                    # デバッグ: どのObsidianノートがヒットしたか
                    path = item.value.get('path', 'unknown')
                    self.logger.info(f"SEARCH  | obsidian hit: {path}")
            
            self.logger.info(f"SEARCH  | found {len(memories)} matches (memory: {memory_count}, obsidian: {obsidian_count})")
            self.logger.info(f"SEARCH  | keyword: {len(keyword_items)} notes matched")
            
            return memories
        except Exception as e:
            self.logger.error(f"SEARCH  | error: {e}")
            return []
    
    def _extract_key_terms(self, query: str) -> str:
        """クエリから重要なキーワードを抽出"""
        import re
        
        # 一般的な助詞・動詞を除去してキーワードを抽出
        stop_words = ['について', '教えて', 'とは', 'って', 'だ', 'である', 'です', 'の', 'を', 'が', 'に', 'は', 'で', 'と', 'から', 'まで', 'より', 'など', 'という', 'といった', 'になんて', 'なんて', '書いてある', '内容', 'ファイル']
        
        # 原文をそのまま返すか、重要部分を抽出するかを判断
        query_clean = query.strip()
        
        # 『』で囲まれた部分を優先的に抽出
        quoted_matches = re.findall(r'『([^』]+)』', query_clean)
        if quoted_matches:
            extracted = quoted_matches[0]
            self.logger.info(f"EXTRACT | found quoted term: '{extracted}' from '{query_clean}'")
            return extracted
        
        # .md拡張子を除去
        query_clean = re.sub(r'\.md\b', '', query_clean)
        
        # ファイル名パターンを抽出（例：「逮捕しちゃうぞ.mdになんて書いてある？」→「逮捕しちゃうぞ」）
        filename_match = re.search(r'([^/.]+)(?:\.md)?(?:に|の|について|って|とは)', query_clean)
        if filename_match:
            extracted = filename_match.group(1).strip()
            self.logger.info(f"EXTRACT | found filename pattern: '{extracted}' from '{query_clean}'")
            return extracted
        
        # 助詞を除去して主要な単語を抽出
        for stop_word in stop_words:
            query_clean = query_clean.replace(stop_word, ' ')
        
        # 複数の空白を1つにし、前後の空白を除去
        extracted = ' '.join(query_clean.split())
        
        # 元のクエリと大きく異なる場合は元のクエリを使用
        if len(extracted) < len(query) * 0.3:
            extracted = query
        
        self.logger.info(f"EXTRACT | '{query}' -> '{extracted}'")
        return extracted
    
    async def _keyword_search_obsidian(self, query: str) -> List[Any]:
        """Obsidianノートのキーワード検索"""
        try:
            obsidian_namespace = (self.thread_id, "obsidian_notes")
            
            # 全Obsidianノートを取得（全102ノート対応）
            all_items = await self.store.asearch(obsidian_namespace, query="", limit=200)
            
            keyword_matches = []
            query_lower = query.lower()
            
            for item in all_items:
                if hasattr(item, 'value') and 'content' in item.value:
                    # ノート名または内容にキーワードが含まれているかチェック
                    content = item.value.get('content', '').lower()
                    full_content = item.value.get('full_content', '').lower()
                    path = item.value.get('path', '').lower()
                    
                    # 正規化した名前でも検索（『』の有無に対応）
                    normalized_query = query_lower.replace('『', '').replace('』', '')
                    
                    if (query_lower in content or 
                        query_lower in full_content or 
                        query_lower in path or
                        normalized_query in content or
                        normalized_query in full_content or
                        normalized_query in path):
                        keyword_matches.append(item)
                        
                        # デバッグ: キーワードマッチの詳細
                        match_location = []
                        if query_lower in path:
                            match_location.append("path")
                        if query_lower in content:
                            match_location.append("content")
                        if query_lower in full_content:
                            match_location.append("full_content")
                        
                        self.logger.info(f"KEYWORD | match in {'/'.join(match_location)}: {item.value.get('path', 'unknown')}")
            
            self.logger.info(f"KEYWORD | searched {len(all_items)} notes, found {len(keyword_matches)} matches")
            
            # デバッグ: 逮捕しちゃうぞノートが存在するか確認
            if "逮捕" in query_lower:
                found_taiho = False
                for item in all_items:
                    if hasattr(item, 'value') and 'path' in item.value:
                        path = item.value.get('path', '')
                        if '逮捕' in path:
                            self.logger.info(f"KEYWORD | DEBUG: found 逮捕 note at: {path}")
                            found_taiho = True
                if not found_taiho:
                    self.logger.info(f"KEYWORD | DEBUG: 逮捕 note NOT found in {len(all_items)} items")
            
            return keyword_matches
            
        except Exception as e:
            self.logger.error(f"KEYWORD | search error: {e}")
            return []
    
    async def _enhanced_keyword_search_obsidian(self, query: str) -> List[Any]:
        """強化Obsidianノートキーワード検索"""
        try:
            obsidian_namespace = (self.thread_id, "obsidian_notes")
            
            # 全Obsidianノートを取得
            all_items = await self.store.asearch(obsidian_namespace, query="", limit=200)
            
            keyword_matches = []
            query_lower = query.lower()
            
            # 検索キーワードを複数に分割
            query_parts = [q.strip() for q in query_lower.split() if len(q.strip()) > 1]
            
            self.logger.info(f"ENHANCED | searching for: '{query_lower}' in {len(all_items)} notes")
            
            for item in all_items:
                if hasattr(item, 'value') and 'content' in item.value:
                    content = item.value.get('content', '').lower()
                    full_content = item.value.get('full_content', '').lower()
                    path = item.value.get('path', '').lower()
                    
                    # スコアリングシステム
                    score = 0
                    match_details = []
                    
                    # 1. 完全一致検索（高スコア）
                    if query_lower in path:
                        score += 100
                        match_details.append("path_exact")
                    if query_lower in content:
                        score += 80
                        match_details.append("content_exact")
                    if query_lower in full_content:
                        score += 60
                        match_details.append("full_content_exact")
                    
                    # 2. 部分キーワード検索（中スコア）
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
                    
                    # 3. 正規化検索（『』括弧の処理）
                    normalized_query = query_lower.replace('『', '').replace('』', '')
                    if normalized_query != query_lower:
                        if normalized_query in path or normalized_query in content or normalized_query in full_content:
                            score += 50
                            match_details.append("normalized")
                    
                    # スコアが閾値を超えた場合にマッチとして追加
                    if score > 0:
                        # スコア情報を含む新しいオブジェクトを作成
                        class ScoredItem:
                            def __init__(self, original_item, score, match_details):
                                self.value = original_item.value
                                self.key = original_item.key if hasattr(original_item, 'key') else None
                                self.search_score = score
                                self.match_details = match_details
                                # 元のアイテムの他の属性もコピー
                                for attr in dir(original_item):
                                    if not attr.startswith('_') and attr not in ['value', 'key']:
                                        try:
                                            setattr(self, attr, getattr(original_item, attr))
                                        except:
                                            pass
                        
                        enhanced_item = ScoredItem(item, score, match_details)
                        keyword_matches.append(enhanced_item)
                        
                        # デバッグ情報
                        note_name = item.value.get('path', 'unknown').split('/')[-1]
                        self.logger.info(f"ENHANCED | MATCH: {note_name} (score={score}, matches={','.join(match_details[:3])})")
            
            # スコア順にソート（search_scoreが設定されているもののみ）
            def get_score(item):
                return getattr(item, 'search_score', 0)
            
            keyword_matches.sort(key=get_score, reverse=True)
            
            self.logger.info(f"ENHANCED | searched {len(all_items)} notes, found {len(keyword_matches)} matches")
            
            # デバッグ：最初の逮捕ノートが見つかるかチェック
            if "逮捕" in query_lower:
                for item in all_items[:5]:  # 最初の5件をチェック
                    if hasattr(item, 'value') and 'path' in item.value:
                        path = item.value.get('path', '')
                        if '逮捕' in path:
                            self.logger.info(f"ENHANCED | DEBUG: found 逮捕 note: {path}")
                            self.logger.info(f"ENHANCED | DEBUG: query_lower='{query_lower}', path.lower()='{path.lower()}'")
                            self.logger.info(f"ENHANCED | DEBUG: match test: {query_lower in path.lower()}")
                            break
            
            return keyword_matches
            
        except Exception as e:
            self.logger.error(f"ENHANCED | search error: {e}")
            return []
    
    async def show_memories(self):
        """長期メモリ一覧表示（PostgresStore）"""
        try:
            # 名前空間から全メモリ取得
            namespace = (self.thread_id, "memories")
            
            # PostgresStoreの正しいasearchメソッドで全件取得
            items = await self.store.asearch(
                namespace, 
                query="",  # 空のクエリで全件取得
                limit=50   # 最大50件
            )
            
            if not items:
                print("長期メモリは空です\n")
                return
            
            print(f"\n=== 長期メモリ ({len(items)}件) ===")
            for i, item in enumerate(items[:10], 1):  # 最新10件
                if hasattr(item, 'value'):
                    timestamp = item.value.get('timestamp', '')[:16]
                    content = item.value.get('content', '')[:50]
                    if len(item.value.get('content', '')) > 50:
                        content += "..."
                    print(f"{i}. {timestamp} | {content}")
            print("")
            
            self.logger.info(f"SHOW    | memories: {len(items)}")
        except Exception as e:
            print(f"メモリ表示エラー: {e}\n")
    
    
    
    def run(self):
        """同期実行でCtrl+C問題を回避"""
        print("AI BOT (長期メモリ: PostgreSQL永続化 + Obsidian自動統合)")
        print("[Ctrl+C で終了, !テキスト で保存, memories で一覧]\n")
        self.logger.info(f"START   | thread={self.thread_id}")
        
        try:
            import readline
        except ImportError:
            pass
        
        import asyncio
        
        while True:
            try:
                user_input = input("> ")
                
                if not user_input.strip():
                    continue
                
                # 終了コマンド
                if user_input.lower() in ['exit', 'quit', '終了', 'やめる']:
                    self.logger.info("EXIT    | command")
                    break
                
                # 長期メモリ保存
                if user_input.startswith('!'):
                    memory_content = user_input[1:].strip()
                    asyncio.run(self.save_memory(memory_content))
                    print(f"保存: {memory_content}\n")
                    continue
                
                # 長期メモリ一覧
                if user_input.lower() == 'memories':
                    asyncio.run(self.show_memories())
                    continue
                
                
                # 通常会話
                response = asyncio.run(self.chat(user_input))
                print(f"{response}\n")
                
            except KeyboardInterrupt:
                print("\n")
                self.logger.info("EXIT    | Ctrl+C")
                break
            except EOFError:
                self.logger.info("EXIT    | EOF")
                break
            except Exception as e:
                self.logger.error(f"ERROR   | {e}")
                print(f"ERROR: {e}\n")