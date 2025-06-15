"""Obsidian Repository Integration - 自動背景知識取得
"""
import os
import requests
from typing import List, Optional, Dict, Any, Tuple
import logging
from datetime import datetime
import hashlib


class ObsidianIntegration:
    """Obsidian GitHub リポジトリ専用統合クラス"""
    
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "LangGraph-CLI-Bot"
        }
        
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"
        
        # Obsidian リポジトリ設定（環境変数で変更可能）
        self.owner = os.getenv("OBSIDIAN_OWNER", "sito-sikino")
        self.repo = os.getenv("OBSIDIAN_REPO", "Obsidian")
        self.branch = os.getenv("OBSIDIAN_BRANCH", "main")
        
        self.logger = logging.getLogger("ObsidianIntegration")
        self.logger.setLevel(logging.INFO)
        
        # コンソールハンドラー追加（デバッグ用）
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def get_obsidian_notes(self) -> List[Dict[str, Any]]:
        """Obsidianノート一覧を取得（再帰的）"""
        try:
            self.logger.info(f"OBSIDIAN | accessing repository: {self.owner}/{self.repo}")
            all_notes = []
            self._collect_notes_recursive("", all_notes)
            
            self.logger.info(f"OBSIDIAN | found {len(all_notes)} notes")
            return all_notes
            
        except Exception as e:
            self.logger.error(f"OBSIDIAN | collection error: {e}")
            return []
    
    def _collect_notes_recursive(self, path: str, notes: List[Dict[str, Any]]):
        """再帰的にノートを収集"""
        try:
            contents = self.get_repository_contents(self.owner, self.repo, path)
            
            if not contents:
                self.logger.warning(f"OBSIDIAN | no contents in path: {path}")
                return
            
            self.logger.info(f"OBSIDIAN | scanning path '{path}' - found {len(contents)} items")
            
            for item in contents:
                if item["type"] == "file" and item["name"].endswith(".md"):
                    # Markdownファイル（ノート）を追加
                    notes.append({
                        "name": item["name"][:-3],  # .md拡張子を除去
                        "path": item["path"],
                        "size": item["size"]
                    })
                    self.logger.info(f"OBSIDIAN | found note: {item['name']}")
                elif item["type"] == "dir":
                    # ディレクトリを再帰探索
                    self._collect_notes_recursive(item["path"], notes)
                    
        except Exception as e:
            self.logger.error(f"OBSIDIAN | recursive scan error in {path}: {e}")
    
    def get_note_content_with_hash(self, note_path: str) -> Tuple[Optional[str], Optional[str]]:
        """ノート内容とハッシュを取得（重複チェック用）"""
        try:
            content = self.get_file_content(self.owner, self.repo, note_path)
            if content is None:
                return None, None
            
            # 内容のハッシュ値を計算（重複チェック用）
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
            
            return content, content_hash
            
        except Exception as e:
            self.logger.error(f"Note content error: {e}")
            return None, None
    
    def get_repository_last_updated(self) -> Optional[str]:
        """リポジトリの最終更新日時を取得"""
        try:
            url = f"{self.base_url}/repos/{self.owner}/{self.repo}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get("updated_at")
            
        except Exception as e:
            self.logger.error(f"Repository last updated error: {e}")
            return None
    
    def get_repository_contents(self, owner: str, repo: str, path: str = "") -> List[Dict[str, Any]]:
        """リポジトリファイル一覧取得"""
        try:
            # ブランチを明示的に指定
            url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}?ref={self.branch}"
            self.logger.info(f"OBSIDIAN | API call: {url}")
            
            response = requests.get(url, headers=self.headers)
            self.logger.info(f"OBSIDIAN | API response: {response.status_code}")
            
            if response.status_code == 404:
                self.logger.error(f"OBSIDIAN | Repository {owner}/{repo} not found or path '{path}' doesn't exist")
                return []
            
            response.raise_for_status()
            
            data = response.json()
            contents = []
            
            if isinstance(data, list):
                for item in data:
                    contents.append({
                        "name": item["name"],
                        "type": item["type"],  # file or dir
                        "size": item.get("size", 0),
                        "path": item["path"]
                    })
            else:
                self.logger.warning(f"OBSIDIAN | Unexpected API response format: {type(data)}")
            
            return contents
            
        except Exception as e:
            self.logger.error(f"OBSIDIAN | API error: {e}")
            return []
    
    def get_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """ファイル内容取得"""
        try:
            # ブランチを明示的に指定
            url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}?ref={self.branch}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("type") == "file" and data.get("content"):
                import base64
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content
            
            return None
            
        except Exception as e:
            self.logger.error(f"File content error: {e}")
            return None

