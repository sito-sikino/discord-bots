"""
Obsidian repository integration for multi-agent system
Handles automatic background knowledge acquisition from sito-sikino/Obsidian
"""
import os
import logging
import requests
import hashlib
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from .config import config


class ObsidianManager:
    """Centralized Obsidian GitHub repository integration"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # GitHub API configuration
        self.github_token = config.github_token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "LangGraph-MultiAgent-Bot"
        }
        
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"
        
        # Repository configuration
        self.owner = config.obsidian_owner
        self.repo = config.obsidian_repo
        self.branch = config.obsidian_branch
        
        self.logger.info(f"ObsidianManager initialized for {self.owner}/{self.repo}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup dedicated Obsidian logger"""
        logger = logging.getLogger("ObsidianManager")
        
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler("bot.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s | OBSIDIAN| %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def get_repository_last_updated(self) -> Optional[str]:
        """Get repository last updated timestamp"""
        try:
            url = f"{self.base_url}/repos/{self.owner}/{self.repo}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            last_updated = data.get("updated_at")
            
            self.logger.info(f"repository last updated: {last_updated}")
            return last_updated
            
        except Exception as e:
            self.logger.error(f"repository update check error: {e}")
            return None
    
    def get_obsidian_notes(self) -> List[Dict[str, Any]]:
        """Get all Obsidian notes recursively"""
        try:
            self.logger.info(f"accessing repository: {self.owner}/{self.repo}")
            all_notes = []
            self._collect_notes_recursive("", all_notes)
            
            self.logger.info(f"found {len(all_notes)} notes")
            return all_notes
            
        except Exception as e:
            self.logger.error(f"collection error: {e}")
            return []
    
    def _collect_notes_recursive(self, path: str, notes: List[Dict[str, Any]]):
        """Recursively collect notes from repository"""
        try:
            contents = self.get_repository_contents(self.owner, self.repo, path)
            
            if not contents:
                self.logger.warning(f"no contents in path: {path}")
                return
            
            self.logger.info(f"scanning path '{path}' - found {len(contents)} items")
            
            for item in contents:
                if item["type"] == "file" and item["name"].endswith(".md"):
                    # Markdown file (note)
                    notes.append({
                        "name": item["name"][:-3],  # Remove .md extension
                        "path": item["path"],
                        "size": item["size"]
                    })
                    self.logger.info(f"found note: {item['name']}")
                elif item["type"] == "dir":
                    # Directory - recurse
                    self._collect_notes_recursive(item["path"], notes)
                    
        except Exception as e:
            self.logger.error(f"recursive scan error in {path}: {e}")
    
    def get_repository_contents(self, owner: str, repo: str, path: str = "") -> List[Dict[str, Any]]:
        """Get repository file listing"""
        try:
            # Explicitly specify branch
            url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}?ref={self.branch}"
            self.logger.info(f"API call: {url}")
            
            response = requests.get(url, headers=self.headers)
            self.logger.info(f"API response: {response.status_code}")
            
            if response.status_code == 404:
                self.logger.error(f"Repository {owner}/{repo} not found or path '{path}' doesn't exist")
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
                self.logger.warning(f"Unexpected API response format: {type(data)}")
            
            return contents
            
        except Exception as e:
            self.logger.error(f"API error: {e}")
            return []
    
    def get_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """Get file content from repository"""
        try:
            # Explicitly specify branch
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
            self.logger.error(f"file content error: {e}")
            return None
    
    def get_note_content_with_hash(self, note_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Get note content and hash for duplicate checking"""
        try:
            content = self.get_file_content(self.owner, self.repo, note_path)
            if content is None:
                return None, None
            
            # Calculate content hash for duplicate checking
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
            
            return content, content_hash
            
        except Exception as e:
            self.logger.error(f"note content error: {e}")
            return None, None
    
    async def load_notes_to_memory(self, memory_manager) -> int:
        """Load Obsidian notes to memory manager"""
        try:
            # Check repository last updated
            last_updated = self.get_repository_last_updated()
            if not last_updated:
                self.logger.warning("repository update check failed")
                return 0
            
            # Check force reload setting
            force_reload = config.force_reload_obsidian
            
            # Define namespace for metadata
            namespace = (memory_manager.thread_id, "obsidian_meta")
            
            if not force_reload:
                # Check existing update
                existing_update = await memory_manager.store.aget(namespace, "last_updated")
                
                if (existing_update and 
                    hasattr(existing_update, 'value') and 
                    existing_update.value.get("value") == last_updated):
                    self.logger.info("notes already up-to-date")
                    return 0
            else:
                self.logger.info("force reload enabled - reloading all notes")
            
            # Get notes list
            notes = self.get_obsidian_notes()
            if not notes:
                self.logger.warning("no notes found")
                return 0
            
            # Load each note to memory (with duplicate checking)
            loaded_count = 0
            
            for i, note in enumerate(notes):
                # API rate limiting (moderate intervals)
                if i > 0 and i % 10 == 0:
                    time.sleep(0.1)  # 100ms rest every 10 notes
                    self.logger.info(f"processed {i+1}/{len(notes)} notes")
                
                content, content_hash = self.get_note_content_with_hash(note["path"])
                
                # Debug logging for first 3 notes
                if i < 3:
                    self.logger.info(f"note {i+1}: {note['name']}")
                    self.logger.info(f"  path: {note['path']}")
                    self.logger.info(f"  content_length: {len(content) if content else 0}")
                    self.logger.info(f"  has_hash: {bool(content_hash)}")
                
                if content and content_hash:
                    # Skip duplicate check if force reload
                    if force_reload:
                        should_save = True
                        if i < 3:
                            self.logger.info("  force_reload: skipping duplicate check")
                    else:
                        # Check existing hash
                        note_namespace = (memory_manager.thread_id, "obsidian_notes")
                        existing_note = await memory_manager.store.aget(note_namespace, note["name"])
                        
                        existing_hash = None
                        if existing_note and hasattr(existing_note, 'value'):
                            existing_hash = existing_note.value.get("hash")
                        elif existing_note:
                            existing_hash = existing_note.get("hash")
                        
                        # Debug info for first 3 notes
                        if i < 3:
                            self.logger.info(f"  existing_note: {bool(existing_note)}")
                            self.logger.info(f"  existing_hash: {existing_hash}")
                            self.logger.info(f"  new_hash: {content_hash}")
                            self.logger.info(f"  hash_different: {existing_hash != content_hash}")
                        
                        should_save = not existing_note or existing_hash != content_hash
                    
                    if should_save:
                        # New or updated note (with filtering)
                        content_length = len(content.strip()) if content else 0
                        
                        # Detailed debug logging
                        self.logger.info(f"[{i+1}/{len(notes)}] {note['name']}")
                        self.logger.info(f"  content_length: {content_length}")
                        self.logger.info(f"  has_content: {bool(content)}")
                        self.logger.info(f"  passes_length_filter: {content_length > 10}")
                        
                        if content and content_length > 10:  # Minimum length check
                            try:
                                self.logger.info(f"  preparing save data for: {note['name']}")
                                note_data = {
                                    "content": f"Obsidianノート「{note['name']}」: {content[:500]}{'...' if len(content) > 500 else ''}",
                                    "hash": content_hash,
                                    "path": note["path"],
                                    "full_content": content,
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                self.logger.info(f"  calling store.aput for: {note['name']}")
                                note_namespace = (memory_manager.thread_id, "obsidian_notes")
                                await memory_manager.store.aput(note_namespace, note["name"], note_data)
                                loaded_count += 1
                                
                                self.logger.info(f"  ✓ SAVED: {note['name']} (total: {loaded_count})")
                            except Exception as save_error:
                                self.logger.error(f"  ✗ SAVE_ERROR: {note['name']} - {save_error}")
                                self.logger.error("  continuing with next note...")
                        else:
                            self.logger.info(f"  ✗ FILTERED: {note['name']} (content_length: {content_length})")
                    else:
                        self.logger.info(f"  ✗ SKIPPED: {note['name']} (existing/duplicate)")
            
            # Record update timestamp
            await memory_manager.store.aput(namespace, "last_updated", {"value": last_updated})
            
            self.logger.info(f"loaded {loaded_count} notes (total: {len(notes)})")
            
            # Log loaded notes sample (first 10)
            if loaded_count > 0:
                self.logger.info("loaded notes sample:")
                sample_count = 0
                for i, note in enumerate(notes):
                    if sample_count >= 10:
                        break
                    content, _ = self.get_note_content_with_hash(note["path"])
                    if content and len(content.strip()) > 10:
                        self.logger.info(f"  - {note['name']}.md")
                        sample_count += 1
            
            return loaded_count
            
        except Exception as e:
            self.logger.error(f"loading error: {e}")
            return 0
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get Obsidian integration statistics"""
        return {
            "owner": self.owner,
            "repo": self.repo,
            "branch": self.branch,
            "has_token": bool(self.github_token),
            "force_reload": config.force_reload_obsidian,
            "last_check": datetime.now().isoformat()
        }