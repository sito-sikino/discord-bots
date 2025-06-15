"""
Semantic routing for multi-agent system
Vector similarity-based agent selection with LLM fallback
"""
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .config import config


class SemanticRouter:
    """Semantic routing using vector similarity"""
    
    def __init__(self, embeddings_model):
        self.embeddings = embeddings_model
        self.logger = self._setup_logging()
        
        # Agent characteristic vectors (will be computed and cached)
        self.agent_vectors = {}
        self.vector_cache_file = Path("agent_vectors.json")
        
        # Agent definitions - 専門特化・排他的特徴量
        self.agent_definitions = {
            "spectra": "挨拶 会話 説明 相談 質問回答 dialogue greeting explanation consultation interpersonal communication social interaction discussion facilitation understanding",
            "lynq": "数学 計算 統計 論理 分析 証明 mathematics calculation statistics logic analysis proof reasoning systematic methodology verification scientific",
            "paz": "芸術 創作 遊び エンターテインメント art creativity entertainment play artistic expression imagination unconventional breakthrough innovation design inspiration"
        }
        
        self.logger.info("SemanticRouter initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup semantic router logger"""
        logger = logging.getLogger("SemanticRouter")
        
        if logger.handlers:
            return logger
        
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler("bot.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s | SEMANTIC| %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    async def initialize_vectors(self) -> bool:
        """Initialize or load agent characteristic vectors"""
        try:
            # Try to load cached vectors
            if self.vector_cache_file.exists():
                self.logger.info("loading cached agent vectors")
                with open(self.vector_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    
                # Convert lists back to numpy arrays
                self.agent_vectors = {
                    agent: np.array(vector) 
                    for agent, vector in cached_data.items()
                }
                
                self.logger.info(f"loaded {len(self.agent_vectors)} cached agent vectors")
                return True
            
            # Compute vectors if cache doesn't exist
            return await self._compute_agent_vectors()
            
        except Exception as e:
            self.logger.error(f"vector initialization error: {e}")
            return False
    
    async def _compute_agent_vectors(self) -> bool:
        """Compute and cache agent characteristic vectors"""
        try:
            self.logger.info("computing agent characteristic vectors...")
            
            computed_vectors = {}
            
            for agent_name, characteristics in self.agent_definitions.items():
                # Use the text directly (already a string now)
                text = characteristics
                
                self.logger.info(f"computing vector for {agent_name}: '{text}'")
                
                # Compute embedding vector
                vector = await self.embeddings.aembed_query(text)
                vector_array = np.array(vector)
                computed_vectors[agent_name] = vector_array
                
                self.logger.info(f"computed vector for {agent_name} (dim: {len(vector)}, first5: {vector_array[:5]})")
            
            # Verify vectors are different
            agents = list(computed_vectors.keys())
            for i, agent1 in enumerate(agents):
                for agent2 in agents[i+1:]:
                    sim = self._cosine_similarity(computed_vectors[agent1], computed_vectors[agent2])
                    self.logger.info(f"similarity {agent1}-{agent2}: {sim:.3f}")
            
            # Cache vectors to file
            cache_data = {
                agent: vector.tolist() 
                for agent, vector in computed_vectors.items()
            }
            
            with open(self.vector_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.agent_vectors = computed_vectors
            self.logger.info(f"cached {len(computed_vectors)} agent vectors to {self.vector_cache_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"vector computation error: {e}")
            return False
    
    async def compute_similarity(self, user_input: str) -> Dict[str, float]:
        """Compute cosine similarity between user input and agent vectors"""
        try:
            if not self.agent_vectors:
                await self.initialize_vectors()
            
            # Compute user input embedding
            user_vector = np.array(await self.embeddings.aembed_query(user_input))
            
            # Compute similarities with each agent
            similarities = {}
            for agent_name, agent_vector in self.agent_vectors.items():
                similarity = self._cosine_similarity(user_vector, agent_vector)
                similarities[agent_name] = similarity
            
            self.logger.info(f"computed similarities for: {user_input[:30]}...")
            return similarities
            
        except Exception as e:
            self.logger.error(f"similarity computation error: {e}")
            return {}
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        try:
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Compute cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            # Ensure result is between -1 and 1
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"cosine similarity error: {e}")
            return 0.0
    
    def select_best_agent(self, similarities: Dict[str, float], threshold: float = 0.7) -> Tuple[str, float]:
        """Select best agent based on similarity scores with enhanced confidence calculation"""
        if not similarities:
            return "spectra", 0.0  # Default fallback
        
        # Sort scores in descending order
        sorted_scores = sorted(similarities.values(), reverse=True)
        
        if len(sorted_scores) < 2:
            return "spectra", 0.0
        
        # Find agent with highest similarity
        best_agent = max(similarities, key=similarities.get)
        best_score = sorted_scores[0]
        second_score = sorted_scores[1]
        
        # Enhanced confidence calculation
        confidence = self._calculate_confidence(best_score, second_score, similarities)
        
        self.logger.info(
            f"best agent: {best_agent} (score: {best_score:.3f}, "
            f"gap: {best_score - second_score:.3f}, confidence: {confidence:.3f}, "
            f"confident: {confidence > threshold})"
        )
        
        return best_agent, confidence
    
    def _calculate_confidence(self, best_score: float, second_score: float, all_similarities: Dict[str, float]) -> float:
        """Calculate confidence based on multiple factors"""
        
        # Factor 1: Absolute score (how good the best match is)
        absolute_factor = best_score
        
        # Factor 2: Relative gap (how much better than second best)
        gap = best_score - second_score
        relative_factor = min(gap * 2.0, 0.3)  # Cap at 0.3 bonus
        
        # Factor 3: Consistency (how much better than average)
        avg_score = sum(all_similarities.values()) / len(all_similarities)
        consistency_factor = min((best_score - avg_score) * 1.5, 0.2)  # Cap at 0.2 bonus
        
        # Combine factors
        confidence = absolute_factor + relative_factor + consistency_factor
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        self.logger.debug(
            f"confidence calculation: abs={absolute_factor:.3f}, "
            f"gap={relative_factor:.3f}, consistency={consistency_factor:.3f}, "
            f"final={confidence:.3f}"
        )
        
        return confidence
    
    def get_vector_info(self) -> Dict[str, any]:
        """Get vector information for debugging"""
        if not self.agent_vectors:
            return {"status": "not_initialized", "count": 0}
        
        return {
            "status": "initialized",
            "count": len(self.agent_vectors),
            "agents": list(self.agent_vectors.keys()),
            "dimensions": len(next(iter(self.agent_vectors.values()))),
            "cache_file": str(self.vector_cache_file),
            "cache_exists": self.vector_cache_file.exists()
        }