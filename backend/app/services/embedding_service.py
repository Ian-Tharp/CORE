"""
Embedding Service

Generates embeddings using the local Ollama instance for vector similarity search.
Supports both single and batch embedding generation.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

import httpx
from openai import AsyncOpenAI

from app.dependencies import get_ollama_client

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using Ollama.
    
    Uses nomic-embed-text or similar models for generating
    embeddings that work with pgvector similarity search.
    """
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.model_name: str = "nomic-embed-text"  # Default embedding model
        self.dimensions: int = 768  # Default dimensions for nomic-embed-text
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> None:
        """Initialize the embedding service and verify model availability."""
        try:
            self.client = get_ollama_client()
            
            # Check if embedding model is available
            available_models = await self._list_available_models()
            
            if self.model_name in available_models:
                logger.info(f"Using embedding model: {self.model_name}")
            else:
                # Try alternative models
                alternatives = ["mxbai-embed-large", "all-minilm", "nomic-embed-text"]
                for alt_model in alternatives:
                    if alt_model in available_models:
                        self.model_name = alt_model
                        logger.info(f"Using alternative embedding model: {self.model_name}")
                        break
                else:
                    logger.warning(f"No embedding models found. Available: {available_models}")
                    # Will fall back to using first available model
                    if available_models:
                        self.model_name = available_models[0]
                        logger.info(f"Falling back to: {self.model_name}")
            
            # Get model dimensions
            await self._get_model_info()
            
            logger.info("EmbeddingService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {e}")
            raise
            
    async def _list_available_models(self) -> List[str]:
        """List available models from Ollama."""
        try:
            # Use httpx to directly query Ollama API
            async with httpx.AsyncClient() as client:
                response = await client.get("http://ollama:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [model["name"].split(":")[0] for model in data.get("models", [])]
                else:
                    logger.warning("Could not fetch available models from Ollama")
                    return []
        except Exception as e:
            logger.error(f"Error listing available models: {e}")
            return []
    
    async def _get_model_info(self) -> None:
        """Get model information and cache dimensions."""
        try:
            if self.model_name in self._model_cache:
                cached_info = self._model_cache[self.model_name]
                self.dimensions = cached_info["dimensions"]
                return
            
            # Try to generate a test embedding to determine dimensions
            test_embedding = await self.generate_embedding("test")
            if test_embedding:
                self.dimensions = len(test_embedding)
                self._model_cache[self.model_name] = {
                    "dimensions": self.dimensions,
                    "model_name": self.model_name
                }
                logger.info(f"Model {self.model_name} has {self.dimensions} dimensions")
            
        except Exception as e:
            logger.warning(f"Could not determine model dimensions: {e}, using default")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        if not self.client:
            raise RuntimeError("EmbeddingService not initialized")
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimensions  # Return zero vector for empty text
        
        try:
            # Use OpenAI-compatible embedding endpoint
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text.strip(),
                encoding_format="float"
            )
            
            if response.data:
                embedding = response.data[0].embedding
                return embedding
            else:
                raise RuntimeError("No embedding data returned")
                
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {str(e)[:100]}")
            # Return zero vector as fallback
            return [0.0] * self.dimensions
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 10
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors (one per input text)
        """
        if not self.client:
            raise RuntimeError("EmbeddingService not initialized")
        
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches to avoid overwhelming the service
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # For Ollama, we need to process one at a time
                # as batch processing isn't consistently supported
                batch_embeddings = []
                
                for text in batch:
                    embedding = await self.generate_embedding(text)
                    batch_embeddings.append(embedding)
                
                all_embeddings.extend(batch_embeddings)
                
                # Small delay between batches to be nice to the service
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                # Add zero vectors for failed batch
                batch_embeddings = [[0.0] * self.dimensions] * len(batch)
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.
        
        Returns:
            Dict containing model name, dimensions, and availability
        """
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "available": self.client is not None,
            "service": "ollama"
        }
    
    async def health_check(self) -> bool:
        """
        Check if the embedding service is healthy.
        
        Returns:
            True if service can generate embeddings, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Try to generate a test embedding
            test_embedding = await self.generate_embedding("health check")
            return len(test_embedding) == self.dimensions
            
        except Exception as e:
            logger.warning(f"Embedding service health check failed: {e}")
            return False


# Global embedding service instance
embedding_service = EmbeddingService()