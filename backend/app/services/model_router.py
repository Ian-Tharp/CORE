"""
CORE Model Router - Dynamic model selection and routing.

Provides:
- Route requests to different LLM providers (Ollama, OpenAI, Anthropic)
- Task-based model selection (simple → fast local, complex → powerful cloud)
- Fallback chains when primary model fails
- Cost tracking and optimization

RSI TODO: Add model performance tracking
RSI TODO: Add automatic model selection based on task type
RSI TODO: Add caching layer for common prompts
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
from enum import Enum
from dataclasses import dataclass

from openai import AsyncOpenAI, OpenAI
import logging

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Supported model providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelTier(str, Enum):
    """Model capability tiers."""
    FAST = "fast"        # Quick, cheap, good for simple tasks
    BALANCED = "balanced"  # Good balance of speed and quality
    POWERFUL = "powerful"  # Best quality, slower/expensive


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    id: str
    provider: ModelProvider
    tier: ModelTier
    display_name: str
    context_window: int
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD
    supports_tools: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider.value,
            "tier": self.tier.value,
            "display_name": self.display_name,
            "context_window": self.context_window,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision
        }


# Available models registry
MODELS: Dict[str, ModelConfig] = {
    # Ollama (local, free)
    "gpt-oss:20b": ModelConfig(
        id="gpt-oss:20b",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.BALANCED,
        display_name="GPT-OSS 20B (Local)",
        context_window=8192,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0
    ),
    "llama3.2:latest": ModelConfig(
        id="llama3.2:latest",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.FAST,
        display_name="Llama 3.2 (Local)",
        context_window=8192,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0
    ),
    "deepseek-r1:32b": ModelConfig(
        id="deepseek-r1:32b",
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.POWERFUL,
        display_name="DeepSeek R1 32B (Local)",
        context_window=32768,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0
    ),
    
    # OpenAI
    "gpt-4o-mini": ModelConfig(
        id="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        tier=ModelTier.FAST,
        display_name="GPT-4o Mini",
        context_window=128000,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        supports_vision=True
    ),
    "gpt-4o": ModelConfig(
        id="gpt-4o",
        provider=ModelProvider.OPENAI,
        tier=ModelTier.BALANCED,
        display_name="GPT-4o",
        context_window=128000,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        supports_vision=True
    ),
    "o1-preview": ModelConfig(
        id="o1-preview",
        provider=ModelProvider.OPENAI,
        tier=ModelTier.POWERFUL,
        display_name="o1 Preview",
        context_window=128000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.06,
        supports_tools=False
    ),
    
    # Anthropic
    "claude-3-5-haiku-latest": ModelConfig(
        id="claude-3-5-haiku-latest",
        provider=ModelProvider.ANTHROPIC,
        tier=ModelTier.FAST,
        display_name="Claude 3.5 Haiku",
        context_window=200000,
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.004,
        supports_vision=True
    ),
    "claude-sonnet-4-20250514": ModelConfig(
        id="claude-sonnet-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        tier=ModelTier.BALANCED,
        display_name="Claude Sonnet 4",
        context_window=200000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supports_vision=True
    ),
    "claude-opus-4-20250514": ModelConfig(
        id="claude-opus-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        tier=ModelTier.POWERFUL,
        display_name="Claude Opus 4",
        context_window=200000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        supports_vision=True
    ),
}


class ModelRouter:
    """
    Routes requests to appropriate models based on task requirements.
    """
    
    def __init__(self):
        self._clients: Dict[ModelProvider, Any] = {}
        self._usage_stats: Dict[str, Dict[str, Any]] = {}
        self._total_cost = 0.0
        
        # Default model preferences
        self.default_model = os.getenv("CORE_DEFAULT_MODEL", "gpt-oss:20b")
        self.fallback_chain = ["gpt-oss:20b", "gpt-4o-mini", "claude-3-5-haiku-latest"]
    
    def get_client(self, provider: ModelProvider):
        """Get or create client for a provider."""
        if provider in self._clients:
            return self._clients[provider]
        
        if provider == ModelProvider.OLLAMA:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
            client = AsyncOpenAI(
                base_url=f"{base_url}/v1",
                api_key="ollama"
            )
        
        elif provider == ModelProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            client = AsyncOpenAI(api_key=api_key)
        
        elif provider == ModelProvider.ANTHROPIC:
            # Anthropic uses its own SDK, but we can use OpenAI-compatible endpoint
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            # Using anthropic's OpenAI-compatible API
            client = AsyncOpenAI(
                base_url="https://api.anthropic.com/v1",
                api_key=api_key
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        self._clients[provider] = client
        return client
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a model."""
        return MODELS.get(model_id)
    
    def list_models(
        self,
        provider: Optional[ModelProvider] = None,
        tier: Optional[ModelTier] = None
    ) -> List[ModelConfig]:
        """List available models with optional filtering."""
        models = list(MODELS.values())
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        if tier:
            models = [m for m in models if m.tier == tier]
        
        return models
    
    def select_model(
        self,
        task_type: Literal["simple", "complex", "creative", "reasoning"],
        require_tools: bool = False,
        require_vision: bool = False,
        prefer_local: bool = True,
        max_cost_per_1k: Optional[float] = None
    ) -> str:
        """
        Automatically select the best model for a task.
        
        Args:
            task_type: Type of task
            require_tools: Whether tool/function calling is needed
            require_vision: Whether vision capability is needed
            prefer_local: Prefer local models when suitable
            max_cost_per_1k: Maximum cost per 1k tokens
            
        Returns:
            Model ID
        """
        # Map task types to tiers
        tier_mapping = {
            "simple": ModelTier.FAST,
            "complex": ModelTier.POWERFUL,
            "creative": ModelTier.BALANCED,
            "reasoning": ModelTier.POWERFUL
        }
        
        target_tier = tier_mapping.get(task_type, ModelTier.BALANCED)
        
        # Filter candidates
        candidates = []
        for model in MODELS.values():
            # Check requirements
            if require_tools and not model.supports_tools:
                continue
            if require_vision and not model.supports_vision:
                continue
            if max_cost_per_1k and model.cost_per_1k_output > max_cost_per_1k:
                continue
            
            candidates.append(model)
        
        if not candidates:
            return self.default_model
        
        # Sort by preference
        def score_model(m: ModelConfig) -> tuple:
            tier_match = 0 if m.tier == target_tier else 1
            is_local = 0 if (prefer_local and m.provider == ModelProvider.OLLAMA) else 1
            cost = m.cost_per_1k_output
            return (tier_match, is_local, cost)
        
        candidates.sort(key=score_model)
        return candidates[0].id
    
    async def complete(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a completion request to the appropriate provider.
        
        Args:
            model_id: Model identifier
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            tools: Optional tool definitions
            **kwargs: Additional provider-specific options
            
        Returns:
            Completion response
        """
        config = self.get_model_config(model_id)
        if not config:
            raise ValueError(f"Unknown model: {model_id}")
        
        start_time = time.time()
        
        try:
            client = self.get_client(config.provider)
            
            # Build request
            request_params = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            if tools and config.supports_tools:
                request_params["tools"] = tools
            
            # Make request
            response = await client.chat.completions.create(**request_params)
            
            # Track usage
            duration = time.time() - start_time
            usage = response.usage
            
            if usage:
                cost = (
                    (usage.prompt_tokens / 1000) * config.cost_per_1k_input +
                    (usage.completion_tokens / 1000) * config.cost_per_1k_output
                )
                self._track_usage(model_id, usage.prompt_tokens, usage.completion_tokens, cost, duration)
            
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls if hasattr(response.choices[0].message, "tool_calls") else None,
                "model": model_id,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0
                },
                "duration_ms": duration * 1000
            }
        
        except Exception as e:
            logger.error(f"Model {model_id} failed: {e}")
            
            # Try fallback
            for fallback_id in self.fallback_chain:
                if fallback_id != model_id:
                    logger.info(f"Trying fallback model: {fallback_id}")
                    try:
                        return await self.complete(
                            fallback_id, messages, temperature, max_tokens, tools, **kwargs
                        )
                    except Exception:
                        continue
            
            raise
    
    def _track_usage(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        duration: float
    ):
        """Track usage statistics for a model."""
        if model_id not in self._usage_stats:
            self._usage_stats[model_id] = {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost": 0.0,
                "total_duration": 0.0
            }
        
        stats = self._usage_stats[model_id]
        stats["requests"] += 1
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += completion_tokens
        stats["total_cost"] += cost
        stats["total_duration"] += duration
        
        self._total_cost += cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all models."""
        return {
            "by_model": self._usage_stats,
            "total_cost": self._total_cost,
            "total_requests": sum(s["requests"] for s in self._usage_stats.values())
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self._usage_stats = {}
        self._total_cost = 0.0


# Global router instance
_model_router: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    """Get the global model router instance."""
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router
