"""
Model Configuration

Centralized configuration for LLM models used in CORE.
Supports local (Ollama) and cloud (OpenAI, Anthropic) providers.

Usage:
    from app.config.models import get_model_config, ModelProvider
    
    config = get_model_config("gpt-4")
    # or
    config = get_model_config(ModelProvider.OLLAMA, "gpt-oss:20b")
"""

import os
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Supported model providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # Generic local


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    
    provider: ModelProvider
    model_name: str
    display_name: str
    
    # Connection settings
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None  # Environment variable name for API key
    
    # Model parameters
    default_temperature: float = 0.7
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    
    # Performance hints
    avg_tokens_per_second: Optional[float] = None  # For estimation
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None
    
    # Recommended use cases
    recommended_for: list[str] = Field(default_factory=list)
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None
    
    def get_base_url(self) -> str:
        """Get base URL for API calls."""
        if self.base_url:
            return self.base_url
        
        if self.provider == ModelProvider.OLLAMA:
            return os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        elif self.provider == ModelProvider.OPENAI:
            return "https://api.openai.com/v1"
        elif self.provider == ModelProvider.ANTHROPIC:
            return "https://api.anthropic.com"
        
        return ""


# Pre-configured models
MODELS: Dict[str, ModelConfig] = {
    # Local Ollama models
    "gpt-oss:20b": ModelConfig(
        provider=ModelProvider.OLLAMA,
        model_name="gpt-oss:20b",
        display_name="GPT-OSS 20B (Local)",
        default_temperature=0.7,
        max_tokens=8192,
        avg_tokens_per_second=15,  # Estimate for RTX 4080
        recommended_for=["simple_tasks", "quick_questions", "local_dev"],
    ),
    "qwen2.5:32b": ModelConfig(
        provider=ModelProvider.OLLAMA,
        model_name="qwen2.5:32b",
        display_name="Qwen 2.5 32B (Local)",
        default_temperature=0.7,
        max_tokens=32768,
        avg_tokens_per_second=10,
        recommended_for=["complex_reasoning", "code_generation"],
    ),
    "nomic-embed-text": ModelConfig(
        provider=ModelProvider.OLLAMA,
        model_name="nomic-embed-text",
        display_name="Nomic Embed (Local)",
        supports_streaming=False,
        supports_tools=False,
        recommended_for=["embeddings", "semantic_search"],
    ),
    
    # OpenAI models
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
        display_name="GPT-4o",
        api_key_env="OPENAI_API_KEY",
        max_tokens=128000,
        supports_vision=True,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        recommended_for=["complex_tasks", "multi_step_plans", "production"],
    ),
    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        display_name="GPT-4o Mini",
        api_key_env="OPENAI_API_KEY",
        max_tokens=128000,
        supports_vision=True,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        recommended_for=["simple_tasks", "high_volume", "cost_sensitive"],
    ),
    
    # Anthropic models
    "claude-3-5-sonnet": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=200000,
        supports_vision=True,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        recommended_for=["complex_reasoning", "code", "analysis"],
    ),
    "claude-3-haiku": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=200000,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        recommended_for=["quick_responses", "high_volume"],
    ),
}


# Default model for each use case
DEFAULT_MODELS = {
    "comprehension": "gpt-oss:20b",  # Use local for dev
    "orchestration": "gpt-oss:20b",
    "reasoning": "gpt-oss:20b",
    "evaluation": "gpt-oss:20b",
    "conversation": "gpt-oss:20b",
    "embeddings": "nomic-embed-text",
}

# Production defaults (when cloud APIs available)
PRODUCTION_MODELS = {
    "comprehension": "gpt-4o-mini",
    "orchestration": "gpt-4o",
    "reasoning": "gpt-4o",
    "evaluation": "gpt-4o-mini",
    "conversation": "gpt-4o-mini",
    "embeddings": "nomic-embed-text",
}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a model by name."""
    return MODELS.get(model_name)


def get_default_model(use_case: str, production: bool = False) -> str:
    """Get default model for a use case."""
    models = PRODUCTION_MODELS if production else DEFAULT_MODELS
    return models.get(use_case, "gpt-oss:20b")


def list_available_models() -> list[ModelConfig]:
    """List all configured models."""
    return list(MODELS.values())


def get_models_by_provider(provider: ModelProvider) -> list[ModelConfig]:
    """Get all models for a specific provider."""
    return [m for m in MODELS.values() if m.provider == provider]


def is_model_available(model_name: str) -> bool:
    """Check if a model is configured and available."""
    config = get_model_config(model_name)
    if not config:
        return False
    
    # Check if API key is available for cloud models
    if config.provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC]:
        return config.get_api_key() is not None
    
    return True
