"""
MCP Registry Configuration

Configuration settings for the MCP registry service.
Uses pydantic-settings for environment variable support.
"""

from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Application settings
    app_name: str = "MCP Server Registry"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 4
    
    # Database settings
    database_url: str = "sqlite:///./mcp_registry.db"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    
    # Redis settings (for caching and task queue)
    redis_url: Optional[str] = "redis://localhost:6379"
    cache_ttl: int = 300  # 5 minutes
    
    # Security settings
    secret_key: str = "your-secret-key-here"  # Change in production
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # MCP Client settings
    max_connections_per_server: int = 5
    connection_timeout: int = 30
    health_check_interval: int = 60  # seconds
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # External services
    core_backend_url: str = "http://localhost:8080"
    
    # Feature flags
    enable_websocket: bool = True
    enable_batch_operations: bool = True
    enable_health_monitoring: bool = True
    
    class Config:
        env_prefix = "MCP_"


# Create global settings instance
settings = Settings()


# Helper functions
def get_database_url() -> str:
    """Get the database URL with proper formatting."""
    return settings.database_url


def get_redis_client():
    """Get Redis client for caching."""
    if settings.redis_url:
        import redis
        return redis.from_url(settings.redis_url)
    return None


def is_production() -> bool:
    """Check if running in production mode."""
    return not settings.debug 