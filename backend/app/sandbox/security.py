"""
Security Configuration for Sandboxed Agents

Defines trust levels and security configurations for agent containers.
Supports configurable policies for different deployment scenarios.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TrustLevel(str, Enum):
    """Trust levels for agent execution environments."""
    TRUSTED = "trusted"        # Your code, full access
    SANDBOXED = "sandboxed"    # Limited network, resource caps
    UNTRUSTED = "untrusted"    # gVisor, no network, strict limits


class AgentSecurityConfig(BaseModel):
    """Security configuration for agent execution."""
    
    trust_level: TrustLevel = TrustLevel.TRUSTED
    
    # Network Configuration
    network_enabled: bool = True
    allowed_hosts: Optional[List[str]] = None  # None = all (if network enabled)
    allowed_ports: Optional[List[int]] = None  # None = all
    dns_servers: Optional[List[str]] = None
    
    # Resource Limits
    memory_limit_mb: int = 512
    cpu_limit: float = 1.0  # cores
    cpu_shares: int = 1024  # relative weight
    timeout_seconds: int = 300
    max_processes: int = 100
    
    # Filesystem Configuration
    writable_paths: List[str] = Field(default_factory=lambda: ["/workspace", "/tmp"])
    read_only_root: bool = True
    max_file_size_mb: int = 100
    
    # Code Execution Permissions
    allow_shell: bool = True
    allow_network_tools: bool = True
    allow_file_tools: bool = True
    allow_subprocess: bool = True
    
    # MCP Tool Restrictions
    mcp_capabilities: Optional[List[str]] = None  # None = all allowed
    blocked_tools: List[str] = Field(default_factory=list)
    
    # Audit and Logging
    log_all_commands: bool = False
    capture_stdout: bool = True
    capture_stderr: bool = True
    
    class Config:
        use_enum_values = True


# Pre-defined trust level presets
TRUST_PRESETS: Dict[TrustLevel, AgentSecurityConfig] = {
    TrustLevel.TRUSTED: AgentSecurityConfig(
        trust_level=TrustLevel.TRUSTED,
        network_enabled=True,
        memory_limit_mb=2048,
        cpu_limit=2.0,
        timeout_seconds=600,
        allow_shell=True,
        allow_network_tools=True,
        allow_subprocess=True,
        read_only_root=False,
    ),
    TrustLevel.SANDBOXED: AgentSecurityConfig(
        trust_level=TrustLevel.SANDBOXED,
        network_enabled=True,
        allowed_hosts=[
            "api.openai.com",
            "api.anthropic.com",
            "api.groq.com",
            "localhost",
            "127.0.0.1",
        ],
        memory_limit_mb=512,
        cpu_limit=1.0,
        timeout_seconds=120,
        allow_shell=False,
        allow_subprocess=False,
        read_only_root=True,
        log_all_commands=True,
    ),
    TrustLevel.UNTRUSTED: AgentSecurityConfig(
        trust_level=TrustLevel.UNTRUSTED,
        network_enabled=False,
        memory_limit_mb=256,
        cpu_limit=0.5,
        timeout_seconds=30,
        read_only_root=True,
        allow_shell=False,
        allow_network_tools=False,
        allow_subprocess=False,
        max_processes=10,
        log_all_commands=True,
    ),
}


def get_security_config(trust_level: TrustLevel, overrides: Optional[Dict[str, Any]] = None) -> AgentSecurityConfig:
    """
    Get security configuration for a trust level with optional overrides.
    
    Args:
        trust_level: Base trust level to use
        overrides: Optional dictionary of config overrides
        
    Returns:
        AgentSecurityConfig with applied overrides
    """
    base_config = TRUST_PRESETS[trust_level].model_copy()
    
    if overrides:
        for key, value in overrides.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
    
    return base_config


def validate_security_config(config: AgentSecurityConfig) -> List[str]:
    """
    Validate security configuration for potential issues.
    
    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []
    
    # Check for dangerous combinations
    if config.trust_level == TrustLevel.UNTRUSTED:
        if config.network_enabled:
            warnings.append("UNTRUSTED level should not have network enabled")
        if config.allow_shell:
            warnings.append("UNTRUSTED level should not allow shell access")
        if config.allow_subprocess:
            warnings.append("UNTRUSTED level should not allow subprocess creation")
    
    if config.trust_level == TrustLevel.SANDBOXED:
        if config.allow_shell and not config.log_all_commands:
            warnings.append("SANDBOXED with shell access should log all commands")
    
    # Resource sanity checks
    if config.memory_limit_mb < 64:
        warnings.append("Memory limit below 64MB may cause issues")
    if config.timeout_seconds < 5:
        warnings.append("Timeout below 5 seconds is very restrictive")
    if config.max_processes < 5:
        warnings.append("Max processes below 5 may prevent basic operations")
    
    return warnings
