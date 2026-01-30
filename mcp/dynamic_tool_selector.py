"""
Dynamic Tool Selector for MCP Registry

Provides capability-based tool selection and routing for the CORE loop.
Matches task requirements to available MCP servers and their tools.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field
import httpx

logger = logging.getLogger(__name__)


class Capability(str, Enum):
    """Standard capability categories for MCP tools."""
    
    # File Operations
    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    FILE_SEARCH = "file:search"
    
    # Code Execution
    CODE_PYTHON = "code:python"
    CODE_JAVASCRIPT = "code:javascript"
    CODE_SHELL = "code:shell"
    
    # Web Operations
    WEB_FETCH = "web:fetch"
    WEB_SEARCH = "web:search"
    WEB_BROWSE = "web:browse"
    
    # Data Operations
    DATA_QUERY = "data:query"
    DATA_TRANSFORM = "data:transform"
    DATA_ANALYZE = "data:analyze"
    
    # External Services
    API_CALL = "api:call"
    DATABASE = "database"
    MESSAGING = "messaging"
    
    # AI Operations
    LLM_CALL = "llm:call"
    EMBEDDING = "embedding"
    IMAGE_GEN = "image:gen"
    
    # System Operations
    SYSTEM_INFO = "system:info"
    PROCESS_MGMT = "process:mgmt"


class ToolDefinition(BaseModel):
    """Definition of an available tool."""
    
    name: str
    description: str
    server_id: str
    server_name: str
    capabilities: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    estimated_latency_ms: Optional[int] = None
    cost_per_call: Optional[float] = None
    rate_limit: Optional[int] = None  # calls per minute
    
    class Config:
        arbitrary_types_allowed = True


class ToolSelection(BaseModel):
    """Result of tool selection."""
    
    selected_tools: List[ToolDefinition]
    required_capabilities: List[str]
    missing_capabilities: List[str]
    selection_reason: str
    confidence: float  # 0.0 to 1.0
    
    class Config:
        arbitrary_types_allowed = True


class TaskAnalysis(BaseModel):
    """Analysis of a task's tool requirements."""
    
    task_description: str
    inferred_capabilities: List[str]
    explicit_capabilities: List[str]
    priority_order: List[str]  # Capabilities in order of importance
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class DynamicToolSelector:
    """
    Selects appropriate MCP tools based on task requirements.
    
    Features:
    - Capability-based matching
    - Priority-based selection
    - Caching of tool definitions
    - Support for explicit and inferred capabilities
    """
    
    # Keyword to capability mapping for inference
    CAPABILITY_KEYWORDS: Dict[str, List[str]] = {
        Capability.FILE_READ: ["read", "open", "load", "file", "document"],
        Capability.FILE_WRITE: ["write", "save", "create", "export", "file"],
        Capability.FILE_SEARCH: ["find", "search", "grep", "locate", "file"],
        Capability.CODE_PYTHON: ["python", "script", "execute", "run", "code"],
        Capability.CODE_JAVASCRIPT: ["javascript", "js", "node"],
        Capability.CODE_SHELL: ["shell", "bash", "command", "terminal"],
        Capability.WEB_FETCH: ["fetch", "download", "url", "http", "request"],
        Capability.WEB_SEARCH: ["search", "google", "find online", "web search"],
        Capability.WEB_BROWSE: ["browse", "navigate", "website", "page"],
        Capability.DATA_QUERY: ["query", "sql", "database", "select"],
        Capability.DATA_TRANSFORM: ["transform", "convert", "format", "parse"],
        Capability.DATA_ANALYZE: ["analyze", "statistics", "insights", "summarize"],
        Capability.API_CALL: ["api", "endpoint", "rest", "graphql"],
        Capability.LLM_CALL: ["llm", "gpt", "claude", "ai", "generate"],
        Capability.EMBEDDING: ["embed", "vector", "similarity"],
        Capability.IMAGE_GEN: ["image", "picture", "generate image", "dall-e"],
    }
    
    def __init__(
        self,
        registry_url: str = "http://localhost:8000",
        cache_ttl_minutes: int = 5
    ):
        """
        Initialize the tool selector.
        
        Args:
            registry_url: URL of the MCP registry service
            cache_ttl_minutes: Cache TTL for tool definitions
        """
        self.registry_url = registry_url
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        # Tool cache
        self._tools_cache: Dict[str, ToolDefinition] = {}
        self._server_tools: Dict[str, List[str]] = {}  # server_id -> tool names
        self._capability_index: Dict[str, Set[str]] = {}  # capability -> tool names
        self._last_cache_update: datetime = datetime.min
        
        # Lock for thread-safe cache updates
        self._lock = asyncio.Lock()
    
    async def refresh_tool_cache(self, force: bool = False):
        """
        Refresh the tool cache from the registry.
        
        Args:
            force: Force refresh even if cache is valid
        """
        async with self._lock:
            if not force and datetime.utcnow() - self._last_cache_update < self.cache_ttl:
                return
            
            logger.debug("Refreshing tool cache from registry...")
            
            try:
                async with httpx.AsyncClient() as client:
                    # Get all active servers
                    response = await client.get(
                        f"{self.registry_url}/servers",
                        params={"status": "active"}
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"Failed to fetch servers: {response.status_code}")
                        return
                    
                    servers = response.json()
                    
                    # Clear existing cache
                    self._tools_cache.clear()
                    self._server_tools.clear()
                    self._capability_index.clear()
                    
                    # Index tools by capability
                    for server in servers:
                        server_id = server["id"]
                        server_name = server["name"]
                        capabilities = server.get("capabilities", {})
                        
                        # Process tools from capabilities
                        tools = capabilities.get("tools", [])
                        self._server_tools[server_id] = []
                        
                        for tool in tools:
                            tool_def = ToolDefinition(
                                name=tool.get("name", "unknown"),
                                description=tool.get("description", ""),
                                server_id=server_id,
                                server_name=server_name,
                                capabilities=tool.get("capabilities", []),
                                parameters=tool.get("parameters", {}),
                                estimated_latency_ms=tool.get("estimated_latency_ms"),
                                cost_per_call=tool.get("cost_per_call"),
                                rate_limit=tool.get("rate_limit")
                            )
                            
                            # Add to cache
                            tool_key = f"{server_id}:{tool_def.name}"
                            self._tools_cache[tool_key] = tool_def
                            self._server_tools[server_id].append(tool_def.name)
                            
                            # Index by capability
                            for cap in tool_def.capabilities:
                                if cap not in self._capability_index:
                                    self._capability_index[cap] = set()
                                self._capability_index[cap].add(tool_key)
                    
                    self._last_cache_update = datetime.utcnow()
                    logger.info(f"Refreshed tool cache: {len(self._tools_cache)} tools from {len(servers)} servers")
            
            except Exception as e:
                logger.error(f"Error refreshing tool cache: {e}")
    
    async def analyze_task(
        self,
        task_description: str,
        explicit_capabilities: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> TaskAnalysis:
        """
        Analyze a task to determine required capabilities.
        
        Args:
            task_description: Natural language task description
            explicit_capabilities: Explicitly required capabilities
            constraints: Additional constraints (max_latency, max_cost, etc.)
            
        Returns:
            TaskAnalysis with inferred and explicit capabilities
        """
        explicit_caps = explicit_capabilities or []
        
        # Infer capabilities from task description
        inferred_caps = self._infer_capabilities(task_description)
        
        # Combine and deduplicate
        all_caps = list(set(explicit_caps + inferred_caps))
        
        # Determine priority order based on task
        priority_order = self._prioritize_capabilities(task_description, all_caps)
        
        return TaskAnalysis(
            task_description=task_description,
            inferred_capabilities=inferred_caps,
            explicit_capabilities=explicit_caps,
            priority_order=priority_order,
            constraints=constraints or {}
        )
    
    async def select_tools(
        self,
        analysis: TaskAnalysis,
        max_tools: int = 10,
        user_id: Optional[str] = None
    ) -> ToolSelection:
        """
        Select appropriate tools based on task analysis.
        
        Args:
            analysis: TaskAnalysis from analyze_task
            max_tools: Maximum number of tools to select
            user_id: Optional user ID for access filtering
            
        Returns:
            ToolSelection with matched tools
        """
        # Ensure cache is fresh
        await self.refresh_tool_cache()
        
        selected_tools: List[ToolDefinition] = []
        matched_capabilities: Set[str] = set()
        
        # Select tools for each required capability
        for capability in analysis.priority_order:
            if capability not in self._capability_index:
                continue
            
            tool_keys = self._capability_index[capability]
            
            for tool_key in tool_keys:
                if len(selected_tools) >= max_tools:
                    break
                
                tool = self._tools_cache.get(tool_key)
                if not tool:
                    continue
                
                # Check constraints
                if not self._check_constraints(tool, analysis.constraints):
                    continue
                
                # Avoid duplicates
                if tool not in selected_tools:
                    selected_tools.append(tool)
                    matched_capabilities.update(tool.capabilities)
        
        # Determine missing capabilities
        required_caps = set(analysis.priority_order)
        missing_caps = list(required_caps - matched_capabilities)
        
        # Calculate confidence
        if required_caps:
            confidence = len(matched_capabilities & required_caps) / len(required_caps)
        else:
            confidence = 1.0 if selected_tools else 0.0
        
        # Generate selection reason
        reason = self._generate_selection_reason(
            selected_tools, matched_capabilities, missing_caps
        )
        
        return ToolSelection(
            selected_tools=selected_tools,
            required_capabilities=analysis.priority_order,
            missing_capabilities=missing_caps,
            selection_reason=reason,
            confidence=confidence
        )
    
    async def select_tools_for_task(
        self,
        task_description: str,
        explicit_capabilities: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        max_tools: int = 10,
        user_id: Optional[str] = None
    ) -> ToolSelection:
        """
        Convenience method to analyze task and select tools in one call.
        
        Args:
            task_description: Natural language task description
            explicit_capabilities: Explicitly required capabilities
            constraints: Additional constraints
            max_tools: Maximum number of tools to select
            user_id: Optional user ID for access filtering
            
        Returns:
            ToolSelection with matched tools
        """
        analysis = await self.analyze_task(
            task_description,
            explicit_capabilities,
            constraints
        )
        return await self.select_tools(analysis, max_tools, user_id)
    
    def _infer_capabilities(self, task_description: str) -> List[str]:
        """Infer capabilities from task description using keyword matching."""
        task_lower = task_description.lower()
        inferred = []
        
        for capability, keywords in self.CAPABILITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in task_lower:
                    inferred.append(capability.value)
                    break
        
        return inferred
    
    def _prioritize_capabilities(
        self,
        task_description: str,
        capabilities: List[str]
    ) -> List[str]:
        """
        Prioritize capabilities based on task context.
        
        Simple heuristic: capabilities mentioned earlier in the task
        or more frequently are higher priority.
        """
        task_lower = task_description.lower()
        
        # Score each capability by position and frequency
        scores: Dict[str, float] = {}
        
        for cap in capabilities:
            # Get keywords for this capability
            cap_enum = None
            for c in Capability:
                if c.value == cap:
                    cap_enum = c
                    break
            
            if cap_enum and cap_enum in self.CAPABILITY_KEYWORDS:
                keywords = self.CAPABILITY_KEYWORDS[cap_enum]
                
                # Find earliest mention
                earliest_pos = len(task_lower)
                mention_count = 0
                
                for keyword in keywords:
                    pos = task_lower.find(keyword)
                    if pos != -1:
                        earliest_pos = min(earliest_pos, pos)
                        mention_count += task_lower.count(keyword)
                
                # Score: lower position = higher score, more mentions = higher score
                scores[cap] = mention_count * 10 - earliest_pos * 0.1
            else:
                scores[cap] = 0
        
        # Sort by score descending
        return sorted(capabilities, key=lambda c: scores.get(c, 0), reverse=True)
    
    def _check_constraints(
        self,
        tool: ToolDefinition,
        constraints: Dict[str, Any]
    ) -> bool:
        """Check if a tool meets the specified constraints."""
        if not constraints:
            return True
        
        # Check max latency
        max_latency = constraints.get("max_latency_ms")
        if max_latency and tool.estimated_latency_ms:
            if tool.estimated_latency_ms > max_latency:
                return False
        
        # Check max cost
        max_cost = constraints.get("max_cost_per_call")
        if max_cost and tool.cost_per_call:
            if tool.cost_per_call > max_cost:
                return False
        
        # Check blocked tools
        blocked = constraints.get("blocked_tools", [])
        if tool.name in blocked:
            return False
        
        # Check required servers
        required_servers = constraints.get("required_servers", [])
        if required_servers and tool.server_id not in required_servers:
            return False
        
        return True
    
    def _generate_selection_reason(
        self,
        selected_tools: List[ToolDefinition],
        matched_caps: Set[str],
        missing_caps: List[str]
    ) -> str:
        """Generate a human-readable reason for the selection."""
        if not selected_tools:
            if missing_caps:
                return f"No tools found for capabilities: {', '.join(missing_caps)}"
            return "No tools required for this task"
        
        tool_summary = ", ".join(t.name for t in selected_tools[:3])
        if len(selected_tools) > 3:
            tool_summary += f" and {len(selected_tools) - 3} more"
        
        reason = f"Selected {len(selected_tools)} tools ({tool_summary}) "
        reason += f"matching {len(matched_caps)} capabilities"
        
        if missing_caps:
            reason += f". Missing: {', '.join(missing_caps)}"
        
        return reason
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the tool selector."""
        return {
            "cached_tools": len(self._tools_cache),
            "indexed_capabilities": len(self._capability_index),
            "servers_indexed": len(self._server_tools),
            "cache_age_seconds": (datetime.utcnow() - self._last_cache_update).total_seconds(),
            "cache_ttl_minutes": self.cache_ttl.total_seconds() / 60
        }
