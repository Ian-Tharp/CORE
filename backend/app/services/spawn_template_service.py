"""
Agent Spawn Template Service

Manages reusable agent spawn templates — create, list, update, delete,
and spawn agents from templates with optional overrides.

Templates are stored in-memory with a set of built-in defaults loaded
at startup.  Future: persist to PostgreSQL for durability across restarts.

Architecture:
    SpawnTemplateService → AgentCreateRequest → Agent Repository → DB
    Templates simplify the "spawn a specialist agent" workflow by
    pre-configuring role, personality, tools, and system prompt.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.models.agent_models import AgentCapability, AgentConfig
from app.models.spawn_template_models import (
    SpawnFromTemplateRequest,
    SpawnResult,
    SpawnTemplate,
    SpawnTemplateCreate,
    SpawnTemplateUpdate,
)
from app.repository.agent_repository import create_agent

logger = logging.getLogger(__name__)


# =============================================================================
# BUILT-IN TEMPLATES
# =============================================================================

_BUILTIN_TEMPLATES: List[SpawnTemplate] = [
    SpawnTemplate(
        id="tmpl-researcher",
        name="Researcher",
        description=(
            "Deep research and information synthesis specialist. "
            "Excels at finding, analyzing, and summarizing information "
            "from multiple sources."
        ),
        role="researcher",
        agent_type="task_agent",
        system_prompt=(
            "You are a meticulous research specialist within the CORE platform. "
            "Your primary function is to gather, analyze, and synthesize "
            "information with precision and thoroughness. When given a research "
            "task, you:\n"
            "1. Break the question into sub-questions\n"
            "2. Search available sources systematically\n"
            "3. Cross-reference findings for accuracy\n"
            "4. Synthesize a clear, well-cited summary\n"
            "5. Flag uncertainties and knowledge gaps\n\n"
            "Always distinguish between established facts and inferences. "
            "Cite sources when available. Prefer depth over breadth."
        ),
        personality_traits={
            "curiosity": 0.95,
            "technical_precision": 0.85,
            "creativity": 0.5,
            "uncertainty": 0.6,
        },
        capabilities=[
            AgentCapability(
                name="deep_research",
                description="Systematic multi-source research and analysis",
            ),
            AgentCapability(
                name="information_synthesis",
                description="Combine findings into coherent summaries",
            ),
        ],
        interests=["research", "analysis", "fact-checking", "knowledge"],
        tags=["research", "analysis", "information", "core-builtin"],
        is_builtin=True,
        author="CORE",
    ),
    SpawnTemplate(
        id="tmpl-analyst",
        name="Analyst",
        description=(
            "Data-driven analytical agent. Evaluates evidence, identifies "
            "patterns, and produces structured assessments."
        ),
        role="analyst",
        agent_type="task_agent",
        system_prompt=(
            "You are an analytical agent within the CORE platform. "
            "You approach every problem with structured reasoning and "
            "evidence-based evaluation. Your workflow:\n"
            "1. Identify the core question and success criteria\n"
            "2. Gather relevant data points\n"
            "3. Apply appropriate analytical frameworks\n"
            "4. Present findings with confidence levels\n"
            "5. Recommend concrete next steps\n\n"
            "Be quantitative when possible. Use tables, comparisons, and "
            "metrics. Always state your assumptions explicitly."
        ),
        personality_traits={
            "curiosity": 0.7,
            "technical_precision": 0.95,
            "creativity": 0.4,
            "uncertainty": 0.5,
        },
        capabilities=[
            AgentCapability(
                name="pattern_recognition",
                description="Identify trends and patterns in data",
            ),
            AgentCapability(
                name="structured_evaluation",
                description="Apply frameworks for systematic assessment",
            ),
        ],
        interests=["analysis", "data", "metrics", "evaluation", "patterns"],
        tags=["analysis", "evaluation", "data", "core-builtin"],
        is_builtin=True,
        author="CORE",
    ),
    SpawnTemplate(
        id="tmpl-coordinator",
        name="Coordinator",
        description=(
            "Multi-agent coordination specialist. Manages task delegation, "
            "tracks progress, and synthesizes results from other agents."
        ),
        role="coordinator",
        agent_type="task_agent",
        system_prompt=(
            "You are a coordination agent within the CORE platform. "
            "Your role is to orchestrate work across multiple agents, "
            "ensuring efficient task delegation and result synthesis. "
            "Your approach:\n"
            "1. Decompose complex tasks into parallelizable sub-tasks\n"
            "2. Match sub-tasks to the best-suited agents\n"
            "3. Monitor progress and handle blockers\n"
            "4. Merge results into a coherent deliverable\n"
            "5. Identify gaps and request follow-up work\n\n"
            "Communicate clearly. Keep status updates concise. "
            "Prioritize unblocking other agents over doing work yourself."
        ),
        personality_traits={
            "curiosity": 0.6,
            "technical_precision": 0.7,
            "creativity": 0.5,
            "uncertainty": 0.3,
        },
        capabilities=[
            AgentCapability(
                name="task_delegation",
                description="Break work into sub-tasks and assign to agents",
            ),
            AgentCapability(
                name="progress_tracking",
                description="Monitor and report on multi-agent workflows",
            ),
        ],
        interests=["coordination", "orchestration", "project-management"],
        tags=["coordination", "management", "orchestration", "core-builtin"],
        is_builtin=True,
        author="CORE",
    ),
    SpawnTemplate(
        id="tmpl-writer",
        name="Writer",
        description=(
            "Creative and technical writing specialist. Produces clear, "
            "engaging content from documentation to narratives."
        ),
        role="writer",
        agent_type="task_agent",
        system_prompt=(
            "You are a writing specialist within the CORE platform. "
            "You craft clear, engaging, and well-structured content for "
            "a variety of purposes — technical documentation, reports, "
            "creative pieces, and communication drafts. Your principles:\n"
            "1. Audience-first — adapt tone and complexity to the reader\n"
            "2. Structure matters — use headings, lists, and logical flow\n"
            "3. Concise over verbose — every sentence earns its place\n"
            "4. Show, don't tell — use examples and specifics\n"
            "5. Iterate — first drafts are starting points, not endpoints\n\n"
            "Ask clarifying questions about audience and purpose before "
            "writing when they aren't clear."
        ),
        personality_traits={
            "curiosity": 0.7,
            "technical_precision": 0.6,
            "creativity": 0.9,
            "uncertainty": 0.4,
        },
        capabilities=[
            AgentCapability(
                name="technical_writing",
                description="Produce clear technical documentation and guides",
            ),
            AgentCapability(
                name="creative_writing",
                description="Craft engaging narratives and creative content",
            ),
        ],
        interests=["writing", "documentation", "communication", "storytelling"],
        tags=["writing", "content", "documentation", "core-builtin"],
        is_builtin=True,
        author="CORE",
    ),
]


# =============================================================================
# SERVICE
# =============================================================================


class SpawnTemplateService:
    """
    Manages spawn templates and creates agents from them.

    In-memory store seeded with built-in templates.  Custom templates
    can be added, updated, and removed at runtime.

    Thread-safety note: This is a single-threaded async service within
    a FastAPI application; no locking is required for the in-memory dict.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, SpawnTemplate] = {}
        self._load_builtins()

    # ── Bootstrap ────────────────────────────────────────────────────────

    def _load_builtins(self) -> None:
        for tmpl in _BUILTIN_TEMPLATES:
            self._templates[tmpl.id] = tmpl.model_copy()
        logger.info(
            "SpawnTemplateService loaded %d built-in templates",
            len(_BUILTIN_TEMPLATES),
        )

    # ── CRUD ─────────────────────────────────────────────────────────────

    def create_template(self, req: SpawnTemplateCreate) -> SpawnTemplate:
        """Create a new custom template. Returns the created template."""
        tmpl = SpawnTemplate(
            name=req.name,
            description=req.description,
            role=req.role,
            agent_type=req.agent_type,
            system_prompt=req.system_prompt,
            personality_traits=req.personality_traits,
            capabilities=req.capabilities,
            interests=req.interests,
            mcp_servers=req.mcp_servers,
            custom_tools=req.custom_tools,
            tags=req.tags,
            version=req.version,
            author=req.author,
            is_builtin=False,
        )
        self._templates[tmpl.id] = tmpl
        logger.info("Created spawn template: %s (%s)", tmpl.id, tmpl.name)
        return tmpl

    def get_template(self, template_id: str) -> Optional[SpawnTemplate]:
        return self._templates.get(template_id)

    def list_templates(
        self,
        *,
        role: Optional[str] = None,
        tag: Optional[str] = None,
        builtin_only: bool = False,
    ) -> List[SpawnTemplate]:
        """List templates with optional filters."""
        results: List[SpawnTemplate] = []
        for tmpl in self._templates.values():
            if role and tmpl.role != role:
                continue
            if tag and tag not in tmpl.tags:
                continue
            if builtin_only and not tmpl.is_builtin:
                continue
            results.append(tmpl)
        return results

    def update_template(
        self, template_id: str, patch: SpawnTemplateUpdate
    ) -> Optional[SpawnTemplate]:
        """
        Apply a partial update to an existing template.

        Built-in templates can be updated (changes are in-memory only and
        reset on restart).  Returns the updated template or None if not found.
        """
        tmpl = self._templates.get(template_id)
        if tmpl is None:
            return None

        update_data = patch.model_dump(exclude_unset=True)
        if not update_data:
            return tmpl  # nothing to update

        current = tmpl.model_dump()
        current.update(update_data)
        current["updated_at"] = datetime.now(timezone.utc)

        updated = SpawnTemplate(**current)
        self._templates[template_id] = updated
        logger.info("Updated spawn template: %s", template_id)
        return updated

    def delete_template(self, template_id: str) -> bool:
        """Delete a template. Returns True if it existed."""
        tmpl = self._templates.get(template_id)
        if tmpl is None:
            return False
        if tmpl.is_builtin:
            logger.warning(
                "Deleting built-in template %s — will reappear on restart",
                template_id,
            )
        del self._templates[template_id]
        logger.info("Deleted spawn template: %s", template_id)
        return True

    # ── Spawn ────────────────────────────────────────────────────────────

    async def spawn_from_template(
        self, template_id: str, req: SpawnFromTemplateRequest
    ) -> SpawnResult:
        """
        Create a new agent from a template with optional overrides.

        Steps:
            1. Load the template
            2. Build an AgentCreateRequest, merging template + overrides
            3. Persist via agent_repository.create_agent()
            4. Return a SpawnResult summary

        Raises:
            ValueError: If template not found
        """
        tmpl = self._templates.get(template_id)
        if tmpl is None:
            raise ValueError(f"Template '{template_id}' not found")

        # Build system prompt (template + optional append)
        system_prompt = tmpl.system_prompt
        if req.system_prompt_append:
            system_prompt = f"{system_prompt}\n\n{req.system_prompt_append}"

        # Merge personality traits
        traits = dict(tmpl.personality_traits)
        if req.personality_overrides:
            traits.update(req.personality_overrides)

        # Merge interests
        interests = list(tmpl.interests)
        if req.extra_interests:
            interests.extend(req.extra_interests)

        # Merge MCP servers
        mcp_servers = list(tmpl.mcp_servers)
        if req.extra_mcp_servers:
            mcp_servers.extend(req.extra_mcp_servers)

        # Determine display name
        short_id = (
            req.agent_id.split("-")[-1][:8] if "-" in req.agent_id else req.agent_id[:8]
        )
        agent_name = req.agent_name or f"{tmpl.name}-{short_id}"

        agent_config = AgentConfig(
            agent_id=req.agent_id,
            agent_name=agent_name,
            agent_type=tmpl.agent_type,
            display_name=f"{agent_name} (from {tmpl.name} template)",
            description=tmpl.description,
            system_prompt=system_prompt,
            personality_traits=traits,
            capabilities=tmpl.capabilities,
            interests=interests,
            mcp_servers=mcp_servers,
            custom_tools=tmpl.custom_tools,
            is_active=req.is_active,
            current_status="offline",
            version=tmpl.version,
            author=tmpl.author,
        )

        # Persist to database
        await create_agent(agent_config)

        logger.info(
            "Spawned agent '%s' from template '%s' (%s)",
            req.agent_id,
            template_id,
            tmpl.name,
        )

        return SpawnResult(
            agent_id=req.agent_id,
            agent_name=agent_name,
            template_id=template_id,
            template_name=tmpl.name,
            agent_type=tmpl.agent_type,
            role=tmpl.role,
        )


# =============================================================================
# SINGLETON
# =============================================================================

_service: Optional[SpawnTemplateService] = None


def get_spawn_template_service() -> SpawnTemplateService:
    """Return the global SpawnTemplateService singleton."""
    global _service
    if _service is None:
        _service = SpawnTemplateService()
    return _service
