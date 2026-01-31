"""
Comprehension Service

Core logic for the Comprehension Engine — the first phase of the CORE loop.
Takes raw input and produces a structured ComprehensionResult that feeds
directly into the Task Router (Orchestration phase).

Flow: Raw Input → **Comprehension** → Orchestration → Reasoning → Evaluation

The service:
1. Parses intent using LLM (extract action, entities, urgency, complexity)
2. Searches all three memory tiers for relevant context
3. Matches against agent capabilities
4. Matches against available tools/services
5. Determines if single-agent or multi-agent handling is needed
6. Scores confidence in comprehension
7. Returns structured ComprehensionResult
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any

from app.models.comprehension_models import (
    ComprehensionInput,
    ComprehensionResult,
    IntentAnalysis,
    ContextMatch,
    CapabilityMatch,
    ComplexityScore,
    ComprehensionFeedback,
    SystemCapability,
    SourceType,
    ActionType,
    UrgencyLevel,
    HandlingMode,
    ComprehensionStatus,
    ExtractedEntity,
    MemoryMatch,
    MatchedCapability,
)
from app.models.task_models import TaskType
from app.repository import comprehension_repository
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


# =============================================================================
# INTENT PARSING PROMPT
# =============================================================================

INTENT_PARSE_SYSTEM_PROMPT = """You are an intent analysis module for the CORE multi-agent orchestration system.

Given a user input, analyze it and return a JSON object with:
{
  "action_type": one of ["query", "command", "analysis", "creation", "monitoring", "conversation", "planning", "unknown"],
  "summary": "brief summary of what is being requested",
  "entities": [{"name": "entity_value", "entity_type": "type", "confidence": 0.9}],
  "urgency": one of ["critical", "high", "medium", "low", "none"],
  "confidence": 0.0-1.0 confidence in your analysis,
  "keywords": ["key", "terms"],
  "requires_clarification": true/false,
  "clarification_questions": ["question if ambiguous"],
  "complexity": {
    "overall": 0.0-1.0,
    "reasoning_depth": 0.0-1.0,
    "breadth": 0.0-1.0,
    "novelty": 0.0-1.0,
    "estimated_duration_seconds": null or integer
  },
  "suggested_task_type": one of ["research", "code", "analysis", "monitoring", "writing", "communication", "planning", "evaluation", "general"],
  "suggested_priority": 1-10 integer,
  "suggested_capabilities": ["capability_name"]
}

Rules:
- Be precise about action_type classification
- Extract all meaningful entities (people, topics, tools, technologies)
- Assess urgency based on language cues (ASAP, urgent, whenever, etc.)
- Estimate complexity honestly — simple questions are low, multi-step tasks are high
- suggested_capabilities should reflect what skills/tools would help

Return ONLY valid JSON, no markdown fencing."""


# =============================================================================
# COMPREHENSION SERVICE
# =============================================================================

class ComprehensionService:
    """
    Central comprehension service for the CORE system.

    Analyzes raw input to produce structured ComprehensionResult objects
    that feed into the Task Router for orchestration.
    """

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the comprehension service."""
        self._initialized = True
        logger.info("ComprehensionService initialized")

    # -------------------------------------------------------------------------
    # MAIN COMPREHENSION FLOW
    # -------------------------------------------------------------------------

    async def comprehend(self, input_data: ComprehensionInput) -> ComprehensionResult:
        """
        Main comprehension flow: analyze raw input and produce a structured result.

        Steps:
        1. Parse intent using LLM
        2. Search memory tiers for relevant context
        3. Match capabilities against available agents
        4. Determine handling mode (single/multi-agent)
        5. Score overall confidence
        6. Store result and return

        Args:
            input_data: Raw input to comprehend

        Returns:
            ComprehensionResult ready for the Task Router
        """
        start_time = time.time()

        try:
            # Step 1: Parse intent
            intent, complexity, suggested_task_type, suggested_priority, suggested_capabilities = (
                await self._parse_intent(input_data)
            )

            # Step 2: Search memory for context
            context = await self._search_memory_context(input_data, intent)

            # Step 3: Match capabilities
            capabilities = await self._match_capabilities(intent, suggested_capabilities)

            # Step 4: Determine handling mode
            handling_mode = self._determine_handling_mode(intent, complexity, capabilities)

            # Step 5: Adjust priority based on context
            adjusted_priority = self._adjust_priority(suggested_priority, intent, context)

            # Step 6: Score overall confidence
            overall_confidence = self._score_confidence(intent, context, capabilities)

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Build result
            result = ComprehensionResult(
                input=input_data,
                intent=intent,
                context=context,
                capabilities=capabilities,
                complexity=complexity,
                suggested_task_type=suggested_task_type,
                suggested_priority=adjusted_priority,
                handling_mode=handling_mode,
                suggested_capabilities=suggested_capabilities,
                confidence=overall_confidence,
                status=ComprehensionStatus.COMPLETED,
                processing_time_ms=processing_time_ms,
            )

            # Step 7: Store result for analytics
            try:
                await comprehension_repository.store_comprehension_result(result)
            except Exception as store_err:
                logger.warning(f"Failed to store comprehension result: {store_err}")

            logger.info(
                f"Comprehension completed in {processing_time_ms}ms — "
                f"action={intent.action_type.value}, confidence={overall_confidence:.2f}, "
                f"handling={handling_mode.value}"
            )

            return result

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Comprehension failed: {e}")

            # Return a partial/failed result
            return ComprehensionResult(
                input=input_data,
                intent=IntentAnalysis(
                    action_type=ActionType.UNKNOWN,
                    summary=f"Comprehension failed: {str(e)}",
                    confidence=0.0,
                ),
                confidence=0.0,
                status=ComprehensionStatus.FAILED,
                processing_time_ms=processing_time_ms,
            )

    # -------------------------------------------------------------------------
    # STEP 1: INTENT PARSING
    # -------------------------------------------------------------------------

    async def _parse_intent(
        self, input_data: ComprehensionInput
    ) -> tuple:
        """
        Parse intent using LLM.

        Returns tuple of:
            (IntentAnalysis, ComplexityScore, suggested_task_type, suggested_priority, suggested_capabilities)
        """
        try:
            llm_result = await self._call_llm_for_intent(input_data.content, input_data.source_type.value)
            return self._build_intent_from_llm(llm_result)
        except Exception as e:
            logger.warning(f"LLM intent parsing failed, falling back to heuristic: {e}")
            return self._heuristic_intent_parse(input_data)

    async def _call_llm_for_intent(self, content: str, source_type: str) -> Dict[str, Any]:
        """Call LLM for intent analysis."""
        try:
            from app.dependencies import get_ollama_client

            client = get_ollama_client()
            user_prompt = f"Source type: {source_type}\n\nInput:\n{content}"

            response = await client.chat.completions.create(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": INTENT_PARSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )

            raw_text = response.choices[0].message.content.strip()

            # Try to parse JSON, handling potential markdown fencing
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                # Remove first and last lines (fences)
                raw_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            return json.loads(raw_text)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            raise
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            raise

    def _build_intent_from_llm(self, llm_result: Dict[str, Any]) -> tuple:
        """Build structured objects from LLM JSON response."""
        # Parse action type
        action_type_str = llm_result.get("action_type", "unknown")
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.UNKNOWN

        # Parse urgency
        urgency_str = llm_result.get("urgency", "medium")
        try:
            urgency = UrgencyLevel(urgency_str)
        except ValueError:
            urgency = UrgencyLevel.MEDIUM

        # Parse entities
        entities = []
        for e in llm_result.get("entities", []):
            if isinstance(e, dict):
                entities.append(ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=e.get("entity_type", "unknown"),
                    confidence=min(1.0, max(0.0, float(e.get("confidence", 0.5)))),
                ))

        # Build IntentAnalysis
        intent = IntentAnalysis(
            action_type=action_type,
            summary=llm_result.get("summary", ""),
            entities=entities,
            urgency=urgency,
            confidence=min(1.0, max(0.0, float(llm_result.get("confidence", 0.5)))),
            keywords=llm_result.get("keywords", []),
            requires_clarification=llm_result.get("requires_clarification", False),
            clarification_questions=llm_result.get("clarification_questions", []),
        )

        # Build ComplexityScore
        complexity_raw = llm_result.get("complexity", {})
        complexity = ComplexityScore(
            overall=min(1.0, max(0.0, float(complexity_raw.get("overall", 0.5)))),
            reasoning_depth=min(1.0, max(0.0, float(complexity_raw.get("reasoning_depth", 0.5)))),
            breadth=min(1.0, max(0.0, float(complexity_raw.get("breadth", 0.5)))),
            novelty=min(1.0, max(0.0, float(complexity_raw.get("novelty", 0.5)))),
            estimated_duration_seconds=complexity_raw.get("estimated_duration_seconds"),
        )

        # Suggested task type
        suggested_task_type = llm_result.get("suggested_task_type", "general")
        valid_task_types = [
            TaskType.RESEARCH, TaskType.CODE, TaskType.ANALYSIS,
            TaskType.MONITORING, TaskType.WRITING, TaskType.COMMUNICATION,
            TaskType.PLANNING, TaskType.EVALUATION, "general",
        ]
        if suggested_task_type not in valid_task_types:
            suggested_task_type = "general"

        # Suggested priority
        try:
            suggested_priority = max(1, min(10, int(llm_result.get("suggested_priority", 5))))
        except (ValueError, TypeError):
            suggested_priority = 5

        # Suggested capabilities
        suggested_capabilities = llm_result.get("suggested_capabilities", [])
        if not isinstance(suggested_capabilities, list):
            suggested_capabilities = []

        return intent, complexity, suggested_task_type, suggested_priority, suggested_capabilities

    def _heuristic_intent_parse(self, input_data: ComprehensionInput) -> tuple:
        """
        Fallback heuristic intent parsing when LLM is unavailable.

        Uses keyword matching and pattern recognition.
        """
        content_lower = input_data.content.lower()

        # Classify action type
        if any(w in content_lower for w in ["create", "build", "make", "generate", "write"]):
            action_type = ActionType.CREATION
            task_type = TaskType.WRITING
        elif any(w in content_lower for w in ["analyze", "evaluate", "assess", "compare"]):
            action_type = ActionType.ANALYSIS
            task_type = TaskType.ANALYSIS
        elif any(w in content_lower for w in ["find", "search", "what", "who", "when", "where", "how", "why", "?"]):
            action_type = ActionType.QUERY
            task_type = TaskType.RESEARCH
        elif any(w in content_lower for w in ["run", "execute", "deploy", "start", "stop", "restart"]):
            action_type = ActionType.COMMAND
            task_type = TaskType.CODE
        elif any(w in content_lower for w in ["monitor", "watch", "track", "observe"]):
            action_type = ActionType.MONITORING
            task_type = TaskType.MONITORING
        elif any(w in content_lower for w in ["plan", "schedule", "roadmap", "strategy"]):
            action_type = ActionType.PLANNING
            task_type = TaskType.PLANNING
        else:
            action_type = ActionType.CONVERSATION
            task_type = "general"

        # Assess urgency
        if any(w in content_lower for w in ["urgent", "asap", "immediately", "critical", "emergency"]):
            urgency = UrgencyLevel.CRITICAL
            priority = 9
        elif any(w in content_lower for w in ["important", "soon", "priority", "fast"]):
            urgency = UrgencyLevel.HIGH
            priority = 7
        elif any(w in content_lower for w in ["whenever", "low priority", "no rush", "when you can"]):
            urgency = UrgencyLevel.LOW
            priority = 3
        else:
            urgency = UrgencyLevel.MEDIUM
            priority = 5

        # Extract simple keywords (words > 3 chars, no stop words)
        stop_words = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her",
                       "was", "one", "our", "out", "has", "have", "from", "this", "that", "with",
                       "they", "been", "said", "each", "which", "their", "will", "other", "about",
                       "many", "then", "them", "these", "some", "would", "make", "like", "just",
                       "over", "such", "take", "than", "into", "could", "what", "there", "please"}
        words = content_lower.split()
        keywords = [w.strip(".,!?;:'\"") for w in words if len(w) > 3 and w not in stop_words][:10]

        # Estimate complexity based on length and structure
        word_count = len(words)
        complexity_overall = min(1.0, word_count / 100.0)

        intent = IntentAnalysis(
            action_type=action_type,
            summary=input_data.content[:200],
            entities=[],
            urgency=urgency,
            confidence=0.4,  # Lower confidence for heuristic
            keywords=keywords,
            requires_clarification=action_type == ActionType.UNKNOWN,
        )

        complexity = ComplexityScore(
            overall=complexity_overall,
            reasoning_depth=min(1.0, complexity_overall * 0.8),
            breadth=min(1.0, complexity_overall * 0.6),
            novelty=0.5,  # Can't assess novelty without context
        )

        return intent, complexity, task_type, priority, keywords[:5]

    # -------------------------------------------------------------------------
    # STEP 2: MEMORY CONTEXT SEARCH
    # -------------------------------------------------------------------------

    async def _search_memory_context(
        self, input_data: ComprehensionInput, intent: IntentAnalysis
    ) -> ContextMatch:
        """Search all three memory tiers for relevant context."""
        try:
            # Generate embedding for the input
            query_text = f"{intent.summary} {' '.join(intent.keywords)}"
            query_embedding = await embedding_service.generate_embedding(query_text)

            if not query_embedding or all(v == 0.0 for v in query_embedding):
                logger.warning("Failed to generate embedding, skipping memory search")
                return ContextMatch()

            # Import memory repository and search all tiers
            from app.repository.memory_repository import get_relevant_context

            agent_id = input_data.source_id if input_data.source_type == SourceType.AGENT else None

            raw_context = await get_relevant_context(
                query_embedding=query_embedding,
                agent_id=agent_id,
                limit_per_tier=5,
                threshold=0.6,
            )

            # Convert to ContextMatch model
            semantic_matches = []
            for mem in raw_context.get("semantic", []):
                semantic_matches.append(MemoryMatch(
                    memory_id=mem.id,
                    content=mem.content[:500],
                    similarity=mem.metadata.get("similarity", 0.0),
                    memory_tier="semantic",
                    metadata=mem.metadata,
                ))

            episodic_matches = []
            for mem in raw_context.get("episodic", []):
                episodic_matches.append(MemoryMatch(
                    memory_id=mem.id,
                    content=mem.content[:500],
                    similarity=mem.metadata.get("similarity", 0.0),
                    memory_tier="episodic",
                    metadata=mem.metadata,
                ))

            procedural_matches = []
            for mem in raw_context.get("procedural", []):
                procedural_matches.append(MemoryMatch(
                    memory_id=mem.id,
                    content=mem.content[:500],
                    similarity=mem.metadata.get("similarity", 0.0),
                    memory_tier="procedural",
                    metadata=mem.metadata,
                ))

            all_scores = (
                [m.similarity for m in semantic_matches] +
                [m.similarity for m in episodic_matches] +
                [m.similarity for m in procedural_matches]
            )
            best_score = max(all_scores) if all_scores else 0.0
            has_relevant = best_score > 0.6

            return ContextMatch(
                semantic_matches=semantic_matches,
                episodic_matches=episodic_matches,
                procedural_matches=procedural_matches,
                has_relevant_context=has_relevant,
                best_match_score=best_score,
            )

        except Exception as e:
            logger.warning(f"Memory context search failed: {e}")
            return ContextMatch()

    # -------------------------------------------------------------------------
    # STEP 3: CAPABILITY MATCHING
    # -------------------------------------------------------------------------

    async def _match_capabilities(
        self, intent: IntentAnalysis, suggested_capabilities: List[str]
    ) -> CapabilityMatch:
        """Match intent against available agent capabilities."""
        try:
            from app.dependencies import get_db_pool

            pool = await get_db_pool()

            # Query agents with their capabilities
            query = """
                SELECT id, agent_id, agent_role, capabilities, status
                FROM agent_instances
                WHERE status = 'ready'
            """

            async with pool.acquire() as conn:
                rows = await conn.fetch(query)

            if not rows:
                return CapabilityMatch()

            # Score each agent against the intent
            matched_capabilities: List[MatchedCapability] = []
            matched_agent_ids: List[str] = []
            best_agent_id: Optional[str] = None
            best_score = 0.0

            # Build capability keywords from intent
            intent_keywords = set(
                [kw.lower() for kw in intent.keywords] +
                [cap.lower() for cap in suggested_capabilities] +
                [intent.action_type.value]
            )

            for row in rows:
                agent_id = row["agent_id"]
                capabilities_raw = row["capabilities"]
                
                # Parse capabilities - could be JSON string or list
                if isinstance(capabilities_raw, str):
                    try:
                        agent_caps = json.loads(capabilities_raw)
                    except (json.JSONDecodeError, TypeError):
                        agent_caps = []
                elif isinstance(capabilities_raw, list):
                    agent_caps = capabilities_raw
                else:
                    agent_caps = []

                if not agent_caps:
                    continue

                # Score capability match
                cap_names = set()
                for cap in agent_caps:
                    if isinstance(cap, str):
                        cap_names.add(cap.lower())
                    elif isinstance(cap, dict):
                        cap_names.add(cap.get("name", "").lower())

                # Calculate overlap score
                if cap_names:
                    overlap = intent_keywords.intersection(cap_names)
                    match_score = len(overlap) / max(len(intent_keywords), 1)
                    
                    # Boost for action type alignment
                    role = (row["agent_role"] or "").lower()
                    action_role_map = {
                        ActionType.QUERY: ["researcher", "analyst"],
                        ActionType.ANALYSIS: ["analyst", "researcher"],
                        ActionType.CREATION: ["writer", "coder", "creator"],
                        ActionType.COMMAND: ["coder", "monitor"],
                        ActionType.MONITORING: ["monitor"],
                        ActionType.PLANNING: ["planner", "researcher"],
                    }
                    aligned_roles = action_role_map.get(intent.action_type, [])
                    if role in aligned_roles:
                        match_score = min(1.0, match_score + 0.3)

                    if match_score > 0.1:
                        matched_agent_ids.append(agent_id)
                        for cap in agent_caps:
                            cap_name = cap if isinstance(cap, str) else cap.get("name", "")
                            cap_desc = "" if isinstance(cap, str) else cap.get("description", "")
                            matched_capabilities.append(MatchedCapability(
                                capability_name=cap_name,
                                description=cap_desc,
                                match_score=match_score,
                                source_type="agent",
                                source_id=agent_id,
                            ))

                        if match_score > best_score:
                            best_score = match_score
                            best_agent_id = agent_id

            return CapabilityMatch(
                matched_capabilities=matched_capabilities,
                matched_agent_ids=matched_agent_ids,
                matched_tools=[],  # TODO: integrate tool registry
                has_capable_agents=len(matched_agent_ids) > 0,
                best_agent_id=best_agent_id,
                best_agent_score=best_score,
            )

        except Exception as e:
            logger.warning(f"Capability matching failed: {e}")
            return CapabilityMatch()

    # -------------------------------------------------------------------------
    # STEP 4: HANDLING MODE DETERMINATION
    # -------------------------------------------------------------------------

    def _determine_handling_mode(
        self,
        intent: IntentAnalysis,
        complexity: ComplexityScore,
        capabilities: CapabilityMatch,
    ) -> HandlingMode:
        """Determine whether single-agent, multi-agent, or no-agent handling is needed."""
        # If no agents can handle it, may need human
        if not capabilities.has_capable_agents:
            if intent.action_type == ActionType.CONVERSATION:
                return HandlingMode.SINGLE_AGENT  # Conversation can use any agent
            return HandlingMode.HUMAN_REQUIRED

        # Simple queries with high-confidence context might not need an agent
        if (
            intent.action_type == ActionType.QUERY
            and intent.confidence > 0.8
            and complexity.overall < 0.2
        ):
            return HandlingMode.NO_AGENT

        # High complexity or broad tasks need multi-agent
        if complexity.overall > 0.7 and complexity.breadth > 0.6:
            return HandlingMode.MULTI_AGENT

        # Multiple distinct capability areas
        if len(capabilities.matched_agent_ids) > 2 and complexity.breadth > 0.5:
            return HandlingMode.MULTI_AGENT

        return HandlingMode.SINGLE_AGENT

    # -------------------------------------------------------------------------
    # STEP 5: PRIORITY ADJUSTMENT
    # -------------------------------------------------------------------------

    def _adjust_priority(
        self,
        base_priority: int,
        intent: IntentAnalysis,
        context: ContextMatch,
    ) -> int:
        """Adjust priority based on urgency and context."""
        priority = base_priority

        # Urgency boost
        urgency_boost = {
            UrgencyLevel.CRITICAL: 3,
            UrgencyLevel.HIGH: 1,
            UrgencyLevel.MEDIUM: 0,
            UrgencyLevel.LOW: -1,
            UrgencyLevel.NONE: -2,
        }
        priority += urgency_boost.get(intent.urgency, 0)

        # If we have strong context, slightly lower priority (less uncertain)
        if context.has_relevant_context and context.best_match_score > 0.8:
            priority -= 1

        # Clamp to valid range
        return max(1, min(10, priority))

    # -------------------------------------------------------------------------
    # STEP 6: CONFIDENCE SCORING
    # -------------------------------------------------------------------------

    def _score_confidence(
        self,
        intent: IntentAnalysis,
        context: ContextMatch,
        capabilities: CapabilityMatch,
    ) -> float:
        """Score overall comprehension confidence."""
        # Weighted average of component confidences
        intent_weight = 0.5
        context_weight = 0.25
        capability_weight = 0.25

        intent_score = intent.confidence

        # Context score: higher if relevant context found
        context_score = 0.3  # Base score
        if context.has_relevant_context:
            context_score = min(1.0, context.best_match_score)

        # Capability score: higher if capable agents found
        capability_score = 0.3  # Base score
        if capabilities.has_capable_agents:
            capability_score = min(1.0, capabilities.best_agent_score + 0.3)

        overall = (
            intent_score * intent_weight
            + context_score * context_weight
            + capability_score * capability_weight
        )

        return min(1.0, max(0.0, overall))

    # -------------------------------------------------------------------------
    # PUBLIC UTILITY METHODS
    # -------------------------------------------------------------------------

    async def get_capability_registry(self) -> List[SystemCapability]:
        """
        Discover what the system can do — all agents, tools, and services.

        Returns a flat list of SystemCapability objects.
        """
        capabilities: List[SystemCapability] = []

        try:
            from app.dependencies import get_db_pool

            pool = await get_db_pool()

            # Get agent capabilities
            agent_query = """
                SELECT agent_id, agent_role, capabilities, status
                FROM agent_instances
                WHERE status IN ('ready', 'busy')
            """

            async with pool.acquire() as conn:
                rows = await conn.fetch(agent_query)

            for row in rows:
                agent_id = row["agent_id"]
                agent_role = row["agent_role"] or ""
                is_available = row["status"] == "ready"
                
                caps_raw = row["capabilities"]
                if isinstance(caps_raw, str):
                    try:
                        caps_list = json.loads(caps_raw)
                    except (json.JSONDecodeError, TypeError):
                        caps_list = []
                elif isinstance(caps_raw, list):
                    caps_list = caps_raw
                else:
                    caps_list = []

                for cap in caps_list:
                    if isinstance(cap, str):
                        capabilities.append(SystemCapability(
                            name=cap,
                            provider_type="agent",
                            provider_id=agent_id,
                            provider_name=agent_role,
                            is_available=is_available,
                            tags=["agent", agent_role],
                        ))
                    elif isinstance(cap, dict):
                        capabilities.append(SystemCapability(
                            name=cap.get("name", "unknown"),
                            description=cap.get("description", ""),
                            provider_type="agent",
                            provider_id=agent_id,
                            provider_name=agent_role,
                            is_available=is_available,
                            tags=["agent", agent_role],
                        ))

        except Exception as e:
            logger.warning(f"Failed to query agent capabilities: {e}")

        # Add well-known system capabilities
        system_caps = [
            SystemCapability(
                name="memory_search",
                description="Search across semantic, episodic, and procedural memory tiers",
                provider_type="service",
                provider_id="memory_service",
                provider_name="LangMem Memory Service",
                is_available=True,
                tags=["memory", "search", "knowledge"],
            ),
            SystemCapability(
                name="embedding_generation",
                description="Generate vector embeddings for semantic similarity search",
                provider_type="service",
                provider_id="embedding_service",
                provider_name="Embedding Service (Ollama)",
                is_available=True,
                tags=["embedding", "search", "similarity"],
            ),
            SystemCapability(
                name="task_routing",
                description="Route tasks to the best available agent",
                provider_type="service",
                provider_id="task_router",
                provider_name="Task Router",
                is_available=True,
                tags=["routing", "orchestration"],
            ),
        ]
        capabilities.extend(system_caps)

        return capabilities

    async def analyze_complexity(self, content: str) -> ComplexityScore:
        """
        Estimate task complexity for routing.

        Standalone method that can be used independently of full comprehension.
        """
        word_count = len(content.split())

        # Heuristic complexity estimation
        overall = min(1.0, word_count / 100.0)

        # Check for multi-step indicators
        multi_step_words = ["then", "after", "next", "also", "additionally", "furthermore", "step"]
        step_count = sum(1 for w in content.lower().split() if w in multi_step_words)
        breadth = min(1.0, step_count * 0.15 + 0.2)

        # Check for analytical depth indicators
        depth_words = ["analyze", "evaluate", "compare", "explain", "why", "how", "reasoning"]
        depth_count = sum(1 for w in content.lower().split() if w in depth_words)
        reasoning_depth = min(1.0, depth_count * 0.2 + 0.2)

        # Novelty is hard without context
        novelty = 0.5

        return ComplexityScore(
            overall=overall,
            reasoning_depth=reasoning_depth,
            breadth=breadth,
            novelty=novelty,
            estimated_duration_seconds=max(5, int(word_count * 0.5)),
        )

    async def submit_feedback(self, feedback: ComprehensionFeedback) -> bool:
        """Submit feedback on a comprehension result for the learning loop."""
        return await comprehension_repository.store_feedback(feedback)

    async def get_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehension analytics."""
        return await comprehension_repository.get_comprehension_analytics(days=days)

    async def health_check(self) -> Dict[str, Any]:
        """Check comprehension service health."""
        embedding_ok = False
        try:
            embedding_ok = await embedding_service.health_check()
        except Exception:
            pass

        return {
            "initialized": self._initialized,
            "embedding_service": embedding_ok,
        }


# Global comprehension service instance
comprehension_service = ComprehensionService()
