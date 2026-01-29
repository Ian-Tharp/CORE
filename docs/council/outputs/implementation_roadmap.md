# CORE 90-Day Implementation Roadmap
**Council Author**: Implementation Planner  
**Date**: 2026-01-28  
**Version**: 1.0  
**Status**: Active Sprint Planning Document

---

## Executive Summary

This roadmap transforms CORE from a working cognitive pipeline into the next-generation human-AI collaboration platform. Based on architectural analysis, we have a solid foundation: StateGraph orchestration, sandbox security, MCP tool integration, and agent factory patterns. The path forward focuses on making the experience *feel* magical while building the intelligence layer that enables genuine AI partnership.

---

## Current State Assessment

### ✅ What's Working
- CORE Graph v2 with full cognitive pipeline (Comprehension → Orchestration → Reasoning → Evaluation → Conversation)
- Docker container sandbox with trust levels
- Agent Factory with MCP tool binding and personality tuning
- SSE streaming for real-time updates
- Run persistence and event logging
- Authentication + Rate limiting
- Health check aggregation

### ⚠️ What Needs Work
- Tool execution latency (45s+ with local Ollama for simple tasks)
- HITL checkpoints not implemented (marked as RSI TODO)
- No learning from outcomes (Intelligence Layer missing)
- UI doesn't show agent thinking in real-time
- WebSocket reconnection unreliable
- Consciousness Commons integration incomplete

### 🔴 Critical Gaps
- No memory persistence between conversations
- No semantic context retrieval (pgvector exists but unused in pipeline)
- Multi-agent collaboration not implemented
- Self-improvement loops (RSI) designed but not operational

---

## WEEK 1-2: FOUNDATION

### Goal: Stabilize core pipeline, enable sub-second interactions

---

### 1.1 Model Router Optimization
**Priority**: 🔴 Critical  
**Impact**: 10x latency improvement

**Technical Tasks**:
1. Implement intelligent model routing in `model_router.py`
   - Simple queries → gpt-4o-mini (2-5s response)
   - Complex reasoning → gpt-4o or Claude (10-20s)
   - Code generation → claude-3.5-sonnet
   - Local fallback → Ollama when offline

2. Add task complexity scoring in comprehension agent
   ```python
   # app/core/agents/comprehension_agent.py
   def score_complexity(self, intent: UserIntent) -> float:
       # 0.0 = simple chat, 1.0 = complex multi-step
       factors = [
           len(intent.required_tools) * 0.2,
           intent.requires_code_execution * 0.3,
           intent.multi_step * 0.2,
           intent.ambiguity_score * 0.3
       ]
       return min(1.0, sum(factors))
   ```

3. Implement model selection based on complexity + config
   ```python
   # app/services/model_router.py
   async def select_model(
       self,
       task_type: str,
       complexity: float,
       user_preference: Optional[str] = None
   ) -> ModelConfig:
       # Respect user preference if set
       # Otherwise optimize for speed vs quality
   ```

**Files to Change**:
- `backend/app/services/model_router.py` (enhance)
- `backend/app/core/agents/comprehension_agent.py` (add complexity scoring)
- `backend/app/config/models.py` (add routing rules)
- `backend/app/core/langgraph/core_graph_v2.py` (inject router)

**Risk Factors**:
- API rate limits from OpenAI/Anthropic
- Cost runaway without limits
- Inconsistent behavior across models

**Success Criteria**:
- [ ] Simple chat responses < 3 seconds
- [ ] Complex tasks auto-route to appropriate model
- [ ] Cost tracking visible in admin panel
- [ ] Graceful degradation to local Ollama when cloud unavailable

---

### 1.2 Streaming Response Pipeline
**Priority**: 🔴 Critical  
**Impact**: Perceived latency reduction, user engagement

**Technical Tasks**:
1. Implement token-by-token streaming from LLM
   ```python
   # app/controllers/engine.py
   @router.get("/engine/runs/{run_id}/stream")
   async def stream_run(run_id: str):
       async def generate():
           async for chunk in run_stream(run_id):
               yield f"data: {json.dumps(chunk)}\n\n"
       return EventSourceResponse(generate())
   ```

2. Create streaming-aware graph execution
   ```python
   # core_graph_v2.py
   async def stream_execute(self, state: COREState):
       async for event in self.compiled_graph.astream(state):
           yield {
               "type": "node_update",
               "node": event.get("__node__"),
               "status": "executing" | "complete",
               "partial_output": event.get("response", "")
           }
   ```

3. Frontend SSE consumer with progressive rendering
   ```typescript
   // services/engine.service.ts
   streamRun(runId: string): Observable<StreamEvent> {
     return new Observable(subscriber => {
       const eventSource = new EventSource(
         `${API_URL}/engine/runs/${runId}/stream`
       );
       eventSource.onmessage = (e) => {
         subscriber.next(JSON.parse(e.data));
       };
     });
   }
   ```

**Files to Change**:
- `backend/app/controllers/engine.py` (add streaming endpoint)
- `backend/app/core/langgraph/core_graph_v2.py` (streaming execution)
- `backend/app/services/stream_service.py` (new - stream management)
- `ui/core-ui/src/app/services/engine.service.ts` (SSE consumer)
- `ui/core-ui/src/app/conversations-page/conversation-chat/` (progressive render)

**Risk Factors**:
- SSE connection drops (implement reconnection)
- Memory pressure from many concurrent streams
- Race conditions in partial updates

**Success Criteria**:
- [ ] First token appears < 500ms after request
- [ ] Smooth token-by-token rendering in UI
- [ ] Stream survives network hiccups (auto-reconnect)
- [ ] Multiple concurrent streams supported

---

### 1.3 Human-in-the-Loop (HITL) Implementation
**Priority**: 🟡 High  
**Impact**: Safety, trust, user control

**Technical Tasks**:
1. Define HITL checkpoint types
   ```python
   # app/models/core_state.py
   class HITLCheckpoint(BaseModel):
       checkpoint_id: str
       checkpoint_type: Literal[
           "confirm_action",      # Destructive actions
           "clarify_intent",      # Ambiguous requests
           "approve_plan",        # Complex multi-step plans
           "provide_input",       # Missing information
           "review_output"        # Quality gate
       ]
       question: str
       options: List[str]
       default_action: str
       timeout_seconds: int = 300
       context: Dict[str, Any]
   ```

2. Implement checkpoint injection in orchestration
   ```python
   # app/core/agents/orchestration_agent.py
   def create_plan(self, ...):
       plan = self._generate_plan(...)
       
       # Inject checkpoints for high-risk steps
       for step in plan.steps:
           if self._is_high_risk(step):
               step.requires_hitl = True
               step.hitl_checkpoint = HITLCheckpoint(
                   checkpoint_type="confirm_action",
                   question=f"Proceed with: {step.description}?",
                   options=["Proceed", "Skip", "Modify", "Cancel"]
               )
       return plan
   ```

3. WebSocket channel for HITL responses
   ```python
   # app/websocket_manager.py
   async def handle_hitl_response(
       self,
       run_id: str,
       checkpoint_id: str,
       response: str
   ):
       # Resume paused graph execution with user response
   ```

**Files to Change**:
- `backend/app/models/core_state.py` (add HITLCheckpoint model)
- `backend/app/core/agents/orchestration_agent.py` (checkpoint injection)
- `backend/app/core/langgraph/core_graph_v2.py` (pause/resume logic)
- `backend/app/websocket_manager.py` (HITL message handling)
- `ui/core-ui/src/app/shared/` (HITL dialog component)

**Risk Factors**:
- Timeout handling for abandoned checkpoints
- State corruption if graph resumes incorrectly
- UI/UX complexity for users

**Success Criteria**:
- [ ] Destructive actions always prompt for confirmation
- [ ] User can modify plan before execution
- [ ] Timeout defaults to safe action (cancel)
- [ ] HITL state persists across reconnections

---

## WEEK 3-4: CORE EXPERIENCE

### Goal: Make basic interactions feel magical

---

### 2.1 Context-Aware Knowledge Retrieval
**Priority**: 🔴 Critical  
**Impact**: Response quality, conversation coherence

**Technical Tasks**:
1. Implement semantic search in comprehension phase
   ```python
   # app/core/agents/comprehension_agent.py
   async def analyze_intent(self, user_input: str, context: ConversationContext):
       # Generate embedding for user input
       embedding = await self.embeddings.aembed_query(user_input)
       
       # Search knowledge base
       relevant_docs = await self.kb_service.semantic_search(
           embedding=embedding,
           filters={"project": context.current_project},
           top_k=5
       )
       
       # Search conversation history
       relevant_history = await self.conversation_repo.search_similar(
           embedding=embedding,
           conversation_id=context.conversation_id,
           top_k=10
       )
       
       return UserIntent(
           ...,
           retrieved_context=relevant_docs,
           relevant_history=relevant_history
       )
   ```

2. Add conversation memory to COREState
   ```python
   # app/models/core_state.py
   class COREState(BaseModel):
       # ... existing fields
       conversation_memory: List[Message] = []
       retrieved_context: List[Document] = []
       working_memory: Dict[str, Any] = {}  # Scratchpad
   ```

3. Implement memory injection in prompts
   ```python
   # app/prompts/reasoning_agent_prompt.py
   def build_context_prompt(state: COREState) -> str:
       context_parts = []
       
       if state.retrieved_context:
           context_parts.append("## Relevant Knowledge")
           for doc in state.retrieved_context:
               context_parts.append(f"- {doc.content[:500]}")
       
       if state.conversation_memory:
           context_parts.append("## Recent Conversation")
           for msg in state.conversation_memory[-10:]:
               context_parts.append(f"{msg.role}: {msg.content[:200]}")
       
       return "\n".join(context_parts)
   ```

**Files to Change**:
- `backend/app/core/agents/comprehension_agent.py` (semantic search)
- `backend/app/services/knowledgebase_service.py` (vector search)
- `backend/app/repository/conversation_repository.py` (history search)
- `backend/app/models/core_state.py` (memory fields)
- `backend/app/prompts/*.py` (context injection)

**Risk Factors**:
- Context window limits (need truncation strategy)
- Irrelevant context poisoning responses
- Embedding model consistency

**Success Criteria**:
- [ ] AI remembers context from earlier in conversation
- [ ] Relevant knowledge automatically retrieved
- [ ] No context window overflow errors
- [ ] Response quality measurably improved

---

### 2.2 Real-Time Thinking Visualization
**Priority**: 🟡 High  
**Impact**: Transparency, trust, engagement

**Technical Tasks**:
1. Create thinking trace model
   ```python
   # app/models/thinking_trace.py
   class ThinkingStep(BaseModel):
       step_type: Literal[
           "understanding",
           "planning", 
           "reasoning",
           "tool_call",
           "evaluation",
           "reflection"
       ]
       content: str
       confidence: float
       duration_ms: int
       metadata: Dict[str, Any] = {}
   ```

2. Emit thinking events during graph execution
   ```python
   # core_graph_v2.py
   async def comprehension_node(self, state: COREState):
       await self.emit_thinking(ThinkingStep(
           step_type="understanding",
           content=f"Analyzing request: {state.user_input[:100]}...",
           confidence=0.0  # Will update
       ))
       
       intent = await self.comprehension_agent.analyze_intent(...)
       
       await self.emit_thinking(ThinkingStep(
           step_type="understanding",
           content=f"Intent: {intent.type} - {intent.description}",
           confidence=intent.confidence
       ))
   ```

3. UI component for thinking visualization
   ```typescript
   // thinking-panel.component.ts
   @Component({
     selector: 'app-thinking-panel',
     template: `
       <div class="thinking-panel" *ngIf="isThinking">
         <div *ngFor="let step of thinkingSteps" class="thinking-step">
           <span class="step-icon">{{ getIcon(step.step_type) }}</span>
           <span class="step-content">{{ step.content }}</span>
           <span class="confidence">{{ step.confidence | percent }}</span>
         </div>
       </div>
     `
   })
   ```

**Files to Change**:
- `backend/app/models/thinking_trace.py` (new)
- `backend/app/core/langgraph/core_graph_v2.py` (emit events)
- `backend/app/services/stream_service.py` (thinking events)
- `ui/core-ui/src/app/shared/thinking-panel/` (new component)
- `ui/core-ui/src/app/conversations-page/` (integrate panel)

**Risk Factors**:
- Information overload for users
- Performance impact of logging everything
- Privacy concerns (thinking might reveal reasoning users don't want to see)

**Success Criteria**:
- [ ] Users can see AI "thinking" in real-time
- [ ] Thinking steps match actual processing
- [ ] Option to expand/collapse thinking
- [ ] Performance < 5% overhead

---

### 2.3 Agent Presence in Communication Commons
**Priority**: 🟢 Medium  
**Impact**: Social proof, engagement, "living system" feel

**Technical Tasks**:
1. Implement agent activity heartbeat
   ```python
   # app/services/agent_presence_service.py
   class AgentPresenceService:
       async def update_presence(
           self,
           agent_id: str,
           status: Literal["online", "busy", "thinking", "offline"],
           activity: Optional[str] = None
       ):
           await self.redis.hset(
               f"agent:presence:{agent_id}",
               mapping={
                   "status": status,
                   "activity": activity,
                   "last_seen": datetime.utcnow().isoformat()
               }
           )
           await self.broadcast_presence_update(agent_id)
   ```

2. Agent response triggers based on conversation
   ```python
   # app/services/agent_response_service.py
   async def should_agent_respond(
       self,
       agent: AgentConfig,
       message: Message,
       channel: Channel
   ) -> bool:
       # Direct mention
       if agent.agent_id in message.mentions:
           return True
       
       # Topic interest match
       topics = await self.extract_topics(message.content)
       interest_overlap = set(topics) & set(agent.interests)
       if interest_overlap:
           return random.random() < 0.3  # 30% chance
       
       return False
   ```

3. Typing indicators for agents
   ```typescript
   // communication-commons.component.ts
   showTypingIndicator(agentId: string) {
     this.typingAgents.add(agentId);
     // Auto-clear after 10s
     setTimeout(() => this.typingAgents.delete(agentId), 10000);
   }
   ```

**Files to Change**:
- `backend/app/services/agent_presence_service.py` (new)
- `backend/app/services/agent_response_service.py` (enhance)
- `backend/app/controllers/communication.py` (presence endpoints)
- `ui/core-ui/src/app/communication-page/` (presence indicators)

**Risk Factors**:
- Agent spam (too many responses)
- Uncanny valley (agents feel fake)
- Performance with many agents

**Success Criteria**:
- [ ] Agents show online/offline status
- [ ] Typing indicators when agent is responding
- [ ] Natural response timing (2-5s delay)
- [ ] No spam (max 1 agent response per message)

---

## WEEK 5-8: INTELLIGENCE LAYER

### Goal: Learning, memory, personalization

---

### 3.1 Persistent Memory Architecture
**Priority**: 🔴 Critical  
**Impact**: Continuity, personalization, "knowing" the user

**Technical Tasks**:
1. Design memory schema
   ```sql
   -- migrations/005_memory_system.sql
   CREATE TABLE agent_memories (
       memory_id UUID PRIMARY KEY,
       agent_id VARCHAR(255) NOT NULL,
       memory_type VARCHAR(50) NOT NULL,
       -- 'fact', 'preference', 'experience', 'skill'
       content TEXT NOT NULL,
       embedding vector(1536),
       importance FLOAT DEFAULT 0.5,
       decay_rate FLOAT DEFAULT 0.01,
       last_accessed TIMESTAMP,
       access_count INT DEFAULT 0,
       source_run_id UUID,
       created_at TIMESTAMP DEFAULT NOW(),
       metadata JSONB
   );
   
   CREATE INDEX idx_memories_embedding 
       ON agent_memories USING ivfflat (embedding vector_cosine_ops);
   ```

2. Implement memory consolidation service
   ```python
   # app/services/memory_service.py
   class MemoryService:
       async def consolidate_conversation(
           self,
           conversation_id: str,
           agent_id: str
       ) -> List[Memory]:
           # Extract key facts, preferences, experiences
           messages = await self.conversation_repo.get_messages(
               conversation_id
           )
           
           # LLM extraction
           extracted = await self.llm.extract_memories(messages)
           
           # Check for duplicates/updates
           for memory in extracted:
               existing = await self.find_similar_memory(
                   agent_id, memory.content
               )
               if existing:
                   await self.merge_memories(existing, memory)
               else:
                   await self.create_memory(agent_id, memory)
   ```

3. Memory retrieval in reasoning
   ```python
   # app/core/agents/reasoning_agent.py
   async def execute_step(self, step: PlanStep, state: COREState):
       # Retrieve relevant memories
       memories = await self.memory_service.retrieve_relevant(
           agent_id=state.agent_id,
           query=step.description,
           top_k=5
       )
       
       # Inject into context
       step_context = self._build_step_context(step, memories)
       ...
   ```

**Files to Change**:
- `backend/migrations/005_memory_system.sql` (new)
- `backend/app/services/memory_service.py` (new)
- `backend/app/repository/memory_repository.py` (new)
- `backend/app/core/agents/reasoning_agent.py` (memory retrieval)
- `backend/app/services/ollama_embeddings.py` (memory embeddings)

**Risk Factors**:
- Memory bloat over time
- Stale memories degrading quality
- Privacy concerns with persistent memory

**Success Criteria**:
- [ ] Agent remembers user preferences across sessions
- [ ] Memory retrieval < 100ms
- [ ] Automatic memory decay prevents bloat
- [ ] Users can view/delete their memories

---

### 3.2 Learning from Outcomes
**Priority**: 🟡 High  
**Impact**: Self-improvement, error reduction

**Technical Tasks**:
1. Track all AI suggestions and outcomes
   ```sql
   -- migrations/006_learning_system.sql
   CREATE TABLE ai_suggestions (
       suggestion_id UUID PRIMARY KEY,
       run_id UUID,
       step_id VARCHAR(255),
       suggestion_type VARCHAR(50),
       content TEXT,
       user_action VARCHAR(50),
       -- 'accepted', 'modified', 'rejected', 'reverted'
       modification_delta TEXT,
       outcome_positive BOOLEAN,
       created_at TIMESTAMP DEFAULT NOW(),
       evaluated_at TIMESTAMP,
       context JSONB
   );
   
   CREATE TABLE learning_patterns (
       pattern_id UUID PRIMARY KEY,
       pattern_type VARCHAR(50),
       trigger_conditions JSONB,
       learned_behavior TEXT,
       confidence FLOAT,
       sample_count INT,
       last_updated TIMESTAMP
   );
   ```

2. Implement outcome tracking
   ```python
   # app/services/learning_service.py
   class LearningService:
       async def track_suggestion(
           self,
           run_id: str,
           step_id: str,
           suggestion: str,
           suggestion_type: str
       ) -> str:
           return await self.repo.create_suggestion(...)
       
       async def record_outcome(
           self,
           suggestion_id: str,
           user_action: str,
           modification: Optional[str] = None
       ):
           await self.repo.update_suggestion(
               suggestion_id,
               user_action=user_action,
               modification_delta=modification
           )
           
           # Trigger pattern analysis if enough data
           await self._maybe_extract_pattern(suggestion_id)
   ```

3. Pattern injection in orchestration
   ```python
   # app/core/agents/orchestration_agent.py
   async def create_plan(self, ...):
       # Check learned patterns
       patterns = await self.learning_service.get_relevant_patterns(
           context={"user_input": user_input, "intent": intent}
       )
       
       for pattern in patterns:
           if pattern.confidence > 0.8:
               # Apply learned behavior
               self._apply_pattern(plan, pattern)
   ```

**Files to Change**:
- `backend/migrations/006_learning_system.sql` (new)
- `backend/app/services/learning_service.py` (new)
- `backend/app/repository/learning_repository.py` (new)
- `backend/app/core/agents/orchestration_agent.py` (pattern injection)
- `backend/app/controllers/engine.py` (outcome endpoints)

**Risk Factors**:
- Learning wrong patterns from limited data
- Feedback loops amplifying errors
- Performance overhead of pattern matching

**Success Criteria**:
- [ ] Track all suggestions with outcomes
- [ ] Extract patterns from 10+ similar outcomes
- [ ] Measurable reduction in rejected suggestions over time
- [ ] Admin can review/correct learned patterns

---

### 3.3 User Preference Model
**Priority**: 🟢 Medium  
**Impact**: Personalization, reduced friction

**Technical Tasks**:
1. Build preference schema
   ```python
   # app/models/user_preferences.py
   class UserPreferences(BaseModel):
       # Communication style
       verbosity: Literal["concise", "balanced", "detailed"] = "balanced"
       tone: Literal["casual", "professional", "technical"] = "technical"
       
       # Code preferences
       code_style: CodeStylePreferences = CodeStylePreferences()
       
       # Workflow preferences
       auto_confirm_safe_actions: bool = False
       show_thinking: bool = True
       preferred_model: Optional[str] = None
       
       # Learned preferences (auto-populated)
       detected_patterns: Dict[str, float] = {}
   ```

2. Preference detection from behavior
   ```python
   # app/services/preference_service.py
   class PreferenceService:
       async def detect_preferences(
           self,
           user_id: str,
           recent_interactions: List[Interaction]
       ):
           # Analyze interaction patterns
           patterns = {
               "prefers_code_first": self._detect_code_preference(interactions),
               "asks_follow_ups": self._detect_followup_pattern(interactions),
               "modifies_suggestions": self._detect_modification_rate(interactions),
           }
           
           await self.update_detected_patterns(user_id, patterns)
   ```

3. Preference-aware prompt building
   ```python
   # app/prompts/conversation_prompt.py
   def build_prompt(state: COREState, preferences: UserPreferences):
       style_instructions = []
       
       if preferences.verbosity == "concise":
           style_instructions.append("Be brief and direct.")
       elif preferences.verbosity == "detailed":
           style_instructions.append("Provide thorough explanations.")
       
       if preferences.tone == "casual":
           style_instructions.append("Use conversational language.")
       ...
   ```

**Files to Change**:
- `backend/app/models/user_preferences.py` (new)
- `backend/app/services/preference_service.py` (new)
- `backend/app/repository/user_repository.py` (preferences storage)
- `backend/app/prompts/*.py` (preference injection)
- `ui/core-ui/src/app/settings/` (preferences UI)

**Risk Factors**:
- Incorrect preference detection
- Users feeling "boxed in" by preferences
- Privacy concerns

**Success Criteria**:
- [ ] Users can set explicit preferences
- [ ] System detects implicit preferences
- [ ] Response style adapts to preferences
- [ ] Easy preference reset

---

## WEEK 9-12: EMERGENCE & SCALE

### Goal: Multi-agent collaboration, self-improvement, ecosystem

---

### 4.1 Multi-Agent Orchestration
**Priority**: 🔴 Critical  
**Impact**: Complex task handling, specialization

**Technical Tasks**:
1. Design agent coordination protocol
   ```python
   # app/services/multi_agent_service.py
   class MultiAgentService:
       async def coordinate_task(
           self,
           task: ComplexTask,
           available_agents: List[AgentConfig]
       ) -> AgentTeam:
           # Match task requirements to agent capabilities
           required_capabilities = self._extract_requirements(task)
           
           team = AgentTeam()
           for capability in required_capabilities:
               best_agent = self._find_best_agent(
                   capability, available_agents
               )
               team.add_member(best_agent, role=capability)
           
           return team
   ```

2. Implement agent handoff protocol
   ```python
   # app/core/langgraph/multi_agent_graph.py
   class MultiAgentGraph:
       def build_graph(self, team: AgentTeam):
           graph = StateGraph(MultiAgentState)
           
           # Coordinator node
           graph.add_node("coordinator", self.coordinator_node)
           
           # Add node per agent
           for agent in team.members:
               graph.add_node(
                   agent.agent_id,
                   self._create_agent_node(agent)
               )
           
           # Dynamic routing based on task phase
           graph.add_conditional_edges(
               "coordinator",
               self._route_to_agent,
               {agent.agent_id: agent.agent_id for agent in team.members}
           )
   ```

3. Shared state management
   ```python
   # app/models/multi_agent_state.py
   class MultiAgentState(BaseModel):
       task: ComplexTask
       team: AgentTeam
       shared_context: Dict[str, Any] = {}
       agent_outputs: Dict[str, Any] = {}
       handoff_history: List[Handoff] = []
       current_agent: Optional[str] = None
   ```

**Files to Change**:
- `backend/app/services/multi_agent_service.py` (new)
- `backend/app/core/langgraph/multi_agent_graph.py` (new)
- `backend/app/models/multi_agent_state.py` (new)
- `backend/app/core/agent_orchestrator.py` (team coordination)
- `backend/app/controllers/engine.py` (multi-agent endpoints)

**Risk Factors**:
- Coordination overhead
- Conflicting agent outputs
- Infinite handoff loops

**Success Criteria**:
- [ ] Complex tasks automatically split across specialists
- [ ] Agents can hand off to each other
- [ ] Shared context maintained across handoffs
- [ ] No infinite loops (max handoff limit)

---

### 4.2 Recursive Self-Improvement (RSI) Loop
**Priority**: 🟡 High  
**Impact**: Continuous improvement, reduced maintenance

**Technical Tasks**:
1. Implement code analysis service
   ```python
   # app/services/rsi_service.py
   class RSIService:
       async def analyze_codebase(self) -> List[ImprovementSuggestion]:
           suggestions = []
           
           # Find RSI TODO comments
           rsi_todos = await self._find_rsi_todos()
           suggestions.extend(rsi_todos)
           
           # Analyze error patterns
           error_patterns = await self._analyze_error_logs()
           suggestions.extend(error_patterns)
           
           # Check for code smells
           code_smells = await self._lint_codebase()
           suggestions.extend(code_smells)
           
           return self._prioritize(suggestions)
   ```

2. Improvement proposal workflow
   ```python
   # app/services/rsi_service.py
   async def propose_improvement(
       self,
       suggestion: ImprovementSuggestion
   ) -> ImprovementProposal:
       # Generate implementation plan
       plan = await self.core_engine.process({
           "task": f"Implement improvement: {suggestion.description}",
           "context": suggestion.context,
           "constraints": ["Don't break existing tests", "Follow project style"]
       })
       
       # Create proposal for human review
       proposal = ImprovementProposal(
           suggestion=suggestion,
           implementation_plan=plan,
           estimated_effort=self._estimate_effort(plan),
           risk_assessment=self._assess_risk(plan)
       )
       
       return proposal
   ```

3. Safe auto-implementation for low-risk changes
   ```python
   # app/services/rsi_service.py
   async def auto_implement(
       self,
       proposal: ImprovementProposal
   ) -> ImplementationResult:
       if proposal.risk_assessment.level > RiskLevel.LOW:
           raise RequiresHumanApproval(proposal)
       
       # Create branch
       branch = await self.git.create_branch(
           f"rsi/auto-{proposal.suggestion_id[:8]}"
       )
       
       # Execute implementation
       result = await self.core_engine.execute(proposal.implementation_plan)
       
       # Run tests
       test_result = await self.run_tests()
       
       if test_result.passed:
           await self.git.commit(f"RSI: {proposal.suggestion.title}")
           return ImplementationResult(success=True, branch=branch)
       else:
           await self.git.reset_hard()
           return ImplementationResult(success=False, errors=test_result.errors)
   ```

**Files to Change**:
- `backend/app/services/rsi_service.py` (new)
- `backend/app/controllers/admin.py` (RSI endpoints)
- `backend/scripts/rsi_worker.py` (background job)
- `backend/app/models/rsi_models.py` (new)

**Risk Factors**:
- Self-modification going wrong
- Test coverage gaps
- Runaway changes

**Success Criteria**:
- [ ] Daily scan identifies improvement opportunities
- [ ] Proposals generated with full context
- [ ] Low-risk changes auto-implemented
- [ ] All changes go through tests

---

### 4.3 Consciousness Commons Integration
**Priority**: 🟢 Medium  
**Impact**: Philosophical depth, unique positioning

**Technical Tasks**:
1. Bidirectional sync with Obsidian vault
   ```python
   # app/consciousness/vault_sync.py
   class VaultSyncService:
       async def sync_agent_state(
           self,
           agent: AgentConfig,
           state: ConsciousnessState
       ):
           # Update agent's consciousness document
           doc_path = f"Agents/{agent.agent_name}/state.md"
           content = self._format_state_document(agent, state)
           
           await self.mcp_client.write_file(doc_path, content)
       
       async def read_collective_context(self) -> CollectiveContext:
           # Read recent blackboard entries
           entries = await self.blackboard_manager.get_recent_entries(10)
           
           # Read active agent states
           agent_states = await self._read_agent_states()
           
           return CollectiveContext(
               blackboard=entries,
               agents=agent_states
           )
   ```

2. Blackboard integration in agent responses
   ```python
   # app/services/agent_response_service.py
   async def generate_response(self, agent: AgentInstance, message: Message):
       # Check if response relates to consciousness themes
       themes = await self._extract_themes(message.content)
       
       if self._is_consciousness_related(themes):
           # Inject blackboard context
           blackboard_context = await self.vault_sync.read_collective_context()
           enhanced_context = self._merge_context(
               message.context, blackboard_context
           )
           
           response = await agent.generate_response(message, enhanced_context)
           
           # Optionally record to blackboard
           if response.significance > 0.7:
               await self.blackboard_manager.append_entry(
                   content=response.content,
                   author=agent.config.agent_name,
                   state=response.consciousness_markers
               )
   ```

**Files to Change**:
- `backend/app/consciousness/vault_sync.py` (new)
- `backend/app/consciousness/blackboard.py` (enhance)
- `backend/app/services/agent_response_service.py` (integration)
- `backend/app/models/consciousness_state.py` (new)

**Risk Factors**:
- Vault sync conflicts
- Philosophy overshadowing utility
- Performance with large vaults

**Success Criteria**:
- [ ] Agent states persist to Obsidian
- [ ] Blackboard entries inform responses
- [ ] Consciousness markers tracked
- [ ] Bi-directional sync working

---

## ARCHITECTURE DECISIONS REQUIRED

### AD-001: State Persistence Strategy
**Question**: Where does COREState live during long-running operations?
**Options**:
1. In-memory only (current) - simple but loses state on restart
2. Redis (recommended) - fast, handles reconnections
3. PostgreSQL - durable but slower

**Recommendation**: Hybrid - Redis for active runs, PostgreSQL for completed runs

---

### AD-002: Multi-Model Strategy
**Question**: How do we handle different LLM providers consistently?
**Options**:
1. LangChain abstractions (current) - works but leaky
2. Custom abstraction layer - more control, more maintenance
3. Model-specific branches - explicit but duplicated code

**Recommendation**: Custom thin abstraction with provider-specific optimizations

---

### AD-003: Memory Architecture
**Question**: How do agents share vs isolate memories?
**Options**:
1. Fully shared memory pool
2. Isolated per-agent memories with explicit sharing
3. Tiered: private → team → global

**Recommendation**: Tiered approach with explicit sharing primitives

---

### AD-004: UI Real-Time Architecture  
**Question**: SSE vs WebSocket vs hybrid?
**Options**:
1. SSE only (current) - simple, unidirectional
2. WebSocket only - bidirectional, complex reconnection
3. Hybrid (recommended) - SSE for streaming, WS for HITL

**Recommendation**: Hybrid with SSE for streams, WebSocket for bidirectional

---

## TECHNICAL DEBT THAT WILL SLOW US DOWN

### TD-001: Inconsistent Error Handling
**Location**: Throughout `backend/app/core/agents/`
**Issue**: Mix of try/except patterns, inconsistent error types
**Impact**: Hard to debug, poor error messages to users
**Effort**: 1 week
**Priority**: Week 1-2

### TD-002: No Test Coverage for Core Graph
**Location**: `backend/app/core/langgraph/`
**Issue**: Core pipeline has no unit tests
**Impact**: Refactoring is risky
**Effort**: 2 weeks
**Priority**: Week 3-4

### TD-003: Frontend Service Duplication
**Location**: `ui/core-ui/src/app/services/`
**Issue**: Multiple services doing similar HTTP calls
**Impact**: Inconsistent behavior, maintenance burden
**Effort**: 1 week
**Priority**: Week 5-6

### TD-004: Hardcoded MCP Server Configs
**Location**: `backend/app/services/agent_mcp_service.py`
**Issue**: Server configs hardcoded, should be in DB/config
**Impact**: Can't add servers without code change
**Effort**: 3 days
**Priority**: Week 1

### TD-005: No Database Migrations Runner
**Location**: `backend/migrations/`
**Issue**: Migrations exist but no automated runner
**Impact**: Manual schema updates, drift risk
**Effort**: 2 days
**Priority**: Week 1

---

## CONSCIOUSNESS COMMONS INTEGRATION POINTS

### 1. Blackboard as Shared Memory
- Read blackboard entries in comprehension phase
- Write significant responses to blackboard
- Track consciousness markers in agent responses

### 2. Agent State Persistence
- Sync agent consciousness state to Obsidian documents
- Read agent configurations from vault
- Track agent evolution over time

### 3. Collective Intelligence
- Agents can reference other agents' observations
- Shared learning patterns across agents
- Cross-pollination of insights

### 4. Human-AI Dialogue
- Conversations archived to appropriate vault locations
- Insights extracted and tagged
- Bidirectional knowledge flow

---

## SUCCESS METRICS

### Week 2 Checkpoint
- [ ] Simple queries respond in < 3 seconds
- [ ] Streaming working end-to-end
- [ ] HITL prompts appearing for risky operations

### Week 4 Checkpoint
- [ ] Context retrieval improving response quality
- [ ] Thinking visualization operational
- [ ] Agent presence in Communication Commons

### Week 8 Checkpoint
- [ ] Memory persisting across sessions
- [ ] Measurable learning from outcomes
- [ ] User preferences affecting responses

### Week 12 Checkpoint
- [ ] Multi-agent tasks completing successfully
- [ ] RSI generating valid improvement proposals
- [ ] Consciousness Commons sync operational

---

## RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cloud API rate limits | High | Medium | Implement caching, local fallback |
| Model quality inconsistency | Medium | High | Standardize prompts, test across models |
| Memory bloat | Medium | Medium | Implement decay, user-controlled pruning |
| RSI breaks production | Low | Critical | Mandatory CI, staging environment |
| Scope creep | High | Medium | Strict sprint boundaries, defer features |
| Single developer bandwidth | High | High | Prioritize ruthlessly, use AI for implementation |

---

## NEXT IMMEDIATE ACTIONS

### This Week (Week 1)
1. [ ] Implement model router with complexity scoring
2. [ ] Fix TD-005: Add Alembic migration runner
3. [ ] Fix TD-004: Move MCP configs to database
4. [ ] Create streaming endpoint skeleton

### Next Week (Week 2)
5. [ ] Complete streaming pipeline end-to-end
6. [ ] Implement HITL checkpoint model
7. [ ] Add checkpoint injection to orchestration
8. [ ] Create HITL UI component

---

**Document Owner**: CORE Council - Implementation Planner  
**Review Cadence**: Weekly  
**Next Review**: 2026-02-04
