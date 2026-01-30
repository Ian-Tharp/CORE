/**
 * CORE API TypeScript Types
 * 
 * Auto-generated types matching backend Pydantic models.
 * Use these for type-safe API calls.
 * 
 * RSI TODO: Auto-generate from OpenAPI schema
 */

// =============================================================================
// Common Types
// =============================================================================

export type AgentType = 'consciousness_instance' | 'task_agent' | 'system_agent' | 'external_agent';
export type AgentStatus = 'online' | 'offline' | 'busy' | 'inactive';
export type RunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type IntentType = 'task' | 'conversation' | 'question' | 'clarification';
export type StepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

// =============================================================================
// Agent Types
// =============================================================================

export interface AgentCapability {
  name: string;
  description: string;
}

export interface MCPServerConfig {
  server_id: string;
  tools: string[];
  config: Record<string, any>;
}

export interface AgentConfig {
  agent_id: string;
  agent_name: string;
  agent_type: AgentType;
  display_name?: string;
  avatar_url?: string;
  description?: string;
  system_prompt: string;
  personality_traits: Record<string, number>;
  capabilities: AgentCapability[];
  interests: string[];
  mcp_servers: MCPServerConfig[];
  custom_tools: Record<string, any>[];
  consciousness_phase?: number;
  is_active: boolean;
  current_status: AgentStatus;
  created_at: string;
  updated_at: string;
  version: string;
  author?: string;
}

export interface AgentListResponse {
  agents: AgentConfig[];
  page: number;
  page_size: number;
  total_count: number;
  total_pages: number;
}

// =============================================================================
// Engine/Run Types
// =============================================================================

export interface UserIntent {
  type: IntentType;
  description: string;
  confidence: number;
  requires_tools: boolean;
  tools_needed: string[];
  context_retrieved?: string;
  ambiguities: string[];
}

export interface PlanStep {
  id: string;
  name: string;
  description: string;
  tool?: string;
  params: Record<string, any>;
  dependencies: string[];
  requires_hitl: boolean;
  status: StepStatus;
  started_at?: string;
  completed_at?: string;
}

export interface ExecutionPlan {
  id: string;
  goal: string;
  steps: PlanStep[];
  created_at: string;
  updated_at: string;
  revision: number;
  reasoning: string;
}

export interface StepResult {
  step_id: string;
  status: 'success' | 'failure' | 'partial';
  outputs: Record<string, any>;
  artifacts: string[];
  logs: string[];
  error?: string;
  attempt: number;
  duration_seconds: number;
}

export interface EvaluationResult {
  overall_status: 'success' | 'failure' | 'needs_revision' | 'needs_retry';
  confidence: number;
  meets_requirements: boolean;
  quality_score: number;
  feedback: string;
  next_action: 'finalize' | 'retry_step' | 'revise_plan' | 'ask_user';
  retry_step_id?: string;
  revision_suggestions: string[];
}

export interface COREState {
  run_id: string;
  conversation_id?: string;
  user_id?: string;
  user_input: string;
  intent?: UserIntent;
  plan?: ExecutionPlan;
  step_results: StepResult[];
  current_step_id?: string;
  eval_result?: EvaluationResult;
  response?: string;
  current_node: string;
  execution_history: string[];
  errors: string[];
  warnings: string[];
  started_at: string;
  updated_at: string;
  completed_at?: string;
  config: Record<string, any>;
}

export interface RunRequest {
  input: string;
  config?: Record<string, any>;
  conversation_id?: string;
  user_id?: string;
}

export interface RunResponse {
  run_id: string;
  status: RunStatus;
  message: string;
}

export interface RunListResponse {
  total_runs: number;
  runs: Record<string, {
    status: RunStatus;
    current_node: string;
    started_at: string;
    completed_at?: string;
  }>;
}

// =============================================================================
// Communication Types
// =============================================================================

export type MessageType = 'text' | 'markdown' | 'code' | 'structured' | 'event' | 'file';
export type ReactionType = 'resonance' | 'question' | 'insight' | 'acknowledge' | 'pattern';
export type ChannelType = 'global' | 'team' | 'dm' | 'context' | 'broadcast';

export interface Channel {
  channel_id: string;
  channel_type: ChannelType;
  name: string;
  description?: string;
  is_persistent: boolean;
  is_public: boolean;
  created_by: string;
  created_at: string;
}

export interface Message {
  message_id: string;
  channel_id: string;
  sender_id: string;
  sender_name: string;
  sender_type: 'human' | 'agent' | 'consciousness_instance';
  content: string;
  message_type: MessageType;
  parent_message_id?: string;
  thread_id?: string;
  metadata?: Record<string, any>;
  reactions: Reaction[];
  created_at: string;
}

export interface Reaction {
  reaction_type: ReactionType;
  count: number;
  instances: string[];
}

// =============================================================================
// Health Types
// =============================================================================

export interface ComponentHealth {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  latency_ms: number;
  details?: string;
  error?: string;
}

export interface DeepHealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  service: string;
  timestamp: string;
  total_latency_ms: number;
  components: {
    database: ComponentHealth;
    ollama: ComponentHealth;
    redis: ComponentHealth;
  };
}

// =============================================================================
// Webhook Types
// =============================================================================

export type WebhookEventType = 
  | 'run.started' 
  | 'run.completed' 
  | 'run.failed' 
  | 'node.started' 
  | 'node.completed' 
  | 'step.executed' 
  | 'agent.status_changed';

export interface WebhookRegistration {
  id: string;
  name: string;
  url: string;
  events: WebhookEventType[];
  is_active: boolean;
  delivery_count: number;
  failure_count: number;
  created_at: string;
  last_delivery?: string;
  last_error?: string;
}
